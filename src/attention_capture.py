"""
Read-only attention capture for per-segment attention analysis.

This module monkey-patches attention modules (same pattern as attention_hook.py)
to record post-softmax attention weights aggregated per rule segment.  No GPU
tensors are retained — only Python floats on CPU.

Typical workflow::

    segment_map = SegmentAttentionMap(...)
    handle = register_capture_hooks(model, segment_map)
    model.generate(...)
    results = get_capture_results(handle)
    unregister_capture_hooks(handle)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from .attention_hook import find_attention_modules


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentAttentionMap:
    """Describes which token positions belong to each rule segment.

    Attributes:
        segment_names: Human-readable segment identifiers (e.g.
            ``"first_bag/row_us_puerto_rico"``).
        segment_token_sets: Parallel list — ``segment_token_sets[i]`` holds
            the token indices for ``segment_names[i]``.
        segment_is_applicable: Whether each segment is relevant (applicable)
            to the current problem.
    """
    segment_names: list[str]
    segment_token_sets: list[list[int]]
    segment_is_applicable: list[bool]

    def __post_init__(self):
        n = len(self.segment_names)
        if len(self.segment_token_sets) != n:
            raise ValueError(
                f"segment_token_sets length ({len(self.segment_token_sets)}) "
                f"must match segment_names length ({n})"
            )
        if len(self.segment_is_applicable) != n:
            raise ValueError(
                f"segment_is_applicable length ({len(self.segment_is_applicable)}) "
                f"must match segment_names length ({n})"
            )


@dataclass
class AttentionCaptureHandle:
    """Handle for managing attention capture hooks.

    Attributes:
        patched_modules: Maps module name → original forward method.
        model: Reference to the patched model.
        segment_map: The segment description used for aggregation.
        step_counter: Mutable ``{"step": 0}`` incremented on each forward pass.
        data: Accumulated attention data.
            ``data[layer_idx][step]`` is a ``list[float]`` of length
            ``len(segment_map.segment_names)``.
        num_layers: Number of attention layers that were patched.
        segment_index_tensors: Pre-built CPU ``LongTensor`` per segment
            for fast ``index_select``.
    """
    patched_modules: dict = field(default_factory=dict)
    model: Optional[nn.Module] = None
    segment_map: Optional[SegmentAttentionMap] = None
    step_counter: dict = field(default_factory=lambda: {"step": 0})
    data: dict = field(default_factory=dict)
    num_layers: int = 0
    segment_index_tensors: list = field(default_factory=list)
    min_seq_k: int = 0  # minimum key-sequence length; softmax calls with seq_k <
                        # this are skipped (avoids capturing auxiliary softmax ops
                        # that precede the real attention softmax in some models)


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------

def _create_capture_patched_forward(original_forward, handle, layer_idx):
    """Wrap an attention module's forward to capture post-softmax weights.

    The patched softmax:
    1. Calls the real softmax to get attention weights.
    2. Aggregates **on GPU** — averages across heads to reduce
       ``[batch, H, Sq, Sk]`` to ``[Sq, Sk]``, avoiding a multi-GB copy.
    3. For each segment, gathers its token columns via ``index_select``
       on GPU, reduces to a single scalar, and extracts via ``.item()``.
    4. Stores one Python float per segment — no tensors retained.
    """

    def patched_forward(*args, **kwargs):
        original_softmax = torch.nn.functional.softmax
        captured = {"done": False}

        def capturing_softmax(input_tensor, dim=-1, dtype=None):
            result = original_softmax(input_tensor, dim=dim, dtype=dtype)

            if (not captured["done"] and dim == -1
                    and len(input_tensor.shape) >= 4
                    and input_tensor.shape[-1] >= handle.min_seq_k):
                captured["done"] = True

                # Increment step counter only for the first layer
                if layer_idx == 0:
                    handle.step_counter["step"] += 1
                step = handle.step_counter["step"] - 1

                # Aggregate ON GPU to avoid copying the full
                # [batch, heads, seq_q, seq_k] tensor (can be multi-GB).
                # Average across heads in native dtype (bf16) FIRST to
                # reduce [1, H, Sq, Sk] → [Sq, Sk] before any float32 cast.
                attn_avg = result.detach()[0].mean(dim=0).float()
                device = attn_avg.device
                seq_k = attn_avg.size(1)

                # Aggregate per segment using index_select on GPU
                seg_attention: list[float] = []
                for idx_tensor in handle.segment_index_tensors:
                    if idx_tensor.numel() == 0:
                        seg_attention.append(0.0)
                        continue
                    # Filter to indices within current seq_k
                    valid_mask = idx_tensor < seq_k
                    valid_indices = idx_tensor[valid_mask]
                    if valid_indices.numel() == 0:
                        seg_attention.append(0.0)
                        continue
                    # Move index tensor to same device, gather, reduce to scalar
                    gpu_indices = valid_indices.to(device)
                    cols = torch.index_select(attn_avg, 1, gpu_indices)
                    val = cols.sum(dim=1).mean(dim=0).item()
                    seg_attention.append(val)

                # Free the GPU intermediate immediately
                del attn_avg

                # Store as plain Python list
                if layer_idx not in handle.data:
                    handle.data[layer_idx] = {}
                handle.data[layer_idx][step] = seg_attention

            return result

        torch.nn.functional.softmax = capturing_softmax
        try:
            output = original_forward(*args, **kwargs)
        finally:
            torch.nn.functional.softmax = original_softmax

        return output

    return patched_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_capture_hooks(
    model: nn.Module,
    segment_map: SegmentAttentionMap,
    layers: Optional[list[int]] = None,
) -> AttentionCaptureHandle:
    """Install attention capture hooks on the model.

    Args:
        model: HuggingFace transformer model (must use
            ``attn_implementation="eager"`` for explicit softmax).
        segment_map: Describes which tokens belong to each segment.
        layers: Optional list of layer indices to capture.  ``None`` means
            all layers.

    Returns:
        An :class:`AttentionCaptureHandle` for later cleanup.
    """
    attention_modules = find_attention_modules(model)
    if not attention_modules:
        raise ValueError("No attention modules found in model")

    # Pre-build CPU index tensors for each segment (allocated once)
    seg_index_tensors = []
    for token_set in segment_map.segment_token_sets:
        seg_index_tensors.append(torch.tensor(token_set, dtype=torch.long))

    # Minimum seq_k that the real attention softmax must have.
    # Any softmax call with seq_k < this is an auxiliary op (e.g. MoE routing)
    # that appears before the actual attention computation in some architectures.
    all_indices = [idx for ts in segment_map.segment_token_sets for idx in ts]
    min_seq_k = max(all_indices) + 1 if all_indices else 0

    handle = AttentionCaptureHandle(
        model=model,
        segment_map=segment_map,
        num_layers=len(attention_modules),
        segment_index_tensors=seg_index_tensors,
        min_seq_k=min_seq_k,
    )

    for layer_idx, (name, module) in enumerate(attention_modules):
        if layers is not None and layer_idx not in layers:
            continue

        original_forward = module.forward
        handle.patched_modules[name] = original_forward
        module.forward = _create_capture_patched_forward(
            original_forward, handle, layer_idx
        )

    if not handle.patched_modules:
        raise ValueError("No attention modules were patched")

    return handle


def unregister_capture_hooks(handle: AttentionCaptureHandle) -> None:
    """Restore original forward methods on all patched modules.

    Args:
        handle: Handle returned by :func:`register_capture_hooks`.
    """
    for name, module in handle.model.named_modules():
        if name in handle.patched_modules:
            module.forward = handle.patched_modules[name]
    handle.patched_modules.clear()


def reset_capture_data(handle: AttentionCaptureHandle) -> None:
    """Clear accumulated attention data so the handle can be reused.

    The hooks remain installed — only the data dict and step counter
    are reset.
    """
    handle.data.clear()
    handle.step_counter["step"] = 0


def get_capture_results(handle: AttentionCaptureHandle) -> dict:
    """Return the accumulated attention data as a plain dict.

    Returns:
        ``{"data": handle.data, "num_layers": handle.num_layers,
        "num_steps": <max step + 1>, "num_segments": <segment count>}``
    """
    num_steps = handle.step_counter["step"]
    return {
        "data": handle.data,
        "num_layers": handle.num_layers,
        "num_steps": num_steps,
        "num_segments": len(handle.segment_map.segment_names),
    }
