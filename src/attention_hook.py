"""
Core attention hooking functionality for InstABoost.

This module modifies pre-softmax attention scores in transformer models by
monkey-patching attention module forward methods to add bias before softmax.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Any
from dataclasses import dataclass, field
from .boost_config import BoostConfig, TokenSubset


@dataclass
class HookHandle:
    """
    Handle for managing patched attention modules.

    Attributes:
        patched_modules: Dict mapping module name to original forward method
        config: Current boost configuration
        model: Reference to the model
        bias_mask: Cached bias mask [seq_length]
    """
    patched_modules: dict = field(default_factory=dict)
    config: Optional[BoostConfig] = None
    model: Optional[nn.Module] = None
    bias_mask: Optional[torch.Tensor] = None


def create_bias_mask(
    subsets: List[TokenSubset],
    seq_length: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a bias mask from token subsets.

    Creates a 1D tensor where each position contains the sum of all bias
    values from subsets that include that token position.

    Args:
        subsets: List of TokenSubset objects
        seq_length: Length of the sequence
        device: Device to create tensor on

    Returns:
        Bias mask tensor of shape [seq_length]
    """
    if device is None:
        device = torch.device("cpu")

    # Initialize with zeros
    bias_mask = torch.zeros(seq_length, device=device, dtype=torch.float32)

    # Add bias for each subset (overlapping indices get summed)
    for subset in subsets:
        for idx in subset.indices:
            if idx >= seq_length:
                raise ValueError(
                    f"Token index {idx} in subset '{subset.name}' exceeds "
                    f"sequence length {seq_length}"
                )
            bias_mask[idx] += subset.bias

    return bias_mask


def _create_patched_forward(original_forward, bias_mask: torch.Tensor, layer_idx: int, config: BoostConfig):
    """
    Create a patched forward method that applies bias before softmax.

    This wraps the original forward method and temporarily replaces the
    softmax function to add our bias right before normalization.

    Args:
        original_forward: Original forward method of the attention module
        bias_mask: Bias mask to apply [seq_length]
        layer_idx: Layer index (for layer filtering)
        config: Boost configuration

    Returns:
        Patched forward method
    """
    def patched_forward(*args, **kwargs):
        # Check if we should apply to this layer
        if config.layers is not None and layer_idx not in config.layers:
            return original_forward(*args, **kwargs)

        # Store reference to original softmax
        original_softmax = torch.nn.functional.softmax
        bias_applied = {"applied": False}  # Track if we've applied bias this forward pass

        def biased_softmax(input_tensor, dim=-1, dtype=None):
            """Softmax with bias applied before normalization."""
            # Only apply bias once per forward pass, and only to the right tensor
            if not bias_applied["applied"] and dim == -1:
                # Check if this looks like attention scores
                # Attention scores typically have shape [batch, heads, seq_q, seq_k]
                if len(input_tensor.shape) >= 2 and input_tensor.size(-1) == bias_mask.size(0):
                    # Move bias to correct device/dtype
                    bias = bias_mask.to(device=input_tensor.device, dtype=input_tensor.dtype)
                    # Add bias before softmax: S' = S + bias
                    # Broadcasting handles batch and head dimensions automatically
                    input_tensor = input_tensor + bias
                    bias_applied["applied"] = True

            # Call original softmax
            return original_softmax(input_tensor, dim=dim, dtype=dtype)

        # Temporarily replace softmax
        torch.nn.functional.softmax = biased_softmax

        try:
            # Call original forward with biased softmax
            result = original_forward(*args, **kwargs)
        finally:
            # Always restore original softmax
            torch.nn.functional.softmax = original_softmax

        return result

    return patched_forward


def register_boost_hooks(
    model: nn.Module,
    config: BoostConfig,
    input_length: Optional[int] = None,
) -> HookHandle:
    """
    Register attention boosting by patching attention module forward methods.

    This finds all attention modules in the model and replaces their forward
    methods with versions that add bias before softmax.

    Args:
        model: HuggingFace transformer model
        config: Boost configuration
        input_length: Expected input length (for validation)

    Returns:
        HookHandle for cleanup and updates

    Example:
        >>> handle = register_boost_hooks(model, config)
        >>> update_bias_mask(handle, seq_length=20)
        >>> output = model.generate(...)
        >>> unregister_boost_hooks(handle)
    """
    # Validate configuration
    if input_length is not None:
        max_idx = config.get_max_token_index()
        if max_idx >= input_length:
            raise ValueError(
                f"Token indices exceed input length: "
                f"max_idx={max_idx}, input_length={input_length}"
            )

    # Find attention modules
    attention_patterns = ["attn", "self_attn", "attention"]
    attention_modules = []

    for name, module in model.named_modules():
        name_lower = name.lower()
        # Match attention module patterns
        if any(pattern in name_lower for pattern in attention_patterns):
            # Exclude projection sub-modules
            if not any(proj in name_lower for proj in ["q_proj", "k_proj", "v_proj", "out_proj", "c_proj"]):
                attention_modules.append((name, module))

    if not attention_modules:
        raise ValueError("No attention modules found in model")

    # Create handle
    handle = HookHandle(config=config, model=model)

    # Patch attention modules
    # Note: We use a dummy bias_mask initially; it will be updated via update_bias_mask()
    dummy_bias = torch.zeros(1)

    for layer_idx, (name, module) in enumerate(attention_modules):
        # Skip if layer filtering is enabled and this layer isn't selected
        if config.layers is not None and layer_idx not in config.layers:
            continue

        # Save original forward
        original_forward = module.forward
        handle.patched_modules[name] = original_forward

        # Create and apply patched forward
        module.forward = _create_patched_forward(
            original_forward,
            dummy_bias,  # Will be updated when update_bias_mask is called
            layer_idx,
            config
        )

    if not handle.patched_modules:
        raise ValueError("No attention modules were patched")

    return handle


def unregister_boost_hooks(handle: HookHandle) -> None:
    """
    Restore original forward methods to all patched modules.

    Args:
        handle: HookHandle from register_boost_hooks
    """
    # Find modules by name and restore original forwards
    for name, module in handle.model.named_modules():
        if name in handle.patched_modules:
            module.forward = handle.patched_modules[name]

    handle.patched_modules.clear()


def update_bias_mask(
    handle: HookHandle,
    seq_length: int,
    device: Optional[torch.device] = None
) -> None:
    """
    Update the bias mask and re-patch modules with new mask.

    This must be called before generation to set the correct bias mask
    for the input sequence length.

    Args:
        handle: HookHandle from register_boost_hooks
        seq_length: Length of input sequence
        device: Device for bias mask (inferred from model if not provided)
    """
    if device is None:
        device = next(handle.model.parameters()).device

    # Create new bias mask
    bias_mask = create_bias_mask(
        subsets=handle.config.subsets,
        seq_length=seq_length,
        device=device
    )
    handle.bias_mask = bias_mask

    # Re-patch all modules with new bias mask
    # Find attention modules again
    attention_patterns = ["attn", "self_attn", "attention"]
    attention_modules = []

    for name, module in handle.model.named_modules():
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in attention_patterns):
            if not any(proj in name_lower for proj in ["q_proj", "k_proj", "v_proj", "out_proj", "c_proj"]):
                if name in handle.patched_modules:  # Only re-patch what we patched before
                    attention_modules.append((name, module))

    # Re-apply patches with new bias mask
    for layer_idx, (name, module) in enumerate(attention_modules):
        original_forward = handle.patched_modules[name]
        module.forward = _create_patched_forward(
            original_forward,
            bias_mask,
            layer_idx,
            handle.config
        )
