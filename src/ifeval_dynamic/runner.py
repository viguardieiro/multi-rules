"""IFEval dynamic boosting runner utilities and execution loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import BoostConfig, TokenSubset, register_boost_hooks, unregister_boost_hooks, update_bias_mask
from src.dynamic_boost import (
    BoundaryChecker,
    BoundaryConfig,
    DynamicBoostController,
    LLMInstructionSelector,
    OllamaSelectorBackend,
    SelectorDecision,
    SelectorRequest,
    TokenStepOutput,
)

from .data_adapter import IFEvalSample
from .eval_adapter import compute_ifeval_aggregate_metrics, evaluate_ifeval_sample
from .instruction_spans import InstructionSpan, compute_instruction_spans
from .selector_context import build_selector_context


@dataclass(frozen=True)
class GenerationConfig:
    """Decode settings shared across methods for fair comparison."""

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def instruction_spans_to_index_map(spans: list[InstructionSpan]) -> dict[str, list[int]]:
    """Convert per-instruction token spans to explicit token index lists."""

    out: dict[str, list[int]] = {}
    for span in spans:
        if span.end_token <= span.start_token:
            raise ValueError(f"Invalid span for '{span.instruction_id}': [{span.start_token}, {span.end_token})")
        out[span.instruction_id] = list(range(span.start_token, span.end_token))
    return out


def build_active_boost_config(
    instruction_token_indices: dict[str, list[int]],
    active_instruction_ids: list[str],
    *,
    boost_bias: float,
) -> BoostConfig:
    """Build BoostConfig for currently active instruction IDs."""

    if isinstance(boost_bias, bool) or not isinstance(boost_bias, (int, float)):
        raise TypeError("boost_bias must be numeric")

    if not active_instruction_ids:
        raise ValueError("active_instruction_ids cannot be empty")

    subsets: list[TokenSubset] = []
    for inst_id in active_instruction_ids:
        if inst_id not in instruction_token_indices:
            raise ValueError(f"Unknown instruction id in active set: '{inst_id}'")
        indices = instruction_token_indices[inst_id]
        if not indices:
            raise ValueError(f"Instruction '{inst_id}' has empty token indices")
        subsets.append(TokenSubset(name=inst_id, indices=indices, bias=float(boost_bias)))

    return BoostConfig(subsets=subsets)


def selector_request_from_context(ctx: dict[str, Any]) -> SelectorRequest:
    """Construct a typed selector request from IFEval selector context payload."""

    return SelectorRequest(
        sample_id=ctx["sample_id"],
        base_prompt=ctx["base_prompt"],
        candidate_instruction_ids=ctx["candidate_instruction_ids"],
        instruction_text_by_id=ctx["instruction_text_by_id"],
        currently_active_instruction_ids=ctx["currently_active_instruction_ids"],
        current_generation=ctx["current_generation"],
        generation_token_count=ctx["generation_token_count"],
        step_index=ctx["step_index"],
        metadata=ctx.get("metadata", {}),
    )


def trace_to_dict(trace: Any) -> dict[str, Any]:
    """Serialize DynamicRunTrace to plain JSON-compatible dict."""

    return {
        "sample_id": trace.sample_id,
        "model_name": trace.model_name,
        "selector_backend": trace.selector_backend,
        "decode_config": dict(trace.decode_config),
        "boundary_events": [asdict(x) for x in trace.boundary_events],
        "selector_decisions": [asdict(x) for x in trace.selector_decisions],
        "fallback_count": trace.fallback_count,
        "selector_calls": trace.selector_calls,
        "total_generated_tokens": trace.total_generated_tokens,
    }


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")

    warped = logits / float(temperature)
    probs = torch.softmax(warped, dim=-1)

    if top_k > 0 and top_k < probs.numel():
        top_values, top_indices = torch.topk(probs, k=top_k, dim=-1)
        masked = torch.zeros_like(probs)
        masked.scatter_(dim=-1, index=top_indices, src=top_values)
        probs = masked / masked.sum(dim=-1, keepdim=True)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        masked = torch.zeros_like(probs)
        kept_values = sorted_probs * keep
        masked.scatter_(dim=-1, index=sorted_indices, src=kept_values)
        total = masked.sum(dim=-1, keepdim=True)
        probs = masked / torch.clamp_min(total, 1e-12)

    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


class _IncrementalGenerator:
    """One-token-at-a-time generator with KV cache for dynamic control."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        full_input: str,
        generation_config: GenerationConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = generation_config
        self.device = device

        encoded = tokenizer(full_input, return_tensors="pt")
        self.prompt_input_ids = encoded["input_ids"].to(device)
        self.prompt_attention_mask = encoded.get("attention_mask")
        if self.prompt_attention_mask is None:
            self.prompt_attention_mask = torch.ones_like(self.prompt_input_ids, dtype=torch.long, device=device)
        else:
            self.prompt_attention_mask = self.prompt_attention_mask.to(device)
        self.full_attention_mask = self.prompt_attention_mask.clone()

        self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())

        eos_ids = set()
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            eos_ids.add(int(eos))
        eos_list = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_list, int):
            eos_ids.add(int(eos_list))
        elif isinstance(eos_list, (list, tuple)):
            eos_ids.update(int(x) for x in eos_list)
        self.eos_ids = eos_ids

        self.past_key_values = None
        self.next_input_ids = None
        self.generated_token_ids: list[int] = []

    @property
    def prompt_length(self) -> int:
        return int(self.prompt_input_ids.shape[1])

    def step(self, _active_instruction_ids: list[str], _step_index: int) -> TokenStepOutput:
        with torch.inference_mode():
            if self.past_key_values is None:
                outputs = self.model(
                    input_ids=self.prompt_input_ids,
                    attention_mask=self.full_attention_mask,
                    use_cache=True,
                )
            else:
                kwargs = {
                    "input_ids": self.next_input_ids,
                    "use_cache": True,
                    "past_key_values": self.past_key_values,
                    "attention_mask": self.full_attention_mask,
                }

                if "position_ids" in self._forward_params:
                    position_ids = (self.full_attention_mask.long().cumsum(dim=-1) - 1)[:, -1:]
                    kwargs["position_ids"] = position_ids
                if "cache_position" in self._forward_params:
                    cache_pos = torch.tensor([self.full_attention_mask.shape[1] - 1], device=self.device)
                    kwargs["cache_position"] = cache_pos

                outputs = self.model(**kwargs)

        self.past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = _sample_next_token(
            next_token_logits,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )

        self.generated_token_ids.append(next_token_id)
        self.next_input_ids = torch.tensor([[next_token_id]], device=self.device)
        mask_extension = torch.ones(
            (self.full_attention_mask.shape[0], 1),
            dtype=self.full_attention_mask.dtype,
            device=self.device,
        )
        self.full_attention_mask = torch.cat([self.full_attention_mask, mask_extension], dim=-1)

        token_text = self.tokenizer.decode(
            [next_token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return TokenStepOutput(
            text=token_text,
            token_id=next_token_id,
            is_eos=(next_token_id in self.eos_ids),
        )


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _pick_dtype(dtype_arg: str) -> torch.dtype | None:
    if dtype_arg == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else None
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def load_transformers_model_and_tokenizer(
    *,
    model_name: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[Any, Any, torch.device]:
    """Load HF model/tokenizer for generation and return (model, tokenizer, device)."""

    target_device = _pick_device(device)
    torch_dtype = _pick_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(target_device)
    model.eval()
    return model, tokenizer, target_device


def build_selector(
    *,
    selector_backend: str,
    selector_model: str,
    selector_base_url: str,
    selector_timeout_s: float,
    selector_retries: int,
    logger: Any = None,
) -> Any:
    """Create selector backend instance for dynamic controller."""

    if selector_backend != "ollama":
        raise ValueError("Only selector_backend='ollama' is currently supported")

    backend = OllamaSelectorBackend(
        model=selector_model,
        base_url=selector_base_url,
        temperature=0.0,
    )
    return LLMInstructionSelector(
        backend=backend,
        timeout_s=selector_timeout_s,
        max_retries=selector_retries,
        retry_backoff_s=0.2,
        logger=logger,
    )


def run_dynamic_sample(
    *,
    sample: IFEvalSample,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    selector: Any,
    selector_backend_name: str,
    generation_config: GenerationConfig,
    boundary_config: BoundaryConfig,
    boost_bias: float,
    fallback_selector: Any | None = None,
) -> dict[str, Any]:
    """Run dynamic boosted generation + evaluation for one IFEval sample."""

    spans = compute_instruction_spans(
        tokenizer,
        full_input=sample.full_input,
        instruction_id_list=sample.instruction_id_list,
        instruction_texts=sample.instruction_texts,
        instruction_block=sample.instruction_block,
    )
    instruction_token_indices = instruction_spans_to_index_map(spans)
    initial_active_instruction_ids = [sample.instruction_id_list[0]]

    boost_config = build_active_boost_config(
        instruction_token_indices,
        initial_active_instruction_ids,
        boost_bias=boost_bias,
    )

    generator = _IncrementalGenerator(
        model=model,
        tokenizer=tokenizer,
        full_input=sample.full_input,
        generation_config=generation_config,
        device=device,
    )

    handle = register_boost_hooks(model, boost_config, input_length=generator.prompt_length)
    update_bias_mask(handle, seq_length=generator.prompt_length, device=device)

    def request_builder(
        current_generation: str,
        active_ids: list[str],
        generation_token_count: int,
        step_index: int,
    ) -> SelectorRequest:
        ctx = build_selector_context(
            sample=sample,
            current_generation=current_generation,
            active_instruction_ids=active_ids,
            generation_token_count=generation_token_count,
            step_index=step_index,
        )
        return selector_request_from_context(ctx)

    def on_selector_update(next_active_ids: list[str], _decision: SelectorDecision, _event: Any) -> None:
        handle.config = build_active_boost_config(
            instruction_token_indices,
            next_active_ids,
            boost_bias=boost_bias,
        )
        update_bias_mask(handle, seq_length=generator.prompt_length, device=device)

    controller = DynamicBoostController(
        model_name=str(getattr(model, "name_or_path", "transformers_model")),
        selector_backend=selector_backend_name,
        selector=selector,
        boundary_checker=BoundaryChecker(boundary_config),
        step_fn=generator.step,
        request_builder=request_builder,
        decode_config=generation_config.to_dict(),
        fallback_selector=fallback_selector,
        on_selector_update=on_selector_update,
    )

    try:
        run_result = controller.run(
            sample_id=sample.sample_id,
            initial_active_instruction_ids=initial_active_instruction_ids,
            max_new_tokens=generation_config.max_new_tokens,
        )
    finally:
        unregister_boost_hooks(handle)

    eval_result = evaluate_ifeval_sample(
        run_result.generation_text,
        instruction_id_list=sample.instruction_id_list,
        kwargs_list=sample.kwargs_list,
    )

    return {
        "sample_id": sample.sample_id,
        "generation": run_result.generation_text,
        "strict_following": bool(eval_result["sample_score"]),
        "instruction_level_score": float(eval_result["instruction_level_score"]),
        "per_instruction_eval": eval_result["per_instruction"],
        "method_metadata": {
            "active_instruction_ids_initial": initial_active_instruction_ids,
            "active_instruction_ids_final": run_result.final_active_instruction_ids,
            "dynamic_trace": trace_to_dict(run_result.trace),
            "instruction_token_spans": [asdict(x) for x in spans],
        },
    }


def run_dynamic_benchmark(
    *,
    samples: list[IFEvalSample],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    selector: Any,
    selector_backend_name: str,
    generation_config: GenerationConfig,
    boundary_config: BoundaryConfig,
    boost_bias: float,
    fallback_selector: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run dynamic benchmark over a list of samples."""

    per_sample_results = []
    eval_results = []
    for sample in samples:
        sample_result = run_dynamic_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            device=device,
            selector=selector,
            selector_backend_name=selector_backend_name,
            generation_config=generation_config,
            boundary_config=boundary_config,
            boost_bias=boost_bias,
            fallback_selector=fallback_selector,
        )
        per_sample_results.append(sample_result)
        eval_results.append(
            {
                "per_instruction": sample_result["per_instruction_eval"],
                "sample_score": 1 if sample_result["strict_following"] else 0,
            }
        )

    aggregate = compute_ifeval_aggregate_metrics(eval_results)
    return per_sample_results, aggregate


def _run_plain_generation(
    *,
    generator: _IncrementalGenerator,
    max_new_tokens: int,
) -> tuple[str, int]:
    chunks: list[str] = []
    generated = 0
    for step_index in range(1, max_new_tokens + 1):
        step = generator.step([], step_index)
        chunks.append(step.text)
        generated += 1
        if step.is_eos:
            break
    return "".join(chunks), generated


def run_baseline_sample(
    *,
    sample: IFEvalSample,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    """Run one IFEval sample without boosting."""

    generator = _IncrementalGenerator(
        model=model,
        tokenizer=tokenizer,
        full_input=sample.full_input,
        generation_config=generation_config,
        device=device,
    )
    generation_text, generated_tokens = _run_plain_generation(
        generator=generator,
        max_new_tokens=generation_config.max_new_tokens,
    )

    eval_result = evaluate_ifeval_sample(
        generation_text,
        instruction_id_list=sample.instruction_id_list,
        kwargs_list=sample.kwargs_list,
    )
    return {
        "sample_id": sample.sample_id,
        "generation": generation_text,
        "strict_following": bool(eval_result["sample_score"]),
        "instruction_level_score": float(eval_result["instruction_level_score"]),
        "per_instruction_eval": eval_result["per_instruction"],
        "method_metadata": {
            "mode": "baseline",
            "generated_tokens": generated_tokens,
        },
    }


def run_static_sample(
    *,
    sample: IFEvalSample,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    generation_config: GenerationConfig,
    boost_bias: float,
    active_instruction_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run one IFEval sample with fixed active instruction boosts."""

    spans = compute_instruction_spans(
        tokenizer,
        full_input=sample.full_input,
        instruction_id_list=sample.instruction_id_list,
        instruction_texts=sample.instruction_texts,
        instruction_block=sample.instruction_block,
    )
    instruction_token_indices = instruction_spans_to_index_map(spans)
    fixed_ids = list(active_instruction_ids or sample.instruction_id_list)
    boost_config = build_active_boost_config(
        instruction_token_indices=instruction_token_indices,
        active_instruction_ids=fixed_ids,
        boost_bias=boost_bias,
    )

    generator = _IncrementalGenerator(
        model=model,
        tokenizer=tokenizer,
        full_input=sample.full_input,
        generation_config=generation_config,
        device=device,
    )

    handle = register_boost_hooks(model, boost_config, input_length=generator.prompt_length)
    update_bias_mask(handle, seq_length=generator.prompt_length, device=device)
    try:
        generation_text, generated_tokens = _run_plain_generation(
            generator=generator,
            max_new_tokens=generation_config.max_new_tokens,
        )
    finally:
        unregister_boost_hooks(handle)

    eval_result = evaluate_ifeval_sample(
        generation_text,
        instruction_id_list=sample.instruction_id_list,
        kwargs_list=sample.kwargs_list,
    )
    return {
        "sample_id": sample.sample_id,
        "generation": generation_text,
        "strict_following": bool(eval_result["sample_score"]),
        "instruction_level_score": float(eval_result["instruction_level_score"]),
        "per_instruction_eval": eval_result["per_instruction"],
        "method_metadata": {
            "mode": "static_instaboost",
            "generated_tokens": generated_tokens,
            "active_instruction_ids_fixed": fixed_ids,
            "instruction_token_spans": [asdict(x) for x in spans],
            "boost_bias": float(boost_bias),
        },
    }


def run_baseline_benchmark(
    *,
    samples: list[IFEvalSample],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    generation_config: GenerationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run baseline benchmark over a list of samples."""

    per_sample_results = []
    eval_results = []
    for sample in samples:
        sample_result = run_baseline_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            device=device,
            generation_config=generation_config,
        )
        per_sample_results.append(sample_result)
        eval_results.append(
            {
                "per_instruction": sample_result["per_instruction_eval"],
                "sample_score": 1 if sample_result["strict_following"] else 0,
            }
        )
    aggregate = compute_ifeval_aggregate_metrics(eval_results)
    return per_sample_results, aggregate


def run_static_benchmark(
    *,
    samples: list[IFEvalSample],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    generation_config: GenerationConfig,
    boost_bias: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run static instaboost benchmark over a list of samples."""

    per_sample_results = []
    eval_results = []
    for sample in samples:
        sample_result = run_static_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            device=device,
            generation_config=generation_config,
            boost_bias=boost_bias,
        )
        per_sample_results.append(sample_result)
        eval_results.append(
            {
                "per_instruction": sample_result["per_instruction_eval"],
                "sample_score": 1 if sample_result["strict_following"] else 0,
            }
        )
    aggregate = compute_ifeval_aggregate_metrics(eval_results)
    return per_sample_results, aggregate
