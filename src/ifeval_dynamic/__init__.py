"""IFEval-specific adapters for dynamic attention boosting experiments."""

from .data_adapter import IFEvalSample
from .eval_adapter import evaluate_ifeval_sample
from .instruction_spans import InstructionSpan
from .runner import (
    GenerationConfig,
    build_active_boost_config,
    instruction_spans_to_index_map,
    run_baseline_benchmark,
    run_dynamic_benchmark,
    run_static_benchmark,
    selector_request_from_context,
    trace_to_dict,
)
from .selector_context import build_selector_context

__all__ = [
    "IFEvalSample",
    "InstructionSpan",
    "GenerationConfig",
    "instruction_spans_to_index_map",
    "build_active_boost_config",
    "run_baseline_benchmark",
    "run_static_benchmark",
    "run_dynamic_benchmark",
    "selector_request_from_context",
    "trace_to_dict",
    "build_selector_context",
    "evaluate_ifeval_sample",
]
