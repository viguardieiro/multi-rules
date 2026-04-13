"""Model-agnostic dynamic attention boosting interfaces."""

from .types import (
    BoundaryConfig,
    BoundaryEvent,
    DynamicRunTrace,
    SelectorDecision,
    SelectorRequest,
)
from .boundaries import BoundaryChecker, BoundaryCheckerState
from .controller import DynamicBoostController, DynamicControllerResult, TokenStepOutput
from .selector_protocol import (
    InstructionSelector,
    decision_from_dict,
    ensure_selector,
    normalize_selector_output,
)
from .selector_llm import (
    DEFAULT_SELECTOR_SYSTEM_PROMPT,
    DeterministicFallbackSelector,
    LLMInstructionSelector,
    OllamaSelectorBackend,
    SelectorLLMBackend,
    SelectorParseError,
    build_selector_payload,
    parse_selector_response,
    sanitize_raw_output,
)
from .result_schema import (
    SCHEMA_VERSION,
    build_run_result,
    validate_run_result_schema,
)

__all__ = [
    "BoundaryConfig",
    "BoundaryChecker",
    "BoundaryCheckerState",
    "BoundaryEvent",
    "DynamicBoostController",
    "DynamicControllerResult",
    "DynamicRunTrace",
    "SelectorDecision",
    "SelectorRequest",
    "TokenStepOutput",
    "InstructionSelector",
    "SelectorLLMBackend",
    "OllamaSelectorBackend",
    "LLMInstructionSelector",
    "DeterministicFallbackSelector",
    "SelectorParseError",
    "DEFAULT_SELECTOR_SYSTEM_PROMPT",
    "build_selector_payload",
    "parse_selector_response",
    "sanitize_raw_output",
    "decision_from_dict",
    "ensure_selector",
    "normalize_selector_output",
    "SCHEMA_VERSION",
    "build_run_result",
    "validate_run_result_schema",
]
