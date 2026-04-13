"""Core model-agnostic types for dynamic attention boosting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DecisionType = Literal["stay", "switch", "add"]
BoundaryReason = Literal["boundary", "max_tokens_fallback"]


@dataclass(frozen=True)
class BoundaryConfig:
    """Configuration for boundary-triggered selector updates."""

    min_tokens_between_checks: int = 8
    max_tokens_without_check: int = 32
    boundary_markers: tuple[str, ...] = (".", "?", "!", "\n\n")
    rolling_buffer_chars: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.min_tokens_between_checks, bool) or not isinstance(self.min_tokens_between_checks, int):
            raise TypeError("min_tokens_between_checks must be an int")
        if isinstance(self.max_tokens_without_check, bool) or not isinstance(self.max_tokens_without_check, int):
            raise TypeError("max_tokens_without_check must be an int")
        if isinstance(self.rolling_buffer_chars, bool) or not isinstance(self.rolling_buffer_chars, int):
            raise TypeError("rolling_buffer_chars must be an int")
        if self.min_tokens_between_checks < 1:
            raise ValueError("min_tokens_between_checks must be >= 1")
        if self.max_tokens_without_check < self.min_tokens_between_checks:
            raise ValueError("max_tokens_without_check must be >= min_tokens_between_checks")
        if not self.boundary_markers:
            raise ValueError("boundary_markers cannot be empty")
        if any((not isinstance(m, str) or not m) for m in self.boundary_markers):
            raise ValueError("boundary_markers must contain non-empty strings")
        if len(set(self.boundary_markers)) != len(self.boundary_markers):
            raise ValueError("boundary_markers contains duplicates")
        if self.rolling_buffer_chars < 1:
            raise ValueError("rolling_buffer_chars must be >= 1")


@dataclass(frozen=True)
class SelectorRequest:
    """Input contract passed to any instruction/rule selector backend."""

    sample_id: str
    base_prompt: str
    candidate_instruction_ids: list[str]
    instruction_text_by_id: dict[str, str]
    currently_active_instruction_ids: list[str]
    current_generation: str
    generation_token_count: int
    step_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not isinstance(self.base_prompt, str):
            raise TypeError("base_prompt must be a string")
        if not isinstance(self.candidate_instruction_ids, list) or not self.candidate_instruction_ids:
            raise ValueError("candidate_instruction_ids must be a non-empty list")
        if any((not isinstance(x, str) or not x) for x in self.candidate_instruction_ids):
            raise ValueError("candidate_instruction_ids must contain non-empty strings")
        if len(set(self.candidate_instruction_ids)) != len(self.candidate_instruction_ids):
            raise ValueError("candidate_instruction_ids must be unique")
        if not isinstance(self.instruction_text_by_id, dict):
            raise TypeError("instruction_text_by_id must be a dict")
        if not isinstance(self.currently_active_instruction_ids, list):
            raise TypeError("currently_active_instruction_ids must be a list")
        if any((not isinstance(x, str) or not x) for x in self.currently_active_instruction_ids):
            raise ValueError("currently_active_instruction_ids must contain non-empty strings")
        if len(set(self.currently_active_instruction_ids)) != len(self.currently_active_instruction_ids):
            raise ValueError("currently_active_instruction_ids must be unique")
        if not isinstance(self.current_generation, str):
            raise TypeError("current_generation must be a string")

        candidate_set = set(self.candidate_instruction_ids)
        missing = [x for x in self.candidate_instruction_ids if x not in self.instruction_text_by_id]
        if missing:
            raise ValueError(f"instruction_text_by_id missing candidate ids: {missing}")
        invalid_text_values = [
            x for x in self.candidate_instruction_ids
            if not isinstance(self.instruction_text_by_id[x], str) or not self.instruction_text_by_id[x]
        ]
        if invalid_text_values:
            raise ValueError(
                f"instruction_text_by_id must map candidate ids to non-empty strings: {invalid_text_values}"
            )

        invalid_active = [x for x in self.currently_active_instruction_ids if x not in candidate_set]
        if invalid_active:
            raise ValueError(f"currently_active_instruction_ids must be subset of candidates: {invalid_active}")

        if isinstance(self.generation_token_count, bool) or not isinstance(self.generation_token_count, int):
            raise TypeError("generation_token_count must be an int")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("step_index must be an int")
        if self.generation_token_count < 0:
            raise ValueError("generation_token_count must be >= 0")
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")


@dataclass(frozen=True)
class SelectorDecision:
    """Normalized selector output used by the dynamic controller."""

    decision: DecisionType
    active_instruction_ids: list[str]
    confidence: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"stay", "switch", "add"}:
            raise ValueError(f"Invalid decision '{self.decision}'")
        if not isinstance(self.active_instruction_ids, list) or not self.active_instruction_ids:
            raise ValueError("active_instruction_ids must be a non-empty list")
        if len(set(self.active_instruction_ids)) != len(self.active_instruction_ids):
            raise ValueError("active_instruction_ids must be unique")
        if any((not isinstance(x, str) or not x) for x in self.active_instruction_ids):
            raise ValueError("active_instruction_ids must contain non-empty strings")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be numeric")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def validate_candidates(self, candidate_instruction_ids: list[str]) -> None:
        """Validate decision IDs against a candidate set."""

        if not isinstance(candidate_instruction_ids, list) or not candidate_instruction_ids:
            raise ValueError("candidate_instruction_ids must be a non-empty list")
        if any((not isinstance(x, str) or not x) for x in candidate_instruction_ids):
            raise ValueError("candidate_instruction_ids must contain non-empty strings")
        candidate_set = set(candidate_instruction_ids)
        invalid = [x for x in self.active_instruction_ids if x not in candidate_set]
        if invalid:
            raise ValueError(f"active_instruction_ids contain unknown ids: {invalid}")


@dataclass(frozen=True)
class BoundaryEvent:
    """Telemetry event for a selector check trigger."""

    token_index: int
    reason: BoundaryReason
    text_suffix: str

    def __post_init__(self) -> None:
        if isinstance(self.token_index, bool) or not isinstance(self.token_index, int):
            raise TypeError("token_index must be an int")
        if self.token_index < 0:
            raise ValueError("token_index must be >= 0")
        if self.reason not in {"boundary", "max_tokens_fallback"}:
            raise ValueError(f"Invalid boundary reason '{self.reason}'")
        if not isinstance(self.text_suffix, str):
            raise TypeError("text_suffix must be a string")


@dataclass
class DynamicRunTrace:
    """Method-agnostic telemetry collected during dynamic generation."""

    sample_id: str
    model_name: str
    selector_backend: str
    decode_config: dict[str, Any] = field(default_factory=dict)
    boundary_events: list[BoundaryEvent] = field(default_factory=list)
    selector_decisions: list[SelectorDecision] = field(default_factory=list)
    fallback_count: int = 0
    total_generated_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not isinstance(self.selector_backend, str) or not self.selector_backend:
            raise ValueError("selector_backend cannot be empty")
        if not isinstance(self.decode_config, dict):
            raise TypeError("decode_config must be a dict")
        if not isinstance(self.boundary_events, list):
            raise TypeError("boundary_events must be a list")
        if not all(isinstance(x, BoundaryEvent) for x in self.boundary_events):
            raise TypeError("boundary_events must contain BoundaryEvent entries")
        if not isinstance(self.selector_decisions, list):
            raise TypeError("selector_decisions must be a list")
        if not all(isinstance(x, SelectorDecision) for x in self.selector_decisions):
            raise TypeError("selector_decisions must contain SelectorDecision entries")
        if isinstance(self.fallback_count, bool) or not isinstance(self.fallback_count, int):
            raise TypeError("fallback_count must be an int")
        if isinstance(self.total_generated_tokens, bool) or not isinstance(self.total_generated_tokens, int):
            raise TypeError("total_generated_tokens must be an int")
        if self.fallback_count < 0:
            raise ValueError("fallback_count must be >= 0")
        if self.total_generated_tokens < 0:
            raise ValueError("total_generated_tokens must be >= 0")

    @property
    def selector_calls(self) -> int:
        return len(self.selector_decisions)

    def add_boundary_event(self, event: BoundaryEvent) -> None:
        if not isinstance(event, BoundaryEvent):
            raise TypeError("event must be a BoundaryEvent")
        self.boundary_events.append(event)

    def add_selector_decision(self, decision: SelectorDecision, used_fallback: bool = False) -> None:
        if not isinstance(decision, SelectorDecision):
            raise TypeError("decision must be a SelectorDecision")
        if not isinstance(used_fallback, bool):
            raise TypeError("used_fallback must be a bool")
        self.selector_decisions.append(decision)
        if used_fallback:
            self.fallback_count += 1
