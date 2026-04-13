"""Single-token dynamic boosting controller.

This controller is model-agnostic. It only requires callback functions for:
- one-token generation step,
- selector-request construction,
- optional immediate booster update side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .boundaries import BoundaryChecker
from .selector_protocol import InstructionSelector, normalize_selector_output
from .types import BoundaryEvent, DynamicRunTrace, SelectorDecision, SelectorRequest


@dataclass(frozen=True)
class TokenStepOutput:
    """Output of one model decoding step (exactly one token step)."""

    text: str
    token_id: int | None = None
    is_eos: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.token_id is not None and (isinstance(self.token_id, bool) or not isinstance(self.token_id, int)):
            raise TypeError("token_id must be an int or None")
        if not isinstance(self.is_eos, bool):
            raise TypeError("is_eos must be a bool")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")


@dataclass(frozen=True)
class DynamicControllerResult:
    """Result payload from one dynamic generation run."""

    generation_text: str
    final_active_instruction_ids: list[str]
    trace: DynamicRunTrace


def _unique_non_empty(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _default_fallback_decision(request: SelectorRequest) -> SelectorDecision:
    active = request.currently_active_instruction_ids or [request.candidate_instruction_ids[0]]
    return SelectorDecision(
        decision="stay",
        active_instruction_ids=active,
        confidence=0.0,
        reason="fallback_selector_default",
        metadata={"fallback": True},
    )


def _apply_decision(
    current_active_instruction_ids: list[str],
    decision: SelectorDecision,
    candidate_instruction_ids: list[str],
) -> list[str]:
    candidate_set = set(candidate_instruction_ids)

    if decision.decision == "add":
        merged = _unique_non_empty(current_active_instruction_ids + decision.active_instruction_ids)
        next_active = merged
    elif decision.decision == "switch":
        next_active = _unique_non_empty(decision.active_instruction_ids)
    else:  # "stay"
        # Stay keeps current active ids if available; otherwise use selector-provided ids.
        next_active = _unique_non_empty(current_active_instruction_ids or decision.active_instruction_ids)

    invalid = [x for x in next_active if x not in candidate_set]
    if invalid:
        raise ValueError(f"Decision produced unknown active ids: {invalid}")
    if not next_active:
        raise ValueError("Decision produced an empty active instruction set")
    return next_active


class DynamicBoostController:
    """Orchestrates boundary-triggered selector updates with one-token decoding."""

    def __init__(
        self,
        *,
        model_name: str,
        selector_backend: str,
        selector: InstructionSelector,
        boundary_checker: BoundaryChecker,
        step_fn: Callable[[list[str], int], TokenStepOutput],
        request_builder: Callable[[str, list[str], int, int], SelectorRequest],
        decode_config: dict[str, Any] | None = None,
        fallback_selector: Callable[[SelectorRequest, Exception], SelectorDecision] | None = None,
        on_selector_update: Callable[[list[str], SelectorDecision, BoundaryEvent], None] | None = None,
    ):
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if not selector_backend:
            raise ValueError("selector_backend cannot be empty")
        if not isinstance(boundary_checker, BoundaryChecker):
            raise TypeError("boundary_checker must be a BoundaryChecker")
        if not callable(step_fn):
            raise TypeError("step_fn must be callable")
        if not callable(request_builder):
            raise TypeError("request_builder must be callable")
        if fallback_selector is not None and not callable(fallback_selector):
            raise TypeError("fallback_selector must be callable when provided")
        if on_selector_update is not None and not callable(on_selector_update):
            raise TypeError("on_selector_update must be callable when provided")

        self.model_name = model_name
        self.selector_backend = selector_backend
        self.selector = selector
        self.boundary_checker = boundary_checker
        self.step_fn = step_fn
        self.request_builder = request_builder
        self.decode_config = decode_config or {}
        self.fallback_selector = fallback_selector
        self.on_selector_update = on_selector_update

    def run(
        self,
        *,
        sample_id: str,
        initial_active_instruction_ids: list[str],
        max_new_tokens: int,
    ) -> DynamicControllerResult:
        """Run dynamic generation with immediate post-boundary selector updates."""

        if not sample_id:
            raise ValueError("sample_id cannot be empty")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive int")
        if not isinstance(initial_active_instruction_ids, list):
            raise TypeError("initial_active_instruction_ids must be a list")

        self.boundary_checker.reset()

        active_ids = _unique_non_empty(initial_active_instruction_ids)
        current_text = ""
        generated_tokens = 0

        trace = DynamicRunTrace(
            sample_id=sample_id,
            model_name=self.model_name,
            selector_backend=self.selector_backend,
            decode_config=self.decode_config,
        )

        for step_index in range(1, max_new_tokens + 1):
            step_out = self.step_fn(active_ids, step_index)
            if not isinstance(step_out, TokenStepOutput):
                raise TypeError("step_fn must return TokenStepOutput")

            current_text += step_out.text
            generated_tokens += 1

            event = self.boundary_checker.ingest(step_out.text, total_generated_tokens=generated_tokens)
            if event is not None:
                trace.add_boundary_event(event)
                request = self.request_builder(current_text, active_ids, generated_tokens, step_index)
                if not isinstance(request, SelectorRequest):
                    raise TypeError("request_builder must return SelectorRequest")

                used_fallback = False
                try:
                    raw = self.selector.select(request)
                    decision = normalize_selector_output(raw)
                    decision.validate_candidates(request.candidate_instruction_ids)
                except Exception as exc:  # noqa: BLE001
                    used_fallback = True
                    if self.fallback_selector is not None:
                        decision = self.fallback_selector(request, exc)
                        if not isinstance(decision, SelectorDecision):
                            raise TypeError("fallback_selector must return SelectorDecision")
                    else:
                        decision = _default_fallback_decision(request)

                next_active = _apply_decision(
                    current_active_instruction_ids=active_ids,
                    decision=decision,
                    candidate_instruction_ids=request.candidate_instruction_ids,
                )

                trace.add_selector_decision(decision, used_fallback=used_fallback)
                active_ids = next_active

                if self.on_selector_update is not None:
                    self.on_selector_update(active_ids, decision, event)

            if step_out.is_eos:
                break

        trace.total_generated_tokens = generated_tokens
        return DynamicControllerResult(
            generation_text=current_text,
            final_active_instruction_ids=active_ids,
            trace=trace,
        )
