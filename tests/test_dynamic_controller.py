"""Tests for src.dynamic_boost.controller."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.dynamic_boost.boundaries import BoundaryChecker
from src.dynamic_boost.controller import DynamicBoostController, TokenStepOutput
from src.dynamic_boost.types import BoundaryConfig, SelectorDecision, SelectorRequest


@dataclass
class ScriptedSelector:
    decisions: list[SelectorDecision]

    def __post_init__(self) -> None:
        self.calls = 0

    def select(self, request: SelectorRequest) -> SelectorDecision:
        idx = self.calls
        self.calls += 1
        return self.decisions[idx]


def _request_builder_factory(candidate_ids: list[str]):
    def _build(current_text: str, active_ids: list[str], generated_tokens: int, step_index: int) -> SelectorRequest:
        return SelectorRequest(
            sample_id="sample-1",
            base_prompt="Q",
            candidate_instruction_ids=candidate_ids,
            instruction_text_by_id={cid: cid for cid in candidate_ids},
            currently_active_instruction_ids=active_ids,
            current_generation=current_text,
            generation_token_count=generated_tokens,
            step_index=step_index,
            metadata={},
        )

    return _build


def test_single_token_loop_switches_before_next_token():
    # Boundaries happen on tokens 2 and 4. We verify active ids seen by step_fn:
    # token 3 must use switch from token 2, token 5 must use switch from token 4.
    tokens = ["A", ".", "B", "!", "C"]
    seen_active = []

    def step_fn(active_ids: list[str], step_index: int) -> TokenStepOutput:
        seen_active.append((step_index, list(active_ids)))
        text = tokens[step_index - 1]
        return TokenStepOutput(text=text, token_id=step_index, is_eos=(step_index == len(tokens)))

    selector = ScriptedSelector(
        decisions=[
            SelectorDecision(decision="switch", active_instruction_ids=["i2"], confidence=0.9),
            SelectorDecision(decision="switch", active_instruction_ids=["i3"], confidence=0.9),
        ]
    )

    updates = []

    def on_update(active_ids, decision, event):
        updates.append((list(active_ids), decision.decision, event.reason, event.token_index))

    controller = DynamicBoostController(
        model_name="test-model",
        selector_backend="test-backend",
        selector=selector,
        boundary_checker=BoundaryChecker(BoundaryConfig(min_tokens_between_checks=1, max_tokens_without_check=100)),
        step_fn=step_fn,
        request_builder=_request_builder_factory(["i1", "i2", "i3"]),
        decode_config={"max_new_tokens": 8},
        on_selector_update=on_update,
    )

    result = controller.run(
        sample_id="sample-1",
        initial_active_instruction_ids=["i1"],
        max_new_tokens=10,
    )

    # Step 1 and 2 use i1. Boundary on step 2 switches to i2 for step 3.
    assert seen_active[0] == (1, ["i1"])
    assert seen_active[1] == (2, ["i1"])
    assert seen_active[2] == (3, ["i2"])
    # Boundary on step 4 switches to i3 for step 5.
    assert seen_active[4] == (5, ["i3"])

    assert result.generation_text == "A.B!C"
    assert result.final_active_instruction_ids == ["i3"]
    assert result.trace.selector_calls == 2
    assert [u[3] for u in updates] == [2, 4]


def test_fallback_used_when_selector_raises():
    tokens = ["x", ".", "y"]

    def step_fn(active_ids: list[str], step_index: int) -> TokenStepOutput:
        return TokenStepOutput(text=tokens[step_index - 1], token_id=step_index, is_eos=(step_index == len(tokens)))

    class FailingSelector:
        def select(self, request: SelectorRequest) -> SelectorDecision:
            raise RuntimeError("selector failure")

    def fallback_selector(request: SelectorRequest, exc: Exception) -> SelectorDecision:
        assert "failure" in str(exc)
        return SelectorDecision(
            decision="switch",
            active_instruction_ids=["i2"],
            confidence=0.2,
            reason="fallback",
            metadata={"fallback": True},
        )

    controller = DynamicBoostController(
        model_name="test-model",
        selector_backend="test-backend",
        selector=FailingSelector(),
        boundary_checker=BoundaryChecker(BoundaryConfig(min_tokens_between_checks=1, max_tokens_without_check=100)),
        step_fn=step_fn,
        request_builder=_request_builder_factory(["i1", "i2"]),
        fallback_selector=fallback_selector,
    )

    result = controller.run(
        sample_id="sample-fallback",
        initial_active_instruction_ids=["i1"],
        max_new_tokens=5,
    )

    assert result.trace.selector_calls == 1
    assert result.trace.fallback_count == 1
    assert result.final_active_instruction_ids == ["i2"]


def test_selector_called_on_max_token_fallback_without_boundary():
    tokens = ["a", "b", "c", "d", "e", "f"]

    def step_fn(active_ids: list[str], step_index: int) -> TokenStepOutput:
        return TokenStepOutput(text=tokens[step_index - 1], token_id=step_index, is_eos=(step_index == len(tokens)))

    selector = ScriptedSelector(
        decisions=[
            SelectorDecision(decision="stay", active_instruction_ids=["i1"], confidence=0.9),
            SelectorDecision(decision="stay", active_instruction_ids=["i1"], confidence=0.9),
        ]
    )

    controller = DynamicBoostController(
        model_name="test-model",
        selector_backend="test-backend",
        selector=selector,
        boundary_checker=BoundaryChecker(BoundaryConfig(min_tokens_between_checks=2, max_tokens_without_check=3)),
        step_fn=step_fn,
        request_builder=_request_builder_factory(["i1"]),
    )

    result = controller.run(
        sample_id="sample-fallback-trigger",
        initial_active_instruction_ids=["i1"],
        max_new_tokens=6,
    )

    assert result.trace.selector_calls == 2
    assert [e.reason for e in result.trace.boundary_events] == ["max_tokens_fallback", "max_tokens_fallback"]


def test_controller_rejects_bad_step_output_type():
    def step_fn(active_ids: list[str], step_index: int):
        return "not-a-step"  # type: ignore[return-value]

    selector = ScriptedSelector(
        decisions=[SelectorDecision(decision="stay", active_instruction_ids=["i1"], confidence=0.9)]
    )

    controller = DynamicBoostController(
        model_name="test-model",
        selector_backend="test-backend",
        selector=selector,
        boundary_checker=BoundaryChecker(BoundaryConfig()),
        step_fn=step_fn,
        request_builder=_request_builder_factory(["i1"]),
    )

    with pytest.raises(TypeError, match="TokenStepOutput"):
        controller.run(sample_id="sample-err", initial_active_instruction_ids=["i1"], max_new_tokens=1)
