"""Tests for deterministic selector fallback behavior."""

from __future__ import annotations

from src.dynamic_boost.selector_llm import DeterministicFallbackSelector
from src.dynamic_boost.types import SelectorRequest


def _request(active_ids):
    return SelectorRequest(
        sample_id="s1",
        base_prompt="Q",
        candidate_instruction_ids=["i1", "i2", "i3"],
        instruction_text_by_id={"i1": "one", "i2": "two", "i3": "three"},
        currently_active_instruction_ids=active_ids,
        current_generation="partial",
        generation_token_count=5,
        step_index=1,
    )


def test_fallback_keeps_current_when_available():
    fallback = DeterministicFallbackSelector(keep_current_if_possible=True)
    decision = fallback(_request(["i2"]), RuntimeError("boom"))

    assert decision.decision == "stay"
    assert decision.active_instruction_ids == ["i2"]
    assert decision.metadata["fallback"] is True


def test_fallback_switches_to_first_candidate_when_no_active():
    fallback = DeterministicFallbackSelector(keep_current_if_possible=True)
    decision = fallback(_request([]), ValueError("bad"))

    assert decision.decision == "switch"
    assert decision.active_instruction_ids == ["i1"]
    assert decision.reason.startswith("fallback_due_to_")


def test_fallback_can_force_switch_even_if_active_exists():
    fallback = DeterministicFallbackSelector(keep_current_if_possible=False)
    decision = fallback(_request(["i3"]), Exception("x"))

    assert decision.decision == "switch"
    assert decision.active_instruction_ids == ["i1"]
