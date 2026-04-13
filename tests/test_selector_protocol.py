"""Tests for src.dynamic_boost.selector_protocol."""

import pytest

from src.dynamic_boost.selector_protocol import (
    decision_from_dict,
    ensure_selector,
    normalize_selector_output,
)
from src.dynamic_boost.types import SelectorDecision, SelectorRequest


class DummySelector:
    def select(self, request: SelectorRequest) -> SelectorDecision:
        return SelectorDecision(
            decision="stay",
            active_instruction_ids=request.currently_active_instruction_ids or [request.candidate_instruction_ids[0]],
            confidence=0.8,
            reason="dummy",
        )


class NotASelector:
    pass


def _request() -> SelectorRequest:
    return SelectorRequest(
        sample_id="s1",
        base_prompt="question",
        candidate_instruction_ids=["i1", "i2"],
        instruction_text_by_id={"i1": "first", "i2": "second"},
        currently_active_instruction_ids=["i1"],
        current_generation="partial answer",
        generation_token_count=10,
        step_index=1,
    )


def test_ensure_selector_accepts_protocol_implementation():
    selector = ensure_selector(DummySelector())
    out = selector.select(_request())
    assert out.decision == "stay"


def test_ensure_selector_rejects_invalid_object():
    with pytest.raises(TypeError, match="InstructionSelector"):
        ensure_selector(NotASelector())


def test_decision_from_dict_and_normalize():
    raw = {
        "decision": "switch",
        "active_instruction_ids": ["i2"],
        "confidence": 0.92,
        "reason": "switch now",
        "metadata": {"source": "external"},
    }

    d1 = decision_from_dict(raw)
    d2 = normalize_selector_output(raw)

    assert d1.decision == "switch"
    assert d1.active_instruction_ids == ["i2"]
    assert d2 == d1


def test_decision_from_dict_missing_required_field():
    with pytest.raises(ValueError, match="Missing selector decision field"):
        decision_from_dict({"decision": "stay", "confidence": 0.5})


def test_normalize_selector_output_rejects_wrong_type():
    with pytest.raises(TypeError, match="Selector output"):
        normalize_selector_output("bad")


def test_decision_from_dict_rejects_bad_metadata_type():
    with pytest.raises(ValueError, match="metadata"):
        decision_from_dict(
            {
                "decision": "stay",
                "active_instruction_ids": ["i1"],
                "confidence": 0.5,
                "metadata": "not-a-dict",
            }
        )


def test_decision_from_dict_rejects_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        decision_from_dict("bad")  # type: ignore[arg-type]
