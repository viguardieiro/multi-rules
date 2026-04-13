"""Tests for src.dynamic_boost.types."""

import pytest

from src.dynamic_boost.types import (
    BoundaryConfig,
    BoundaryEvent,
    DynamicRunTrace,
    SelectorDecision,
    SelectorRequest,
)


def test_boundary_config_valid_defaults():
    cfg = BoundaryConfig()
    assert cfg.min_tokens_between_checks == 8
    assert cfg.max_tokens_without_check == 32
    assert cfg.boundary_markers == (".", "?", "!", "\n\n")


def test_boundary_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="max_tokens_without_check"):
        BoundaryConfig(min_tokens_between_checks=10, max_tokens_without_check=5)


def test_boundary_config_rejects_non_int_values():
    with pytest.raises(TypeError, match="must be an int"):
        BoundaryConfig(min_tokens_between_checks=1.5, max_tokens_without_check=8)
    with pytest.raises(TypeError, match="must be an int"):
        BoundaryConfig(min_tokens_between_checks=2, max_tokens_without_check=8, rolling_buffer_chars=4.2)


@pytest.mark.parametrize(
    "markers",
    [
        (),
        (".", "."),
        ("",),
    ],
)
def test_boundary_config_rejects_bad_markers(markers):
    with pytest.raises(ValueError):
        BoundaryConfig(boundary_markers=markers)


def test_selector_request_requires_candidate_texts_and_subset_active_ids():
    with pytest.raises(ValueError, match="missing candidate ids"):
        SelectorRequest(
            sample_id="s1",
            base_prompt="q",
            candidate_instruction_ids=["i1", "i2"],
            instruction_text_by_id={"i1": "one"},
            currently_active_instruction_ids=["i1"],
            current_generation="",
            generation_token_count=0,
            step_index=0,
        )

    with pytest.raises(ValueError, match="subset of candidates"):
        SelectorRequest(
            sample_id="s1",
            base_prompt="q",
            candidate_instruction_ids=["i1", "i2"],
            instruction_text_by_id={"i1": "one", "i2": "two"},
            currently_active_instruction_ids=["i3"],
            current_generation="",
            generation_token_count=0,
            step_index=0,
        )


def test_selector_decision_candidate_validation():
    d = SelectorDecision(
        decision="switch",
        active_instruction_ids=["i2"],
        confidence=0.9,
        reason="need next rule",
    )
    d.validate_candidates(["i1", "i2"])

    with pytest.raises(ValueError, match="unknown ids"):
        d.validate_candidates(["i1"])


@pytest.mark.parametrize("confidence", [-0.01, 1.1])
def test_selector_decision_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        SelectorDecision(
            decision="stay",
            active_instruction_ids=["i1"],
            confidence=confidence,
        )


def test_selector_decision_rejects_bool_confidence():
    with pytest.raises(TypeError, match="numeric"):
        SelectorDecision(
            decision="stay",
            active_instruction_ids=["i1"],
            confidence=True,
        )


def test_selector_request_rejects_non_string_instruction_text():
    with pytest.raises(ValueError, match="instruction_text_by_id"):
        SelectorRequest(
            sample_id="s1",
            base_prompt="q",
            candidate_instruction_ids=["i1"],
            instruction_text_by_id={"i1": 123},
            currently_active_instruction_ids=[],
            current_generation="",
            generation_token_count=0,
            step_index=0,
        )


def test_dynamic_run_trace_counters_and_events():
    trace = DynamicRunTrace(
        sample_id="sample-1",
        model_name="openai/gpt-oss-20b",
        selector_backend="ollama",
        decode_config={"max_new_tokens": 64},
    )

    trace.add_boundary_event(BoundaryEvent(token_index=5, reason="boundary", text_suffix="end."))
    trace.add_selector_decision(
        SelectorDecision(
            decision="add",
            active_instruction_ids=["i1", "i2"],
            confidence=0.7,
        ),
        used_fallback=True,
    )

    assert len(trace.boundary_events) == 1
    assert trace.selector_calls == 1
    assert trace.fallback_count == 1


def test_dynamic_run_trace_rejects_non_typed_events():
    trace = DynamicRunTrace(
        sample_id="sample-2",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        selector_backend="ollama",
    )
    with pytest.raises(TypeError, match="BoundaryEvent"):
        trace.add_boundary_event("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SelectorDecision"):
        trace.add_selector_decision("bad")  # type: ignore[arg-type]
