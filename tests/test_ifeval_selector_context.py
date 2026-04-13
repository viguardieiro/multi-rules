"""Tests for src.ifeval_dynamic.selector_context."""

from __future__ import annotations

import pytest

from src.ifeval_dynamic.data_adapter import IFEvalSample
from src.ifeval_dynamic.selector_context import build_selector_context


def _sample() -> IFEvalSample:
    return IFEvalSample(
        key=7,
        sample_id="ifeval_7",
        original_prompt="Original prompt",
        base_question="Base question",
        instruction_id_list=["i1", "i2"],
        kwargs_list=[{}, {}],
        instruction_texts=["First", "Second"],
        instruction_block="Your response should follow the instructions below:\n- First\n- Second",
        full_input="Base question\n\nYour response should follow the instructions below:\n- First\n- Second",
    )


def test_build_selector_context_basic():
    sample = _sample()
    ctx = build_selector_context(
        sample=sample,
        current_generation="hello world",
        active_instruction_ids=["i1"],
        generation_token_count=10,
        step_index=2,
        max_generation_chars=5,
    )

    assert ctx["sample_id"] == "ifeval_7"
    assert ctx["base_prompt"] == sample.full_input
    assert ctx["candidate_instruction_ids"] == ["i1", "i2"]
    assert ctx["instruction_text_by_id"]["i2"] == "Second"
    assert ctx["current_generation"] == "world"
    assert ctx["metadata"]["ifeval_key"] == 7


def test_build_selector_context_rejects_unknown_active_ids():
    with pytest.raises(ValueError, match="unknown ids"):
        build_selector_context(
            sample=_sample(),
            current_generation="x",
            active_instruction_ids=["bad"],
            generation_token_count=1,
            step_index=1,
        )
