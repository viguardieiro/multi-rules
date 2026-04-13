"""Tests for src.ifeval_dynamic.instruction_spans."""

from __future__ import annotations

import pytest

from src.ifeval_dynamic.instruction_spans import (
    InstructionSpan,
    compute_instruction_block_token_span,
    compute_instruction_spans,
)


class _CharTokenizer:
    """Tokenizer stub where each character is one token with exact offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if not return_offsets_mapping:
            raise ValueError("return_offsets_mapping=True required")
        return {
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_compute_instruction_block_token_span_char_tokenizer():
    tokenizer = _CharTokenizer()
    full_input = "Q\n\nYour response should follow the instructions below:\n- A\n- B"
    block = "Your response should follow the instructions below:\n- A\n- B"

    start, end = compute_instruction_block_token_span(tokenizer, full_input, block)
    expected_start = full_input.find(block)
    expected_end = expected_start + len(block)

    assert start == expected_start
    assert end == expected_end


def test_compute_instruction_spans_char_tokenizer():
    tokenizer = _CharTokenizer()
    full_input = "Q\n\nYour response should follow the instructions below:\n- first\n- second"
    ids = ["i1", "i2"]
    texts = ["first", "second"]

    spans = compute_instruction_spans(tokenizer, full_input, ids, texts)
    assert len(spans) == 2
    assert all(isinstance(s, InstructionSpan) for s in spans)

    bullet_1_start = full_input.find("- first")
    bullet_1_end = bullet_1_start + len("- first")
    assert spans[0].instruction_id == "i1"
    assert spans[0].start_token == bullet_1_start
    assert spans[0].end_token == bullet_1_end


def test_compute_instruction_spans_raises_when_missing_bullet():
    tokenizer = _CharTokenizer()
    with pytest.raises(ValueError, match="Could not find bullet"):
        compute_instruction_spans(
            tokenizer,
            full_input="Q\n\nYour response should follow the instructions below:\n- first",
            instruction_id_list=["i1", "i2"],
            instruction_texts=["first", "second"],
        )
