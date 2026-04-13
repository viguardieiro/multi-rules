"""Transformers integration tests for boundary checker behavior.

These tests validate boundary detection against real streamed text pieces produced
by a Hugging Face tokenizer, which is the path used in token-by-token generation.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from src.dynamic_boost.boundaries import BoundaryChecker
from src.dynamic_boost.types import BoundaryConfig


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Local GPT-2 tokenizer cache unavailable for offline integration test: {exc}")


def _stream_pieces(tokenizer, text: str) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = [
        tokenizer.decode(
            [tok_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for tok_id in ids
    ]
    return pieces


def test_transformers_stream_end_dot_newline_newline_triggers_once(gpt2_tokenizer):
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=8, max_tokens_without_check=100))

    # The stream may represent "\n\n" as one token or multiple tokens.
    text = "end.\n\nNext line"
    pieces = _stream_pieces(gpt2_tokenizer, text)

    events = []
    for i, piece in enumerate(pieces, start=1):
        event = checker.ingest(piece, total_generated_tokens=i)
        if event is not None:
            events.append(event)

    # We should trigger on boundary exactly once inside cooldown.
    assert len(events) == 1
    assert events[0].reason == "boundary"


def test_transformers_stream_fallback_without_punctuation(gpt2_tokenizer):
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=2, max_tokens_without_check=3))

    text = "abcdefghij"  # intentionally no punctuation/newlines
    pieces = _stream_pieces(gpt2_tokenizer, text)

    events = []
    for i, piece in enumerate(pieces, start=1):
        event = checker.ingest(piece, total_generated_tokens=i)
        if event is not None:
            events.append((i, event.reason))

    # Fallback should fire every 3 streamed token steps.
    expected = []
    for idx in range(3, len(pieces) + 1, 3):
        expected.append((idx, "max_tokens_fallback"))

    assert events == expected
