"""Tests for src.dynamic_boost.boundaries."""

import pytest

from src.dynamic_boost.boundaries import BoundaryChecker
from src.dynamic_boost.types import BoundaryConfig


def test_single_boundary_triggers_once():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=1, max_tokens_without_check=100))

    assert checker.ingest("Hello", total_generated_tokens=5) is None
    event = checker.ingest(".", total_generated_tokens=6)
    assert event is not None
    assert event.reason == "boundary"

    # No new boundary here.
    assert checker.ingest(" world", total_generated_tokens=7) is None


def test_end_dot_newline_newline_triggers_once_with_cooldown():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=8, max_tokens_without_check=100))

    text = "end.\n\n"
    events = []
    for idx, ch in enumerate(text, start=1):
        event = checker.ingest(ch, total_generated_tokens=idx)
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert events[0].reason == "boundary"


def test_forced_fallback_when_no_boundary():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=2, max_tokens_without_check=5))

    events = []
    for i in range(1, 11):
        event = checker.ingest("a", total_generated_tokens=i)
        if event is not None:
            events.append((i, event.reason))

    assert events == [(5, "max_tokens_fallback"), (10, "max_tokens_fallback")]


def test_no_duplicate_checks_inside_min_token_gap():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=4, max_tokens_without_check=100))

    # First boundary fires.
    assert checker.ingest(".", total_generated_tokens=1) is not None
    # Second boundary too soon, suppressed.
    assert checker.ingest("?", total_generated_tokens=2) is None
    # At 4-token gap from last check, boundary can fire again.
    assert checker.ingest("!", total_generated_tokens=5) is not None


def test_multiple_punctuation_only_one_event_inside_cooldown():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=5, max_tokens_without_check=100))

    events = []
    for i, ch in enumerate("!?...", start=1):
        event = checker.ingest(ch, total_generated_tokens=i)
        if event is not None:
            events.append((i, event.reason))

    assert events == [(1, "boundary")]


def test_boundary_marker_split_across_fragments():
    checker = BoundaryChecker(BoundaryConfig(min_tokens_between_checks=1, max_tokens_without_check=100))

    assert checker.ingest("\n", total_generated_tokens=1) is None
    event = checker.ingest("\n", total_generated_tokens=2)
    assert event is not None
    assert event.reason == "boundary"


def test_input_validation_and_monotonicity():
    checker = BoundaryChecker(BoundaryConfig())

    with pytest.raises(TypeError, match="string"):
        checker.ingest(123, total_generated_tokens=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="int"):
        checker.ingest("x", total_generated_tokens=1.2)  # type: ignore[arg-type]

    checker.ingest("x", total_generated_tokens=1)
    with pytest.raises(ValueError, match="monotonic"):
        checker.ingest("y", total_generated_tokens=0)
