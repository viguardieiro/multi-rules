"""Boundary-trigger checker with debounce and max-token fallback.

This module is model-agnostic: it only consumes generated text fragments and
monotonic token counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import BoundaryConfig, BoundaryEvent


@dataclass
class BoundaryCheckerState:
    """Internal mutable state for boundary checks."""

    tail: str = ""
    last_observed_token_count: int = 0
    last_check_token_index: int | None = None


class BoundaryChecker:
    """Decides when to run selector updates during generation."""

    def __init__(self, config: BoundaryConfig):
        if not isinstance(config, BoundaryConfig):
            raise TypeError("config must be a BoundaryConfig")
        self.config = config
        self.state = BoundaryCheckerState()

    def reset(self) -> None:
        """Reset checker state for a new sample."""

        self.state = BoundaryCheckerState()

    def ingest(self, text_fragment: str, total_generated_tokens: int) -> BoundaryEvent | None:
        """Ingest new generated text and decide whether to trigger an update.

        Args:
            text_fragment: Newly generated text fragment since the previous call.
            total_generated_tokens: Monotonic total number of generated tokens.

        Returns:
            A BoundaryEvent when a selector update should be triggered; otherwise None.
        """

        if not isinstance(text_fragment, str):
            raise TypeError("text_fragment must be a string")
        if isinstance(total_generated_tokens, bool) or not isinstance(total_generated_tokens, int):
            raise TypeError("total_generated_tokens must be an int")
        if total_generated_tokens < 0:
            raise ValueError("total_generated_tokens must be >= 0")
        if total_generated_tokens < self.state.last_observed_token_count:
            raise ValueError("total_generated_tokens must be monotonic")

        previous_tail = self.state.tail
        combined = (previous_tail + text_fragment)[-self.config.rolling_buffer_chars :]

        boundary_detected = self._has_new_boundary(previous_tail, text_fragment)
        boundary_due = boundary_detected and self._cooldown_satisfied(total_generated_tokens)
        fallback_due = self._fallback_due(total_generated_tokens)

        event = None
        if boundary_due or fallback_due:
            reason = "boundary" if boundary_due else "max_tokens_fallback"
            event = BoundaryEvent(
                token_index=total_generated_tokens,
                reason=reason,
                text_suffix=combined,
            )
            self.state.last_check_token_index = total_generated_tokens

        self.state.tail = combined
        self.state.last_observed_token_count = total_generated_tokens
        return event

    def _cooldown_satisfied(self, total_generated_tokens: int) -> bool:
        if self.state.last_check_token_index is None:
            return True
        return (total_generated_tokens - self.state.last_check_token_index) >= self.config.min_tokens_between_checks

    def _fallback_due(self, total_generated_tokens: int) -> bool:
        if total_generated_tokens == 0:
            return False
        base = self.state.last_check_token_index if self.state.last_check_token_index is not None else 0
        return (total_generated_tokens - base) >= self.config.max_tokens_without_check

    def _has_new_boundary(self, previous_tail: str, text_fragment: str) -> bool:
        if not text_fragment:
            return False

        composite = previous_tail + text_fragment
        prev_len = len(previous_tail)

        for marker in self.config.boundary_markers:
            start = 0
            while True:
                marker_pos = composite.find(marker, start)
                if marker_pos == -1:
                    break

                end_pos = marker_pos + len(marker)
                # Trigger only for boundaries completed in the newly appended region.
                if end_pos > prev_len:
                    return True

                start = marker_pos + 1

        return False
