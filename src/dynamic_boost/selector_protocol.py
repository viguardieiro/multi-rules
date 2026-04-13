"""Model-agnostic selector protocol and normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .types import SelectorDecision, SelectorRequest


@runtime_checkable
class InstructionSelector(Protocol):
    """Protocol for external or heuristic rule/instruction selectors."""

    def select(self, request: SelectorRequest) -> SelectorDecision:
        """Return the next selector decision for the current generation state."""


def ensure_selector(selector: Any) -> InstructionSelector:
    """Validate that an object implements the selector protocol."""

    if not isinstance(selector, InstructionSelector) or not callable(getattr(selector, "select", None)):
        raise TypeError("selector must implement InstructionSelector protocol")
    return selector


def decision_from_dict(raw: Mapping[str, Any]) -> SelectorDecision:
    """Build a typed SelectorDecision from a raw JSON-like dictionary."""

    if not isinstance(raw, Mapping):
        raise TypeError("raw selector output must be a mapping")

    try:
        decision = raw["decision"]
        active = raw["active_instruction_ids"]
        confidence = raw["confidence"]
    except KeyError as exc:
        raise ValueError(f"Missing selector decision field: {exc.args[0]}") from exc

    reason = raw.get("reason", "")
    metadata = raw.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict when provided")

    return SelectorDecision(
        decision=decision,
        active_instruction_ids=active,
        confidence=confidence,
        reason=reason,
        metadata=metadata,
    )


def normalize_selector_output(raw: SelectorDecision | Mapping[str, Any]) -> SelectorDecision:
    """Normalize selector output to SelectorDecision."""

    if isinstance(raw, SelectorDecision):
        return raw
    if isinstance(raw, Mapping):
        return decision_from_dict(raw)
    raise TypeError("Selector output must be SelectorDecision or dict")
