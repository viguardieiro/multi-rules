"""Tests for selector_llm parser and retry logic."""

from __future__ import annotations

import pytest

from src.dynamic_boost.selector_llm import (
    LLMInstructionSelector,
    SelectorParseError,
    build_selector_payload,
    parse_selector_response,
    sanitize_raw_output,
)
from src.dynamic_boost.types import SelectorRequest


class FlakyBackend:
    name = "flaky"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, system_prompt, user_payload, timeout_s):
        out = self.outputs[self.calls]
        self.calls += 1
        return out


def _request() -> SelectorRequest:
    return SelectorRequest(
        sample_id="s1",
        base_prompt="Q",
        candidate_instruction_ids=["i1", "i2"],
        instruction_text_by_id={"i1": "first", "i2": "second"},
        currently_active_instruction_ids=["i1"],
        current_generation="Some generated text. " * 20,
        generation_token_count=42,
        step_index=3,
    )


def test_parse_selector_response_direct_json():
    raw = '{"decision":"switch","active_instruction_ids":["i2"],"confidence":0.9,"reason":"go"}'
    decision = parse_selector_response(raw)
    assert decision.decision == "switch"
    assert decision.active_instruction_ids == ["i2"]


def test_parse_selector_response_with_wrappers_and_fence():
    raw = (
        "Result:\n```json\n"
        '{"decision":"add","active_instruction_ids":["i1","i2"],"confidence":0.74,"reason":"need both"}'
        "\n```"
    )
    decision = parse_selector_response(raw)
    assert decision.decision == "add"
    assert decision.active_instruction_ids == ["i1", "i2"]


def test_parse_selector_response_raises_when_no_json():
    with pytest.raises(SelectorParseError, match="valid JSON selector object"):
        parse_selector_response("not-json output")


def test_build_selector_payload_truncates_generation_tail():
    req = _request()
    payload = build_selector_payload(req, max_generation_chars=25)
    assert "sample_id" not in payload
    assert payload["model_input"] == req.base_prompt
    assert payload["current_generation"] == req.current_generation[-25:]
    assert payload["instruction_texts"][0]["id"] == "i1"


def test_sanitize_raw_output_compacts_and_truncates():
    raw = "A\n\nB\tC      D"
    compact = sanitize_raw_output(raw, max_chars=4)
    assert compact.endswith("...")
    assert "  " not in compact


def test_llm_selector_retries_and_succeeds_on_second_attempt():
    backend = FlakyBackend(
        outputs=[
            "bad output",
            '{"decision":"switch","active_instruction_ids":["i2"],"confidence":0.8,"reason":"ok"}',
        ]
    )

    logs = []
    selector = LLMInstructionSelector(
        backend=backend,
        max_retries=1,
        retry_backoff_s=0.0,
        logger=logs.append,
    )

    decision = selector.select(_request())
    assert decision.decision == "switch"
    assert backend.calls == 2
    assert len(logs) == 1


def test_llm_selector_raises_after_retry_exhaustion():
    backend = FlakyBackend(outputs=["bad output", "still bad"])
    selector = LLMInstructionSelector(backend=backend, max_retries=1, retry_backoff_s=0.0)

    with pytest.raises(SelectorParseError, match="failed after"):
        selector.select(_request())
    assert backend.calls == 2
