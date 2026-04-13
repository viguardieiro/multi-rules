"""Tests for src.ifeval_dynamic.runner utility functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.dynamic_boost.types import BoundaryEvent, DynamicRunTrace, SelectorDecision
from src.ifeval_dynamic.instruction_spans import InstructionSpan
from src.ifeval_dynamic.runner import (
    GenerationConfig,
    _IncrementalGenerator,
    build_active_boost_config,
    instruction_spans_to_index_map,
    selector_request_from_context,
    trace_to_dict,
)


def test_instruction_spans_to_index_map():
    spans = [
        InstructionSpan(instruction_id="a", start_token=2, end_token=5),
        InstructionSpan(instruction_id="b", start_token=8, end_token=10),
    ]
    got = instruction_spans_to_index_map(spans)
    assert got == {
        "a": [2, 3, 4],
        "b": [8, 9],
    }


def test_instruction_spans_to_index_map_rejects_invalid_span():
    with pytest.raises(ValueError, match="Invalid span"):
        instruction_spans_to_index_map(
            [InstructionSpan(instruction_id="a", start_token=5, end_token=5)]
        )


def test_build_active_boost_config():
    index_map = {"i1": [1, 2], "i2": [5]}
    cfg = build_active_boost_config(index_map, ["i2"], boost_bias=7.5)
    assert len(cfg.subsets) == 1
    assert cfg.subsets[0].name == "i2"
    assert cfg.subsets[0].indices == [5]
    assert cfg.subsets[0].bias == 7.5


def test_selector_request_from_context():
    ctx = {
        "sample_id": "ifeval_1",
        "base_prompt": "q",
        "candidate_instruction_ids": ["i1", "i2"],
        "instruction_text_by_id": {"i1": "A", "i2": "B"},
        "currently_active_instruction_ids": ["i1"],
        "current_generation": "abc",
        "generation_token_count": 3,
        "step_index": 3,
        "metadata": {"k": 1},
    }
    req = selector_request_from_context(ctx)
    assert req.sample_id == "ifeval_1"
    assert req.currently_active_instruction_ids == ["i1"]
    assert req.instruction_text_by_id["i2"] == "B"


def test_trace_to_dict():
    trace = DynamicRunTrace(
        sample_id="ifeval_2",
        model_name="m",
        selector_backend="b",
        decode_config={"max_new_tokens": 4},
    )
    trace.add_boundary_event(BoundaryEvent(token_index=2, reason="boundary", text_suffix="end."))
    trace.add_selector_decision(
        SelectorDecision(
            decision="stay",
            active_instruction_ids=["i1"],
            confidence=0.4,
            reason="x",
        ),
        used_fallback=True,
    )
    payload = trace_to_dict(trace)
    assert payload["sample_id"] == "ifeval_2"
    assert payload["fallback_count"] == 1
    assert payload["selector_calls"] == 1
    assert payload["boundary_events"][0]["reason"] == "boundary"


class _DummyTokenizer:
    eos_token_id = None

    def __call__(self, _text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, _ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "x"


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        past_key_values=None,
        position_ids=None,
        cache_position=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
                "use_cache": use_cache,
                "past_key_values": past_key_values,
                "position_ids": None if position_ids is None else position_ids.clone(),
                "cache_position": None if cache_position is None else cache_position.clone(),
            }
        )

        vocab = 8
        logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)
        logits[:, -1, 4] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=("pkv",))


def test_incremental_generator_passes_step_attention_and_positions():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    gen = _IncrementalGenerator(
        model=model,
        tokenizer=tokenizer,
        full_input="q",
        generation_config=GenerationConfig(max_new_tokens=4, do_sample=False),
        device=torch.device("cpu"),
    )

    gen.step([], 1)
    gen.step([], 2)

    assert len(model.calls) == 2
    second = model.calls[1]
    assert second["attention_mask"] is not None
    assert tuple(second["attention_mask"].shape) == (1, 4)
    assert second["position_ids"] is not None
    assert tuple(second["position_ids"].shape) == (1, 1)
    assert int(second["position_ids"][0, 0]) == 3
    assert second["cache_position"] is not None
    assert tuple(second["cache_position"].shape) == (1,)
    assert int(second["cache_position"][0]) == 3
