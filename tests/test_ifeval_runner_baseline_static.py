"""Tests for baseline/static paths in src.ifeval_dynamic.runner."""

from __future__ import annotations

from dataclasses import dataclass

from src.boost_config import BoostConfig, TokenSubset
from src.ifeval_dynamic.data_adapter import IFEvalSample
from src.ifeval_dynamic.instruction_spans import InstructionSpan
from src.ifeval_dynamic.runner import GenerationConfig, run_baseline_sample, run_static_sample


def _sample() -> IFEvalSample:
    return IFEvalSample(
        key=1,
        sample_id="ifeval_1",
        original_prompt="orig",
        base_question="q",
        instruction_id_list=["i1", "i2"],
        kwargs_list=[{}, {}],
        instruction_texts=["first", "second"],
        instruction_block="Your response should follow the instructions below:\n- first\n- second",
        full_input="q\n\nYour response should follow the instructions below:\n- first\n- second",
    )


class _FakeGenerator:
    def __init__(self, **_kwargs):
        self.i = 0
        self.prompt_length = 10

    def step(self, _active_ids, _step_index):
        from src.dynamic_boost import TokenStepOutput

        self.i += 1
        if self.i == 1:
            return TokenStepOutput(text="A")
        return TokenStepOutput(text="B", is_eos=True)


def test_run_baseline_sample(monkeypatch):
    monkeypatch.setattr("src.ifeval_dynamic.runner._IncrementalGenerator", _FakeGenerator)
    monkeypatch.setattr(
        "src.ifeval_dynamic.runner.evaluate_ifeval_sample",
        lambda *_args, **_kwargs: {
            "sample_score": 1,
            "instruction_level_score": 0.5,
            "per_instruction": [{"instruction_id": "i1", "score": 1, "passed": True}],
        },
    )

    out = run_baseline_sample(
        sample=_sample(),
        model=object(),
        tokenizer=object(),
        device=object(),
        generation_config=GenerationConfig(max_new_tokens=5),
    )

    assert out["generation"] == "AB"
    assert out["strict_following"] is True
    assert out["method_metadata"]["mode"] == "baseline"
    assert out["method_metadata"]["generated_tokens"] == 2


@dataclass
class _FakeHandle:
    config: object | None = None


def test_run_static_sample(monkeypatch):
    monkeypatch.setattr("src.ifeval_dynamic.runner._IncrementalGenerator", _FakeGenerator)
    monkeypatch.setattr(
        "src.ifeval_dynamic.runner.evaluate_ifeval_sample",
        lambda *_args, **_kwargs: {
            "sample_score": 0,
            "instruction_level_score": 0.25,
            "per_instruction": [{"instruction_id": "i1", "score": 0, "passed": False}],
        },
    )
    monkeypatch.setattr(
        "src.ifeval_dynamic.runner.compute_instruction_spans",
        lambda *_args, **_kwargs: [
            InstructionSpan(instruction_id="i1", start_token=2, end_token=4),
            InstructionSpan(instruction_id="i2", start_token=5, end_token=7),
        ],
    )
    monkeypatch.setattr(
        "src.ifeval_dynamic.runner.build_active_boost_config",
        lambda instruction_token_indices, active_instruction_ids, boost_bias: BoostConfig(
            subsets=[
                TokenSubset(name=inst_id, indices=instruction_token_indices[inst_id], bias=boost_bias)
                for inst_id in active_instruction_ids
            ]
        ),
    )
    monkeypatch.setattr("src.ifeval_dynamic.runner.register_boost_hooks", lambda *_args, **_kwargs: _FakeHandle())
    monkeypatch.setattr("src.ifeval_dynamic.runner.update_bias_mask", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ifeval_dynamic.runner.unregister_boost_hooks", lambda *_args, **_kwargs: None)

    out = run_static_sample(
        sample=_sample(),
        model=object(),
        tokenizer=object(),
        device=object(),
        generation_config=GenerationConfig(max_new_tokens=5),
        boost_bias=8.0,
    )

    assert out["generation"] == "AB"
    assert out["strict_following"] is False
    assert out["method_metadata"]["mode"] == "static_instaboost"
    assert out["method_metadata"]["active_instruction_ids_fixed"] == ["i1", "i2"]
