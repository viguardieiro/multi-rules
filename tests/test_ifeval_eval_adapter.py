"""Tests for src.ifeval_dynamic.eval_adapter."""

from __future__ import annotations

import pytest

from src.ifeval_dynamic.eval_adapter import (
    compute_ifeval_aggregate_metrics,
    evaluate_ifeval_sample,
)


class _ContainsChecker:
    def __init__(self, inst_id: str):
        self.inst_id = inst_id

    def build_description(self, phrase: str | None = None) -> str:
        self._phrase = phrase or ""
        return f"must include {self._phrase}"

    def check_following(self, generation: str) -> bool:
        return self._phrase in generation


def _instruction_dict():
    return {
        "i1": lambda inst_id: _ContainsChecker(inst_id),
        "i2": lambda inst_id: _ContainsChecker(inst_id),
    }


def test_evaluate_ifeval_sample_mixed_pass_fail_and_unknown():
    res = evaluate_ifeval_sample(
        generation="alpha present only",
        instruction_id_list=["i1", "i2", "unknown"],
        kwargs_list=[{"phrase": "alpha"}, {"phrase": "beta"}, {}],
        instruction_dict=_instruction_dict(),
    )

    assert len(res["per_instruction"]) == 3
    assert res["per_instruction"][0]["passed"] is True
    assert res["per_instruction"][1]["passed"] is False
    assert res["per_instruction"][2]["note"] == "unknown instruction_id"
    assert res["sample_score"] == 0
    assert 0.0 <= res["instruction_level_score"] <= 1.0


def test_evaluate_ifeval_sample_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        evaluate_ifeval_sample(
            generation="x",
            instruction_id_list=["i1"],
            kwargs_list=[],
            instruction_dict=_instruction_dict(),
        )


def test_compute_ifeval_aggregate_metrics_with_ci_fn():
    eval_results = [
        {
            "sample_score": 1,
            "per_instruction": [{"score": 1}, {"score": 0}],
        },
        {
            "sample_score": 0,
            "per_instruction": [{"score": 0}, {"score": 0}],
        },
    ]

    def ci(values):
        return (min(values), max(values))

    out = compute_ifeval_aggregate_metrics(eval_results, ci_fn=ci)
    assert out["prompt_level_strict_acc"] == 0.5
    assert out["instruction_level_strict_acc"] == 0.25
    assert out["prompt_level_ci"] == {"lower": 0.0, "upper": 1.0}
    assert out["n_samples"] == 2
    assert out["n_instructions"] == 4
