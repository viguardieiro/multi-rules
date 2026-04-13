"""Parity-style checks for IFEval evaluation semantics."""

from __future__ import annotations

from src.ifeval_dynamic.eval_adapter import evaluate_ifeval_sample


class _Checker:
    def __init__(self, inst_id: str):
        self.inst_id = inst_id

    def build_description(self, keyword: str | None = None) -> str:
        self.keyword = keyword or ""
        return f"Include {self.keyword}"

    def check_following(self, generation: str) -> bool:
        return self.keyword in generation


def _instruction_dict():
    return {
        "i1": lambda inst_id: _Checker(inst_id),
        "i2": lambda inst_id: _Checker(inst_id),
    }


def _reference_evaluate(generation, instruction_id_list, kwargs_list, instruction_dict):
    # Mirrors run_ifeval_benchmark_instr_last.evaluate_sample semantics.
    per_instruction = []
    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        if inst_id not in instruction_dict:
            per_instruction.append({"instruction_id": inst_id, "passed": False, "score": 0, "note": "unknown instruction_id"})
            continue
        checker = instruction_dict[inst_id](inst_id)
        checker.build_description(**{k: v for k, v in kwargs.items() if v is not None})
        passed = bool(checker.check_following(generation))
        per_instruction.append({"instruction_id": inst_id, "passed": passed, "score": 1 if passed else 0})

    all_passed = all(item["passed"] for item in per_instruction) if per_instruction else False
    inst_level = (sum(item["score"] for item in per_instruction) / len(per_instruction)) if per_instruction else 0.0
    return {
        "per_instruction": per_instruction,
        "sample_score": 1 if all_passed else 0,
        "instruction_level_score": inst_level,
    }


def test_eval_adapter_matches_reference_semantics():
    generation = "alpha and beta"
    ids = ["i1", "i2", "unknown"]
    kwargs = [{"keyword": "alpha"}, {"keyword": "beta"}, {}]

    ref = _reference_evaluate(generation, ids, kwargs, _instruction_dict())
    out = evaluate_ifeval_sample(
        generation,
        instruction_id_list=ids,
        kwargs_list=kwargs,
        instruction_dict=_instruction_dict(),
    )

    assert out == ref
