"""IFEval evaluation adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .data_adapter import clean_kwargs


def _resolve_instruction_dict(instruction_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    if instruction_dict is not None:
        return instruction_dict

    import sys

    from .data_adapter import _datasets_import_paths

    for datasets_path in _datasets_import_paths():
        if datasets_path not in sys.path:
            sys.path.insert(0, datasets_path)

    from ifeval_scripts import instructions_registry  # type: ignore

    return instructions_registry.INSTRUCTION_DICT


def evaluate_ifeval_sample(
    generation: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    *,
    instruction_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one generated response against IFEval instruction checkers."""

    if len(instruction_id_list) != len(kwargs_list):
        raise ValueError("instruction_id_list and kwargs_list must have the same length")

    resolved = _resolve_instruction_dict(instruction_dict)

    per_instruction = []
    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        if inst_id not in resolved:
            per_instruction.append(
                {
                    "instruction_id": inst_id,
                    "passed": False,
                    "score": 0,
                    "note": "unknown instruction_id",
                }
            )
            continue

        checker = resolved[inst_id](inst_id)
        checker.build_description(**clean_kwargs(kwargs))
        passed = bool(checker.check_following(generation))
        per_instruction.append(
            {
                "instruction_id": inst_id,
                "passed": passed,
                "score": 1 if passed else 0,
            }
        )

    all_passed = all(item["passed"] for item in per_instruction) if per_instruction else False
    instruction_level_score = (
        float(sum(item["score"] for item in per_instruction) / len(per_instruction))
        if per_instruction
        else 0.0
    )

    return {
        "per_instruction": per_instruction,
        "sample_score": 1 if all_passed else 0,
        "instruction_level_score": instruction_level_score,
    }


def compute_ifeval_aggregate_metrics(
    eval_results: list[dict[str, Any]],
    *,
    ci_fn: Callable[[list[float]], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compute prompt/instruction strict accuracy summary from sample evaluations."""

    sample_scores = [float(item["sample_score"]) for item in eval_results]
    instruction_scores = [
        float(inst["score"])
        for item in eval_results
        for inst in item["per_instruction"]
    ]

    prompt_acc = float(np.mean(sample_scores)) if sample_scores else 0.0
    inst_acc = float(np.mean(instruction_scores)) if instruction_scores else 0.0

    if ci_fn is not None and len(sample_scores) >= 2:
        prompt_ci = ci_fn(sample_scores)
    else:
        prompt_ci = (0.0, 0.0)

    if ci_fn is not None and len(instruction_scores) >= 2:
        inst_ci = ci_fn(instruction_scores)
    else:
        inst_ci = (0.0, 0.0)

    return {
        "prompt_level_strict_acc": prompt_acc,
        "prompt_level_ci": {"lower": float(prompt_ci[0]), "upper": float(prompt_ci[1])},
        "instruction_level_strict_acc": inst_acc,
        "instruction_level_ci": {"lower": float(inst_ci[0]), "upper": float(inst_ci[1])},
        "n_samples": len(sample_scores),
        "n_instructions": len(instruction_scores),
    }
