"""Canonical result schema helpers for benchmark method comparability."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

SCHEMA_VERSION = "dynamic_boost_results_v1"


def build_run_result(
    *,
    method_name: str,
    model_name: str,
    split_name: str,
    seed: int,
    decode_config: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    per_sample_results: list[dict[str, Any]],
    selector_backend: str | None = None,
    selector_model: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a schema-compliant run result dictionary."""

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_name": method_name,
        "model_name": model_name,
        "split_name": split_name,
        "seed": seed,
        "decode_config": decode_config,
        "selector": {
            "backend": selector_backend,
            "model": selector_model,
        },
        "aggregate_metrics": aggregate_metrics,
        "per_sample_results": per_sample_results,
        "extra_metadata": extra_metadata or {},
    }
    validate_run_result_schema(result)
    return result


def validate_run_result_schema(result: Mapping[str, Any]) -> None:
    """Validate required keys and core typing constraints."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required = {
        "schema_version",
        "created_at_utc",
        "method_name",
        "model_name",
        "split_name",
        "seed",
        "decode_config",
        "selector",
        "aggregate_metrics",
        "per_sample_results",
        "extra_metadata",
    }
    missing = sorted(required - set(result.keys()))
    if missing:
        raise ValueError(f"Missing top-level schema keys: {missing}")

    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported schema_version")

    if not isinstance(result["created_at_utc"], str) or not result["created_at_utc"]:
        raise ValueError("created_at_utc must be a non-empty string")
    # Accepts timezone-aware ISO strings like "2026-04-12T17:00:00+00:00".
    datetime.fromisoformat(result["created_at_utc"])

    if not isinstance(result["method_name"], str) or not result["method_name"]:
        raise ValueError("method_name cannot be empty")
    if not isinstance(result["model_name"], str) or not result["model_name"]:
        raise ValueError("model_name cannot be empty")
    if not isinstance(result["split_name"], str) or not result["split_name"]:
        raise ValueError("split_name cannot be empty")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        raise TypeError("seed must be int")

    if not isinstance(result["decode_config"], Mapping):
        raise TypeError("decode_config must be a mapping")
    if not isinstance(result["selector"], Mapping):
        raise TypeError("selector must be a mapping")
    if "backend" not in result["selector"] or "model" not in result["selector"]:
        raise ValueError("selector must include 'backend' and 'model' keys")
    backend = result["selector"]["backend"]
    model = result["selector"]["model"]
    if backend is not None and (not isinstance(backend, str) or not backend):
        raise ValueError("selector.backend must be None or non-empty string")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("selector.model must be None or non-empty string")
    if not isinstance(result["aggregate_metrics"], Mapping):
        raise TypeError("aggregate_metrics must be a mapping")
    if not isinstance(result["per_sample_results"], list):
        raise TypeError("per_sample_results must be a list")
    if not isinstance(result["extra_metadata"], Mapping):
        raise TypeError("extra_metadata must be a mapping")

    for idx, sample in enumerate(result["per_sample_results"]):
        if not isinstance(sample, Mapping):
            raise TypeError(f"per_sample_results[{idx}] must be a mapping")
        for key in [
            "sample_id",
            "generation",
            "strict_following",
            "instruction_level_score",
            "per_instruction_eval",
            "method_metadata",
        ]:
            if key not in sample:
                raise ValueError(f"per_sample_results[{idx}] missing '{key}'")

        if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
            raise ValueError(f"per_sample_results[{idx}].sample_id must be non-empty string")
        if not isinstance(sample["generation"], str):
            raise TypeError(f"per_sample_results[{idx}].generation must be string")
        if not isinstance(sample["strict_following"], bool):
            raise TypeError(f"per_sample_results[{idx}].strict_following must be bool")

        score = sample["instruction_level_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"per_sample_results[{idx}].instruction_level_score must be numeric")
        if not (0.0 <= float(score) <= 1.0):
            raise ValueError(f"per_sample_results[{idx}].instruction_level_score must be in [0,1]")

        if not isinstance(sample["per_instruction_eval"], list):
            raise TypeError(f"per_sample_results[{idx}].per_instruction_eval must be list")
        if not isinstance(sample["method_metadata"], Mapping):
            raise TypeError(f"per_sample_results[{idx}].method_metadata must be a mapping")
