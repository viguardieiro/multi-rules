"""Tests for src.dynamic_boost.result_schema."""

import pytest

from src.dynamic_boost.result_schema import (
    SCHEMA_VERSION,
    build_run_result,
    validate_run_result_schema,
)


def _sample_record(sample_id: str = "s1") -> dict:
    return {
        "sample_id": sample_id,
        "generation": "Answer text",
        "strict_following": True,
        "instruction_level_score": 1.0,
        "per_instruction_eval": [{"instruction_id": "i1", "passed": True, "score": 1}],
        "method_metadata": {"notes": "ok"},
    }


def test_build_run_result_produces_valid_schema():
    result = build_run_result(
        method_name="dynamic_instaboost",
        model_name="openai/gpt-oss-20b",
        split_name="val",
        seed=42,
        decode_config={"max_new_tokens": 128, "temperature": 0.0},
        aggregate_metrics={"strict_accuracy": 0.75},
        per_sample_results=[_sample_record()],
        selector_backend="ollama",
        selector_model="gpt-oss:20b",
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["selector"]["backend"] == "ollama"
    validate_run_result_schema(result)


def test_validate_schema_rejects_missing_top_level_key():
    result = build_run_result(
        method_name="baseline",
        model_name="google/gemma-2-9b",
        split_name="test",
        seed=1,
        decode_config={"max_new_tokens": 128},
        aggregate_metrics={"strict_accuracy": 0.5},
        per_sample_results=[_sample_record()],
    )
    del result["decode_config"]

    with pytest.raises(ValueError, match="Missing top-level schema keys"):
        validate_run_result_schema(result)


def test_validate_schema_rejects_bad_sample_score_range():
    result = build_run_result(
        method_name="static_instaboost",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        split_name="val",
        seed=7,
        decode_config={"max_new_tokens": 64},
        aggregate_metrics={"strict_accuracy": 0.4},
        per_sample_results=[_sample_record()],
    )
    result["per_sample_results"][0]["instruction_level_score"] = 1.2

    with pytest.raises(ValueError, match="instruction_level_score"):
        validate_run_result_schema(result)


def test_validate_schema_rejects_missing_per_sample_field():
    result = build_run_result(
        method_name="baseline",
        model_name="google/gemma-2-9b",
        split_name="val",
        seed=0,
        decode_config={"max_new_tokens": 16},
        aggregate_metrics={"strict_accuracy": 0.0},
        per_sample_results=[_sample_record()],
    )
    del result["per_sample_results"][0]["generation"]

    with pytest.raises(ValueError, match="missing 'generation'"):
        validate_run_result_schema(result)


def test_validate_schema_rejects_bool_seed():
    result = build_run_result(
        method_name="baseline",
        model_name="google/gemma-2-9b",
        split_name="val",
        seed=1,
        decode_config={"max_new_tokens": 16},
        aggregate_metrics={"strict_accuracy": 0.0},
        per_sample_results=[_sample_record()],
    )
    result["seed"] = True

    with pytest.raises(TypeError, match="seed must be int"):
        validate_run_result_schema(result)


def test_validate_schema_rejects_bool_instruction_level_score():
    result = build_run_result(
        method_name="baseline",
        model_name="google/gemma-2-9b",
        split_name="val",
        seed=1,
        decode_config={"max_new_tokens": 16},
        aggregate_metrics={"strict_accuracy": 0.0},
        per_sample_results=[_sample_record()],
    )
    result["per_sample_results"][0]["instruction_level_score"] = True

    with pytest.raises(TypeError, match="instruction_level_score"):
        validate_run_result_schema(result)
