"""Smoke tests for IFEval runner scripts with mocked heavy dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fake_samples():
    return ["s1", "s2"]


def test_run_ifeval_dynamic_script_smoke(monkeypatch, tmp_path):
    import scripts.run_ifeval_dynamic as mod

    out = tmp_path / "dynamic.json"
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: argparse.Namespace(
            model_name="m",
            selector_model="sel",
            selector_base_url="http://127.0.0.1:11434",
            selector_timeout_s=1.0,
            selector_retries=0,
            split="val",
            seed=42,
            n_val=2,
            n_test=0,
            limit=1,
            ms_jsonl_path=None,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            boost_bias=8.0,
            min_tokens_between_checks=8,
            max_tokens_without_check=32,
            rolling_buffer_chars=32,
            device="cpu",
            dtype="float32",
            trust_remote_code=False,
            output_json=str(out),
            log_file=None,
        ),
    )
    monkeypatch.setattr(mod, "require_ifeval_runtime_dependencies", lambda: None)
    monkeypatch.setattr(mod, "load_ifeval_samples", lambda **_kwargs: (_fake_samples(), []))
    monkeypatch.setattr(mod, "load_transformers_model_and_tokenizer", lambda **_kwargs: ("model", "tok", "cpu"))
    monkeypatch.setattr(mod, "build_selector", lambda **_kwargs: object())
    def _fake_dynamic(**kwargs):
        assert kwargs.get("fallback_selector") is not None
        return (
            [
                {
                    "sample_id": "ifeval_1",
                    "generation": "x",
                    "strict_following": True,
                    "instruction_level_score": 1.0,
                    "per_instruction_eval": [],
                    "method_metadata": {},
                }
            ],
            {
                "prompt_level_strict_acc": 1.0,
                "instruction_level_strict_acc": 1.0,
                "n_samples": 1,
                "n_instructions": 0,
            },
        )

    monkeypatch.setattr(mod, "run_dynamic_benchmark", _fake_dynamic)

    mod.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["method_name"] == "dynamic_instaboost"
    assert out.with_suffix(".log").exists()


def test_run_ifeval_baseline_static_script_smoke(monkeypatch, tmp_path):
    import scripts.run_ifeval_baseline_static_compare as mod

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: argparse.Namespace(
            model_name="m",
            split="val",
            seed=42,
            n_val=2,
            n_test=0,
            limit=1,
            ms_jsonl_path=None,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            boost_bias=8.0,
            device="cpu",
            dtype="float32",
            trust_remote_code=False,
            output_dir=str(out_dir),
            log_file=None,
        ),
    )
    monkeypatch.setattr(mod, "require_ifeval_runtime_dependencies", lambda: None)
    monkeypatch.setattr(mod, "load_ifeval_samples", lambda **_kwargs: (_fake_samples(), []))
    monkeypatch.setattr(mod, "load_transformers_model_and_tokenizer", lambda **_kwargs: ("model", "tok", "cpu"))
    mock_result = (
        [
            {
                "sample_id": "ifeval_1",
                "generation": "x",
                "strict_following": True,
                "instruction_level_score": 1.0,
                "per_instruction_eval": [],
                "method_metadata": {},
            }
        ],
        {
            "prompt_level_strict_acc": 1.0,
            "instruction_level_strict_acc": 1.0,
            "n_samples": 1,
            "n_instructions": 0,
        },
    )
    monkeypatch.setattr(mod, "run_baseline_benchmark", lambda **_kwargs: mock_result)
    monkeypatch.setattr(mod, "run_static_benchmark", lambda **_kwargs: mock_result)

    mod.main()
    assert (out_dir / "baseline.json").exists()
    assert (out_dir / "static_instaboost.json").exists()
    assert (out_dir / "run.log").exists()


def test_run_ifeval_all_methods_script_smoke(monkeypatch, tmp_path):
    import scripts.run_ifeval_all_methods as mod

    out_dir = tmp_path / "all"
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: argparse.Namespace(
            model_name="m",
            selector_model="sel",
            selector_base_url="http://127.0.0.1:11434",
            selector_timeout_s=1.0,
            selector_retries=0,
            split="val",
            seed=42,
            n_val=2,
            n_test=0,
            limit=1,
            ms_jsonl_path=None,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            boost_bias=8.0,
            min_tokens_between_checks=8,
            max_tokens_without_check=32,
            rolling_buffer_chars=32,
            device="cpu",
            dtype="float32",
            trust_remote_code=False,
            output_dir=str(out_dir),
            log_file=None,
        ),
    )
    monkeypatch.setattr(mod, "require_ifeval_runtime_dependencies", lambda: None)
    monkeypatch.setattr(mod, "load_ifeval_samples", lambda **_kwargs: (_fake_samples(), []))
    monkeypatch.setattr(mod, "load_transformers_model_and_tokenizer", lambda **_kwargs: ("model", "tok", "cpu"))
    monkeypatch.setattr(mod, "build_selector", lambda **_kwargs: object())
    base_result = (
        [
            {
                "sample_id": "ifeval_1",
                "generation": "x",
                "strict_following": True,
                "instruction_level_score": 1.0,
                "per_instruction_eval": [],
                "method_metadata": {},
            }
        ],
        {
            "prompt_level_strict_acc": 1.0,
            "instruction_level_strict_acc": 1.0,
            "n_samples": 1,
            "n_instructions": 0,
        },
    )
    dyn_result = (
        [
            {
                "sample_id": "ifeval_1",
                "generation": "x",
                "strict_following": True,
                "instruction_level_score": 1.0,
                "per_instruction_eval": [],
                "method_metadata": {"dynamic_trace": {"selector_calls": 1}},
            }
        ],
        {
            "prompt_level_strict_acc": 1.0,
            "instruction_level_strict_acc": 1.0,
            "n_samples": 1,
            "n_instructions": 0,
        },
    )
    monkeypatch.setattr(mod, "run_baseline_benchmark", lambda **_kwargs: base_result)
    monkeypatch.setattr(mod, "run_static_benchmark", lambda **_kwargs: base_result)
    monkeypatch.setattr(mod, "run_dynamic_benchmark", lambda **_kwargs: dyn_result)

    mod.main()
    assert (out_dir / "baseline.json").exists()
    assert (out_dir / "static_instaboost.json").exists()
    assert (out_dir / "dynamic_instaboost.json").exists()
    assert (out_dir / "comparison.md").exists()
    assert (out_dir / "run.log").exists()
