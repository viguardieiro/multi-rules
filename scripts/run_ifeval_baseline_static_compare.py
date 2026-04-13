"""Run baseline and static InstABoost on IFEval with shared settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dynamic_boost.result_schema import build_run_result
from src.ifeval_dynamic.data_adapter import load_ifeval_samples
from src.ifeval_dynamic.logging_utils import ExperimentLogger
from src.ifeval_dynamic.runtime_checks import require_ifeval_runtime_dependencies
from src.ifeval_dynamic.runner import (
    GenerationConfig,
    load_transformers_model_and_tokenizer,
    run_baseline_benchmark,
    run_static_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IFEval baseline and static InstABoost")
    parser.add_argument("--model-name", type=str, required=True, help="HF model id/path used for generation")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ms-jsonl-path", type=str, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--boost-bias", type=float, default=8.0)

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to persistent run log")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    default_log = str(Path(args.output_dir) / "run.log")
    logger = ExperimentLogger(log_file=args.log_file, default_log_file=default_log)
    require_ifeval_runtime_dependencies()

    val_samples, test_samples = load_ifeval_samples(
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        ms_jsonl_path=args.ms_jsonl_path,
    )
    samples = val_samples if args.split == "val" else test_samples
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"No samples found for split='{args.split}' with current filters")

    model, tokenizer, device = load_transformers_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    logger.info(
        "Running baseline/static:",
        f"split={args.split}",
        f"samples={len(samples)}",
        f"model={args.model_name}",
    )

    baseline_samples, baseline_agg = run_baseline_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_config,
    )
    static_samples, static_agg = run_static_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_config,
        boost_bias=args.boost_bias,
    )

    baseline_run = build_run_result(
        method_name="baseline",
        model_name=args.model_name,
        split_name=args.split,
        seed=args.seed,
        decode_config=generation_config.to_dict(),
        aggregate_metrics=baseline_agg,
        per_sample_results=baseline_samples,
        selector_backend=None,
        selector_model=None,
        extra_metadata={
            "n_samples": len(samples),
        },
    )
    static_run = build_run_result(
        method_name="static_instaboost",
        model_name=args.model_name,
        split_name=args.split,
        seed=args.seed,
        decode_config=generation_config.to_dict(),
        aggregate_metrics=static_agg,
        per_sample_results=static_samples,
        selector_backend=None,
        selector_model=None,
        extra_metadata={
            "n_samples": len(samples),
            "boost_bias": args.boost_bias,
        },
    )

    out_dir = Path(args.output_dir)
    baseline_path = out_dir / "baseline.json"
    static_path = out_dir / "static_instaboost.json"
    _write_json(baseline_path, baseline_run)
    _write_json(static_path, static_run)

    logger.info(f"Wrote baseline to {baseline_path}")
    logger.info(f"Wrote static to {static_path}")


if __name__ == "__main__":
    main()
