"""Run baseline, static, and dynamic IFEval in one command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dynamic_boost import BoundaryConfig, DeterministicFallbackSelector
from src.dynamic_boost.result_schema import build_run_result
from src.ifeval_dynamic.data_adapter import load_ifeval_samples
from src.ifeval_dynamic.logging_utils import ExperimentLogger
from src.ifeval_dynamic.runtime_checks import require_ifeval_runtime_dependencies
from src.ifeval_dynamic.runner import (
    GenerationConfig,
    build_selector,
    load_transformers_model_and_tokenizer,
    run_baseline_benchmark,
    run_dynamic_benchmark,
    run_static_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all IFEval methods in one command")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--selector-model", type=str, required=True)
    p.add_argument("--selector-base-url", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--selector-timeout-s", type=float, default=30.0)
    p.add_argument("--selector-retries", type=int, default=1)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ms-jsonl-path", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--boost-bias", type=float, default=8.0)
    p.add_argument("--min-tokens-between-checks", type=int, default=8)
    p.add_argument("--max-tokens-without-check", type=int, default=32)
    p.add_argument("--rolling-buffer-chars", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--log-file", type=str, default=None, help="Optional path to persistent run log")
    return p.parse_args()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _table(baseline: dict, static: dict, dynamic: dict) -> str:
    rows = [
        ("baseline", baseline["aggregate_metrics"]),
        ("static_instaboost", static["aggregate_metrics"]),
        ("dynamic_instaboost", dynamic["aggregate_metrics"]),
    ]
    lines = [
        "| method | prompt_strict_acc | instruction_strict_acc | n_samples | n_instructions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, agg in rows:
        lines.append(
            f"| {name} | {float(agg['prompt_level_strict_acc']):.4f} | "
            f"{float(agg['instruction_level_strict_acc']):.4f} | "
            f"{int(agg['n_samples'])} | {int(agg['n_instructions'])} |"
        )
    return "\n".join(lines) + "\n"


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
        raise ValueError("No samples found with current split/limit settings")

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
    boundary_config = BoundaryConfig(
        min_tokens_between_checks=args.min_tokens_between_checks,
        max_tokens_without_check=args.max_tokens_without_check,
        rolling_buffer_chars=args.rolling_buffer_chars,
    )

    logger.info("Running baseline...")
    baseline_samples, baseline_agg = run_baseline_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_config,
    )
    logger.info("Running static instaboost...")
    static_samples, static_agg = run_static_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_config,
        boost_bias=args.boost_bias,
    )
    logger.info("Running dynamic instaboost...")
    selector = build_selector(
        selector_backend="ollama",
        selector_model=args.selector_model,
        selector_base_url=args.selector_base_url,
        selector_timeout_s=args.selector_timeout_s,
        selector_retries=args.selector_retries,
        logger=lambda m: logger.info("[selector]", m),
    )
    fallback = DeterministicFallbackSelector(keep_current_if_possible=True)
    dynamic_samples, dynamic_agg = run_dynamic_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        selector=selector,
        selector_backend_name=f"ollama:{args.selector_model}",
        generation_config=generation_config,
        boundary_config=boundary_config,
        boost_bias=args.boost_bias,
        fallback_selector=fallback,
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
        extra_metadata={"n_samples": len(samples)},
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
        extra_metadata={"n_samples": len(samples), "boost_bias": args.boost_bias},
    )
    dynamic_run = build_run_result(
        method_name="dynamic_instaboost",
        model_name=args.model_name,
        split_name=args.split,
        seed=args.seed,
        decode_config=generation_config.to_dict(),
        aggregate_metrics=dynamic_agg,
        per_sample_results=dynamic_samples,
        selector_backend="ollama",
        selector_model=args.selector_model,
        extra_metadata={
            "n_samples": len(samples),
            "boost_bias": args.boost_bias,
            "boundary_config": {
                "min_tokens_between_checks": args.min_tokens_between_checks,
                "max_tokens_without_check": args.max_tokens_without_check,
                "rolling_buffer_chars": args.rolling_buffer_chars,
            },
        },
    )

    out_dir = Path(args.output_dir)
    _write(out_dir / "baseline.json", baseline_run)
    _write(out_dir / "static_instaboost.json", static_run)
    _write(out_dir / "dynamic_instaboost.json", dynamic_run)
    table = _table(baseline_run, static_run, dynamic_run)
    (out_dir / "comparison.md").write_text(table, encoding="utf-8")
    logger.info(table.rstrip("\n"))
    logger.info(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
