"""Run dynamic InstABoost on IFEval with boundary-triggered selector updates."""

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
    run_dynamic_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic InstABoost on IFEval")
    parser.add_argument("--model-name", type=str, required=True, help="HF model id/path used for generation")
    parser.add_argument("--selector-model", type=str, required=True, help="Selector model for Ollama backend")
    parser.add_argument("--selector-base-url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--selector-timeout-s", type=float, default=30.0)
    parser.add_argument("--selector-retries", type=int, default=1)

    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on selected split size")
    parser.add_argument("--ms-jsonl-path", type=str, default=None, help="Optional local MS JSONL path")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--boost-bias", type=float, default=8.0)
    parser.add_argument("--min-tokens-between-checks", type=int, default=8)
    parser.add_argument("--max-tokens-without-check", type=int, default=32)
    parser.add_argument("--rolling-buffer-chars", type=int, default=32)

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to persistent run log")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = ExperimentLogger(
        log_file=args.log_file,
        default_log_file=str(Path(args.output_json).with_suffix(".log")),
    )
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

    llm_selector = build_selector(
        selector_backend="ollama",
        selector_model=args.selector_model,
        selector_base_url=args.selector_base_url,
        selector_timeout_s=args.selector_timeout_s,
        selector_retries=args.selector_retries,
        logger=lambda msg: logger.info("[selector]", msg),
    )
    fallback_selector = DeterministicFallbackSelector(keep_current_if_possible=True)

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

    logger.info(
        "Running dynamic benchmark:",
        f"split={args.split}",
        f"samples={len(samples)}",
        f"model={args.model_name}",
        f"selector={args.selector_model}",
    )

    per_sample_results, aggregate = run_dynamic_benchmark(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        selector=llm_selector,
        selector_backend_name=f"ollama:{args.selector_model}",
        generation_config=generation_config,
        boundary_config=boundary_config,
        boost_bias=args.boost_bias,
        fallback_selector=fallback_selector,
    )

    run_result = build_run_result(
        method_name="dynamic_instaboost",
        model_name=args.model_name,
        split_name=args.split,
        seed=args.seed,
        decode_config=generation_config.to_dict(),
        aggregate_metrics=aggregate,
        per_sample_results=per_sample_results,
        selector_backend="ollama",
        selector_model=args.selector_model,
        extra_metadata={
            "n_samples": len(samples),
            "boundary_config": {
                "min_tokens_between_checks": args.min_tokens_between_checks,
                "max_tokens_without_check": args.max_tokens_without_check,
                "rolling_buffer_chars": args.rolling_buffer_chars,
            },
            "boost_bias": args.boost_bias,
        },
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    logger.info(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
