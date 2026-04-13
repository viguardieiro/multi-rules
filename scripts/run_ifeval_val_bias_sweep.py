"""Sweep boost_bias on validation for static and dynamic methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dynamic_boost import BoundaryConfig, DeterministicFallbackSelector
from src.ifeval_dynamic.data_adapter import load_ifeval_samples
from src.ifeval_dynamic.logging_utils import ExperimentLogger
from src.ifeval_dynamic.runtime_checks import require_ifeval_runtime_dependencies
from src.ifeval_dynamic.runner import (
    GenerationConfig,
    build_selector,
    load_transformers_model_and_tokenizer,
    run_dynamic_benchmark,
    run_static_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation boost-bias sweep for IFEval")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--selector-model", type=str, required=True)
    p.add_argument("--selector-base-url", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--selector-timeout-s", type=float, default=30.0)
    p.add_argument("--selector-retries", type=int, default=1)
    p.add_argument("--bias-values", type=str, default="0,2,4,6,8,10,12")
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
    p.add_argument("--min-tokens-between-checks", type=int, default=8)
    p.add_argument("--max-tokens-without-check", type=int, default=32)
    p.add_argument("--rolling-buffer-chars", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--output-md", type=str, default=None)
    p.add_argument("--log-file", type=str, default=None, help="Optional path to persistent run log")
    return p.parse_args()


def _parse_bias_values(s: str) -> list[float]:
    values = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("No valid --bias-values provided")
    return values


def _pick_best(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda r: (
            float(r["prompt_level_strict_acc"]),
            float(r["instruction_level_strict_acc"]),
            -int(r.get("selector_calls", 0)),
        ),
        reverse=True,
    )[0]


def _table(rows: list[dict]) -> str:
    lines = [
        "| method | boost_bias | prompt_strict_acc | instruction_strict_acc | selector_calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {float(r['boost_bias']):.2f} | "
            f"{float(r['prompt_level_strict_acc']):.4f} | {float(r['instruction_level_strict_acc']):.4f} | "
            f"{int(r.get('selector_calls', 0))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    logger = ExperimentLogger(
        log_file=args.log_file,
        default_log_file=str(Path(args.output_json).with_suffix(".log")),
    )
    require_ifeval_runtime_dependencies()
    biases = _parse_bias_values(args.bias_values)

    val_samples, _ = load_ifeval_samples(
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        ms_jsonl_path=args.ms_jsonl_path,
    )
    samples = val_samples[: args.limit] if args.limit > 0 else val_samples
    if not samples:
        raise ValueError("No validation samples found")

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
    selector = build_selector(
        selector_backend="ollama",
        selector_model=args.selector_model,
        selector_base_url=args.selector_base_url,
        selector_timeout_s=args.selector_timeout_s,
        selector_retries=args.selector_retries,
        logger=lambda m: logger.info("[selector]", m),
    )
    fallback = DeterministicFallbackSelector(keep_current_if_possible=True)

    static_rows = []
    dynamic_rows = []
    for bias in biases:
        logger.info(f"Running bias={bias} static...")
        _, static_agg = run_static_benchmark(
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            generation_config=generation_config,
            boost_bias=bias,
        )
        static_rows.append(
            {
                "method": "static_instaboost",
                "boost_bias": bias,
                "prompt_level_strict_acc": static_agg["prompt_level_strict_acc"],
                "instruction_level_strict_acc": static_agg["instruction_level_strict_acc"],
                "selector_calls": 0,
            }
        )

        logger.info(f"Running bias={bias} dynamic...")
        dynamic_samples, dynamic_agg = run_dynamic_benchmark(
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            selector=selector,
            selector_backend_name=f"ollama:{args.selector_model}",
            generation_config=generation_config,
            boundary_config=boundary_config,
            boost_bias=bias,
            fallback_selector=fallback,
        )
        total_selector_calls = sum(
            int(x["method_metadata"]["dynamic_trace"]["selector_calls"])
            for x in dynamic_samples
        )
        dynamic_rows.append(
            {
                "method": "dynamic_instaboost",
                "boost_bias": bias,
                "prompt_level_strict_acc": dynamic_agg["prompt_level_strict_acc"],
                "instruction_level_strict_acc": dynamic_agg["instruction_level_strict_acc"],
                "selector_calls": total_selector_calls,
            }
        )

    best_static = _pick_best(static_rows)
    best_dynamic = _pick_best(dynamic_rows)
    payload = {
        "model_name": args.model_name,
        "selector_model": args.selector_model,
        "seed": args.seed,
        "n_samples": len(samples),
        "bias_values": biases,
        "static_results": static_rows,
        "dynamic_results": dynamic_rows,
        "selected": {
            "static_best_bias": best_static["boost_bias"],
            "dynamic_best_bias": best_dynamic["boost_bias"],
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote sweep json to {out_json}")

    if args.output_md:
        md = "# IFEval Validation Bias Sweep\n\n"
        md += "## Static\n\n" + _table(static_rows) + "\n"
        md += "## Dynamic\n\n" + _table(dynamic_rows) + "\n"
        md += (
            f"Selected static bias: `{best_static['boost_bias']}`\n\n"
            f"Selected dynamic bias: `{best_dynamic['boost_bias']}`\n"
        )
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        logger.info(f"Wrote sweep markdown to {out_md}")


if __name__ == "__main__":
    main()
