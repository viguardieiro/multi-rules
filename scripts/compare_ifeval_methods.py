"""Compare baseline/static/dynamic IFEval result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IFEval method outputs")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--static", type=str, required=True)
    parser.add_argument("--dynamic", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(name: str, payload: dict) -> dict:
    agg = payload["aggregate_metrics"]
    return {
        "method": name,
        "prompt_acc": float(agg["prompt_level_strict_acc"]),
        "inst_acc": float(agg["instruction_level_strict_acc"]),
        "n_samples": int(agg["n_samples"]),
        "n_instructions": int(agg["n_instructions"]),
    }


def _format_table(rows: list[dict]) -> str:
    lines = [
        "| method | prompt_strict_acc | instruction_strict_acc | n_samples | n_instructions |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['prompt_acc']:.4f} | {r['inst_acc']:.4f} | "
            f"{r['n_samples']} | {r['n_instructions']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    baseline = _load(args.baseline)
    static = _load(args.static)
    dynamic = _load(args.dynamic)

    rows = [
        _row("baseline", baseline),
        _row("static_instaboost", static),
        _row("dynamic_instaboost", dynamic),
    ]
    table = _format_table(rows)
    print(table)

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote markdown report to {out}")


if __name__ == "__main__":
    main()
