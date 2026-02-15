#!/usr/bin/env python3
"""
Hyperparameter sweep for InstABoost bias on the RuleArena benchmark.

Runs eval_rulearena.py with multiple bias values on a small validation set
(first --val_size problems), then collects and compares the results.

The final evaluation should use --start_idx <val_size> to skip the
validation samples and evaluate on the remaining (unseen) problems.

Usage:
    # 1. Sweep (validation on first 20 samples)
    python scripts/sweep_boost_bias.py \\
        --model openai/gpt-oss-20b \\
        --domain airline \\
        --complexity 0 \\
        --bias_values 1.0 2.0 3.0 4.0 5.0 \\
        --val_size 20

    # 2. Final eval with best bias (skip first 20, evaluate remaining 80)
    python scripts/eval_rulearena.py \\
        --model openai/gpt-oss-20b \\
        --domain airline \\
        --complexity 0 \\
        --boost_strategy uniform_rules --bias <best> \\
        --start_idx 20

Results are saved under results/rulearena/<model>/<domain>/comp_<N>/val_<run_id>/
with one subdirectory per bias value. A comparison table is printed at the end
and saved to results/rulearena/<model>/<domain>/comp_<N>/sweep_summary.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent



def read_summary(results_dir: Path) -> dict | None:
    """Read summary.json from a results directory, if it exists."""
    path = results_dir / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_single_eval(model: str, domain: str, complexity: int, bias: float,
                    val_size: int, log_dir: str, boost_strategy: str,
                    extra_args: list[str]) -> dict | None:
    """Run eval_rulearena.py for a single bias value. Returns summary dict or None."""
    # Validation results go under <log_dir>/val/ so they don't collide with
    # final test runs. eval_rulearena.py builds its own directory structure
    # inside the --log_dir we pass it.
    val_log_dir = Path(log_dir) / "val"
    strategy = "none" if bias == 0.0 else boost_strategy
    eval_run_id = "none" if bias == 0.0 else f"{boost_strategy}_bias{bias}"
    safe_model = model.replace("/", "_")
    eval_rdir = (val_log_dir / "rulearena" / safe_model / domain
                 / f"comp_{complexity}" / eval_run_id)

    # Skip if summary already exists with the right sample count
    existing = read_summary(eval_rdir)
    if existing and existing.get("total_count", 0) >= val_size:
        print(f"\n  [skip] bias={bias} — already have {existing['total_count']} samples")
        return existing

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "eval_rulearena.py"),
        "--model", model,
        "--domain", domain,
        "--complexity", str(complexity),
        "--boost_strategy", strategy,
        "--bias", str(bias),
        "--start_idx", "0",
        "--max_problems", str(val_size),
        "--log_dir", str(val_log_dir),
        *extra_args,
    ]

    print(f"\n{'='*60}")
    print(f"  Running bias={bias} ({strategy}) — {val_size} samples")
    print(f"  Results dir: {eval_rdir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  [ERROR] eval_rulearena.py exited with code {result.returncode}")
        return None

    return read_summary(eval_rdir)


def print_comparison(summaries: list[tuple[float, dict | None]]) -> None:
    """Print a comparison table of bias values vs accuracy."""
    print(f"\n{'='*70}")
    print(f"  SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"  {'Bias':>6}  {'Accuracy':>10}  {'Correct':>9}  {'Finished':>10}  {'Avg Tokens':>11}  {'Time':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*11}  {'─'*8}")

    for bias, summary in summaries:
        if summary is None:
            print(f"  {bias:>6.1f}  {'FAILED':>10}")
            continue
        acc = summary["accuracy"]
        correct = summary["correct_count"]
        total = summary["total_count"]
        finished_ratio = summary["generation_finished_ratio"]
        avg_tokens = summary.get("avg_output_length_tokens", 0)
        wall = summary.get("wall_time_seconds", 0)
        wall_min = wall / 60
        print(f"  {bias:>6.1f}  {acc:>9.1%}  {correct:>4}/{total:<4}  {finished_ratio:>9.1%}  {avg_tokens:>11.0f}  {wall_min:>6.1f}m")

    print(f"{'='*70}\n")


def save_sweep_summary(log_dir: str, model: str, domain: str,
                       complexity: int, val_size: int,
                       summaries: list[tuple[float, dict | None]],
                       wall_time: float) -> Path:
    """Save the sweep comparison to a JSON file."""
    safe_model = model.replace("/", "_")
    sweep_dir = Path(log_dir) / "val" / "rulearena" / safe_model / domain / f"comp_{complexity}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_path = sweep_dir / "sweep_summary.json"

    rows = []
    for bias, summary in summaries:
        row = {"bias": bias}
        if summary:
            row.update({
                "accuracy": summary["accuracy"],
                "correct_count": summary["correct_count"],
                "total_count": summary["total_count"],
                "generation_finished_ratio": summary["generation_finished_ratio"],
                "avg_output_length_tokens": summary.get("avg_output_length_tokens"),
                "wall_time_seconds": summary.get("wall_time_seconds"),
            })
        else:
            row["error"] = True
        rows.append(row)

    sweep = {
        "model": model,
        "domain": domain,
        "complexity": complexity,
        "val_size": val_size,
        "bias_values": [b for b, _ in summaries],
        "results": rows,
        "total_sweep_time_seconds": round(wall_time, 1),
    }
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"  Sweep summary saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for InstABoost bias on RuleArena"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["airline", "nba", "tax"])
    parser.add_argument("--complexity", type=int, default=0,
                        choices=[0, 1, 2])
    parser.add_argument("--boost_strategy", type=str, default="uniform_rules",
                        choices=["uniform_rules", "uniform_question"],
                        help="Boost strategy to use (default: uniform_rules)")
    parser.add_argument("--bias_values", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 4.0, 5.0],
                        help="Bias values to sweep (default: 1.0 2.0 3.0 4.0 5.0)")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Include a bias=0.0 baseline run (off by default)")
    parser.add_argument("--val_size", type=int, default=20,
                        help="Number of validation samples per bias value (default: 20)")
    parser.add_argument("--log_dir", type=str, default="results",
                        help="Base directory for results")
    # Pass-through args for eval_rulearena.py
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--textual", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)

    args = parser.parse_args()

    # Build the list of bias values to test
    bias_values = sorted(set(args.bias_values))
    if args.include_baseline and 0.0 not in bias_values:
        bias_values = [0.0] + bias_values

    # Build extra args to forward
    extra_args = []
    if args.use_example:
        extra_args.append("--use_example")
    if args.textual:
        extra_args.append("--textual")
    if args.max_new_tokens is not None:
        extra_args.extend(["--max_new_tokens", str(args.max_new_tokens)])

    print(f"\n  InstABoost Bias Sweep")
    print(f"  Model:      {args.model}")
    print(f"  Domain:     {args.domain}/comp_{args.complexity}")
    print(f"  Val size:   {args.val_size} samples (problems 0–{args.val_size - 1})")
    print(f"  Bias values: {bias_values}")
    print(f"  Extra args:  {extra_args or '(none)'}")
    print(f"\n  Note: Final eval should use --start_idx {args.val_size} to skip validation samples")

    # Run evaluations
    summaries = []
    sweep_start = time.time()

    for bias in bias_values:
        summary = run_single_eval(
            model=args.model,
            domain=args.domain,
            complexity=args.complexity,
            bias=bias,
            val_size=args.val_size,
            log_dir=args.log_dir,
            boost_strategy=args.boost_strategy,
            extra_args=extra_args,
        )
        summaries.append((bias, summary))

    sweep_time = time.time() - sweep_start

    # Print and save comparison
    print_comparison(summaries)
    save_sweep_summary(args.log_dir, args.model, args.domain,
                       args.complexity, args.val_size, summaries, sweep_time)

    sweep_min = sweep_time / 60
    print(f"  Total sweep time: {sweep_min:.1f} minutes")
    print(f"\n  Next step: run final eval with the best bias using --start_idx {args.val_size}\n")


if __name__ == "__main__":
    main()
