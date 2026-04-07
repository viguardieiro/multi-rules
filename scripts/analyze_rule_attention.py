#!/usr/bin/env python3
"""
Analyse per-rule-segment attention for airline RuleArena problems.

For each problem the script:
1. Builds the prompt (same as eval_rulearena.py).
2. Maps fine-grained rule segments to token indices.
3. Registers read-only attention capture hooks.
4. Runs greedy generation and records post-softmax attention per segment.
5. Evaluates accuracy.
6. Saves per-sample JSON / NPZ and a cross-sample summary.

Requires ``attn_implementation="eager"`` so that softmax is explicit.
"""

import os
import pwd
import tempfile
# Fix containers where uid has no passwd entry (LOGNAME may be set but empty)
try:
    pwd.getpwuid(os.getuid())
except KeyError:
    uid = os.getuid()
    if not os.environ.get("LOGNAME") and not os.environ.get("USER"):
        os.environ["LOGNAME"] = f"uid{uid}"
    # Set TORCHINDUCTOR_CACHE_DIR to bypass torch._inductor's getuser() call
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"torchinductor_uid{uid}"),
    )

import kernels  # must be imported before transformers for gpt-oss MXFP4 models

import sys

# torchvision 0.19.1 is incompatible with torch 2.10.0; mark it unavailable so
# transformers skips all torchvision-dependent code paths.
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None  # find_spec returns None → is_torchvision_available() = False

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_rulearena import (
    SYSTEM_PROMPTS,
    load_problems_airline,
    load_reference_rules,
    build_prompt_airline,
    format_chat_prompt,
    eval_accuracy_airline,
    parse_gpt_oss_output,
    _NumpyEncoder,
)
from src.rulearena.rulebook_segments import get_fine_segments, get_coarse_segments
from src.rulearena.rule_applicability import get_applied_rules, build_filtered_rulebook
from src.token_utils import segments_to_token_indices
from src.attention_capture import (
    SegmentAttentionMap,
    register_capture_hooks,
    unregister_capture_hooks,
    reset_capture_data,
    get_capture_results,
)


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyse per-rule attention on airline RuleArena problems"
    )
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name")
    p.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--max_problems", type=int, default=20)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--use_example", action="store_true",
                   help="Include few-shot example in prompt")
    p.add_argument("--log_dir", type=str, default="results")
    p.add_argument("--max_new_tokens", type=int, default=16000)
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices to capture (default: all)")
    p.add_argument("--rules_strategy", type=str, default="full",
                   choices=["full", "applicable_only"],
                   help="Whether to use the full rulebook or only applicable rules")
    p.add_argument("--drop_fee_summaries", action="store_true",
                   help="Drop fee summary sections when rules_strategy=applicable_only")
    return p.parse_args(argv)


# ===================================================================
# Helpers
# ===================================================================

def build_segment_attention_map(
    fine_segments: list[dict],
    formatted_prompt: str,
    reference_rules: str,
    tokenizer,
    applied_rule_names: set[str],
) -> SegmentAttentionMap:
    """Build a SegmentAttentionMap from fine segments + token indices."""
    seg_with_tokens = segments_to_token_indices(
        fine_segments, formatted_prompt, reference_rules, tokenizer,
        add_special_tokens=False,
    )
    return SegmentAttentionMap(
        segment_names=[s["name"] for s in seg_with_tokens],
        segment_token_sets=[s["token_indices"] for s in seg_with_tokens],
        segment_is_applicable=[
            s["name"] in applied_rule_names for s in seg_with_tokens
        ],
    )


def build_attention_cube(results: dict, num_segments: int) -> np.ndarray:
    """Build a ``[num_layers, num_steps, num_segments]`` numpy array."""
    data = results["data"]
    num_layers = results["num_layers"]
    num_steps = results["num_steps"]
    cube = np.zeros((num_layers, num_steps, num_segments), dtype=np.float32)
    for layer_idx, steps_dict in data.items():
        for step, seg_list in steps_dict.items():
            cube[layer_idx, step, :len(seg_list)] = seg_list
    return cube


def compute_applicable_vs_nonapplicable(
    cube: np.ndarray,
    is_applicable: list[bool],
    num_tokens_per_segment: list[int],
) -> dict:
    """Compare attention on applicable vs non-applicable segments."""
    app_mask = np.array(is_applicable, dtype=bool)
    non_mask = ~app_mask

    # Average across layers and steps → [num_segments]
    avg_per_seg = cube.mean(axis=(0, 1))

    app_tokens = sum(n for n, a in zip(num_tokens_per_segment, is_applicable) if a)
    non_tokens = sum(n for n, a in zip(num_tokens_per_segment, is_applicable) if not a)

    total_app = float(avg_per_seg[app_mask].sum()) if app_mask.any() else 0.0
    total_non = float(avg_per_seg[non_mask].sum()) if non_mask.any() else 0.0

    return {
        "total_applicable_attention": round(total_app, 6),
        "total_nonapplicable_attention": round(total_non, 6),
        "applicable_per_token": round(total_app / app_tokens, 6) if app_tokens else 0.0,
        "nonapplicable_per_token": round(total_non / non_tokens, 6) if non_tokens else 0.0,
    }


# ===================================================================
# Main
# ===================================================================

def main(argv=None):
    args = parse_args(argv)

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    # Results directory
    safe_model = args.model.replace("/", "_")
    strategy_suffix = f"_{args.rules_strategy}" if args.rules_strategy != "full" else ""
    results_dir = (
        Path(args.log_dir) / "attention_analysis" / safe_model
        / "airline" / f"comp_{args.complexity}{strategy_suffix}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "model": args.model,
        "domain": "airline",
        "complexity": args.complexity,
        "max_problems": args.max_problems,
        "start_idx": args.start_idx,
        "use_example": args.use_example,
        "max_new_tokens": args.max_new_tokens,
        "layers": layers,
        "rules_strategy": args.rules_strategy,
        "drop_fee_summaries": args.drop_fee_summaries,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load model with eager attention
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Load problems & reference rules
    problems = load_problems_airline(args.complexity)
    reference_rules = load_reference_rules("airline")
    fine_segments = get_fine_segments(reference_rules)
    coarse_segments = get_coarse_segments(reference_rules) if args.rules_strategy == "applicable_only" else None
    print(f"Loaded {len(fine_segments)} fine segments")

    system_prompt = SYSTEM_PROMPTS["airline"]

    problems = problems[args.start_idx:]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    print(f"Processing {len(problems)} problems "
          f"(start_idx={args.start_idx}, complexity={args.complexity}, "
          f"rules_strategy={args.rules_strategy})")

    sample_results = []
    correct_count = 0
    wall_start = time.time()
    handle = None

    for local_idx, problem in enumerate(problems):
        idx = local_idx + args.start_idx
        print(f"\n--- Problem {idx} ({local_idx + 1}/{len(problems)}) ---")
        sample_start = time.time()

        # Get applicable rules (needed for both strategies)
        applied = get_applied_rules(
            problem["info"], fine_segments,
            drop_fee_summaries=args.drop_fee_summaries,
        )
        applied_names = {seg["name"] for seg in applied}

        # Determine which rules text and segments to use
        if args.rules_strategy == "applicable_only":
            prompt_rules = build_filtered_rulebook(
                problem["info"], reference_rules, fine_segments, coarse_segments,
                drop_fee_summaries=args.drop_fee_summaries,
            )
            # Recompute char_start relative to prompt_rules (filtered text)
            # since the original char_start values are relative to reference_rules
            adjusted = []
            for seg in applied:
                pos = prompt_rules.find(seg["substring"])
                if pos == -1:
                    raise ValueError(
                        f"Segment '{seg['name']}' substring not found in filtered rulebook"
                    )
                adjusted.append({**seg, "char_start": pos})
            tracked_segments = adjusted
        else:
            prompt_rules = reference_rules
            tracked_segments = fine_segments  # all segments

        # Build prompt
        user_prompt, rules_text, question_text = build_prompt_airline(
            problem, prompt_rules, args.use_example
        )
        formatted_prompt = format_chat_prompt(system_prompt, user_prompt, tokenizer)

        # Build segment attention map
        seg_map = build_segment_attention_map(
            tracked_segments, formatted_prompt, prompt_rules,
            tokenizer, applied_names,
        )

        # Register hooks (or reset existing)
        if handle is None:
            handle = register_capture_hooks(model, seg_map, layers=layers)
        else:
            # Unregister old hooks and register fresh (segment map changes per sample)
            unregister_capture_hooks(handle)
            handle = register_capture_hooks(model, seg_map, layers=layers)

        # Tokenize
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=False,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Collect attention results
        cap_results = get_capture_results(handle)
        num_segments = cap_results["num_segments"]

        # Build attention cube [layers, steps, segments]
        cube = build_attention_cube(cap_results, num_segments)

        # Decode response
        generated_tokens = output_ids[0][input_length:]
        output_length_tokens = len(generated_tokens)

        response_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        parsed = parse_gpt_oss_output(response_raw)
        if parsed["final"] is not None:
            response = parsed["final"]
        else:
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Evaluate accuracy
        is_correct, predicted, truth = eval_accuracy_airline(
            response, problem["info"]
        )
        if is_correct:
            correct_count += 1

        # Compute metrics
        num_tokens_per_seg = [len(ts) for ts in seg_map.segment_token_sets]
        app_vs_non = compute_applicable_vs_nonapplicable(
            cube, seg_map.segment_is_applicable, num_tokens_per_seg,
        )

        # Average attention per segment across layers+steps → [segments]
        avg_attention_per_seg = cube.mean(axis=(0, 1)).tolist()
        # Per-layer average across steps → [layers, segments]
        attention_per_layer_seg = cube.mean(axis=1).tolist()
        # Per-step average across layers → [steps, segments]
        attention_per_step_seg = cube.mean(axis=0).tolist()

        sample = {
            "sample_idx": idx,
            "rules_strategy": args.rules_strategy,
            "problem_info": problem["info"],
            "input_length_tokens": int(input_length),
            "output_length_tokens": int(output_length_tokens),
            "num_layers": int(cap_results["num_layers"]),
            "num_steps": int(cap_results["num_steps"]),
            "num_segments": num_segments,
            "segment_names": seg_map.segment_names,
            "segment_is_applicable": seg_map.segment_is_applicable,
            "segment_num_tokens": num_tokens_per_seg,
            "attention_per_segment": avg_attention_per_seg,
            "attention_per_layer_segment": attention_per_layer_seg,
            "attention_per_step_segment": attention_per_step_seg,
            "applicable_vs_nonapplicable": app_vs_non,
            "predicted_answer": predicted,
            "ground_truth_answer": truth,
            "is_correct": is_correct,
        }
        sample_results.append(sample)

        # Save per-sample files
        with open(results_dir / f"{idx}.json", "w") as f:
            json.dump(sample, f, indent=2, cls=_NumpyEncoder)
        np.savez_compressed(results_dir / f"{idx}_attention.npz", attention=cube)

        sample_time = time.time() - sample_start
        done = local_idx + 1
        elapsed = time.time() - wall_start
        eta = (elapsed / done) * (len(problems) - done)
        eta_min, eta_sec = divmod(int(eta), 60)
        acc_so_far = correct_count / done
        print(f"  Correct: {is_correct} | Predicted: {predicted} | Truth: {truth}"
              f" | steps={cap_results['num_steps']}"
              f" | {sample_time:.1f}s"
              f" | Running: {correct_count}/{done} ({acc_so_far:.1%})"
              f" | ETA: {eta_min}m{eta_sec:02d}s")
        print(f"  Applicable attn: {app_vs_non['total_applicable_attention']:.4f}"
              f" | Non-applicable: {app_vs_non['total_nonapplicable_attention']:.4f}"
              f" | Per-token ratio: "
              f"{app_vs_non['applicable_per_token'] / max(app_vs_non['nonapplicable_per_token'], 1e-10):.2f}x")

    # Unregister hooks
    if handle is not None:
        unregister_capture_hooks(handle)

    wall_time = time.time() - wall_start

    # --- Cross-sample summary ---
    total = len(sample_results)
    correct = sum(1 for s in sample_results if s["is_correct"])

    # Average applicable vs non-applicable across samples
    avg_app = np.mean([s["applicable_vs_nonapplicable"]["total_applicable_attention"]
                       for s in sample_results]) if sample_results else 0.0
    avg_non = np.mean([s["applicable_vs_nonapplicable"]["total_nonapplicable_attention"]
                       for s in sample_results]) if sample_results else 0.0
    avg_app_per_tok = np.mean([s["applicable_vs_nonapplicable"]["applicable_per_token"]
                               for s in sample_results]) if sample_results else 0.0
    avg_non_per_tok = np.mean([s["applicable_vs_nonapplicable"]["nonapplicable_per_token"]
                               for s in sample_results]) if sample_results else 0.0

    # Attention on applicable rules: correct vs incorrect samples
    correct_samples = [s for s in sample_results if s["is_correct"]]
    incorrect_samples = [s for s in sample_results if not s["is_correct"]]
    app_attn_correct = (
        np.mean([s["applicable_vs_nonapplicable"]["applicable_per_token"]
                 for s in correct_samples])
        if correct_samples else None
    )
    app_attn_incorrect = (
        np.mean([s["applicable_vs_nonapplicable"]["applicable_per_token"]
                 for s in incorrect_samples])
        if incorrect_samples else None
    )

    summary = {
        "config": config_dict,
        "accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "total_count": total,
        "avg_applicable_attention": float(avg_app),
        "avg_nonapplicable_attention": float(avg_non),
        "avg_applicable_per_token": float(avg_app_per_tok),
        "avg_nonapplicable_per_token": float(avg_non_per_tok),
        "accuracy_attention_correlation": {
            "applicable_per_token_correct_samples": (
                float(app_attn_correct) if app_attn_correct is not None else None
            ),
            "applicable_per_token_incorrect_samples": (
                float(app_attn_incorrect) if app_attn_incorrect is not None else None
            ),
        },
        "wall_time_seconds": round(wall_time, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    print(f"\n=== Final: {correct}/{total} = {correct / total if total else 0:.4f} ===")
    print(f"Avg applicable per-token attention: {avg_app_per_tok:.6f}")
    print(f"Avg non-applicable per-token attention: {avg_non_per_tok:.6f}")
    if app_attn_correct is not None and app_attn_incorrect is not None:
        print(f"Applicable per-token (correct): {app_attn_correct:.6f}")
        print(f"Applicable per-token (incorrect): {app_attn_incorrect:.6f}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
