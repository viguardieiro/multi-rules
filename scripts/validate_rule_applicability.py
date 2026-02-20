#!/usr/bin/env python3
"""Cross-validate get_applied_rules() against RuleArena's ground-truth fee computation.

For each of the 300 airline samples, this script:
  1. Runs get_applied_rules() to get applicable fine segment names.
  2. Runs compute_answer() to get the fee breakdown per bag.
  3. Checks that every fee-relevant rule (base fee row, overweight bracket,
     oversize row) used by compute_answer is present in the applicable rules.

Usage:
    python scripts/validate_rule_applicability.py              # validate only
    python scripts/validate_rule_applicability.py --generate   # validate + generate annotation JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
AIRLINE_DIR = ROOT / "datasets" / "RuleArena" / "airline"
RULEBOOK_PATH = AIRLINE_DIR / "reference_rules.txt"
PROBLEMS_DIR = AIRLINE_DIR / "synthesized_problems"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AIRLINE_DIR))

import os  # noqa: E402

from src.rulearena.rulebook_segments import get_fine_segments, get_coarse_segments  # noqa: E402
from src.rulearena.rule_applicability import (  # noqa: E402
    get_applied_rules,
    get_applied_rules_with_coarse,
    _get_fee_row_name,
    _get_overweight_names,
    _get_oversize_name,
)

# compute_answer.py loads fee tables at module level using relative paths,
# so we must chdir before importing it.
_orig_cwd = os.getcwd()
os.chdir(str(AIRLINE_DIR))
from compute_answer import compute_answer, load_checking_fee  # noqa: E402
check_base_tables = load_checking_fee()
os.chdir(_orig_cwd)

# ---------------------------------------------------------------------------
# Load segments
# ---------------------------------------------------------------------------
rulebook_text = RULEBOOK_PATH.read_text()
fine_segments = get_fine_segments(rulebook_text)
coarse_segments = get_coarse_segments(rulebook_text)

ANNOTATION_OUTPUT = ROOT / "results" / "rulearena" / "rules" / "airline_sample_rules.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_samples() -> list[dict]:
    """Load all 300 airline samples across complexity levels."""
    samples = []
    for comp in range(3):
        path = PROBLEMS_DIR / f"comp_{comp}.jsonl"
        with open(path) as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                obj["complexity"] = comp
                obj["sample_idx"] = idx
                samples.append(obj)
    return samples


def validate_sample(info: dict) -> list[str]:
    """Validate one sample. Returns a list of error messages (empty = pass)."""
    errors: list[str] = []

    routine = info["routine"]
    direction = info["direction"]
    customer_class = info["customer_class"]
    bag_list = info["bag_list"]
    checked_bags = bag_list[1:]  # index 0 is carry-on

    # Get applicable fine rules
    applicable = get_applied_rules(info, fine_segments)
    applicable_names = {seg["name"] for seg in applicable}

    # Run ground-truth computation
    os.chdir(str(AIRLINE_DIR))
    _, gt_info = compute_answer(
        base_price=info.get("base_price", 100),
        direction=direction,
        routine=routine,
        customer_class=customer_class,
        bag_list=bag_list,
        check_base_tables=check_base_tables,
    )
    os.chdir(_orig_cwd)

    # --- 1. Check base fee table rows ---
    # compute_answer reorders bags by complementary gain, but gt_info["base"]
    # is in original order after invert_order.
    for bag_idx in range(len(checked_bags)):
        table_idx = min(3, bag_idx)
        expected_row = _get_fee_row_name(table_idx, routine, direction)
        if expected_row and expected_row not in applicable_names:
            errors.append(
                f"Missing base fee row for bag {bag_idx}: {expected_row}"
            )

    # --- 2. Check overweight rules ---
    for bag_idx, bag in enumerate(checked_bags):
        weight = bag["weight"]
        if weight <= 50:
            continue

        # Check that the correct overweight bracket row is present
        ow_names = _get_overweight_names(weight, routine)
        for ow_name in ow_names:
            if ow_name not in applicable_names:
                errors.append(
                    f"Missing overweight rule for bag {bag_idx} "
                    f"(weight={weight}): {ow_name}"
                )

        # Also check the "more than one fee" intro rule
        if "weight_and_size/overweight_bags/more_than_one_fee" not in applicable_names:
            errors.append(
                "Missing 'more_than_one_fee' intro for overweight bag"
            )

    # --- 3. Check oversize rules ---
    for bag_idx, bag in enumerate(checked_bags):
        total_size = sum(bag["size"])
        if total_size <= 62:
            continue

        os_name = _get_oversize_name(total_size, routine)
        if os_name and os_name not in applicable_names:
            errors.append(
                f"Missing oversize rule for bag {bag_idx} "
                f"(size={total_size}): {os_name}"
            )

    # --- 4. Cross-check ground-truth overweight/oversize against applicable rules ---
    # compute_answer reorders bags to optimize complementary assignment, so we
    # cannot check per-bag position base fees. Instead verify the ground-truth
    # overweight/oversize amounts are non-zero only for bags where our rules
    # include the corresponding bracket.
    ow_list = gt_info["overweight"]
    os_list = gt_info["oversize"]
    for bag_idx, bag in enumerate(checked_bags):
        weight = bag["weight"]
        ow_fee = ow_list[bag_idx]

        if ow_fee > 0 and weight > 50:
            # Verify the correct overweight bracket is in applicable rules
            ow_names = _get_overweight_names(weight, routine)
            if not ow_names:
                errors.append(
                    f"Ground-truth overweight fee {ow_fee} for bag {bag_idx} "
                    f"(weight={weight}) but no matching bracket found"
                )

        total_size = sum(bag["size"])
        os_fee = os_list[bag_idx]
        if os_fee > 0 and total_size > 62:
            os_name = _get_oversize_name(total_size, routine)
            if not os_name:
                errors.append(
                    f"Ground-truth oversize fee {os_fee} for bag {bag_idx} "
                    f"(size={total_size}) but no matching oversize rule found"
                )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_annotations(samples: list[dict]) -> list[dict]:
    """Generate the pre-computed rule annotation for every sample."""
    annotations = []
    for sample in samples:
        info = sample["info"]
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        annotations.append({
            "complexity": sample["complexity"],
            "sample_idx": sample["sample_idx"],
            "info": info,
            "applicable_fine_segments": [s["name"] for s in result["fine"]],
            "applicable_coarse_segments": [s["name"] for s in result["coarse"]],
            "fine_to_coarse": result["mapping"],
            "num_fine_rules": len(result["fine"]),
            "num_coarse_sections": len(result["coarse"]),
        })
    return annotations


def main():
    do_generate = "--generate" in sys.argv

    samples = load_all_samples()
    print(f"Loaded {len(samples)} samples across 3 complexity levels.\n")

    total_pass = 0
    total_fail = 0
    all_errors: list[tuple[int, int, list[str]]] = []

    for sample in samples:
        info = sample["info"]
        comp = sample["complexity"]
        idx = sample["sample_idx"]

        errs = validate_sample(info)
        if errs:
            total_fail += 1
            all_errors.append((comp, idx, errs))
        else:
            total_pass += 1

    # Report results
    print(f"Results: {total_pass} passed, {total_fail} failed "
          f"out of {len(samples)} samples.\n")

    if all_errors:
        print("FAILURES:")
        for comp, idx, errs in all_errors:
            print(f"\n  comp_{comp} sample {idx}:")
            for e in errs:
                print(f"    - {e}")
        sys.exit(1)
    else:
        print("All samples passed cross-validation!")

    if do_generate:
        print(f"\nGenerating annotation file...")
        annotations = generate_annotations(samples)
        ANNOTATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(ANNOTATION_OUTPUT, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Wrote {len(annotations)} annotations to {ANNOTATION_OUTPUT}")


if __name__ == "__main__":
    main()
