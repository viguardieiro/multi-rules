"""Rule application ordering oracle for RuleArena airline problems.

This module complements ``get_applied_rules()`` by producing an *execution
order* trace that mirrors RuleArena's fee computation logic in
``datasets/RuleArena/airline/compute_answer.py``.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from src.rulearena.rule_applicability import (
    _get_fee_row_name,
    _get_overweight_names,
    _get_oversize_name,
    get_applied_rules,
)


def _compute_oversize_fee(total_size: float, routine: str) -> int:
    """Return oversize fee exactly as RuleArena's ``compute_oversize``."""
    if total_size <= 62:
        return 0
    if total_size <= 65:
        return 30
    if routine in {
        "Panama",
        "South America",
        "Peru",
        "Colombia",
        "Ecuador",
        "Europe",
        "Israel",
        "Qatar",
    }:
        return 150
    return 200


def _compute_overweight_fee(
    weight: float,
    routine: str,
    customer_class: str,
    complimentary: bool,
) -> int:
    """Return overweight fee exactly as RuleArena's ``compute_overweight``."""
    if routine in {"Australia", "New Zealand"}:
        if complimentary:
            return 0 if weight <= 70 else 200
        if weight <= 50:
            return 0
        if weight <= 53:
            return 30
        if weight <= 70:
            return 200 if routine == "Cuba" else 100
        if routine in {"India", "China", "Japan", "South Korea", "Hong Kong"}:
            return 450
        return 200

    if complimentary and customer_class in {"Business", "First"}:
        if weight <= 70:
            return 0
        if routine in {"India", "China", "Japan", "South Korea", "Hong Kong"}:
            return 450
        return 200

    if weight <= 50:
        return 0
    if weight <= 53:
        return 30
    if weight <= 70:
        return 200 if routine == "Cuba" else 100
    if routine in {"India", "China", "Japan", "South Korea", "Hong Kong"}:
        return 450
    return 200


def _build_name_index(fine_segments: list[dict]) -> dict[str, dict]:
    """Build ``segment_name -> segment_dict`` lookup."""
    return {seg["name"]: seg for seg in fine_segments}


@lru_cache(maxsize=1)
def _load_base_fee_lookup() -> dict[tuple[int, int, str, str], int]:
    """Load RuleArena base fee table values from CSV files.

    Key format: ``(table_idx, direction, customer_class, routine)``.
    """
    project_root = Path(__file__).resolve().parents[2]
    airline_dir = project_root / "datasets" / "RuleArena" / "airline"

    fee_lookup: dict[tuple[int, int, str, str], int] = {}
    for table_idx in range(4):
        table_number = table_idx + 1
        for direction in (0, 1):
            csv_path = airline_dir / "fee_tables" / f"bag_{table_number}" / f"{direction}.csv"
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    routine = row[""]
                    for customer_class, value in row.items():
                        if customer_class == "":
                            continue
                        fee_lookup[(table_idx, direction, customer_class, routine)] = int(value)

    return fee_lookup


def _get_base_fee(
    table_idx: int,
    direction: int,
    customer_class: str,
    routine: str,
) -> int:
    """Return base checked-bag fee from RuleArena CSV tables."""
    lookup = _load_base_fee_lookup()
    return lookup[(table_idx, direction, customer_class, routine)]


def _is_per_bag_row_rule(name: str) -> bool:
    """Return True when the segment is a table-row-like per-bag rule."""
    return (
        name.startswith("first_bag/row_")
        or name.startswith("second_bag/row_")
        or name.startswith("third_bag/row_")
        or name.startswith("fourth_bag/row_")
        or "/row_" in name and name.startswith("weight_and_size/")
    )


def _complementary_gain_for_bag(
    bag: dict,
    routine: str,
    customer_class: str,
) -> int:
    """Return RuleArena complementary gain for one checked bag."""
    total_size = sum(bag["size"])
    oversize = _compute_oversize_fee(total_size, routine)
    overweight_if_comp = _compute_overweight_fee(
        bag["weight"], routine, customer_class, complimentary=True
    )
    overweight_if_not = _compute_overweight_fee(
        bag["weight"], routine, customer_class, complimentary=False
    )
    violation_if_comp = max(oversize, overweight_if_comp)
    violation_if_not = max(oversize, overweight_if_not)
    return violation_if_not - violation_if_comp


def get_checked_bag_processing_order(info: dict) -> list[dict]:
    """Return checked bags in RuleArena processing order.

    Sorting rule mirrors RuleArena:
    1) Descending ``complementary_gain``
    2) Stable tie-break by original bag index
    """
    checked_bags = info["bag_list"][1:]
    routine = info["routine"]
    customer_class = info["customer_class"]

    order = sorted(
        range(len(checked_bags)),
        key=lambda i: (-_complementary_gain_for_bag(checked_bags[i], routine, customer_class), i),
    )

    ranked: list[dict] = []
    for rank, original_idx in enumerate(order):
        bag = checked_bags[original_idx]
        ranked.append(
            {
                "rank": rank + 1,
                "original_bag_index": original_idx + 1,  # 1-based among checked bags
                "complementary_gain": _complementary_gain_for_bag(
                    bag, routine, customer_class
                ),
                "weight": bag["weight"],
                "total_size": sum(bag["size"]),
            }
        )
    return ranked


def _append_step(
    steps: list[dict],
    *,
    rule_name: str,
    phase: str,
    segment_index: dict[str, dict],
    computed_fee: int | None = None,
    original_bag_index: int | None = None,
    bag_rank: int | None = None,
    fee_table_number: int | None = None,
    winner: str | None = None,
) -> None:
    segment = segment_index.get(rule_name)
    steps.append(
        {
            "step_index": len(steps) + 1,
            "phase": phase,
            "rule_name": rule_name,
            "rule_text": segment["substring"] if segment else None,
            "original_bag_index": original_bag_index,
            "bag_rank": bag_rank,
            "fee_table_number": fee_table_number,
            "computed_fee": computed_fee,
            "winner": winner,
        }
    )


def get_rule_application_trace(
    info: dict,
    fine_segments: list[dict],
    *,
    drop_fee_summaries: bool = False,
) -> dict:
    """Return an ordered rule-application trace for one airline sample.

    The trace has two phases:
    1) ``global`` rules (always applicable context)
    2) ``bag`` rules applied per checked bag in RuleArena processing order
    """
    routine = info["routine"]
    direction = info["direction"]
    customer_class = info["customer_class"]
    checked_bags = info["bag_list"][1:]

    segment_index = _build_name_index(fine_segments)
    applicable = get_applied_rules(
        info, fine_segments, drop_fee_summaries=drop_fee_summaries
    )
    applicable_names = {seg["name"] for seg in applicable}

    steps: list[dict] = []

    # Global phase: keep document order but remove per-bag rows and tie-break rule.
    for seg in applicable:
        name = seg["name"]
        if _is_per_bag_row_rule(name):
            continue
        if name == "weight_and_size/overweight_bags/more_than_one_fee":
            continue
        if name in {"first_bag/post_table", "second_bag/post_table"}:
            continue
        _append_step(steps, rule_name=name, phase="global", segment_index=segment_index)

    processing_order = get_checked_bag_processing_order(info)

    for bag_info in processing_order:
        bag_rank = bag_info["rank"]
        original_bag_index = bag_info["original_bag_index"]
        bag = checked_bags[original_bag_index - 1]
        table_idx = min(3, bag_rank - 1)
        table_number = table_idx + 1
        base_fee = _get_base_fee(table_idx, direction, customer_class, routine)

        base_rule_name = _get_fee_row_name(table_idx, routine, direction)
        if base_rule_name and base_rule_name in applicable_names:
            _append_step(
                steps,
                rule_name=base_rule_name,
                phase="bag",
                segment_index=segment_index,
                computed_fee=base_fee,
                original_bag_index=original_bag_index,
                bag_rank=bag_rank,
                fee_table_number=table_number,
            )

        if customer_class == "Main Plus" and table_number in {1, 2}:
            footnote = "first_bag/post_table" if table_number == 1 else "second_bag/post_table"
            if footnote in applicable_names:
                _append_step(
                    steps,
                    rule_name=footnote,
                    phase="bag",
                    segment_index=segment_index,
                    original_bag_index=original_bag_index,
                    bag_rank=bag_rank,
                    fee_table_number=table_number,
                )

        # Complimentary status in RuleArena is based on base fee == 0.
        complimentary = base_fee == 0

        total_size = sum(bag["size"])
        weight = bag["weight"]

        overweight_names = _get_overweight_names(weight, routine)
        overweight_fee = _compute_overweight_fee(
            weight, routine, customer_class, complimentary
        )
        for name in overweight_names:
            if name in applicable_names:
                _append_step(
                    steps,
                    rule_name=name,
                    phase="bag",
                    segment_index=segment_index,
                    computed_fee=overweight_fee,
                    original_bag_index=original_bag_index,
                    bag_rank=bag_rank,
                    fee_table_number=table_number,
                )

        oversize_name = _get_oversize_name(total_size, routine)
        oversize_fee = _compute_oversize_fee(total_size, routine)
        if oversize_name and oversize_name in applicable_names:
            _append_step(
                steps,
                rule_name=oversize_name,
                phase="bag",
                segment_index=segment_index,
                computed_fee=oversize_fee,
                original_bag_index=original_bag_index,
                bag_rank=bag_rank,
                fee_table_number=table_number,
            )

        if (
            (overweight_names and any(n in applicable_names for n in overweight_names))
            or (oversize_name and oversize_name in applicable_names)
        ) and "weight_and_size/overweight_bags/more_than_one_fee" in applicable_names:
            winner = "none"
            if max(overweight_fee, oversize_fee) > 0:
                winner = "oversize" if oversize_fee >= overweight_fee else "overweight"
            _append_step(
                steps,
                rule_name="weight_and_size/overweight_bags/more_than_one_fee",
                phase="bag",
                segment_index=segment_index,
                computed_fee=max(overweight_fee, oversize_fee),
                original_bag_index=original_bag_index,
                bag_rank=bag_rank,
                fee_table_number=table_number,
                winner=winner,
            )

    return {
        "info": info,
        "bag_processing_order": processing_order,
        "steps": steps,
    }
