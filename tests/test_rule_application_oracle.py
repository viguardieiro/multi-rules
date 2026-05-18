"""Tests for rule application ordering oracle."""

import json
from pathlib import Path

import pytest

from src.rulearena.rule_application_oracle import (
    get_checked_bag_processing_order,
    get_rule_application_trace,
)
from src.rulearena.rule_applicability import get_applied_rules
from src.rulearena.rulebook_segments import get_fine_segments


ROOT = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = ROOT / "datasets" / "RuleArena" / "airline" / "reference_rules.txt"
PROBLEMS_DIR = ROOT / "datasets" / "RuleArena" / "airline" / "synthesized_problems"


def _load_sample(comp: int, sample_idx: int) -> dict:
    path = PROBLEMS_DIR / f"comp_{comp}.jsonl"
    with open(path) as f:
        for idx, line in enumerate(f):
            if idx == sample_idx:
                return json.loads(line)
    raise IndexError(f"No sample {sample_idx} in {path}")


@pytest.fixture
def fine_segments():
    return get_fine_segments(RULEBOOK_PATH.read_text())


class TestCheckedBagProcessingOrder:
    def test_reorders_by_complementary_gain_comp0_sample1(self):
        sample = _load_sample(0, 1)
        order = get_checked_bag_processing_order(sample["info"])
        # In this sample, checked bag #3 has highest gain and is processed first.
        assert order[0]["original_bag_index"] == 3
        assert order[0]["complementary_gain"] > 0

    def test_tie_break_uses_original_index_comp2_sample0(self):
        sample = _load_sample(2, 0)
        order = get_checked_bag_processing_order(sample["info"])
        # Bags #3 and #7 tie on gain (70); lower original index comes first.
        assert order[0]["original_bag_index"] == 3
        assert order[1]["original_bag_index"] == 7
        assert order[0]["complementary_gain"] == order[1]["complementary_gain"] == 70


class TestRuleApplicationTrace:
    def test_first_rank_uses_first_bag_table_row(self, fine_segments):
        sample = _load_sample(0, 1)
        trace = get_rule_application_trace(sample["info"], fine_segments)

        first_rank_steps = [s for s in trace["steps"] if s["phase"] == "bag" and s["bag_rank"] == 1]
        base_steps = [s for s in first_rank_steps if s["rule_name"].startswith("first_bag/row_")]

        assert len(base_steps) == 1
        assert base_steps[0]["rule_name"] == "first_bag/row_us_puerto_rico"
        assert base_steps[0]["computed_fee"] == 0

    def test_tie_break_step_present_for_violations(self, fine_segments):
        sample = _load_sample(0, 1)
        trace = get_rule_application_trace(sample["info"], fine_segments)

        first_rank_steps = [s for s in trace["steps"] if s["phase"] == "bag" and s["bag_rank"] == 1]
        tie_break = [
            s
            for s in first_rank_steps
            if s["rule_name"] == "weight_and_size/overweight_bags/more_than_one_fee"
        ]
        assert len(tie_break) == 1
        assert tie_break[0]["winner"] in {"oversize", "overweight"}
        assert tie_break[0]["computed_fee"] > 0

    def test_trace_rule_names_are_subset_of_applicable(self, fine_segments):
        sample = _load_sample(1, 0)
        info = sample["info"]
        applicable_names = {seg["name"] for seg in get_applied_rules(info, fine_segments)}
        trace = get_rule_application_trace(info, fine_segments)
        trace_names = {step["rule_name"] for step in trace["steps"]}

        assert trace_names.issubset(applicable_names)

