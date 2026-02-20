"""Tests for src.rulearena.rule_applicability."""

from pathlib import Path

import pytest

from src.rulearena.rulebook_segments import get_fine_segments
from src.rulearena.rule_applicability import (
    get_applied_rules,
    get_coarse_for_fine,
    get_applied_rules_with_coarse,
    build_filtered_rulebook,
    COMPLEMENTARY_FIRST_DESTINATIONS,
)
from src.rulearena.rulebook_segments import get_coarse_segments

RULEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "RuleArena"
    / "airline"
    / "reference_rules.txt"
)


@pytest.fixture
def rulebook_text():
    return RULEBOOK_PATH.read_text()


@pytest.fixture
def fine_segments(rulebook_text):
    return get_fine_segments(rulebook_text)


@pytest.fixture
def coarse_segments(rulebook_text):
    return get_coarse_segments(rulebook_text)


def _names(result: list[dict]) -> list[str]:
    """Extract segment names from result list."""
    return [seg["name"] for seg in result]


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestBasicStructure:
    def test_returns_list_of_dicts(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},  # carry-on
                {"size": [28, 18, 12], "weight": 40},  # 1st checked
            ],
        }
        result = get_applied_rules(info, fine_segments)
        assert isinstance(result, list)
        assert all(isinstance(seg, dict) for seg in result)
        assert all("name" in seg and "substring" in seg for seg in result)

    def test_result_is_subset_of_fine_segments(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        result = get_applied_rules(info, fine_segments)
        fine_names = {seg["name"] for seg in fine_segments}
        for seg in result:
            assert seg["name"] in fine_names

    def test_preserves_rulebook_order(self, fine_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
                {"size": [30, 20, 14], "weight": 45},
            ],
        }
        result = get_applied_rules(info, fine_segments)
        fine_order = {seg["name"]: i for i, seg in enumerate(fine_segments)}
        indices = [fine_order[seg["name"]] for seg in result]
        assert indices == sorted(indices)

    def test_no_duplicate_segments(self, fine_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
                {"size": [30, 20, 14], "weight": 65},
            ],
        }
        result = get_applied_rules(info, fine_segments)
        names = _names(result)
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Always-included sections
# ---------------------------------------------------------------------------


class TestAlwaysIncluded:
    def test_carry_on_rules_always_present(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Basic Economy",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        carry_on = [n for n in names if n.startswith("carry_on/")]
        assert len(carry_on) >= 4

    def test_checked_bags_intro_always_present(self, fine_segments):
        info = {
            "routine": "Europe",
            "direction": 1,
            "customer_class": "First",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        intro = [n for n in names if n.startswith("checked_bags_intro/")]
        assert len(intro) >= 3

    def test_weight_size_intro_always_present(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/intro/we_calculate_the_size" in names

    def test_preamble_always_present(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "preamble/all_published_bag_fees" in names


# ---------------------------------------------------------------------------
# Cuba direction handling
# ---------------------------------------------------------------------------


class TestCubaDirection:
    def test_to_cuba(self, fine_segments):
        info = {
            "routine": "Cuba",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_to_cuba" in names
        assert "first_bag/row_from_cuba" not in names

    def test_from_cuba(self, fine_segments):
        info = {
            "routine": "Cuba",
            "direction": 1,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_from_cuba" in names
        assert "first_bag/row_to_cuba" not in names

    def test_cuba_second_bag_direction(self, fine_segments):
        info = {
            "routine": "Cuba",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "second_bag/row_to_cuba" in names
        assert "second_bag/row_from_cuba" not in names


# ---------------------------------------------------------------------------
# Fee table row selection
# ---------------------------------------------------------------------------


class TestFeeTableRows:
    def test_us_domestic_first_bag(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_us_puerto_rico" in names
        # Should NOT have other first_bag rows
        other_first = [
            n for n in names
            if n.startswith("first_bag/row_") and n != "first_bag/row_us_puerto_rico"
        ]
        assert other_first == []

    def test_india_route_fee_row(self, fine_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_india_china" in names

    def test_multiple_bags_multiple_tables(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_us_puerto_rico" in names
        assert "second_bag/row_us_canada_puerto" in names
        assert "third_bag/row_us_canada_puerto" in names

    def test_four_bags(self, fine_segments):
        info = {
            "routine": "Europe",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/row_europe" in names
        assert "second_bag/row_europe_israel_qatar" in names
        assert "third_bag/row_europe_israel_qatar" in names
        assert "fourth_bag/row_europe_israel_qatar" in names

    def test_main_plus_gets_post_table(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Plus",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/post_table" in names

    def test_non_main_plus_no_post_table(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "first_bag/post_table" not in names


# ---------------------------------------------------------------------------
# Overweight / oversize
# ---------------------------------------------------------------------------


class TestOverweightOversize:
    def test_no_overweight_rules_for_light_bags(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        overweight = [n for n in names if "over_50" in n or "over_53" in n or "over_70" in n]
        assert overweight == []

    def test_overweight_50_53_bracket(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 52},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/over_50_lbs_23_kgs_to_53_lbs_24_kgs/row_regions" in names
        assert "weight_and_size/overweight_bags/more_than_one_fee" in names

    def test_overweight_53_70_bracket_india(self, fine_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china" in names

    def test_overweight_70_100_bracket(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 75},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_us_canada_puerto" in names
        # Should NOT have 53-70 bracket
        assert all("over_53" not in n for n in names)

    def test_no_oversize_for_small_bags(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [22, 18, 12], "weight": 40},  # total = 52
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        oversize = [n for n in names if "oversize" in n]
        assert oversize == []

    def test_oversize_us_route(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [30, 22, 14], "weight": 40},  # total = 66 > 62
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/oversize_bags/row_us_puerto_rico" in names

    def test_oversize_panama_route(self, fine_segments):
        info = {
            "routine": "Panama",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [30, 22, 14], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/oversize_bags/row_panama_south_america" in names


# ---------------------------------------------------------------------------
# Weight/size intro region selection
# ---------------------------------------------------------------------------


class TestWeightSizeIntro:
    def test_non_australia_gets_all_regions(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/intro/for_all_regions_except" in names
        assert "weight_and_size/intro/for_all_confirmed_customers" not in names

    def test_australia_gets_confirmed_customers(self, fine_segments):
        info = {
            "routine": "Australia",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/intro/for_all_confirmed_customers" in names
        assert "weight_and_size/intro/for_all_regions_except" not in names

    def test_new_zealand_gets_confirmed_customers(self, fine_segments):
        info = {
            "routine": "New Zealand",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/intro/for_all_confirmed_customers" in names


# ---------------------------------------------------------------------------
# Complimentary bag rules
# ---------------------------------------------------------------------------


class TestComplimentaryRules:
    def test_intro_always_present(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Basic Economy",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "complimentary_bags/in_some_cases_you" in names

    def test_comp_destination_includes_1st_bag_rule(self, fine_segments):
        for dest in ["India", "Cuba", "Israel", "Panama"]:
            info = {
                "routine": dest,
                "direction": 0,
                "customer_class": "Main Cabin",
                "bag_list": [
                    {"size": [20, 14, 8], "weight": 10},
                    {"size": [28, 18, 12], "weight": 40},
                ],
            }
            names = _names(get_applied_rules(info, fine_segments))
            assert "complimentary_bags/1st_checked_bag_is" in names, (
                f"Missing 1st_checked_bag_is for {dest}"
            )
            assert "complimentary_bags/or_when_traveling_to" in names

    def test_business_class_includes_1st_2nd(self, fine_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "complimentary_bags/1st_and_2nd_checked" in names

    def test_basic_economy_us_no_1st_2nd(self, fine_segments):
        """Basic Economy to U.S. should not include the 1st+2nd complimentary rule."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Basic Economy",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "complimentary_bags/1st_and_2nd_checked" not in names


# ---------------------------------------------------------------------------
# Spot checks from plan
# ---------------------------------------------------------------------------


class TestSpotChecks:
    def test_simple_us_domestic_count(self, fine_segments):
        """Simple U.S. domestic with 1 checked bag should have ~15 rules."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        result = get_applied_rules(info, fine_segments)
        names = _names(result)
        # Verify key sections are present
        assert any(n.startswith("carry_on/") for n in names)
        assert any(n.startswith("checked_bags_intro/") for n in names)
        assert "first_bag/row_us_puerto_rico" in names
        assert "weight_and_size/intro/we_calculate_the_size" in names
        # Reasonable count: carry-on(6) + intro(4) + preamble(1) +
        # complimentary(5-7) + first_bag(1) + weight_size_intro(2)
        assert 15 <= len(result) <= 25

    def test_india_overweight_75lb(self, fine_segments):
        """India with a 75 lb bag should include the 70-100 bracket, not 53-70."""
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 75},
            ],
        }
        names = _names(get_applied_rules(info, fine_segments))
        assert "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china" in names
        # Should NOT include 53-70 bracket
        ow_53 = [n for n in names if "over_53" in n]
        assert ow_53 == []


# ---------------------------------------------------------------------------
# Fine-to-coarse mapping
# ---------------------------------------------------------------------------


class TestGetCoarseForFine:
    def test_first_bag_row_maps_to_first_bag(self, fine_segments, coarse_segments):
        seg = next(s for s in fine_segments if s["name"] == "first_bag/row_india_china")
        coarse = get_coarse_for_fine(seg, coarse_segments)
        assert coarse["name"] == "first_bag"

    def test_weight_size_deep_nested_maps_correctly(self, fine_segments, coarse_segments):
        seg = next(
            s for s in fine_segments
            if s["name"] == "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china"
        )
        coarse = get_coarse_for_fine(seg, coarse_segments)
        assert coarse["name"] == "weight_and_size"

    def test_preamble_maps_to_preamble(self, fine_segments, coarse_segments):
        seg = next(s for s in fine_segments if s["name"] == "preamble/all_published_bag_fees")
        coarse = get_coarse_for_fine(seg, coarse_segments)
        assert coarse["name"] == "preamble"

    def test_every_fine_has_a_coarse_parent(self, fine_segments, coarse_segments):
        for seg in fine_segments:
            coarse = get_coarse_for_fine(seg, coarse_segments)
            assert coarse is not None
            assert coarse["char_start"] <= seg["char_start"]
            assert seg["char_end"] <= coarse["char_end"]

    def test_raises_for_invalid_segment(self, coarse_segments):
        fake = {"name": "fake", "char_start": 999999, "char_end": 999999 + 10}
        with pytest.raises(ValueError, match="No coarse segment"):
            get_coarse_for_fine(fake, coarse_segments)


# ---------------------------------------------------------------------------
# Combined API
# ---------------------------------------------------------------------------


class TestGetAppliedRulesWithCoarse:
    def test_returns_expected_keys(self, fine_segments, coarse_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        assert set(result.keys()) == {"fine", "coarse", "mapping"}

    def test_fine_matches_get_applied_rules(self, fine_segments, coarse_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
            ],
        }
        direct = get_applied_rules(info, fine_segments)
        combined = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        assert _names(combined["fine"]) == _names(direct)

    def test_coarse_is_deduplicated(self, fine_segments, coarse_segments):
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        coarse_names = [s["name"] for s in result["coarse"]]
        assert len(coarse_names) == len(set(coarse_names))

    def test_coarse_preserves_order(self, fine_segments, coarse_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
                {"size": [30, 20, 14], "weight": 45},
            ],
        }
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        coarse_order = {seg["name"]: i for i, seg in enumerate(coarse_segments)}
        indices = [coarse_order[s["name"]] for s in result["coarse"]]
        assert indices == sorted(indices)

    def test_mapping_has_entry_per_fine(self, fine_segments, coarse_segments):
        info = {
            "routine": "Europe",
            "direction": 1,
            "customer_class": "First",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [30, 22, 14], "weight": 75},
            ],
        }
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        fine_names = set(_names(result["fine"]))
        assert set(result["mapping"].keys()) == fine_names

    def test_weight_size_fine_maps_to_weight_and_size_coarse(self, fine_segments, coarse_segments):
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
            ],
        }
        result = get_applied_rules_with_coarse(info, fine_segments, coarse_segments)
        ow_fine = "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china"
        assert result["mapping"][ow_fine] == "weight_and_size"


# ---------------------------------------------------------------------------
# build_filtered_rulebook tests
# ---------------------------------------------------------------------------


class TestBuildFilteredRulebook:
    def test_filtered_is_proper_subset(self, rulebook_text, fine_segments, coarse_segments):
        """Filtered text must be shorter than the full rulebook."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        assert 0 < len(filtered) < len(rulebook_text)

    def test_contains_all_applicable_fine_segments(self, rulebook_text, fine_segments, coarse_segments):
        """Every applicable fine segment's substring must appear in the filtered text."""
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        applicable = get_applied_rules(info, fine_segments)
        for seg in applicable:
            assert seg["substring"] in filtered, (
                f"Missing applicable segment: {seg['name']}"
            )

    def test_excludes_non_applicable_fee_rows(self, rulebook_text, fine_segments, coarse_segments):
        """Non-applicable fee table rows must NOT appear in the filtered text."""
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        # U.S. domestic first-bag row should NOT be present
        us_row = next(s for s in fine_segments if s["name"] == "first_bag/row_us_puerto_rico")
        assert us_row["substring"] not in filtered
        # Canada first-bag row should NOT be present
        ca_row = next(s for s in fine_segments if s["name"] == "first_bag/row_canada")
        assert ca_row["substring"] not in filtered

    def test_table_column_headers_preserved(self, rulebook_text, fine_segments, coarse_segments):
        """Table column headers must be preserved when a row from that table is applicable."""
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        # The first_bag table header row should be present
        assert "| Regions" in filtered
        # The separator row should be present
        assert "|---" in filtered

    def test_section_headers_preserved(self, rulebook_text, fine_segments, coarse_segments):
        """Markdown section headers should be present for applicable sections."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        assert "# Bag fees" in filtered
        assert "## Carry-on bags" in filtered
        assert "## Checked bags" in filtered
        assert "### First Bag" in filtered
        assert "### Complimentary Bags" in filtered
        assert "### Weight and Size" in filtered

    def test_skipped_sections_not_present(self, rulebook_text, fine_segments, coarse_segments):
        """Sections for bags the passenger doesn't have should be absent."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},  # only 1 checked bag
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        # With only 1 checked bag, second/third/fourth bag sections should be absent
        assert "### Second Bag" not in filtered
        assert "### Third Bag" not in filtered
        assert "### Fourth Bag" not in filtered

    def test_overweight_section_excluded_for_light_bags(self, rulebook_text, fine_segments, coarse_segments):
        """Overweight bracket sub-headers should be absent when all bags are under 50 lbs."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Main Cabin",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        assert "Over 50 lbs" not in filtered
        assert "Over 53 lbs" not in filtered
        assert "Over 70 lbs" not in filtered

    def test_all_characters_from_original(self, rulebook_text, fine_segments, coarse_segments):
        """Every character in the filtered text must come from the original rulebook."""
        info = {
            "routine": "India",
            "direction": 0,
            "customer_class": "Business",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 60},
                {"size": [30, 20, 14], "weight": 45},
            ],
        }
        filtered = build_filtered_rulebook(info, rulebook_text, fine_segments, coarse_segments)
        # Filtered text should be constructable from non-overlapping slices of the original
        remaining = rulebook_text
        for chunk in filtered.split("\n"):
            if chunk:
                assert chunk in remaining or chunk in rulebook_text

    def test_drop_fee_summaries_excludes_summary_lines(self, fine_segments):
        """drop_fee_summaries=True should exclude generic fee summary sentences."""
        info = {
            "routine": "U.S.",
            "direction": 0,
            "customer_class": "Premium Economy",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        names_with = {s["name"] for s in get_applied_rules(info, fine_segments)}
        names_without = {s["name"] for s in get_applied_rules(
            info, fine_segments, drop_fee_summaries=True
        )}

        assert "checked_bags_intro/travel_within_between_the" in names_with
        assert "checked_bags_intro/travel_to_from_canada" in names_with
        assert "checked_bags_intro/travel_within_between_the" not in names_without
        assert "checked_bags_intro/travel_to_from_canada" not in names_without
        # Non-summary intro segments should still be present
        assert "checked_bags_intro/bag_fees_have_been" in names_without
        assert "checked_bags_intro/all_bag_fees_are" in names_without

    def test_drop_fee_summaries_in_filtered_rulebook(
        self, rulebook_text, fine_segments, coarse_segments
    ):
        """drop_fee_summaries should remove summary text from filtered rulebook."""
        info = {
            "routine": "Europe",
            "direction": 1,
            "customer_class": "First",
            "bag_list": [
                {"size": [20, 14, 8], "weight": 10},
                {"size": [28, 18, 12], "weight": 40},
            ],
        }
        filtered_with = build_filtered_rulebook(
            info, rulebook_text, fine_segments, coarse_segments,
        )
        filtered_without = build_filtered_rulebook(
            info, rulebook_text, fine_segments, coarse_segments,
            drop_fee_summaries=True,
        )
        assert "1st checked bag fee is $40" in filtered_with
        assert "1st checked bag fee is $40" not in filtered_without
        assert len(filtered_without) < len(filtered_with)
        # Structural elements should still be present
        assert "## Checked bags" in filtered_without
