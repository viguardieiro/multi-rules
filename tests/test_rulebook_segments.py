"""Tests for src.rulearena.rulebook_segments."""

from pathlib import Path

import pytest

from src.rulearena.rulebook_segments import (
    get_coarse_segments,
    get_fine_segments,
    parse_rulebook_sections,
    parse_table_rows,
)

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


# --- parse_rulebook_sections ---


class TestParseRulebookSections:
    def test_returns_nonempty(self, rulebook_text):
        sections = parse_rulebook_sections(rulebook_text)
        assert len(sections) > 0

    def test_sections_are_contiguous(self, rulebook_text):
        sections = parse_rulebook_sections(rulebook_text)
        concatenated = "".join(s["substring"] for s in sections)
        assert concatenated == rulebook_text

    def test_char_offsets_match_substrings(self, rulebook_text):
        for s in parse_rulebook_sections(rulebook_text):
            assert rulebook_text[s["char_start"]:s["char_end"]] == s["substring"]

    def test_first_section_is_bag_fees(self, rulebook_text):
        sections = parse_rulebook_sections(rulebook_text)
        assert sections[0]["title"] == "Bag fees"
        assert sections[0]["level"] == 1

    def test_no_header_text(self):
        text = "Just some plain text without any headers."
        sections = parse_rulebook_sections(text)
        assert len(sections) == 1
        assert sections[0]["substring"] == text


# --- get_coarse_segments ---


class TestCoarseSegments:
    def test_segment_count(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        assert len(coarse) == 9

    def test_expected_names(self, rulebook_text):
        names = [s["name"] for s in get_coarse_segments(rulebook_text)]
        assert names == [
            "preamble",
            "carry_on",
            "checked_bags_intro",
            "first_bag",
            "second_bag",
            "third_bag",
            "fourth_bag",
            "complimentary_bags",
            "weight_and_size",
        ]

    def test_contiguity(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        concatenated = "".join(s["substring"] for s in coarse)
        assert concatenated == rulebook_text

    def test_no_gaps_no_overlaps(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        for i in range(len(coarse) - 1):
            assert coarse[i]["char_end"] == coarse[i + 1]["char_start"]
        assert coarse[0]["char_start"] == 0
        assert coarse[-1]["char_end"] == len(rulebook_text)

    def test_char_offsets_match_substrings(self, rulebook_text):
        for s in get_coarse_segments(rulebook_text):
            assert rulebook_text[s["char_start"]:s["char_end"]] == s["substring"]

    def test_preamble_starts_with_bag_fees(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        assert coarse[0]["substring"].startswith("# Bag fees")

    def test_carry_on_contains_personal_items(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        carry_on = coarse[1]
        assert "Personal items" in carry_on["substring"]
        assert "Carry-on items" in carry_on["substring"]

    def test_weight_and_size_contains_overweight(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        ws = [s for s in coarse if s["name"] == "weight_and_size"][0]
        assert "Overweight" in ws["substring"]
        assert "Oversize" in ws["substring"]


# --- get_fine_segments ---


class TestFineSegments:
    def test_more_segments_than_coarse(self, rulebook_text):
        coarse = get_coarse_segments(rulebook_text)
        fine = get_fine_segments(rulebook_text)
        assert len(fine) > len(coarse)

    def test_no_overlaps(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        for i in range(len(fine) - 1):
            assert fine[i]["char_end"] <= fine[i + 1]["char_start"], (
                f"Overlap between {fine[i]['name']} and {fine[i+1]['name']}: "
                f"{fine[i]['char_end']} > {fine[i+1]['char_start']}"
            )

    def test_segments_within_bounds(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        for s in fine:
            assert 0 <= s["char_start"] < s["char_end"] <= len(rulebook_text), (
                f"Segment '{s['name']}' out of bounds: "
                f"[{s['char_start']}, {s['char_end']}) vs text length {len(rulebook_text)}"
            )

    def test_segments_ordered(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        starts = [s["char_start"] for s in fine]
        assert starts == sorted(starts)

    def test_char_offsets_match_substrings(self, rulebook_text):
        for s in get_fine_segments(rulebook_text):
            assert rulebook_text[s["char_start"]:s["char_end"]] == s["substring"]

    def test_no_headers_in_segments(self, rulebook_text):
        """Fine segments should not contain markdown headers as standalone segments."""
        fine = get_fine_segments(rulebook_text)
        for s in fine:
            # No segment should start with a markdown header
            assert not s["substring"].lstrip().startswith("#"), (
                f"Segment '{s['name']}' starts with a header: "
                f"{s['substring'][:80]!r}"
            )

    def test_no_table_headers_in_segments(self, rulebook_text):
        """Fine segments should not be table column header rows."""
        fine = get_fine_segments(rulebook_text)
        names = [s["name"] for s in fine]
        table_header_names = [n for n in names if n.endswith("/table_header")]
        assert table_header_names == [], f"Found table headers: {table_header_names}"

    def test_no_pre_table_segments(self, rulebook_text):
        """Fine segments should not include pre_table structural segments."""
        fine = get_fine_segments(rulebook_text)
        names = [s["name"] for s in fine]
        pre_table_names = [n for n in names if n.endswith("/pre_table")]
        assert pre_table_names == [], f"Found pre_table segments: {pre_table_names}"

    def test_carry_on_split_into_rules(self, rulebook_text):
        """carry_on should be split into multiple prose rules, not kept as one block."""
        fine = get_fine_segments(rulebook_text)
        carry_on_rules = [s for s in fine if s["name"].startswith("carry_on/")]
        assert len(carry_on_rules) >= 4  # at least: allowance, personal items, exempt items, carry-on items, sizes

    def test_complimentary_bags_split_into_rules(self, rulebook_text):
        """complimentary_bags should be split into individual rule blocks."""
        fine = get_fine_segments(rulebook_text)
        comp_rules = [s for s in fine if s["name"].startswith("complimentary_bags/")]
        assert len(comp_rules) >= 5  # intro, status, codeshare, 1st free, destinations, 1st+2nd, 1st+2nd+3rd

    def test_first_bag_has_rows(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        first_bag_rows = [s for s in fine if s["name"].startswith("first_bag/row_")]
        assert len(first_bag_rows) == 13  # 13 data rows in First Bag table

    def test_first_bag_has_footnote(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        footnotes = [s for s in fine if s["name"] == "first_bag/post_table"]
        assert len(footnotes) == 1
        assert "Main Plus" in footnotes[0]["substring"]

    def test_oversize_rows_exist(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        oversize_rows = [
            s for s in fine if s["name"].startswith("weight_and_size/oversize_bags/row_")
        ]
        assert len(oversize_rows) == 5

    def test_weight_and_size_prose_split(self, rulebook_text):
        """weight_and_size intro and overweight intro should be split into prose rules."""
        fine = get_fine_segments(rulebook_text)
        ws_intro_rules = [s for s in fine if s["name"].startswith("weight_and_size/intro/")]
        assert len(ws_intro_rules) >= 2  # dimension intro, all-regions allowance, australia/NZ allowance

    def test_unique_names(self, rulebook_text):
        fine = get_fine_segments(rulebook_text)
        names = [s["name"] for s in fine]
        assert len(names) == len(set(names)), (
            f"Duplicate names: {[n for n in names if names.count(n) > 1]}"
        )


# --- parse_table_rows ---


class TestParseTableRows:
    def test_simple_table(self):
        table = (
            "| Region | Fee |\n"
            "|--------|-----|\n"
            "| U.S.   | $40 |\n"
            "| Canada | $35 |\n"
        )
        rows = parse_table_rows("test", table, base_offset=100)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[0]["name"] == "test/table_header"
        assert rows[1]["name"].startswith("test/row_")
        assert rows[2]["name"].startswith("test/row_")

    def test_offsets_are_contiguous(self):
        table = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
        )
        rows = parse_table_rows("sec", table, base_offset=0)
        concat = "".join(r["substring"] for r in rows)
        assert concat == table
