"""Map RuleArena airline problems to their applicable fine-grained rule segments.

Given a problem's ``info`` dict (routine, direction, customer_class, bag_list)
and the list of fine segments from :func:`get_fine_segments`, returns the ordered
subset of segments that are needed to solve that problem.

This enables attention analysis: comparing model attention on *relevant* rules
versus irrelevant ones during generation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Route constants
# ---------------------------------------------------------------------------

# Destinations where the 1st checked bag is complimentary (excluding Basic Economy).
# Mirrors ``complementary_first`` in ``datasets/RuleArena/airline/compute_answer.py``.
COMPLEMENTARY_FIRST_DESTINATIONS: set[str] = {
    "China", "Hong Kong", "Japan", "South Korea", "India",
    "Qatar", "Haiti", "Cuba", "Panama", "Colombia",
    "Ecuador", "Peru", "South America", "Israel",
}

# Customer classes that get 1st+2nd bags complimentary.
COMP_1ST_2ND_CLASSES: set[str] = {"Business", "First", "Premium Economy"}

# ---------------------------------------------------------------------------
# Per-table route → segment-name maps
# ---------------------------------------------------------------------------
# Each dict maps a ``routine`` value to the segment name suffix (after the
# table prefix).  Cuba is direction-dependent and handled via dedicated
# ``_CUBA_*`` entries or separate logic.

# ── first_bag ──────────────────────────────────────────────────────────────

FIRST_BAG_ROUTE: dict[str, str] = {
    "U.S.": "first_bag/row_us_puerto_rico",
    "Puerto Rico": "first_bag/row_us_puerto_rico",
    "Canada": "first_bag/row_canada",
    "Mexico": "first_bag/row_mexico_caribbean_central",
    "Haiti": "first_bag/row_haiti",
    "Panama": "first_bag/row_panama_colombia_ecuador",
    "Colombia": "first_bag/row_panama_colombia_ecuador",
    "Ecuador": "first_bag/row_panama_colombia_ecuador",
    "Peru": "first_bag/row_panama_colombia_ecuador",
    "South America": "first_bag/row_south_america",
    "Israel": "first_bag/row_israel",
    "Qatar": "first_bag/row_qatar",
    "Europe": "first_bag/row_europe",
    "India": "first_bag/row_india_china",
    "China": "first_bag/row_india_china",
    "Japan": "first_bag/row_india_china",
    "South Korea": "first_bag/row_india_china",
    "Hong Kong": "first_bag/row_india_china",
    "Australia": "first_bag/row_india_china",
    "New Zealand": "first_bag/row_india_china",
}
FIRST_BAG_CUBA = {0: "first_bag/row_to_cuba", 1: "first_bag/row_from_cuba"}

# ── second_bag ─────────────────────────────────────────────────────────────

SECOND_BAG_ROUTE: dict[str, str] = {
    "U.S.": "second_bag/row_us_canada_puerto",
    "Puerto Rico": "second_bag/row_us_canada_puerto",
    "Canada": "second_bag/row_us_canada_puerto",
    "Mexico": "second_bag/row_mexico_caribbean_or",
    "Haiti": "second_bag/row_haiti",
    "Panama": "second_bag/row_panama",
    "Colombia": "second_bag/row_south_america",
    "Ecuador": "second_bag/row_south_america",
    "Peru": "second_bag/row_south_america",
    "South America": "second_bag/row_south_america",
    "Israel": "second_bag/row_europe_israel_qatar",
    "Qatar": "second_bag/row_europe_israel_qatar",
    "Europe": "second_bag/row_europe_israel_qatar",
    "India": "second_bag/row_india_china",
    "China": "second_bag/row_india_china",
    "Japan": "second_bag/row_india_china",
    "South Korea": "second_bag/row_india_china",
    "Hong Kong": "second_bag/row_india_china",
    "Australia": "second_bag/row_india_china",
    "New Zealand": "second_bag/row_india_china",
}
SECOND_BAG_CUBA = {0: "second_bag/row_to_cuba", 1: "second_bag/row_from_cuba"}

# ── third_bag ──────────────────────────────────────────────────────────────

THIRD_BAG_ROUTE: dict[str, str] = {
    "U.S.": "third_bag/row_us_canada_puerto",
    "Puerto Rico": "third_bag/row_us_canada_puerto",
    "Canada": "third_bag/row_us_canada_puerto",
    "Mexico": "third_bag/row_mexico_caribbean_central",
    "Cuba": "third_bag/row_mexico_caribbean_central",
    "Haiti": "third_bag/row_mexico_caribbean_central",
    "Panama": "third_bag/row_mexico_caribbean_central",
    "Colombia": "third_bag/row_mexico_caribbean_central",
    "Ecuador": "third_bag/row_mexico_caribbean_central",
    "Peru": "third_bag/row_mexico_caribbean_central",
    "South America": "third_bag/row_mexico_caribbean_central",
    "Israel": "third_bag/row_europe_israel_qatar",
    "Qatar": "third_bag/row_europe_israel_qatar",
    "Europe": "third_bag/row_europe_israel_qatar",
    "India": "third_bag/row_india_china",
    "China": "third_bag/row_india_china",
    "Japan": "third_bag/row_india_china",
    "South Korea": "third_bag/row_india_china",
    "Hong Kong": "third_bag/row_india_china",
    "Australia": "third_bag/row_india_china",
    "New Zealand": "third_bag/row_india_china",
}

# ── fourth_bag ─────────────────────────────────────────────────────────────

FOURTH_BAG_ROUTE: dict[str, str] = {
    "U.S.": "fourth_bag/row_us_canada_puerto",
    "Puerto Rico": "fourth_bag/row_us_canada_puerto",
    "Canada": "fourth_bag/row_us_canada_puerto",
    "Mexico": "fourth_bag/row_mexico_caribbean_or",
    "Cuba": "fourth_bag/row_mexico_caribbean_or",
    "Haiti": "fourth_bag/row_mexico_caribbean_or",
    "Panama": "fourth_bag/row_mexico_caribbean_or",
    "Colombia": "fourth_bag/row_mexico_caribbean_or",
    "Ecuador": "fourth_bag/row_mexico_caribbean_or",
    "Peru": "fourth_bag/row_mexico_caribbean_or",
    "South America": "fourth_bag/row_mexico_caribbean_or",
    "Israel": "fourth_bag/row_europe_israel_qatar",
    "Qatar": "fourth_bag/row_europe_israel_qatar",
    "Europe": "fourth_bag/row_europe_israel_qatar",
    "India": "fourth_bag/row_india_china",
    "China": "fourth_bag/row_india_china",
    "Japan": "fourth_bag/row_india_china",
    "South Korea": "fourth_bag/row_india_china",
    "Hong Kong": "fourth_bag/row_india_china",
    "Australia": "fourth_bag/row_india_china",
    "New Zealand": "fourth_bag/row_india_china",
}

# ── overweight: 53–70 lbs ─────────────────────────────────────────────────

OW_53_70_ROUTE: dict[str, str] = {
    "U.S.": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_us_canada_puerto",
    "Puerto Rico": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_us_canada_puerto",
    "Canada": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_us_canada_puerto",
    "Cuba": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_cuba",
    "Mexico": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Haiti": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Panama": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Colombia": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Ecuador": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Peru": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "South America": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_mexico_caribbean_or",
    "Europe": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_europe_israel_qatar",
    "Israel": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_europe_israel_qatar",
    "Qatar": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_europe_israel_qatar",
    "Australia": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_australia_and",
    "New Zealand": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_australia_and",
    "India": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china",
    "China": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china",
    "Japan": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china",
    "South Korea": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china",
    "Hong Kong": "weight_and_size/over_53_lbs_24_kgs_to_70_lbs_32_kgs/row_india_china",
}

# ── overweight: 70–100 lbs ────────────────────────────────────────────────

OW_70_100_ROUTE: dict[str, str] = {
    "U.S.": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_us_canada_puerto",
    "Puerto Rico": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_us_canada_puerto",
    "Canada": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_us_canada_puerto",
    "Cuba": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Mexico": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Haiti": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Panama": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Colombia": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Ecuador": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Peru": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "South America": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_mexico_caribbean_central",
    "Europe": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_europe_israel_qatar",
    "Israel": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_europe_israel_qatar",
    "Qatar": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_europe_israel_qatar",
    "Australia": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_australia_and",
    "New Zealand": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_australia_and",
    "India": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china",
    "China": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china",
    "Japan": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china",
    "South Korea": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china",
    "Hong Kong": "weight_and_size/over_70_lbs_32kgs_to_100_lbs_45_kgs/row_india_china",
}

# ── oversize ───────────────────────────────────────────────────────────────

OVERSIZE_ROUTE: dict[str, str] = {
    "U.S.": "weight_and_size/oversize_bags/row_us_puerto_rico",
    "Puerto Rico": "weight_and_size/oversize_bags/row_us_puerto_rico",
    "Canada": "weight_and_size/oversize_bags/row_us_puerto_rico",
    "Mexico": "weight_and_size/oversize_bags/row_mexico_caribbean_central",
    "Cuba": "weight_and_size/oversize_bags/row_mexico_caribbean_central",
    "Haiti": "weight_and_size/oversize_bags/row_mexico_caribbean_central",
    "Panama": "weight_and_size/oversize_bags/row_panama_south_america",
    "Colombia": "weight_and_size/oversize_bags/row_panama_south_america",
    "Ecuador": "weight_and_size/oversize_bags/row_panama_south_america",
    "Peru": "weight_and_size/oversize_bags/row_panama_south_america",
    "South America": "weight_and_size/oversize_bags/row_panama_south_america",
    "Europe": "weight_and_size/oversize_bags/row_europe_israel_qatar",
    "Israel": "weight_and_size/oversize_bags/row_europe_israel_qatar",
    "Qatar": "weight_and_size/oversize_bags/row_europe_israel_qatar",
    "India": "weight_and_size/oversize_bags/row_china_japan_south",
    "China": "weight_and_size/oversize_bags/row_china_japan_south",
    "Japan": "weight_and_size/oversize_bags/row_china_japan_south",
    "South Korea": "weight_and_size/oversize_bags/row_china_japan_south",
    "Hong Kong": "weight_and_size/oversize_bags/row_china_japan_south",
    "Australia": "weight_and_size/oversize_bags/row_china_japan_south",
    "New Zealand": "weight_and_size/oversize_bags/row_china_japan_south",
}

# Fee table maps indexed by bag position (0-based for checked bags).
_BAG_TABLE_MAPS: list[tuple[dict[str, str], dict[int, str]]] = [
    (FIRST_BAG_ROUTE, FIRST_BAG_CUBA),
    (SECOND_BAG_ROUTE, SECOND_BAG_CUBA),
    (THIRD_BAG_ROUTE, {}),   # Cuba not direction-dependent for 3rd/4th
    (FOURTH_BAG_ROUTE, {}),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_name_index(fine_segments: list[dict]) -> dict[str, dict]:
    """Build a name → segment dict for O(1) lookups."""
    return {seg["name"]: seg for seg in fine_segments}


def _get_fee_row_name(
    table_idx: int,
    routine: str,
    direction: int,
) -> str | None:
    """Return the segment name for a fee-table row given bag position, routine, direction."""
    route_map, cuba_map = _BAG_TABLE_MAPS[table_idx]
    if routine == "Cuba" and cuba_map:
        return cuba_map.get(direction)
    return route_map.get(routine)


def _get_overweight_names(weight: float, routine: str) -> list[str]:
    """Return segment names for applicable overweight bracket rows."""
    if weight <= 50:
        return []
    names: list[str] = []
    if weight > 50 and weight <= 53:
        names.append(
            "weight_and_size/over_50_lbs_23_kgs_to_53_lbs_24_kgs/row_regions"
        )
    elif weight > 53 and weight <= 70:
        name = OW_53_70_ROUTE.get(routine)
        if name:
            names.append(name)
    elif weight > 70:
        name = OW_70_100_ROUTE.get(routine)
        if name:
            names.append(name)
    return names


def _get_oversize_name(total_size: float, routine: str) -> str | None:
    """Return the segment name for the applicable oversize row, or None."""
    if total_size <= 62:
        return None
    return OVERSIZE_ROUTE.get(routine)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Checked-bags intro segments that state class-agnostic fee summaries.
# These can mislead the model when the passenger's cabin class gets $0 per
# the detailed fee table (e.g. Premium Economy, Business, First).
_FEE_SUMMARY_SEGMENTS: set[str] = {
    "checked_bags_intro/travel_within_between_the",
    "checked_bags_intro/travel_to_from_canada",
}


def get_applied_rules(
    info: dict,
    fine_segments: list[dict],
    *,
    drop_fee_summaries: bool = False,
) -> list[dict]:
    """Return the ordered list of fine segments needed to solve a problem.

    Parameters
    ----------
    info : dict
        A RuleArena airline problem ``info`` dict with keys:
        ``routine``, ``direction``, ``customer_class``, ``bag_list``.
        ``bag_list[0]`` is the carry-on; ``bag_list[1:]`` are checked bags.
    fine_segments : list[dict]
        The full list of fine segments from :func:`get_fine_segments`.
    drop_fee_summaries : bool, optional
        If True, exclude the generic checked-bag fee summary sentences
        (e.g. "1st checked bag fee is $40") that can conflict with the
        cabin-specific fee tables.  Default False (include them).

    Returns
    -------
    list[dict]
        Ordered subset of *fine_segments* applicable to this problem.
        Order matches the original segment order (i.e. the rulebook order).
    """
    routine: str = info["routine"]
    direction: int = info["direction"]
    customer_class: str = info["customer_class"]
    bag_list: list[dict] = info["bag_list"]
    checked_bags = bag_list[1:]  # index 0 is carry-on

    idx = _build_name_index(fine_segments)
    needed_names: set[str] = set()

    # 1. Preamble — always
    needed_names.add("preamble/all_published_bag_fees")

    # 2. Carry-on rules — always
    for seg in fine_segments:
        if seg["name"].startswith("carry_on/"):
            needed_names.add(seg["name"])

    # 3. Checked bags intro — always (optionally drop fee summaries)
    for seg in fine_segments:
        if seg["name"].startswith("checked_bags_intro/"):
            if drop_fee_summaries and seg["name"] in _FEE_SUMMARY_SEGMENTS:
                continue
            needed_names.add(seg["name"])

    # 4. Complimentary bags — always include intro + general info;
    #    conditionally include tier-specific rules.
    needed_names.add("complimentary_bags/in_some_cases_you")
    needed_names.add("complimentary_bags/if_your_status_level")
    needed_names.add("complimentary_bags/free_checked_bags_may")

    # 1st bag complimentary rules: if destination qualifies or customer has status
    if (routine in COMPLEMENTARY_FIRST_DESTINATIONS
            or customer_class != "Basic Economy"):
        needed_names.add("complimentary_bags/1st_checked_bag_is")
        needed_names.add("complimentary_bags/or_when_traveling_to")

    # 1st+2nd complimentary: if customer class qualifies
    if customer_class in COMP_1ST_2ND_CLASSES:
        needed_names.add("complimentary_bags/1st_and_2nd_checked")

    # 1st+2nd+3rd complimentary: always include (Flagship / Executive Platinum /
    # military status — cannot be determined from info dict alone, but the rule
    # is relevant for understanding the fee structure).
    needed_names.add("complimentary_bags/1st_2nd_and_3rd")

    # 5. Per-bag fee table rows — one row per checked bag (up to 4 tables)
    for bag_idx in range(len(checked_bags)):
        table_idx = min(3, bag_idx)  # bags beyond 4th use 4th table
        row_name = _get_fee_row_name(table_idx, routine, direction)
        if row_name:
            needed_names.add(row_name)

    # Post-table footnotes — if customer_class is "Main Plus"
    if customer_class == "Main Plus":
        needed_names.add("first_bag/post_table")
        needed_names.add("second_bag/post_table")

    # 6. Weight & size intro — always
    needed_names.add("weight_and_size/intro/we_calculate_the_size")
    if routine in ("Australia", "New Zealand"):
        needed_names.add("weight_and_size/intro/for_all_confirmed_customers")
    else:
        needed_names.add("weight_and_size/intro/for_all_regions_except")

    # 7. Overweight rules — only if any checked bag > 50 lbs
    any_overweight = any(b["weight"] > 50 for b in checked_bags)
    if any_overweight:
        needed_names.add("weight_and_size/overweight_bags/more_than_one_fee")
        for bag in checked_bags:
            for name in _get_overweight_names(bag["weight"], routine):
                needed_names.add(name)

    # 8. Oversize rules — only if any checked bag total dimensions > 62
    any_oversize = any(sum(b["size"]) > 62 for b in checked_bags)
    if any_oversize:
        # Include the "more_than_one_fee" rule if not already included
        needed_names.add("weight_and_size/overweight_bags/more_than_one_fee")
        for bag in checked_bags:
            name = _get_oversize_name(sum(bag["size"]), routine)
            if name:
                needed_names.add(name)

    # Filter and preserve original segment order
    return [seg for seg in fine_segments if seg["name"] in needed_names]


def build_filtered_rulebook(
    info: dict,
    rulebook_text: str,
    fine_segments: list[dict],
    coarse_segments: list[dict],
    *,
    drop_fee_summaries: bool = False,
) -> str:
    """Build a filtered rulebook containing only rules applicable to a problem.

    Preserves document structure (section headers, table column headers) while
    stripping non-applicable fee table rows, weight brackets, and oversize rows.

    Parameters
    ----------
    info : dict
        A RuleArena airline problem ``info`` dict.
    rulebook_text : str
        Full text of ``reference_rules.txt``.
    fine_segments : list[dict]
        All fine segments from :func:`get_fine_segments`.
    coarse_segments : list[dict]
        All coarse segments from :func:`get_coarse_segments`.
    drop_fee_summaries : bool, optional
        Forwarded to :func:`get_applied_rules`.

    Returns
    -------
    str
        Filtered rulebook text — a proper subset of *rulebook_text* that
        retains structural elements (headers, table column headers) around
        applicable rules.
    """
    result = get_applied_rules_with_coarse(
        info, fine_segments, coarse_segments, drop_fee_summaries=drop_fee_summaries
    )
    applicable_fine_names: set[str] = {seg["name"] for seg in result["fine"]}
    applicable_coarse = result["coarse"]

    pieces: list[str] = []

    for coarse_seg in applicable_coarse:
        c_start = coarse_seg["char_start"]
        c_end = coarse_seg["char_end"]

        # Collect all fine segments within this coarse section
        fines_in_coarse = [
            f for f in fine_segments
            if c_start <= f["char_start"] and f["char_end"] <= c_end
        ]
        fines_in_coarse.sort(key=lambda f: f["char_start"])

        # Build interleaved list: (type, start, end, segment_or_None)
        entries: list[tuple[str, int, int, dict | None]] = []
        pos = c_start
        for fine in fines_in_coarse:
            if fine["char_start"] > pos:
                entries.append(("gap", pos, fine["char_start"], None))
            entries.append(("fine", fine["char_start"], fine["char_end"], fine))
            pos = fine["char_end"]
        if pos < c_end:
            entries.append(("gap", pos, c_end, None))

        # Walk entries: include applicable fines and their preceding structural gaps
        for i, (etype, start, end, seg) in enumerate(entries):
            if etype == "fine":
                if seg["name"] in applicable_fine_names:
                    pieces.append(rulebook_text[start:end])
            else:  # gap
                # Include this gap only if at least one applicable fine
                # follows before the next gap.
                include = False
                for j in range(i + 1, len(entries)):
                    if entries[j][0] == "gap":
                        break
                    if (entries[j][0] == "fine"
                            and entries[j][3]["name"] in applicable_fine_names):
                        include = True
                        break
                if include:
                    pieces.append(rulebook_text[start:end])

    return "".join(pieces)


def get_coarse_for_fine(fine_segment: dict, coarse_segments: list[dict]) -> dict:
    """Return the coarse segment that contains a given fine segment.

    Uses character-range containment: a fine segment's [char_start, char_end]
    must fall within exactly one coarse segment's range.

    Raises ``ValueError`` if no matching coarse segment is found.
    """
    f_start = fine_segment["char_start"]
    f_end = fine_segment["char_end"]
    for coarse in coarse_segments:
        if coarse["char_start"] <= f_start and f_end <= coarse["char_end"]:
            return coarse
    raise ValueError(
        f"No coarse segment contains fine segment {fine_segment['name']!r} "
        f"[{f_start}:{f_end}]"
    )


def get_applied_rules_with_coarse(
    info: dict,
    fine_segments: list[dict],
    coarse_segments: list[dict],
    *,
    drop_fee_summaries: bool = False,
) -> dict:
    """Return applicable rules at both fine and coarse granularity.

    Parameters
    ----------
    info : dict
        A RuleArena airline problem ``info`` dict.
    fine_segments : list[dict]
        All fine segments from :func:`get_fine_segments`.
    coarse_segments : list[dict]
        All coarse segments from :func:`get_coarse_segments`.
    drop_fee_summaries : bool, optional
        Forwarded to :func:`get_applied_rules`.

    Returns
    -------
    dict
        ``"fine"``  — list of applicable fine segment dicts (ordered).
        ``"coarse"`` — list of applicable coarse segment dicts (deduplicated,
        ordered by their position in *coarse_segments*).
        ``"mapping"`` — ``{fine_name: coarse_name, ...}`` for each applicable
        fine segment.
    """
    fine = get_applied_rules(info, fine_segments, drop_fee_summaries=drop_fee_summaries)

    mapping: dict[str, str] = {}
    coarse_names_seen: set[str] = set()
    coarse_ordered: list[dict] = []

    for f_seg in fine:
        c_seg = get_coarse_for_fine(f_seg, coarse_segments)
        mapping[f_seg["name"]] = c_seg["name"]
        if c_seg["name"] not in coarse_names_seen:
            coarse_names_seen.add(c_seg["name"])
            coarse_ordered.append(c_seg)

    # Re-sort coarse segments to match original coarse_segments order
    coarse_order = {seg["name"]: i for i, seg in enumerate(coarse_segments)}
    coarse_ordered.sort(key=lambda s: coarse_order[s["name"]])

    return {
        "fine": fine,
        "coarse": coarse_ordered,
        "mapping": mapping,
    }
