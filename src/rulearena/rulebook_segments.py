"""Airline rulebook segmentation for RuleArena analysis.

Provides coarse (section-level) and fine (row-level) segmentation of the
airline ``reference_rules.txt`` rulebook.  Each segment is a dict with
keys ``name``, ``substring``, ``char_start``, ``char_end`` — designed to be
passed directly to ``find_substring_token_indices`` from ``src.token_utils``.
"""

import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _header_to_name(title: str) -> str:
    """Convert a markdown header title to a snake_case segment name."""
    name = title.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def _find_line_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character ranges for each line including its newline."""
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        nl = text.find("\n", start)
        if nl == -1:
            ranges.append((start, len(text)))
            break
        ranges.append((start, nl + 1))
        start = nl + 1
    return ranges


def _has_table(text: str) -> bool:
    """Return True if *text* contains a markdown table."""
    return bool(re.search(r"^\|.+\|$", text, re.MULTILINE))


def _extract_first_column(row_line: str) -> str:
    """Extract the text of the first data column from a markdown table row."""
    parts = row_line.split("|")
    # parts[0] is empty (before first |), parts[1] is first column
    return parts[1].strip() if len(parts) > 1 else ""


def _region_to_slug(region: str) -> str:
    """Derive a short slug from a region description in a table row."""
    text = region.strip().lower()
    text = text.replace("u.s.", "us").replace("u.s", "us")
    text = re.sub(r"\s*\(.*?\)", "", text)  # remove parentheticals

    # "To or from X"
    m = re.match(r"to or from\s+(.+)", text)
    if m:
        words = re.findall(r"[a-z]+", m.group(1))[:2]
        return "_".join(words)

    # "To / From X"  (overweight / oversize tables)
    m = re.match(r"to / from\s+(.+)", text)
    if m:
        words = re.findall(r"[a-z]+", m.group(1))[:2]
        return "_".join(words)

    # "From X to Y" — pick the shorter side
    m = re.match(r"from\s+(.+?)\s+to\s+(.+)", text)
    if m:
        src = re.findall(r"[a-z]+", m.group(1))
        dst = re.findall(r"[a-z]+", m.group(2))
        if len(src) <= len(dst):
            return "from_" + "_".join(src[:2])
        return "to_" + "_".join(dst[:2])

    # Common prefixes
    for prefix in [
        "within and between ",
        "between ",
        "to / from ",
        "all ",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    # Remove "and us ..." / "and the us ..." tails
    text = re.sub(r"\s+and\s+(?:the\s+)?us\b.*$", "", text)
    words = re.findall(r"[a-z]+", text)[:3]
    return "_".join(words) if words else "unknown"


def _prose_rule_slug(text: str) -> str:
    """Derive a short slug from the first few words of a prose rule block."""
    first_line = text.strip().split("\n")[0]
    first_line = first_line.lstrip("* ").lstrip("- ")
    words = re.findall(r"[a-z0-9]+", first_line.lower())[:4]
    return "_".join(words) if words else "unknown"


def _merge_sections(
    sections: list[dict],
    start_idx: int,
    end_idx: int,
    name: str,
    full_text: str,
) -> dict:
    """Merge a range of parsed sections [start_idx, end_idx) into one segment."""
    char_start = sections[start_idx]["char_start"]
    char_end = sections[end_idx - 1]["char_end"]
    return {
        "name": name,
        "substring": full_text[char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
    }


# ---------------------------------------------------------------------------
# Table splitting (used by fine segmentation)
# ---------------------------------------------------------------------------

def parse_table_rows(
    section_name: str,
    table_text: str,
    base_offset: int,
) -> list[dict]:
    """Split a markdown table into header + individual data rows.

    Parameters
    ----------
    section_name : str
        Prefix for the segment names (e.g. ``"first_bag"``).
    table_text : str
        The raw text of the table (header row, separator, data rows).
    base_offset : int
        Character offset of *table_text* within the full rulebook.

    Returns
    -------
    list[dict]
        One dict per piece (table_header + one per data row), each with
        ``name``, ``substring``, ``char_start``, ``char_end``.
    """
    line_ranges = _find_line_ranges(table_text)
    lines = [table_text[s:e] for s, e in line_ranges]

    result: list[dict] = []
    data_row_idx = 0
    seen_slugs: set[str] = set()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        if line.startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
            # Header row + separator → table_header segment
            hdr_start = line_ranges[i][0]
            hdr_end = line_ranges[i + 1][1]
            result.append({
                "name": f"{section_name}/table_header",
                "substring": table_text[hdr_start:hdr_end],
                "char_start": base_offset + hdr_start,
                "char_end": base_offset + hdr_end,
            })
            i += 2
            continue

        if line.startswith("|"):
            # Data row
            region = _extract_first_column(line)
            slug = _region_to_slug(region)
            if slug in seen_slugs:
                slug = f"{slug}_{data_row_idx}"
            seen_slugs.add(slug)
            row_start, row_end = line_ranges[i]
            result.append({
                "name": f"{section_name}/row_{slug}",
                "substring": table_text[row_start:row_end],
                "char_start": base_offset + row_start,
                "char_end": base_offset + row_end,
            })
            data_row_idx += 1
            i += 1
            continue

        i += 1

    return result


def _split_around_table(
    section_name: str,
    text: str,
    base_offset: int,
) -> list[dict]:
    """Split a section's text into pre-table, table rows, and post-table segments."""
    line_ranges = _find_line_ranges(text)
    lines = [text[s:e].rstrip("\n") for s, e in line_ranges]

    # Find table boundaries (first and last lines starting with |)
    table_start: int | None = None
    table_end: int | None = None  # exclusive
    for idx, line in enumerate(lines):
        if line.startswith("|"):
            if table_start is None:
                table_start = idx
            table_end = idx + 1

    if table_start is None:
        # No table found — return the whole section as-is
        return [{
            "name": section_name,
            "substring": text,
            "char_start": base_offset,
            "char_end": base_offset + len(text),
        }]

    result: list[dict] = []

    # Pre-table text
    pre_end = line_ranges[table_start][0]
    if pre_end > 0:
        result.append({
            "name": f"{section_name}/pre_table",
            "substring": text[:pre_end],
            "char_start": base_offset,
            "char_end": base_offset + pre_end,
        })

    # Table itself → split into rows via parse_table_rows
    table_char_start = line_ranges[table_start][0]
    table_char_end = line_ranges[table_end - 1][1]
    table_text = text[table_char_start:table_char_end]
    result.extend(
        parse_table_rows(section_name, table_text, base_offset + table_char_start)
    )

    # Post-table text
    if table_char_end < len(text):
        result.append({
            "name": f"{section_name}/post_table",
            "substring": text[table_char_end:],
            "char_start": base_offset + table_char_end,
            "char_end": base_offset + len(text),
        })

    return result


# ---------------------------------------------------------------------------
# Prose splitting (used by fine segmentation)
# ---------------------------------------------------------------------------

def _split_prose_into_rules(
    section_name: str,
    text: str,
    base_offset: int,
) -> list[dict]:
    """Split prose text into individual rule blocks.

    Headers (lines starting with ``#``) are skipped.  Blocks are separated
    by blank lines.  A paragraph followed immediately by a bullet list
    (no intervening blank line) is kept as a single rule block.

    Returns a list of segment dicts with ``name``, ``substring``,
    ``char_start``, ``char_end``.
    """
    line_ranges = _find_line_ranges(text)

    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end: int | None = None

    for start, end in line_ranges:
        line_text = text[start:end].rstrip("\n")

        # Skip markdown headers
        if line_text.lstrip().startswith("#"):
            if block_start is not None:
                blocks.append((block_start, block_end))
                block_start = None
            continue

        # Blank line → end current block
        if not line_text.strip():
            if block_start is not None:
                blocks.append((block_start, block_end))
                block_start = None
            continue

        # Content line
        if block_start is None:
            block_start = start
        block_end = end

    # Last block
    if block_start is not None:
        blocks.append((block_start, block_end))

    # Convert to segment dicts
    result: list[dict] = []
    seen_slugs: set[str] = set()
    for i, (start, end) in enumerate(blocks):
        substring = text[start:end]
        slug = _prose_rule_slug(substring)
        if slug in seen_slugs:
            slug = f"{slug}_{i}"
        seen_slugs.add(slug)
        result.append({
            "name": f"{section_name}/{slug}",
            "substring": substring,
            "char_start": base_offset + start,
            "char_end": base_offset + end,
        })

    return result


def _extract_table_rules(
    section_name: str,
    text: str,
    base_offset: int,
) -> list[dict]:
    """Extract only data rows and meaningful post-table text from a section.

    Unlike :func:`_split_around_table`, this drops structural elements
    (section titles, table column headers, separator lines) and returns
    only actual rule content.
    """
    line_ranges = _find_line_ranges(text)
    lines = [text[s:e].rstrip("\n") for s, e in line_ranges]

    # Locate table boundaries and separator
    table_start: int | None = None
    table_end: int | None = None  # exclusive
    separator_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.startswith("|"):
            if table_start is None:
                table_start = idx
            if "---" in line and separator_idx is None:
                separator_idx = idx
            table_end = idx + 1

    if table_start is None:
        return []

    result: list[dict] = []
    seen_slugs: set[str] = set()
    data_row_counter = 0

    # Data rows start after separator
    data_start = (separator_idx + 1) if separator_idx is not None else table_start + 2
    for idx in range(data_start, table_end):
        line = lines[idx]
        if line.startswith("|"):
            region = _extract_first_column(line)
            slug = _region_to_slug(region)
            if slug in seen_slugs:
                slug = f"{slug}_{data_row_counter}"
            seen_slugs.add(slug)
            row_start, row_end = line_ranges[idx]
            result.append({
                "name": f"{section_name}/row_{slug}",
                "substring": text[row_start:row_end],
                "char_start": base_offset + row_start,
                "char_end": base_offset + row_end,
            })
            data_row_counter += 1

    # Post-table text (footnotes) — only if non-whitespace content
    if table_end < len(lines):
        post_start = line_ranges[table_end][0]
        post_text = text[post_start:]
        if post_text.strip():
            result.append({
                "name": f"{section_name}/post_table",
                "substring": post_text,
                "char_start": base_offset + post_start,
                "char_end": base_offset + len(text),
            })

    return result


# ---------------------------------------------------------------------------
# Explode a coarse segment into fine segments
# ---------------------------------------------------------------------------

def _explode_segment(seg: dict) -> list[dict]:
    """Break a table-containing segment into fine rule segments.

    If the segment has sub-headers (e.g. ``weight_and_size``), split at those
    first, then extract table rules or prose rules from each sub-section.
    Structural elements (headers, table column headers) are dropped.
    """
    text = seg["substring"]
    base = seg["char_start"]
    name = seg["name"]

    header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headers = list(header_re.finditer(text))

    # Drop the segment's own header (first match at position 0)
    if headers and headers[0].start() == 0:
        headers = headers[1:]

    if not headers:
        # No sub-headers → just extract table rules
        return _extract_table_rules(name, text, base)

    # --- Has sub-headers: split into sub-sections first ---
    sub_sections: list[dict] = []

    # Text before first sub-header (intro)
    if headers[0].start() > 0:
        intro_end = headers[0].start()
        sub_sections.append({
            "name": f"{name}/intro",
            "substring": text[:intro_end],
            "char_start": base,
            "char_end": base + intro_end,
        })

    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sub_name = _header_to_name(h.group(2).strip())
        sub_sections.append({
            "name": f"{name}/{sub_name}",
            "substring": text[start:end],
            "char_start": base + start,
            "char_end": base + end,
        })

    # Extract rules from each sub-section
    result: list[dict] = []
    for sub in sub_sections:
        if _has_table(sub["substring"]):
            result.extend(
                _extract_table_rules(sub["name"], sub["substring"], sub["char_start"])
            )
        else:
            result.extend(
                _split_prose_into_rules(sub["name"], sub["substring"], sub["char_start"])
            )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_rulebook_sections(rulebook_text: str) -> list[dict]:
    """Split rulebook into sections at every markdown header boundary.

    Each section runs from its header to the start of the next header (or end
    of text).  Returns a list of dicts with:

    * ``title`` — header text (e.g. ``"Carry-on bags"``)
    * ``level`` — heading depth (``1`` for ``#``, ``2`` for ``##``, etc.)
    * ``substring`` — full text of the section
    * ``char_start``, ``char_end`` — character offsets in *rulebook_text*
    """
    header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headers = list(header_re.finditer(rulebook_text))

    if not headers:
        return [{
            "title": "(no headers)",
            "level": 0,
            "substring": rulebook_text,
            "char_start": 0,
            "char_end": len(rulebook_text),
        }]

    sections: list[dict] = []

    # Text before the very first header (if any)
    if headers[0].start() > 0:
        sections.append({
            "title": "(preamble)",
            "level": 0,
            "substring": rulebook_text[: headers[0].start()],
            "char_start": 0,
            "char_end": headers[0].start(),
        })

    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(rulebook_text)
        sections.append({
            "title": match.group(2).strip(),
            "level": len(match.group(1)),
            "substring": rulebook_text[start:end],
            "char_start": start,
            "char_end": end,
        })

    return sections


def get_coarse_segments(rulebook_text: str) -> list[dict]:
    """Return coarse (section-level) segmentation of the airline rulebook.

    Returns a list of dicts with ``name``, ``substring``, ``char_start``,
    ``char_end``.  Segments are contiguous and their concatenation equals the
    full *rulebook_text*.

    Coarse segments:

    ========== ===================================
    preamble   ``# Bag fees`` + intro text
    carry_on   ``## Carry-on bags`` (full section)
    checked_bags_intro  ``## Checked bags`` intro
    first_bag  ``### First Bag`` + table + footnote
    second_bag ``### Second Bag`` + table + footnote
    third_bag  ``### Third Bag`` + table
    fourth_bag ``### Fourth Bag +`` + table
    complimentary_bags  ``### Complimentary Bags``
    weight_and_size     ``### Weight and Size``
    ========== ===================================
    """
    sections = parse_rulebook_sections(rulebook_text)
    coarse: list[dict] = []
    i = 0

    while i < len(sections):
        s = sections[i]

        if s["level"] <= 1:
            # # Bag fees (level 1) or any pre-header text (level 0) → preamble
            coarse.append({
                "name": "preamble",
                "substring": s["substring"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
            })
            i += 1

        elif s["level"] == 2 and "Carry-on" in s["title"]:
            # ## Carry-on bags → merge all subsections until next level-2+
            start_idx = i
            j = i + 1
            while j < len(sections) and sections[j]["level"] > 2:
                j += 1
            coarse.append(
                _merge_sections(sections, start_idx, j, "carry_on", rulebook_text)
            )
            i = j

        elif s["level"] == 2 and "Checked" in s["title"]:
            # ## Checked bags → intro only (until first ### subsection)
            coarse.append({
                "name": "checked_bags_intro",
                "substring": s["substring"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
            })
            i += 1

        elif s["level"] == 3:
            # ### subsection → merge with its #### / ##### children
            name = _header_to_name(s["title"])
            start_idx = i
            j = i + 1
            while j < len(sections) and sections[j]["level"] > 3:
                j += 1
            coarse.append(
                _merge_sections(sections, start_idx, j, name, rulebook_text)
            )
            i = j

        else:
            # Fallback for unexpected header levels
            coarse.append({
                "name": _header_to_name(s["title"]),
                "substring": s["substring"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
            })
            i += 1

    return coarse


def get_fine_segments(rulebook_text: str) -> list[dict]:
    """Return fine (rule-level) segmentation of the airline rulebook.

    Each segment represents an individual rule:

    * Table sections → one segment per data row, plus footnotes
    * Prose sections → one segment per paragraph / bullet-list block
    * Structural elements (markdown headers, table column headers) are
      **excluded** — they are not rules.

    Unlike coarse segments, fine segments are **not** contiguous and do not
    cover the full rulebook text.  Use coarse segments for full coverage.
    """
    coarse = get_coarse_segments(rulebook_text)
    fine: list[dict] = []

    for seg in coarse:
        if _has_table(seg["substring"]):
            fine.extend(_explode_segment(seg))
        else:
            fine.extend(
                _split_prose_into_rules(
                    seg["name"], seg["substring"], seg["char_start"]
                )
            )

    return fine
