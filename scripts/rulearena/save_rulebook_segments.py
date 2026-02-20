#!/usr/bin/env python3
"""Save coarse and fine airline rulebook segmentations to JSON files.

Usage::

    python scripts/rulearena/save_rulebook_segments.py

Outputs:
    results/rulearena/rules/airline_rulebook_segments_coarse.json
    results/rulearena/rules/airline_rulebook_segments_fine.json
"""

import json
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rulearena.rulebook_segments import get_coarse_segments, get_fine_segments

RULEBOOK_PATH = Path(__file__).resolve().parents[2] / "datasets" / "RuleArena" / "airline" / "reference_rules.txt"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "rulearena" / "rules"


def main() -> None:
    rulebook_text = RULEBOOK_PATH.read_text()

    coarse = get_coarse_segments(rulebook_text)
    fine = get_fine_segments(rulebook_text)

    # Verify coarse contiguity: concatenation of substrings == full text
    coarse_concat = "".join(seg["substring"] for seg in coarse)
    assert coarse_concat == rulebook_text, "Coarse segments are not contiguous!"

    # Verify fine segments: no overlaps, ordered, within bounds
    for i in range(len(fine) - 1):
        assert fine[i]["char_end"] <= fine[i + 1]["char_start"], (
            f"Fine overlap: {fine[i]['name']} ends at {fine[i]['char_end']} "
            f"but {fine[i+1]['name']} starts at {fine[i+1]['char_start']}"
        )
    for seg in fine:
        assert rulebook_text[seg["char_start"]:seg["char_end"]] == seg["substring"]

    # Coverage stats
    fine_chars = sum(seg["char_end"] - seg["char_start"] for seg in fine)
    coverage = fine_chars / len(rulebook_text) * 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coarse_path = OUTPUT_DIR / "airline_rulebook_segments_coarse.json"
    fine_path = OUTPUT_DIR / "airline_rulebook_segments_fine.json"

    with open(coarse_path, "w") as f:
        json.dump(coarse, f, indent=2)

    with open(fine_path, "w") as f:
        json.dump(fine, f, indent=2)

    print(f"Coarse: {len(coarse)} segments (contiguous, full coverage) → {coarse_path}")
    print(f"Fine:   {len(fine)} segments (rules only, {coverage:.1f}% coverage) → {fine_path}")

    # Print summary
    print("\n--- Coarse segments ---")
    for seg in coarse:
        print(f"  {seg['name']:30s}  chars {seg['char_start']:5d}–{seg['char_end']:5d}  ({seg['char_end'] - seg['char_start']:5d} chars)")

    print(f"\n--- Fine segments ({len(fine)}) ---")
    for seg in fine:
        print(f"  {seg['name']:55s}  chars {seg['char_start']:5d}–{seg['char_end']:5d}")


if __name__ == "__main__":
    main()
