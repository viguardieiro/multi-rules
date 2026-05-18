"""RuleArena helpers for segmentation, applicability, and rule-order tracing."""

from .rule_application_oracle import (
    get_checked_bag_processing_order,
    get_rule_application_trace,
)
from .rule_applicability import (
    get_applied_rules,
    get_applied_rules_with_coarse,
    build_filtered_rulebook,
)
from .rulebook_segments import (
    get_coarse_segments,
    get_fine_segments,
    parse_rulebook_sections,
)

__all__ = [
    "parse_rulebook_sections",
    "get_coarse_segments",
    "get_fine_segments",
    "get_applied_rules",
    "get_applied_rules_with_coarse",
    "build_filtered_rulebook",
    "get_checked_bag_processing_order",
    "get_rule_application_trace",
]
