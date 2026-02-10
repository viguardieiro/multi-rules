"""
Multi-Subset InstABoost

An extended implementation of the InstABoost method for instruction following
in Large Language Models, supporting multiple token subsets with different bias parameters.
"""

__version__ = "0.1.0"

from .boost_config import TokenSubset, BoostConfig
from .attention_hook import (
    register_boost_hooks,
    unregister_boost_hooks,
    update_bias_mask,
)
from .token_utils import (
    find_substring_token_indices,
    create_token_subset_from_substring,
    validate_token_indices,
)

__all__ = [
    "TokenSubset",
    "BoostConfig",
    "register_boost_hooks",
    "unregister_boost_hooks",
    "update_bias_mask",
    "find_substring_token_indices",
    "create_token_subset_from_substring",
    "validate_token_indices",
]
