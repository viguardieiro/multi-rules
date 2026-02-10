"""
Configuration classes for multi-subset InstABoost.

This module defines the data structures used to configure attention boosting:
- TokenSubset: Defines a subset of tokens with a specific bias value
- BoostConfig: Manages multiple token subsets and application settings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TokenSubset:
    """
    Defines a subset of tokens to boost with a specific bias value.

    Attributes:
        name: Descriptive name for this subset (e.g., "instruction", "examples")
        indices: List of token positions (absolute indices in the input sequence)
        bias: Bias value B to add to attention logits for these tokens
    """
    name: str
    indices: List[int]
    bias: float

    def __post_init__(self):
        """Validate TokenSubset parameters."""
        if not self.name:
            raise ValueError("TokenSubset name cannot be empty")

        if not isinstance(self.indices, list):
            raise TypeError(f"indices must be a list, got {type(self.indices)}")

        if not self.indices:
            raise ValueError(f"TokenSubset '{self.name}' has empty indices list")

        if not all(isinstance(idx, int) for idx in self.indices):
            raise TypeError(f"All indices must be integers in TokenSubset '{self.name}'")

        if any(idx < 0 for idx in self.indices):
            raise ValueError(f"All indices must be non-negative in TokenSubset '{self.name}'")

        if not isinstance(self.bias, (int, float)):
            raise TypeError(f"bias must be a number, got {type(self.bias)}")

    def __repr__(self) -> str:
        return f"TokenSubset(name='{self.name}', indices={self.indices[:3]}{'...' if len(self.indices) > 3 else ''}, bias={self.bias})"


@dataclass
class BoostConfig:
    """
    Configuration for multi-subset attention boosting.

    Attributes:
        subsets: List of TokenSubset objects defining which tokens to boost
        layers: Optional list of layer indices to apply boosting (None = all layers)
        heads: Optional list of attention head indices to apply boosting (None = all heads)
        combination: Strategy for combining biases when tokens belong to multiple subsets
                    Currently only "sum" is supported
    """
    subsets: List[TokenSubset]
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    combination: Literal["sum"] = "sum"

    def __post_init__(self):
        """Validate BoostConfig parameters."""
        if not isinstance(self.subsets, list):
            raise TypeError(f"subsets must be a list, got {type(self.subsets)}")

        if not self.subsets:
            raise ValueError("BoostConfig must have at least one TokenSubset")

        if not all(isinstance(subset, TokenSubset) for subset in self.subsets):
            raise TypeError("All subsets must be TokenSubset instances")

        # Check for duplicate subset names
        names = [subset.name for subset in self.subsets]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate subset names found: {set(duplicates)}")

        # Validate layers
        if self.layers is not None:
            if not isinstance(self.layers, list):
                raise TypeError(f"layers must be a list or None, got {type(self.layers)}")
            if not all(isinstance(layer, int) for layer in self.layers):
                raise TypeError("All layer indices must be integers")
            if any(layer < 0 for layer in self.layers):
                raise ValueError("All layer indices must be non-negative")

        # Validate heads
        if self.heads is not None:
            if not isinstance(self.heads, list):
                raise TypeError(f"heads must be a list or None, got {type(self.heads)}")
            if not all(isinstance(head, int) for head in self.heads):
                raise TypeError("All head indices must be integers")
            if any(head < 0 for head in self.heads):
                raise ValueError("All head indices must be non-negative")

        # Validate combination strategy
        if self.combination not in ["sum"]:
            raise ValueError(f"combination must be 'sum', got '{self.combination}'")

    def get_max_token_index(self) -> int:
        """
        Get the maximum token index across all subsets.

        Returns:
            Maximum token index, or -1 if no subsets exist
        """
        if not self.subsets:
            return -1
        return max(max(subset.indices) for subset in self.subsets)

    def get_min_token_index(self) -> int:
        """
        Get the minimum token index across all subsets.

        Returns:
            Minimum token index, or 0 if no subsets exist
        """
        if not self.subsets:
            return 0
        return min(min(subset.indices) for subset in self.subsets)

    def __repr__(self) -> str:
        subset_summary = f"{len(self.subsets)} subset{'s' if len(self.subsets) != 1 else ''}"
        layers_str = f"layers={self.layers}" if self.layers else "all layers"
        heads_str = f"heads={self.heads}" if self.heads else "all heads"
        return f"BoostConfig({subset_summary}, {layers_str}, {heads_str}, combination='{self.combination}')"
