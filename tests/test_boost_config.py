"""Tests for boost_config module."""

import pytest
from src.boost_config import TokenSubset, BoostConfig


class TestTokenSubset:
    """Test TokenSubset dataclass."""

    def test_valid_subset(self):
        """Test creating a valid TokenSubset."""
        subset = TokenSubset(name="test", indices=[0, 1, 2], bias=1.5)
        assert subset.name == "test"
        assert subset.indices == [0, 1, 2]
        assert subset.bias == 1.5

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            TokenSubset(name="", indices=[0, 1], bias=1.0)

    def test_empty_indices(self):
        """Test that empty indices raises ValueError."""
        with pytest.raises(ValueError, match="empty indices"):
            TokenSubset(name="test", indices=[], bias=1.0)

    def test_negative_indices(self):
        """Test that negative indices raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            TokenSubset(name="test", indices=[0, -1, 2], bias=1.0)

    def test_non_int_indices(self):
        """Test that non-integer indices raise TypeError."""
        with pytest.raises(TypeError, match="must be integers"):
            TokenSubset(name="test", indices=[0, 1.5, 2], bias=1.0)

    def test_non_list_indices(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            TokenSubset(name="test", indices=(0, 1, 2), bias=1.0)


class TestBoostConfig:
    """Test BoostConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid BoostConfig."""
        subset1 = TokenSubset(name="inst", indices=[0, 1], bias=2.0)
        subset2 = TokenSubset(name="query", indices=[3, 4], bias=1.0)
        config = BoostConfig(subsets=[subset1, subset2])

        assert len(config.subsets) == 2
        assert config.layers is None
        assert config.heads is None
        assert config.combination == "sum"

    def test_empty_subsets(self):
        """Test that empty subsets list raises ValueError."""
        with pytest.raises(ValueError, match="at least one TokenSubset"):
            BoostConfig(subsets=[])

    def test_duplicate_names(self):
        """Test that duplicate subset names raise ValueError."""
        subset1 = TokenSubset(name="test", indices=[0, 1], bias=1.0)
        subset2 = TokenSubset(name="test", indices=[2, 3], bias=2.0)

        with pytest.raises(ValueError, match="Duplicate subset names"):
            BoostConfig(subsets=[subset1, subset2])

    def test_specific_layers(self):
        """Test config with specific layers."""
        subset = TokenSubset(name="test", indices=[0, 1], bias=1.0)
        config = BoostConfig(subsets=[subset], layers=[0, 2, 4])

        assert config.layers == [0, 2, 4]

    def test_negative_layers(self):
        """Test that negative layer indices raise ValueError."""
        subset = TokenSubset(name="test", indices=[0, 1], bias=1.0)

        with pytest.raises(ValueError, match="non-negative"):
            BoostConfig(subsets=[subset], layers=[0, -1, 2])

    def test_get_max_token_index(self):
        """Test getting maximum token index."""
        subset1 = TokenSubset(name="a", indices=[0, 1, 5], bias=1.0)
        subset2 = TokenSubset(name="b", indices=[10, 15], bias=2.0)
        config = BoostConfig(subsets=[subset1, subset2])

        assert config.get_max_token_index() == 15

    def test_get_min_token_index(self):
        """Test getting minimum token index."""
        subset1 = TokenSubset(name="a", indices=[5, 10], bias=1.0)
        subset2 = TokenSubset(name="b", indices=[2, 3], bias=2.0)
        config = BoostConfig(subsets=[subset1, subset2])

        assert config.get_min_token_index() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
