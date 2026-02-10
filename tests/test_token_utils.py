"""Tests for token_utils module."""

import pytest
from transformers import AutoTokenizer
from src.token_utils import (
    find_substring_token_indices,
    create_token_subset_from_substring,
    validate_token_indices
)
from src.boost_config import TokenSubset


class TestValidateTokenIndices:
    """Test validate_token_indices function."""

    def test_valid_indices(self):
        """Test with valid indices."""
        validate_token_indices([0, 1, 2, 5], max_length=10)
        # Should not raise

    def test_negative_index(self):
        """Test that negative indices raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_token_indices([0, -1, 2], max_length=10)

    def test_out_of_range(self):
        """Test that out-of-range indices raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_token_indices([0, 1, 10], max_length=10)

    def test_non_list(self):
        """Test that non-list raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            validate_token_indices((0, 1, 2), max_length=10)


class TestFindSubstringTokenIndices:
    """Test find_substring_token_indices function."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer for testing."""
        # Use GPT-2 tokenizer as it's small and widely available
        return AutoTokenizer.from_pretrained("gpt2")

    def test_simple_substring(self, tokenizer):
        """Test finding a simple substring."""
        text = "Hello world, how are you?"
        substring = "world"

        indices = find_substring_token_indices(text, substring, tokenizer)

        # Verify the indices correspond to the substring
        assert isinstance(indices, list)
        assert len(indices) > 0
        assert all(isinstance(i, int) for i in indices)

        # Decode to verify
        full_tokens = tokenizer.encode(text, add_special_tokens=False)
        substring_tokens = [full_tokens[i] for i in indices]
        decoded = tokenizer.decode(substring_tokens)
        assert substring.lower() in decoded.lower() or decoded.lower() in substring.lower()

    def test_substring_not_found(self, tokenizer):
        """Test that ValueError is raised when substring not in text."""
        text = "Hello world"
        substring = "goodbye"

        with pytest.raises(ValueError, match="Substring not found"):
            find_substring_token_indices(text, substring, tokenizer)

    def test_empty_substring(self, tokenizer):
        """Test that empty substring raises ValueError."""
        text = "Hello world"
        substring = ""

        with pytest.raises(ValueError, match="cannot be empty"):
            find_substring_token_indices(text, substring, tokenizer)

    def test_multi_word_substring(self, tokenizer):
        """Test finding a multi-word substring."""
        text = "The quick brown fox jumps over the lazy dog"
        substring = "brown fox jumps"

        indices = find_substring_token_indices(text, substring, tokenizer)

        # Verify
        assert isinstance(indices, list)
        assert len(indices) > 0

        # The substring should decode correctly
        full_tokens = tokenizer.encode(text, add_special_tokens=False)
        substring_tokens = [full_tokens[i] for i in indices]
        decoded = tokenizer.decode(substring_tokens).strip()

        # Check that the key words are present
        assert "brown" in decoded.lower()
        assert "fox" in decoded.lower()


class TestCreateTokenSubsetFromSubstring:
    """Test create_token_subset_from_substring function."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer for testing."""
        return AutoTokenizer.from_pretrained("gpt2")

    def test_create_subset(self, tokenizer):
        """Test creating a TokenSubset from substring."""
        text = "Instruction: Answer in French."
        substring = "Instruction:"
        bias = 2.0

        subset = create_token_subset_from_substring(
            name="instruction",
            text=text,
            substring=substring,
            tokenizer=tokenizer,
            bias=bias
        )

        assert isinstance(subset, TokenSubset)
        assert subset.name == "instruction"
        assert subset.bias == 2.0
        assert len(subset.indices) > 0
        assert all(isinstance(i, int) for i in subset.indices)

    def test_create_subset_not_found(self, tokenizer):
        """Test that ValueError is raised when substring not found."""
        text = "Hello world"
        substring = "goodbye"

        with pytest.raises(ValueError, match="Substring not found"):
            create_token_subset_from_substring(
                name="test",
                text=text,
                substring=substring,
                tokenizer=tokenizer,
                bias=1.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
