"""
Token manipulation utilities for finding and validating token indices.

This module provides functions to map text substrings to token indices,
with proper validation and error handling for tokenization edge cases.
"""

from typing import List
from .boost_config import TokenSubset


def validate_token_indices(indices: List[int], max_length: int) -> None:
    """
    Validate that token indices are within valid range.

    Args:
        indices: List of token indices to validate
        max_length: Maximum valid index (exclusive)

    Raises:
        ValueError: If any index is out of range
        TypeError: If indices is not a list or contains non-integers
    """
    if not isinstance(indices, list):
        raise TypeError(f"indices must be a list, got {type(indices)}")

    if not all(isinstance(idx, int) for idx in indices):
        raise TypeError("All indices must be integers")

    if any(idx < 0 for idx in indices):
        raise ValueError("Token indices cannot be negative")

    if any(idx >= max_length for idx in indices):
        invalid = [idx for idx in indices if idx >= max_length]
        raise ValueError(
            f"Token indices {invalid} are out of range. "
            f"Valid range is [0, {max_length})"
        )


def find_substring_token_indices(
    text: str,
    substring: str,
    tokenizer,
    occurrence: int = 0,
    add_special_tokens: bool = True,
) -> List[int]:
    """
    Find token indices corresponding to a substring within text.

    This function handles BPE tokenization edge cases by finding the character
    position of the substring in the text, then determining which tokens
    correspond to that character range.

    Important: ``add_special_tokens`` must match the setting used when
    tokenizing the model input, so that the returned indices align with
    the actual token positions the model sees.

    Args:
        text: Full input text
        substring: Substring to find token indices for
        tokenizer: HuggingFace tokenizer instance
        occurrence: Which occurrence to find (0 = first, 1 = second, etc.)
        add_special_tokens: Whether to include special tokens (e.g. BOS)
            when tokenizing.  Must match the setting used for the model
            input so that indices are consistent.  Defaults to ``True``
            (the HuggingFace tokenizer default).

    Returns:
        List of absolute token indices where substring appears

    Raises:
        ValueError: If substring not found in text, or if tokenization matching fails
        TypeError: If inputs have wrong types
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)}")

    if not isinstance(substring, str):
        raise TypeError(f"substring must be a string, got {type(substring)}")

    if not substring:
        raise ValueError("substring cannot be empty")

    if occurrence < 0:
        raise ValueError(f"occurrence must be non-negative, got {occurrence}")

    # Check if substring exists in text
    if substring not in text:
        raise ValueError(
            f"Substring not found in text.\n"
            f"Substring: '{substring}'\n"
            f"Text: '{text}'"
        )

    # Find the character position of the substring
    char_positions = []
    start = 0
    while True:
        pos = text.find(substring, start)
        if pos == -1:
            break
        char_positions.append(pos)
        start = pos + 1

    if occurrence >= len(char_positions):
        raise ValueError(
            f"Requested occurrence {occurrence} but only found {len(char_positions)} "
            f"occurrence(s) of substring in text"
        )

    char_pos = char_positions[occurrence]
    char_end = char_pos + len(substring)

    # Tokenize the full text and get character spans
    # HuggingFace tokenizers can return char-to-token mappings
    encoding = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)

    offset_mapping = encoding['offset_mapping']
    token_ids = encoding['input_ids']

    # Find which tokens overlap with the substring's character range
    matching_indices = []
    for idx, (start_char, end_char) in enumerate(offset_mapping):
        # Check if this token overlaps with the substring
        # Token overlaps if it starts before substring ends AND ends after substring starts
        if start_char < char_end and end_char > char_pos:
            matching_indices.append(idx)

    if not matching_indices:
        raise ValueError(
            f"Could not find tokens corresponding to substring.\n"
            f"Substring: '{substring}' at position {char_pos}\n"
            f"This may be due to special tokenization handling."
        )

    return matching_indices


def create_token_subset_from_substring(
    name: str,
    text: str,
    substring: str,
    tokenizer,
    bias: float,
    occurrence: int = 0,
    add_special_tokens: bool = True,
) -> TokenSubset:
    """
    Convenience function to create a TokenSubset from a substring.

    This combines substring finding and TokenSubset creation in one step.

    Args:
        name: Name for the token subset
        text: Full input text
        substring: Substring to find and boost
        tokenizer: HuggingFace tokenizer instance
        bias: Bias value to apply to these tokens
        occurrence: Which occurrence to find (0 = first, 1 = second, etc.)
        add_special_tokens: Whether to include special tokens when
            tokenizing.  Must match the model input tokenization.
            Defaults to ``True``.

    Returns:
        TokenSubset instance with indices corresponding to the substring

    Raises:
        ValueError: If substring not found or tokenization fails
        TypeError: If inputs have wrong types
    """
    indices = find_substring_token_indices(
        text, substring, tokenizer, occurrence, add_special_tokens
    )
    return TokenSubset(name=name, indices=indices, bias=bias)


def segments_to_token_indices(
    segments: list[dict],
    full_prompt: str,
    rulebook_text: str,
    tokenizer,
    add_special_tokens: bool = True,
) -> list[dict]:
    """Convert segment dicts to token index ranges.

    Each segment in *segments* must have a ``substring`` key whose value
    appears in *rulebook_text* (and, by extension, in *full_prompt*).

    This function locates each substring inside *full_prompt* and calls
    :func:`find_substring_token_indices` to obtain the matching token
    indices.

    Parameters
    ----------
    segments : list[dict]
        Segment dicts as returned by
        ``src.rulearena.rulebook_segments.get_coarse_segments`` (or fine).
        Each must contain at least ``name`` and ``substring``.
    full_prompt : str
        The complete model prompt that contains the rulebook.
    rulebook_text : str
        The raw rulebook text, used to compute the correct occurrence
        offset when *full_prompt* contains the rulebook exactly once.
    tokenizer
        A HuggingFace tokenizer instance.
    add_special_tokens : bool
        Must match the setting used when tokenizing the model input.

    Returns
    -------
    list[dict]
        A copy of each input segment dict, augmented with a
        ``token_indices`` key (a ``list[int]``).
    """
    # Find where the rulebook starts inside the full prompt so that
    # we can resolve each segment's substring unambiguously.
    rulebook_offset = full_prompt.find(rulebook_text)
    if rulebook_offset == -1:
        raise ValueError("rulebook_text not found in full_prompt")

    result: list[dict] = []
    for seg in segments:
        substring = seg["substring"]
        # Compute the character position of this segment within full_prompt
        char_start_in_prompt = rulebook_offset + seg["char_start"]
        # Verify the substring matches at the expected position
        expected = full_prompt[char_start_in_prompt:char_start_in_prompt + len(substring)]
        if expected != substring:
            raise ValueError(
                f"Segment '{seg['name']}' substring does not match at expected "
                f"position {char_start_in_prompt} in full_prompt."
            )
        indices = find_substring_token_indices(
            full_prompt, substring, tokenizer,
            add_special_tokens=add_special_tokens,
        )
        result.append({**seg, "token_indices": indices})

    return result
