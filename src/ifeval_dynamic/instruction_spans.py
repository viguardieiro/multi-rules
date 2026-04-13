"""Instruction span mapping for IFEval instruction-last prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data_adapter import INSTRUCTION_HEADER, build_instruction_block


@dataclass(frozen=True)
class InstructionSpan:
    """Token span for a single instruction inside the full prompt."""

    instruction_id: str
    start_token: int
    end_token: int


def _offsets_from_tokenizer(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset_mapping")
    return [tuple(pair) for pair in offsets]


def _char_span_to_token_span(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int]:
    if start_char < 0 or end_char < start_char:
        raise ValueError("Invalid character span")

    start_token = None
    end_token = None

    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break

        if start_token is None:
            start_token = idx
        end_token = idx + 1

    if start_token is None or end_token is None:
        raise ValueError("Could not map character span to token span")
    return start_token, end_token


def compute_instruction_block_token_span(tokenizer: Any, full_input: str, instruction_block: str) -> tuple[int, int]:
    """Compute token span of the full instruction block in full_input."""

    block_start = full_input.find(instruction_block)
    if block_start == -1:
        raise ValueError("instruction_block not found inside full_input")
    block_end = block_start + len(instruction_block)

    offsets = _offsets_from_tokenizer(tokenizer, full_input)
    return _char_span_to_token_span(offsets, block_start, block_end)


def compute_instruction_spans(
    tokenizer: Any,
    full_input: str,
    instruction_id_list: list[str],
    instruction_texts: list[str],
    *,
    instruction_block: str | None = None,
) -> list[InstructionSpan]:
    """Compute token spans for each instruction bullet inside full_input."""

    if len(instruction_id_list) != len(instruction_texts):
        raise ValueError("instruction_id_list and instruction_texts must have same length")

    if instruction_block is None:
        instruction_block = build_instruction_block(instruction_texts)

    block_start = full_input.find(instruction_block)
    if block_start == -1:
        # Fallback for partially formatted prompts: anchor to the instruction
        # header so per-bullet diagnostics can still run.
        block_start = full_input.find(INSTRUCTION_HEADER)
        if block_start == -1:
            raise ValueError("instruction_block not found inside full_input")

    offsets = _offsets_from_tokenizer(tokenizer, full_input)

    spans: list[InstructionSpan] = []
    search_pos = block_start
    for inst_id, inst_text in zip(instruction_id_list, instruction_texts):
        bullet = f"- {inst_text}"
        local_pos = full_input.find(bullet, search_pos)
        if local_pos == -1:
            raise ValueError(f"Could not find bullet for instruction '{inst_id}'")

        start_char = local_pos
        end_char = local_pos + len(bullet)
        start_tok, end_tok = _char_span_to_token_span(offsets, start_char, end_char)

        spans.append(InstructionSpan(instruction_id=inst_id, start_token=start_tok, end_token=end_tok))
        search_pos = end_char

    return spans
