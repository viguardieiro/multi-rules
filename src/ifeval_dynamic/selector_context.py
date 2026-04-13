"""Selector context construction for IFEval dynamic boosting."""

from __future__ import annotations

from typing import Any

from .data_adapter import IFEvalSample


def build_selector_context(
    *,
    sample: IFEvalSample,
    current_generation: str,
    active_instruction_ids: list[str],
    generation_token_count: int,
    step_index: int,
    max_generation_chars: int = 1600,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact selector context for IFEval samples."""

    if isinstance(max_generation_chars, bool) or not isinstance(max_generation_chars, int) or max_generation_chars < 1:
        raise ValueError("max_generation_chars must be a positive int")

    if not isinstance(current_generation, str):
        raise TypeError("current_generation must be a string")

    if isinstance(generation_token_count, bool) or not isinstance(generation_token_count, int) or generation_token_count < 0:
        raise ValueError("generation_token_count must be a non-negative int")

    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative int")

    instruction_text_by_id = {
        inst_id: text
        for inst_id, text in zip(sample.instruction_id_list, sample.instruction_texts)
    }

    invalid = [x for x in active_instruction_ids if x not in instruction_text_by_id]
    if invalid:
        raise ValueError(f"active_instruction_ids contain unknown ids: {invalid}")

    metadata = {
        "ifeval_key": sample.key,
        "original_prompt": sample.original_prompt,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "sample_id": sample.sample_id,
        "base_prompt": sample.full_input,
        "candidate_instruction_ids": list(sample.instruction_id_list),
        "instruction_text_by_id": instruction_text_by_id,
        "currently_active_instruction_ids": list(active_instruction_ids),
        "current_generation": current_generation[-max_generation_chars:],
        "generation_token_count": generation_token_count,
        "step_index": step_index,
        "metadata": metadata,
    }
