"""IFEval data loading and normalization adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any

INSTRUCTION_HEADER = "Your response should follow the instructions below:"
DEFAULT_MS_JSONL_URL = (
    "https://raw.githubusercontent.com/microsoft/llm-steer-instruct/"
    "refs/heads/main/data/ifeval_wo_instructions.jsonl"
)
DEFAULT_IFEVAL_JSONL_URL = (
    "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
)


@dataclass(frozen=True)
class IFEvalSample:
    """Normalized IFEval sample used by dynamic benchmark runners."""

    key: int
    sample_id: str
    original_prompt: str
    base_question: str
    instruction_id_list: list[str]
    kwargs_list: list[dict[str, Any]]
    instruction_texts: list[str]
    instruction_block: str
    full_input: str


def clean_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Drop None values to match checker.build_description expectations."""

    return {k: v for k, v in (kwargs or {}).items() if v is not None}


def build_instruction_block(instruction_texts: list[str]) -> str:
    """Format instruction texts as a labeled bullet list."""

    if not instruction_texts:
        raise ValueError("instruction_texts cannot be empty")
    bullets = "\n".join(f"- {text}" for text in instruction_texts)
    return f"{INSTRUCTION_HEADER}\n{bullets}"


def _resolve_instruction_dict(instruction_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    if instruction_dict is not None:
        return instruction_dict

    # Local import to avoid hard dependency during unit tests.
    import sys

    for datasets_path in _datasets_import_paths():
        if datasets_path not in sys.path:
            sys.path.insert(0, datasets_path)

    from ifeval_scripts import instructions_registry  # type: ignore

    return instructions_registry.INSTRUCTION_DICT


def _datasets_import_paths() -> list[str]:
    """Return candidate `datasets` directories for ifeval_scripts imports.

    Search order:
    1. project-root relative to this file (robust to arbitrary CWD),
    2. current working directory fallback.
    """

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        str(project_root / "datasets"),
        str(Path.cwd() / "datasets"),
    ]

    out = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        if os.path.isdir(path):
            out.append(path)
            seen.add(path)
    return out


def get_instruction_texts(
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    instruction_dict: dict[str, Any] | None = None,
) -> list[str]:
    """Build one checker description string per instruction id."""

    if len(instruction_id_list) != len(kwargs_list):
        raise ValueError("instruction_id_list and kwargs_list must have the same length")

    resolved = _resolve_instruction_dict(instruction_dict)
    texts: list[str] = []
    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        if inst_id not in resolved:
            continue
        checker = resolved[inst_id](inst_id)
        texts.append(checker.build_description(**clean_kwargs(kwargs)))
    return texts


def build_samples_from_rows(
    ifeval_rows: list[dict[str, Any]],
    ms_rows_by_key: dict[int, dict[str, Any]],
    instruction_dict: dict[str, Any] | None = None,
) -> list[IFEvalSample]:
    """Merge raw IFEval rows with MS base-question rows and normalize."""

    resolved = _resolve_instruction_dict(instruction_dict)
    samples: list[IFEvalSample] = []

    for row in ifeval_rows:
        key = row["key"]
        if key not in ms_rows_by_key:
            continue

        base_question = str(ms_rows_by_key[key].get("model_output", "")).strip()
        if not base_question:
            continue

        valid_ids: list[str] = []
        valid_kwargs: list[dict[str, Any]] = []
        for inst_id, kwargs in zip(row.get("instruction_id_list", []), row.get("kwargs", [])):
            if inst_id in resolved:
                valid_ids.append(inst_id)
                valid_kwargs.append(kwargs)

        if not valid_ids:
            continue

        instruction_texts = get_instruction_texts(valid_ids, valid_kwargs, instruction_dict=resolved)
        if not instruction_texts:
            continue

        instruction_block = build_instruction_block(instruction_texts)
        full_input = base_question + "\n\n" + instruction_block

        samples.append(
            IFEvalSample(
                key=key,
                sample_id=f"ifeval_{key}",
                original_prompt=str(row.get("prompt", "")),
                base_question=base_question,
                instruction_id_list=valid_ids,
                kwargs_list=valid_kwargs,
                instruction_texts=instruction_texts,
                instruction_block=instruction_block,
                full_input=full_input,
            )
        )

    return samples


def split_samples(
    samples: list[IFEvalSample],
    *,
    n_val: int = 100,
    n_test: int = 400,
    seed: int = 42,
) -> tuple[list[IFEvalSample], list[IFEvalSample]]:
    """Shuffle and split into validation and test sets."""

    if isinstance(n_val, bool) or not isinstance(n_val, int) or n_val < 0:
        raise ValueError("n_val must be a non-negative int")
    if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test < 0:
        raise ValueError("n_test must be a non-negative int")

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_samples = shuffled[:n_val]
    test_samples = shuffled[n_val : n_val + n_test]
    return val_samples, test_samples


def _load_ms_rows_from_jsonl_path(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows[int(record["key"])] = record
    return rows


def load_ifeval_samples(
    *,
    n_val: int = 100,
    n_test: int = 400,
    seed: int = 42,
    ifeval_rows: list[dict[str, Any]] | None = None,
    ms_rows_by_key: dict[int, dict[str, Any]] | None = None,
    ms_jsonl_path: str | None = None,
    ms_jsonl_url: str = DEFAULT_MS_JSONL_URL,
    ifeval_jsonl_url: str = DEFAULT_IFEVAL_JSONL_URL,
    instruction_dict: dict[str, Any] | None = None,
) -> tuple[list[IFEvalSample], list[IFEvalSample]]:
    """Load normalized IFEval samples.

    Offline-friendly usage is supported by providing `ifeval_rows` and
    `ms_rows_by_key` (or `ms_jsonl_path`).
    """

    if ms_rows_by_key is None:
        if ms_jsonl_path is not None:
            ms_rows_by_key = _load_ms_rows_from_jsonl_path(ms_jsonl_path)
        else:
            import requests

            resp = requests.get(ms_jsonl_url, timeout=(10, 60))
            resp.raise_for_status()
            ms_rows_by_key = {
                int(json.loads(line)["key"]): json.loads(line)
                for line in resp.text.splitlines()
                if line.strip()
            }

    if ifeval_rows is None:
        try:
            from datasets import load_dataset

            ds = load_dataset("google/IFEval", split="train")
            ifeval_rows = list(ds)
        except Exception:
            import requests

            resp = requests.get(ifeval_jsonl_url, timeout=(10, 60))
            resp.raise_for_status()
            ifeval_rows = [
                json.loads(line)
                for line in resp.text.splitlines()
                if line.strip()
            ]

    samples = build_samples_from_rows(
        ifeval_rows=ifeval_rows,
        ms_rows_by_key=ms_rows_by_key,
        instruction_dict=instruction_dict,
    )
    return split_samples(samples, n_val=n_val, n_test=n_test, seed=seed)
