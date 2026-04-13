"""Tests for src.ifeval_dynamic.data_adapter."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from src.ifeval_dynamic.data_adapter import (
    IFEvalSample,
    _datasets_import_paths,
    build_instruction_block,
    build_samples_from_rows,
    get_instruction_texts,
    load_ifeval_samples,
    split_samples,
)


class _Checker:
    def __init__(self, inst_id: str):
        self.inst_id = inst_id

    def build_description(self, target: str | None = None) -> str:
        tgt = target or "x"
        return f"Instruction {self.inst_id}: include {tgt}"

    def check_following(self, generation: str) -> bool:
        return True


def _instruction_dict():
    return {
        "i1": lambda inst_id: _Checker(inst_id),
        "i2": lambda inst_id: _Checker(inst_id),
    }


def test_build_instruction_block():
    block = build_instruction_block(["A", "B"])
    assert "Your response should follow the instructions below:" in block
    assert "- A" in block
    assert "- B" in block


def test_get_instruction_texts_filters_unknown_ids():
    texts = get_instruction_texts(
        ["i1", "unknown", "i2"],
        [{"target": "alpha"}, {}, {"target": "beta"}],
        instruction_dict=_instruction_dict(),
    )
    assert len(texts) == 2
    assert "alpha" in texts[0]
    assert "beta" in texts[1]


def test_build_samples_from_rows_and_split():
    ifeval_rows = [
        {
            "key": 1,
            "prompt": "orig prompt 1",
            "instruction_id_list": ["i1", "i2"],
            "kwargs": [{"target": "apple"}, {"target": "banana"}],
        },
        {
            "key": 2,
            "prompt": "orig prompt 2",
            "instruction_id_list": ["unknown"],
            "kwargs": [{}],
        },
    ]
    ms_rows_by_key = {
        1: {"key": 1, "model_output": "Base question 1"},
        2: {"key": 2, "model_output": "Base question 2"},
    }

    samples = build_samples_from_rows(
        ifeval_rows=ifeval_rows,
        ms_rows_by_key=ms_rows_by_key,
        instruction_dict=_instruction_dict(),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, IFEvalSample)
    assert sample.key == 1
    assert sample.sample_id == "ifeval_1"
    assert sample.base_question == "Base question 1"
    assert sample.instruction_id_list == ["i1", "i2"]
    assert "Your response should follow" in sample.full_input

    val, test = split_samples(samples, n_val=1, n_test=1, seed=123)
    assert len(val) == 1
    assert len(test) == 0


def test_load_ifeval_samples_falls_back_to_jsonl_when_datasets_fails(monkeypatch):
    class _Resp:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    fake_datasets = types.ModuleType("datasets")

    def _boom(*_args, **_kwargs):
        raise ValueError("boom")

    fake_datasets.load_dataset = _boom
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    ifeval_line = '{"key": 1, "prompt": "orig", "instruction_id_list": ["i1"], "kwargs": [{}]}'
    ms_line = '{"key": 1, "model_output": "base question"}'

    def _fake_get(url, timeout):
        if "ifeval_input_data.jsonl" in url:
            return _Resp(ifeval_line + "\n")
        return _Resp(ms_line + "\n")

    monkeypatch.setattr("requests.get", _fake_get)

    instruction_dict = {
        "i1": lambda _inst_id: types.SimpleNamespace(build_description=lambda **_kwargs: "desc")
    }

    val, test = load_ifeval_samples(
        n_val=1,
        n_test=0,
        seed=0,
        instruction_dict=instruction_dict,
    )
    assert len(val) == 1
    assert len(test) == 0
    assert val[0].full_input.startswith("base question")


def test_datasets_import_paths_prefers_project_root(monkeypatch):
    monkeypatch.chdir("/tmp")
    paths = _datasets_import_paths()
    expected = str(Path(__file__).resolve().parents[1] / "datasets")
    assert paths
    assert paths[0] == expected
