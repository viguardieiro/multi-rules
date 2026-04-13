"""Tests for src.ifeval_dynamic.runtime_checks."""

from __future__ import annotations

import pytest

from src.ifeval_dynamic.runtime_checks import require_ifeval_runtime_dependencies


def test_runtime_checks_reports_missing_package(monkeypatch):
    import src.ifeval_dynamic.runtime_checks as mod

    class _NltkOK:
        class data:
            @staticmethod
            def find(_path):
                return True

    def _fake_import(name: str):
        if name == "absl":
            raise ImportError("missing")
        if name in {"langdetect", "immutabledict"}:
            return object()
        if name == "nltk":
            return _NltkOK
        raise ImportError(name)

    monkeypatch.setattr(mod.importlib, "import_module", _fake_import)

    with pytest.raises(RuntimeError, match="absl-py"):
        require_ifeval_runtime_dependencies()


def test_runtime_checks_reports_missing_punkt(monkeypatch):
    import src.ifeval_dynamic.runtime_checks as mod

    class _NltkNoPunkt:
        class data:
            @staticmethod
            def find(_path):
                raise LookupError("missing punkt")

    def _fake_import(name: str):
        if name in {"absl", "langdetect", "immutabledict"}:
            return object()
        if name == "nltk":
            return _NltkNoPunkt
        raise ImportError(name)

    monkeypatch.setattr(mod.importlib, "import_module", _fake_import)

    with pytest.raises(RuntimeError, match="punkt"):
        require_ifeval_runtime_dependencies()
