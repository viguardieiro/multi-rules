"""Runtime dependency checks for IFEval runner scripts."""

from __future__ import annotations

import importlib
from typing import Any


def require_ifeval_runtime_dependencies() -> None:
    """Ensure required Python packages and NLTK data are available."""

    missing = []
    for module_name, package_name in [
        ("absl", "absl-py"),
        ("langdetect", "langdetect"),
        ("immutabledict", "immutabledict"),
        ("nltk", "nltk"),
    ]:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_name)

    if missing:
        unique = sorted(set(missing))
        raise RuntimeError(
            "Missing required runtime packages for IFEval checkers: "
            + ", ".join(unique)
            + ". Install with: python -m pip install -U "
            + " ".join(unique)
        )

    nltk_mod: Any = importlib.import_module("nltk")
    try:
        nltk_mod.data.find("tokenizers/punkt/english.pickle")
    except LookupError as exc:
        raise RuntimeError(
            "Missing required NLTK tokenizer data 'punkt'. "
            "Install with: python -c \"import nltk; nltk.download('punkt')\""
        ) from exc
