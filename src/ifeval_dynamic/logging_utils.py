"""Simple experiment logging helpers for IFEval scripts."""

from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
from typing import TextIO


class ExperimentLogger:
    """Dual logger that writes to stdout and an optional log file."""

    def __init__(self, *, log_file: str | None = None, default_log_file: str | None = None):
        self.log_path = log_file or default_log_file
        self._fh: TextIO | None = None
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
            atexit.register(self.close)

    def info(self, *parts: object) -> None:
        msg = " ".join(str(p) for p in parts)
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
