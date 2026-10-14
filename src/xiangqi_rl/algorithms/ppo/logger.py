"""Tiny dependency-free CSV logger."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append rows with a fixed set of columns to a CSV file.

    The header is written only when the file is new or empty, so a resumed run
    continues the same file. Unknown keys raise; missing keys are left blank.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = not self.path.exists() or self.path.stat().st_size == 0

    def log(self, row: Mapping[str, Any]) -> None:
        extra = set(row) - set(self.fieldnames)
        if extra:
            raise KeyError(f"unknown columns for {self.path.name}: {sorted(extra)}")
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
