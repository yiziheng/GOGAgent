"""Small JSON/JSONL file helpers shared by training and eval code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one mapping as a JSONL row."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write one JSON object."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as JSONL and return the row count."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
