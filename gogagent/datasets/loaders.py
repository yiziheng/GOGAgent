"""Dependency-free loaders for the first three supported benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class DatasetExample:
    """Keep labels out of inference by construction.

    Pass only ``public_task`` to graph execution. The ``gold`` field is reserved
    for reward/evaluation code after inference has completed.
    """

    dataset: str
    public_task: Mapping[str, Any]
    gold: Any


def load_gsm8k_jsonl(path: str | Path) -> Iterator[DatasetExample]:
    """Load canonical GSM8K JSONL rows with ``question`` and ``answer``."""

    for line_number, row in _read_jsonl(path):
        question = _require(row, "question", path, line_number)
        answer = _require(row, "answer", path, line_number)
        yield DatasetExample(
            dataset="gsm8k",
            public_task={
                "task_id": str(row.get("task_id", f"gsm8k-{line_number}")),
                "question": str(question),
            },
            gold={"answer": str(answer)},
        )


def load_humaneval_jsonl(path: str | Path) -> Iterator[DatasetExample]:
    """Load canonical HumanEval JSONL without exposing tests to the runtime."""

    for line_number, row in _read_jsonl(path):
        prompt = _require(row, "prompt", path, line_number)
        task_id = str(row.get("task_id", f"humaneval-{line_number}"))
        public_task = {
            "task_id": task_id,
            "prompt": str(prompt),
            "entry_point": str(row.get("entry_point", "")),
        }
        gold = {
            key: row[key]
            for key in ("canonical_solution", "test")
            if key in row
        }
        yield DatasetExample(dataset="humaneval", public_task=public_task, gold=gold)


def load_mmlu_directory(
    directory: str | Path,
    split: str = "test",
) -> Iterator[DatasetExample]:
    """Load MMLU CSV files named ``<subject>_<split>.csv``.

    Canonical rows contain question, four answer choices, then one label.
    """

    root = Path(directory)
    suffix = f"_{split}.csv"
    for path in sorted(root.glob(f"*{suffix}")):
        subject = path.name[: -len(suffix)]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if len(row) < 6:
                    raise ValueError(
                        f"{path}:{row_number} expected 6 CSV columns, got {len(row)}"
                    )
                question, *tail = row
                options = dict(zip(("A", "B", "C", "D"), tail[:4], strict=True))
                yield DatasetExample(
                    dataset="mmlu",
                    public_task={
                        "task_id": f"{subject}-{split}-{row_number}",
                        "subject": subject,
                        "question": question,
                        "options": options,
                    },
                    gold=tail[4],
                )


def _read_jsonl(path: str | Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_number} expected one JSON object")
            yield line_number, row


def _require(
    row: Mapping[str, Any],
    key: str,
    path: str | Path,
    line_number: int,
) -> Any:
    try:
        return row[key]
    except KeyError as error:
        raise ValueError(f"{path}:{line_number} missing required field {key!r}") from error
