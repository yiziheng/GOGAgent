"""Dataset loading helpers for BC trajectory generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from gogagent.datasets import (
    DatasetExample,
    load_gsm8k_jsonl,
    load_humaneval_jsonl,
    load_mmlu_directory,
)


def load_bc_examples(
    *,
    dataset: str,
    data_path: str | Path,
    split: str,
    limit: int | None = None,
) -> Iterator[DatasetExample]:
    """Load public tasks for teacher trajectory generation."""

    normalized = dataset.strip().lower()
    if normalized == "mmlu":
        iterator = load_mmlu_directory(data_path, split=split)
    elif normalized == "gsm8k":
        iterator = load_gsm8k_jsonl(data_path)
    elif normalized == "humaneval":
        iterator = load_humaneval_jsonl(data_path)
    else:
        raise ValueError(f"unsupported dataset for BC trajectory generation: {dataset}")

    for index, example in enumerate(iterator, start=1):
        if limit is not None and index > limit:
            break
        yield example
