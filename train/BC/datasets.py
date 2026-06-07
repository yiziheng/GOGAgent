"""Dataset loading helpers for BC trajectory generation."""

from __future__ import annotations

from typing import Iterator

from gogagent.datasets import DatasetExample, iter_examples


def load_bc_examples(
    *,
    dataset: str,
    data_path: str | Path,
    split: str,
    limit: int | None = None,
) -> Iterator[DatasetExample]:
    """Load public tasks for teacher trajectory generation."""

    yield from iter_examples(dataset=dataset, data_path=data_path, split=split, limit=limit)
