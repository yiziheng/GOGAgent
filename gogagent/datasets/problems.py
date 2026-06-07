"""Dataset-aware problem shaping helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from gogagent.datasets.loaders import DatasetExample, normalize_dataset


def make_problem(example: DatasetExample) -> dict[str, Any]:
    """Build the LLM-facing problem without exposing gold labels."""

    dataset_name = normalize_dataset(example.dataset)
    problem = {
        **dict(example.public_task),
        "dataset": dataset_name,
    }
    problem.setdefault("answer_format", answer_format_for_dataset(dataset_name))
    return problem


def answer_format_for_dataset(dataset: str) -> str:
    """Return the default GraphMessage answer-format instruction."""

    dataset_name = normalize_dataset(dataset)
    if dataset_name == "mmlu":
        return (
            "The GraphMessage answer field must be exactly one option letter: "
            "A, B, C, or D."
        )
    if dataset_name == "gsm8k":
        return "The GraphMessage answer field must contain only the final numeric answer."
    if dataset_name == "humaneval":
        return "The GraphMessage answer field must contain the final Python solution code."
    raise AssertionError(f"unhandled dataset: {dataset_name}")


def enrich_example(
    example: DatasetExample,
    selection: Mapping[tuple[str, str], Mapping[str, Any]],
) -> DatasetExample:
    """Attach sampled-set ids when a GPTSwarm-style MMLU selection file is provided."""

    if not selection or example.dataset.lower() != "mmlu":
        return example
    public_task = dict(example.public_task)
    local_task_id = public_task.get("task_id")
    key = (
        str(public_task.get("subject", "")),
        str(public_task.get("question", "")),
    )
    source = selection.get(key)
    if source is not None:
        public_task["task_id"] = str(source.get("task_id", local_task_id))
        public_task["source_row"] = source.get("source_row")
        public_task["source_file"] = source.get("source_file")
        if "rank" in source:
            public_task["selection_rank"] = source.get("rank")
    public_task["local_task_id"] = local_task_id
    return DatasetExample(dataset=example.dataset, public_task=public_task, gold=example.gold)


def load_selection(path: str | Path | None) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Load a GPTSwarm-style MMLU selection JSONL keyed by subject/question."""

    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("subject", "")), str(row.get("question", "")))
            rows[key] = row
    return rows
