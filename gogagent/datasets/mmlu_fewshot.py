"""MMLU subject-specific few-shot example loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from gogagent.datasets.loaders import load_mmlu_directory


def load_mmlu_fewshot_by_subject(
    dev_path: str | Path,
    *,
    shot_count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Load the first ``shot_count`` dev examples for every MMLU subject."""

    if shot_count <= 0:
        raise ValueError("shot_count must be positive")
    examples_by_subject: dict[str, list[dict[str, Any]]] = {}
    for example in load_mmlu_directory(dev_path, split="dev"):
        subject = str(example.public_task.get("subject", "")).strip()
        if not subject:
            raise ValueError(f"MMLU dev example is missing subject: {example}")
        bucket = examples_by_subject.setdefault(subject, [])
        if len(bucket) >= shot_count:
            continue
        bucket.append(
            {
                "question": example.public_task["question"],
                "options": dict(example.public_task["options"]),
                "answer": str(example.gold).strip().upper(),
            }
        )
    if not examples_by_subject:
        raise RuntimeError(f"no MMLU dev examples found at {dev_path}")
    missing = [
        subject
        for subject, examples in examples_by_subject.items()
        if len(examples) < shot_count
    ]
    if missing:
        raise RuntimeError(
            f"MMLU dev path {dev_path} has fewer than {shot_count} examples for "
            f"subjects: {', '.join(sorted(missing))}"
        )
    return examples_by_subject


def attach_mmlu_fewshot_examples(
    public_task: Mapping[str, Any],
    fewshot_by_subject: Mapping[str, list[Mapping[str, Any]]],
    *,
    shot_count: int = 5,
) -> dict[str, Any]:
    """Return a public task copy with same-subject MMLU dev shots attached."""

    subject = str(public_task.get("subject", "")).strip()
    if not subject:
        raise ValueError("MMLU few-shot prompting requires a subject field")
    examples = fewshot_by_subject.get(subject)
    if examples is None:
        raise RuntimeError(f"missing MMLU dev few-shot examples for subject {subject!r}")
    if len(examples) < shot_count:
        raise RuntimeError(
            f"subject {subject!r} has only {len(examples)} dev examples; "
            f"{shot_count} requested"
        )
    task = dict(public_task)
    task["mmlu_fewshot_examples"] = [dict(example) for example in examples[:shot_count]]
    return task
