"""Dependency-free loaders for supported benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


SUPPORTED_DATASETS = ("mmlu", "mmlu_pro", "gsm8k", "humaneval", "multiagentbench")

_MULTIAGENTBENCH_GOLD_KEYS = (
    "answer",
    "gold",
    "target",
    "label",
    "final_answer",
    "canonical_answer",
    "expected",
    "reference",
    "solution",
)


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


def load_multiagentbench_jsonl(path: str | Path) -> Iterator[DatasetExample]:
    """Load MultiAgentBench/MARBLE-style JSONL tasks.

    Official MultiAgentBench rows are scenario configs rather than one uniform
    QA schema. This adapter keeps only LLM-visible task fields in ``public_task``
    and extracts optional gold labels from common benchmark keys when present.
    Rows without gold are still executable, but the dedicated eval script reports
    them as unscored instead of counting them as wrong.
    """

    source = Path(path)
    paths = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    if not paths:
        raise RuntimeError(f"no MultiAgentBench JSONL files found at {source}")

    for data_path in paths:
        for line_number, row in _read_jsonl(data_path):
            raw_task = row.get("task")
            task_text = _task_text(row)
            if not task_text:
                raise ValueError(
                    f"{data_path}:{line_number} missing task/content/question field"
                )
            scenario = str(row.get("scenario") or data_path.stem).strip()
            raw_task_id = row.get("task_id", line_number)
            task_id = f"{scenario}-{raw_task_id}"
            output_format = _task_field(raw_task, "output_format") or row.get(
                "output_format"
            )

            public_task: dict[str, Any] = {
                "task_id": task_id,
                "source_file": str(data_path),
                "source_row": line_number,
                "scenario": scenario,
                "task": str(task_text),
                "output_format": _optional_str(output_format),
                "dataset_protocol": "multiagentbench_jsonl_v1",
            }
            options = _normalize_options(
                row.get("options")
                or row.get("choices")
                or _task_field(raw_task, "options")
                or _task_field(raw_task, "choices")
            )
            if options:
                public_task["options"] = options

            for key in (
                "answer_type",
                "metric",
                "eval_metric",
                "context",
                "agents",
                "relationships",
                "environment",
                "communication",
                "memory",
                "metrics",
            ):
                if key in row:
                    public_task[key] = row[key]

            yield DatasetExample(
                dataset="multiagentbench",
                public_task=public_task,
                gold=_extract_multiagentbench_gold(row),
            )


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


def load_examples(
    *,
    dataset: str,
    data_path: str | Path,
    split: str,
    limit: int | None = None,
) -> list[DatasetExample]:
    """Load a supported benchmark into public-task/gold examples."""

    dataset_name = normalize_dataset(dataset)
    if dataset_name == "mmlu":
        iterator = load_mmlu_directory(data_path, split=split)
    elif dataset_name == "mmlu_pro":
        del split
        from gogagent.datasets.mmlu_pro import load_mmlu_pro_jsonl

        iterator = load_mmlu_pro_jsonl(data_path)
    elif dataset_name == "gsm8k":
        iterator = load_gsm8k_jsonl(data_path)
    elif dataset_name == "humaneval":
        iterator = load_humaneval_jsonl(data_path)
    elif dataset_name == "multiagentbench":
        del split
        iterator = load_multiagentbench_jsonl(data_path)
    else:
        raise AssertionError(f"unhandled dataset: {dataset_name}")

    examples: list[DatasetExample] = []
    for index, example in enumerate(iterator, start=1):
        if limit is not None and index > limit:
            break
        examples.append(example)
    if not examples:
        raise RuntimeError(f"no {dataset_name} examples found at {data_path}")
    return examples


def iter_examples(
    *,
    dataset: str,
    data_path: str | Path,
    split: str,
    limit: int | None = None,
) -> Iterator[DatasetExample]:
    """Yield supported benchmark examples with an optional row limit."""

    yield from load_examples(
        dataset=dataset,
        data_path=data_path,
        split=split,
        limit=limit,
    )


def normalize_dataset(dataset: str) -> str:
    """Normalize and validate a supported dataset name."""

    dataset_name = dataset.strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}"
        )
    return dataset_name


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


def _task_text(row: Mapping[str, Any]) -> str | None:
    raw_task = row.get("task")
    if isinstance(raw_task, Mapping):
        for key in ("content", "question", "prompt", "instruction", "task"):
            value = raw_task.get(key)
            if value:
                return str(value)
        return json.dumps(raw_task, ensure_ascii=False, sort_keys=True)
    if raw_task:
        return str(raw_task)
    for key in ("question", "prompt", "instruction", "content", "problem"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _task_field(raw_task: Any, key: str) -> Any:
    if isinstance(raw_task, Mapping):
        return raw_task.get(key)
    return None


def _normalize_options(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        options = {str(key).upper(): choice for key, choice in value.items()}
        return options or None
    if isinstance(value, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {
            labels[index]: choice
            for index, choice in enumerate(value)
            if index < len(labels)
        } or None
    return None


def _extract_multiagentbench_gold(row: Mapping[str, Any]) -> Any | None:
    for key in _MULTIAGENTBENCH_GOLD_KEYS:
        if key in row:
            return row[key]
    for container_key in ("eval", "evaluation", "golden"):
        container = row.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in _MULTIAGENTBENCH_GOLD_KEYS:
            if key in container:
                return container[key]
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
