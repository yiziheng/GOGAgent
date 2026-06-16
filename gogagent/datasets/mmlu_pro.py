"""MMLU-Pro JSONL loader with dynamic option-label normalization."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

from gogagent.datasets.loaders import DatasetExample


OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_GOLD_KEYS = (
    "answer",
    "gold",
    "target",
    "label",
    "correct_answer",
    "final_answer",
)
_INDEX_KEYS = (
    "answer_index",
    "answer_idx",
    "label_index",
    "target_index",
    "correct_index",
)


def load_mmlu_pro_jsonl(path: str | Path) -> Iterator[DatasetExample]:
    """Load normalized MMLU-Pro examples from one JSONL file or a JSONL directory."""

    source = Path(path)
    paths = _jsonl_paths(source)
    if not paths:
        raise RuntimeError(f"no MMLU-Pro JSONL files found at {source}")
    for data_path in paths:
        for line_number, row in _read_jsonl(data_path):
            yield normalize_mmlu_pro_row(row, source_path=data_path, line_number=line_number)


def normalize_mmlu_pro_row(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    line_number: int,
) -> DatasetExample:
    """Normalize one MMLU-Pro row into the project public-task/gold contract."""

    question = _first_present(row, ("question", "prompt", "problem", "input"))
    if question is None or not str(question).strip():
        raise ValueError(f"{source_path}:{line_number} missing MMLU-Pro question")

    options = normalize_mmlu_pro_options(_raw_options(row))
    if len(options) < 2:
        raise ValueError(f"{source_path}:{line_number} requires at least two options")

    gold = normalize_mmlu_pro_gold(row, options)
    labels = tuple(options)
    subject = str(
        row.get("subject")
        or row.get("category")
        or row.get("discipline")
        or row.get("field")
        or source_path.stem
    ).strip()
    task_id = str(
        row.get("task_id")
        or row.get("question_id")
        or row.get("id")
        or f"{source_path.stem}-{line_number}"
    )
    public_task = {
        "task_id": task_id,
        "subject": subject,
        "category": str(row.get("category") or subject),
        "question": str(question).strip(),
        "options": dict(options),
        "choice_labels": list(labels),
        "option_count": len(labels),
        "source_file": str(source_path),
        "source_row": line_number,
        "dataset_protocol": "mmlu_pro_jsonl_v1",
    }
    return DatasetExample(dataset="mmlu_pro", public_task=public_task, gold=gold)


def normalize_mmlu_pro_options(raw_options: Any) -> dict[str, str]:
    """Normalize mapping/list options to ordered A/B/C... labels."""

    if isinstance(raw_options, Mapping):
        options = {
            _normalize_label(label): str(value).strip()
            for label, value in raw_options.items()
            if _normalize_label(label) in OPTION_LABELS and str(value).strip()
        }
        return {
            label: options[label]
            for label in sorted(options, key=_label_sort_key)
        }

    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        if len(raw_options) > len(OPTION_LABELS):
            raise ValueError(f"too many MMLU-Pro options: {len(raw_options)}")
        return {
            label: str(value).strip()
            for label, value in zip(OPTION_LABELS, raw_options, strict=False)
            if str(value).strip()
        }

    return {}


def normalize_mmlu_pro_gold(
    row: Mapping[str, Any],
    options: Mapping[str, str],
) -> str:
    """Return the gold option label for a normalized MMLU-Pro row."""

    labels = tuple(options)
    for key in _INDEX_KEYS:
        if key in row and row[key] is not None and row[key] != "":
            return _label_from_index(row[key], labels)

    for key in _GOLD_KEYS:
        if key in row and row[key] is not None and row[key] != "":
            return extract_mmlu_pro_label(row[key], labels=labels, options=options)

    raise ValueError("MMLU-Pro row missing a gold answer field")


def extract_mmlu_pro_label(
    value: Any,
    *,
    labels: Sequence[str],
    options: Mapping[str, str] | None = None,
) -> str:
    """Extract a dynamic MMLU-Pro option label from common row fields."""

    if isinstance(value, int) and not isinstance(value, bool):
        return _label_from_index(value, labels)

    text = str(value).strip()
    if not text:
        raise ValueError("empty MMLU-Pro answer")
    upper = text.upper()
    if upper in labels:
        return upper

    if text.isdigit():
        return _label_from_index(int(text), labels)

    if options:
        normalized_text = _normalize_option_text(text)
        for label, option_text in options.items():
            if _normalize_option_text(option_text) == normalized_text:
                return label

    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    explicit = re.findall(
        rf"(?:final\s+answer|answer|option|choice|letter)\s*(?:is|:|\-)?\s*\(?({label_pattern})\)?",
        upper,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()

    leading = re.match(rf"^\s*\(?({label_pattern})\)?\s*[\.\):\-]", upper)
    if leading:
        return leading.group(1).upper()

    raise ValueError(f"cannot extract MMLU-Pro option from {value!r}")


def _jsonl_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("*.jsonl"))
    return [source] if source.suffix == ".jsonl" else []


def _read_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} expected one JSON object")
            yield line_number, row


def _raw_options(row: Mapping[str, Any]) -> Any:
    for key in ("options", "choices", "candidates"):
        value = row.get(key)
        if value:
            return value

    option_columns = {}
    for label in OPTION_LABELS:
        for key in (label, label.lower(), f"option_{label}", f"option_{label.lower()}"):
            if key in row and str(row[key]).strip():
                option_columns[label] = row[key]
                break
    return option_columns


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_label(label: Any) -> str:
    text = str(label).strip().upper()
    if len(text) == 1:
        return text
    match = re.search(r"([A-Z])$", text)
    return match.group(1) if match else text


def _label_sort_key(label: str) -> tuple[int, str]:
    return (OPTION_LABELS.index(label), label) if label in OPTION_LABELS else (999, label)


def _label_from_index(value: Any, labels: Sequence[str]) -> str:
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"MMLU-Pro option index must be an integer, got {value!r}") from error
    if 0 <= index < len(labels):
        return labels[index]
    raise ValueError(
        f"MMLU-Pro option index must be in [0, {len(labels) - 1}], got {index}"
    )


def _normalize_option_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())
