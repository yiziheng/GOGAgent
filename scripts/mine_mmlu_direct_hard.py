#!/usr/bin/env python3
"""Mine MMLU validation examples that DeepSeek direct answers incorrectly."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.config.env import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT_SECONDS,
    load_project_env,
    require_env,
)
from gogagent.datasets.prompt_specs import (  # noqa: E402
    MMLU_DIRECT_SYSTEM_PROMPT,
    format_mmlu_direct_task,
)
from gogagent.reward.oracle import score_answer  # noqa: E402


@dataclass(frozen=True)
class MMLURow:
    """One MMLU CSV row with source metadata."""

    subject: str
    split: str
    source_file: Path
    source_row: int
    values: tuple[str, str, str, str, str, str]

    @property
    def question(self) -> str:
        return self.values[0]

    @property
    def options(self) -> dict[str, str]:
        return dict(zip(("A", "B", "C", "D"), self.values[1:5], strict=True))

    @property
    def answer(self) -> str:
        return self.values[5]

    @property
    def task_id(self) -> str:
        return f"{self.subject}-{self.split}-{self.source_row}"

    @property
    def content_key(self) -> str:
        return stable_content_key(self.question, self.options, self.answer)

    def public_task(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "question": self.question,
            "options": self.options,
            "source_file": relative_path(self.source_file),
            "source_row": self.source_row,
        }


def main() -> None:
    args = parse_args()
    load_project_env(args.env)

    source_dir = resolve_source_dir(args.source_dir, args.split)
    output_root = args.output_root.resolve()
    prepare_output_root(output_root, overwrite=args.overwrite, resume=args.resume)

    excluded_keys = load_excluded_content_keys(args.exclude_selection_jsonl)
    rows = [
        row
        for row in load_ordered_rows(source_dir, split=args.split)
        if row.content_key not in excluded_keys
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no rows left after applying exclusions and limit")
    if args.dry_run:
        summary = {
            "dry_run": True,
            "source_dir": str(source_dir),
            "source_split": args.split,
            "excluded_selection_jsonl": str(args.exclude_selection_jsonl),
            "excluded_content_count": len(excluded_keys),
            "total_to_call": len(rows),
            "output_root": str(output_root),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    results_path = output_root / "results.jsonl"
    existing_results = load_existing_results(results_path) if args.resume else {}
    client = OpenAI(
        api_key=require_env(args.api_key_env),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    total_usage = usage_from_results(existing_results.values())
    rows_by_key = {row.content_key: row for row in rows}
    pending_rows = [row for row in rows if row.content_key not in existing_results]
    completed_rows: list[dict[str, Any]] = [
        existing_results[row.content_key]
        for row in rows
        if row.content_key in existing_results
    ]

    iterator = tqdm(
        pending_rows,
        total=len(pending_rows),
        desc="Mine MMLU direct hard",
        unit="item",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for index, row in enumerate(iterator, start=len(completed_rows) + 1):
        result = run_direct_call(
            client=client,
            row=row,
            index=index,
            model=args.model,
            temperature=args.temperature,
            thinking=args.thinking,
        )
        append_jsonl(results_path, result)
        completed_rows.append(result)
        for key in total_usage:
            total_usage[key] += int(result.get("usage", {}).get(key, 0) or 0)
        if result["status"] == "error" and not args.continue_on_error:
            raise RuntimeError(f"DeepSeek request failed at {result['task_id']}: {result['error']}")
        iterator.set_postfix(
            {
                "acc": f"{running_accuracy(completed_rows):.3f}",
                "hard": count_hard(completed_rows),
                "errors": count_errors(completed_rows),
            }
        )
        progress = {
            "task_id": result["task_id"],
            "subject": result["subject"],
            "prediction": result.get("prediction"),
            "gold": result.get("gold"),
            "correct": result.get("correct"),
            "status": result["status"],
        }
        if args.no_progress:
            print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
        else:
            tqdm.write(json.dumps(progress, ensure_ascii=False, sort_keys=True))

    refreshed_results = load_existing_results(results_path)
    ordered_results = [refreshed_results.get(row.content_key) for row in rows]
    final_results = [result for result in ordered_results if result is not None]
    write_subsets(
        output_root,
        final_results,
        rows_by_key=rows_by_key,
        output_split=args.output_split,
    )
    summary = summarize(
        rows=final_results,
        source_dir=source_dir,
        output_root=output_root,
        args=args,
        excluded_count=len(excluded_keys),
        total_usage=total_usage,
    )
    write_json(output_root / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing original MMLU *_val.csv files.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-split", default="val")
    parser.add_argument(
        "--exclude-selection-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "gptswarm153" / "selection.jsonl",
        help="Selection JSONL to exclude, usually GPTSwarm153.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "direct_hard_val",
    )
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--api-key-env", default="GOGAGENT_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled", "none"),
        default=DEFAULT_THINKING,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without calling the API")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def run_direct_call(
    *,
    client: OpenAI,
    row: MMLURow,
    index: int,
    model: str,
    temperature: float,
    thinking: str,
) -> dict[str, Any]:
    public_task = row.public_task()
    prompt = format_mmlu_direct_task(public_task)
    started_at = time.monotonic()
    base = {
        "index": index,
        "content_key": row.content_key,
        "task_id": row.task_id,
        "subject": row.subject,
        "source_split": row.split,
        "source_file": relative_path(row.source_file),
        "source_row": row.source_row,
        "question": row.question,
        "options": row.options,
        "gold": row.answer,
        "row_values": list(row.values),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MMLU_DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            **extra_body_for_thinking(thinking),
        )
    except (APIConnectionError, APITimeoutError, APIError) as error:
        return {
            **base,
            "status": "error",
            "error": f"{type(error).__name__}: {safe_error_message(error)}",
            "prediction": None,
            "correct": None,
            "latency_seconds": round(time.monotonic() - started_at, 6),
            "usage": zero_usage(),
        }

    payload = response.model_dump(mode="json")
    text = extract_text(payload)
    oracle = score_answer("mmlu", public_task, {"answer": text}, gold=row.answer)
    return {
        **base,
        "status": "completed",
        "prediction": text.strip(),
        "exact_letter": text.strip() in {"A", "B", "C", "D"},
        "correct": oracle.correct,
        "oracle_reason": oracle.reason,
        "model": payload.get("model", model),
        "prompt_style": "direct",
        "system_prompt": MMLU_DIRECT_SYSTEM_PROMPT,
        "latency_seconds": round(time.monotonic() - started_at, 6),
        "usage": usage_to_dict(payload.get("usage", {})),
    }


def write_subsets(
    output_root: Path,
    rows: list[Mapping[str, Any]],
    *,
    rows_by_key: Mapping[str, MMLURow],
    output_split: str,
) -> None:
    hard = [row for row in rows if row.get("status") == "completed" and row.get("correct") is False]
    easy = [row for row in rows if row.get("status") == "completed" and row.get("correct") is True]
    errors = [row for row in rows if row.get("status") == "error"]
    write_result_subset(output_root / "hard", hard, rows_by_key=rows_by_key, output_split=output_split)
    write_result_subset(output_root / "easy", easy, rows_by_key=rows_by_key, output_split=output_split)
    write_result_subset(output_root / "error", errors, rows_by_key=rows_by_key, output_split=output_split)


def write_result_subset(
    subset_root: Path,
    results: list[Mapping[str, Any]],
    *,
    rows_by_key: Mapping[str, MMLURow],
    output_split: str,
) -> None:
    if subset_root.exists():
        shutil.rmtree(subset_root)
    split_dir = subset_root / output_split
    split_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[MMLURow]] = {}
    for result in results:
        row = rows_by_key.get(str(result.get("content_key", "")))
        if row is not None:
            grouped.setdefault(row.subject, []).append(row)
    for subject, subject_rows in sorted(grouped.items()):
        output_path = split_dir / f"{subject}_{output_split}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(row.values for row in subject_rows)
    write_jsonl(subset_root / "selection.jsonl", list(results))


def summarize(
    *,
    rows: list[Mapping[str, Any]],
    source_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    excluded_count: int,
    total_usage: Mapping[str, int],
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    correct = [row for row in completed if row.get("correct") is True]
    hard = [row for row in completed if row.get("correct") is False]
    errors = [row for row in rows if row.get("status") == "error"]
    return {
        "name": output_root.name,
        "purpose": "MMLU validation examples outside GPTSwarm153 mined by one DS direct call.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "source_split": args.split,
        "output_root": str(output_root),
        "output_split": args.output_split,
        "excluded_selection_jsonl": str(args.exclude_selection_jsonl),
        "excluded_content_count": excluded_count,
        "total": len(rows),
        "completed": len(completed),
        "failed": len(errors),
        "correct": len(correct),
        "hard": len(hard),
        "accuracy": round(len(correct) / len(completed), 6) if completed else None,
        "backend": {
            "base_url": args.base_url,
            "model": args.model,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "api_key_configured": True,
        },
        "usage": dict(total_usage),
        "results_jsonl": str(output_root / "results.jsonl"),
        "hard_dir": str(output_root / "hard"),
        "easy_dir": str(output_root / "easy"),
        "error_dir": str(output_root / "error"),
        "hard_selection_jsonl": str(output_root / "hard" / "selection.jsonl"),
        "easy_selection_jsonl": str(output_root / "easy" / "selection.jsonl"),
        "error_selection_jsonl": str(output_root / "error" / "selection.jsonl"),
        "system_prompt": MMLU_DIRECT_SYSTEM_PROMPT,
    }


def resolve_source_dir(source_dir: Path | None, split: str) -> Path:
    candidates = [source_dir] if source_dir is not None else [
        REPO_ROOT / "data" / "MMLU" / "data" / split,
        REPO_ROOT / "data" / "MMLU" / split,
        REPO_ROOT / "data" / "MMLU" / "data" / "validation",
        REPO_ROOT / "data" / "MMLU" / "validation",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists() and list(resolved.glob(f"*_{split}.csv")):
            return resolved
    searched = "\n".join(str(path) for path in candidates if path is not None)
    raise RuntimeError(f"could not find MMLU *_{split}.csv files. Searched:\n{searched}")


def load_ordered_rows(source_dir: Path, *, split: str) -> list[MMLURow]:
    rows: list[MMLURow] = []
    suffix = f"_{split}.csv"
    source_files = sorted(source_dir.glob(f"*{suffix}"))
    if not source_files:
        raise RuntimeError(f"no MMLU files found in {source_dir} for split={split!r}")
    for path in source_files:
        subject = subject_name(path, split)
        with path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 6:
                    raise ValueError(f"{path}:{row_index} expected 6 columns, got {len(row)}")
                rows.append(
                    MMLURow(
                        subject=subject,
                        split=split,
                        source_file=path,
                        source_row=row_index,
                        values=tuple(str(value) for value in row[:6]),  # type: ignore[arg-type]
                    )
                )
    return rows


def load_excluded_content_keys(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} expected JSON object")
            options = row.get("options") or {}
            if not isinstance(options, Mapping):
                raise ValueError(f"{path}:{line_number} expected options object")
            keys.add(stable_content_key(row.get("question", ""), options, row.get("answer", "")))
    return keys


def stable_content_key(question: Any, options: Mapping[str, Any], answer: Any) -> str:
    payload = {
        "question": str(question),
        "options": {
            label: str(options.get(label, options.get(label.lower(), "")))
            for label in ("A", "B", "C", "D")
        },
        "answer": str(answer),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} expected JSON object")
            key = str(row.get("content_key", ""))
            if key:
                results[key] = row
    return results


def prepare_output_root(output_root: Path, *, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_root.exists():
        if resume:
            return
        if not overwrite:
            raise RuntimeError(f"output root already exists: {output_root}")
        if output_root == REPO_ROOT or REPO_ROOT not in output_root.parents:
            raise RuntimeError(f"refusing to overwrite unsafe output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def extra_body_for_thinking(thinking: str) -> dict[str, Any]:
    if thinking == "none":
        return {}
    return {"extra_body": {"thinking": {"type": thinking}}}


def extract_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("chat completions response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("chat completions message content is not a string")
    return content


def usage_to_dict(usage: Any) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        return zero_usage()
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def usage_from_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = zero_usage()
    for row in rows:
        usage = row.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def running_accuracy(rows: list[Mapping[str, Any]]) -> float:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return 0.0
    return sum(1 for row in completed if row.get("correct") is True) / len(completed)


def count_hard(rows: list[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("status") == "completed" and row.get("correct") is False
    )


def count_errors(rows: list[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("status") == "error")


def subject_name(path: Path, split: str) -> str:
    suffix = f"_{split}.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected MMLU filename for split={split!r}: {path.name}")
    return path.name[: -len(suffix)]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_error_message(error: Exception) -> str:
    return str(error).replace("\n", " ")[:500]


if __name__ == "__main__":
    main()
