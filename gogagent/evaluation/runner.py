"""Resumable batch evaluation for real benchmark files and real LLM backends."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from threading import Lock
from time import monotonic
from typing import Any, Iterable, Mapping

from gogagent.adapters.mmlu import subject_profile
from gogagent.adapters.registry import get_adapter
from gogagent.core.rollout import RolloutEngine
from gogagent.datasets import (
    DatasetExample,
    load_gsm8k_jsonl,
    load_humaneval_jsonl,
    load_mmlu_directory,
)
from gogagent.gog.memory import OrganizationGoG
from gogagent.llm.base import LLMBackend


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str
    data_path: Path
    artifact_root: Path
    run_id: str
    split: str = "test"
    workers: int = 1
    start_index: int = 0
    limit: int | None = None
    resume: bool = False
    gog_memory: Path | None = None


class BenchmarkRunner:
    """Run label-blind rollouts and score only after each rollout returns."""

    def __init__(self, config: EvaluationConfig, backend: LLMBackend) -> None:
        if config.workers < 1:
            raise ValueError("workers must be positive")
        self.config = config
        self.backend = backend
        self.run_directory = config.artifact_root / config.run_id
        self.items_directory = self.run_directory / "items"
        self._event_lock = Lock()

    def run(self) -> dict[str, Any]:
        self.items_directory.mkdir(parents=True, exist_ok=True)
        examples = list(self._selected_examples())
        _write_json(
            self.run_directory / "manifest.json",
            {
                "created_at": _now(),
                "config": _config_dict(self.config),
                "backend": self.backend.describe(),
                "label_boundary": "gold is scored only after RolloutEngine.run(public_task)",
            },
        )
        pending = []
        skipped = []
        for example in examples:
            item_directory = self.items_directory / _stable_item_key(example)
            if self.config.resume and _is_completed(item_directory):
                skipped.append(_read_json(item_directory / "result.json"))
                continue
            _write_status(item_directory, "pending")
            pending.append((example, item_directory))

        completed: list[dict[str, Any]] = []
        if self.config.workers == 1:
            completed.extend(self._run_one(example, item_directory) for example, item_directory in pending)
        else:
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                futures = {
                    executor.submit(self._run_one, example, item_directory): item_directory
                    for example, item_directory in pending
                }
                for future in as_completed(futures):
                    completed.append(future.result())

        records = skipped + completed
        summary = _summarize(self.config.dataset, records, skipped_count=len(skipped))
        _write_json(self.run_directory / "summary.json", summary)
        _write_summary_tsv(self.run_directory / "summary.tsv", summary)
        return summary

    def _run_one(self, example: DatasetExample, item_directory: Path) -> dict[str, Any]:
        started = monotonic()
        task_id = str(example.public_task.get("task_id", item_directory.name))
        _write_status(item_directory, "running")
        _write_json(item_directory / "input.json", dict(example.public_task))
        self._append_event("running", task_id, item_directory)
        _reset_rollout_directory(item_directory / "rollout")
        try:
            memory = OrganizationGoG.load(self.config.gog_memory) if self.config.gog_memory else None
            rollout = RolloutEngine(
                get_adapter(example.dataset),
                self.backend,
                artifact_root=self.run_directory / "rollouts",
                gog_memory=memory,
            ).run(
                example.public_task,
                episode_id=item_directory.name,
                artifact_directory=item_directory / "rollout",
            )
            correct = _score(example, rollout["final_output"])
            result = {
                "status": "completed",
                "dataset": example.dataset,
                "task_id": task_id,
                "subject": example.public_task.get("subject"),
                "subject_profile": (
                    subject_profile(str(example.public_task.get("subject", "")))
                    if example.dataset == "mmlu"
                    else None
                ),
                "prediction": rollout["final_output"],
                "correct": correct,
                "elapsed_seconds": round(monotonic() - started, 6),
                "used_tokens": rollout["used_tokens"],
                "llm_calls": rollout["llm_calls"],
                "artifact_directory": rollout["artifact_directory"],
            }
            _write_json(item_directory / "result.json", result)
            _write_status(item_directory, "completed")
            self._append_event("completed", task_id, item_directory)
            return result
        except Exception as error:  # Keep one bad item from aborting a benchmark.
            result = {
                "status": "failed",
                "dataset": example.dataset,
                "task_id": task_id,
                "correct": False,
                "elapsed_seconds": round(monotonic() - started, 6),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(item_directory / "error.json", result)
            _write_status(item_directory, "failed")
            self._append_event("failed", task_id, item_directory, error=str(error))
            return result

    def _selected_examples(self) -> Iterable[DatasetExample]:
        if self.config.dataset == "gsm8k":
            examples = load_gsm8k_jsonl(self.config.data_path)
        elif self.config.dataset == "mmlu":
            examples = load_mmlu_directory(self.config.data_path, self.config.split)
        elif self.config.dataset == "humaneval":
            examples = load_humaneval_jsonl(self.config.data_path)
        else:
            raise ValueError(f"unsupported dataset {self.config.dataset!r}")
        selected = list(examples)[self.config.start_index :]
        return selected[: self.config.limit] if self.config.limit is not None else selected

    def _append_event(
        self,
        status: str,
        task_id: str,
        item_directory: Path,
        **extra: Any,
    ) -> None:
        record = {
            "timestamp": _now(),
            "status": status,
            "task_id": task_id,
            "item_directory": str(item_directory),
            **extra,
        }
        with self._event_lock:
            with (self.run_directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _score(example: DatasetExample, output: str) -> bool:
    # Deliberately imported here: inference modules do not depend on train/eval oracles.
    from gogagent.oracle.registry import get_oracle

    return bool(get_oracle(example.dataset).score(example.public_task, output, example.gold))


def _stable_item_key(example: DatasetExample) -> str:
    task_id = str(example.public_task.get("task_id", "unknown"))
    safe_id = "".join(character if character.isalnum() or character in "-_." else "-" for character in task_id)
    digest = sha256(
        json.dumps(dict(example.public_task), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"{safe_id[:80]}-{digest}"


def _is_completed(item_directory: Path) -> bool:
    result_path = item_directory / "result.json"
    return result_path.exists() and _read_json(result_path).get("status") == "completed"


def _summarize(dataset: str, records: list[dict[str, Any]], skipped_count: int) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    correct = sum(bool(record.get("correct")) for record in completed)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "total": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "skipped_completed": skipped_count,
        "correct": correct,
        "accuracy": round(correct / len(records), 6) if records else None,
        "completed_accuracy": round(correct / len(completed), 6) if completed else None,
        "total_used_tokens": sum(int(record.get("used_tokens", 0)) for record in completed),
        "total_llm_calls": sum(int(record.get("llm_calls", 0)) for record in completed),
        "updated_at": _now(),
    }
    if dataset == "mmlu":
        summary["per_subject"] = _group_accuracy(completed, "subject")
        summary["per_subject_profile"] = _group_accuracy(completed, "subject_profile")
    return summary


def _group_accuracy(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(bool(record.get("correct")))
    return {
        group: {
            "count": len(values),
            "correct": sum(values),
            "accuracy": round(sum(values) / len(values), 6),
        }
        for group, values in sorted(groups.items())
    }


def _write_status(item_directory: Path, status: str) -> None:
    item_directory.mkdir(parents=True, exist_ok=True)
    _write_json(item_directory / "status.json", {"status": status, "updated_at": _now()})


def _reset_rollout_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_dict(config: EvaluationConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_summary_tsv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = ["metric\tvalue"]
    for key in ("dataset", "total", "completed", "failed", "skipped_completed", "correct", "accuracy", "completed_accuracy", "total_used_tokens", "total_llm_calls"):
        rows.append(f"{key}\t{summary.get(key)}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
