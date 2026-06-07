"""File writers and summary helpers for BC trajectory artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from gogagent.artifacts import write_json, write_jsonl


class JsonlWriter:
    """Streaming JSONL writer with a row counter."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.count = 0
        self._handle: Any | None = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, row: Mapping[str, Any]) -> None:
        """Write one JSONL row."""

        if self._handle is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        self._handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
        self._handle.write("\n")
        self.count += 1


@dataclass
class TrajectorySummaryAccumulator:
    """Streaming summary accumulator for BC trajectory generation."""

    run_dir: str | Path
    total_tasks: int = 0
    total_trajectories: int = 0
    valid_trajectories: int = 0
    invalid_trajectories: int = 0
    valid_step_count: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    invalid_action_counts: Counter[str] = field(default_factory=Counter)
    style_counts: Counter[str] = field(default_factory=Counter)

    def observe_task(self) -> None:
        """Count one loaded dataset task."""

        self.total_tasks += 1

    def observe_trajectory(self, row: Mapping[str, Any]) -> None:
        """Update aggregate trajectory counters."""

        self.total_trajectories += 1
        self.style_counts[str(row.get("style", ""))] += 1
        if row.get("valid"):
            self.valid_trajectories += 1
        else:
            self.invalid_trajectories += 1
        for action in row.get("actions", []):
            self.action_counts[str(action)] += 1
        for invalid_step in row.get("invalid_steps", []):
            action = invalid_step.get("action")
            self.invalid_action_counts[str(action)] += 1

    def observe_steps(self, count: int) -> None:
        """Count valid step-level BC rows."""

        self.valid_step_count += count

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate stats as a JSON-serializable object."""

        return {
            "run_dir": str(self.run_dir),
            "total_tasks": self.total_tasks,
            "total_trajectories": self.total_trajectories,
            "valid_trajectories": self.valid_trajectories,
            "invalid_trajectories": self.invalid_trajectories,
            "action_distribution": dict(sorted(self.action_counts.items())),
            "invalid_action_distribution": dict(
                sorted(self.invalid_action_counts.items())
            ),
            "style_distribution": dict(sorted(self.style_counts.items())),
            "valid_step_count": self.valid_step_count,
        }


def summarize_trajectories(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Return aggregate stats for trajectory generation."""

    accumulator = TrajectorySummaryAccumulator(run_dir=run_dir)
    task_ids = set()
    for row in rows:
        task_ids.add(str(row.get("task_id", "")))
        accumulator.observe_trajectory(row)
    accumulator.total_tasks = len(task_ids)
    return accumulator.to_dict()
