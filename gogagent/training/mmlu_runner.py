"""Resumable MMLU training runner for label-free Organization GoG memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Any, Iterable, Mapping

from gogagent.adapters.mmlu import MMLUAdapter, subject_profile
from gogagent.core.actions import MacroAction
from gogagent.core.rollout import RolloutEngine
from gogagent.core.types import EdgeSpec, NodeSpec, OrgGraphSnapshot, TransitionEdge
from gogagent.datasets import DatasetExample, load_mmlu_directory
from gogagent.gog.memory import OrganizationGoG
from gogagent.llm.base import LLMBackend
from gogagent.oracle.registry import get_oracle
from gogagent.training.credit import TransitionCreditInput
from gogagent.training.recorder import TrainingEpisodeRecorder


_STATUS_VALUES = {
    "failed": -0.25,
    "needs_review": 0.0,
    "unknown": 0.0,
    "ready": 0.25,
    "passed": 0.25,
}


@dataclass(frozen=True)
class MMLUTrainingConfig:
    """Filesystem and slicing controls for one real MMLU dev training run."""

    data_path: Path
    artifact_root: Path
    run_id: str
    split: str = "dev"
    start_index: int = 0
    limit: int | None = None
    resume: bool = False
    gog_memory: Path | None = None
    token_budget: int = 4096


class MMLUMemoryTrainer:
    """Train shared GoG experience memory from label-blind MMLU rollouts."""

    def __init__(self, config: MMLUTrainingConfig, backend: LLMBackend) -> None:
        if config.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if config.limit is not None and config.limit < 0:
            raise ValueError("limit must be non-negative")
        if config.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.config = config
        self.backend = backend
        self.adapter = MMLUAdapter()
        self.recorder = TrainingEpisodeRecorder(get_oracle("mmlu"))
        self.run_directory = config.artifact_root / config.run_id
        self.items_directory = self.run_directory / "items"
        self.memory_path = self.run_directory / "memory.json"
        self.gog = OrganizationGoG()

    def run(self) -> dict[str, Any]:
        self._prepare_run()
        examples = list(self._selected_examples())
        _write_json(
            self.run_directory / "manifest.json",
            {
                "updated_at": _now(),
                "config": _config_dict(self.config),
                "backend": dict(self.backend.describe()),
                "label_boundary": (
                    "RolloutEngine.run receives public_task only; gold enters "
                    "TrainingEpisodeRecorder.record only after rollout returns"
                ),
            },
        )

        skipped: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []
        for example in examples:
            item_directory = self.items_directory / _stable_item_key(example)
            task_id = str(example.public_task.get("task_id", item_directory.name))
            if self.config.resume and _is_completed(item_directory):
                result = _read_json(item_directory / "result.json")
                skipped.append(result)
                self._append_event("skipped_completed", task_id, item_directory)
                continue
            _write_status(item_directory, "pending", task_id=task_id)
            self._append_event("pending", task_id, item_directory)
            completed.append(self._run_one(example, item_directory))

        records = skipped + completed
        summary = _summarize(records, len(skipped), self.gog, self.memory_path)
        _write_json(self.run_directory / "train_summary.json", summary)
        return summary

    def _prepare_run(self) -> None:
        if self.run_directory.exists() and not self.config.resume:
            if any(self.run_directory.iterdir()):
                raise FileExistsError(
                    f"training run directory already exists: {self.run_directory}; "
                    "choose a new run_id or set resume=True"
                )
        self.items_directory.mkdir(parents=True, exist_ok=True)
        if self.config.resume and self.memory_path.exists():
            self.gog = OrganizationGoG.load(self.memory_path)
            return
        if self.config.resume and _has_completed_items(self.items_directory):
            raise FileNotFoundError(
                f"cannot resume completed items without checkpoint: {self.memory_path}"
            )
        self.gog = (
            OrganizationGoG.load(self.config.gog_memory)
            if self.config.gog_memory is not None
            else OrganizationGoG()
        )
        _save_memory(self.gog, self.memory_path)

    def _run_one(
        self,
        example: DatasetExample,
        item_directory: Path,
    ) -> dict[str, Any]:
        started = monotonic()
        task_id = str(example.public_task.get("task_id", item_directory.name))
        _write_status(item_directory, "running", task_id=task_id)
        _write_json(item_directory / "input.json", dict(example.public_task))
        self._append_event("running", task_id, item_directory)
        _reset_rollout_directory(item_directory / "rollout")
        try:
            rollout = RolloutEngine(
                self.adapter,
                self.backend,
                artifact_root=self.run_directory / "rollouts",
                gog_memory=self.gog,
                token_budget=self.config.token_budget,
            ).run(
                example.public_task,
                episode_id=item_directory.name,
                artifact_directory=item_directory / "rollout",
            )
            trace_path = Path(rollout["artifact_directory"]) / "trace.jsonl"
            trace = _read_trace(trace_path)
            steps = _credit_steps_from_trace(trace)
            updated_gog = self.gog.fork_for_rollout()
            _merge_trace_snapshots(updated_gog, self.adapter, trace)
            training = self.recorder.record(
                gog=updated_gog,
                task=example.public_task,
                task_features=self.adapter.task_features(example.public_task),
                output=str(rollout["final_output"]),
                gold=example.gold,
                steps=steps,
            )
            _save_memory(updated_gog, self.memory_path)
            self.gog = updated_gog
            result = {
                "status": "completed",
                "dataset": "mmlu",
                "task_id": task_id,
                "subject": example.public_task.get("subject"),
                "subject_profile": subject_profile(
                    str(example.public_task.get("subject", ""))
                ),
                "prediction": rollout["final_output"],
                "terminal_reward": training.terminal_reward,
                "correct": training.terminal_reward > 0.0,
                "credit_count": training.experience_count,
                "memory_experience_count": len(self.gog.experiences),
                "elapsed_seconds": round(monotonic() - started, 6),
                "used_tokens": rollout["used_tokens"],
                "llm_calls": rollout["llm_calls"],
                "artifact_directory": rollout["artifact_directory"],
                "memory_checkpoint": str(self.memory_path),
            }
            _write_json(item_directory / "result.json", result)
            _write_status(item_directory, "completed", task_id=task_id)
            self._append_event(
                "completed",
                task_id,
                item_directory,
                credit_count=training.experience_count,
            )
            return result
        except Exception as error:  # Preserve progress when one real item is malformed.
            result = {
                "status": "failed",
                "dataset": "mmlu",
                "task_id": task_id,
                "elapsed_seconds": round(monotonic() - started, 6),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(item_directory / "result.json", result)
            _write_status(item_directory, "failed", task_id=task_id)
            self._append_event("failed", task_id, item_directory, error=str(error))
            return result

    def _selected_examples(self) -> Iterable[DatasetExample]:
        examples = list(load_mmlu_directory(self.config.data_path, split=self.config.split))
        selected = examples[self.config.start_index :]
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
        with (self.run_directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _credit_steps_from_trace(
    trace: list[dict[str, Any]],
) -> tuple[TransitionCreditInput, ...]:
    """Convert selected edits into credit inputs using public trace signals only."""

    steps: list[TransitionCreditInput] = []
    current_graph_id: str | None = None
    for index, record in enumerate(trace):
        event = record.get("event")
        if event == "snapshot":
            current_graph_id = str(_mapping(record, "graph")["graph_id"])
            continue
        if event != "policy_decision":
            continue
        action = MacroAction(str(_mapping(record, "decision")["action"]))
        if action is MacroAction.STOP:
            continue
        successor = _successor_snapshot(trace, index)
        transition = successor.get("transition", {})
        if not isinstance(transition, Mapping):
            raise ValueError("successor snapshot transition must be an object")
        src_graph_id = str(transition.get("src_graph_id") or current_graph_id or "")
        if not src_graph_id:
            raise ValueError("cannot infer policy source graph_id from trace")
        if current_graph_id is not None and src_graph_id != current_graph_id:
            raise ValueError(
                f"trace transition source {src_graph_id!r} does not match "
                f"policy graph {current_graph_id!r}"
            )
        transition_action = transition.get("action")
        if transition_action is not None and str(transition_action) != action.value:
            raise ValueError(
                f"trace transition action {transition_action!r} does not match "
                f"policy action {action.value!r}"
            )
        execution = _mapping(successor, "execution")
        successor_feedback = _mapping(execution, "visible_feedback")
        source_feedback = _mapping(_mapping(record, "state"), "observable_feedback")
        steps.append(
            TransitionCreditInput(
                graph_id=src_graph_id,
                action=action,
                token_cost=int(execution.get("token_cost", 0)),
                feedback_type=str(successor_feedback.get("status", "unknown")),
                visible_delta=_visible_delta(source_feedback, successor_feedback),
            )
        )
    return tuple(steps)


def _successor_snapshot(
    trace: list[dict[str, Any]],
    policy_index: int,
) -> dict[str, Any]:
    for record in trace[policy_index + 1 :]:
        if record.get("event") == "snapshot":
            return record
        if record.get("event") == "policy_decision":
            break
    raise ValueError("non-STOP policy action is missing its successor snapshot")


def _visible_delta(
    source_feedback: Mapping[str, Any],
    successor_feedback: Mapping[str, Any],
) -> float:
    source = _STATUS_VALUES.get(str(source_feedback.get("status", "unknown")), 0.0)
    successor = _STATUS_VALUES.get(str(successor_feedback.get("status", "unknown")), 0.0)
    return round(successor - source, 6)


def _merge_trace_snapshots(
    gog: OrganizationGoG,
    adapter: MMLUAdapter,
    trace: list[dict[str, Any]],
) -> None:
    """Merge rollout-local graph nodes so recorded experiences stay reusable."""

    for record in trace:
        if record.get("event") != "snapshot":
            continue
        snapshot = _snapshot_from_dict(_mapping(record, "graph"))
        if snapshot.graph_id in gog.snapshots:
            continue
        transition_data = record.get("transition")
        transition = (
            _transition_from_dict(transition_data)
            if isinstance(transition_data, Mapping)
            else None
        )
        gog.add_snapshot(snapshot, adapter.signature(snapshot), transition)


def _snapshot_from_dict(data: Mapping[str, Any]) -> OrgGraphSnapshot:
    created_by = data.get("created_by")
    return OrgGraphSnapshot(
        graph_id=str(data["graph_id"]),
        domain=str(data["domain"]),
        step=int(data["step"]),
        nodes=tuple(
            NodeSpec(
                node_id=str(node["node_id"]),
                role=str(node["role"]),
                runner=str(node.get("runner", "openai_compatible")),
                profile=str(node.get("profile", "")),
                metadata=node.get("metadata", {}),
            )
            for node in data.get("nodes", ())
        ),
        edges=tuple(
            EdgeSpec(
                src=str(edge["src"]),
                dst=str(edge["dst"]),
                payload=str(edge.get("payload", "default")),
            )
            for edge in data.get("edges", ())
        ),
        parent_graph_id=(
            str(data["parent_graph_id"]) if data.get("parent_graph_id") is not None else None
        ),
        created_by=MacroAction(str(created_by)) if created_by is not None else None,
        metadata=data.get("metadata", {}),
    )


def _transition_from_dict(data: Mapping[str, Any]) -> TransitionEdge:
    return TransitionEdge(
        src_graph_id=str(data["src_graph_id"]),
        dst_graph_id=str(data["dst_graph_id"]),
        action=MacroAction(str(data["action"])),
    )


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"trace field {key!r} must be an object")
    return value


def _stable_item_key(example: DatasetExample) -> str:
    task_id = str(example.public_task.get("task_id", "unknown"))
    safe_id = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in task_id
    )
    digest = sha256(
        json.dumps(dict(example.public_task), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"{safe_id[:80]}-{digest}"


def _is_completed(item_directory: Path) -> bool:
    result_path = item_directory / "result.json"
    return result_path.exists() and _read_json(result_path).get("status") == "completed"


def _has_completed_items(items_directory: Path) -> bool:
    return any(_is_completed(item_directory) for item_directory in items_directory.glob("*"))


def _reset_rollout_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def _save_memory(gog: OrganizationGoG, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    gog.save(temporary)
    temporary.replace(path)


def _summarize(
    records: list[dict[str, Any]],
    skipped_count: int,
    gog: OrganizationGoG,
    memory_path: Path,
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    correct = sum(bool(record.get("correct")) for record in completed)
    return {
        "dataset": "mmlu",
        "total": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "skipped_completed": skipped_count,
        "correct": correct,
        "accuracy": round(correct / len(records), 6) if records else None,
        "completed_accuracy": round(correct / len(completed), 6) if completed else None,
        "episode_credit_count": sum(int(record.get("credit_count", 0)) for record in completed),
        "memory_experience_count": len(gog.experiences),
        "memory_snapshot_count": len(gog.snapshots),
        "memory_transition_count": len(gog.transitions),
        "total_used_tokens": sum(int(record.get("used_tokens", 0)) for record in completed),
        "total_llm_calls": sum(int(record.get("llm_calls", 0)) for record in completed),
        "per_subject": _group_accuracy(completed, "subject"),
        "per_subject_profile": _group_accuracy(completed, "subject_profile"),
        "memory_checkpoint": str(memory_path),
        "updated_at": _now(),
    }


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


def _write_status(item_directory: Path, status: str, **extra: Any) -> None:
    item_directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        item_directory / "status.json",
        {"status": status, "updated_at": _now(), **extra},
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _config_dict(config: MMLUTrainingConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
