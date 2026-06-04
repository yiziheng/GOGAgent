"""Resumable MMLU training runner for label-free Organization GoG memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
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
from gogagent.policy.hierarchical_gnn import HierarchicalGNNPolicy
from gogagent.training.credit import TransitionCreditInput
from gogagent.training.learner import DQNStyleLearner
from gogagent.training.recorder import TrainingEpisodeRecorder
from gogagent.training.replay import DenseRewardBreakdown, ReplayTransition


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
    token_budget: int = 4096
    policy_checkpoint_in: Path | None = None
    policy_checkpoint_out: Path | None = None
    policy_learning_rate: float = 0.01
    policy_gamma: float = 0.9
    policy_epsilon: float = 0.05


class MMLUMemoryTrainer:
    """Train shared GoG experience memory from label-blind MMLU rollouts."""

    def __init__(self, config: MMLUTrainingConfig, backend: LLMBackend) -> None:
        if config.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if config.limit is not None and config.limit < 0:
            raise ValueError("limit must be non-negative")
        if config.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if config.policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be positive")
        if not 0.0 <= config.policy_gamma <= 1.0:
            raise ValueError("policy_gamma must be between 0 and 1")
        if not 0.0 <= config.policy_epsilon <= 1.0:
            raise ValueError("policy_epsilon must be between 0 and 1")
        self.config = config
        self.backend = backend
        self.adapter = MMLUAdapter()
        self.recorder = TrainingEpisodeRecorder(get_oracle("mmlu"))
        self.run_directory = config.artifact_root / config.run_id
        self.items_directory = self.run_directory / "items"
        self.memory_path = self.run_directory / "memory.json"
        self.policy_path = config.policy_checkpoint_out or (self.run_directory / "policy.pt")
        self.gog = OrganizationGoG()
        self.policy = HierarchicalGNNPolicy(epsilon=config.policy_epsilon)
        self.learner = DQNStyleLearner(
            self.policy,
            gamma=config.policy_gamma,
            learning_rate=config.policy_learning_rate,
        )

    def run(self) -> dict[str, Any]:
        self._prepare_run()
        examples = list(self._selected_examples())
        _write_json(
            self.run_directory / "manifest.json",
            {
                "updated_at": _now(),
                "config": _config_dict(self.config),
                "backend": dict(self.backend.describe()),
                "policy_checkpoint": str(self.policy_path),
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
        _save_policy(self.policy, self.policy_path)
        summary = _summarize(
            records,
            len(skipped),
            self.gog,
            self.memory_path,
            self.policy_path,
        )
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
        elif self.config.resume and _has_completed_items(self.items_directory):
            raise FileNotFoundError(
                f"cannot resume completed items without checkpoint: {self.memory_path}"
            )
        else:
            self.gog = OrganizationGoG()
            _save_memory(self.gog, self.memory_path)
        self.policy = self._load_policy()
        self.learner = DQNStyleLearner(
            self.policy,
            gamma=self.config.policy_gamma,
            learning_rate=self.config.policy_learning_rate,
        )
        _save_policy(self.policy, self.policy_path)

    def _load_policy(self) -> HierarchicalGNNPolicy:
        if self.config.policy_checkpoint_in is not None:
            policy = HierarchicalGNNPolicy.load(str(self.config.policy_checkpoint_in))
        elif self.config.resume and self.policy_path.exists():
            policy = HierarchicalGNNPolicy.load(str(self.policy_path))
        elif self.config.resume and _has_completed_items(self.items_directory):
            raise FileNotFoundError(
                f"cannot resume completed items without policy checkpoint: {self.policy_path}"
            )
        else:
            policy = HierarchicalGNNPolicy(epsilon=self.config.policy_epsilon)
        policy.epsilon = self.config.policy_epsilon
        return policy

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
                gog_archive=self.gog,
                policy=self.policy,
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
            policy_updates = self._train_policy_from_trace(
                trace,
                terminal_reward=training.terminal_reward,
            )
            _save_memory(updated_gog, self.memory_path)
            self.gog = updated_gog
            _save_policy(self.policy, self.policy_path)
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
                "policy_update_count": len(policy_updates),
                "policy_loss_mean": _mean(
                    update.loss for update in policy_updates
                ),
                "policy_checkpoint": str(self.policy_path),
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

    def _train_policy_from_trace(
        self,
        trace: list[dict[str, Any]],
        *,
        terminal_reward: float,
    ) -> list[Any]:
        snapshots = {
            snapshot.graph_id: snapshot
            for snapshot in (
                _snapshot_from_dict(_mapping(record, "graph"))
                for record in trace
                if record.get("event") == "snapshot"
            )
        }
        updates = []
        transitions = _replay_transitions_from_trace(trace)
        total = len(transitions)
        for index, transition in enumerate(transitions):
            graph = snapshots.get(transition.graph_id)
            next_graph = snapshots.get(transition.next_graph_id)
            if graph is None or next_graph is None:
                continue
            shaped_transition = replace(
                transition,
                reward=round(
                    transition.reward
                    + _terminal_action_bonus(terminal_reward, index, total),
                    6,
                ),
                done=index == total - 1,
            )
            updates.append(
                self.learner.train_one(
                    graph=graph,
                    next_graph=next_graph,
                    transition=shaped_transition,
                )
            )
        stop_transition = _stop_transition_from_trace(trace, terminal_reward)
        if stop_transition is not None:
            graph = snapshots.get(stop_transition.graph_id)
            if graph is not None:
                updates.append(
                    self.learner.train_one(
                        graph=graph,
                        next_graph=graph,
                        transition=stop_transition,
                    )
                )
        return updates

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


def _replay_transitions_from_trace(trace: list[dict[str, Any]]) -> tuple[ReplayTransition, ...]:
    transitions: list[ReplayTransition] = []
    for record in trace:
        if record.get("event") != "replay_transition":
            continue
        dense_reward = _dense_reward_from_dict(_mapping(record, "dense_reward"))
        transitions.append(
            ReplayTransition(
                graph_id=str(record["graph_id"]),
                next_graph_id=str(record["next_graph_id"]),
                action=MacroAction(str(record["action"])),
                reward=float(record["reward"]),
                done=bool(record.get("done", False)),
                state=_mapping(record, "state"),
                next_state=_mapping(record, "next_state"),
                action_mask={
                    str(action): bool(is_legal)
                    for action, is_legal in _mapping(record, "action_mask").items()
                },
                next_action_mask={
                    str(action): bool(is_legal)
                    for action, is_legal in _mapping(record, "next_action_mask").items()
                },
                dense_reward=dense_reward,
            )
        )
    return tuple(transitions)


def _stop_transition_from_trace(
    trace: list[dict[str, Any]],
    terminal_reward: float,
) -> ReplayTransition | None:
    terminal = next(
        (record for record in reversed(trace) if record.get("event") == "terminal"),
        None,
    )
    if terminal is None:
        return None
    src_graph_id = str(terminal.get("src_graph_id", ""))
    if not src_graph_id:
        return None
    stop_decision = None
    for record in reversed(trace):
        if record.get("event") != "policy_decision":
            continue
        decision = _mapping(record, "decision")
        if str(decision.get("action")) == MacroAction.STOP.value:
            stop_decision = record
            break
    if stop_decision is None:
        return None
    state = _mapping(stop_decision, "state")
    action_mask = {
        str(action): bool(is_legal)
        for action, is_legal in _mapping(stop_decision, "legal_action_mask").items()
    }
    return ReplayTransition(
        graph_id=src_graph_id,
        next_graph_id=src_graph_id,
        action=MacroAction.STOP,
        reward=_terminal_stop_reward(terminal_reward),
        done=True,
        state=state,
        next_state=state,
        action_mask=action_mask,
        next_action_mask={action: False for action in action_mask},
        dense_reward=_zero_dense_reward(),
    )


def _dense_reward_from_dict(data: Mapping[str, Any]) -> DenseRewardBreakdown:
    return DenseRewardBreakdown(
        visible_quality_delta=float(data["visible_quality_delta"]),
        issue_resolution_delta=float(data["issue_resolution_delta"]),
        token_penalty=float(data["token_penalty"]),
        call_penalty=float(data["call_penalty"]),
        complexity_penalty=float(data["complexity_penalty"]),
        step_penalty=float(data["step_penalty"]),
        reward=float(data["reward"]),
    )


def _zero_dense_reward() -> DenseRewardBreakdown:
    return DenseRewardBreakdown(
        visible_quality_delta=0.0,
        issue_resolution_delta=0.0,
        token_penalty=0.0,
        call_penalty=0.0,
        complexity_penalty=0.0,
        step_penalty=0.0,
        reward=0.0,
    )


def _terminal_action_bonus(terminal_reward: float, index: int, total: int) -> float:
    """Shape each graph edit by final correctness while favoring later edits."""

    if total <= 0:
        return 0.0
    progress = (index + 1) / total
    decay = 0.6 + 0.4 * progress
    base = 1.0 if terminal_reward > 0.0 else -0.5
    return round(base * decay, 6)


def _terminal_stop_reward(terminal_reward: float) -> float:
    """Directly teach STOP from final correctness."""

    return 0.8 if terminal_reward > 0.0 else -1.1


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
        nodes=tuple(_node_from_dict(node) for node in data.get("nodes", ())),
        edges=tuple(_edge_from_dict(edge) for edge in data.get("edges", ())),
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


def _node_from_dict(node: Mapping[str, Any]) -> NodeSpec:
    return NodeSpec(
        node_id=str(node["node_id"]),
        role=str(node["role"]),
        runner=str(node.get("runner", "openai_compatible")),
        profile=str(node.get("profile", "")),
        node_kind=str(node.get("node_kind", node.get("node_type", "atomic"))),
        module_type=str(node.get("module_type", "")),
        model_tier=str(node.get("model_tier", "standard")),
        input_ports=tuple(str(item) for item in node.get("input_ports", ())),
        output_ports=tuple(str(item) for item in node.get("output_ports", ())),
        internal_nodes=tuple(_node_from_dict(child) for child in node.get("internal_nodes", ())),
        internal_edges=tuple(_edge_from_dict(edge) for edge in node.get("internal_edges", ())),
        metadata=node.get("metadata", {}),
    )


def _edge_from_dict(edge: Mapping[str, Any]) -> EdgeSpec:
    return EdgeSpec(
        src=str(edge["src"]),
        dst=str(edge["dst"]),
        payload=str(edge.get("payload", "default")),
        edge_kind=str(edge.get("edge_kind", "exec")),
        metadata=edge.get("metadata", {}),
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


def _save_policy(policy: HierarchicalGNNPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    policy.save(str(temporary))
    temporary.replace(path)


def _summarize(
    records: list[dict[str, Any]],
    skipped_count: int,
    gog: OrganizationGoG,
    memory_path: Path,
    policy_path: Path,
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
        "policy_update_count": sum(
            int(record.get("policy_update_count", 0)) for record in completed
        ),
        "policy_loss_mean": _mean(
            float(record["policy_loss_mean"])
            for record in completed
            if record.get("policy_loss_mean") is not None
        ),
        "memory_experience_count": len(gog.experiences),
        "memory_snapshot_count": len(gog.snapshots),
        "memory_transition_count": len(gog.transitions),
        "total_used_tokens": sum(int(record.get("used_tokens", 0)) for record in completed),
        "total_llm_calls": sum(int(record.get("llm_calls", 0)) for record in completed),
        "per_subject": _group_accuracy(completed, "subject"),
        "per_subject_profile": _group_accuracy(completed, "subject_profile"),
        "memory_checkpoint": str(memory_path),
        "policy_checkpoint": str(policy_path),
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


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(items) / len(items), 6)


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
