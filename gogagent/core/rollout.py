"""End-to-end label-blind graph construction rollout."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from gogagent.adapters.base import DomainAdapter
from gogagent.core.actions import MacroAction
from gogagent.core.compiler import MacroCompiler
from gogagent.core.constraint_engine import ConstraintEngine
from gogagent.core.executor import IncrementalExecutor
from gogagent.core.supervisor import SupervisorAgent
from gogagent.core.types import TransitionEdge
from gogagent.gog.memory import OrganizationGoG
from gogagent.gog.visualization import export_gog, export_snapshot
from gogagent.llm.base import LLMBackend
from gogagent.policy.features import compressed_state
from gogagent.policy.hierarchical_gnn import HierarchicalGNNPolicy
from gogagent.training.replay import DenseConstructionReward, ReplayTransition


class RolloutEngine:
    """Construct one task-specific DAG while persisting every visible graph artifact."""

    def __init__(
        self,
        adapter: DomainAdapter,
        llm: LLMBackend,
        artifact_root: str | Path = "artifacts/runs",
        constraints: ConstraintEngine | None = None,
        policy: Any | None = None,
        supervisor: SupervisorAgent | None = None,
        gog_archive: OrganizationGoG | None = None,
        token_budget: int = 4096,
        dense_reward: DenseConstructionReward | None = None,
    ) -> None:
        self.adapter = adapter
        self.llm = llm
        self.artifact_root = Path(artifact_root)
        self.constraints = constraints or ConstraintEngine()
        self.policy = policy or HierarchicalGNNPolicy()
        self.supervisor = supervisor or SupervisorAgent()
        self.gog_archive = gog_archive or OrganizationGoG()
        self.token_budget = token_budget
        self.dense_reward = dense_reward or DenseConstructionReward()

    def run(
        self,
        task: Mapping[str, Any],
        episode_id: str | None = None,
        artifact_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        episode_id = episode_id or uuid4().hex[:10]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            Path(artifact_directory)
            if artifact_directory is not None
            else self.artifact_root / run_id / self.adapter.name / episode_id
        )
        snapshot_directory = directory / "snapshots"
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        trace_path = directory / "trace.jsonl"

        gog = self.gog_archive.fork_for_rollout()
        history_snapshot_count = len(gog.snapshots)
        history_transition_count = len(gog.transitions)
        compiler = MacroCompiler(self.adapter, self.constraints)
        executor = IncrementalExecutor(self.adapter, self.llm)
        graph = _with_runner(self.adapter.base_graph(task), _backend_name(self.llm))
        self.constraints.validate(graph)
        execution = executor.execute(graph, task)
        used_tokens = execution.token_cost
        summary = self.supervisor.summarize(execution, used_tokens, self.token_budget)
        gog.add_snapshot(graph, self.adapter.signature(graph))
        episode_graph_ids = [graph.graph_id]
        export_snapshot(graph, snapshot_directory)
        _append_trace(
            trace_path,
            "snapshot",
            {
                "graph": graph.to_dict(),
                "execution": execution.to_dict(),
                "supervisor": summary.to_dict(),
            },
        )

        while True:
            candidates = self.constraints.legal_candidates(graph, execution.visible_feedback)
            if used_tokens >= self.token_budget:
                candidates = tuple(candidate for candidate in candidates if candidate.action is MacroAction.STOP)
            state = compressed_state(
                self.adapter.task_features(task),
                self.adapter.signature(graph),
                execution.visible_feedback,
                summary,
                used_tokens,
                self.token_budget,
            )
            action_mask = {
                action.value: is_legal
                for action, is_legal in self.constraints.action_mask(
                    graph, execution.visible_feedback
                ).items()
            }
            decision = _decide_policy(self.policy, state, graph, candidates)
            _append_trace(
                trace_path,
                "policy_decision",
                {
                    "state": state,
                    "decision": decision.to_dict(),
                    "legal_action_mask": action_mask,
                },
            )
            if decision.action is MacroAction.STOP:
                _append_trace(
                    trace_path,
                    "terminal",
                    {"src_graph_id": graph.graph_id, "action": "STOP", "absorbing_state": "BOTTOM"},
                )
                break
            previous_graph = graph
            previous_execution = execution
            previous_state = state
            graph = _with_runner(
                compiler.compile(graph, decision.action, execution.visible_feedback),
                _backend_name(self.llm),
            )
            execution = executor.execute(graph, task, execution)
            used_tokens += execution.token_cost
            summary = self.supervisor.summarize(execution, used_tokens, self.token_budget)
            next_state = compressed_state(
                self.adapter.task_features(task),
                self.adapter.signature(graph),
                execution.visible_feedback,
                summary,
                used_tokens,
                self.token_budget,
            )
            next_action_mask = {
                action.value: is_legal
                for action, is_legal in self.constraints.action_mask(
                    graph, execution.visible_feedback
                ).items()
            }
            dense = self.dense_reward.score(
                previous_graph,
                graph,
                previous_execution,
                execution,
            )
            transition = TransitionEdge(previous_graph.graph_id, graph.graph_id, decision.action)
            gog.add_snapshot(graph, self.adapter.signature(graph), transition)
            episode_graph_ids.append(graph.graph_id)
            export_snapshot(graph, snapshot_directory)
            _append_trace(
                trace_path,
                "snapshot",
                {
                    "graph": graph.to_dict(),
                    "execution": execution.to_dict(),
                    "supervisor": summary.to_dict(),
                    "transition": transition.to_dict(),
                },
            )
            replay_transition = ReplayTransition(
                graph_id=previous_graph.graph_id,
                next_graph_id=graph.graph_id,
                action=decision.action,
                reward=dense.reward,
                done=False,
                state=previous_state,
                next_state=next_state,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
                dense_reward=dense,
            )
            _append_trace(trace_path, "replay_transition", replay_transition.to_dict())

        episode_id_set = set(episode_graph_ids)
        episode_snapshots = [gog.snapshots[graph_id] for graph_id in episode_graph_ids]
        episode_transitions = [
            transition
            for transition in gog.transitions
            if transition.src_graph_id in episode_id_set and transition.dst_graph_id in episode_id_set
        ]
        export_gog(
            episode_snapshots,
            episode_transitions,
            directory,
            title="Current Episode Graph-of-Graphs",
        )
        result = {
            "episode_id": episode_id,
            "domain": self.adapter.name,
            "artifact_directory": str(directory),
            "final_graph_id": graph.graph_id,
            "final_output": execution.final_output,
            "snapshot_count": len(gog.snapshots) - history_snapshot_count,
            "transition_count": len(gog.transitions) - history_transition_count,
            "history_snapshot_count": history_snapshot_count,
            "history_transition_count": history_transition_count,
            "used_tokens": used_tokens,
            "llm_calls": sum(
                record.get("execution", {}).get("llm_calls", 0)
                for record in _read_trace(trace_path)
                if record.get("event") == "snapshot"
            ),
            "backend": _backend_name(self.llm),
        }
        (directory / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


def _append_trace(path: Path, event: str, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _backend_name(llm: LLMBackend) -> str:
    return str(getattr(llm, "name", type(llm).__name__))


def _with_runner(graph: Any, runner: str) -> Any:
    return replace(
        graph,
        nodes=tuple(
            replace(
                node,
                runner=runner,
                internal_nodes=tuple(
                    replace(child, runner=runner) for child in node.internal_nodes
                ),
            )
            for node in graph.nodes
        ),
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decide_policy(
    policy: Any,
    state: Mapping[str, Any],
    graph: Any,
    candidates: Any,
) -> Any:
    return policy.decide(state, graph, candidates)
