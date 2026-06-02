"""End-to-end label-blind graph construction rollout."""

from __future__ import annotations

import json
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
from gogagent.policy.q_scorer import QScorer


class RolloutEngine:
    """Construct one task-specific DAG while persisting every visible graph artifact."""

    def __init__(
        self,
        adapter: DomainAdapter,
        llm: LLMBackend,
        artifact_root: str | Path = "artifacts/runs",
        constraints: ConstraintEngine | None = None,
        policy: QScorer | None = None,
        supervisor: SupervisorAgent | None = None,
        gog_memory: OrganizationGoG | str | Path | None = None,
        token_budget: int = 4096,
    ) -> None:
        self.adapter = adapter
        self.llm = llm
        self.artifact_root = Path(artifact_root)
        self.constraints = constraints or ConstraintEngine()
        self.policy = policy or QScorer()
        self.supervisor = supervisor or SupervisorAgent()
        self.gog_memory = (
            OrganizationGoG.load(gog_memory)
            if isinstance(gog_memory, (str, Path))
            else gog_memory or OrganizationGoG()
        )
        self.token_budget = token_budget

    def run(self, task: Mapping[str, Any], episode_id: str | None = None) -> dict[str, Any]:
        episode_id = episode_id or uuid4().hex[:10]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = self.artifact_root / run_id / self.adapter.name / episode_id
        snapshot_directory = directory / "snapshots"
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        trace_path = directory / "trace.jsonl"

        gog = self.gog_memory.fork_for_rollout()
        history_snapshot_count = len(gog.snapshots)
        history_transition_count = len(gog.transitions)
        compiler = MacroCompiler(self.adapter, self.constraints)
        executor = IncrementalExecutor(self.adapter, self.llm)
        graph = self.adapter.base_graph(task)
        self.constraints.validate(graph)
        execution = executor.execute(graph, task)
        used_tokens = execution.token_cost
        summary = self.supervisor.summarize(execution, used_tokens, self.token_budget)
        gog.add_snapshot(graph, self.adapter.signature(graph))
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
            decision = self.policy.decide(state, graph.graph_id, candidates, gog)
            neighbor_stats = {
                candidate.action.value: gog.neighbor_stats(graph.graph_id, candidate.action)
                for candidate in candidates
            }
            _append_trace(
                trace_path,
                "policy_decision",
                {
                    "state": state,
                    "decision": decision.to_dict(),
                    "neighbor_stats": neighbor_stats,
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
            graph = compiler.compile(graph, decision.action, execution.visible_feedback)
            execution = executor.execute(graph, task, execution)
            used_tokens += execution.token_cost
            summary = self.supervisor.summarize(execution, used_tokens, self.token_budget)
            transition = TransitionEdge(previous_graph.graph_id, graph.graph_id, decision.action)
            gog.add_snapshot(graph, self.adapter.signature(graph), transition)
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

        export_gog(gog.snapshots.values(), gog.transitions, gog.similarities, directory)
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
