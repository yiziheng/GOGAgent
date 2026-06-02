"""Adapter-backed typed macro compiler."""

from __future__ import annotations

from gogagent.adapters.base import DomainAdapter
from gogagent.core.actions import MacroAction
from gogagent.core.constraint_engine import ConstraintEngine
from gogagent.core.graph_ops import apply_compiled_edit
from gogagent.core.types import OrgGraphSnapshot, VisibleFeedback


class MacroCompiler:
    def __init__(self, adapter: DomainAdapter, constraints: ConstraintEngine) -> None:
        self.adapter = adapter
        self.constraints = constraints

    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> OrgGraphSnapshot:
        if action is MacroAction.STOP:
            raise ValueError("STOP is a terminal action and does not compile into a GoG node")
        legal = {candidate.action for candidate in self.constraints.legal_candidates(graph, feedback)}
        if action not in legal:
            raise ValueError(f"illegal action {action.value} for graph {graph.graph_id}")
        snapshot = apply_compiled_edit(graph, action, self.adapter.compile(graph, action, feedback))
        self.constraints.validate(snapshot)
        return snapshot
