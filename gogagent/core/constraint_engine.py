"""Deterministic legality masks for bounded graph construction."""

from __future__ import annotations

from gogagent.core.actions import MacroAction
from gogagent.core.graph_ops import topological_order
from gogagent.core.types import MacroCandidate, OrgGraphSnapshot, VisibleFeedback


class ConstraintEngine:
    """Keep the learned policy inside a small executable DAG grammar."""

    def __init__(self, max_steps: int = 4, max_nodes: int = 8) -> None:
        self.max_steps = max_steps
        self.max_nodes = max_nodes

    def validate(self, graph: OrgGraphSnapshot) -> None:
        if len(graph.nodes) > self.max_nodes:
            raise ValueError(f"graph exceeds max_nodes={self.max_nodes}")
        topological_order(graph)

    def legal_candidates(
        self,
        graph: OrgGraphSnapshot,
        feedback: VisibleFeedback,
    ) -> tuple[MacroCandidate, ...]:
        """Return label-blind legal actions; STOP is always available."""

        candidates = [MacroCandidate(MacroAction.STOP, "finish with the current graph")]
        if graph.step >= self.max_steps:
            return tuple(candidates)
        capabilities = {_capability(node.role) for node in graph.nodes}
        if "ANALYST" not in capabilities:
            candidates.append(MacroCandidate(MacroAction.ATTACH_ANALYST, "add domain analysis"))
        if "CHECKER" not in capabilities:
            candidates.append(MacroCandidate(MacroAction.ATTACH_CHECKER, "inspect the current answer"))
        has_check_feedback = (
            "CHECKER" in capabilities
            and (
                bool(feedback.issue_codes)
                or feedback.disagreement_level != "none"
                or feedback.status not in {"ready", "passed"}
            )
        )
        if has_check_feedback and "REVISER" not in capabilities:
            candidates.append(MacroCandidate(MacroAction.ATTACH_REVISER, "revise after inspection"))
        uncertain = feedback.confidence_bucket == "low" or feedback.disagreement_level != "none"
        if uncertain and "ALTERNATIVE" not in capabilities:
            candidates.append(MacroCandidate(MacroAction.ATTACH_ALTERNATIVE, "request an independent opinion"))
        return tuple(candidates[:4])


def _capability(role: str) -> str:
    lowered = role.lower()
    if any(word in lowered for word in ("analyst", "planner", "decomposer")):
        return "ANALYST"
    if any(word in lowered for word in ("checker", "critic", "inspector", "tester")):
        return "CHECKER"
    if any(word in lowered for word in ("reviser", "resolver", "debugger", "fixer")):
        return "REVISER"
    if any(word in lowered for word in ("alternative", "independent", "secondopinion")):
        return "ALTERNATIVE"
    return "SOLVER"
