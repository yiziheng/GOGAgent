"""UP template for SolverAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_linear_subgraph


INTERNAL_AGENT_NAMES = ("PlanSketchAgent", "SolverAgent")


def build_subgraph(target_node: Any) -> Any:
    """Expand SolverAgent into PlanSketchAgent -> SolverAgent."""

    return make_linear_subgraph(
        target_node,
        INTERNAL_AGENT_NAMES,
        graph_type="solver_planner_solver",
    )
