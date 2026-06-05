"""UP template for PlanSketchAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_linear_subgraph


INTERNAL_AGENT_NAMES = ("TaskBriefAgent", "PlanSketchAgent")


def build_subgraph(target_node: Any) -> Any:
    """Expand PlanSketchAgent into TaskBrief -> PlanSketch."""

    return make_linear_subgraph(
        target_node,
        INTERNAL_AGENT_NAMES,
        graph_type="brief_plan_sketch",
    )
