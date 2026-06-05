"""Legality rules for ADD_PLAN_SKETCH."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import (
    ActionConstraints,
    LegalityResult,
    expansion_fits,
    graph_has_agent,
    hierarchy_depth,
)


def is_legal(graph: Any, constraints: ActionConstraints) -> LegalityResult:
    """ADD_PLAN_SKETCH is illegal once a planner already exists."""

    if graph_has_agent(graph, {"plan_sketch", "planner"}):
        return LegalityResult(False, "planner already exists")
    return expansion_fits(
        graph,
        constraints,
        added_nodes=1,
        resulting_depth=hierarchy_depth(graph),
    )
