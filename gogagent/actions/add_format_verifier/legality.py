"""Legality rules for ADD_FORMAT_VERIFIER."""

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
    """ADD_FORMAT_VERIFIER is illegal once a verifier already exists."""

    if graph_has_agent(graph, {"format_verifier", "verifier"}):
        return LegalityResult(False, "verifier already exists")
    return expansion_fits(
        graph,
        constraints,
        added_nodes=1,
        resulting_depth=hierarchy_depth(graph),
    )
