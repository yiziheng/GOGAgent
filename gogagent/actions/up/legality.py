"""Legality rules for UP."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import (
    ActionConstraints,
    LegalityResult,
    expansion_fits,
    hierarchy_depth,
    last_atomic_node,
    node_id,
    node_is_subgraph,
)
from gogagent.actions.up.apply import get_upgrade_builder, upgrade_internal_node_count


def is_legal(graph: Any, constraints: ActionConstraints) -> LegalityResult:
    """UP is legal only when the last atomic node can become a depth-2 subgraph."""

    target = last_atomic_node(graph)
    if target is None:
        return LegalityResult(False, "UP has no atomic target")
    if node_is_subgraph(target):
        return LegalityResult(False, f"target is already a subgraph: {node_id(target)}")
    if get_upgrade_builder(target) is None:
        return LegalityResult(False, f"unsupported UP target type: {node_id(target)}")
    resulting_depth = max(hierarchy_depth(graph), 2)
    return expansion_fits(
        graph,
        constraints,
        added_nodes=upgrade_internal_node_count(target),
        resulting_depth=resulting_depth,
    )
