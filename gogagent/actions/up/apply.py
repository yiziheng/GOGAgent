"""Graph mutation for UP."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gogagent.actions.base import (
    last_atomic_node,
    node_agent_key,
    node_id,
    replace_graph_node,
    with_node_executor,
)


_UP_MODULES: dict[str, str] = {
    "solver": "gogagent.actions.up.solver",
    "adversarial_judge": "gogagent.actions.up.adversarial_judge",
    "format_verifier": "gogagent.actions.up.format_verifier",
    "task_brief": "gogagent.actions.up.task_brief",
    "plan_sketch": "gogagent.actions.up.plan_sketch",
}


def apply(graph: Any) -> Any:
    """Upgrade the current graph's last atomic node into a subgraph node."""

    target = last_atomic_node(graph)
    if target is None:
        raise ValueError("UP has no atomic target")
    builder = get_upgrade_builder(target)
    if builder is None:
        raise ValueError(f"UP has no template for node {node_id(target)}")
    subgraph = builder(target)
    upgraded = with_node_executor(
        target,
        subgraph,
        depth=2,
        metadata={"upgraded_by": "UP", "upgraded_from": node_agent_key(target)},
    )
    return replace_graph_node(graph, upgraded)


def apply_up(graph: Any) -> Any:
    """Explicit alias for callers that name the UP mutation directly."""

    return apply(graph)


def get_upgrade_builder(target_node: Any):
    """Return the type-specific UP builder for a target node."""

    module_path = _UP_MODULES.get(node_agent_key(target_node))
    if module_path is None:
        return None
    module = import_module(module_path)
    return module.build_subgraph


def upgrade_internal_node_count(target_node: Any) -> int:
    """Return how many internal nodes the target's UP template will create."""

    builder = get_upgrade_builder(target_node)
    if builder is None:
        return 0
    module = import_module(builder.__module__)
    return len(getattr(module, "INTERNAL_AGENT_NAMES", ()))
