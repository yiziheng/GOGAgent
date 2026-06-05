"""Shared action types and graph helpers for the refactored runtime."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from gogagent.graph.schema import Edge, Graph, Node


class ActionName(str, Enum):
    """Flat action space exposed to the graph policy."""

    UP = "UP"
    ADD_TASK_BRIEF = "ADD_TASK_BRIEF"
    ADD_PLAN_SKETCH = "ADD_PLAN_SKETCH"
    ADD_ADVERSARIAL_JUDGE = "ADD_ADVERSARIAL_JUDGE"
    ADD_FORMAT_VERIFIER = "ADD_FORMAT_VERIFIER"
    STOP = "STOP"


@dataclass(frozen=True)
class ActionSpec:
    """Static action metadata used by policy, teacher prompts, and rewards."""

    name: ActionName
    description: str
    complexity_penalty: float = 0.0
    is_expansion: bool = False


@dataclass(frozen=True)
class ActionConstraints:
    """Version-1 graph construction limits."""

    max_depth: int = 2
    max_nodes: int = 8


@dataclass(frozen=True)
class LegalityResult:
    """A boolean legality decision plus a compact diagnostic reason."""

    legal: bool
    reason: str = ""


GraphLike = Graph
NodeLike = Node
EdgeLike = Edge
UpgradeBuilder = Callable[[NodeLike], GraphLike]


def normalize_action_name(action: ActionName | str) -> ActionName:
    """Return a typed action name from either enum or string input."""

    if isinstance(action, ActionName):
        return action
    return ActionName(str(action))


def normalize_agent_key(name: str) -> str:
    """Normalize class-like agent names into stable snake-ish keys."""

    cleaned = name.replace("Agent", "")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
    return cleaned.strip("_").lower()


def graph_nodes(graph: GraphLike) -> list[NodeLike]:
    """Return top-level graph nodes in stable execution/order-list order."""

    nodes = getattr(graph, "nodes", {})
    if isinstance(nodes, Mapping):
        return list(nodes.values())
    return list(nodes)


def graph_edges(graph: GraphLike) -> list[EdgeLike]:
    """Return graph edges as a mutable list copy."""

    return list(getattr(graph, "edges", ()))


def graph_node_map(graph: GraphLike) -> dict[str, NodeLike]:
    """Return top-level nodes keyed by node id."""

    nodes = getattr(graph, "nodes", {})
    if isinstance(nodes, Mapping):
        return dict(nodes)
    return {node_id(node): node for node in nodes}


def node_id(node: NodeLike) -> str:
    """Best-effort stable node id accessor."""

    for attr in ("node_id", "id"):
        value = getattr(node, attr, None)
        if value:
            return str(value)
    name = getattr(node, "name", None) or getattr(node, "role", None)
    if name:
        return str(name)
    raise ValueError(f"node has no node_id/id/name: {node!r}")


def node_name(node: NodeLike) -> str:
    """Best-effort display/agent name accessor."""

    for attr in ("name", "role", "agent_name"):
        value = getattr(node, attr, None)
        if value:
            return str(value)
    executor = getattr(node, "executor", None)
    if executor is not None:
        return executor.__class__.__name__ if not isinstance(executor, str) else executor
    return node_id(node)


def node_agent_key(node: NodeLike) -> str:
    """Return a normalized key for an agent node."""

    executor = getattr(node, "executor", None)
    if executor is not None and not _looks_like_graph(executor):
        if isinstance(executor, str):
            return normalize_agent_key(executor)
        return normalize_agent_key(executor.__class__.__name__)
    return normalize_agent_key(node_name(node))


def node_is_subgraph(node: NodeLike) -> bool:
    """Return whether a node already executes a nested graph."""

    executor = getattr(node, "executor", None)
    if executor is not None and _looks_like_graph(executor):
        return True
    if getattr(node, "depth", 1) and int(getattr(node, "depth", 1)) > 1:
        return True
    if getattr(node, "node_kind", "") == "graph":
        return True
    return bool(getattr(node, "internal_nodes", ()))


def last_atomic_node(graph: GraphLike) -> NodeLike | None:
    """Return the last top-level node that is still atomic."""

    for node in reversed(graph_nodes(graph)):
        if not node_is_subgraph(node):
            return node
    return None


def graph_has_agent(graph: GraphLike, agent_keys: Iterable[str]) -> bool:
    """Return whether a top-level or nested node matches one of the keys."""

    wanted = {normalize_agent_key(key) for key in agent_keys}
    for node in graph_nodes(graph):
        if node_agent_key(node) in wanted:
            return True
        executor = getattr(node, "executor", None)
        if _looks_like_graph(executor) and graph_has_agent(executor, wanted):
            return True
        for internal in getattr(node, "internal_nodes", ()):
            if node_agent_key(internal) in wanted:
                return True
    return False


def total_node_count(graph: GraphLike) -> int:
    """Count top-level nodes plus nested subgraph nodes."""

    total = 0
    for node in graph_nodes(graph):
        total += 1
        executor = getattr(node, "executor", None)
        if _looks_like_graph(executor):
            total += total_node_count(executor)
        else:
            total += len(tuple(getattr(node, "internal_nodes", ())))
    return total


def hierarchy_depth(graph: GraphLike) -> int:
    """Return recursive Graph-of-Graphs depth, not topological path length."""

    max_depth = 1
    for node in graph_nodes(graph):
        max_depth = max(max_depth, int(getattr(node, "depth", 1) or 1))
        executor = getattr(node, "executor", None)
        if _looks_like_graph(executor):
            max_depth = max(max_depth, 1 + hierarchy_depth(executor))
        elif getattr(node, "internal_nodes", ()):
            max_depth = max(max_depth, 2)
    return max_depth


def expansion_fits(
    graph: GraphLike,
    constraints: ActionConstraints,
    *,
    added_nodes: int,
    resulting_depth: int | None = None,
) -> LegalityResult:
    """Check max-node and max-depth constraints for a proposed expansion."""

    next_nodes = total_node_count(graph) + added_nodes
    if next_nodes > constraints.max_nodes:
        return LegalityResult(
            False,
            f"max_nodes exceeded: {next_nodes}>{constraints.max_nodes}",
        )
    next_depth = resulting_depth if resulting_depth is not None else hierarchy_depth(graph)
    if next_depth > constraints.max_depth:
        return LegalityResult(
            False,
            f"max_depth exceeded: {next_depth}>{constraints.max_depth}",
        )
    return LegalityResult(True)


def unique_node_id(graph: GraphLike, base_name: str) -> str:
    """Create a deterministic id that is unique in the top-level graph."""

    base = normalize_agent_key(base_name) or "node"
    existing = set(graph_node_map(graph))
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def make_node(
    agent_name: str,
    *,
    node_id_value: str | None = None,
    name: str | None = None,
    executor: Any | None = None,
    depth: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> NodeLike:
    """Construct a graph runtime node."""

    node_id_value = node_id_value or normalize_agent_key(agent_name)
    executor = executor if executor is not None else _make_agent(agent_name)
    return Node(
        node_id=node_id_value,
        name=name or agent_name,
        executor=executor,
        depth=depth,
        final_node_id=None,
        left_step=0,
        metadata=dict(metadata or {}),
    )


def make_edge(
    src: str,
    dst: str,
    *,
    payload: str = "default",
    edge_kind: str = "exec",
    metadata: Mapping[str, Any] | None = None,
) -> EdgeLike:
    """Construct a graph runtime edge."""

    return Edge(
        source=src,
        target=dst,
        role=edge_kind or payload,
        metadata=dict(metadata or {}),
    )


def make_graph(
    *,
    in_node: str,
    out_node: str,
    nodes: Mapping[str, NodeLike] | Sequence[NodeLike],
    edges: Sequence[EdgeLike],
    metadata: Mapping[str, Any] | None = None,
) -> GraphLike:
    """Construct a graph runtime object."""

    node_map = dict(nodes) if isinstance(nodes, Mapping) else {node_id(n): n for n in nodes}
    return Graph(
        in_node=in_node,
        out_node=out_node,
        nodes=node_map,
        edges=list(edges),
        metadata=dict(metadata or {}),
    )


def make_linear_subgraph(
    target_node: NodeLike,
    agent_names: Sequence[str],
    *,
    graph_type: str,
) -> GraphLike:
    """Build a depth-2 linear subgraph used by UP templates."""

    parent_id = node_id(target_node)
    nodes: dict[str, NodeLike] = {}
    previous_id: str | None = None
    edges: list[EdgeLike] = []
    for index, agent_name in enumerate(agent_names, start=1):
        local_id = f"{parent_id}_{normalize_agent_key(agent_name)}_{index}"
        node = make_node(
            agent_name,
            node_id_value=local_id,
            metadata={"created_by": "UP", "up_parent": parent_id},
        )
        nodes[local_id] = node
        if previous_id is not None:
            edges.append(make_edge(previous_id, local_id))
        previous_id = local_id
    if not nodes:
        raise ValueError("UP template must contain at least one internal node")
    ordered_ids = list(nodes)
    return make_graph(
        in_node=ordered_ids[0],
        out_node=ordered_ids[-1],
        nodes=nodes,
        edges=edges,
        metadata={"graph_type": graph_type, "created_by": "UP", "parent": parent_id},
    )


def replace_graph_node(graph: GraphLike, new_node: NodeLike) -> GraphLike:
    """Return a graph copy with one top-level node replaced."""

    target_id = node_id(new_node)
    node_map = graph_node_map(graph)
    if target_id not in node_map:
        raise ValueError(f"cannot replace missing node: {target_id}")
    node_map[target_id] = new_node
    return clone_graph(graph, nodes=_nodes_like_graph(graph, node_map))


def with_node_executor(
    node: NodeLike,
    executor: Any,
    *,
    depth: int,
    metadata: Mapping[str, Any] | None = None,
) -> NodeLike:
    """Return a node copy whose executor is a nested subgraph."""

    updates: dict[str, Any] = {"executor": executor, "depth": depth}
    if hasattr(node, "metadata"):
        merged = dict(getattr(node, "metadata") or {})
        merged.update(metadata or {})
        updates["metadata"] = merged
    return clone_object(node, updates)


def append_as_output_node(graph: GraphLike, node: NodeLike) -> GraphLike:
    """Append a new node after the current output node."""

    node_map = graph_node_map(graph)
    new_id = node_id(node)
    if new_id in node_map:
        raise ValueError(f"duplicate node id: {new_id}")
    old_out = get_out_node_id(graph)
    node_map[new_id] = node
    edges = graph_edges(graph)
    if old_out is not None:
        edges.append(make_edge(old_out, new_id))
    return clone_graph(
        graph,
        nodes=_nodes_like_graph(graph, node_map),
        edges=_edges_like_graph(graph, edges),
        in_node=get_in_node_id(graph) or new_id,
        out_node=new_id,
    )


def prepend_as_input_node(graph: GraphLike, node: NodeLike) -> GraphLike:
    """Prepend a new node before the current input node."""

    node_map = graph_node_map(graph)
    new_id = node_id(node)
    if new_id in node_map:
        raise ValueError(f"duplicate node id: {new_id}")
    old_in = get_in_node_id(graph)
    ordered = {new_id: node}
    ordered.update(node_map)
    edges = graph_edges(graph)
    if old_in is not None:
        edges.insert(0, make_edge(new_id, old_in))
    return clone_graph(
        graph,
        nodes=_nodes_like_graph(graph, ordered),
        edges=_edges_like_graph(graph, edges),
        in_node=new_id,
        out_node=get_out_node_id(graph) or new_id,
    )


def get_in_node_id(graph: GraphLike) -> str | None:
    """Return graph input node id if available."""

    value = getattr(graph, "in_node", None)
    if value is not None:
        return str(value)
    nodes = graph_nodes(graph)
    return node_id(nodes[0]) if nodes else None


def get_out_node_id(graph: GraphLike) -> str | None:
    """Return graph output node id if available."""

    value = getattr(graph, "out_node", None)
    if value is not None:
        return str(value)
    nodes = graph_nodes(graph)
    return node_id(nodes[-1]) if nodes else None


def clone_graph(graph: GraphLike, **updates: Any) -> GraphLike:
    """Clone a graph object while preserving its concrete class when possible."""

    return clone_object(graph, updates)


def clone_object(obj: Any, updates: Mapping[str, Any]) -> Any:
    """Clone a dataclass or mutable object with best-effort field filtering."""

    if is_dataclass(obj):
        allowed = set(getattr(obj, "__dataclass_fields__", {}))
        filtered = {key: value for key, value in updates.items() if key in allowed}
        if filtered:
            return replace(obj, **filtered)
        return obj
    cloned = copy.copy(obj)
    for key, value in updates.items():
        if hasattr(cloned, key):
            setattr(cloned, key, value)
    return cloned


def _nodes_like_graph(graph: GraphLike, node_map: Mapping[str, NodeLike]) -> Any:
    original = getattr(graph, "nodes", {})
    if isinstance(original, MutableMapping) or isinstance(original, Mapping):
        return dict(node_map)
    if isinstance(original, tuple):
        return tuple(node_map.values())
    if isinstance(original, list):
        return list(node_map.values())
    return dict(node_map)


def _edges_like_graph(graph: GraphLike, edges: Sequence[EdgeLike]) -> Any:
    original = getattr(graph, "edges", ())
    if isinstance(original, tuple):
        return tuple(edges)
    if isinstance(original, list):
        return list(edges)
    return list(edges)


def _looks_like_graph(value: Any) -> bool:
    return value is not None and hasattr(value, "nodes") and hasattr(value, "edges")


def _make_agent(agent_name: str) -> Any:
    from gogagent.agents.registry import get_agent

    return get_agent(agent_name)
