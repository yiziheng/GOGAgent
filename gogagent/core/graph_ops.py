"""Pure graph helpers for immutable organization DAG snapshots."""

from __future__ import annotations

from collections import deque
from uuid import uuid4

from gogagent.core.actions import MacroAction
from gogagent.core.types import CompiledEdit, EdgeSpec, GraphSignature, OrgGraphSnapshot


def make_graph_id() -> str:
    return f"g-{uuid4().hex[:8]}"


def apply_compiled_edit(
    graph: OrgGraphSnapshot,
    action: MacroAction,
    edit: CompiledEdit,
) -> OrgGraphSnapshot:
    """Create a new immutable DAG snapshot from a deterministic compiled edit."""

    existing_ids = {node.node_id for node in graph.nodes}
    duplicate_ids = existing_ids.intersection(node.node_id for node in edit.added_nodes)
    if duplicate_ids:
        raise ValueError(f"compiled edit adds duplicate node ids: {sorted(duplicate_ids)}")
    metadata = dict(graph.metadata)
    metadata.update(edit.metadata)
    return OrgGraphSnapshot(
        graph_id=make_graph_id(),
        domain=graph.domain,
        step=graph.step + 1,
        nodes=graph.nodes + edit.added_nodes,
        edges=graph.edges + edit.added_edges,
        parent_graph_id=graph.graph_id,
        created_by=action,
        metadata=metadata,
    )


def topological_order(graph: OrgGraphSnapshot) -> tuple[str, ...]:
    """Validate the graph and return one stable topological order."""

    node_ids = {node.node_id for node in graph.nodes}
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in graph.edges:
        if edge.src not in node_ids or edge.dst not in node_ids:
            raise ValueError(f"edge references missing node: {edge}")
        adjacency[edge.src].append(edge.dst)
        indegree[edge.dst] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for dst in sorted(adjacency[node_id]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    if len(order) != len(node_ids):
        raise ValueError(f"organization graph {graph.graph_id} contains a cycle")
    return tuple(order)


def graph_depth(graph: OrgGraphSnapshot) -> int:
    """Return the longest path node count in a DAG."""

    order = topological_order(graph)
    depth = {node_id: 1 for node_id in order}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in order}
    for edge in graph.edges:
        outgoing[edge.src].append(edge.dst)
    for src in order:
        for dst in outgoing[src]:
            depth[dst] = max(depth[dst], depth[src] + 1)
    return max(depth.values(), default=0)


def default_signature(graph: OrgGraphSnapshot) -> GraphSignature:
    return GraphSignature(
        roles=tuple(sorted(node.role for node in graph.nodes)),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        depth=graph_depth(graph),
        payload_modes=tuple(sorted({edge.payload for edge in graph.edges})),
    )
