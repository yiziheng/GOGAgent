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
    """Create a new immutable hierarchical GoG from a deterministic edit."""

    removed_ids = set(edit.removed_nodes)
    existing_by_id = {
        node.node_id: node for node in graph.nodes if node.node_id not in removed_ids
    }
    updated_ids = {node.node_id for node in edit.updated_nodes}
    missing_updates = sorted(updated_ids - set(existing_by_id))
    if missing_updates:
        raise ValueError(f"compiled edit updates missing node ids: {missing_updates}")
    for node in edit.updated_nodes:
        existing_by_id[node.node_id] = node
    duplicate_ids = set(existing_by_id).intersection(
        node.node_id for node in edit.added_nodes
    )
    if duplicate_ids:
        raise ValueError(f"compiled edit adds duplicate node ids: {sorted(duplicate_ids)}")
    removed_edge_keys = {_edge_key(edge) for edge in edit.removed_edges}
    retained_edges = tuple(
        edge
        for edge in graph.edges
        if edge.src not in removed_ids
        and edge.dst not in removed_ids
        and _edge_key(edge) not in removed_edge_keys
    )
    metadata = dict(graph.metadata)
    metadata.update(edit.metadata)
    return OrgGraphSnapshot(
        graph_id=make_graph_id(),
        domain=graph.domain,
        step=graph.step + 1,
        nodes=tuple(existing_by_id.values()) + edit.added_nodes,
        edges=retained_edges + edit.added_edges,
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
    graph_agents = tuple(node for node in graph.nodes if node.node_kind == "graph")
    atomic_agents = tuple(node for node in graph.nodes if node.node_kind != "graph")
    return GraphSignature(
        roles=tuple(sorted(node.role for node in graph.nodes)),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        depth=graph_depth(graph),
        payload_modes=tuple(sorted({edge.payload for edge in graph.edges})),
        graph_agent_count=len(graph_agents),
        atomic_agent_count=len(atomic_agents),
        module_types=tuple(sorted(node.module_type for node in graph_agents if node.module_type)),
        max_graphagent_internal_nodes=max(
            (len(node.internal_nodes) for node in graph_agents),
            default=0,
        ),
    )


def _edge_key(edge: EdgeSpec) -> tuple[str, str, str, str]:
    return (edge.src, edge.dst, edge.payload, edge.edge_kind)
