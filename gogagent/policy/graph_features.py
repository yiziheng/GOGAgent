"""Generic graph feature encoding for policy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gogagent.agents.registry import AGENT_CLASSES
from gogagent.graph.schema import Edge, Graph, Node


UNKNOWN_AGENT_TYPE = "Unknown"
SUBGRAPH_AGENT_TYPE = "Subgraph"
AGENT_TYPES: tuple[str, ...] = tuple(AGENT_CLASSES) + (
    UNKNOWN_AGENT_TYPE,
    SUBGRAPH_AGENT_TYPE,
)

AGENT_TYPE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"agent_type:{agent_type}"
    for agent_type in AGENT_TYPES
)
SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "depth",
    "is_subgraph",
    "in_degree",
    "out_degree",
    "is_in_node",
    "is_out_node",
    "left_step",
)
GRAPH_FEATURE_NAMES: tuple[str, ...] = AGENT_TYPE_FEATURE_NAMES + SCALAR_FEATURE_NAMES


@dataclass(frozen=True)
class GraphTensor:
    """Tensorized graph inputs for policy models."""

    node_features: torch.FloatTensor
    edge_index: torch.LongTensor
    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class GraphFeatureBuilder:
    """Build generic, dataset-agnostic node features for a GOG graph."""

    max_degree: float = 8.0
    max_depth: float = 4.0
    max_left_step: float = 8.0

    def build(
        self,
        graph: Graph,
        *,
        device: torch.device | str | None = None,
    ) -> GraphTensor:
        """Encode a Graph into generic node and edge tensors."""

        node_ids = tuple(graph.nodes)
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        in_degree, out_degree = _degree_tables(graph, node_index)

        rows = [
            self.node_features(
                node,
                node_id=node_id,
                graph=graph,
                in_degree=in_degree[index],
                out_degree=out_degree[index],
            )
            for index, (node_id, node) in enumerate(graph.nodes.items())
        ]
        node_features = torch.tensor(rows, dtype=torch.float32, device=device)
        if not rows:
            node_features = torch.empty(
                (0, len(GRAPH_FEATURE_NAMES)),
                dtype=torch.float32,
                device=device,
            )

        edge_index = _edge_index(graph.edges, node_index, device=device)
        return GraphTensor(
            node_features=node_features,
            edge_index=edge_index,
            node_ids=node_ids,
            feature_names=GRAPH_FEATURE_NAMES,
        )

    def node_features(
        self,
        node: Node,
        *,
        node_id: str,
        graph: Graph,
        in_degree: int,
        out_degree: int,
    ) -> list[float]:
        """Return one node feature row."""

        agent_type = _node_agent_type(node)
        one_hot = [0.0] * len(AGENT_TYPES)
        one_hot[AGENT_TYPES.index(agent_type)] = 1.0

        is_subgraph = _node_is_subgraph(node)
        return [
            *one_hot,
            _bounded(float(getattr(node, "depth", 1) or 0), self.max_depth),
            float(is_subgraph),
            _bounded(float(in_degree), self.max_degree),
            _bounded(float(out_degree), self.max_degree),
            float(node_id == graph.in_node),
            float(node_id == graph.out_node),
            _bounded(float(getattr(node, "left_step", 0) or 0), self.max_left_step),
        ]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in node feature order."""

        return GRAPH_FEATURE_NAMES

    @property
    def feature_dim(self) -> int:
        """Return the node feature width."""

        return len(GRAPH_FEATURE_NAMES)


def encode_graph(graph: Graph, *, device: torch.device | str | None = None) -> GraphTensor:
    """Encode a Graph into generic node and edge tensors."""

    return GraphFeatureBuilder().build(graph, device=device)


def graph_to_tensor(graph: Graph, *, device: torch.device | str | None = None) -> GraphTensor:
    """Alias for encode_graph."""

    return encode_graph(graph, device=device)


def feature_dim() -> int:
    """Return the node feature width produced by encode_graph."""

    return len(GRAPH_FEATURE_NAMES)


def _node_agent_type(node: Node) -> str:
    if _node_is_subgraph(node):
        return SUBGRAPH_AGENT_TYPE

    executor = getattr(node, "executor", None)
    agent_type = _executor_agent_type(executor)
    if agent_type in AGENT_CLASSES:
        return agent_type
    return UNKNOWN_AGENT_TYPE


def _executor_agent_type(executor: Any) -> str | None:
    if executor is None:
        return None
    if isinstance(executor, str):
        return executor

    agent_type = getattr(executor, "agent_type", None)
    if agent_type:
        return str(agent_type)

    if hasattr(executor, "to_dict"):
        data = executor.to_dict()
        if isinstance(data, dict):
            value = data.get("type", data.get("agent_type"))
            if value:
                return str(value)

    return executor.__class__.__name__


def _node_is_subgraph(node: Node) -> bool:
    executor = getattr(node, "executor", None)
    if isinstance(executor, Graph):
        return True
    return bool(getattr(node, "is_subgraph", False))


def _degree_tables(graph: Graph, node_index: dict[str, int]) -> tuple[list[int], list[int]]:
    in_degree = [0] * len(node_index)
    out_degree = [0] * len(node_index)
    for edge in graph.edges:
        source = _edge_source(edge)
        target = _edge_target(edge)
        if source not in node_index:
            raise ValueError(f"edge source does not exist: {source}")
        if target not in node_index:
            raise ValueError(f"edge target does not exist: {target}")
        out_degree[node_index[source]] += 1
        in_degree[node_index[target]] += 1
    return in_degree, out_degree


def _edge_index(
    edges: list[Edge],
    node_index: dict[str, int],
    *,
    device: torch.device | str | None,
) -> torch.LongTensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    encoded_edges = [
        (node_index[_edge_source(edge)], node_index[_edge_target(edge)])
        for edge in edges
    ]
    return torch.tensor(encoded_edges, dtype=torch.long, device=device).t().contiguous()


def _edge_source(edge: Edge) -> str:
    return str(getattr(edge, "source"))


def _edge_target(edge: Edge) -> str:
    return str(getattr(edge, "target"))


def _bounded(value: float, maximum: float) -> float:
    if maximum <= 0:
        return value
    return max(0.0, min(value, maximum)) / maximum


__all__ = [
    "AGENT_TYPES",
    "GRAPH_FEATURE_NAMES",
    "GraphFeatureBuilder",
    "GraphTensor",
    "encode_graph",
    "feature_dim",
    "graph_to_tensor",
]
