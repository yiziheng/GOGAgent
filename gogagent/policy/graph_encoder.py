"""Lightweight torch-only graph encoder for policy decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from gogagent.graph.schema import Graph
from gogagent.policy.graph_features import GraphFeatureBuilder, GraphTensor, feature_dim

if TYPE_CHECKING:
    TensorLikeGraph = GraphTensor | Graph
else:
    TensorLikeGraph = Any


class GCNLayer(nn.Module):
    """A small directed GCN-style message passing layer implemented in torch."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one normalized message-passing step."""

        if node_features.dim() != 2:
            raise ValueError("node_features must have shape [num_nodes, feature_dim]")

        num_nodes = node_features.size(0)
        transformed = self.linear(node_features)
        if num_nodes == 0:
            return transformed

        edge_index = _coerce_edge_index(edge_index, device=node_features.device)
        self_loops = torch.arange(num_nodes, device=node_features.device, dtype=torch.long)
        self_loops = torch.stack((self_loops, self_loops), dim=0)
        edge_index = torch.cat((edge_index, self_loops), dim=1)

        source, target = edge_index
        valid = (
            (source >= 0)
            & (source < num_nodes)
            & (target >= 0)
            & (target < num_nodes)
        )
        source = source[valid]
        target = target[valid]

        degree = torch.zeros(num_nodes, device=node_features.device, dtype=transformed.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=transformed.dtype))
        norm = degree.clamp_min(1.0).pow(-0.5)
        weights = norm[source] * norm[target]

        output = transformed.new_zeros(transformed.shape)
        output.index_add_(0, target, transformed[source] * weights.unsqueeze(-1))
        return output


class GraphEncoder(nn.Module):
    """Encode a GraphTensor or Graph into a fixed-size graph embedding."""

    def __init__(
        self,
        input_dim: int | None = None,
        embedding_dim: int = 64,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        max_subgraph_depth: int = 2,
        dropout: float = 0.0,
        pooling: str = "mean",
        feature_builder: GraphFeatureBuilder | None = None,
    ) -> None:
        super().__init__()
        input_dim = input_dim or feature_dim()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_subgraph_depth < 0:
            raise ValueError("max_subgraph_depth must be non-negative")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of: mean, sum, max")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_subgraph_depth = max_subgraph_depth
        self.dropout = dropout
        self.pooling = pooling
        self.feature_builder = feature_builder or GraphFeatureBuilder()

        hidden_dim = hidden_dim or embedding_dim
        dims = [input_dim]
        dims.extend([hidden_dim] * max(0, num_layers - 1))
        dims.append(embedding_dim)
        self.layers = nn.ModuleList(
            GCNLayer(dims[index], dims[index + 1])
            for index in range(len(dims) - 1)
        )
        self.subgraph_projection = nn.Linear(embedding_dim, input_dim)
        self.empty_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, graph: TensorLikeGraph) -> torch.Tensor:
        """Return a graph embedding with shape [embedding_dim]."""

        return self._encode(graph, depth=0, active_graph_ids=set())

    def _encode(
        self,
        graph: TensorLikeGraph,
        *,
        depth: int,
        active_graph_ids: set[int],
    ) -> torch.Tensor:
        if _is_graph_instance(graph):
            node_features, edge_index = self._graph_to_tensors(
                graph,
                depth=depth,
                active_graph_ids=active_graph_ids,
            )
        else:
            node_features, edge_index = _graph_tensor_to_tensors(graph)
            node_features = self._fit_feature_dim(node_features)

        device, dtype = self._parameter_device_dtype()
        node_features = node_features.to(device=device, dtype=dtype)
        edge_index = edge_index.to(device=device)
        return self._encode_tensors(node_features, edge_index)

    def _encode_tensors(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_features = self._fit_feature_dim(node_features)
        if node_features.size(0) == 0:
            return self.empty_embedding

        hidden = node_features
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index + 1 < len(self.layers):
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        return self._pool(hidden)

    def _graph_to_tensors(
        self,
        graph: Graph,
        *,
        depth: int,
        active_graph_ids: set[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_identity = id(graph)
        if graph_identity in active_graph_ids:
            raise ValueError("cycle detected while encoding nested graph")

        active_graph_ids.add(graph_identity)
        try:
            device, dtype = self._parameter_device_dtype()
            graph_tensor = self.feature_builder.build(graph, device=device)
            node_features = self._fit_feature_dim(
                graph_tensor.node_features.to(device=device, dtype=dtype)
            )
            if depth + 1 <= self.max_subgraph_depth:
                enriched_rows = []
                for index, node_id_value in enumerate(graph_tensor.node_ids):
                    feature = node_features[index]
                    subgraph = _node_subgraph(graph.nodes[node_id_value])
                    if subgraph is not None:
                        subgraph_embedding = self._encode(
                            subgraph,
                            depth=depth + 1,
                            active_graph_ids=active_graph_ids,
                        )
                        feature = feature + self.subgraph_projection(subgraph_embedding)
                    enriched_rows.append(feature)
                if enriched_rows:
                    node_features = torch.stack(enriched_rows, dim=0)
            edge_index = graph_tensor.edge_index.to(device=device)
            return node_features, edge_index
        finally:
            active_graph_ids.remove(graph_identity)

    def _fit_feature_dim(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        if features.dim() != 2:
            raise ValueError("features must have shape [num_nodes, feature_dim]")

        current_dim = features.size(-1)
        if current_dim < self.input_dim:
            padding = features.new_zeros(features.size(0), self.input_dim - current_dim)
            features = torch.cat((features, padding), dim=-1)
        elif current_dim > self.input_dim:
            features = features[..., : self.input_dim]

        return features.squeeze(0) if squeezed else features

    def _pool(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.pooling == "sum":
            return node_embeddings.sum(dim=0)
        if self.pooling == "max":
            return node_embeddings.max(dim=0).values
        return node_embeddings.mean(dim=0)

    def _parameter_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype


def _graph_tensor_to_tensors(graph_tensor: Any) -> tuple[torch.Tensor, torch.Tensor]:
    node_features = _first_attr(
        graph_tensor,
        "node_features",
        "features",
        "x",
        "node_feature",
        "feature_matrix",
    )
    if node_features is None:
        raise TypeError("GraphTensor must expose node_features/features/x")
    if not isinstance(node_features, torch.Tensor):
        node_features = torch.as_tensor(node_features, dtype=torch.float32)
    if node_features.dim() == 1:
        node_features = node_features.unsqueeze(0)
    if node_features.dim() != 2:
        raise ValueError("GraphTensor node features must have shape [num_nodes, feature_dim]")

    edge_data = _first_attr(
        graph_tensor,
        "edge_index",
        "edge_indices",
        "edges",
        "edge_list",
        "adjacency",
    )
    edge_index = _coerce_edge_index(edge_data, device=node_features.device)
    return node_features.float(), edge_index


def _coerce_edge_index(edge_data: Any, *, device: torch.device) -> torch.Tensor:
    if edge_data is None:
        return torch.empty(2, 0, device=device, dtype=torch.long)

    if isinstance(edge_data, torch.Tensor):
        edge_tensor = edge_data.to(device=device)
        if edge_tensor.dim() == 2 and edge_tensor.size(0) == 2:
            return edge_tensor.long()
        if edge_tensor.dim() == 2 and edge_tensor.size(1) == 2:
            return edge_tensor.t().contiguous().long()
        if edge_tensor.dim() == 2 and edge_tensor.size(0) == edge_tensor.size(1):
            return edge_tensor.nonzero(as_tuple=False).t().contiguous().long()
        raise ValueError("edge_index tensor must be [2, E], [E, 2], or adjacency [N, N]")

    pairs = []
    for edge in edge_data:
        if hasattr(edge, "source") and hasattr(edge, "target"):
            pairs.append((int(edge.source), int(edge.target)))
        else:
            source, target = edge
            pairs.append((int(source), int(target)))
    if not pairs:
        return torch.empty(2, 0, device=device, dtype=torch.long)
    return torch.tensor(pairs, device=device, dtype=torch.long).t().contiguous()


def _node_subgraph(node: Any) -> Graph | None:
    executor = getattr(node, "executor", None)
    if _is_graph_instance(executor):
        return executor
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("subgraph") or metadata.get("graph")
        if _is_graph_instance(value):
            return value
    return None


def _is_graph_instance(value: Any) -> bool:
    return isinstance(value, Graph) or (
        hasattr(value, "nodes")
        and hasattr(value, "edges")
        and not isinstance(value, torch.Tensor)
    )


def _first_attr(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None

    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            return item
    return None


__all__ = ["GCNLayer", "GraphEncoder"]
