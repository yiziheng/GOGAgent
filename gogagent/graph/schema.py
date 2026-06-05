"""Serializable Graph-of-Graphs data schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


JsonDict = dict[str, Any]


@dataclass
class GraphMessage:
    """Structured JSON message passed between graph nodes."""

    role: str
    content: str
    sender: str | None = None
    answer: str | None = None
    confidence: float | None = None
    notes: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable representation."""

        return {
            "sender": self.sender,
            "role": self.role,
            "content": self.content,
            "answer": self.answer,
            "confidence": self.confidence,
            "notes": dict(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "GraphMessage") -> "GraphMessage":
        """Build a message from a JSON mapping or return an existing message."""

        if isinstance(data, GraphMessage):
            return data
        return cls(
            sender=_optional_str(data.get("sender")),
            role=str(data.get("role", "agent")),
            content=str(data.get("content", "")),
            answer=_optional_str(data.get("answer")),
            confidence=_optional_float(data.get("confidence")),
            notes=dict(data.get("notes") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Edge:
    """Directed graph edge from one node id to another."""

    source: str
    target: str
    role: str = "message"
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable representation."""

        return {
            "source": self.source,
            "target": self.target,
            "role": self.role,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        """Build an edge from a JSON mapping."""

        source = data.get("source", data.get("src", data.get("from")))
        target = data.get("target", data.get("dst", data.get("to")))
        if source is None or target is None:
            raise ValueError("edge requires source and target")
        return cls(
            source=str(source),
            target=str(target),
            role=str(data.get("role", "message")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Node:
    """A graph node whose executor can be either an Agent or another Graph."""

    node_id: str
    name: str
    executor: Any
    depth: int = 1
    final_node_id: str | None = None
    left_step: int = 1
    metadata: JsonDict = field(default_factory=dict)

    @property
    def is_subgraph(self) -> bool:
        """Whether this node executes a nested Graph."""

        return isinstance(self.executor, Graph)

    def process(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        *,
        context: Any | None = None,
    ) -> GraphMessage:
        """Execute this node with structured predecessor messages."""

        from gogagent.graph.executor import execute_node

        return execute_node(self, problem, inputs, context=context)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable representation."""

        return {
            "node_id": self.node_id,
            "name": self.name,
            "executor": _executor_to_dict(self.executor),
            "depth": self.depth,
            "final_node_id": self.final_node_id,
            "left_step": self.left_step,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        """Build a node from a JSON mapping."""

        return cls(
            node_id=str(data["node_id"]),
            name=str(data.get("name", data["node_id"])),
            executor=_executor_from_dict(data.get("executor")),
            depth=int(data.get("depth", 1)),
            final_node_id=_optional_str(data.get("final_node_id")),
            left_step=int(data.get("left_step", 1)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Graph:
    """Serializable directed execution graph.

    A Graph can itself be used as a Node executor, which is the core GOG
    structure for the refactor.
    """

    graph_id: str = "graph"
    in_node: str | None = None
    out_node: str | None = None
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    max_steps: int = 16
    metadata: JsonDict = field(default_factory=dict)

    def execute(self, problem: Mapping[str, Any], *, context: Any | None = None) -> GraphMessage:
        """Execute this graph with bounded DAG-style scheduling."""

        from gogagent.graph.executor import execute_graph

        return execute_graph(self, problem, context=context)

    def add_node(self, node: Node) -> None:
        """Add a node and initialize endpoints if this is the first node."""

        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node
        if self.in_node is None:
            self.in_node = node.node_id
        if self.out_node is None:
            self.out_node = node.node_id

    def add_edge(self, edge: Edge) -> None:
        """Add an edge after validating endpoint ids."""

        if edge.source not in self.nodes:
            raise ValueError(f"edge source does not exist: {edge.source}")
        if edge.target not in self.nodes:
            raise ValueError(f"edge target does not exist: {edge.target}")
        self.edges.append(edge)

    def predecessors(self, node_id: str) -> list[str]:
        """Return predecessor ids in edge insertion order."""

        return [edge.source for edge in self.edges if edge.target == node_id]

    def successors(self, node_id: str) -> list[str]:
        """Return successor ids in edge insertion order."""

        return [edge.target for edge in self.edges if edge.source == node_id]

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable representation."""

        return {
            "graph_id": self.graph_id,
            "in_node": self.in_node,
            "out_node": self.out_node,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Graph":
        """Build a graph from a JSON mapping."""

        raw_nodes = data.get("nodes") or {}
        if isinstance(raw_nodes, Mapping):
            nodes = {
                str(node_id): Node.from_dict(node_data)
                for node_id, node_data in raw_nodes.items()
            }
        else:
            node_list = [Node.from_dict(node_data) for node_data in raw_nodes]
            nodes = {node.node_id: node for node in node_list}

        return cls(
            graph_id=str(data.get("graph_id", "graph")),
            in_node=_optional_str(data.get("in_node")),
            out_node=_optional_str(data.get("out_node")),
            nodes=nodes,
            edges=[Edge.from_dict(edge) for edge in data.get("edges", [])],
            max_steps=int(data.get("max_steps", 16)),
            metadata=dict(data.get("metadata") or {}),
        )


def _executor_to_dict(executor: Any) -> JsonDict:
    if isinstance(executor, Graph):
        return {
            "kind": "graph",
            "graph": executor.to_dict(),
        }
    if hasattr(executor, "to_dict"):
        return {
            "kind": "agent",
            "agent": executor.to_dict(),
        }
    raise TypeError(f"unsupported node executor: {type(executor).__name__}")


def _executor_from_dict(data: Any) -> Any:
    if not isinstance(data, Mapping):
        raise ValueError("node executor must be a mapping")

    kind = str(data.get("kind", "agent"))
    if kind == "graph":
        return Graph.from_dict(data["graph"])
    if kind == "agent":
        from gogagent.agents.registry import agent_from_dict

        return agent_from_dict(data.get("agent", data))
    raise ValueError(f"unknown executor kind: {kind}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
