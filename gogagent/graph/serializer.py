"""Serialization helpers for Graph-of-Graphs artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from gogagent.graph.schema import Graph


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    """Convert a graph to a JSON-serializable dictionary."""

    return graph.to_dict()


def graph_from_dict(data: Mapping[str, Any]) -> Graph:
    """Load a graph from a dictionary."""

    return Graph.from_dict(data)


def graph_to_json(graph: Graph, *, indent: int | None = 2) -> str:
    """Convert a graph to a JSON string."""

    return json.dumps(graph.to_dict(), ensure_ascii=False, indent=indent)


def graph_from_json(text: str) -> Graph:
    """Load a graph from a JSON string."""

    return Graph.from_dict(json.loads(text))


def save_graph(graph: Graph, path: str | Path) -> None:
    """Save a graph JSON artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(graph_to_json(graph), encoding="utf-8")


def load_graph(path: str | Path) -> Graph:
    """Load a graph JSON artifact."""

    return graph_from_json(Path(path).read_text(encoding="utf-8"))
