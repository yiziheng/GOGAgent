"""Canonical graph factories shared by BC, RL, and eval."""

from __future__ import annotations

from gogagent.agents.registry import create_agent
from gogagent.graph.schema import Graph, Node


def make_initial_graph(*, graph_id: str = "initial_graph") -> Graph:
    """Return the canonical minimal executable graph: one SolverAgent node."""

    return Graph(
        graph_id=graph_id,
        in_node="solver",
        out_node="solver",
        nodes={
            "solver": Node(
                node_id="solver",
                name="SolverAgent",
                executor=create_agent("SolverAgent"),
                depth=1,
            )
        },
        edges=[],
        metadata={"created_for": "graph_construction"},
    )
