"""Initial graph factory for BC teacher trajectory generation."""

from __future__ import annotations

from gogagent.agents.registry import create_agent
from gogagent.graph.schema import Graph, Node


def make_initial_graph(*, graph_id: str = "bc_initial_graph") -> Graph:
    """Return the canonical initial graph used for BC action replay."""

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
        metadata={"created_for": "bc_teacher_trajectory"},
    )
