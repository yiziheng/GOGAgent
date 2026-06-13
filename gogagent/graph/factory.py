"""Canonical graph factories shared by BC, RL, and eval."""

from __future__ import annotations

from gogagent.agents.registry import create_agent
from gogagent.graph.schema import Edge, Graph, Node


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


def make_solver_supervisor_graph(*, graph_id: str = "solver_supervisor_graph") -> Graph:
    """Return the fixed Solver -> Supervisor graph used for supervision experiments."""

    return Graph(
        graph_id=graph_id,
        in_node="solver",
        out_node="supervisor",
        nodes={
            "solver": Node(
                node_id="solver",
                name="SolverAgent",
                executor=create_agent("SolverAgent"),
                depth=1,
            ),
            "supervisor": Node(
                node_id="supervisor",
                name="SupervisorAgent",
                executor=create_agent("SupervisorAgent"),
                depth=1,
            ),
        },
        edges=[Edge(source="solver", target="supervisor")],
        metadata={"created_for": "fixed_solver_supervisor"},
    )
