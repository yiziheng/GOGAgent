#!/usr/bin/env python3
"""Round 2 policy smoke checks: GOG -> GCN -> legal action."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.actions.registry import apply_action, is_action_legal
from gogagent.agents.registry import create_agent
from gogagent.graph.schema import Edge, Graph, Node
from gogagent.policy import (
    ACTION_SPACE,
    GraphEncoder,
    PolicyNetwork,
    action_count,
    action_to_index,
    encode_graph,
    encode_task,
    feature_dim,
    mask_action_logits,
    select_action,
    task_embedding_dim,
)


def main() -> None:
    torch.manual_seed(7)
    test_atomic_graph_to_legal_action()
    test_subgraph_encoding_to_legal_action()
    test_mask_blocks_illegal_high_logit()
    print("Round2 policy graph encoding passed")


def test_atomic_graph_to_legal_action() -> None:
    graph = make_solver_graph()
    graph = apply_action(graph, ActionName.ADD_PLAN_SKETCH)

    graph_tensor = encode_graph(graph)
    assert graph_tensor.node_features.shape == (2, feature_dim())
    assert graph_tensor.edge_index.shape == (2, 1)
    assert graph_tensor.node_ids == ("plan_sketch", "solver")

    action = policy_select(graph)
    assert is_action_legal(graph, action).legal


def test_subgraph_encoding_to_legal_action() -> None:
    graph = apply_action(make_solver_graph(), ActionName.UP)

    graph_tensor = encode_graph(graph)
    assert graph_tensor.node_features.shape == (1, feature_dim())
    assert graph_tensor.node_features[0, list(graph_tensor.feature_names).index("is_subgraph")] == 1

    embedding = GraphEncoder(embedding_dim=32)(graph)
    assert embedding.shape == (32,)

    action = policy_select(graph)
    assert is_action_legal(graph, action).legal


def test_mask_blocks_illegal_high_logit() -> None:
    graph = make_solver_graph(include_planner=True)
    constraints = ActionConstraints(max_depth=2, max_nodes=8)
    logits = torch.zeros(action_count(), dtype=torch.float32)
    logits[action_to_index(ActionName.ADD_PLAN_SKETCH)] = 1000.0
    logits[action_to_index(ActionName.STOP)] = 1.0

    masked_logits, legal_actions = mask_action_logits(logits, graph, constraints)
    assert ActionName.ADD_PLAN_SKETCH not in legal_actions
    assert torch.isneginf(masked_logits[action_to_index(ActionName.ADD_PLAN_SKETCH)])

    selected = select_action(logits, graph, constraints, mode="argmax")
    assert selected != ActionName.ADD_PLAN_SKETCH
    assert is_action_legal(graph, selected, constraints).legal


def policy_select(graph: Graph) -> ActionName:
    graph_encoder = GraphEncoder(embedding_dim=32)
    task_dim = task_embedding_dim()
    policy = PolicyNetwork(graph_embedding_dim=32, task_embedding_dim=task_dim)
    graph_embedding = graph_encoder(graph)
    task_embedding = encode_task(
        {
            "question": "Which option is correct?",
            "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
        }
    )
    assert task_embedding.shape == (task_dim,)
    logits = policy(graph_embedding, task_embedding)
    assert logits.shape == (len(ACTION_SPACE),)
    return select_action(logits, graph, ActionConstraints(max_depth=2, max_nodes=8))


def make_solver_graph(*, include_planner: bool = False) -> Graph:
    nodes = {
        "solver": Node(
            node_id="solver",
            name="SolverAgent",
            executor=create_agent("SolverAgent"),
            depth=1,
        )
    }
    edges: list[Edge] = []
    in_node = "solver"
    if include_planner:
        nodes["plan_sketch"] = Node(
            node_id="plan_sketch",
            name="PlanSketchAgent",
            executor=create_agent("PlanSketchAgent"),
            depth=1,
        )
        edges.append(Edge(source="plan_sketch", target="solver"))
        in_node = "plan_sketch"
    return Graph(
        graph_id="policy_test_graph",
        in_node=in_node,
        out_node="solver",
        nodes=nodes,
        edges=edges,
    )


if __name__ == "__main__":
    main()
