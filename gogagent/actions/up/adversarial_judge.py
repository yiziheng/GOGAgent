"""UP template for AdversarialJudgeAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_edge, make_graph, make_node, node_id


INTERNAL_AGENT_NAMES = (
    "ShuffledMMLUSolverAgent",
    "ShuffledMMLUSolverAgent",
    "MMLUMajorityVoteAgent",
)


def build_subgraph(target_node: Any) -> Any:
    """Expand AdversarialJudgeAgent into two independent shuffled-choice votes."""

    parent_id = node_id(target_node)
    vote_1 = f"{parent_id}_mmlu_shuffle_vote_1"
    vote_2 = f"{parent_id}_mmlu_shuffle_vote_2"
    voter = f"{parent_id}_mmlu_majority_vote_3"
    return make_graph(
        in_node=vote_1,
        out_node=voter,
        nodes={
            vote_1: make_node(
                "ShuffledMMLUSolverAgent",
                node_id_value=vote_1,
                metadata={"created_by": "UP", "up_parent": parent_id},
            ),
            vote_2: make_node(
                "ShuffledMMLUSolverAgent",
                node_id_value=vote_2,
                metadata={"created_by": "UP", "up_parent": parent_id},
            ),
            voter: make_node(
                "MMLUMajorityVoteAgent",
                node_id_value=voter,
                metadata={"created_by": "UP", "up_parent": parent_id},
            ),
        },
        edges=[make_edge(vote_1, voter), make_edge(vote_2, voter)],
        metadata={
            "graph_type": "shuffled_choice_self_consistency",
            "created_by": "UP",
            "parent": parent_id,
        },
    )
