"""Masked stochastic action sampling for RL rollouts."""

from __future__ import annotations

import torch

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.graph.schema import Graph
from gogagent.policy import (
    ACTION_SPACE,
    GraphEncoder,
    PolicyNetwork,
    mask_action_logits,
    top_action_scores,
)
from train.RL.buffer import RolloutStep


def sample_action_step(
    *,
    graph: Graph,
    task_embedding: torch.Tensor,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    constraints: ActionConstraints,
    temperature: float,
    step: int,
    generator: torch.Generator | None = None,
) -> RolloutStep:
    """Sample one legal action and record policy diagnostics."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        graph_embedding = graph_encoder(graph)
        logits = policy_network(graph_embedding, task_embedding)
        masked_logits, legal_actions = mask_action_logits(
            logits,
            graph,
            constraints,
            action_space=ACTION_SPACE,
            temperature=temperature,
        )
        if not legal_actions:
            raise RuntimeError("no legal actions available for RL sampling")
        probabilities = torch.softmax(masked_logits, dim=-1)
        selected_index = int(
            torch.multinomial(probabilities, num_samples=1, generator=generator).item()
        )
        action = ACTION_SPACE[selected_index]
        if action not in legal_actions:
            raise RuntimeError(f"sampled illegal action after masking: {action.value}")

    return RolloutStep(
        step=step,
        graph_before=graph.to_dict(),
        action=action,
        legal_actions=tuple(legal_actions),
        top_actions=top_action_scores(masked_logits),
    )
