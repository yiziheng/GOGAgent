"""GRPO-style losses for graph-construction RL."""

from __future__ import annotations

from typing import Any

import torch

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.graph.schema import Graph
from gogagent.policy import ACTION_SPACE, GraphEncoder, PolicyNetwork, action_to_index, mask_action_logits
from train.RL.buffer import TrajectoryRollout


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Return GRPO-style within-question relative advantages."""

    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    return [reward - mean_reward for reward in rewards]


def grpo_rollout_loss(
    *,
    rollout: TrajectoryRollout,
    advantage: float,
    task_embedding: torch.Tensor,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    reference_graph_encoder: GraphEncoder,
    reference_policy_network: PolicyNetwork,
    constraints: ActionConstraints,
    temperature: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return one trajectory policy-gradient loss plus reference KL."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")
    if not rollout.steps:
        zero = torch.zeros((), device=task_embedding.device, requires_grad=True)
        return zero, {"logprob_sum": 0.0, "kl": 0.0, "step_count": 0}

    logprob_sum = torch.zeros((), device=task_embedding.device)
    kl_sum = torch.zeros((), device=task_embedding.device)
    for step in rollout.steps:
        graph = Graph.from_dict(step.graph_before)
        action = ActionName(step.action)
        current_logits = policy_network(graph_encoder(graph), task_embedding)
        current_masked, _ = mask_action_logits(
            current_logits,
            graph,
            constraints,
            action_space=ACTION_SPACE,
            temperature=temperature,
        )
        current_log_probs = torch.log_softmax(current_masked, dim=-1)
        logprob_sum = logprob_sum + current_log_probs[action_to_index(action)]

        if kl_beta > 0:
            with torch.no_grad():
                reference_logits = reference_policy_network(
                    reference_graph_encoder(graph),
                    task_embedding,
                )
                reference_masked, _ = mask_action_logits(
                    reference_logits,
                    graph,
                    constraints,
                    action_space=ACTION_SPACE,
                    temperature=temperature,
                )
                reference_log_probs = torch.log_softmax(reference_masked, dim=-1)
            current_probs = torch.softmax(current_masked, dim=-1)
            finite = torch.isfinite(current_log_probs) & torch.isfinite(reference_log_probs)
            step_kl = torch.sum(
                current_probs[finite] * (current_log_probs[finite] - reference_log_probs[finite])
            )
            kl_sum = kl_sum + step_kl

    advantage_tensor = torch.tensor(
        float(advantage),
        device=task_embedding.device,
        dtype=logprob_sum.dtype,
    )
    policy_loss = -advantage_tensor.detach() * logprob_sum
    kl_loss = float(kl_beta) * kl_sum
    loss = policy_loss + kl_loss
    return loss, {
        "logprob_sum": float(logprob_sum.detach().cpu().item()),
        "kl": float(kl_sum.detach().cpu().item()),
        "step_count": len(rollout.steps),
    }
