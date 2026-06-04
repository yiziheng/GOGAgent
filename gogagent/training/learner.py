"""Minimal DQN-style learner for the lightweight hierarchical GCN policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gogagent.core.actions import MacroAction
from gogagent.core.types import OrgGraphSnapshot
from gogagent.policy.hierarchical_gnn import HierarchicalGNNPolicy
from gogagent.training.replay import ReplayTransition


@dataclass(frozen=True)
class LearnerStep:
    action: MacroAction
    target: float
    loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "target": self.target,
            "loss": self.loss,
        }


class DQNStyleLearner:
    """One-step TD learner over replay transitions.

    The learner is intentionally compact: it updates the policy's action head
    from stored dense rewards while leaving terminal oracle scoring isolated in
    the training recorder.
    """

    def __init__(
        self,
        policy: HierarchicalGNNPolicy,
        *,
        gamma: float = 0.9,
        learning_rate: float = 0.01,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.policy = policy
        self.gamma = gamma
        self.learning_rate = learning_rate

    def train_one(
        self,
        *,
        graph: OrgGraphSnapshot,
        next_graph: OrgGraphSnapshot,
        transition: ReplayTransition,
    ) -> LearnerStep:
        state_embedding = self.policy.state_embedding(graph, transition.state)
        next_embedding = self.policy.state_embedding(next_graph, transition.next_state)
        with torch.no_grad():
            next_values = [
                float(self.policy.q_value(next_embedding, MacroAction(action_name)).item())
                for action_name, is_legal in transition.next_action_mask.items()
                if is_legal and action_name in MacroAction._value2member_map_
            ]
        bootstrap = 0.0 if transition.done or not next_values else max(next_values)
        target = transition.reward + self.gamma * bootstrap
        loss = self.policy.td_update(
            state_embedding=state_embedding,
            action=transition.action,
            target=target,
            learning_rate=self.learning_rate,
        )
        return LearnerStep(
            action=transition.action,
            target=round(target, 6),
            loss=loss,
        )
