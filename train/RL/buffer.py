"""Lightweight rollout data structures for RL policy refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gogagent.actions.base import ActionName


@dataclass
class RolloutStep:
    """One sampled graph-construction decision."""

    step: int
    graph_before: Mapping[str, Any]
    action: ActionName
    legal_actions: tuple[ActionName, ...]
    top_actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "graph_before": dict(self.graph_before),
            "action": self.action.value,
            "legal_actions": [action.value for action in self.legal_actions],
            "top_actions": list(self.top_actions),
            "legal": self.action in self.legal_actions,
        }

@dataclass
class TrajectoryRollout:
    """One complete sampled graph rollout and its reward signal."""

    epoch: int
    example_index: int
    group_index: int
    rollout_index: int
    task_id: str
    dataset: str
    subject: str | None
    question: str | None
    gold: Any
    action_sequence: list[str]
    steps: list[RolloutStep]
    final_graph: Mapping[str, Any]
    status: str
    reward: float
    reward_breakdown: Mapping[str, Any]
    prediction: Any | None = None
    correct: bool | None = None
    format_valid: bool | None = None
    output: Mapping[str, Any] | None = None
    error: str | None = None
    llm_call_count: int = 0
    llm_calls: list[Mapping[str, Any]] = field(default_factory=list)
    item_dir: str | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    advantage: float | None = None
    loss: float | None = None
    logprob_sum: float | None = None
    kl: float | None = None

    @property
    def stopped(self) -> bool:
        return bool(self.action_sequence and self.action_sequence[-1] == ActionName.STOP.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "example_index": self.example_index,
            "group_index": self.group_index,
            "rollout_index": self.rollout_index,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "subject": self.subject,
            "question": self.question,
            "gold": self.gold,
            "action_sequence": list(self.action_sequence),
            "steps": [step.to_dict() for step in self.steps],
            "final_graph": dict(self.final_graph),
            "status": self.status,
            "reward": self.reward,
            "reward_breakdown": dict(self.reward_breakdown),
            "prediction": self.prediction,
            "correct": self.correct,
            "format_valid": self.format_valid,
            "output": dict(self.output) if self.output is not None else None,
            "error": self.error,
            "llm_call_count": self.llm_call_count,
            "llm_calls": [dict(call) for call in self.llm_calls],
            "item_dir": self.item_dir,
            "artifacts": dict(self.artifacts),
            "advantage": self.advantage,
            "loss": self.loss,
            "logprob_sum": self.logprob_sum,
            "kl": self.kl,
            "stopped": self.stopped,
        }
