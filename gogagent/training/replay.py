"""Replay transitions and label-blind dense construction rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from gogagent.core.actions import MacroAction
from gogagent.core.types import ExecutionResult, OrgGraphSnapshot


_STATUS_VALUE = {
    "failed": -0.4,
    "unknown": 0.0,
    "needs_review": 0.0,
    "observed": 0.15,
    "ready": 0.35,
    "passed": 0.45,
}
_CONFIDENCE_VALUE = {"low": -0.15, "unknown": 0.0, "medium": 0.05, "high": 0.2}


@dataclass(frozen=True)
class DenseRewardBreakdown:
    """Label-blind reward attached to one graph construction edit."""

    visible_quality_delta: float
    issue_resolution_delta: float
    token_penalty: float
    call_penalty: float
    complexity_penalty: float
    step_penalty: float
    reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayTransition:
    """Serializable RL transition; never stores gold labels."""

    graph_id: str
    next_graph_id: str
    action: MacroAction
    reward: float
    done: bool
    state: Mapping[str, Any]
    next_state: Mapping[str, Any]
    action_mask: Mapping[str, bool]
    next_action_mask: Mapping[str, bool]
    dense_reward: DenseRewardBreakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "next_graph_id": self.next_graph_id,
            "action": self.action.value,
            "reward": self.reward,
            "done": self.done,
            "state": dict(self.state),
            "next_state": dict(self.next_state),
            "action_mask": dict(self.action_mask),
            "next_action_mask": dict(self.next_action_mask),
            "dense_reward": self.dense_reward.to_dict(),
            "label_blind": True,
        }


class DenseConstructionReward:
    """Reward each module-level construction step before terminal oracle scoring."""

    def __init__(
        self,
        *,
        token_weight: float = 0.0005,
        call_weight: float = 0.02,
        complexity_weight: float = 0.03,
        step_penalty: float = 0.01,
    ) -> None:
        self.token_weight = token_weight
        self.call_weight = call_weight
        self.complexity_weight = complexity_weight
        self.step_penalty = step_penalty

    def score(
        self,
        before_graph: OrgGraphSnapshot,
        after_graph: OrgGraphSnapshot,
        before: ExecutionResult,
        after: ExecutionResult,
    ) -> DenseRewardBreakdown:
        before_quality = _visible_quality(before)
        after_quality = _visible_quality(after)
        visible_delta = after_quality - before_quality
        issue_delta = len(before.visible_feedback.issue_codes) - len(after.visible_feedback.issue_codes)
        token_penalty = self.token_weight * max(after.token_cost, 0)
        call_penalty = self.call_weight * max(after.llm_calls, 0)
        complexity_penalty = self.complexity_weight * max(
            _complexity(after_graph) - _complexity(before_graph),
            0,
        )
        reward = (
            visible_delta
            + 0.1 * issue_delta
            - token_penalty
            - call_penalty
            - complexity_penalty
            - self.step_penalty
        )
        return DenseRewardBreakdown(
            visible_quality_delta=round(visible_delta, 6),
            issue_resolution_delta=round(0.1 * issue_delta, 6),
            token_penalty=round(token_penalty, 6),
            call_penalty=round(call_penalty, 6),
            complexity_penalty=round(complexity_penalty, 6),
            step_penalty=round(self.step_penalty, 6),
            reward=round(reward, 6),
        )


def _visible_quality(result: ExecutionResult) -> float:
    feedback = result.visible_feedback
    return (
        _STATUS_VALUE.get(feedback.status, 0.0)
        + _CONFIDENCE_VALUE.get(feedback.confidence_bucket, 0.0)
        - 0.05 * len(feedback.issue_codes)
        - (0.05 if feedback.disagreement_level != "none" else 0.0)
    )


def _complexity(graph: OrgGraphSnapshot) -> int:
    graph_agents = sum(1 for node in graph.nodes if node.node_kind == "graph")
    internal_nodes = sum(len(node.internal_nodes) for node in graph.nodes)
    return len(graph.nodes) + len(graph.edges) + graph_agents + internal_nodes
