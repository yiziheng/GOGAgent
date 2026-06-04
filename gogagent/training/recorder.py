"""Train-only bridge from oracle outcomes to reusable GoG experiences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gogagent.core.types import ExperienceRecord
from gogagent.gog.memory import OrganizationGoG
from gogagent.oracle.base import TrainOnlyRewardOracle
from gogagent.training.credit import (
    FineGrainedCreditAssigner,
    StepCredit,
    TransitionCreditInput,
)


@dataclass(frozen=True)
class TrainingSummary:
    """Compact training metrics. Gold labels are deliberately not retained."""

    terminal_reward: float
    credits: tuple[StepCredit, ...]
    experience_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "terminal_reward": self.terminal_reward,
            "credits": [credit.to_dict() for credit in self.credits],
            "experience_count": self.experience_count,
        }


class TrainingEpisodeRecorder:
    """Score one episode and persist per-edit credit records for audit."""

    def __init__(
        self,
        oracle: TrainOnlyRewardOracle,
        credit_assigner: FineGrainedCreditAssigner | None = None,
    ) -> None:
        self.oracle = oracle
        self.credit_assigner = credit_assigner or FineGrainedCreditAssigner()

    def record(
        self,
        *,
        gog: OrganizationGoG,
        task: Mapping[str, Any],
        task_features: Mapping[str, Any],
        output: str,
        gold: Any,
        steps: tuple[TransitionCreditInput, ...],
    ) -> TrainingSummary:
        """Store shaped returns but never retain ``gold`` or raw oracle inputs."""

        terminal_reward = self.oracle.score(task, output, gold)
        credits = self.credit_assigner.assign(steps, terminal_reward)
        successful = terminal_reward > 0.0
        for step, credit in zip(steps, credits, strict=True):
            gog.add_experience(
                ExperienceRecord(
                    graph_id=step.graph_id,
                    domain=str(task_features.get("dataset", task_features.get("domain", ""))),
                    task_features=dict(task_features),
                    feedback_type=step.feedback_type,
                    cost_bucket=_cost_bucket(step.token_cost),
                    action=step.action,
                    return_value=credit.return_value,
                    success=successful,
                )
            )
        return TrainingSummary(terminal_reward, credits, len(credits))


def _cost_bucket(token_cost: int) -> str:
    if token_cost < 128:
        return "low"
    if token_cost < 512:
        return "medium"
    return "high"
