"""Fine-grained macro-edit credit assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gogagent.core.actions import MacroAction


@dataclass(frozen=True)
class TransitionCreditInput:
    """Training-only description of one selected graph edit."""

    graph_id: str
    action: MacroAction
    token_cost: int
    feedback_type: str
    visible_delta: float = 0.0


@dataclass(frozen=True)
class StepCredit:
    """A shaped return attached to one graph-construction action."""

    graph_id: str
    action: MacroAction
    return_value: float
    token_cost: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


class FineGrainedCreditAssigner:
    """Push terminal quality down to each edit while penalizing local cost."""

    def __init__(self, gamma: float = 0.9, cost_weight: float = 0.001) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if cost_weight < 0.0:
            raise ValueError("cost_weight must be non-negative")
        self.gamma = gamma
        self.cost_weight = cost_weight

    def assign(
        self,
        steps: tuple[TransitionCreditInput, ...],
        terminal_reward: float,
    ) -> tuple[StepCredit, ...]:
        credits = []
        final_index = len(steps) - 1
        for index, step in enumerate(steps):
            distance_to_terminal = final_index - index
            shaped_return = (
                terminal_reward * (self.gamma**distance_to_terminal)
                + step.visible_delta
                - self.cost_weight * step.token_cost
            )
            credits.append(
                StepCredit(
                    graph_id=step.graph_id,
                    action=step.action,
                    return_value=round(shaped_return, 6),
                    token_cost=step.token_cost,
                )
            )
        return tuple(credits)
