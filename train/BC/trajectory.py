"""Replay and validate teacher action sequences for BC datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.actions.mask import compute_action_mask_with_reasons
from gogagent.actions.registry import ACTION_ORDER, apply_action
from gogagent.graph.schema import Graph


@dataclass(frozen=True)
class TrajectoryStep:
    """One supervised BC step: graph state before action plus target action."""

    step: int
    graph_before: Mapping[str, Any]
    legal_actions: tuple[ActionName, ...]
    target_action: ActionName
    legal: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable step."""

        return {
            "step": self.step,
            "graph_before": dict(self.graph_before),
            "legal_actions": [action.value for action in self.legal_actions],
            "target_action": self.target_action.value,
            "legal": self.legal,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrajectoryBuildResult:
    """Validated trajectory plus replay diagnostics."""

    valid: bool
    actions: tuple[ActionName, ...]
    steps: tuple[TrajectoryStep, ...]
    final_graph: Mapping[str, Any]
    invalid_steps: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trajectory build result."""

        return {
            "valid": self.valid,
            "actions": [action.value for action in self.actions],
            "steps": [step.to_dict() for step in self.steps],
            "final_graph": dict(self.final_graph),
            "invalid_steps": [dict(step) for step in self.invalid_steps],
        }


def build_trajectory(
    *,
    initial_graph: Graph,
    actions: tuple[ActionName, ...],
    constraints: ActionConstraints,
    require_stop: bool = True,
) -> TrajectoryBuildResult:
    """Replay teacher actions and record step-level BC examples."""

    graph = initial_graph
    steps: list[TrajectoryStep] = []
    invalid_steps: list[Mapping[str, Any]] = []
    replayed_actions: list[ActionName] = []

    for step_index, action in enumerate(actions, start=1):
        decisions = compute_action_mask_with_reasons(graph, constraints)
        legal_actions = tuple(
            candidate for candidate in ACTION_ORDER if decisions[candidate].legal
        )
        decision = decisions[action]
        step = TrajectoryStep(
            step=step_index,
            graph_before=graph.to_dict(),
            legal_actions=legal_actions,
            target_action=action,
            legal=decision.legal,
            reason=decision.reason,
        )
        steps.append(step)

        if not decision.legal:
            invalid_steps.append(
                {
                    "step": step_index,
                    "action": action.value,
                    "reason": decision.reason,
                    "legal_actions": [candidate.value for candidate in legal_actions],
                }
            )
            break

        replayed_actions.append(action)
        if action == ActionName.STOP:
            if step_index < len(actions):
                invalid_steps.append(
                    {
                        "step": step_index + 1,
                        "action": actions[step_index].value,
                        "reason": "trajectory contains actions after STOP",
                        "legal_actions": [],
                    }
                )
            break
        graph = apply_action(graph, action)

    if require_stop and (not replayed_actions or replayed_actions[-1] != ActionName.STOP):
        invalid_steps.append(
            {
                "step": len(steps) + 1,
                "action": None,
                "reason": "trajectory did not terminate with STOP",
                "legal_actions": [],
            }
        )

    return TrajectoryBuildResult(
        valid=not invalid_steps,
        actions=tuple(replayed_actions),
        steps=tuple(steps),
        final_graph=graph.to_dict(),
        invalid_steps=tuple(invalid_steps),
    )
