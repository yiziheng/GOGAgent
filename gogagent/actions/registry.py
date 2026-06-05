"""Registry for flat graph-construction actions."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gogagent.actions.base import ActionConstraints, ActionName, ActionSpec, LegalityResult


ACTION_ORDER: tuple[ActionName, ...] = (
    ActionName.UP,
    ActionName.ADD_TASK_BRIEF,
    ActionName.ADD_PLAN_SKETCH,
    ActionName.ADD_ADVERSARIAL_JUDGE,
    ActionName.ADD_FORMAT_VERIFIER,
    ActionName.STOP,
)

_ACTION_PACKAGES: dict[ActionName, str] = {
    ActionName.UP: "gogagent.actions.up",
    ActionName.ADD_TASK_BRIEF: "gogagent.actions.add_task_brief",
    ActionName.ADD_PLAN_SKETCH: "gogagent.actions.add_plan_sketch",
    ActionName.ADD_ADVERSARIAL_JUDGE: "gogagent.actions.add_adversarial_judge",
    ActionName.ADD_FORMAT_VERIFIER: "gogagent.actions.add_format_verifier",
    ActionName.STOP: "gogagent.actions.stop",
}


def get_action_spec(action: ActionName | str) -> ActionSpec:
    """Return static metadata for an action."""

    action_name = ActionName(action)
    module = import_module(f"{_ACTION_PACKAGES[action_name]}.spec")
    return module.SPEC


def is_action_legal(
    graph: Any,
    action: ActionName | str,
    constraints: ActionConstraints | None = None,
) -> LegalityResult:
    """Return the v1 legality decision for an action."""

    action_name = ActionName(action)
    module = import_module(f"{_ACTION_PACKAGES[action_name]}.legality")
    return module.is_legal(graph, constraints or ActionConstraints())


def apply_action(graph: Any, action: ActionName | str) -> Any:
    """Apply an already-legal action to a graph."""

    action_name = ActionName(action)
    module = import_module(f"{_ACTION_PACKAGES[action_name]}.apply")
    return module.apply(graph)


def teacher_action_descriptions() -> str:
    """Return human-readable action meanings for DeepSeek teacher prompts."""

    lines = []
    for action in ACTION_ORDER:
        spec = get_action_spec(action)
        lines.append(f"- {spec.name.value}: {spec.description}")
    return "\n".join(lines)
