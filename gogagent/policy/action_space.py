"""Flat policy action-space helpers."""

from __future__ import annotations

from gogagent.actions.base import ActionName
from gogagent.actions.registry import ACTION_ORDER


ACTION_SPACE: tuple[ActionName, ...] = ACTION_ORDER
_ACTION_TO_INDEX: dict[ActionName, int] = {
    action: index
    for index, action in enumerate(ACTION_SPACE)
}


def action_to_index(action: ActionName | str) -> int:
    """Return the stable policy index for an action."""

    return _ACTION_TO_INDEX[ActionName(action)]


def index_to_action(index: int) -> ActionName:
    """Return the action at a stable policy index."""

    if index < 0 or index >= len(ACTION_SPACE):
        raise IndexError(f"action index out of range: {index}")
    return ACTION_SPACE[index]


def action_count() -> int:
    """Return the size of the flat policy action space."""

    return len(ACTION_SPACE)


__all__ = [
    "ACTION_SPACE",
    "action_count",
    "action_to_index",
    "index_to_action",
]
