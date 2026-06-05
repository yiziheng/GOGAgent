"""Action-mask computation for the flat action space."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import ActionConstraints, ActionName, LegalityResult
from gogagent.actions.registry import ACTION_ORDER, is_action_legal


def compute_action_mask(
    graph: Any,
    constraints: ActionConstraints | None = None,
) -> dict[ActionName, bool]:
    """Return a boolean legality table for every policy action."""

    decisions = compute_action_mask_with_reasons(graph, constraints)
    return {action: result.legal for action, result in decisions.items()}


def compute_action_mask_with_reasons(
    graph: Any,
    constraints: ActionConstraints | None = None,
) -> dict[ActionName, LegalityResult]:
    """Return action legality decisions with compact diagnostic reasons."""

    active_constraints = constraints or ActionConstraints()
    return {
        action: is_action_legal(graph, action, active_constraints)
        for action in ACTION_ORDER
    }
