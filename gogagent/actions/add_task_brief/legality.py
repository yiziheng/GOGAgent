"""Legality rules for ADD_TASK_BRIEF."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import ActionConstraints, LegalityResult, expansion_fits, hierarchy_depth


def is_legal(graph: Any, constraints: ActionConstraints) -> LegalityResult:
    """ADD_TASK_BRIEF is legal when adding one node stays within constraints."""

    return expansion_fits(
        graph,
        constraints,
        added_nodes=1,
        resulting_depth=hierarchy_depth(graph),
    )
