"""Legality rules for STOP."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import ActionConstraints, LegalityResult


def is_legal(graph: Any, constraints: ActionConstraints) -> LegalityResult:
    """STOP is always legal."""

    return LegalityResult(True)
