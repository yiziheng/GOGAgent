"""Flat graph-construction actions for the refactored GOGAgent runtime."""

from gogagent.actions.base import ActionConstraints, ActionName, ActionSpec, LegalityResult
from gogagent.actions.mask import compute_action_mask, compute_action_mask_with_reasons
from gogagent.actions.registry import ACTION_ORDER, apply_action, get_action_spec, is_action_legal

__all__ = [
    "ACTION_ORDER",
    "ActionConstraints",
    "ActionName",
    "ActionSpec",
    "LegalityResult",
    "apply_action",
    "compute_action_mask",
    "compute_action_mask_with_reasons",
    "get_action_spec",
    "is_action_legal",
]
