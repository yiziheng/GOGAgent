"""STOP action package."""

from gogagent.actions.stop.apply import apply
from gogagent.actions.stop.legality import is_legal
from gogagent.actions.stop.spec import SPEC

__all__ = ["SPEC", "apply", "is_legal"]
