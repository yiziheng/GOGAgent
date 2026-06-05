"""UP action package."""

from gogagent.actions.up.apply import apply
from gogagent.actions.up.legality import is_legal
from gogagent.actions.up.spec import SPEC

__all__ = ["SPEC", "apply", "is_legal"]
