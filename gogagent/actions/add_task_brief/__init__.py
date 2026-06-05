"""ADD_TASK_BRIEF action package."""

from gogagent.actions.add_task_brief.apply import apply
from gogagent.actions.add_task_brief.legality import is_legal
from gogagent.actions.add_task_brief.spec import SPEC

__all__ = ["SPEC", "apply", "is_legal"]
