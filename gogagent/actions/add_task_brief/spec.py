"""Static metadata for ADD_TASK_BRIEF."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.ADD_TASK_BRIEF,
    description="Add a TaskBriefAgent as the first context-building node.",
    complexity_penalty=-0.01,
    is_expansion=True,
)
