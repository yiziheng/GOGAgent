"""Static metadata for STOP."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.STOP,
    description="Stop graph construction and execute the current graph.",
    complexity_penalty=0.0,
    is_expansion=False,
)
