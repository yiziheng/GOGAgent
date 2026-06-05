"""Static metadata for ADD_FORMAT_VERIFIER."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.ADD_FORMAT_VERIFIER,
    description="Add a FormatVerifierAgent as the final output-side node.",
    complexity_penalty=-0.01,
    is_expansion=True,
)
