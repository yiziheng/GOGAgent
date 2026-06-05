"""Static metadata for ADD_PLAN_SKETCH."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.ADD_PLAN_SKETCH,
    description=(
        "Add a PlanSketchAgent near the input side. It produces at most two "
        "concise solving steps and does not create an execution loop."
    ),
    complexity_penalty=-0.01,
    is_expansion=True,
)
