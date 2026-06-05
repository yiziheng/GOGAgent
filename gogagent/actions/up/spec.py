"""Static metadata for UP."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.UP,
    description=(
        "Upgrade the current graph's last atomic node into a type-specific "
        "subgraph. Solver becomes PlanSketch -> Solver; AdversarialJudge "
        "becomes Challenger -> Defender -> Judge; verifier and context nodes "
        "expand into their own internal helper graphs."
    ),
    complexity_penalty=-0.02,
    is_expansion=True,
)
