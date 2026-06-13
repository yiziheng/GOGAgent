"""Static metadata for UP."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.UP,
    description=(
        "Upgrade the current graph's last atomic node into a type-specific "
        "subgraph. Solver becomes a light PlanSketch -> Solver subgraph and "
        "should be used mainly for clear multi-step reasoning. On MMLU, "
        "AdversarialJudge becomes two shuffled-option solver votes followed by "
        "a deterministic majority/fallback voter. Context nodes expand into their "
        "own internal helper graphs. FormatVerifier is not upgraded in this version."
    ),
    complexity_penalty=-0.02,
    is_expansion=True,
)
