"""Static metadata for ADD_ADVERSARIAL_JUDGE."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.ADD_ADVERSARIAL_JUDGE,
    description=(
        "Add an AdversarialJudgeAgent near the output side. The standalone "
        "agent uses two LLM calls: challenge the answer, then fairly judge "
        "whether revision is needed."
    ),
    complexity_penalty=-0.01,
    is_expansion=True,
)
