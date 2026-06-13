"""Static metadata for ADD_ADVERSARIAL_JUDGE."""

from gogagent.actions.base import ActionName, ActionSpec


SPEC = ActionSpec(
    name=ActionName.ADD_ADVERSARIAL_JUDGE,
    description=(
        "Add an AdversarialJudgeAgent near the output side. On MMLU this runs "
        "one shuffled-option vote and falls back to the anchor solver on "
        "disagreement. Other datasets keep the generic independent-answer plus "
        "arbitration behavior."
    ),
    complexity_penalty=-0.01,
    is_expansion=True,
)
