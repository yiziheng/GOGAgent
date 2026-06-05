"""ADD_ADVERSARIAL_JUDGE action package."""

from gogagent.actions.add_adversarial_judge.apply import apply
from gogagent.actions.add_adversarial_judge.legality import is_legal
from gogagent.actions.add_adversarial_judge.spec import SPEC

__all__ = ["SPEC", "apply", "is_legal"]
