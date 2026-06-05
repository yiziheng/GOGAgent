"""UP template for AdversarialJudgeAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_linear_subgraph


INTERNAL_AGENT_NAMES = ("ChallengerAgent", "DefenderAgent", "JudgeAgent")


def build_subgraph(target_node: Any) -> Any:
    """Expand AdversarialJudgeAgent into Challenger -> Defender -> Judge."""

    return make_linear_subgraph(
        target_node,
        INTERNAL_AGENT_NAMES,
        graph_type="adversarial_challenge_defense_judge",
    )
