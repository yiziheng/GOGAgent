"""AdversarialJudgeAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "First challenge the current answer, then fairly judge whether revision is "
    "justified. Return the best final answer as a GraphMessage JSON object."
)


@dataclass
class AdversarialJudgeAgent(Agent):
    """Challenge-then-judge prompt wrapper."""

    agent_type: ClassVar[str] = "AdversarialJudgeAgent"
    role: ClassVar[str] = "adversarial_judge"
    description: ClassVar[str] = (
        "Uses two LLM calls: first challenge the answer, then fairly judge whether to revise."
    )
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
