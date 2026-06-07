"""ChallengerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Challenge the upstream answer. Look for incorrect assumptions, missing "
    "evidence, tempting distractors, or format risks. Do not output a final "
    "answer; write a concise critique only."
)


@dataclass
class ChallengerAgent(Agent):
    """UP-only adversarial challenge prompt wrapper."""

    agent_type: ClassVar[str] = "ChallengerAgent"
    role: ClassVar[str] = "challenger"
    description: ClassVar[str] = "Challenges an answer and searches for weaknesses."
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "text"
