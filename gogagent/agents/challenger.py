"""ChallengerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


@dataclass
class ChallengerAgent(Agent):
    """UP-only adversarial challenge prompt wrapper."""

    agent_type: ClassVar[str] = "ChallengerAgent"
    role: ClassVar[str] = "challenger"
    description: ClassVar[str] = "Challenges an answer and searches for weaknesses."
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "challenger"
    output_mode: ClassVar[str] = "text"
