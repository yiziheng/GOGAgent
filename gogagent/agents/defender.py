"""DefenderAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Respond to the challenge. Preserve the original answer if it is still "
    "best, or revise it if the challenge is valid. Return a GraphMessage JSON object."
)


@dataclass
class DefenderAgent(Agent):
    """UP-only answer defense prompt wrapper."""

    agent_type: ClassVar[str] = "DefenderAgent"
    role: ClassVar[str] = "defender"
    description: ClassVar[str] = (
        "Defends or revises the original answer in response to the challenge."
    )
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
