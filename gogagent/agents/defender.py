"""DefenderAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


@dataclass
class DefenderAgent(Agent):
    """UP-only answer defense prompt wrapper."""

    agent_type: ClassVar[str] = "DefenderAgent"
    role: ClassVar[str] = "defender"
    description: ClassVar[str] = (
        "Defends or revises the original answer in response to the challenge."
    )
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "defender"
    output_mode: ClassVar[str] = "candidate_answer"
