"""JudgeAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


@dataclass
class JudgeAgent(Agent):
    """UP-only fair judgment prompt wrapper."""

    agent_type: ClassVar[str] = "JudgeAgent"
    role: ClassVar[str] = "judge"
    description: ClassVar[str] = (
        "Fairly decides whether the defended answer should replace the original answer."
    )
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "judge"
    output_mode: ClassVar[str] = "answer"
