"""JudgeAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Compare the challenge and defense fairly, then select the better final "
    "answer. Output only the final answer."
)


@dataclass
class JudgeAgent(Agent):
    """UP-only fair judgment prompt wrapper."""

    agent_type: ClassVar[str] = "JudgeAgent"
    role: ClassVar[str] = "judge"
    description: ClassVar[str] = (
        "Fairly decides whether the defended answer should replace the original answer."
    )
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "answer"
