"""TaskClassifierAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Identify the rough task type or required skill in a dataset-agnostic way. "
    "Do not output a final answer."
)


@dataclass
class TaskClassifierAgent(Agent):
    """UP-only generic task classification prompt wrapper."""

    agent_type: ClassVar[str] = "TaskClassifierAgent"
    role: ClassVar[str] = "task_classifier"
    description: ClassVar[str] = "Identifies the rough task type or required skill."
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "text"
