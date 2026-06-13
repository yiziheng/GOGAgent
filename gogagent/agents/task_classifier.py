"""TaskClassifierAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


@dataclass
class TaskClassifierAgent(Agent):
    """UP-only generic task classification prompt wrapper."""

    agent_type: ClassVar[str] = "TaskClassifierAgent"
    role: ClassVar[str] = "task_classifier"
    description: ClassVar[str] = "Identifies the rough task type or required skill."
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "task_classifier"
    output_mode: ClassVar[str] = "text"
