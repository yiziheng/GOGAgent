"""TaskBriefAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


@dataclass
class TaskBriefAgent(Agent):
    """One-sentence task briefing prompt wrapper."""

    agent_type: ClassVar[str] = "TaskBriefAgent"
    role: ClassVar[str] = "task_brief"
    description: ClassVar[str] = "Summarizes what the problem is asking in one sentence."
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = "task_brief"
    output_mode: ClassVar[str] = "text"
