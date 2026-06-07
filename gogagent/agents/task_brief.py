"""TaskBriefAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Create a compact one-sentence task brief for downstream agents. Do not "
    "output a final answer."
)


@dataclass
class TaskBriefAgent(Agent):
    """One-sentence task briefing prompt wrapper."""

    agent_type: ClassVar[str] = "TaskBriefAgent"
    role: ClassVar[str] = "task_brief"
    description: ClassVar[str] = "Summarizes what the problem is asking in one sentence."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "text"
