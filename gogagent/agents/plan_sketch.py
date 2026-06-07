"""PlanSketchAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Propose a bounded plan with at most two concise solving steps. Do not "
    "solve the problem and do not output a final answer."
)


@dataclass
class PlanSketchAgent(Agent):
    """Two-step planning prompt wrapper."""

    agent_type: ClassVar[str] = "PlanSketchAgent"
    role: ClassVar[str] = "plan_sketch"
    description: ClassVar[str] = "Produces at most two concise solving steps. No loop execution."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "text"
