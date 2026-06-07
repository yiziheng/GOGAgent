"""SolverAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Solve the task directly. If upstream context is provided, use it only as "
    "helpful context. Your output must be the final answer only."
)


@dataclass
class SolverAgent(Agent):
    """Initial task solver prompt wrapper."""

    agent_type: ClassVar[str] = "SolverAgent"
    role: ClassVar[str] = "solver"
    description: ClassVar[str] = "Solves the task and produces an initial answer."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "answer"
