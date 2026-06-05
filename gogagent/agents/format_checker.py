"""FormatCheckerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Check whether the upstream output follows the benchmark-required format "
    "and has a parseable answer field. Return a GraphMessage JSON object."
)


@dataclass
class FormatCheckerAgent(Agent):
    """UP-only format checking prompt wrapper."""

    agent_type: ClassVar[str] = "FormatCheckerAgent"
    role: ClassVar[str] = "format_checker"
    description: ClassVar[str] = "Checks whether the output follows the benchmark-required format."
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
