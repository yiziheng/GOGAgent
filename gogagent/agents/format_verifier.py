"""FormatVerifierAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Check whether the final message exposes a parseable answer field and "
    "normalize the answer when possible. Return a GraphMessage JSON object."
)


@dataclass
class FormatVerifierAgent(Agent):
    """Final format verification prompt wrapper."""

    agent_type: ClassVar[str] = "FormatVerifierAgent"
    role: ClassVar[str] = "format_verifier"
    description: ClassVar[str] = "Checks output format and normalizes the final answer."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
