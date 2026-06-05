"""AnswerNormalizerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gogagent.agents.base import Agent


AGENT_PROMPT = (
    "Convert the best available answer into the expected final answer string. "
    "Return a GraphMessage JSON object."
)


@dataclass
class AnswerNormalizerAgent(Agent):
    """UP-only answer normalization prompt wrapper."""

    agent_type: ClassVar[str] = "AnswerNormalizerAgent"
    role: ClassVar[str] = "answer_normalizer"
    description: ClassVar[str] = "Converts the answer into the expected final format."
    standalone: ClassVar[bool] = False
    prompt: ClassVar[str] = AGENT_PROMPT
