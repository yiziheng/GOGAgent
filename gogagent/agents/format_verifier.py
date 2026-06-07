"""FormatVerifierAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import Agent, latest_parseable_answer
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext


AGENT_PROMPT = (
    "Check whether the final message exposes a parseable answer field and "
    "normalize the answer when possible."
)


@dataclass
class FormatVerifierAgent(Agent):
    """Final format verification prompt wrapper."""

    agent_type: ClassVar[str] = "FormatVerifierAgent"
    role: ClassVar[str] = "format_verifier"
    description: ClassVar[str] = "Checks output format and normalizes the final answer."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        del context
        answer = latest_parseable_answer(problem, inputs)
        return self.make_message(
            content=answer,
            answer=answer,
            metadata={"source": "deterministic_format_verification"},
        )
