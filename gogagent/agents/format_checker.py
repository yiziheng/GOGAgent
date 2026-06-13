"""FormatCheckerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import Agent, latest_parseable_answer
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext


@dataclass
class FormatCheckerAgent(Agent):
    """UP-only format checking prompt wrapper."""

    agent_type: ClassVar[str] = "FormatCheckerAgent"
    role: ClassVar[str] = "format_checker"
    description: ClassVar[str] = "Checks whether the output follows the benchmark-required format."
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "format_checker"
    output_mode: ClassVar[str] = "text"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        del context
        answer = latest_parseable_answer(problem, inputs)
        return self.make_message(
            content=f"format_valid: {answer}",
            answer=answer,
            metadata={"source": "deterministic_format_check"},
        )
