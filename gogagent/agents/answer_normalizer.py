"""AnswerNormalizerAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import Agent, latest_parseable_answer
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext


@dataclass
class AnswerNormalizerAgent(Agent):
    """UP-only answer normalization prompt wrapper."""

    agent_type: ClassVar[str] = "AnswerNormalizerAgent"
    role: ClassVar[str] = "answer_normalizer"
    description: ClassVar[str] = "Converts the answer into the expected final format."
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "answer_normalizer"
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
            metadata={"source": "deterministic_answer_normalization"},
        )
