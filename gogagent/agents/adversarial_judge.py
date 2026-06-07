"""AdversarialJudgeAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import (
    Agent,
    aggregate_llm_metadata,
    answer_instruction,
    format_problem,
    format_upstream_context,
    parse_answer_text,
)
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext


AGENT_PROMPT = (
    "First challenge the current answer, then fairly judge whether revision is "
    "justified. The final output must be the final answer only."
)


@dataclass
class AdversarialJudgeAgent(Agent):
    """Challenge-then-judge prompt wrapper."""

    agent_type: ClassVar[str] = "AdversarialJudgeAgent"
    role: ClassVar[str] = "adversarial_judge"
    description: ClassVar[str] = (
        "Uses two LLM calls: first challenge the answer, then fairly judge whether to revise."
    )
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = AGENT_PROMPT
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Use two calls: critique first, then answer-only judgment."""

        if context is None or context.llm_client is None:
            raise RuntimeError(
                f"{self.agent_type}.execute requires AgentContext with llm_client"
            )
        base_prompt = "\n\n".join(
            section
            for section in (
                format_problem(problem),
                format_upstream_context(inputs),
            )
            if section.strip()
        )
        challenge_response = context.llm_client.chat_text(
            role="challenger",
            prompt=(
                "Challenge the current answer. Look for incorrect assumptions, "
                "tempting distractors, or missing evidence. Do not output a "
                f"final answer.\n\n{base_prompt}"
            ),
        )
        challenge = challenge_response.text.strip()
        if not challenge:
            raise RuntimeError("AdversarialJudgeAgent challenge response must not be empty")

        judge_response = context.llm_client.chat_text(
            role=self.role,
            prompt=(
                "Fairly judge the original answer using the challenge below. "
                "Revise only if the challenge is stronger than the original "
                "answer.\n\n"
                f"{base_prompt}\n\nChallenge:\n{challenge}\n\n"
                f"{answer_instruction(problem)}"
            ),
        )
        answer = parse_answer_text(judge_response.text, problem)
        responses = [
            ("challenge", challenge_response),
            ("judge", judge_response),
        ]
        aggregate = aggregate_llm_metadata(responses)
        return self.make_message(
            content=answer,
            answer=answer,
            notes={"challenge": challenge},
            metadata={
                "raw_output": judge_response.text,
                "llm": aggregate,
                "llm_calls": aggregate["calls"],
            },
        )
