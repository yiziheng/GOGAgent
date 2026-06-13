"""SupervisorAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import (
    Agent,
    answer_instruction,
    effective_context_inputs,
    format_upstream_context,
    latest_answer,
    llm_audit_metadata,
    llm_metadata,
    parse_answer_text,
)
from gogagent.datasets.prompt_specs import format_problem
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext
from gogagent.prompt import agent_system_prompt


@dataclass
class SupervisorAgent(Agent):
    """Conservative final supervisor for solver outputs."""

    agent_type: ClassVar[str] = "SupervisorAgent"
    role: ClassVar[str] = "supervisor"
    description: ClassVar[str] = (
        "Conservatively reviews the solver answer and changes it only for clear errors."
    )
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = "supervisor"
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Review the upstream answer with a conservative one-call supervisor."""

        if context is None or context.llm_client is None:
            raise RuntimeError(
                f"{self.agent_type}.execute requires AgentContext with llm_client"
            )
        normalized_inputs = effective_context_inputs(problem, inputs)
        prompt = self.build_supervisor_prompt(problem, normalized_inputs)
        response = context.llm_client.chat_text(
            role=self.role,
            prompt=prompt,
            system_prompt=agent_system_prompt(problem, self.prompt_key, role=self.role),
        )
        answer = parse_answer_text(response.text, problem)
        upstream_answer = latest_answer(normalized_inputs)
        return self.make_message(
            content=answer,
            answer=answer,
            notes={
                "upstream_answer": upstream_answer,
                "changed_answer": upstream_answer is not None and answer != upstream_answer,
            },
            metadata={
                "raw_output": response.text,
                "llm": llm_metadata(response),
                "llm_audit": [llm_audit_metadata(response, phase="supervise")],
            },
        )

    def build_supervisor_prompt(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
    ) -> str:
        """Build a compact supervisor prompt that keeps context advisory."""

        sections = [
            "Task:",
            format_problem(problem),
            "",
            format_upstream_context(inputs),
            "",
            "Supervisor instruction:",
            "Treat the upstream solver as a strong baseline.",
            "Use the upstream answer as advice, not as ground truth.",
            "Check for clear task mismatch, format error, arithmetic/logical error, or factual/domain-rule error.",
            "Keep the upstream answer unless another output is clearly better.",
            answer_instruction(problem),
        ]
        return "\n".join(section for section in sections if section.strip())
