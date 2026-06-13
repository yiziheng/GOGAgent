"""Base Agent interface for the refactored GOG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.datasets.prompt_specs import (
    answer_instruction,
    format_problem,
    parse_answer_text,
)
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext, LLMTextResponse
from gogagent.prompt import (
    GENERIC_CONTEXT_INSTRUCTION,
    agent_system_prompt,
    context_instruction,
)


@dataclass
class Agent:
    """Strict LLM-backed Agent."""

    name: str | None = None

    agent_type: ClassVar[str] = "Agent"
    role: ClassVar[str] = "agent"
    description: ClassVar[str] = "Base graph agent."
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = ""
    prompt: ClassVar[str] = ""
    output_mode: ClassVar[str] = "text"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Execute this agent through the configured LLM client."""

        if context is None or context.llm_client is None:
            raise RuntimeError(
                f"{self.agent_type}.execute requires AgentContext with llm_client"
            )
        if self.output_mode in {"answer", "candidate_answer"}:
            return self._execute_answer(problem, inputs, context)
        return self._execute_text(problem, inputs, context)

    def build_prompt(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
    ) -> str:
        """Build the natural-language prompt sent to the LLM."""

        context_inputs = effective_context_inputs(problem, inputs)
        sections = [
            format_task_with_context(
                format_problem(problem),
                context_inputs,
                instruction=self.context_instruction_for(problem),
            ),
        ]
        if self.output_mode in {"answer", "candidate_answer"}:
            sections.append(answer_instruction(problem))
        return "\n\n".join(section for section in sections if section.strip())

    def system_prompt_for(self, problem: Mapping[str, Any]) -> str:
        """Return this agent's dataset-specific system prompt."""

        return agent_system_prompt(
            problem,
            self.prompt_key or self.role,
            role=self.role,
        )

    def context_instruction_for(self, problem: Mapping[str, Any]) -> str:
        """Return this agent's dataset-specific context instruction."""

        return context_instruction(problem)

    def _execute_text(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext,
    ) -> GraphMessage:
        prompt = self.build_prompt(problem, inputs)
        response = context.llm_client.chat_text(
            role=self.role,
            prompt=prompt,
            system_prompt=self.system_prompt_for(problem),
        )
        content = response.text.strip()
        if not content:
            raise RuntimeError(f"{self.agent_type} LLM response must not be empty")
        return self.make_message(
            content=content,
            metadata={
                "llm": llm_metadata(response),
                "llm_audit": [llm_audit_metadata(response)],
            },
        )

    def _execute_answer(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext,
    ) -> GraphMessage:
        prompt = self.build_prompt(problem, inputs)
        response = context.llm_client.chat_text(
            role=self.role,
            prompt=prompt,
            system_prompt=self.system_prompt_for(problem),
        )
        answer = parse_answer_text(response.text, problem)
        return self.make_message(
            content=answer,
            answer=answer,
            metadata={
                "raw_output": response.text,
                "llm": llm_metadata(response),
                "llm_audit": [llm_audit_metadata(response)],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable agent descriptor."""

        return {
            "type": self.agent_type,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "standalone": self.standalone,
            "prompt_key": self.prompt_key,
            "output_mode": self.output_mode,
        }

    @property
    def display_name(self) -> str:
        """Stable display name used in messages and artifacts."""

        return self.name or self.agent_type

    def make_message(
        self,
        *,
        content: str,
        answer: str | None = None,
        confidence: float | None = None,
        notes: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GraphMessage:
        """Build a role-tagged GraphMessage."""

        return GraphMessage(
            sender=self.display_name,
            role=self.role,
            content=content,
            answer=answer,
            confidence=confidence,
            notes=dict(notes or {}),
            metadata={
                "agent_type": self.agent_type,
                "standalone": self.standalone,
                **dict(metadata or {}),
            },
        )


def problem_statement(problem: Mapping[str, Any]) -> str:
    """Extract a generic problem statement without dataset-specific logic."""

    for key in ("question", "prompt", "problem", "task", "input"):
        value = problem.get(key)
        if value:
            return str(value)
    return str(problem)


def format_task_with_context(
    task: str,
    inputs: Mapping[str, GraphMessage],
    *,
    instruction: str = GENERIC_CONTEXT_INSTRUCTION,
) -> str:
    """Wrap a task with advisory predecessor context when context exists."""

    if not inputs:
        return task.strip()
    return "\n\n".join(
        section
        for section in (
            f"Task:\n{task.strip()}",
            format_upstream_context(inputs),
            f"Instruction:\n{instruction.strip()}",
        )
        if section.strip()
    )


def effective_context_inputs(
    problem: Mapping[str, Any],
    inputs: Mapping[str, GraphMessage],
) -> dict[str, GraphMessage]:
    """Merge subgraph parent inputs with direct predecessor inputs."""

    merged = parent_context_inputs(problem)
    merged.update(
        {
            str(node_id): GraphMessage.from_dict(message)
            for node_id, message in inputs.items()
        }
    )
    return merged


def parent_context_inputs(problem: Mapping[str, Any]) -> dict[str, GraphMessage]:
    """Return parent-node inputs injected when an agent runs inside a UP subgraph."""

    raw_inputs = problem.get("_parent_inputs")
    if not isinstance(raw_inputs, Mapping):
        return {}
    parent_inputs: dict[str, GraphMessage] = {}
    for node_id, message in raw_inputs.items():
        if not isinstance(message, Mapping) and not isinstance(message, GraphMessage):
            continue
        parent_inputs[f"parent:{node_id}"] = GraphMessage.from_dict(message)
    return parent_inputs


def format_upstream_context(inputs: Mapping[str, GraphMessage]) -> str:
    """Format predecessor messages as advisory context."""

    if not inputs:
        return ""
    lines = ["Available context:"]
    for index, (node_id, message) in enumerate(inputs.items(), start=1):
        lines.append(f"{index}. {node_id} ({message.role})")
        if message.answer is not None:
            lines.append(f"   answer: {message.answer}")
        if message.content.strip():
            lines.append(f"   content: {short_text(message.content, 600)}")
    return "\n".join(lines)


def latest_parseable_answer(
    problem: Mapping[str, Any],
    inputs: Mapping[str, GraphMessage],
) -> str:
    """Return the latest upstream answer after strict normalization."""

    answer = latest_answer(inputs)
    if answer is None:
        raise RuntimeError("no upstream answer available to normalize")
    return parse_answer_text(str(answer), problem)


def llm_metadata(response: LLMTextResponse) -> dict[str, Any]:
    """Return provider metadata for one raw-text LLM call."""

    return {
        "model": response.model,
        "usage": response.usage.to_dict(),
        "latency_seconds": response.latency_seconds,
    }


def llm_audit_metadata(
    response: LLMTextResponse,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    """Return full LLM request/response metadata for audit-only artifacts."""

    record: dict[str, Any] = {
        "call_type": "chat_text",
        "request": {"messages": list(response.request_messages)},
        "response": {"text": response.text},
        "model": response.model,
        "usage": response.usage.to_dict(),
        "latency_seconds": response.latency_seconds,
    }
    if phase is not None:
        record["phase"] = phase
    return record


def aggregate_llm_metadata(responses: list[tuple[str, LLMTextResponse]]) -> dict[str, Any]:
    """Return aggregate metadata for a multi-call agent."""

    usage = {
        "prompt_tokens": sum(response.usage.prompt_tokens for _, response in responses),
        "completion_tokens": sum(response.usage.completion_tokens for _, response in responses),
        "total_tokens": sum(response.usage.total_tokens for _, response in responses),
    }
    return {
        "model": responses[-1][1].model,
        "usage": usage,
        "latency_seconds": sum(response.latency_seconds for _, response in responses),
        "calls": [
            {"phase": phase, **llm_metadata(response)}
            for phase, response in responses
        ],
    }


def short_text(text: str, limit: int = 240) -> str:
    """Return a compact one-line text snippet."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def latest_answer(inputs: Mapping[str, GraphMessage]) -> str | None:
    """Return the last available upstream answer, if any."""

    for message in reversed(list(inputs.values())):
        if message.answer is not None:
            return str(message.answer)
    return None


def input_summary(inputs: Mapping[str, GraphMessage]) -> list[dict[str, Any]]:
    """Summarize predecessor messages for downstream prompt wrappers."""

    return [
        {
            "node_id": node_id,
            "role": message.role,
            "answer": message.answer,
            "content": short_text(message.content, 160),
        }
        for node_id, message in inputs.items()
    ]
