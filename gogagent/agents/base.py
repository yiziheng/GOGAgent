"""Base Agent interface for the refactored GOG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext


@dataclass
class Agent:
    """Strict LLM-backed Agent."""

    name: str | None = None

    agent_type: ClassVar[str] = "Agent"
    role: ClassVar[str] = "agent"
    description: ClassVar[str] = "Base graph agent."
    standalone: ClassVar[bool] = True
    prompt: ClassVar[str] = "Return a structured GraphMessage JSON object."

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
        response = context.llm_client.chat_json(
            role=self.role,
            prompt=self.prompt,
            payload={
                "problem": dict(problem),
                "inputs": {
                    node_id: message.to_dict()
                    for node_id, message in inputs.items()
                },
                "agent": self.to_dict(),
            },
        )
        data = dict(response.data)
        if "role" not in data or not str(data.get("role", "")).strip():
            data["role"] = self.role
        if "content" not in data or not isinstance(data.get("content"), str):
            raise RuntimeError(
                f"{self.agent_type} LLM response must include string content"
            )
        message = GraphMessage.from_dict(data)
        message.sender = self.display_name
        message.metadata.update(
            {
                "agent_type": self.agent_type,
                "standalone": self.standalone,
                "llm": {
                    "model": response.model,
                    "usage": response.usage.to_dict(),
                    "latency_seconds": response.latency_seconds,
                },
            }
        )
        return message

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable agent descriptor."""

        return {
            "type": self.agent_type,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "standalone": self.standalone,
            "prompt": self.prompt,
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
