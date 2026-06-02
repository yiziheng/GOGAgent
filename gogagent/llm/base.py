"""Minimal model backend contract used by executable DAG nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMResponse:
    """One generated response with provider-reported usage metadata."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_seconds: float


class LLMBackend(ABC):
    """Generate model responses without exposing backend credentials."""

    name = "llm_backend"

    @abstractmethod
    def generate(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        """Generate one node output."""

    def describe(self) -> Mapping[str, Any]:
        """Return credential-free backend metadata suitable for artifacts."""

        return {"name": self.name}
