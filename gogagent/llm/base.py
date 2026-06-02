"""Minimal model backend contract used by executable DAG nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None = None,
    ) -> str:
        """Generate one node output."""
