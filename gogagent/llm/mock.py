"""Deterministic offline backend for compilation and smoke checks."""

from __future__ import annotations

from typing import Mapping

from gogagent.llm.base import LLMBackend


class MockLLM(LLMBackend):
    """Return deterministic text so the runtime can be tested without network I/O."""

    def generate(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None = None,
    ) -> str:
        context_size = len(context or {})
        return f"[{role}] mock response; context_items={context_size}; prompt={prompt[:80]}"
