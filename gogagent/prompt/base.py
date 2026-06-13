"""Prompt-set data structures for dataset-specific GOGAgent prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AgentPromptSet:
    """Dataset-specific prompt bundle for all graph agents."""

    dataset: str
    agent_system_prompts: Mapping[str, str]
    default_text_system_template: str
    json_system_template: str
    context_instruction: str

    def system_for(self, key: str, role: str, *, fallback: str = "") -> str:
        """Return a dataset-specific system prompt for an agent key."""

        prompt = self.agent_system_prompts.get(key)
        if prompt:
            return str(prompt)
        if fallback:
            return str(fallback)
        raise KeyError(
            f"missing system prompt for dataset={self.dataset!r}, key={key!r}, role={role!r}"
        )

    def text_system_for(self, role: str) -> str:
        """Return the default text-call system prompt for a role."""

        return self.default_text_system_template.format(role=role)

    def json_system_for(self, role: str) -> str:
        """Return the default JSON-call system prompt for a role."""

        return self.json_system_template.format(role=role)
