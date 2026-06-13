"""Dataset-specific prompt registry for GOGAgent."""

from __future__ import annotations

from typing import Any, Mapping

from gogagent.prompt.base import AgentPromptSet
from gogagent.prompt.generic import (
    GENERIC_AGENT_SYSTEM_PROMPTS,
    GENERIC_CONTEXT_INSTRUCTION,
    GENERIC_DEFAULT_TEXT_SYSTEM_TEMPLATE,
    GENERIC_JSON_SYSTEM_TEMPLATE,
    GENERIC_PROMPTS,
)
from gogagent.prompt.gsm8k import GSM8K_AGENT_SYSTEM_PROMPTS, GSM8K_PROMPTS
from gogagent.prompt.humaneval import HUMANEVAL_AGENT_SYSTEM_PROMPTS, HUMANEVAL_PROMPTS
from gogagent.prompt.mmlu import (
    MMLU_AGENT_SYSTEM_PROMPTS,
    MMLU_PROMPTS,
    MMLU_SOLVER_SYSTEM_PROMPT,
)
from gogagent.prompt.multiagentbench import (
    MULTIAGENTBENCH_AGENT_SYSTEM_PROMPTS,
    MULTIAGENTBENCH_PROMPTS,
    MULTIAGENTBENCH_SOLVER_SYSTEM_PROMPT,
)


PROMPT_SETS: dict[str, AgentPromptSet] = {
    "mmlu": MMLU_PROMPTS,
    "gsm8k": GSM8K_PROMPTS,
    "humaneval": HUMANEVAL_PROMPTS,
    "multiagentbench": MULTIAGENTBENCH_PROMPTS,
}


def get_prompt_set(problem_or_dataset: Mapping[str, Any] | str | None) -> AgentPromptSet:
    """Return the prompt set for a dataset or problem mapping."""

    if isinstance(problem_or_dataset, Mapping):
        dataset = str(problem_or_dataset.get("dataset", "")).strip().lower()
    else:
        dataset = str(problem_or_dataset or "").strip().lower()
    return PROMPT_SETS.get(dataset, GENERIC_PROMPTS)


def agent_system_prompt(
    problem_or_dataset: Mapping[str, Any] | str | None,
    key: str,
    *,
    role: str | None = None,
    fallback: str = "",
) -> str:
    """Return a dataset-specific system prompt for an agent key."""

    return get_prompt_set(problem_or_dataset).system_for(
        key,
        role or key,
        fallback=fallback,
    )


def context_instruction(problem_or_dataset: Mapping[str, Any] | str | None) -> str:
    """Return the dataset-specific context instruction."""

    return get_prompt_set(problem_or_dataset).context_instruction


def default_text_system_prompt(role: str) -> str:
    """Return the default system prompt for non-special text-call agents."""

    return GENERIC_PROMPTS.system_for(role, role)


def json_system_prompt(role: str) -> str:
    """Return the default system prompt for JSON-call agents."""

    return GENERIC_PROMPTS.json_system_for(role)


__all__ = [
    "AgentPromptSet",
    "GENERIC_AGENT_SYSTEM_PROMPTS",
    "GENERIC_CONTEXT_INSTRUCTION",
    "GENERIC_DEFAULT_TEXT_SYSTEM_TEMPLATE",
    "GENERIC_JSON_SYSTEM_TEMPLATE",
    "GENERIC_PROMPTS",
    "GSM8K_AGENT_SYSTEM_PROMPTS",
    "GSM8K_PROMPTS",
    "HUMANEVAL_AGENT_SYSTEM_PROMPTS",
    "HUMANEVAL_PROMPTS",
    "MMLU_AGENT_SYSTEM_PROMPTS",
    "MMLU_PROMPTS",
    "MMLU_SOLVER_SYSTEM_PROMPT",
    "MULTIAGENTBENCH_AGENT_SYSTEM_PROMPTS",
    "MULTIAGENTBENCH_PROMPTS",
    "MULTIAGENTBENCH_SOLVER_SYSTEM_PROMPT",
    "PROMPT_SETS",
    "agent_system_prompt",
    "context_instruction",
    "default_text_system_prompt",
    "get_prompt_set",
    "json_system_prompt",
]
