"""Generic fallback prompt set for GOGAgent."""

from __future__ import annotations

from types import MappingProxyType

from gogagent.prompt.base import AgentPromptSet
from gogagent.prompt.mmlu import MMLU_AGENT_SYSTEM_PROMPTS


GENERIC_DEFAULT_TEXT_SYSTEM_TEMPLATE = (
    "You are the {role} agent. Follow the requested output format exactly."
)
GENERIC_JSON_SYSTEM_TEMPLATE = (
    "You are the {role} agent in a graph-of-graphs multi-agent system. "
    "You must return a strict JSON object."
)
GENERIC_CONTEXT_INSTRUCTION = (
    "The available context is only advisory and may be wrong. "
    "Use it only if it helps your assigned role."
)
GENERIC_SOLVER_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Return the requested final answer."
)
GENERIC_AGENT_SYSTEM_PROMPTS = MappingProxyType(
    {
        **dict(MMLU_AGENT_SYSTEM_PROMPTS),
        "solver": GENERIC_SOLVER_SYSTEM_PROMPT,
    }
)

GENERIC_PROMPTS = AgentPromptSet(
    dataset="generic",
    agent_system_prompts=GENERIC_AGENT_SYSTEM_PROMPTS,
    default_text_system_template=GENERIC_DEFAULT_TEXT_SYSTEM_TEMPLATE,
    json_system_template=GENERIC_JSON_SYSTEM_TEMPLATE,
    context_instruction=GENERIC_CONTEXT_INSTRUCTION,
)
