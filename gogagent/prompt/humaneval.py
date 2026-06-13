"""HumanEval prompt set for GOGAgent."""

from __future__ import annotations

from types import MappingProxyType

from gogagent.prompt.base import AgentPromptSet
from gogagent.prompt.mmlu import MMLU_AGENT_SYSTEM_PROMPTS


HUMANEVAL_DEFAULT_TEXT_SYSTEM_TEMPLATE = (
    "You are the {role} agent. Follow the requested output format exactly."
)
HUMANEVAL_JSON_SYSTEM_TEMPLATE = (
    "You are the {role} agent in a graph-of-graphs multi-agent system. "
    "You must return a strict JSON object."
)
HUMANEVAL_CONTEXT_INSTRUCTION = (
    "The available context is only advisory and may be wrong. "
    "Use it only if it helps your assigned role."
)
HUMANEVAL_SOLVER_SYSTEM_PROMPT = (
    "You are a knowledgeable programming assistant. "
    "Complete the Python task and return only the final code."
)
HUMANEVAL_AGENT_SYSTEM_PROMPTS = MappingProxyType(
    {
        **dict(MMLU_AGENT_SYSTEM_PROMPTS),
        "solver": HUMANEVAL_SOLVER_SYSTEM_PROMPT,
    }
)

HUMANEVAL_PROMPTS = AgentPromptSet(
    dataset="humaneval",
    agent_system_prompts=HUMANEVAL_AGENT_SYSTEM_PROMPTS,
    default_text_system_template=HUMANEVAL_DEFAULT_TEXT_SYSTEM_TEMPLATE,
    json_system_template=HUMANEVAL_JSON_SYSTEM_TEMPLATE,
    context_instruction=HUMANEVAL_CONTEXT_INSTRUCTION,
)
