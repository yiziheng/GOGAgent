"""GSM8K prompt set for GOGAgent."""

from __future__ import annotations

from types import MappingProxyType

from gogagent.prompt.base import AgentPromptSet
from gogagent.prompt.mmlu import MMLU_AGENT_SYSTEM_PROMPTS


GSM8K_DEFAULT_TEXT_SYSTEM_TEMPLATE = (
    "You are the {role} agent. Follow the requested output format exactly."
)
GSM8K_JSON_SYSTEM_TEMPLATE = (
    "You are the {role} agent in a graph-of-graphs multi-agent system. "
    "You must return a strict JSON object."
)
GSM8K_CONTEXT_INSTRUCTION = (
    "The available context is only advisory and may be wrong. "
    "Use it only if it helps your assigned role."
)
GSM8K_SOLVER_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Solve the grade-school math problem and return only the final numeric answer."
)
GSM8K_AGENT_SYSTEM_PROMPTS = MappingProxyType(
    {
        **dict(MMLU_AGENT_SYSTEM_PROMPTS),
        "solver": GSM8K_SOLVER_SYSTEM_PROMPT,
    }
)

GSM8K_PROMPTS = AgentPromptSet(
    dataset="gsm8k",
    agent_system_prompts=GSM8K_AGENT_SYSTEM_PROMPTS,
    default_text_system_template=GSM8K_DEFAULT_TEXT_SYSTEM_TEMPLATE,
    json_system_template=GSM8K_JSON_SYSTEM_TEMPLATE,
    context_instruction=GSM8K_CONTEXT_INSTRUCTION,
)
