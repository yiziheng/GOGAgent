"""LLM backend abstractions."""

from gogagent.llm.base import LLMBackend, LLMResponse
from gogagent.llm.openai_compatible import OpenAICompatibleLLM

__all__ = ["LLMBackend", "LLMResponse", "OpenAICompatibleLLM"]
