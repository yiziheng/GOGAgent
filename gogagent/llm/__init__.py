"""LLM backend abstractions."""

from gogagent.llm.base import LLMBackend
from gogagent.llm.mock import MockLLM

__all__ = ["LLMBackend", "MockLLM"]
