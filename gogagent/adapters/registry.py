"""Lazy adapter registry."""

from __future__ import annotations

from gogagent.adapters.base import DomainAdapter
from gogagent.adapters.gsm8k import GSM8KAdapter
from gogagent.adapters.humaneval import HumanEvalAdapter
from gogagent.adapters.mmlu import MMLUAdapter


def get_adapter(name: str) -> DomainAdapter:
    adapters: dict[str, type[DomainAdapter]] = {
        "gsm8k": GSM8KAdapter,
        "mmlu": MMLUAdapter,
        "humaneval": HumanEvalAdapter,
    }
    try:
        return adapters[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unsupported adapter {name!r}; choose from {sorted(adapters)}") from error
