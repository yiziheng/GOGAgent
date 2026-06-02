"""Incremental executor facade."""

from __future__ import annotations

from typing import Any, Mapping

from gogagent.adapters.base import DomainAdapter
from gogagent.core.types import ExecutionResult, OrgGraphSnapshot
from gogagent.llm.base import LLMBackend


class IncrementalExecutor:
    def __init__(self, adapter: DomainAdapter, llm: LLMBackend) -> None:
        self.adapter = adapter
        self.llm = llm

    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        return self.adapter.execute(graph, task, self.llm, previous)
