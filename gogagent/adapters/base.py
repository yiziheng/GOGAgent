"""Domain adapter interface for the shared Organization GoG runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from gogagent.core.actions import MacroAction
from gogagent.core.types import (
    CompiledEdit,
    ExecutionResult,
    GraphSignature,
    OrgGraphSnapshot,
    VisibleFeedback,
)
from gogagent.llm.base import LLMBackend


class DomainAdapter(ABC):
    """Compile shared actions into domain-specific executable DAG edits."""

    name: str

    @abstractmethod
    def base_graph(self, task: Mapping[str, Any]) -> OrgGraphSnapshot:
        """Return the minimum executable graph for a task."""

    @abstractmethod
    def task_features(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return label-blind features available during inference."""

    @abstractmethod
    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        """Compile a macro action into deterministic nodes and typed edges."""

    @abstractmethod
    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        llm: LLMBackend,
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        """Execute or simulate a candidate graph without consulting gold labels."""

    @abstractmethod
    def signature(self, graph: OrgGraphSnapshot) -> GraphSignature:
        """Build a lightweight structure-only graph signature."""
