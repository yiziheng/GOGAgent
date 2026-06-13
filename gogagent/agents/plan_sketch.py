"""PlanSketchAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import Agent
from gogagent.graph.schema import GraphMessage
from gogagent.llm import AgentContext


_REPEAT_COUNT_RE = re.compile(
    r"\brepeat\s*(?:count|times|rounds?)?\s*[:：]\s*([12])\b",
    re.IGNORECASE,
)


@dataclass
class PlanSketchAgent(Agent):
    """Two-step planning prompt wrapper."""

    agent_type: ClassVar[str] = "PlanSketchAgent"
    role: ClassVar[str] = "plan_sketch"
    description: ClassVar[str] = "Produces at most two concise solving steps. No loop execution."
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = "plan_sketch"
    output_mode: ClassVar[str] = "text"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Execute the plan and annotate the requested bounded repeat count."""

        output = super().execute(problem, inputs, context=context)
        repeat_count = extract_repeat_count(output.content)
        output.metadata["repeat_count"] = repeat_count
        return output


def extract_repeat_count(text: str) -> int:
    """Extract a bounded repeat count from free-form plan text."""

    match = _REPEAT_COUNT_RE.search(text)
    if not match:
        return 1
    return int(match.group(1))
