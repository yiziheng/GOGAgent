"""UP template for TaskBriefAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_linear_subgraph


INTERNAL_AGENT_NAMES = ("TaskClassifierAgent", "TaskBriefAgent")


def build_subgraph(target_node: Any) -> Any:
    """Expand TaskBriefAgent into TaskClassifier -> TaskBrief."""

    return make_linear_subgraph(
        target_node,
        INTERNAL_AGENT_NAMES,
        graph_type="task_classify_brief",
    )
