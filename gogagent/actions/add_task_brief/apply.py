"""Graph mutation for ADD_TASK_BRIEF."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_node, prepend_as_input_node, unique_node_id


def apply(graph: Any) -> Any:
    """Insert TaskBriefAgent before the current input node."""

    node = make_node(
        "TaskBriefAgent",
        node_id_value=unique_node_id(graph, "task_brief"),
    )
    return prepend_as_input_node(graph, node)
