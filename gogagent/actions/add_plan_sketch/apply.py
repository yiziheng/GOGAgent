"""Graph mutation for ADD_PLAN_SKETCH."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_node, prepend_as_input_node, unique_node_id


def apply(graph: Any) -> Any:
    """Insert PlanSketchAgent near the input side."""

    node = make_node(
        "PlanSketchAgent",
        node_id_value=unique_node_id(graph, "plan_sketch"),
    )
    return prepend_as_input_node(graph, node)
