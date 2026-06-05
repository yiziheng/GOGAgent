"""Graph mutation for ADD_ADVERSARIAL_JUDGE."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import append_as_output_node, make_node, unique_node_id


def apply(graph: Any) -> Any:
    """Append AdversarialJudgeAgent after the current output node."""

    node = make_node(
        "AdversarialJudgeAgent",
        node_id_value=unique_node_id(graph, "adversarial_judge"),
    )
    return append_as_output_node(graph, node)
