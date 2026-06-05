"""Graph mutation for ADD_FORMAT_VERIFIER."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import append_as_output_node, make_node, unique_node_id


def apply(graph: Any) -> Any:
    """Append FormatVerifierAgent after the current output node."""

    node = make_node(
        "FormatVerifierAgent",
        node_id_value=unique_node_id(graph, "format_verifier"),
    )
    return append_as_output_node(graph, node)
