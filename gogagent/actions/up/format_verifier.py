"""UP template for FormatVerifierAgent nodes."""

from __future__ import annotations

from typing import Any

from gogagent.actions.base import make_linear_subgraph


INTERNAL_AGENT_NAMES = ("FormatCheckerAgent", "AnswerNormalizerAgent")


def build_subgraph(target_node: Any) -> Any:
    """Expand FormatVerifierAgent into FormatChecker -> AnswerNormalizer."""

    return make_linear_subgraph(
        target_node,
        INTERNAL_AGENT_NAMES,
        graph_type="format_check_normalize",
    )
