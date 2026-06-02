"""Policy-visible feature compression."""

from __future__ import annotations

from typing import Any, Mapping

from gogagent.core.types import GraphSignature, SupervisorFeedback, VisibleFeedback


def compressed_state(
    task_features: Mapping[str, Any],
    signature: GraphSignature,
    visible: VisibleFeedback,
    supervisor: SupervisorFeedback,
    used_tokens: int,
    token_budget: int,
) -> dict[str, Any]:
    return {
        "task_features": dict(task_features),
        "graph_signature": signature.to_dict(),
        "observable_feedback": visible.to_dict(),
        "supervisor_feedback": supervisor.to_dict(),
        "used_tokens": used_tokens,
        "remaining_tokens": max(token_budget - used_tokens, 0),
    }
