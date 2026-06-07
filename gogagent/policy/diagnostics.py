"""Policy diagnostics for logging action preferences."""

from __future__ import annotations

import math
from typing import Any

import torch

from gogagent.policy.action_space import ACTION_SPACE


def top_action_scores(masked_logits: torch.Tensor | Any, *, k: int = 6) -> list[dict[str, Any]]:
    """Return the top-k action scores from a masked logits vector."""

    if not isinstance(masked_logits, torch.Tensor):
        masked_logits = torch.as_tensor(masked_logits, dtype=torch.float32)
    values = masked_logits.detach().cpu().flatten().tolist()
    ranked = sorted(
        enumerate(values),
        key=lambda item: item[1],
        reverse=True,
    )[:k]
    return [
        {
            "action": ACTION_SPACE[index].value,
            "score": float(score) if math.isfinite(float(score)) else None,
        }
        for index, score in ranked
    ]
