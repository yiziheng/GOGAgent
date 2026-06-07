"""RL reward helpers for graph rollouts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from gogagent.reward import RewardBreakdown, compute_reward


def compute_rl_reward(
    *,
    dataset: str,
    public_task: Mapping[str, Any],
    gold: Any,
    final_output: Any,
    action_records: Iterable[Mapping[str, Any]],
) -> RewardBreakdown:
    """Compute the current v1 reward for one sampled rollout."""

    return compute_reward(
        dataset=dataset,
        example=public_task,
        final_output=final_output,
        action_records=action_records,
        gold=gold,
    )
