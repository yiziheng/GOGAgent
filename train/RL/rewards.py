"""RL reward helpers for graph rollouts."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping

from gogagent.reward import RewardBreakdown, check_output_format, compute_reward, score_answer


RewardMode = Literal["dense", "answer_only"]
REWARD_MODES: tuple[RewardMode, ...] = ("dense", "answer_only")


def compute_rl_reward(
    *,
    dataset: str,
    public_task: Mapping[str, Any],
    gold: Any,
    final_output: Any,
    action_records: Iterable[Mapping[str, Any]],
    reward_mode: RewardMode = "dense",
) -> RewardBreakdown:
    """Compute the selected RL reward for one sampled rollout."""

    if reward_mode == "answer_only":
        return compute_answer_only_reward(
            dataset=dataset,
            public_task=public_task,
            gold=gold,
            final_output=final_output,
        )
    if reward_mode != "dense":
        raise ValueError(
            f"unknown reward_mode {reward_mode!r}; expected one of {REWARD_MODES!r}"
        )
    return compute_reward(
        dataset=dataset,
        example=public_task,
        final_output=final_output,
        action_records=action_records,
        gold=gold,
    )


def compute_answer_only_reward(
    *,
    dataset: str,
    public_task: Mapping[str, Any],
    gold: Any,
    final_output: Any,
) -> RewardBreakdown:
    """Return a sparse reward based only on final-answer correctness."""

    format_result = check_output_format(final_output)
    oracle_result = score_answer(dataset, public_task, final_output, gold=gold)
    return RewardBreakdown(
        answer_correctness=oracle_result.reward,
        format_correctness=0.0,
        graph_validity=0.0,
        graph_complexity=0.0,
        total=oracle_result.reward,
        format_result=format_result,
        oracle_result=oracle_result,
        details={"reward_mode": "answer_only"},
    )


__all__ = [
    "REWARD_MODES",
    "RewardMode",
    "compute_answer_only_reward",
    "compute_rl_reward",
]
