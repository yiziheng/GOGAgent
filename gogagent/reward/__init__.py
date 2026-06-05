"""Reward utilities for the Round 1 GOG refactor."""

from gogagent.reward.format import FormatResult, check_output_format
from gogagent.reward.oracle import OracleResult, score_answer
from gogagent.reward.reward import RewardBreakdown, compute_reward

__all__ = [
    "FormatResult",
    "OracleResult",
    "RewardBreakdown",
    "check_output_format",
    "compute_reward",
    "score_answer",
]
