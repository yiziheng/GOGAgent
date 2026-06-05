"""Simplified Round 1 reward computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from gogagent.reward.format import FormatResult, check_output_format
from gogagent.reward.oracle import OracleResult, score_answer


VALIDITY_ILLEGAL_REWARD = -0.2
VALIDITY_LEGAL_REWARD = 0.0
ADD_NODE_PENALTY = -0.01
UP_NODE_PENALTY = -0.02


@dataclass(frozen=True)
class RewardBreakdown:
    """Named reward components for one completed rollout."""

    answer_correctness: float
    format_correctness: float
    graph_validity: float
    graph_complexity: float
    total: float
    format_result: FormatResult
    oracle_result: OracleResult
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_correctness": self.answer_correctness,
            "format_correctness": self.format_correctness,
            "graph_validity": self.graph_validity,
            "graph_complexity": self.graph_complexity,
            "total": self.total,
            "format_result": self.format_result.to_dict(),
            "oracle_result": self.oracle_result.to_dict(),
            "details": self.details,
        }


def compute_reward(
    *,
    dataset: str,
    example: Mapping[str, Any],
    final_output: Any,
    action_records: Iterable[Any] = (),
    gold: Any | None = None,
) -> RewardBreakdown:
    """Compute the documented v1 reward.

    Components:
    - correct answer: +1.0, wrong: 0.0
    - valid GraphMessage with parseable answer: 0.0, invalid: -0.2
    - legal action: 0.0, illegal action: -0.2
    - ADD_* action: -0.01, UP action: -0.02
    - no token penalty
    """

    records = tuple(action_records)
    format_result = check_output_format(final_output)
    oracle_result = score_answer(dataset, example, final_output, gold=gold)
    graph_validity = validity_reward(records)
    graph_complexity = complexity_penalty(records)
    total = (
        oracle_result.reward
        + format_result.reward
        + graph_validity
        + graph_complexity
    )
    return RewardBreakdown(
        answer_correctness=oracle_result.reward,
        format_correctness=format_result.reward,
        graph_validity=graph_validity,
        graph_complexity=graph_complexity,
        total=total,
        format_result=format_result,
        oracle_result=oracle_result,
        details={
            "illegal_action_count": count_illegal_actions(records),
            "complexity_actions": [_action_name(record) for record in records],
        },
    )


def validity_reward(action_records: Iterable[Any]) -> float:
    """Return -0.2 for each selected illegal action."""

    return VALIDITY_ILLEGAL_REWARD * count_illegal_actions(action_records)


def count_illegal_actions(action_records: Iterable[Any]) -> int:
    """Count action records marked illegal by common field names."""

    count = 0
    for record in action_records:
        if _record_is_illegal(record):
            count += 1
    return count


def complexity_penalty(action_records: Iterable[Any]) -> float:
    """Apply the documented flat complexity penalties."""

    penalty = 0.0
    for record in action_records:
        action = _action_name(record)
        if action == "UP":
            penalty += UP_NODE_PENALTY
        elif action.startswith("ADD_"):
            penalty += ADD_NODE_PENALTY
    return penalty


def _record_is_illegal(record: Any) -> bool:
    for key in ("legal", "is_legal", "selected_action_legal", "valid"):
        value = _field(record, key)
        if value is not None:
            return not bool(value)
    for key in ("illegal", "invalid", "masked"):
        value = _field(record, key)
        if value is not None:
            return bool(value)
    return False


def _action_name(record: Any) -> str:
    action = _field(record, "action")
    if action is None:
        action = _field(record, "action_name")
    if action is None:
        action = _field(record, "name")
    if action is None:
        action = record
    value = getattr(action, "value", action)
    return str(value).strip().upper()


def _field(record: Any, key: str) -> Any | None:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)
