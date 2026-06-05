"""Format reward helpers for structured GraphMessage outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from typing import Any, Mapping


FORMAT_INVALID_REWARD = -0.2
FORMAT_VALID_REWARD = 0.0


@dataclass(frozen=True)
class FormatResult:
    """Result of checking whether an output is a parseable GraphMessage."""

    valid: bool
    reason: str
    message: dict[str, Any] | None = None
    answer: Any | None = None

    @property
    def reward(self) -> float:
        """Return the Round 1 format reward."""

        return FORMAT_VALID_REWARD if self.valid else FORMAT_INVALID_REWARD

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "message": self.message,
            "answer": self.answer,
            "reward": self.reward,
        }


def check_output_format(output: Any) -> FormatResult:
    """Check that ``output`` is JSON/GraphMessage-like and has a parseable answer.

    Parseability is deliberately shallow here: the answer only needs to be present
    and non-empty. Dataset-specific correctness is handled by ``oracle.py``.
    """

    message = coerce_graph_message(output)
    if message is None:
        return FormatResult(
            valid=False,
            reason="output is not a JSON object or GraphMessage",
        )
    if "answer" not in message:
        return FormatResult(
            valid=False,
            reason="missing answer field",
            message=message,
        )
    answer = message.get("answer")
    if not is_parseable_answer(answer):
        return FormatResult(
            valid=False,
            reason="answer field is empty or unparseable",
            message=message,
            answer=answer,
        )
    return FormatResult(
        valid=True,
        reason="valid GraphMessage",
        message=message,
        answer=answer,
    )


def coerce_graph_message(output: Any) -> dict[str, Any] | None:
    """Best-effort conversion of a GraphMessage-like object into a plain dict."""

    if output is None:
        return None
    if isinstance(output, Mapping):
        return dict(output)
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    to_dict = getattr(output, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return dict(converted) if isinstance(converted, Mapping) else None
    if is_dataclass(output):
        converted = asdict(output)
        return dict(converted) if isinstance(converted, Mapping) else None
    return None


def is_parseable_answer(answer: Any) -> bool:
    """Return whether ``answer`` is present enough for a dataset oracle to inspect."""

    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    if isinstance(answer, bool):
        return True
    if isinstance(answer, (int, float)):
        return True
    return False
