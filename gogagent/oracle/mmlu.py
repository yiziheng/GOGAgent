"""Train-only MMLU reward oracle, isolated from inference adapters."""

from __future__ import annotations

import re
from typing import Any, Mapping

from gogagent.oracle.base import TrainOnlyRewardOracle


_OPTION_LABELS = ("A", "B", "C", "D")


class MMLURewardOracle(TrainOnlyRewardOracle):
    """Compare one predicted MMLU option against a training-only gold option."""

    def score(self, task: Mapping[str, Any], output: str, gold: Any) -> float:
        del task
        return float(_extract_option(output) == _extract_option(gold))

    def is_correct(self, output: str, gold: Any) -> bool:
        return bool(self.score({}, output, gold))


def _extract_option(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(_OPTION_LABELS):
            return _OPTION_LABELS[value]
        raise ValueError(f"MMLU option index must be in [0, 3], got {value}")

    text = str(value).strip().upper()
    if text in _OPTION_LABELS:
        return text
    explicit = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*[:=]?\s*\(?([A-D])\)?\b", text)
    if explicit:
        return explicit.group(1)
    standalone = re.findall(r"\b([A-D])\b", text)
    if len(standalone) == 1:
        return standalone[0]
    raise ValueError(f"cannot extract exactly one MMLU option from {value!r}")
