"""Train-only GSM8K reward oracle kept outside the label-blind adapter."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Any, Mapping

from gogagent.oracle.base import TrainOnlyRewardOracle


_NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?"
)


class GSM8KRewardOracle(TrainOnlyRewardOracle):
    """Score a generated answer by normalized numeric equality with the gold answer."""

    name = "gsm8k"

    def score(self, task: Mapping[str, Any], output: str, gold: Any) -> float:
        del task  # The isolated oracle only needs output and training-only gold.
        return float(self.is_correct(output, gold))

    def is_correct(self, prediction: str, gold: str | Mapping[str, Any]) -> bool:
        predicted_value = self.normalized_numeric(prediction)
        gold_value = self.normalized_numeric(self._gold_text(gold))
        return predicted_value is not None and predicted_value == gold_value

    @classmethod
    def normalized_numeric(cls, text: Any) -> Fraction | None:
        """Return the final numeric value, matching GSM8K's final-answer convention."""

        matches = _NUMBER_RE.findall(str(text))
        if not matches:
            return None
        token = matches[-1].replace(",", "").replace(" ", "")
        try:
            if "/" in token:
                numerator, denominator = token.split("/", maxsplit=1)
                return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
            return Fraction(Decimal(token))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _gold_text(gold: str | Mapping[str, Any]) -> str:
        if isinstance(gold, Mapping):
            return str(gold.get("answer", ""))
        return str(gold)
