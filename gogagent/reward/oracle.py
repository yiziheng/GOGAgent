"""Dataset correctness reward helpers.

The oracle only answers "is the final answer correct?". It does not decide whether
the output format is valid; that is handled separately in ``format.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
import re
import shlex
import subprocess
from typing import Any, Mapping

from gogagent.reward.format import coerce_graph_message


_MMLU_OPTION_LABELS = ("A", "B", "C", "D")
_MMLU_CHOICE_RE = re.compile(
    r"(?:^|[\s\(\[\{\"'`])(?:option|answer|choice|letter)?\s*[:\-]?\s*([A-D])(?:[\s\)\]\}.:,;\"'`]|$)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?"
)
_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_HUMANEVAL_SANDBOX_ENV = "GOGAGENT_HUMANEVAL_SANDBOX_COMMAND"


@dataclass(frozen=True)
class OracleResult:
    """Dataset oracle result used by the simplified Round 1 reward."""

    correct: bool
    dataset: str
    prediction: Any
    gold: Any
    reason: str

    @property
    def reward(self) -> float:
        """Return the Round 1 answer correctness reward."""

        return 1.0 if self.correct else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "dataset": self.dataset,
            "prediction": self.prediction,
            "gold": self.gold,
            "reason": self.reason,
            "reward": self.reward,
        }


def score_answer(
    dataset: str,
    example: Mapping[str, Any],
    output: Any,
    *,
    gold: Any | None = None,
) -> OracleResult:
    """Score answer correctness for MMLU, GSM8K, HumanEval, or exact-match tasks."""

    dataset_name = dataset.lower()
    prediction = extract_answer(output)
    gold_value = extract_gold(example, gold)
    if prediction is None:
        return OracleResult(False, dataset_name, prediction, gold_value, "missing prediction")
    if gold_value is None:
        return OracleResult(False, dataset_name, prediction, gold_value, "missing gold answer")

    try:
        if dataset_name == "mmlu":
            correct = _score_mmlu(example, prediction, gold_value)
        elif dataset_name == "gsm8k":
            correct = _score_gsm8k(prediction, gold_value)
        elif dataset_name == "humaneval":
            correct = _score_humaneval(example, prediction, gold_value)
        else:
            correct = str(prediction).strip() == str(gold_value).strip()
    except Exception as error:  # noqa: BLE001 - oracle failures become wrong, not format failures.
        return OracleResult(False, dataset_name, prediction, gold_value, str(error))

    return OracleResult(
        bool(correct),
        dataset_name,
        prediction,
        gold_value,
        "correct" if correct else "wrong",
    )


def extract_answer(output: Any) -> Any | None:
    """Extract the candidate answer from a GraphMessage-like object."""

    message = coerce_graph_message(output)
    if message is None:
        return None
    return message.get("answer")


def extract_gold(example: Mapping[str, Any], gold: Any | None = None) -> Any | None:
    """Extract a gold answer from an explicit value or common dataset fields."""

    if gold is not None:
        return gold
    for key in (
        "answer",
        "gold",
        "target",
        "label",
        "final_answer",
        "canonical_answer",
        "expected",
    ):
        if key in example:
            return example[key]
    return None


def _score_mmlu(example: Mapping[str, Any], prediction: Any, gold: Any) -> bool:
    del example
    return _extract_mmlu_option(prediction) == _extract_mmlu_option(gold)


def _extract_mmlu_option(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(_MMLU_OPTION_LABELS):
            return _MMLU_OPTION_LABELS[value]
        raise ValueError(f"MMLU option index must be in [0, 3], got {value}")

    text = str(value).strip().upper()
    if text in _MMLU_OPTION_LABELS:
        return text
    match = _MMLU_CHOICE_RE.search(text)
    if match:
        return match.group(1).upper()
    raise ValueError(f"cannot extract exactly one MMLU option from {value!r}")


def _score_gsm8k(prediction: Any, gold: Any) -> bool:
    predicted_value = _normalized_numeric(prediction)
    gold_value = _normalized_numeric(_gold_text(gold))
    return predicted_value is not None and predicted_value == gold_value


def _normalized_numeric(text: Any) -> Fraction | None:
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


def _gold_text(gold: Any) -> str:
    if isinstance(gold, Mapping):
        return str(gold.get("answer", ""))
    return str(gold)


def _score_humaneval(
    example: Mapping[str, Any],
    prediction: Any,
    gold: Any,
) -> bool:
    """Score HumanEval only when a trusted sandbox is configured."""

    from gogagent.config import load_project_env

    load_project_env()
    command_text = os.environ.get(_HUMANEVAL_SANDBOX_ENV)
    if not command_text:
        raise RuntimeError(f"HumanEval correctness requires {_HUMANEVAL_SANDBOX_ENV}")
    command = tuple(shlex.split(command_text))
    if not command:
        raise RuntimeError("HumanEval sandbox command must not be empty")
    if not isinstance(gold, Mapping) or "test" not in gold:
        raise RuntimeError("HumanEval sandbox scoring requires the gold test")
    entry_point = str(example.get("entry_point", "")).strip()
    if not entry_point:
        raise RuntimeError("HumanEval sandbox scoring requires an entry_point")

    payload = {
        "candidate_code": _extract_python_code(str(prediction)),
        "test": str(gold["test"]),
        "entry_point": entry_point,
    }
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("HumanEval external sandbox timed out") from error
    except OSError as error:
        raise RuntimeError("HumanEval external sandbox could not be started") from error

    if completed.returncode != 0:
        raise RuntimeError(
            f"HumanEval external sandbox exited with status {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("HumanEval external sandbox returned invalid JSON") from error
    if not isinstance(result, Mapping) or type(result.get("passed")) is not bool:
        raise RuntimeError("HumanEval external sandbox JSON must contain boolean 'passed'")
    return bool(result["passed"])


def _extract_python_code(output: str) -> str:
    matches = _PYTHON_FENCE_RE.findall(output)
    if matches:
        return matches[0].strip()
    return output.strip()
