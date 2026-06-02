"""Train-only, compile-based HumanEval reward oracle."""

from __future__ import annotations

import re
from typing import Any, Mapping

from gogagent.oracle.base import TrainOnlyRewardOracle


_PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class HumanEvalRewardOracle(TrainOnlyRewardOracle):
    """Score Python syntax without executing code or reading tests or gold output."""

    name = "humaneval"

    def score(self, task: Mapping[str, Any], output: str, gold: Any = None) -> float:
        # HumanEval tests, gold code, and canonical solutions are intentionally ignored.
        del task, gold
        candidate = extract_python_code(output)
        if not candidate:
            return 0.0
        try:
            compile(candidate, "<humaneval-candidate>", "exec")
        except (SyntaxError, ValueError, TypeError):
            return 0.0
        return 1.0


def extract_python_code(output: str) -> str:
    """Prefer fenced code while accepting plain Python from simple backends."""

    matches = _PYTHON_FENCE.findall(output)
    if matches:
        return matches[0].strip()
    return output.strip()
