"""Train-only HumanEval reward oracle backed by an external sandbox."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from typing import Any, Mapping

from gogagent.oracle.base import TrainOnlyRewardOracle


_PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SANDBOX_COMMAND_ENV = "GOGAGENT_HUMANEVAL_SANDBOX_COMMAND"


class HumanEvalRewardOracle(TrainOnlyRewardOracle):
    """Score candidates only through a separately configured trusted sandbox."""

    name = "humaneval"

    def __init__(
        self,
        sandbox_command: str | None = None,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        configured_command = (
            sandbox_command
            if sandbox_command is not None
            else os.environ.get(_SANDBOX_COMMAND_ENV)
        )
        if not configured_command:
            raise RuntimeError(
                "HumanEval scoring requires a trusted external sandbox command via "
                "sandbox_command or GOGAGENT_HUMANEVAL_SANDBOX_COMMAND"
            )
        command = tuple(shlex.split(configured_command))
        if not command:
            raise RuntimeError("HumanEval sandbox command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("HumanEval sandbox timeout must be positive")
        self._sandbox_command = command
        self._timeout_seconds = timeout_seconds

    def score(self, task: Mapping[str, Any], output: str, gold: Any = None) -> float:
        candidate = extract_python_code(output)
        if not candidate:
            return 0.0

        if not isinstance(gold, Mapping) or "test" not in gold:
            raise RuntimeError("HumanEval sandbox scoring requires the training-only test")
        entry_point = str(task.get("entry_point", "")).strip()
        if not entry_point:
            raise RuntimeError("HumanEval sandbox scoring requires an entry_point")

        payload = {
            "candidate_code": candidate,
            "test": str(gold["test"]),
            "entry_point": entry_point,
        }
        try:
            # This configured launcher is the trust boundary; there is no local executor fallback.
            completed = subprocess.run(
                self._sandbox_command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
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
        return 1.0 if result["passed"] else 0.0


def extract_python_code(output: str) -> str:
    """Prefer fenced code while accepting plain Python from simple backends."""

    matches = _PYTHON_FENCE.findall(output)
    if matches:
        return matches[0].strip()
    return output.strip()
