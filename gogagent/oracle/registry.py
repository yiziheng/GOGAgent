"""Train-only reward registry.

Inference code must not import this module.
"""

from __future__ import annotations

from gogagent.oracle.base import TrainOnlyRewardOracle
from gogagent.oracle.gsm8k import GSM8KRewardOracle
from gogagent.oracle.humaneval import HumanEvalRewardOracle
from gogagent.oracle.mmlu import MMLURewardOracle


def get_oracle(name: str) -> TrainOnlyRewardOracle:
    oracles: dict[str, type[TrainOnlyRewardOracle]] = {
        "gsm8k": GSM8KRewardOracle,
        "mmlu": MMLURewardOracle,
        "humaneval": HumanEvalRewardOracle,
    }
    try:
        return oracles[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unsupported oracle {name!r}; choose from {sorted(oracles)}") from error
