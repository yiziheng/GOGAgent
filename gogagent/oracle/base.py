"""Physically isolated training reward interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class TrainOnlyRewardOracle(ABC):
    """Score completed candidate outputs during training only."""

    @abstractmethod
    def score(self, task: Mapping[str, Any], output: str, gold: Any) -> float:
        """Return an objective training score. Never expose this value to policy state."""
