"""Step-level behavior cloning dataset for graph-construction policy training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.utils.data import Dataset

from gogagent.actions.base import ActionName
from gogagent.graph.schema import Graph
from gogagent.policy import ACTION_SPACE, action_to_index


@dataclass(frozen=True)
class BCStepExample:
    """One supervised graph-construction decision."""

    trajectory_id: str
    task_id: str
    dataset: str
    task: Mapping[str, Any]
    style: str
    step: int
    graph_before: Graph
    legal_actions: tuple[ActionName, ...]
    target_action: ActionName

    @property
    def target_index(self) -> int:
        """Return the stable action index for the target action."""

        return action_to_index(self.target_action)

    def legal_action_mask(self, *, device: str | torch.device | None = None) -> torch.Tensor:
        """Return a boolean mask aligned with ``ACTION_SPACE``."""

        legal = set(self.legal_actions)
        return torch.tensor(
            [action in legal for action in ACTION_SPACE],
            dtype=torch.bool,
            device=device,
        )


class BCStepDataset(Dataset[BCStepExample]):
    """Load valid BC step rows from a JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.examples = list(load_bc_step_examples(self.path, limit=limit))
        if not self.examples:
            raise ValueError(f"no BC step examples found in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BCStepExample:
        return self.examples[index]


def load_bc_step_examples(
    path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[BCStepExample]:
    """Yield BC step examples from ``steps.jsonl``."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"BC steps file does not exist: {source}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    yielded = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_number} expected one JSON object")
            example = bc_step_from_row(row, source=source, line_number=line_number)
            yield example
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def bc_step_from_row(
    row: Mapping[str, Any],
    *,
    source: str | Path = "<memory>",
    line_number: int = 0,
) -> BCStepExample:
    """Parse and validate one step JSON object."""

    graph_before = _required_mapping(row, "graph_before", source, line_number)
    task = _required_mapping(row, "task", source, line_number)
    legal_actions = tuple(
        _parse_action(action, "legal_actions", source, line_number)
        for action in _required_list(row, "legal_actions", source, line_number)
    )
    target_action = _parse_action(
        row.get("target_action"),
        "target_action",
        source,
        line_number,
    )
    if target_action not in legal_actions:
        raise ValueError(
            f"{source}:{line_number} target_action {target_action.value!r} "
            "is not in legal_actions"
        )

    return BCStepExample(
        trajectory_id=str(row.get("trajectory_id", "")),
        task_id=str(row.get("task_id", task.get("task_id", ""))),
        dataset=str(row.get("dataset", "")),
        task=dict(task),
        style=str(row.get("style", "")),
        step=int(row.get("step", 0)),
        graph_before=Graph.from_dict(graph_before),
        legal_actions=legal_actions,
        target_action=target_action,
    )


def _required_mapping(
    row: Mapping[str, Any],
    key: str,
    source: str | Path,
    line_number: int,
) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}:{line_number} missing mapping field {key!r}")
    return value


def _required_list(
    row: Mapping[str, Any],
    key: str,
    source: str | Path,
    line_number: int,
) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{source}:{line_number} missing list field {key!r}")
    if not value:
        raise ValueError(f"{source}:{line_number} field {key!r} must not be empty")
    return value


def _parse_action(
    value: Any,
    field: str,
    source: str | Path,
    line_number: int,
) -> ActionName:
    try:
        return ActionName(str(value))
    except ValueError as exc:
        raise ValueError(
            f"{source}:{line_number} unknown action in {field}: {value!r}"
        ) from exc


__all__ = [
    "BCStepDataset",
    "BCStepExample",
    "bc_step_from_row",
    "load_bc_step_examples",
]
