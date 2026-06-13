#!/usr/bin/env python3
"""Build a balanced BC step dataset from hard-rescue and easy-STOP trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName  # noqa: E402
from gogagent.datasets import DatasetExample, load_examples  # noqa: E402
from gogagent.graph.factory import make_initial_graph  # noqa: E402
from train.BC.generate_trajectories import (  # noqa: E402
    safe_id,
    step_rows_from_trajectory,
    task_id,
)
from train.BC.io_utils import JsonlWriter, write_json  # noqa: E402
from train.BC.trajectory import build_trajectory  # noqa: E402


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_id)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    hard_rows = load_valid_hard_rows(args.hard_trajectories)
    if not hard_rows:
        raise RuntimeError(f"no valid hard trajectories found in {args.hard_trajectories}")

    easy_examples = load_examples(
        dataset=args.dataset,
        data_path=args.easy_data,
        split=args.split,
        limit=None,
    )
    easy_count = args.easy_count if args.easy_count is not None else len(hard_rows)
    if easy_count < 0:
        raise ValueError("--easy-count must be non-negative")
    if easy_count > len(easy_examples) and not args.allow_replacement:
        raise ValueError(
            f"requested {easy_count} easy examples, but only {len(easy_examples)} are available; "
            "use --allow-replacement to sample with replacement"
        )
    easy_rows = build_easy_stop_rows(
        examples=easy_examples,
        easy_count=easy_count,
        seed=args.seed,
        allow_replacement=args.allow_replacement,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )

    combined_rows = [*hard_rows, *easy_rows]
    if args.shuffle:
        random.Random(args.seed).shuffle(combined_rows)

    trajectories_path = run_dir / "trajectories.jsonl"
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"
    step_rows: list[dict[str, Any]] = []
    with (
        JsonlWriter(trajectories_path) as trajectory_writer,
        JsonlWriter(steps_path) as step_writer,
    ):
        for row in tqdm(
            combined_rows,
            desc="Writing balanced BC",
            unit="traj",
            dynamic_ncols=True,
            disable=args.no_progress,
        ):
            trajectory_writer.write(row)
            rows = step_rows_from_trajectory(row)
            step_rows.extend(rows)
            for step_row in rows:
                step_writer.write(step_row)

    summary = make_summary(
        run_dir=run_dir,
        hard_rows=hard_rows,
        easy_rows=easy_rows,
        combined_rows=combined_rows,
        step_rows=step_rows,
        args=args,
        trajectories_path=trajectories_path,
        steps_path=steps_path,
        summary_path=summary_path,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-trajectories", type=Path, required=True)
    parser.add_argument("--easy-data", type=Path, required=True)
    parser.add_argument("--dataset", default="mmlu", choices=("mmlu", "gsm8k", "humaneval"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "bc_trajectories")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--easy-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--allow-replacement", action="store_true")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(shuffle=True)
    return parser.parse_args()


def load_valid_hard_rows(path: Path) -> list[dict[str, Any]]:
    """Load valid non-STOP-only hard-rescue trajectories."""

    rows = []
    for row in read_jsonl(path):
        if not row.get("valid"):
            continue
        actions = tuple(str(action) for action in row.get("actions", []))
        if not actions or actions == ("STOP",):
            continue
        rows.append(dict(row))
    return rows


def build_easy_stop_rows(
    *,
    examples: list[DatasetExample],
    easy_count: int,
    seed: int,
    allow_replacement: bool,
    max_depth: int,
    max_nodes: int,
) -> list[dict[str, Any]]:
    """Create synthetic valid STOP trajectories for direct-easy examples."""

    rng = random.Random(seed)
    if allow_replacement:
        selected = [rng.choice(examples) for _ in range(easy_count)]
    else:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        selected = shuffled[:easy_count]

    constraints = ActionConstraints(max_depth=max_depth, max_nodes=max_nodes)
    rows = []
    for index, example in enumerate(selected, start=1):
        graph = make_initial_graph(graph_id=f"bc_easy_stop_{safe_id(task_id(example))}_{index}")
        build = build_trajectory(
            initial_graph=graph,
            actions=(ActionName.STOP,),
            constraints=constraints,
            require_stop=True,
        )
        row = {
            "trajectory_id": f"{task_id(example)}::easy_stop::{index}",
            "example_index": index,
            "task_id": task_id(example),
            "dataset": example.dataset,
            "task": dict(example.public_task),
            "style": "easy_stop",
            "solver_probe": {
                "source": "direct_easy_subset",
                "correct": True,
                "output": None,
            },
            "valid": build.valid,
            "actions": [action.value for action in build.actions],
            "proposal": {
                "actions": [ActionName.STOP.value],
                "reason": "Direct Solver-only answer was correct in hard-mining, so STOP is the preferred cheap action.",
                "difficulty": "easy",
                "failure_type": "none",
                "expected_graph_shape": "SolverAgent only",
                "raw_response": {"source": "synthetic_easy_stop"},
            },
            "steps": [step.to_dict() for step in build.steps],
            "final_graph": dict(build.final_graph),
            "invalid_steps": [dict(step) for step in build.invalid_steps],
        }
        if not row["valid"]:
            raise RuntimeError(f"synthetic STOP trajectory unexpectedly invalid: {row}")
        rows.append(row)
    return rows


def make_summary(
    *,
    run_dir: Path,
    hard_rows: list[Mapping[str, Any]],
    easy_rows: list[Mapping[str, Any]],
    combined_rows: list[Mapping[str, Any]],
    step_rows: list[Mapping[str, Any]],
    args: argparse.Namespace,
    trajectories_path: Path,
    steps_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Return a compact summary for the combined BC dataset."""

    sequence_counts = Counter(action_sequence(row) for row in combined_rows)
    target_counts = Counter(str(row["target_action"]) for row in step_rows)
    return {
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hard_trajectories": str(args.hard_trajectories),
        "easy_data": str(args.easy_data),
        "hard_valid_non_stop_trajectories": len(hard_rows),
        "easy_stop_trajectories": len(easy_rows),
        "total_trajectories": len(combined_rows),
        "total_steps": len(step_rows),
        "target_action_distribution": dict(sorted(target_counts.items())),
        "trajectory_sequence_distribution": dict(sorted(sequence_counts.items())),
        "seed": args.seed,
        "shuffle": args.shuffle,
        "constraints": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        "trajectories_jsonl": str(trajectories_path),
        "steps_jsonl": str(steps_path),
        "summary_json": str(summary_path),
    }


def action_sequence(row: Mapping[str, Any]) -> str:
    """Return a readable action sequence key."""

    return " -> ".join(str(action) for action in row.get("actions", []))


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} expected a JSON object")
            yield row


def make_run_dir(output_dir: Path, run_id: str | None) -> Path:
    """Return the output run directory."""

    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / run_name


if __name__ == "__main__":
    main()
