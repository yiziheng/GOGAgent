#!/usr/bin/env python3
"""Generate BC trajectories by trying curated graph methods until one succeeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints
from gogagent.config import llm_client_from_env
from train.BC.datasets import load_bc_examples
from train.BC.execution_teacher import (
    DEFAULT_EXECUTION_METHODS,
    ExecutionTeacherError,
    ExecutionVerifiedTeacher,
    normalize_methods,
)
from train.BC.generate_trajectories import (
    make_run_dir,
    print_json,
    safe_id,
    step_rows_from_trajectory,
    task_id,
)
from train.BC.io_utils import JsonlWriter, TrajectorySummaryAccumulator, write_json


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    client = llm_client_from_env(args.env)
    methods = load_methods(args.methods_json)
    teacher = ExecutionVerifiedTeacher(
        llm_client=client,
        methods=methods,
        require_stop=args.require_stop,
    )
    examples = load_bc_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )

    trajectories_path = run_dir / "trajectories.jsonl"
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"
    summary_accumulator = TrajectorySummaryAccumulator(run_dir=run_dir)

    with (
        JsonlWriter(trajectories_path) as trajectory_writer,
        JsonlWriter(steps_path) as step_writer,
        tqdm(
            total=None,
            desc="Execution BC trajectories",
            unit="task",
            dynamic_ncols=True,
        ) as progress_bar,
    ):
        for example_index, example in enumerate(examples, start=1):
            summary_accumulator.observe_task()
            row = generate_one(
                example=example,
                example_index=example_index,
                constraints=constraints,
                teacher=teacher,
            )
            trajectory_writer.write(row)
            summary_accumulator.observe_trajectory(row)
            if row["valid"]:
                step_rows = step_rows_from_trajectory(row)
                for step_row in step_rows:
                    step_writer.write(step_row)
                summary_accumulator.observe_steps(len(step_rows))
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "valid": summary_accumulator.valid_trajectories,
                    "invalid": summary_accumulator.invalid_trajectories,
                }
            )
            print_progress(row, generated=trajectory_writer.count)

    if summary_accumulator.total_tasks == 0:
        raise RuntimeError(
            f"no examples found for dataset={args.dataset!r} path={args.data_path}"
        )

    summary = {
        **summary_accumulator.to_dict(),
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "split": args.split,
        "teacher": "execution_verified_first_success",
        "method_library": [[action.value for action in method] for method in methods],
        "constraints": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        "backend": dict(client.describe()),
        "trajectories_jsonl": str(trajectories_path),
        "steps_jsonl": str(steps_path),
        "summary_json": str(summary_path),
    }
    write_json(summary_path, summary)
    print_json(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("mmlu", "mmlu_pro", "gsm8k", "humaneval"),
        required=True,
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "bc_trajectories",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument(
        "--methods-json",
        type=Path,
        default=None,
        help="Optional JSON array of action arrays, e.g. [[\"STOP\"], [\"ADD_PLAN_SKETCH\", \"STOP\"]].",
    )
    parser.add_argument(
        "--allow-missing-stop",
        action="store_false",
        dest="require_stop",
        help="accept methods that do not end with STOP",
    )
    parser.set_defaults(require_stop=True)
    return parser.parse_args()


def load_methods(path: Path | None):
    """Load custom methods or return the curated default library."""

    if path is None:
        return DEFAULT_EXECUTION_METHODS
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("methods JSON must be an array of action arrays")
    return normalize_methods(data)


def generate_one(
    *,
    example: Any,
    example_index: int,
    constraints: ActionConstraints,
    teacher: ExecutionVerifiedTeacher,
) -> dict[str, Any]:
    """Generate one first-success trajectory row."""

    trajectory_id = f"{task_id(example)}::execution_verified"
    try:
        result = teacher.propose(
            task=example.public_task,
            dataset=example.dataset,
            gold=example.gold,
            constraints=constraints,
            graph_id_prefix=f"bc_exec_{safe_id(task_id(example))}",
        )
    except ExecutionTeacherError as error:
        return {
            "trajectory_id": trajectory_id,
            "example_index": example_index,
            "task_id": task_id(example),
            "dataset": example.dataset,
            "task": dict(example.public_task),
            "style": "execution_verified_first_success",
            "solver_probe": {},
            "valid": False,
            "actions": [],
            "proposal": {
                "source": "execution_verified_teacher",
                "error": str(error),
            },
            "execution_teacher": {
                "attempts": [attempt.to_dict() for attempt in error.attempts],
            },
            "steps": [],
            "final_graph": {},
            "invalid_steps": [{"reason": str(error)}],
        }

    return {
        "trajectory_id": trajectory_id,
        "example_index": example_index,
        "task_id": task_id(example),
        "dataset": example.dataset,
        "task": dict(example.public_task),
        "style": "execution_verified_first_success",
        "solver_probe": {},
        "valid": result.build.valid,
        "actions": [action.value for action in result.proposal.actions],
        "proposal": result.proposal.to_dict(),
        "execution_teacher": {
            "attempts": [attempt.to_dict() for attempt in result.attempts],
            "oracle_result": result.oracle_result.to_dict(),
            "output": result.output.to_dict(),
        },
        "steps": [step.to_dict() for step in result.build.steps],
        "final_graph": dict(result.build.final_graph),
        "invalid_steps": [dict(step) for step in result.build.invalid_steps],
    }


def print_progress(row: Mapping[str, Any], *, generated: int) -> None:
    """Print one compact JSON progress row."""

    progress = {
        "trajectory_id": row["trajectory_id"],
        "valid": row["valid"],
        "actions": row.get("actions", []),
        "generated": generated,
        "attempts": len(row.get("execution_teacher", {}).get("attempts", [])),
        "style": row["style"],
        "task_id": row["task_id"],
    }
    print_json(progress)


if __name__ == "__main__":
    main()
