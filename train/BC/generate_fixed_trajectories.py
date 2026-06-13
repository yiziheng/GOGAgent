#!/usr/bin/env python3
"""Generate BC trajectories from one fixed action sequence.

This is useful when we want to teach the policy a task-level construction
template without asking an LLM teacher to invent actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName  # noqa: E402
from gogagent.graph.factory import make_initial_graph  # noqa: E402
from train.BC.datasets import load_bc_examples  # noqa: E402
from train.BC.generate_trajectories import (  # noqa: E402
    make_run_dir,
    print_json,
    safe_id,
    step_rows_from_trajectory,
    task_id,
)
from train.BC.io_utils import (  # noqa: E402
    JsonlWriter,
    TrajectorySummaryAccumulator,
    write_json,
)
from train.BC.trajectory import build_trajectory  # noqa: E402


DEFAULT_ACTIONS = (
    ActionName.ADD_ADVERSARIAL_JUDGE,
    ActionName.UP,
    ActionName.STOP,
)


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    actions = tuple(ActionName(action) for action in args.actions)
    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
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
            desc="Fixed BC trajectories",
            unit="task",
            dynamic_ncols=True,
        ) as progress_bar,
    ):
        for example_index, example in enumerate(examples, start=1):
            summary_accumulator.observe_task()
            initial_graph = make_initial_graph(
                graph_id=f"bc_fixed_{safe_id(task_id(example))}"
            )
            row = generate_one(
                example=example,
                example_index=example_index,
                actions=actions,
                initial_graph=initial_graph,
                constraints=constraints,
                require_stop=args.require_stop,
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
        "teacher": "fixed_action_sequence",
        "actions": [action.value for action in actions],
        "constraints": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        "trajectories_jsonl": str(trajectories_path),
        "steps_jsonl": str(steps_path),
        "summary_json": str(summary_path),
    }
    write_json(summary_path, summary)
    print_json(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mmlu", "gsm8k", "humaneval"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "bc_trajectories",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=[action.value for action in DEFAULT_ACTIONS],
        help="Fixed action sequence. Default: ADD_ADVERSARIAL_JUDGE UP STOP",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument(
        "--allow-missing-stop",
        action="store_false",
        dest="require_stop",
        help="accept trajectories that do not end with STOP",
    )
    parser.set_defaults(require_stop=True)
    return parser.parse_args()


def generate_one(
    *,
    example: Any,
    example_index: int,
    actions: tuple[ActionName, ...],
    initial_graph: Any,
    constraints: ActionConstraints,
    require_stop: bool,
) -> dict[str, Any]:
    """Replay one fixed action sequence for one task."""

    build = build_trajectory(
        initial_graph=initial_graph,
        actions=actions,
        constraints=constraints,
        require_stop=require_stop,
    )
    return {
        "trajectory_id": f"{task_id(example)}::fixed_actions",
        "example_index": example_index,
        "task_id": task_id(example),
        "dataset": example.dataset,
        "task": dict(example.public_task),
        "style": "fixed_action_sequence",
        "solver_probe": {},
        "valid": build.valid,
        "actions": [action.value for action in actions],
        "proposal": {
            "source": "fixed_action_sequence",
            "reason": "Task-level BC template supplied by configuration.",
            "expected_graph_shape": " -> ".join(action.value for action in actions),
        },
        "steps": [step.to_dict() for step in build.steps],
        "final_graph": dict(build.final_graph),
        "invalid_steps": [dict(step) for step in build.invalid_steps],
    }


def print_progress(row: Mapping[str, Any], *, generated: int) -> None:
    """Print one compact progress row."""

    print_json(
        {
            "trajectory_id": row["trajectory_id"],
            "valid": row["valid"],
            "actions": row.get("actions", []),
            "generated": generated,
            "style": row["style"],
            "task_id": row["task_id"],
        }
    )


if __name__ == "__main__":
    main()
