#!/usr/bin/env python3
"""Generate BC teacher action trajectories with DeepSeek."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.config import llm_client_from_env
from gogagent.datasets import DatasetExample
from gogagent.graph.factory import make_initial_graph
from train.BC.datasets import load_bc_examples
from train.BC.io_utils import JsonlWriter, TrajectorySummaryAccumulator, write_json
from train.BC.probe import SolverProbeResult, run_solver_probe
from train.BC.teacher import (
    DEFAULT_TEACHER_STYLES,
    TeacherActionProposal,
    TeacherTrajectoryClient,
)
from train.BC.trajectory import build_trajectory


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    client = llm_client_from_env(args.env)
    teacher = TeacherTrajectoryClient(llm_client=client, max_actions=args.max_actions)
    examples = load_bc_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )

    styles = tuple(args.styles or DEFAULT_TEACHER_STYLES)
    trajectories_path = run_dir / "trajectories.jsonl"
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"
    summary_accumulator = TrajectorySummaryAccumulator(run_dir=run_dir)

    with (
        JsonlWriter(trajectories_path) as trajectory_writer,
        JsonlWriter(steps_path) as step_writer,
        tqdm(
            total=None,
            desc="BC trajectories",
            unit="traj",
            dynamic_ncols=True,
        ) as progress_bar,
    ):
        for example_index, example in enumerate(examples, start=1):
            summary_accumulator.observe_task()
            probe_graph = make_initial_graph(
                graph_id=f"bc_probe_{safe_id(task_id(example))}"
            )
            solver_probe = run_solver_probe(
                example=example,
                initial_graph=probe_graph,
                llm_client=client,
            )
            active_styles = ("solver_probe_correct",) if solver_probe.correct else styles
            for style in active_styles:
                initial_graph = make_initial_graph(
                    graph_id=f"bc_initial_{safe_id(task_id(example))}_{style}"
                )
                row = generate_one(
                    example=example,
                    example_index=example_index,
                    style=style,
                    initial_graph=initial_graph,
                    constraints=constraints,
                    teacher=teacher,
                    solver_probe=solver_probe,
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
        "styles": list(styles),
        "probe_policy": "solver_correct_stop_wrong_repair",
        "correct_probe_style": "solver_probe_correct",
        "max_actions": args.max_actions,
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
    parser.add_argument("--dataset", choices=("mmlu", "gsm8k", "humaneval"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "bc_trajectories")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--styles", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument(
        "--allow-missing-stop",
        action="store_false",
        dest="require_stop",
        help="accept teacher trajectories that do not end with STOP",
    )
    parser.set_defaults(require_stop=True)
    return parser.parse_args()


def generate_one(
    *,
    example: DatasetExample,
    example_index: int,
    style: str,
    initial_graph: Any,
    constraints: ActionConstraints,
    teacher: TeacherTrajectoryClient,
    solver_probe: SolverProbeResult,
    require_stop: bool,
) -> dict[str, Any]:
    """Generate and validate one trajectory row."""

    trajectory_id = f"{task_id(example)}::{style}"
    if solver_probe.correct:
        proposal = stop_proposal_from_correct_probe(solver_probe)
    else:
        proposal = teacher.propose(
            task=example.public_task,
            style=style,
            initial_graph=initial_graph,
            constraints=constraints,
            solver_probe=solver_probe.to_dict(),
        )
    build = build_trajectory(
        initial_graph=initial_graph,
        actions=proposal.actions,
        constraints=constraints,
        require_stop=require_stop,
    )
    return {
        "trajectory_id": trajectory_id,
        "example_index": example_index,
        "task_id": task_id(example),
        "dataset": example.dataset,
        "task": dict(example.public_task),
        "style": style,
        "solver_probe": solver_probe.to_dict(),
        "valid": build.valid,
        "actions": [action.value for action in proposal.actions],
        "proposal": proposal.to_dict(),
        "steps": [step.to_dict() for step in build.steps],
        "final_graph": dict(build.final_graph),
        "invalid_steps": [dict(step) for step in build.invalid_steps],
    }


def stop_proposal_from_correct_probe(
    solver_probe: SolverProbeResult,
) -> TeacherActionProposal:
    """Return the canonical STOP trajectory when Solver-only already answered correctly."""

    return TeacherActionProposal(
        actions=(ActionName.STOP,),
        reason="Initial SolverAgent answer matched the train-time gold answer.",
        difficulty="easy",
        failure_type="none",
        expected_graph_shape="SolverAgent only",
        raw_response={
            "source": "solver_probe_correct",
            "solver_probe": {
                "correct": solver_probe.correct,
                "prediction": solver_probe.oracle_result.prediction,
                "reason": solver_probe.oracle_result.reason,
            },
        },
    )


def step_rows_from_trajectory(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a valid trajectory row into step-level BC examples."""

    output: list[dict[str, Any]] = []
    for step in row.get("steps", []):
        output.append(
            {
                "trajectory_id": row["trajectory_id"],
                "task_id": row["task_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "style": row["style"],
                "solver_probe_output": row.get("solver_probe", {}).get("output"),
                "step": step["step"],
                "graph_before": step["graph_before"],
                "legal_actions": step["legal_actions"],
                "target_action": step["target_action"],
            }
        )
    return output


def task_id(example: DatasetExample) -> str:
    """Return a stable task id for an example."""

    return str(example.public_task.get("task_id", "task"))


def safe_id(value: str) -> str:
    """Return a filesystem- and graph-id-friendly id segment."""

    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return cleaned[:80] or "task"


def make_run_dir(output_dir: Path, run_id: str | None) -> Path:
    """Return the run artifact directory."""

    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / run_name


def print_progress(
    row: Mapping[str, Any],
    *,
    generated: int,
) -> None:
    """Print one compact JSON progress row."""

    progress = {
        "trajectory_id": row["trajectory_id"],
        "valid": row["valid"],
        "actions": row.get("actions", []),
        "generated": generated,
        "solver_probe_correct": row.get("solver_probe", {}).get("correct"),
        "style": row["style"],
        "task_id": row["task_id"],
    }
    print_json(progress)


def print_json(data: Mapping[str, Any]) -> None:
    """Print compact JSON without importing another writer."""

    import json

    tqdm.write(json.dumps(dict(data), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
