#!/usr/bin/env python3
"""Evaluate a deterministic graph built from a fixed action sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (  # noqa: E402
    compact_sequence,
    execute_graph_eval_item,
    running_accuracy,
    summarize_rows,
    write_json,
    write_jsonl,
    write_tsv,
)
from gogagent.artifacts import prepare_run_dir, safe_path  # noqa: E402
from gogagent.actions.base import ActionConstraints, ActionName  # noqa: E402
from gogagent.actions.registry import apply_action, get_action_spec, is_action_legal  # noqa: E402
from gogagent.config import llm_client_from_env  # noqa: E402
from gogagent.datasets import DatasetExample, enrich_example, load_examples, load_selection  # noqa: E402
from gogagent.graph.factory import make_initial_graph  # noqa: E402
from gogagent.graph.schema import Graph  # noqa: E402
from gogagent.llm import AgentContext  # noqa: E402


def main() -> None:
    args = parse_args()
    actions = parse_actions(args.actions)
    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    validate_action_sequence(actions)
    construct_fixed_graph(
        actions=actions,
        graph_id="fixed_action_validation",
        constraints=constraints,
    )

    examples = load_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )
    selection = load_selection(args.selection_jsonl)
    run_dir = prepare_run_dir(args.output_dir, args.run_id, overwrite=args.overwrite)

    client = None if args.construct_only else llm_client_from_env(args.env)
    context = None if client is None else AgentContext(llm_client=client)

    rows: list[dict[str, Any]] = []
    progress_enabled = not args.no_progress
    iterator = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc=f"Fixed {args.dataset} eval",
        unit="item",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for index, example in iterator:
        enriched = enrich_example(example, selection)
        row = run_one(
            index=index,
            example=enriched,
            actions=actions,
            context=context,
            client_description=dict(client.describe()) if client is not None else None,
            run_dir=run_dir,
            constraints=constraints,
            construct_only=args.construct_only,
            save_item_artifacts=not args.no_item_artifacts,
        )
        rows.append(row)
        iterator.set_postfix(
            {
                "acc": f"{running_accuracy(rows):.3f}",
                "seq": compact_sequence(row.get("action_sequence", [])),
            }
        )
        progress_message = json.dumps(
            {
                "index": index,
                "total": len(examples),
                "task_id": row["task_id"],
                "dataset": row.get("dataset"),
                "subject": row.get("subject"),
                "actions": row.get("action_sequence"),
                "prediction": row.get("prediction"),
                "gold": row.get("gold"),
                "correct": row.get("correct"),
                "llm_call_count": row.get("llm_call_count"),
                "status": row["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if progress_enabled:
            tqdm.write(progress_message)
        else:
            print(progress_message, flush=True)

    summary = summarize_rows(
        rows=rows,
        run_dir=run_dir,
        metadata={
            "eval_type": "fixed_actions",
            "dataset": args.dataset,
            "data_path": str(args.data_path),
            "split": args.split,
            "selection_jsonl": str(args.selection_jsonl) if args.selection_jsonl else None,
            "fixed_actions": [action.value for action in actions],
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        backend=client.describe() if client is not None else None,
    )
    write_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "results.jsonl", rows)
    write_tsv(run_dir / "results.tsv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mmlu", choices=("mmlu", "gsm8k", "humaneval"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection-jsonl", type=Path, default=None)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["STOP"],
        help=(
            "Fixed action sequence. Accepts space-separated tokens or comma-separated "
            "chunks, e.g. --actions ADD_PLAN_SKETCH UP STOP."
        ),
    )
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "evals" / "fixed",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--no-item-artifacts", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_one(
    *,
    index: int,
    example: DatasetExample,
    actions: Sequence[ActionName],
    context: AgentContext | None,
    client_description: Mapping[str, Any] | None,
    run_dir: Path,
    constraints: ActionConstraints,
    construct_only: bool,
    save_item_artifacts: bool,
) -> dict[str, Any]:
    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    graph, construction = construct_fixed_graph(
        actions=actions,
        graph_id=f"fixed_{safe_path(task_id)}",
        constraints=constraints,
    )
    return execute_graph_eval_item(
        index=index,
        example=example,
        graph=graph,
        construction=construction,
        context=context,
        client_description=client_description,
        run_dir=run_dir,
        construct_only=construct_only,
        save_item_artifacts=save_item_artifacts,
    )


def construct_fixed_graph(
    *,
    actions: Sequence[ActionName],
    graph_id: str,
    constraints: ActionConstraints,
) -> tuple[Graph, dict[str, Any]]:
    graph = make_initial_graph(graph_id=graph_id)
    graph.metadata.update({"construction": "fixed_actions"})
    records: list[dict[str, Any]] = []
    sequence: list[str] = []
    stopped = False

    for step, action in enumerate(actions, start=1):
        before = graph.to_dict()
        legality = is_action_legal(graph, action, constraints)
        spec = get_action_spec(action)
        record = {
            "event": "fixed_action",
            "step": step,
            "action": action.value,
            "description": spec.description,
            "legal": legality.legal,
            "reason": legality.reason,
            "before": before,
        }
        sequence.append(action.value)
        if not legality.legal:
            record["after"] = before
            records.append(record)
            raise RuntimeError(f"fixed action {action.value} is illegal: {legality.reason}")
        if action == ActionName.STOP:
            stopped = True
            record["after"] = before
            records.append(record)
            break
        graph = apply_action(graph, action)
        record["after"] = graph.to_dict()
        records.append(record)

    graph.metadata.update(
        {
            "fixed_actions": list(sequence),
            "fixed_stopped": stopped,
            "constraints": {
                "max_depth": constraints.max_depth,
                "max_nodes": constraints.max_nodes,
            },
        }
    )
    return graph, {
        "action_sequence": sequence,
        "action_records": records,
        "stopped": stopped,
        "constraints": {
            "max_depth": constraints.max_depth,
            "max_nodes": constraints.max_nodes,
        },
        "stop_only": sequence == [ActionName.STOP.value],
    }


def parse_actions(raw_actions: Sequence[str]) -> list[ActionName]:
    actions: list[ActionName] = []
    for raw in raw_actions:
        for token in raw.split(","):
            cleaned = token.strip().upper()
            if not cleaned or cleaned in {"NONE", "[]"}:
                continue
            actions.append(ActionName(cleaned))
    return actions


def validate_action_sequence(actions: Sequence[ActionName]) -> None:
    if ActionName.STOP in actions and actions[-1] != ActionName.STOP:
        raise ValueError("STOP terminates construction and must be the last fixed action")


if __name__ == "__main__":
    main()
