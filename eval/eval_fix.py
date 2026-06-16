#!/usr/bin/env python3
"""Evaluate a deterministic graph built from a fixed action sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
from gogagent.datasets import (  # noqa: E402
    DatasetExample,
    attach_mmlu_fewshot_examples,
    enrich_example,
    load_examples,
    load_mmlu_fewshot_by_subject,
    load_selection,
)
from gogagent.graph.factory import make_initial_graph, make_solver_supervisor_graph  # noqa: E402
from gogagent.graph.schema import Graph  # noqa: E402
from gogagent.llm import AgentContext  # noqa: E402


def main() -> None:
    args = parse_args()
    actions = parse_actions(args.actions)
    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    if args.graph_template == "fixed_actions":
        validate_action_sequence(actions)
        construct_fixed_graph(
            actions=actions,
            graph_id="fixed_action_validation",
            constraints=constraints,
        )
    else:
        construct_template_graph(
            graph_template=args.graph_template,
            graph_id="fixed_template_validation",
            constraints=constraints,
        )

    examples = load_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )
    selection = load_selection(args.selection_jsonl)
    examples = sorted(
        (enrich_example(example, selection) for example in examples),
        key=selection_order_key,
    )
    fewshot_by_subject = None
    if args.mmlu_shot_count:
        if args.dataset != "mmlu":
            raise ValueError("--mmlu-shot-count is only supported for --dataset mmlu")
        fewshot_by_subject = load_mmlu_fewshot_by_subject(
            args.mmlu_dev_path,
            shot_count=args.mmlu_shot_count,
        )
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
        row = run_one(
            index=index,
            example=example,
            actions=actions,
            context=context,
            client_description=dict(client.describe()) if client is not None else None,
            run_dir=run_dir,
            constraints=constraints,
            graph_template=args.graph_template,
            construct_only=args.construct_only,
            save_item_artifacts=args.debug_artifacts and not args.no_item_artifacts,
            fewshot_by_subject=fewshot_by_subject,
            mmlu_shot_count=args.mmlu_shot_count,
            mmlu_brief_rationale=args.mmlu_brief_rationale,
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
            "graph_template": args.graph_template,
            "fixed_actions": (
                [action.value for action in actions]
                if args.graph_template == "fixed_actions"
                else None
            ),
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "mmlu_shot_count": args.mmlu_shot_count,
            "mmlu_dev_path": str(args.mmlu_dev_path) if args.mmlu_shot_count else None,
            "mmlu_brief_rationale": args.mmlu_brief_rationale,
        },
        backend=client.describe() if client is not None else None,
        include_debug=args.debug_artifacts,
    )
    write_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "results.jsonl", rows)
    write_tsv(run_dir / "results.tsv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="mmlu",
        choices=("mmlu", "mmlu_pro", "gsm8k", "humaneval"),
    )
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
    parser.add_argument(
        "--graph-template",
        choices=("fixed_actions", "solver_supervisor"),
        default="fixed_actions",
        help="Use fixed action replay or a built-in deterministic graph template.",
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
    parser.add_argument(
        "--mmlu-shot-count",
        type=int,
        default=0,
        help="Attach this many same-subject MMLU dev examples to Solver prompts.",
    )
    parser.add_argument(
        "--mmlu-dev-path",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU" / "data" / "dev",
        help="Directory containing canonical <subject>_dev.csv files.",
    )
    parser.add_argument(
        "--mmlu-brief-rationale",
        action="store_true",
        help=(
            "Ask MMLU solver-style agents to return Answer/Reason/Risk while "
            "keeping GraphMessage.answer as one parsed option letter."
        ),
    )
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Save per-item gog.json/gog.svg/trace/summary and verbose debug fields.",
    )
    parser.add_argument(
        "--no-item-artifacts",
        action="store_true",
        help="Deprecated compatibility flag; per-item artifacts are disabled by default.",
    )
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
    graph_template: str,
    construct_only: bool,
    save_item_artifacts: bool,
    fewshot_by_subject: Mapping[str, list[Mapping[str, Any]]] | None,
    mmlu_shot_count: int,
    mmlu_brief_rationale: bool,
) -> dict[str, Any]:
    public_task = dict(example.public_task)
    if fewshot_by_subject is not None:
        public_task = attach_mmlu_fewshot_examples(
            public_task,
            fewshot_by_subject,
            shot_count=mmlu_shot_count,
        )
        example = DatasetExample(dataset=example.dataset, public_task=public_task, gold=example.gold)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    if graph_template == "fixed_actions":
        graph, construction = construct_fixed_graph(
            actions=actions,
            graph_id=f"fixed_{safe_path(task_id)}",
            constraints=constraints,
        )
    else:
        graph, construction = construct_template_graph(
            graph_template=graph_template,
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
        problem_overrides=(
            {"mmlu_agent_output_style": "brief_rationale"}
            if mmlu_brief_rationale
            else None
        ),
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


def construct_template_graph(
    *,
    graph_template: str,
    graph_id: str,
    constraints: ActionConstraints,
) -> tuple[Graph, dict[str, Any]]:
    """Construct a built-in deterministic graph template."""

    if graph_template == "solver_supervisor":
        graph = make_solver_supervisor_graph(graph_id=graph_id)
        sequence = ["SOLVER_SUPERVISOR"]
    else:
        raise ValueError(f"unknown graph template: {graph_template}")

    graph.metadata.update(
        {
            "construction": "graph_template",
            "graph_template": graph_template,
            "constraints": {
                "max_depth": constraints.max_depth,
                "max_nodes": constraints.max_nodes,
            },
        }
    )
    return graph, {
        "action_sequence": sequence,
        "action_records": [
            {
                "event": "graph_template",
                "graph_template": graph_template,
                "action": sequence[0],
                "legal": True,
                "reason": "built-in fixed graph template",
                "after": graph.to_dict(),
            }
        ],
        "stopped": True,
        "constraints": {
            "max_depth": constraints.max_depth,
            "max_nodes": constraints.max_nodes,
        },
        "stop_only": False,
        "graph_template": graph_template,
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


def selection_order_key(example: DatasetExample) -> tuple[bool, int, str]:
    task = example.public_task
    rank = task.get("selection_rank")
    return (
        rank is None,
        int(rank) if rank is not None else 10**9,
        str(task.get("task_id", "")),
    )


if __name__ == "__main__":
    main()
