#!/usr/bin/env python3
"""Evaluate a checkpointed graph-construction policy on a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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
from gogagent.actions.mask import compute_action_mask_with_reasons  # noqa: E402
from gogagent.actions.registry import ACTION_ORDER, apply_action  # noqa: E402
from gogagent.config import llm_client_from_env  # noqa: E402
from gogagent.datasets import DatasetExample, enrich_example, load_examples, load_selection  # noqa: E402
from gogagent.graph.factory import make_initial_graph  # noqa: E402
from gogagent.graph.schema import Graph  # noqa: E402
from gogagent.llm import AgentContext  # noqa: E402
from gogagent.policy import ACTION_SPACE, PolicyRunner, mask_action_logits, top_action_scores  # noqa: E402


def main() -> None:
    args = parse_args()
    examples = load_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )
    selection = load_selection(args.selection_jsonl)
    run_dir = prepare_run_dir(args.output_dir, args.run_id, overwrite=args.overwrite)

    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    runner = PolicyRunner.from_checkpoint(
        args.checkpoint,
        device=args.device,
        task_encoder_device=args.task_encoder_device,
    )
    client = None if args.construct_only else llm_client_from_env(args.env)
    context = None if client is None else AgentContext(llm_client=client)

    rows: list[dict[str, Any]] = []
    progress_enabled = not args.no_progress
    iterator = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc=f"Policy {args.dataset} eval",
        unit="item",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for index, example in iterator:
        enriched = enrich_example(example, selection)
        row = run_one(
            index=index,
            example=enriched,
            runner=runner,
            context=context,
            client_description=dict(client.describe()) if client is not None else None,
            run_dir=run_dir,
            constraints=constraints,
            max_actions=args.max_actions,
            temperature=args.temperature,
            construct_only=args.construct_only,
            save_item_artifacts=args.debug_artifacts and not args.no_item_artifacts,
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
            "eval_type": "policy",
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "data_path": str(args.data_path),
            "split": args.split,
            "selection_jsonl": str(args.selection_jsonl) if args.selection_jsonl else None,
            "max_actions": args.max_actions,
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "temperature": args.temperature,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default="mmlu",
        choices=("mmlu", "mmlu_pro", "gsm8k", "humaneval"),
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection-jsonl", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "evals" / "policy",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-encoder-device", default=None)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
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
    runner: PolicyRunner,
    context: AgentContext | None,
    client_description: Mapping[str, Any] | None,
    run_dir: Path,
    constraints: ActionConstraints,
    max_actions: int,
    temperature: float,
    construct_only: bool,
    save_item_artifacts: bool,
    mmlu_brief_rationale: bool,
) -> dict[str, Any]:
    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    policy_task = dict(public_task)
    graph, construction = construct_policy_graph(
        task=policy_task,
        task_id=task_id,
        runner=runner,
        constraints=constraints,
        max_actions=max_actions,
        temperature=temperature,
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
        extra_base={"policy": construction} if save_item_artifacts else None,
        problem_overrides=(
            {"mmlu_agent_output_style": "brief_rationale"}
            if mmlu_brief_rationale
            else None
        ),
    )


def construct_policy_graph(
    *,
    task: Mapping[str, Any],
    task_id: str,
    runner: PolicyRunner,
    constraints: ActionConstraints,
    max_actions: int,
    temperature: float,
) -> tuple[Graph, dict[str, Any]]:
    graph = make_initial_graph(graph_id=f"policy_{safe_path(task_id)}")
    records: list[dict[str, Any]] = []
    sequence: list[str] = []
    stopped = False

    for step in range(1, max_actions + 1):
        before = graph.to_dict()
        logits = runner.logits(graph, task)
        masked_logits, legal_actions = mask_action_logits(
            logits,
            graph,
            constraints,
            action_space=ACTION_SPACE,
            temperature=temperature,
        )
        action = ACTION_SPACE[int(masked_logits.argmax().item())]
        decisions = compute_action_mask_with_reasons(graph, constraints)
        record = {
            "event": "policy_action",
            "step": step,
            "action": action.value,
            "legal_actions": [item.value for item in legal_actions],
            "legality": {
                candidate.value: decisions[candidate].reason
                for candidate in ACTION_ORDER
                if not decisions[candidate].legal
            },
            "top_actions": top_action_scores(masked_logits),
            "before": before,
        }
        sequence.append(action.value)
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
            "construction": "policy",
            "policy_action_sequence": list(sequence),
            "policy_stopped": stopped,
            "policy_max_actions": max_actions,
        }
    )
    return graph, {
        "action_sequence": sequence,
        "action_records": records,
        "stopped": stopped,
        "max_actions": max_actions,
        "constraints": {
            "max_depth": constraints.max_depth,
            "max_nodes": constraints.max_nodes,
        },
        "planner_only": sequence == [
            ActionName.ADD_PLAN_SKETCH.value,
            ActionName.STOP.value,
        ],
        "stop_only": sequence == [ActionName.STOP.value],
    }

if __name__ == "__main__":
    main()
