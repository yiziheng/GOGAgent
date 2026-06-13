#!/usr/bin/env python3
"""Evaluate a fixed GOG graph on MultiAgentBench-style JSONL tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (  # noqa: E402
    compact_format_result,
    compact_message,
    compact_sequence,
    write_llm_audit,
    write_tsv,
)
from eval.eval_fix import (  # noqa: E402
    construct_fixed_graph,
    construct_template_graph,
    parse_actions,
    validate_action_sequence,
)
from gogagent.actions.base import ActionConstraints, ActionName, total_node_count  # noqa: E402
from gogagent.artifacts import RunRecorder, prepare_run_dir, safe_path, write_json, write_jsonl  # noqa: E402
from gogagent.config.env import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT_SECONDS,
    load_project_env,
    require_env,
)
from gogagent.datasets import DatasetExample, load_examples, make_problem  # noqa: E402
from gogagent.graph.executor import execute_graph  # noqa: E402
from gogagent.llm import AgentContext, OpenAICompatibleClient  # noqa: E402
from gogagent.reward import check_output_format, score_answer  # noqa: E402


def main() -> None:
    args = parse_args()
    actions = parse_actions(args.actions)
    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    if args.graph_template == "fixed_actions":
        validate_action_sequence(actions)
        construct_fixed_graph(
            actions=actions,
            graph_id="multiagentbench_fixed_validation",
            constraints=constraints,
        )
    else:
        construct_template_graph(
            graph_template=args.graph_template,
            graph_id="multiagentbench_template_validation",
            constraints=constraints,
        )

    examples = load_examples(
        dataset="multiagentbench",
        data_path=args.data_path,
        split="main",
        limit=args.limit,
    )
    run_dir = prepare_run_dir(args.output_dir, args.run_id, overwrite=args.overwrite)
    client = None if args.construct_only else client_from_args(args)
    context = None if client is None else AgentContext(llm_client=client)

    rows: list[dict[str, Any]] = []
    iterator = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc="MultiAgentBench fixed eval",
        unit="item",
        dynamic_ncols=True,
        disable=args.no_progress,
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
            save_item_artifacts=args.debug_artifacts,
        )
        rows.append(row)
        iterator.set_postfix(
            {
                "score": _running_task_score(rows),
                "scored": _scored_count(rows),
                "seq": compact_sequence(row.get("action_sequence", [])),
            }
        )
        progress = {
            "index": index,
            "total": len(examples),
            "task_id": row.get("task_id"),
            "scenario": row.get("scenario"),
            "prediction": row.get("prediction"),
            "gold": row.get("gold"),
            "correct": row.get("correct"),
            "status": row.get("status"),
            "llm_call_count": row.get("llm_call_count"),
        }
        if args.no_progress:
            print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
        else:
            tqdm.write(json.dumps(progress, ensure_ascii=False, sort_keys=True))

    summary = summarize_multiagentbench(
        rows=rows,
        run_dir=run_dir,
        metadata={
            "eval_type": "multiagentbench_fixed_actions",
            "dataset": "multiagentbench",
            "data_path": str(args.data_path),
            "graph_template": args.graph_template,
            "fixed_actions": (
                [action.value for action in actions]
                if args.graph_template == "fixed_actions"
                else None
            ),
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "scoring": "multiple-choice label, numeric equivalence, otherwise normalized exact match; missing gold is unscored",
        },
        backend=client.describe() if client is not None else None,
    )
    write_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "results.jsonl", rows)
    write_tsv(run_dir / "results.tsv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["STOP"],
        help="Fixed action sequence, e.g. --actions ADD_TASK_BRIEF ADD_ADVERSARIAL_JUDGE STOP.",
    )
    parser.add_argument(
        "--graph-template",
        choices=("fixed_actions", "solver_supervisor"),
        default="fixed_actions",
    )
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument(
        "--thinking",
        choices=("env", "enabled", "disabled", "none"),
        default="env",
        help="Override GOGAGENT_THINKING for this eval; 'none' sends no provider thinking flag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "evals" / "multiagentbench",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--debug-artifacts", action="store_true")
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
) -> dict[str, Any]:
    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    if graph_template == "fixed_actions":
        graph, construction = construct_fixed_graph(
            actions=actions,
            graph_id=f"multiagentbench_{safe_path(task_id)}",
            constraints=constraints,
        )
    else:
        graph, construction = construct_template_graph(
            graph_template=graph_template,
            graph_id=f"multiagentbench_{safe_path(task_id)}",
            constraints=constraints,
        )

    item_dir = run_dir / f"{index:03d}-{safe_path(task_id)}"
    recorder = RunRecorder(item_dir) if save_item_artifacts else None
    problem = make_problem(example)
    if recorder is not None:
        for record in construction.get("action_records", []):
            recorder.record_trace(record)
        graph_json, graph_svg = recorder.save_graph(graph)
        write_json(item_dir / "input.json", {"problem": problem, "gold": example.gold})
    else:
        graph_json = graph_svg = None

    base_result = {
        "index": index,
        "task_id": task_id,
        "dataset": "multiagentbench",
        "scenario": public_task.get("scenario"),
        "task": public_task.get("task"),
        "question": public_task.get("task"),
        "gold": example.gold,
        "action_sequence": list(construction.get("action_sequence") or []),
        "graph_node_count": total_node_count(graph),
    }
    if recorder is not None:
        base_result["item_dir"] = str(item_dir)
        base_result["construction"] = dict(construction)
        base_result["backend"] = dict(client_description or {})
        base_result["artifacts"] = {
            **recorder.paths(),
            "gog_json": str(graph_json),
            "gog_svg": str(graph_svg),
        }

    if construct_only:
        row = {
            **base_result,
            "status": "constructed",
            "correct": None,
            "format_valid": None,
            "llm_call_count": 0,
        }
        if recorder is not None:
            recorder.save_summary(row)
        return row
    if context is None:
        raise RuntimeError("full evaluation requires AgentContext with llm_client")

    llm_start = len(context.llm_calls)
    audit_start = len(context.llm_audit)
    try:
        output = execute_graph(graph, problem, context=context)
        llm_calls = list(context.llm_calls[llm_start:])
        llm_audit = list(context.llm_audit[audit_start:])
        write_llm_audit(run_dir, base_result, llm_audit)
        format_result = check_output_format(output)
        if _has_gold(example.gold):
            oracle = score_answer("multiagentbench", public_task, output, gold=example.gold)
            correct: bool | None = oracle.correct
            oracle_dict: Mapping[str, Any] | None = oracle.to_dict()
        else:
            correct = None
            oracle_dict = {
                "correct": None,
                "dataset": "multiagentbench",
                "prediction": output.answer,
                "gold": None,
                "reason": "missing gold answer; item is executable but unscored",
                "reward": None,
            }
        if recorder is not None:
            recorder.record_trace(
                {
                    "event": "final_output",
                    "output": output.to_dict(),
                    "format": format_result.to_dict(),
                    "oracle": dict(oracle_dict),
                    "llm_call_count": len(llm_calls),
                    "llm_calls": llm_calls,
                    "llm_audit_count": len(llm_audit),
                }
            )
        row = {
            **base_result,
            "status": "ok",
            "prediction": output.answer,
            "correct": correct,
            "format_valid": format_result.valid,
            "output": compact_message(output),
            "format": compact_format_result(format_result.to_dict()),
            "oracle": dict(oracle_dict),
            "llm_call_count": len(llm_calls),
        }
        if recorder is not None:
            row["llm_calls"] = llm_calls
    except Exception as exc:  # noqa: BLE001 - batch eval records item-level failures.
        llm_calls = list(context.llm_calls[llm_start:])
        llm_audit = list(context.llm_audit[audit_start:])
        write_llm_audit(run_dir, base_result, llm_audit)
        row = {
            **base_result,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "llm_call_count": len(llm_calls),
        }
        if recorder is not None:
            row["llm_calls"] = llm_calls
    if recorder is not None:
        recorder.save_summary(row)
    return row


def summarize_multiagentbench(
    *,
    rows: list[dict[str, Any]],
    run_dir: Path,
    metadata: Mapping[str, Any],
    backend: Mapping[str, Any] | None,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    scored = [row for row in completed if row.get("correct") is not None]
    correct = [row for row in scored if row.get("correct") is True]
    failed = [row for row in rows if row.get("status") == "error"]
    constructed = [row for row in rows if row.get("status") == "constructed"]
    accuracy = round(len(correct) / len(scored), 6) if scored else None
    return {
        **dict(metadata),
        "run_dir": str(run_dir),
        "total": len(rows),
        "completed": len(completed),
        "constructed": len(constructed),
        "failed": len(failed),
        "scored": len(scored),
        "unscored": len(completed) - len(scored),
        "correct": len(correct),
        "accuracy": accuracy,
        "task_score": accuracy,
        "format_valid": sum(1 for row in completed if row.get("format_valid") is True),
        "llm_call_count": sum(int(row.get("llm_call_count") or 0) for row in rows),
        "model": backend.get("model") if backend is not None else None,
        "thinking": backend.get("thinking") if backend is not None else None,
        "results_jsonl": str(run_dir / "results.jsonl"),
        "results_tsv": str(run_dir / "results.tsv"),
        "llm_audit_jsonl": str(run_dir / "llm_audit.jsonl"),
        "summary_json": str(run_dir / "summary.json"),
    }


def client_from_args(args: argparse.Namespace) -> OpenAICompatibleClient:
    load_project_env(args.env)
    thinking = os.environ.get("GOGAGENT_THINKING", DEFAULT_THINKING).strip() or None
    if args.thinking != "env":
        thinking = None if args.thinking == "none" else args.thinking
    return OpenAICompatibleClient(
        base_url=os.environ.get("GOGAGENT_BASE_URL", DEFAULT_BASE_URL).strip()
        or DEFAULT_BASE_URL,
        model=os.environ.get("GOGAGENT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        api_key=require_env("GOGAGENT_API_KEY"),
        timeout=_float_env("GOGAGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_retries=_int_env("GOGAGENT_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        temperature=_float_env("GOGAGENT_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_optional_int_env("GOGAGENT_MAX_TOKENS"),
        thinking=thinking,
    )


def _has_gold(value: Any) -> bool:
    return value is not None and value != "" and value != {}


def _running_task_score(rows: list[Mapping[str, Any]]) -> str:
    scored = [row for row in rows if row.get("status") == "ok" and row.get("correct") is not None]
    if not scored:
        return "n/a"
    correct = sum(1 for row in scored if row.get("correct") is True)
    return f"{correct / len(scored):.3f}"


def _scored_count(rows: list[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("status") == "ok" and row.get("correct") is not None
    )


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


if __name__ == "__main__":
    main()
