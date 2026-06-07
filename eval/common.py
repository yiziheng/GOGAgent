"""Shared helpers for dataset evaluation scripts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from gogagent.actions.base import total_node_count
from gogagent.artifacts import RunRecorder, safe_path, write_json, write_jsonl
from gogagent.datasets import DatasetExample, make_problem
from gogagent.graph.executor import execute_graph
from gogagent.graph.schema import Graph
from gogagent.llm import AgentContext
from gogagent.reward import check_output_format, score_answer


REPO_ROOT = Path(__file__).resolve().parents[1]


def execute_graph_eval_item(
    *,
    index: int,
    example: DatasetExample,
    graph: Graph,
    construction: Mapping[str, Any],
    context: AgentContext | None,
    client_description: Mapping[str, Any] | None,
    run_dir: Path,
    construct_only: bool,
    save_item_artifacts: bool,
    extra_base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one already-constructed graph and write standard eval artifacts."""

    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    item_dir = run_dir / f"{index:03d}-{safe_path(task_id)}"
    recorder = RunRecorder(item_dir)
    problem = make_problem(example)
    for record in construction.get("action_records", []):
        recorder.record_trace(record)

    base_result = {
        "index": index,
        "task_id": task_id,
        "local_task_id": public_task.get("local_task_id"),
        "dataset": example.dataset,
        "subject": public_task.get("subject"),
        "question": public_task.get("question", public_task.get("prompt")),
        "gold": example.gold,
        "item_dir": str(item_dir),
        "construction": dict(construction),
        "action_sequence": list(construction.get("action_sequence") or []),
        "graph_node_count": total_node_count(graph),
        **dict(extra_base or {}),
    }

    if save_item_artifacts:
        graph_json, graph_svg = recorder.save_graph(graph)
        base_result["artifacts"] = {
            **recorder.paths(),
            "gog_json": str(graph_json),
            "gog_svg": str(graph_svg),
        }
    else:
        base_result["artifacts"] = recorder.paths()

    write_json(item_dir / "input.json", {"problem": problem, "gold": example.gold})

    if construct_only:
        item_summary = {
            **base_result,
            "status": "constructed",
            "correct": None,
            "format_valid": None,
            "llm_call_count": 0,
            "backend": client_description,
        }
        recorder.save_summary(item_summary)
        return item_summary

    if context is None:
        raise RuntimeError("full evaluation requires AgentContext with llm_client")

    llm_start = len(context.llm_calls)
    try:
        output = execute_graph(graph, problem, context=context)
        llm_calls = list(context.llm_calls[llm_start:])
        format_result = check_output_format(output)
        oracle_result = score_answer(example.dataset, public_task, output, gold=example.gold)
        recorder.record_trace(
            {
                "event": "final_output",
                "output": output.to_dict(),
                "format": format_result.to_dict(),
                "oracle": oracle_result.to_dict(),
                "llm_call_count": len(llm_calls),
                "llm_calls": llm_calls,
            }
        )
        item_summary = {
            **base_result,
            "status": "ok",
            "prediction": output.answer,
            "correct": oracle_result.correct,
            "format_valid": format_result.valid,
            "output": output.to_dict(),
            "format": format_result.to_dict(),
            "oracle": oracle_result.to_dict(),
            "llm_call_count": len(llm_calls),
            "llm_calls": llm_calls,
            "backend": client_description,
        }
    except Exception as exc:  # noqa: BLE001 - batch eval records item-level failures.
        llm_calls = list(context.llm_calls[llm_start:])
        item_summary = {
            **base_result,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "llm_call_count": len(llm_calls),
            "llm_calls": llm_calls,
            "backend": client_description,
        }
    recorder.save_summary(item_summary)
    return item_summary


def summarize_rows(
    *,
    rows: list[dict[str, Any]],
    run_dir: Path,
    metadata: Mapping[str, Any],
    backend: Mapping[str, Any] | None,
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "ok"]
    correct = [row for row in completed if row.get("correct") is True]
    failed = [row for row in rows if row["status"] == "error"]
    constructed = [row for row in rows if row["status"] == "constructed"]
    stop_only = [row for row in rows if row.get("construction", {}).get("stop_only")]
    non_stop = [row for row in rows if not row.get("construction", {}).get("stop_only")]

    action_counter: Counter[str] = Counter()
    first_action_counter: Counter[str] = Counter()
    sequence_counter: Counter[str] = Counter()
    for row in rows:
        sequence = tuple(row.get("action_sequence") or ())
        sequence_counter[" -> ".join(sequence) or "<empty>"] += 1
        if sequence:
            first_action_counter[str(sequence[0])] += 1
        action_counter.update(str(action) for action in sequence)

    return {
        **dict(metadata),
        "run_dir": str(run_dir),
        "total": len(rows),
        "completed": len(completed),
        "constructed": len(constructed),
        "failed": len(failed),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(completed), 6) if completed else None,
        "llm_call_count": sum(int(row.get("llm_call_count") or 0) for row in rows),
        "action_distribution": dict(sorted(action_counter.items())),
        "first_action_distribution": dict(sorted(first_action_counter.items())),
        "action_sequence_distribution": dict(sorted(sequence_counter.items())),
        "stop_only_count": len(stop_only),
        "stop_only_accuracy": accuracy_for(stop_only),
        "non_stop_count": len(non_stop),
        "non_stop_accuracy": accuracy_for(non_stop),
        "backend": dict(backend) if backend is not None else None,
        "results_jsonl": str(run_dir / "results.jsonl"),
        "results_tsv": str(run_dir / "results.tsv"),
        "summary_json": str(run_dir / "summary.json"),
    }


def accuracy_for(rows: list[Mapping[str, Any]]) -> float | None:
    completed = [row for row in rows if row.get("status") == "ok"]
    if not completed:
        return None
    correct = [row for row in completed if row.get("correct") is True]
    return round(len(correct) / len(completed), 6)


def running_accuracy(rows: list[Mapping[str, Any]]) -> float:
    completed = [row for row in rows if row.get("status") == "ok"]
    if not completed:
        return 0.0
    return sum(1 for row in completed if row.get("correct") is True) / len(completed)


def compact_sequence(sequence: Iterable[Any]) -> str:
    values = [str(action) for action in sequence]
    if not values:
        return "-"
    return ">".join(action.replace("ADD_", "").replace("_", "") for action in values)


def write_tsv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "index",
        "task_id",
        "dataset",
        "subject",
        "gold",
        "prediction",
        "correct",
        "action_sequence",
        "llm_call_count",
        "status",
        "item_dir",
        "question",
        "error",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns))
        handle.write("\n")
        for row in rows:
            handle.write("\t".join(clean_cell(row.get(column, "")) for column in columns))
            handle.write("\n")


def clean_cell(value: Any) -> str:
    if isinstance(value, list):
        value = " -> ".join(str(item) for item in value)
    return " ".join(str(value).split())
