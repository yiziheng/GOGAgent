"""Shared helpers for dataset evaluation scripts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from gogagent.actions.base import total_node_count
from gogagent.artifacts import RunRecorder, append_jsonl, safe_path, write_json, write_jsonl
from gogagent.datasets import DatasetExample, make_problem
from gogagent.graph.executor import execute_graph
from gogagent.graph.schema import Graph, GraphMessage
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
    problem_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one already-constructed graph and write standard eval artifacts."""

    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    item_dir = run_dir / f"{index:03d}-{safe_path(task_id)}"
    recorder = RunRecorder(item_dir) if save_item_artifacts else None
    problem = make_problem(example)
    problem.update(dict(problem_overrides or {}))
    if recorder is not None:
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
        "action_sequence": list(construction.get("action_sequence") or []),
        "graph_node_count": total_node_count(graph),
        **dict(extra_base or {}),
    }

    if recorder is not None:
        base_result["item_dir"] = str(item_dir)
        base_result["construction"] = dict(construction)
        base_result["backend"] = client_description
        graph_json, graph_svg = recorder.save_graph(graph)
        base_result["artifacts"] = {
            **recorder.paths(),
            "gog_json": str(graph_json),
            "gog_svg": str(graph_svg),
        }
        write_json(item_dir / "input.json", {"problem": problem, "gold": example.gold})

    if construct_only:
        item_summary = {
            **base_result,
            "status": "constructed",
            "correct": None,
            "format_valid": None,
            "llm_call_count": 0,
        }
        if recorder is not None:
            recorder.save_summary(item_summary)
        return item_summary

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
        oracle_result = score_answer(example.dataset, public_task, output, gold=example.gold)
        if recorder is not None:
            recorder.record_trace(
                {
                    "event": "final_output",
                    "output": output.to_dict(),
                    "format": format_result.to_dict(),
                    "oracle": oracle_result.to_dict(),
                    "llm_call_count": len(llm_calls),
                    "llm_calls": llm_calls,
                    "llm_audit_count": len(llm_audit),
                }
            )
        item_summary = {
            **base_result,
            "status": "ok",
            "prediction": output.answer,
            "correct": oracle_result.correct,
            "format_valid": format_result.valid,
            "output": compact_message(output),
            "format": compact_format_result(format_result.to_dict()),
            "oracle": oracle_result.to_dict(),
            "llm_call_count": len(llm_calls),
        }
        if recorder is not None:
            item_summary["llm_calls"] = llm_calls
    except Exception as exc:  # noqa: BLE001 - batch eval records item-level failures.
        llm_calls = list(context.llm_calls[llm_start:])
        llm_audit = list(context.llm_audit[audit_start:])
        write_llm_audit(run_dir, base_result, llm_audit)
        item_summary = {
            **base_result,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "llm_call_count": len(llm_calls),
        }
        if recorder is not None:
            item_summary["llm_calls"] = llm_calls
    if recorder is not None:
        recorder.save_summary(item_summary)
    return item_summary


def summarize_rows(
    *,
    rows: list[dict[str, Any]],
    run_dir: Path,
    metadata: Mapping[str, Any],
    backend: Mapping[str, Any] | None,
    include_debug: bool = False,
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "ok"]
    correct = [row for row in completed if row.get("correct") is True]
    failed = [row for row in rows if row["status"] == "error"]
    constructed = [row for row in rows if row["status"] == "constructed"]
    stop_only = [row for row in rows if is_stop_only(row)]
    non_stop = [row for row in rows if not is_stop_only(row)]

    action_counter: Counter[str] = Counter()
    first_action_counter: Counter[str] = Counter()
    sequence_counter: Counter[str] = Counter()
    for row in rows:
        sequence = tuple(row.get("action_sequence") or ())
        sequence_counter[" -> ".join(sequence) or "<empty>"] += 1
        if sequence:
            first_action_counter[str(sequence[0])] += 1
        action_counter.update(str(action) for action in sequence)

    summary = {
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
        "model": backend.get("model") if backend is not None else None,
        "results_jsonl": str(run_dir / "results.jsonl"),
        "results_tsv": str(run_dir / "results.tsv"),
        "llm_audit_jsonl": str(run_dir / "llm_audit.jsonl"),
        "summary_json": str(run_dir / "summary.json"),
    }
    if include_debug:
        summary["backend"] = dict(backend) if backend is not None else None
    return summary


def is_stop_only(row: Mapping[str, Any]) -> bool:
    construction = row.get("construction")
    if isinstance(construction, Mapping) and construction.get("stop_only") is True:
        return True
    return list(row.get("action_sequence") or []) == ["STOP"]


def compact_message(message: GraphMessage) -> dict[str, Any]:
    """Return a compact message for daily experiment result files."""

    data = {
        "sender": message.sender,
        "role": message.role,
        "content": message.content,
        "answer": message.answer,
        "confidence": message.confidence,
    }
    if message.notes:
        data["notes"] = dict(message.notes)
    metadata = {
        key: value
        for key, value in message.metadata.items()
        if key
        not in {
            "llm",
            "llm_calls",
            "llm_audit",
            "raw_output",
            "subgraph_output",
        }
    }
    if metadata:
        data["metadata"] = metadata
    return data


def compact_format_result(format_result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact format-check record without duplicating LLM audits."""

    compact = {
        "valid": format_result.get("valid"),
        "reason": format_result.get("reason"),
        "answer": format_result.get("answer"),
        "reward": format_result.get("reward"),
    }
    message = format_result.get("message")
    if isinstance(message, Mapping):
        compact["message"] = compact_message(GraphMessage.from_dict(message))
    return compact


def write_llm_audit(
    run_dir: Path,
    base_result: Mapping[str, Any],
    events: list[Mapping[str, Any]],
) -> None:
    """Append full LLM request/response events to a run-level audit JSONL."""

    for call_index, event in enumerate(events, start=1):
        append_jsonl(
            run_dir / "llm_audit.jsonl",
            {
                "index": base_result.get("index"),
                "task_id": base_result.get("task_id"),
                "local_task_id": base_result.get("local_task_id"),
                "dataset": base_result.get("dataset"),
                "subject": base_result.get("subject"),
                "call_index": call_index,
                **dict(event),
            },
        )


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
