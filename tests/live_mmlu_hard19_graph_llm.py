#!/usr/bin/env python3
"""Run the live fixed-action GOG flow on the local hard19 MMLU subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from gogagent.artifacts import RunRecorder
from gogagent.config import llm_client_from_env
from gogagent.datasets import DatasetExample, load_mmlu_directory
from gogagent.graph.executor import execute_graph
from gogagent.llm import AgentContext
from gogagent.reward import check_output_format, score_answer

from live_mmlu_graph_llm import build_fixed_action_graph


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    examples = list(load_mmlu_directory(data_dir, split=args.split))
    if not examples:
        raise RuntimeError(f"no MMLU examples found in {data_dir} for split {args.split!r}")
    selection = load_selection(args.selection_jsonl)

    client = llm_client_from_env(args.env)
    context = AgentContext(llm_client=client)
    run_dir = make_run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        enriched = enrich_example(example, selection)
        result = run_one(
            index=index,
            total=len(examples),
            example=enriched,
            context=context,
            client_description=dict(client.describe()),
            run_dir=run_dir,
        )
        rows.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(examples),
                    "task_id": result["task_id"],
                    "subject": result["subject"],
                    "prediction": result.get("prediction"),
                    "gold": result.get("gold"),
                    "correct": result.get("correct"),
                    "llm_call_count": result.get("llm_call_count"),
                    "status": result["status"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    summary = summarize(rows, run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "results.jsonl", rows)
    write_tsv(run_dir / "results.tsv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "hard19" / "val",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--selection-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "hard19" / "selection.jsonl",
    )
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def run_one(
    *,
    index: int,
    total: int,
    example: DatasetExample,
    context: AgentContext,
    client_description: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    del total
    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{index}"))
    item_dir = run_dir / f"{index:02d}-{safe_path(task_id)}"
    item_dir.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(item_dir)

    problem = {
        **public_task,
        "dataset": "mmlu",
        "answer_format": (
            "The GraphMessage answer field must be exactly one option letter: "
            "A, B, C, or D."
        ),
    }
    (item_dir / "input.json").write_text(
        json.dumps({"problem": problem, "gold": example.gold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    graph, action_records = build_fixed_action_graph()
    for record in action_records:
        recorder.record_trace(record)

    base_result = {
        "index": index,
        "task_id": task_id,
        "local_task_id": public_task.get("local_task_id"),
        "subject": public_task.get("subject"),
        "question": public_task.get("question"),
        "gold": example.gold,
        "item_dir": str(item_dir),
    }
    llm_start = len(context.llm_calls)
    try:
        output = execute_graph(graph, problem, context=context)
        llm_calls = list(context.llm_calls[llm_start:])
        format_result = check_output_format(output)
        oracle_result = score_answer("mmlu", {"answer": example.gold}, output)
        graph_json, graph_svg = recorder.save_graph(graph)
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
            "artifacts": {
                **recorder.paths(),
                "gog_json": str(graph_json),
                "gog_svg": str(graph_svg),
            },
        }
    except Exception as exc:  # noqa: BLE001 - batch runner records failures per item.
        llm_calls = list(context.llm_calls[llm_start:])
        graph_json, graph_svg = recorder.save_graph(graph)
        item_summary = {
            **base_result,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "llm_call_count": len(llm_calls),
            "llm_calls": llm_calls,
            "backend": client_description,
            "artifacts": {
                **recorder.paths(),
                "gog_json": str(graph_json),
                "gog_svg": str(graph_svg),
            },
        }
    recorder.save_summary(item_summary)
    return item_summary


def enrich_example(
    example: DatasetExample,
    selection: Mapping[tuple[str, str], Mapping[str, Any]],
) -> DatasetExample:
    public_task = dict(example.public_task)
    local_task_id = public_task.get("task_id")
    key = (
        str(public_task.get("subject", "")),
        str(public_task.get("question", "")),
    )
    source = selection.get(key)
    if source is not None:
        public_task["task_id"] = str(source.get("task_id", local_task_id))
        public_task["hard19_rank"] = source.get("rank")
        public_task["source_row"] = source.get("source_row")
        public_task["source_file"] = source.get("source_file")
    public_task["local_task_id"] = local_task_id
    return DatasetExample(dataset=example.dataset, public_task=public_task, gold=example.gold)


def load_selection(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("subject", "")), str(row.get("question", "")))
            rows[key] = row
    return rows


def summarize(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "ok"]
    correct = [row for row in completed if row.get("correct") is True]
    failed = [row for row in rows if row["status"] != "ok"]
    return {
        "run_dir": str(run_dir),
        "total": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(completed), 6) if completed else 0.0,
        "llm_call_count": sum(int(row.get("llm_call_count") or 0) for row in rows),
        "results_jsonl": str(run_dir / "results.jsonl"),
        "results_tsv": str(run_dir / "results.tsv"),
        "summary_json": str(run_dir / "summary.json"),
    }


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_tsv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "index",
        "task_id",
        "subject",
        "gold",
        "prediction",
        "correct",
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


def make_run_dir(run_id: str | None = None) -> Path:
    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "artifacts" / "tests" / "live_mmlu_hard19_graph_llm" / run_name


def clean_cell(value: Any) -> str:
    return " ".join(str(value).split())


def safe_path(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return cleaned[:96] or "item"


if __name__ == "__main__":
    main()
