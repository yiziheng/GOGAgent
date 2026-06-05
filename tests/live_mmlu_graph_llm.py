#!/usr/bin/env python3
"""Build a fixed-action GOG and execute one MMLU-style item with a real LLM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.actions.registry import apply_action, get_action_spec, is_action_legal
from gogagent.agents.registry import create_agent
from gogagent.artifacts import RunRecorder
from gogagent.config import llm_client_from_env
from gogagent.graph.executor import execute_graph
from gogagent.graph.schema import Graph, Node
from gogagent.llm import AgentContext
from gogagent.reward import check_output_format, score_answer


FIXED_ACTIONS: tuple[ActionName, ...] = (
    ActionName.ADD_PLAN_SKETCH,
    ActionName.UP,
    ActionName.ADD_FORMAT_VERIFIER,
)


def main() -> None:
    client = llm_client_from_env()
    context = AgentContext(llm_client=client)

    graph, action_records = build_fixed_action_graph()
    problem, gold = mmlu_problem()
    llm_start = len(context.llm_calls)
    output = execute_graph(graph, problem, context=context)
    llm_calls = list(context.llm_calls[llm_start:])
    format_result = check_output_format(output)
    oracle_result = score_answer("mmlu", {"answer": gold}, output)

    recorder = RunRecorder(make_run_dir())
    for record in action_records:
        recorder.record_trace(record)
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
    graph_json, graph_svg = recorder.save_graph(graph)
    summary = {
        "backend": client.describe(),
        "problem": problem,
        "gold": gold,
        "output": output.to_dict(),
        "format": format_result.to_dict(),
        "oracle": oracle_result.to_dict(),
        "llm_call_count": len(llm_calls),
        "llm_calls": llm_calls,
        "actions": action_records,
        "artifacts": {
            **recorder.paths(),
            "gog_json": str(graph_json),
            "gog_svg": str(graph_svg),
        },
    }
    recorder.save_summary(summary)

    print(
        json.dumps(
            {
                "answer": output.answer,
                "correct": oracle_result.correct,
                "format_valid": format_result.valid,
                "llm_call_count": len(llm_calls),
                "artifact_dir": str(recorder.run_dir),
                "gog_json": str(graph_json),
                "gog_svg": str(graph_svg),
                "summary": str(recorder.summary_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def build_fixed_action_graph() -> tuple[Graph, list[dict[str, Any]]]:
    graph = Graph(
        graph_id="live_mmlu_fixed_action_gog",
        in_node="solver",
        out_node="solver",
        nodes={
            "solver": Node(
                node_id="solver",
                name="SolverAgent",
                executor=create_agent("SolverAgent"),
                depth=1,
            )
        },
        edges=[],
        metadata={"construction": "fixed_action_test"},
    )
    constraints = ActionConstraints(max_depth=2, max_nodes=8)
    records: list[dict[str, Any]] = []
    for step, action in enumerate(FIXED_ACTIONS, start=1):
        legality = is_action_legal(graph, action, constraints)
        spec = get_action_spec(action)
        record = {
            "step": step,
            "action": action.value,
            "description": spec.description,
            "legal": legality.legal,
            "reason": legality.reason,
            "before": graph.to_dict(),
        }
        if not legality.legal:
            raise RuntimeError(f"fixed action {action.value} is illegal: {legality.reason}")
        graph = apply_action(graph, action)
        record["after"] = graph.to_dict()
        records.append(record)
    graph.metadata["fixed_actions"] = [action.value for action in FIXED_ACTIONS]
    return graph, records


def mmlu_problem() -> tuple[dict[str, Any], str]:
    return (
        {
            "dataset": "mmlu",
            "subject": "high_school_biology",
            "question": (
                "Which molecule carries genetic information in most living organisms?"
            ),
            "choices": {
                "A": "DNA",
                "B": "Glucose",
                "C": "Sodium chloride",
                "D": "Cholesterol",
            },
            "answer_format": (
                "The GraphMessage answer field must be exactly one option letter: "
                "A, B, C, or D."
            ),
        },
        "A",
    )


def make_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "artifacts" / "tests" / "live_mmlu_graph_llm" / timestamp


if __name__ == "__main__":
    main()
