#!/usr/bin/env python3
"""Round 1 refactor smoke checks for the real project runtime."""

from __future__ import annotations

from dataclasses import asdict
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.artifacts import RunRecorder
from gogagent.llm import AgentContext, LLMClient, LLMJsonResponse, LLMUsage
from gogagent.reward import check_output_format, compute_reward


class ScriptedLLMClient(LLMClient):
    """Explicit test LLM client for smoke checks."""

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
    ) -> LLMJsonResponse:
        del prompt
        data = {
            "role": role,
            "content": f"{role} produced a structured test response.",
            "answer": latest_payload_answer(payload) or "A",
            "confidence": 1.0,
            "notes": {"source": "scripted_test_client"},
            "metadata": {"agent_role": role},
        }
        return LLMJsonResponse(
            data=data,
            raw_text=json.dumps(data, ensure_ascii=False),
            model="scripted-test-client",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_seconds=0.0,
        )


def main() -> None:
    runtime = load_runtime()
    smoke_graph_message(runtime)
    graph = smoke_execute_solver_graph(runtime)
    smoke_action_mask(runtime, graph)
    upgraded_graph = smoke_up_solver(runtime, graph)
    smoke_execute_upgraded_graph(runtime, upgraded_graph)
    smoke_agent_registry()
    smoke_graph_roundtrip(runtime, upgraded_graph)
    smoke_reward()
    smoke_artifacts(upgraded_graph)
    print("Round1 smoke passed (project-runtime)")


def load_runtime() -> dict[str, Any]:
    """Load the real Round 1 graph/action runtime."""

    from gogagent.actions.base import ActionConstraints
    from gogagent.actions.mask import compute_action_mask
    from gogagent.actions.up.apply import apply_up
    from gogagent.agents.registry import create_agent
    from gogagent.graph.executor import execute_graph
    from gogagent.graph.schema import Edge, Graph, GraphMessage, Node

    return {
        "ActionConstraints": ActionConstraints,
        "Edge": Edge,
        "Graph": Graph,
        "GraphMessage": GraphMessage,
        "Node": Node,
        "apply_up": apply_up,
        "compute_action_mask": compute_action_mask,
        "create_agent": create_agent,
        "execute_graph": execute_graph,
    }


def smoke_graph_message(runtime: dict[str, Any]) -> None:
    GraphMessage = runtime["GraphMessage"]
    message = GraphMessage(role="solver", content="answer is B", answer="B")
    assert not isinstance(message, str), "GraphMessage must not be a raw string"
    json.dumps(to_dict(message), ensure_ascii=False)
    assert check_output_format(message).valid


def smoke_execute_solver_graph(runtime: dict[str, Any]) -> Any:
    graph = make_solver_graph(runtime)
    try:
        runtime["execute_graph"](graph, {"question": "2+2?"})
    except RuntimeError as error:
        assert "llm_client" in str(error)
    else:
        raise AssertionError("graph execution without LLMClient must fail")

    result = runtime["execute_graph"](graph, {"question": "2+2?"}, context=test_context())
    assert not isinstance(result, str), "graph execution must return GraphMessage, not str"
    assert check_output_format(result).valid
    return graph


def smoke_action_mask(runtime: dict[str, Any], graph: Any) -> None:
    graph_with_planner = make_solver_graph(runtime, include_planner=True)
    graph_with_verifier = make_solver_graph(runtime, include_verifier=True)
    subgraph_tail = runtime["apply_up"](make_solver_graph(runtime))
    maxed_graph = make_solver_graph(runtime, max_nodes=True)

    assert not action_allowed(
        action_mask(runtime, graph_with_planner, max_nodes=6, max_depth=2),
        "ADD_PLAN_SKETCH",
    )
    assert not action_allowed(
        action_mask(runtime, graph_with_verifier, max_nodes=6, max_depth=2),
        "ADD_FORMAT_VERIFIER",
    )
    assert not action_allowed(
        action_mask(runtime, subgraph_tail, max_nodes=6, max_depth=2),
        "UP",
    )
    assert not any(
        action_allowed(action_mask(runtime, maxed_graph, max_nodes=2, max_depth=2), action)
        for action in ("UP", "ADD_PLAN_SKETCH", "ADD_FORMAT_VERIFIER", "ADD_TASK_BRIEF")
    )
    assert action_allowed(
        action_mask(runtime, graph, max_nodes=6, max_depth=2),
        "UP",
    )


def smoke_up_solver(runtime: dict[str, Any], graph: Any) -> Any:
    upgraded = runtime["apply_up"](graph)
    upgraded_dict = to_dict(upgraded)
    nodes = upgraded_dict.get("nodes", {})
    subgraph_nodes = [
        node for node in nodes.values()
        if isinstance(node, dict) and int(node.get("depth", 1)) > 1
    ]
    assert subgraph_nodes, "UP solver must produce a subgraph node"
    executor = subgraph_nodes[0].get("executor", {})
    assert executor.get("kind") == "graph", "UP node executor must be a nested graph"
    return upgraded


def smoke_execute_upgraded_graph(runtime: dict[str, Any], graph: Any) -> None:
    result = runtime["execute_graph"](graph, {"question": "2+2?"}, context=test_context())
    assert not isinstance(result, str), "upgraded GOG execution must return GraphMessage"
    assert check_output_format(result).valid
    result_dict = to_dict(result)
    assert result_dict["metadata"]["executor"] == "Graph"
    assert "subgraph_output" in result_dict["metadata"]


def smoke_agent_registry() -> None:
    from gogagent.agents.registry import is_standalone_agent

    assert is_standalone_agent("SolverAgent")
    assert is_standalone_agent("PlanSketchAgent")
    assert not is_standalone_agent("ChallengerAgent")
    assert not is_standalone_agent("DefenderAgent")
    assert not is_standalone_agent("JudgeAgent")


def smoke_graph_roundtrip(runtime: dict[str, Any], graph: Any) -> None:
    Graph = runtime["Graph"]
    graph_dict = graph.to_dict()
    restored = Graph.from_dict(graph_dict)
    assert restored.to_dict() == graph_dict
    result = runtime["execute_graph"](restored, {"question": "roundtrip?"}, context=test_context())
    assert check_output_format(result).valid


def smoke_reward() -> None:
    output = {"role": "solver", "content": "I choose B", "answer": "B"}
    reward = compute_reward(
        dataset="mmlu",
        example={"answer": "B"},
        final_output=output,
        action_records=[
            {"action": "ADD_PLAN_SKETCH", "legal": True},
            {"action": "UP", "legal": True},
            {"action": "ADD_FORMAT_VERIFIER", "legal": False},
        ],
    )
    assert reward.answer_correctness == 1.0
    assert reward.format_correctness == 0.0
    assert reward.graph_validity == -0.2
    assert reward.graph_complexity == -0.04
    assert abs(reward.total - 0.76) < 1e-9

    invalid = compute_reward(
        dataset="mmlu",
        example={"answer": "A"},
        final_output={"role": "solver", "content": "missing answer"},
        action_records=[],
    )
    assert invalid.answer_correctness == 0.0
    assert invalid.format_correctness == -0.2


def smoke_artifacts(graph: Any) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="gogagent_round1_smoke_"))
    try:
        recorder = RunRecorder(temp_dir)
        recorder.record_trace({"step": 1, "action": "UP"})
        recorder.save_summary({"ok": True})
        graph_json, graph_svg = recorder.save_graph(graph)
        assert recorder.trace_path.exists() and recorder.trace_path.read_text(encoding="utf-8")
        assert recorder.summary_path.exists()
        assert graph_json.exists() and json.loads(graph_json.read_text(encoding="utf-8"))
        assert graph_svg.exists() and "<svg" in graph_svg.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir)


def make_solver_graph(
    runtime: dict[str, Any],
    *,
    include_planner: bool = False,
    include_verifier: bool = False,
    max_nodes: bool = False,
) -> Any:
    Graph = runtime["Graph"]
    Node = runtime["Node"]
    Edge = runtime["Edge"]
    create_agent = runtime["create_agent"]
    nodes: dict[str, Any] = {
        "solver": Node(
            node_id="solver",
            name="SolverAgent",
            executor=create_agent("SolverAgent"),
            depth=1,
        )
    }
    edges = []
    in_node = "solver"
    out_node = "solver"
    if include_planner:
        nodes["planner"] = Node(
            node_id="planner",
            name="PlanSketchAgent",
            executor=create_agent("PlanSketchAgent"),
            depth=1,
        )
        edges.append(Edge(source="planner", target="solver"))
        in_node = "planner"
    if include_verifier:
        nodes["verifier"] = Node(
            node_id="verifier",
            name="FormatVerifierAgent",
            executor=create_agent("FormatVerifierAgent"),
            depth=1,
        )
        edges.append(Edge(source=out_node, target="verifier"))
        out_node = "verifier"
    if max_nodes:
        nodes["brief"] = Node(
            node_id="brief",
            name="TaskBriefAgent",
            executor=create_agent("TaskBriefAgent"),
            depth=1,
        )
        edges.append(Edge(source="brief", target=in_node))
        in_node = "brief"
    return Graph(in_node=in_node, out_node=out_node, nodes=nodes, edges=edges)


def action_mask(runtime: dict[str, Any], graph: Any, *, max_nodes: int, max_depth: int) -> Any:
    constraints = runtime["ActionConstraints"](max_nodes=max_nodes, max_depth=max_depth)
    return runtime["compute_action_mask"](graph, constraints)


def action_allowed(mask: Any, action: str) -> bool:
    if isinstance(mask, dict):
        for key, value in mask.items():
            if str(getattr(key, "value", key)).upper() == action.upper():
                return bool(value)
        return False
    value = getattr(mask, action, None) or getattr(mask, action.lower(), None)
    return bool(value)


def test_context() -> AgentContext:
    return AgentContext(llm_client=ScriptedLLMClient())


def latest_payload_answer(payload: Mapping[str, Any]) -> str | None:
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return None
    for message in reversed(list(inputs.values())):
        if isinstance(message, Mapping):
            answer = message.get("answer")
            if answer is not None and str(answer).strip():
                return str(answer)
    return None


def to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value


if __name__ == "__main__":
    main()
