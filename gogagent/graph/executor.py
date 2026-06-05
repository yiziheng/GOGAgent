"""Bounded execution for Graph-of-Graphs structures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from gogagent.graph.schema import Graph, GraphMessage, Node
from gogagent.llm.client import AgentContext


@dataclass
class GraphExecutionError(RuntimeError):
    """Raised when a graph cannot be executed under v1 bounded semantics."""

    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message}: {self.details}"


def execute_graph(
    graph: Graph,
    problem: Mapping[str, Any],
    *,
    context: AgentContext | None = None,
    max_steps: int | None = None,
) -> GraphMessage:
    """Execute a graph with bounded DAG-style scheduling.

    Version 1 does not support arbitrary execution cycles. If the scheduler
    cannot reach the output node within the step budget, it raises an explicit
    error instead of silently looping.
    """

    if not graph.nodes:
        return GraphMessage(
            sender=graph.graph_id,
            role="graph",
            content="Empty graph executed with no nodes.",
            metadata={"empty_graph": True},
        )

    _validate_graph(graph)

    in_node = graph.in_node or next(iter(graph.nodes))
    out_node = graph.out_node or _infer_out_node(graph)
    step_budget = max_steps if max_steps is not None else graph.max_steps
    step_budget = max(1, int(step_budget))

    incoming = _incoming_edges(graph)
    outgoing = _outgoing_edges(graph)
    completed: dict[str, GraphMessage] = {}
    ready = deque(
        node_id
        for node_id in graph.nodes
        if not incoming[node_id] or node_id == in_node
    )

    steps = 0
    while ready and steps < step_budget:
        node_id = ready.popleft()
        if node_id in completed:
            continue
        if node_id != in_node and not _predecessors_complete(incoming[node_id], completed):
            continue

        node = graph.nodes[node_id]
        inputs = {
            predecessor: completed[predecessor]
            for predecessor in incoming[node_id]
            if predecessor in completed
        }
        completed[node_id] = execute_node(node, problem, inputs, context=context)
        steps += 1

        if node_id == out_node:
            return completed[node_id]

        for successor in _successors_for_message(node, completed[node_id], outgoing[node_id]):
            if successor not in completed and _predecessors_complete(incoming[successor], completed):
                ready.append(successor)

        if not ready:
            ready.extend(_newly_ready_nodes(graph, incoming, completed))

    if out_node in completed:
        return completed[out_node]

    raise GraphExecutionError(
        "graph execution did not reach out_node",
        {
            "graph_id": graph.graph_id,
            "out_node": out_node,
            "completed_nodes": sorted(completed),
            "max_steps": step_budget,
        },
    )


def execute_node(
    node: Node,
    problem: Mapping[str, Any],
    inputs: Mapping[str, GraphMessage],
    *,
    context: AgentContext | None = None,
) -> GraphMessage:
    """Execute one node whose executor is either an Agent or a nested Graph."""

    normalized_inputs = {
        node_id: GraphMessage.from_dict(message)
        for node_id, message in inputs.items()
    }

    if isinstance(node.executor, Graph):
        sub_problem = dict(problem)
        sub_problem["_parent_node_id"] = node.node_id
        sub_problem["_parent_inputs"] = {
            input_id: message.to_dict()
            for input_id, message in normalized_inputs.items()
        }
        output = execute_graph(
            node.executor,
            sub_problem,
            context=context,
            max_steps=node.executor.max_steps,
        )
        return GraphMessage(
            sender=node.node_id,
            role=node.name,
            content=output.content,
            answer=output.answer,
            confidence=output.confidence,
            notes=dict(output.notes),
            metadata={
                **dict(output.metadata),
                "executor": "Graph",
                "subgraph_id": node.executor.graph_id,
                "subgraph_output": output.to_dict(),
            },
        )

    if not hasattr(node.executor, "execute"):
        raise GraphExecutionError(
            "node executor has no execute method",
            {"node_id": node.node_id, "executor_type": type(node.executor).__name__},
        )

    raw_output = node.executor.execute(problem, normalized_inputs, context=context)
    if isinstance(raw_output, str):
        raise TypeError(
            "Agent.execute must return GraphMessage or JSON dict, not raw str"
        )
    output = GraphMessage.from_dict(raw_output)
    output.sender = node.node_id
    if not output.role:
        output.role = node.name
    if context is not None and output.metadata.get("llm"):
        context.record_llm_call(
            {
                "node_id": node.node_id,
                "node_name": node.name,
                "role": output.role,
                "agent_type": output.metadata.get("agent_type"),
                "llm": output.metadata["llm"],
            }
        )
    return output


def _validate_graph(graph: Graph) -> None:
    if graph.in_node is not None and graph.in_node not in graph.nodes:
        raise GraphExecutionError(
            "graph in_node does not exist",
            {"graph_id": graph.graph_id, "in_node": graph.in_node},
        )
    if graph.out_node is not None and graph.out_node not in graph.nodes:
        raise GraphExecutionError(
            "graph out_node does not exist",
            {"graph_id": graph.graph_id, "out_node": graph.out_node},
        )
    for edge in graph.edges:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            raise GraphExecutionError(
                "graph edge references missing node",
                {
                    "graph_id": graph.graph_id,
                    "source": edge.source,
                    "target": edge.target,
                },
            )


def _infer_out_node(graph: Graph) -> str:
    sources = {edge.source for edge in graph.edges}
    for node_id in reversed(list(graph.nodes)):
        if node_id not in sources:
            return node_id
    return next(reversed(graph.nodes))


def _incoming_edges(graph: Graph) -> dict[str, list[str]]:
    incoming = {node_id: [] for node_id in graph.nodes}
    for edge in graph.edges:
        incoming[edge.target].append(edge.source)
    return incoming


def _outgoing_edges(graph: Graph) -> dict[str, list[str]]:
    outgoing = {node_id: [] for node_id in graph.nodes}
    for edge in graph.edges:
        outgoing[edge.source].append(edge.target)
    return outgoing


def _predecessors_complete(
    predecessors: list[str],
    completed: Mapping[str, GraphMessage],
) -> bool:
    return all(predecessor in completed for predecessor in predecessors)


def _successors_for_message(
    node: Node,
    message: GraphMessage,
    default_successors: list[str],
) -> list[str]:
    route_to_final = (
        bool(message.metadata.get("complete"))
        or bool(message.metadata.get("route_to_final"))
        or node.left_step <= 0
    )
    if route_to_final and node.final_node_id is not None:
        return [node.final_node_id]
    return list(default_successors)


def _newly_ready_nodes(
    graph: Graph,
    incoming: Mapping[str, list[str]],
    completed: Mapping[str, GraphMessage],
) -> list[str]:
    return [
        node_id
        for node_id in graph.nodes
        if node_id not in completed and _predecessors_complete(incoming[node_id], completed)
    ]
