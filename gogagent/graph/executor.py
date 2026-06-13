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
        inputs = _execution_context_inputs(completed)
        completed[node_id] = execute_node(node, problem, inputs, context=context)
        steps += 1

        if node_id == out_node:
            return _maybe_execute_plan_repeat(
                graph=graph,
                problem=problem,
                incoming=incoming,
                outgoing=outgoing,
                completed=completed,
                out_node=out_node,
                context=context,
                step_budget=step_budget,
            )

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
    if context is not None:
        llm_calls = output.metadata.get("llm_calls")
        if isinstance(llm_calls, list):
            for call in llm_calls:
                context.record_llm_call(
                    {
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "role": output.role,
                        "agent_type": output.metadata.get("agent_type"),
                        "llm": call,
                    }
                )
        elif output.metadata.get("llm"):
            context.record_llm_call(
                {
                    "node_id": node.node_id,
                    "node_name": node.name,
                    "role": output.role,
                    "agent_type": output.metadata.get("agent_type"),
                    "llm": output.metadata["llm"],
                }
            )
        llm_audit = output.metadata.get("llm_audit")
        if isinstance(llm_audit, list):
            for audit_event in llm_audit:
                if not isinstance(audit_event, Mapping):
                    continue
                context.record_llm_audit(
                    {
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "role": output.role,
                        "agent_type": output.metadata.get("agent_type"),
                        **dict(audit_event),
                    }
                )
    return output


def _maybe_execute_plan_repeat(
    *,
    graph: Graph,
    problem: Mapping[str, Any],
    incoming: Mapping[str, list[str]],
    outgoing: Mapping[str, list[str]],
    completed: Mapping[str, GraphMessage],
    out_node: str,
    context: AgentContext | None,
    step_budget: int,
) -> GraphMessage:
    """Repeat the path after PlanSketchAgent once when the plan asks for it."""

    output = completed[out_node]
    repeat_request = _find_plan_repeat_request(completed)
    if repeat_request is None:
        return output

    plan_node_id, plan_message, repeat_count = repeat_request
    if repeat_count <= 1:
        return output

    downstream = _reachable_nodes(outgoing, plan_node_id)
    if out_node not in downstream:
        return output

    working_completed = dict(completed)
    for iteration in range(2, repeat_count + 1):
        augmented_plan = _append_step_output_to_context(
            plan_message,
            step_number=iteration - 1,
            output=output,
        )
        repeated = _execute_downstream_once(
            graph=graph,
            problem=problem,
            incoming=incoming,
            outgoing=outgoing,
            downstream=downstream,
            plan_node_id=plan_node_id,
            plan_message=augmented_plan,
            completed=working_completed,
            out_node=out_node,
            context=context,
            step_budget=step_budget,
        )
        if out_node not in repeated:
            raise GraphExecutionError(
                "plan repeat did not reach out_node",
                {
                    "graph_id": graph.graph_id,
                    "plan_node": plan_node_id,
                    "out_node": out_node,
                    "repeated_nodes": sorted(repeated),
                    "iteration": iteration,
                },
            )
        output = repeated[out_node]
        output.metadata["plan_repeat"] = {
            "plan_node": plan_node_id,
            "requested_count": repeat_count,
            "iteration": iteration,
            "step_context": f"Step {iteration - 1}",
        }
        working_completed.update(repeated)

    return output


def _find_plan_repeat_request(
    completed: Mapping[str, GraphMessage],
) -> tuple[str, GraphMessage, int] | None:
    for node_id, message in completed.items():
        if message.role != "plan_sketch":
            continue
        repeat_count = int(message.metadata.get("repeat_count") or 1)
        repeat_count = 2 if repeat_count == 2 else 1
        return node_id, message, repeat_count
    return None


def _execute_downstream_once(
    *,
    graph: Graph,
    problem: Mapping[str, Any],
    incoming: Mapping[str, list[str]],
    outgoing: Mapping[str, list[str]],
    downstream: list[str],
    plan_node_id: str,
    plan_message: GraphMessage,
    completed: Mapping[str, GraphMessage],
    out_node: str,
    context: AgentContext | None,
    step_budget: int,
) -> dict[str, GraphMessage]:
    downstream_set = set(downstream)
    repeated: dict[str, GraphMessage] = {}
    ready = deque(
        node_id
        for node_id in downstream
        if _repeat_inputs_for_node(
            node_id,
            incoming,
            downstream_set=downstream_set,
            plan_node_id=plan_node_id,
            plan_message=plan_message,
            completed=completed,
            repeated=repeated,
        )
        is not None
    )

    steps = 0
    while ready and steps < step_budget:
        node_id = ready.popleft()
        if node_id in repeated:
            continue
        direct_inputs = _repeat_inputs_for_node(
            node_id,
            incoming,
            downstream_set=downstream_set,
            plan_node_id=plan_node_id,
            plan_message=plan_message,
            completed=completed,
            repeated=repeated,
        )
        if direct_inputs is None:
            continue

        transcript = dict(completed)
        transcript[plan_node_id] = plan_message
        transcript.update(repeated)
        inputs = _execution_context_inputs(transcript)
        repeated[node_id] = execute_node(graph.nodes[node_id], problem, inputs, context=context)
        steps += 1
        if node_id == out_node:
            break

        for successor in outgoing[node_id]:
            if successor in downstream_set and successor not in repeated:
                ready.append(successor)

    return repeated


def _repeat_inputs_for_node(
    node_id: str,
    incoming: Mapping[str, list[str]],
    *,
    downstream_set: set[str],
    plan_node_id: str,
    plan_message: GraphMessage,
    completed: Mapping[str, GraphMessage],
    repeated: Mapping[str, GraphMessage],
) -> dict[str, GraphMessage] | None:
    inputs: dict[str, GraphMessage] = {}
    for predecessor in incoming[node_id]:
        if predecessor == plan_node_id:
            inputs[predecessor] = plan_message
        elif predecessor in downstream_set:
            if predecessor not in repeated:
                return None
            inputs[predecessor] = repeated[predecessor]
        elif predecessor in completed:
            inputs[predecessor] = completed[predecessor]
        else:
            return None
    return inputs


def _reachable_nodes(
    outgoing: Mapping[str, list[str]],
    start_node: str,
) -> list[str]:
    reachable: list[str] = []
    seen: set[str] = set()
    queue = deque(outgoing[start_node])
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        reachable.append(node_id)
        queue.extend(outgoing[node_id])
    return reachable


def _append_step_output_to_context(
    message: GraphMessage,
    *,
    step_number: int,
    output: GraphMessage,
) -> GraphMessage:
    content = message.content.rstrip()
    step_output = _message_output_text(output)
    if content:
        content = f"{content}\n\nStep {step_number}: {step_output}"
    else:
        content = f"Step {step_number}: {step_output}"
    return GraphMessage(
        sender=message.sender,
        role=message.role,
        content=content,
        answer=message.answer,
        confidence=message.confidence,
        notes=dict(message.notes),
        metadata={
            **dict(message.metadata),
            "repeat_context_step": step_number,
            "repeat_context_output": step_output,
        },
    )


def _message_output_text(message: GraphMessage) -> str:
    if message.answer is not None:
        return str(message.answer).strip()
    return " ".join(message.content.split())


def _execution_context_inputs(
    completed: Mapping[str, GraphMessage],
) -> dict[str, GraphMessage]:
    """Return the compact transcript visible to the next node.

    Topology still controls readiness and routing. This helper only changes the
    prompt context: each node sees the messages produced before it in execution
    order, instead of only its direct graph predecessors.
    """

    return {
        node_id: GraphMessage.from_dict(message)
        for node_id, message in completed.items()
    }


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
