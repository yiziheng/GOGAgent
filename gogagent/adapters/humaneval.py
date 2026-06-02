"""HumanEval adapter for label-blind Python code-generation rollouts."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from hashlib import sha256
import json
from typing import Any, Mapping

from gogagent.adapters.base import DomainAdapter
from gogagent.core.actions import MacroAction
from gogagent.core.types import (
    CompiledEdit,
    EdgeSpec,
    ExecutionResult,
    GraphSignature,
    NodeSpec,
    OrgGraphSnapshot,
    VisibleFeedback,
)
from gogagent.llm.base import LLMBackend


_ROLE_TO_NODE_ID = {
    "Solver": "solver",
    "CodePlanner": "code_planner",
    "CodeChecker": "code_checker",
    "CodeReviser": "code_reviser",
    "Rechecker": "rechecker",
    "IndependentCodeSolver": "independent_code_solver",
    "Adjudicator": "adjudicator",
}


class HumanEvalAdapter(DomainAdapter):
    """Compile shared macro actions into small HumanEval execution DAGs."""

    name = "humaneval"

    def base_graph(self, task: Mapping[str, Any]) -> OrgGraphSnapshot:
        task_id = str(task.get("task_id", "unknown"))
        return OrgGraphSnapshot(
            graph_id=f"humaneval-{_slug(task_id)}-g000",
            domain=self.name,
            step=0,
            nodes=(_node("Solver", profile="python_solution"),),
            edges=(),
            metadata={"task_id": task_id, "adapter": self.name},
        )

    def task_features(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Expose public prompt features only; tests and gold code stay isolated."""

        prompt = _public_prompt(task)
        return {
            "task_id": str(task.get("task_id", "unknown")),
            "entry_point": str(task.get("entry_point", "")),
            "language": "python",
            "prompt_length": len(prompt),
            "line_count": prompt.count("\n") + 1,
        }

    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        del feedback  # Macro expansion is deterministic and label-blind.
        roles = {node.role for node in graph.nodes}

        if action is MacroAction.STOP:
            return CompiledEdit(added_nodes=(), added_edges=())

        if action is MacroAction.ATTACH_ANALYST:
            _ensure_absent(roles, "CodePlanner")
            invalidated = _downstream_nodes(graph, "solver") | {"solver"}
            return _edit(
                graph,
                nodes=(_node("CodePlanner", profile="python_plan"),),
                edges=(EdgeSpec("code_planner", "solver", "implementation_plan"),),
                invalidated=invalidated,
            )

        if action is MacroAction.ATTACH_CHECKER:
            _ensure_absent(roles, "CodeChecker")
            source = _preferred_code_node(graph)
            return _edit(
                graph,
                nodes=(_node("CodeChecker", profile="static_code_review"),),
                edges=(EdgeSpec(source, "code_checker", "candidate_code"),),
            )

        if action is MacroAction.ATTACH_REVISER:
            _ensure_present(roles, "CodeChecker")
            _ensure_absent(roles, "CodeReviser", "Rechecker")
            return _edit(
                graph,
                nodes=(
                    _node("CodeReviser", profile="review_guided_revision"),
                    _node("Rechecker", profile="static_code_recheck"),
                ),
                edges=(
                    EdgeSpec("code_checker", "code_reviser", "review_report"),
                    EdgeSpec("code_reviser", "rechecker", "revised_code"),
                ),
            )

        if action is MacroAction.ATTACH_ALTERNATIVE:
            _ensure_absent(roles, "IndependentCodeSolver", "Adjudicator")
            source = _preferred_code_node(graph)
            return _edit(
                graph,
                nodes=(
                    _node("IndependentCodeSolver", profile="independent_python_solution"),
                    _node("Adjudicator", profile="candidate_adjudication"),
                ),
                edges=(
                    EdgeSpec(source, "adjudicator", "primary_candidate"),
                    EdgeSpec(
                        "independent_code_solver",
                        "adjudicator",
                        "alternative_candidate",
                    ),
                ),
            )

        raise ValueError(f"Unsupported HumanEval macro action: {action.value}")

    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        llm: LLMBackend,
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        """Execute the DAG without inspecting tests, gold code, or oracle scores."""

        prompt = _public_prompt(task)
        outputs: dict[str, str] = {}
        cache = dict(previous.cache) if previous else {}
        token_cost = 0
        llm_calls = 0

        for node in _topological_nodes(graph):
            context = {
                edge.src: outputs[edge.src]
                for edge in graph.edges
                if edge.dst == node.node_id
            }
            role_prompt = _role_prompt(node.role, prompt)
            cache_key = _cache_key(node, role_prompt, context)
            if cache_key in cache:
                output = cache[cache_key]
            else:
                output = llm.generate(role=node.role, prompt=role_prompt, context=context)
                cache[cache_key] = output
                token_cost += _token_estimate(role_prompt, output, context)
                llm_calls += 1
            outputs[node.node_id] = output

        roles = {node.role for node in graph.nodes}
        final_node = _final_output_node(roles)
        feedback = _visible_feedback(roles)
        return ExecutionResult(
            graph_id=graph.graph_id,
            final_output=outputs[final_node],
            node_outputs=outputs,
            visible_feedback=feedback,
            token_cost=token_cost,
            llm_calls=llm_calls,
            cache=cache,
        )

    def signature(self, graph: OrgGraphSnapshot) -> GraphSignature:
        return GraphSignature(
            roles=tuple(sorted(node.role for node in graph.nodes)),
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            depth=_graph_depth(graph),
            payload_modes=tuple(sorted({edge.payload for edge in graph.edges})),
        )


def _node(role: str, profile: str) -> NodeSpec:
    return NodeSpec(
        node_id=_ROLE_TO_NODE_ID[role],
        role=role,
        profile=profile,
        metadata={"capability": "python_code_generation"},
    )


def _edit(
    graph: OrgGraphSnapshot,
    *,
    nodes: tuple[NodeSpec, ...],
    edges: tuple[EdgeSpec, ...],
    invalidated: set[str] | None = None,
) -> CompiledEdit:
    invalidated_nodes = tuple(sorted(invalidated or set()))
    reusable = tuple(
        sorted(
            node.node_id
            for node in graph.nodes
            if node.node_id not in invalidated_nodes
        )
    )
    return CompiledEdit(
        added_nodes=nodes,
        added_edges=edges,
        invalidated_nodes=invalidated_nodes,
        reusable_cache_keys=reusable,
    )


def _public_prompt(task: Mapping[str, Any]) -> str:
    return str(task.get("prompt", task.get("question", "")))


def _cache_key(node: NodeSpec, prompt: str, context: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"node": node.to_dict(), "prompt": prompt, "context": dict(context)},
        ensure_ascii=True,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _token_estimate(prompt: str, output: str, context: Mapping[str, str]) -> int:
    text = " ".join((prompt, output, *context.values()))
    return max(1, len(text.split()))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return slug or "unknown"


def _ensure_absent(roles: set[str], *required_absent: str) -> None:
    duplicates = sorted(set(required_absent) & roles)
    if duplicates:
        raise ValueError(f"HumanEval graph already contains: {duplicates}")


def _ensure_present(roles: set[str], *required_present: str) -> None:
    missing = sorted(set(required_present) - roles)
    if missing:
        raise ValueError(f"HumanEval graph is missing prerequisite roles: {missing}")


def _preferred_code_node(graph: OrgGraphSnapshot) -> str:
    roles = {node.role for node in graph.nodes}
    if "CodeReviser" in roles:
        return _ROLE_TO_NODE_ID["CodeReviser"]
    return _ROLE_TO_NODE_ID["Solver"]


def _final_output_node(roles: set[str]) -> str:
    if "Adjudicator" in roles:
        return _ROLE_TO_NODE_ID["Adjudicator"]
    if "CodeReviser" in roles:
        return _ROLE_TO_NODE_ID["CodeReviser"]
    return _ROLE_TO_NODE_ID["Solver"]


def _downstream_nodes(graph: OrgGraphSnapshot, node_id: str) -> set[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.src].append(edge.dst)

    downstream: set[str] = set()
    queue = deque(adjacency[node_id])
    while queue:
        current = queue.popleft()
        if current in downstream:
            continue
        downstream.add(current)
        queue.extend(adjacency[current])
    return downstream


def _topological_nodes(graph: OrgGraphSnapshot) -> tuple[NodeSpec, ...]:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    indegree = {node_id: 0 for node_id in nodes_by_id}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges:
        if edge.src not in nodes_by_id or edge.dst not in nodes_by_id:
            raise ValueError(f"Dangling HumanEval edge: {edge.src} -> {edge.dst}")
        adjacency[edge.src].append(edge.dst)
        indegree[edge.dst] += 1

    ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    ordered: list[NodeSpec] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(nodes_by_id[node_id])
        for destination in sorted(adjacency[node_id]):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)

    if len(ordered) != len(nodes_by_id):
        raise ValueError("HumanEval execution graph must be acyclic")
    return tuple(ordered)


def _graph_depth(graph: OrgGraphSnapshot) -> int:
    depth_by_node: dict[str, int] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.dst].append(edge.src)

    for node in _topological_nodes(graph):
        parents = incoming[node.node_id]
        depth_by_node[node.node_id] = 1 + max(
            (depth_by_node[parent] for parent in parents),
            default=0,
        )
    return max(depth_by_node.values(), default=0)


def _visible_feedback(roles: set[str]) -> VisibleFeedback:
    if "Adjudicator" in roles:
        return VisibleFeedback(
            status="adjudicated",
            confidence_bucket="high",
            disagreement_level="low",
            signals={"alternative_attached": True, "label_blind": True},
        )
    if "Rechecker" in roles:
        return VisibleFeedback(
            status="rechecked",
            confidence_bucket="high",
            signals={"revision_attached": True, "label_blind": True},
        )
    if "CodeChecker" in roles:
        return VisibleFeedback(
            status="reviewed",
            confidence_bucket="medium",
            issue_codes=("revision_not_attempted",),
            signals={"checker_attached": True, "label_blind": True},
        )
    return VisibleFeedback(
        status="unchecked",
        confidence_bucket="low",
        issue_codes=("verification_missing",),
        signals={"checker_attached": False, "label_blind": True},
    )


def _role_prompt(role: str, task_prompt: str) -> str:
    instructions = {
        "Solver": "Write a concise Python implementation for the HumanEval task.",
        "CodePlanner": "Outline a concise implementation plan and edge cases.",
        "CodeChecker": "Review the candidate implementation for likely defects.",
        "CodeReviser": "Return an improved Python implementation using the review.",
        "Rechecker": "Recheck the revised implementation for likely defects.",
        "IndependentCodeSolver": "Solve the Python task independently.",
        "Adjudicator": "Choose or synthesize the strongest Python implementation.",
    }
    return f"{instructions[role]}\n\nTask:\n{task_prompt}"
