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


_PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_CHECKER_ISSUE_RE = re.compile(
    r"\b(?:bug|bugs|defect|defects|error|errors|fail|failed|failure|incorrect|issue|issues|mistake|mistakes|wrong)\b",
    re.IGNORECASE,
)
_NEGATED_CHECKER_ISSUE_RE = re.compile(
    r"\b(?:no|not|without|zero)\s+"
    r"(?:apparent\s+|detected\s+|obvious\s+|remaining\s+)?"
    r"(?:bugs?|defects?|errors?|failures?|incorrect|issues?|mistakes?|wrong)\b",
    re.IGNORECASE,
)
_CHECKER_ROLES = {"CodeChecker", "Rechecker"}
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
                response = llm.generate(role=node.role, prompt=role_prompt, context=context)
                output = response.text
                cache[cache_key] = output
                token_cost += response.total_tokens
                llm_calls += 1
            outputs[node.node_id] = output

        roles = {node.role for node in graph.nodes}
        final_node = _final_output_node(roles)
        feedback = _visible_feedback(graph, final_node, outputs, llm_calls)
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


def _visible_feedback(
    graph: OrgGraphSnapshot,
    answer_source: str,
    outputs: Mapping[str, str],
    llm_calls: int,
) -> VisibleFeedback:
    roles = {node.role for node in graph.nodes}
    candidate = _extract_python_code(outputs[answer_source])
    code_extractable = bool(candidate)
    python_compile_ok = _python_compiles(candidate)
    checker_issue_nodes = tuple(
        node.node_id
        for node in graph.nodes
        if node.role in _CHECKER_ROLES and _checker_reports_issue(outputs[node.node_id])
    )
    has_checker = "CodeChecker" in roles

    issues: list[str] = []
    if not code_extractable:
        issues.append("python_code_missing")
    elif not python_compile_ok:
        issues.append("python_compile_failed")
    if checker_issue_nodes:
        issues.append("checker_reported_issue")
    if not has_checker:
        issues.append("static_review_missing")

    confidence = (
        "low"
        if not code_extractable or not python_compile_ok
        else "medium"
        if checker_issue_nodes or not has_checker
        else "high"
    )
    return VisibleFeedback(
        status="needs_review" if issues else "ready",
        confidence_bucket=confidence,
        disagreement_level="medium" if checker_issue_nodes else "none",
        issue_codes=tuple(issues),
        signals={
            "has_plan": "CodePlanner" in roles,
            "has_checker": has_checker,
            "has_revision": "CodeReviser" in roles,
            "has_rechecker": "Rechecker" in roles,
            "has_alternative": "Adjudicator" in roles,
            "answer_source": answer_source,
            "code_extractable": code_extractable,
            "python_compile_ok": python_compile_ok,
            "checker_reported_issue": bool(checker_issue_nodes),
            "checker_issue_nodes": checker_issue_nodes,
            "node_count": len(graph.nodes),
            "executed_llm_calls": llm_calls,
            "label_blind": True,
        },
    )


def _extract_python_code(output: str) -> str:
    matches = _PYTHON_FENCE.findall(output)
    if matches:
        return matches[0].strip()
    return output.strip()


def _python_compiles(candidate: str) -> bool:
    if not candidate:
        return False
    try:
        compile(candidate, "<humaneval-public-feedback>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def _checker_reports_issue(text: str) -> bool:
    observable_text = _NEGATED_CHECKER_ISSUE_RE.sub("", text)
    return _CHECKER_ISSUE_RE.search(observable_text) is not None


def _role_prompt(role: str, task_prompt: str) -> str:
    instructions = {
        "Solver": "Write a concise Python implementation for the HumanEval task. Return only the complete implementation fenced as ```python ... ```.",
        "CodePlanner": "Outline a concise implementation plan and edge cases.",
        "CodeChecker": "Review the candidate implementation for likely defects.",
        "CodeReviser": "Return an improved complete Python implementation using the review. Return only the implementation fenced as ```python ... ```.",
        "Rechecker": "Recheck the revised implementation for likely defects.",
        "IndependentCodeSolver": "Solve the Python task independently. Return only the complete implementation fenced as ```python ... ```.",
        "Adjudicator": "Choose or synthesize the strongest complete Python implementation. Return only the implementation fenced as ```python ... ```.",
    }
    return f"{instructions[role]}\n\nTask:\n{task_prompt}"
