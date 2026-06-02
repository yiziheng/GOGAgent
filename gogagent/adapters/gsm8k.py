"""GSM8K adapter for label-blind math graph construction."""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import json
import re
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


_ROLE_TO_ID = {
    "Solver": "solver",
    "MathDecomposer": "math_decomposer",
    "ArithmeticChecker": "arithmetic_checker",
    "MathReviser": "math_reviser",
    "Rechecker": "rechecker",
    "IndependentMathSolver": "independent_math_solver",
    "Adjudicator": "adjudicator",
}


class GSM8KAdapter(DomainAdapter):
    """Compile shared macro actions into a small executable math DAG."""

    name = "gsm8k"

    def base_graph(self, task: Mapping[str, Any]) -> OrgGraphSnapshot:
        task_key = str(task.get("task_id") or task.get("id") or task.get("question", ""))
        suffix = sha256(task_key.encode("utf-8")).hexdigest()[:10]
        return OrgGraphSnapshot(
            graph_id=f"gsm8k-{suffix}-g000",
            domain=self.name,
            step=0,
            nodes=(self._node("Solver"),),
            edges=(),
            metadata={"adapter": self.name, "scale": "S0"},
        )

    def task_features(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        question = self._question(task)
        return {
            "domain": self.name,
            "question_length": len(question),
            "word_count": len(question.split()),
            "number_count": len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", question)),
            "has_currency": "$" in question,
        }

    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        del feedback  # Compilation is deterministic and never consults labels.
        roles = {node.role for node in graph.nodes}

        if action is MacroAction.STOP:
            return CompiledEdit(added_nodes=(), added_edges=(), metadata={"stop": True})

        if action is MacroAction.ATTACH_ANALYST:
            self._require_absent(roles, "MathDecomposer", action)
            return CompiledEdit(
                added_nodes=(self._node("MathDecomposer"),),
                added_edges=(EdgeSpec("math_decomposer", "solver", "decomposition"),),
                invalidated_nodes=tuple(node.node_id for node in graph.nodes),
                metadata={"scale": "S1", "capability": "analysis"},
            )

        if action is MacroAction.ATTACH_CHECKER:
            self._require_absent(roles, "ArithmeticChecker", action)
            source = self._answer_source(graph)
            return CompiledEdit(
                added_nodes=(self._node("ArithmeticChecker"),),
                added_edges=(EdgeSpec(source, "arithmetic_checker", "candidate_answer"),),
                reusable_cache_keys=(source,),
                metadata={"scale": "S1", "capability": "verification"},
            )

        if action is MacroAction.ATTACH_REVISER:
            self._require_present(roles, "ArithmeticChecker", action)
            self._require_absent(roles, "MathReviser", action)
            return CompiledEdit(
                added_nodes=(self._node("MathReviser"), self._node("Rechecker")),
                added_edges=(
                    EdgeSpec("arithmetic_checker", "math_reviser", "arithmetic_review"),
                    EdgeSpec("math_reviser", "rechecker", "revised_answer"),
                ),
                reusable_cache_keys=("arithmetic_checker",),
                metadata={"scale": "S2", "capability": "revision"},
            )

        if action is MacroAction.ATTACH_ALTERNATIVE:
            self._require_absent(roles, "IndependentMathSolver", action)
            source = self._answer_source(graph)
            return CompiledEdit(
                added_nodes=(
                    self._node("IndependentMathSolver"),
                    self._node("Adjudicator"),
                ),
                added_edges=(
                    EdgeSpec(source, "adjudicator", "primary_answer"),
                    EdgeSpec("independent_math_solver", "adjudicator", "alternative_answer"),
                ),
                reusable_cache_keys=(source,),
                metadata={"scale": "S2", "capability": "alternative"},
            )

        raise ValueError(f"Unsupported GSM8K macro action: {action}")

    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        llm: LLMBackend,
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        """Execute the DAG without reading task labels, even if the task contains them."""

        question = self._question(task)
        ordered_nodes = self._topological_nodes(graph)
        incoming: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            incoming[edge.dst].append(edge.src)

        cache = dict(previous.cache) if previous else {}
        outputs: dict[str, str] = {}
        llm_calls = 0
        token_cost = 0

        for node in ordered_nodes:
            context = {src: outputs[src] for src in sorted(incoming[node.node_id])}
            prompt = self._prompt(node.role, question)
            cache_key = self._cache_key(node, prompt, context)
            if cache_key in cache:
                output = cache[cache_key]
            else:
                output = llm.generate(node.role, prompt, context)
                cache[cache_key] = output
                llm_calls += 1
                token_cost += self._token_estimate(prompt, output, context)
            outputs[node.node_id] = output

        answer_source = self._answer_source(graph)
        final_output = outputs[answer_source]
        feedback = self._visible_feedback(graph, answer_source, llm_calls)
        return ExecutionResult(
            graph_id=graph.graph_id,
            final_output=final_output,
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
            depth=self._depth(graph),
            payload_modes=tuple(sorted({edge.payload for edge in graph.edges})),
        )

    @staticmethod
    def _node(role: str) -> NodeSpec:
        return NodeSpec(
            node_id=_ROLE_TO_ID[role],
            role=role,
            profile=f"gsm8k_{_ROLE_TO_ID[role]}",
            metadata={"capability": role},
        )

    @staticmethod
    def _question(task: Mapping[str, Any]) -> str:
        # GSM8K stores the gold rationale in ``answer``. Never read it here.
        question = task.get("question") or task.get("prompt") or ""
        return str(question)

    @staticmethod
    def _prompt(role: str, question: str) -> str:
        instructions = {
            "Solver": "Solve the word problem. End with a single numeric answer.",
            "MathDecomposer": "Break the word problem into explicit arithmetic steps.",
            "ArithmeticChecker": "Check the candidate arithmetic and report any issue.",
            "MathReviser": "Revise the candidate answer using the arithmetic review.",
            "Rechecker": "Recheck the revised answer and end with a single numeric answer.",
            "IndependentMathSolver": "Solve independently and end with a single numeric answer.",
            "Adjudicator": "Compare both candidates and end with the best single numeric answer.",
        }
        return f"{instructions[role]}\nQuestion: {question}"

    @staticmethod
    def _cache_key(node: NodeSpec, prompt: str, context: Mapping[str, str]) -> str:
        payload = json.dumps(
            {"node": node.to_dict(), "prompt": prompt, "context": dict(context)},
            ensure_ascii=True,
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _token_estimate(prompt: str, output: str, context: Mapping[str, str]) -> int:
        text = " ".join((prompt, output, *context.values()))
        return max(1, len(text.split()))

    @staticmethod
    def _answer_source(graph: OrgGraphSnapshot) -> str:
        answer_roles = {
            "Adjudicator",
            "Rechecker",
            "MathReviser",
            "Solver",
            "IndependentMathSolver",
        }
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = {node.node_id: 0 for node in graph.nodes}
        nodes = {node.node_id: node for node in graph.nodes}
        for edge in graph.edges:
            outgoing[edge.src].append(edge.dst)
            incoming[edge.dst].append(edge.src)
            indegree[edge.dst] += 1

        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        depths = {node_id: 1 for node_id in nodes}
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for dst in outgoing[node_id]:
                depths[dst] = max(depths[dst], depths[node_id] + 1)
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        if visited != len(nodes):
            raise ValueError("GSM8K executable graph must be a DAG")

        priority = {
            "Solver": 0,
            "IndependentMathSolver": 1,
            "MathReviser": 2,
            "Rechecker": 3,
            "Adjudicator": 4,
        }
        candidates = [node for node in graph.nodes if node.role in answer_roles]
        if not candidates:
            raise ValueError("GSM8K graph has no answer-producing node")
        selected = max(candidates, key=lambda node: (depths[node.node_id], priority[node.role]))
        return selected.node_id

    @staticmethod
    def _require_present(roles: set[str], role: str, action: MacroAction) -> None:
        if role not in roles:
            raise ValueError(f"{action.value} requires {role}")

    @staticmethod
    def _require_absent(roles: set[str], role: str, action: MacroAction) -> None:
        if role in roles:
            raise ValueError(f"{action.value} cannot add duplicate {role}")

    @classmethod
    def _topological_nodes(cls, graph: OrgGraphSnapshot) -> list[NodeSpec]:
        nodes = {node.node_id: node for node in graph.nodes}
        indegree = {node_id: 0 for node_id in nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.src not in nodes or edge.dst not in nodes:
                raise ValueError(f"Unknown GSM8K edge endpoint: {edge.src} -> {edge.dst}")
            outgoing[edge.src].append(edge.dst)
            indegree[edge.dst] += 1

        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[NodeSpec] = []
        while queue:
            node_id = queue.popleft()
            ordered.append(nodes[node_id])
            for dst in sorted(outgoing[node_id]):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        if len(ordered) != len(nodes):
            raise ValueError("GSM8K executable graph must be a DAG")
        return ordered

    @staticmethod
    def _depth(graph: OrgGraphSnapshot) -> int:
        if not graph.nodes:
            return 0
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = {node.node_id: 0 for node in graph.nodes}
        for edge in graph.edges:
            incoming[edge.dst].append(edge.src)
            outgoing[edge.src].append(edge.dst)
            indegree[edge.dst] += 1

        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        depths = {node_id: 1 for node_id in indegree}
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for dst in outgoing[node_id]:
                depths[dst] = max(depths[dst], depths[node_id] + 1)
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        if visited != len(indegree):
            raise ValueError("GSM8K executable graph must be a DAG")
        return max(depths.values())

    @staticmethod
    def _visible_feedback(
        graph: OrgGraphSnapshot,
        answer_source: str,
        llm_calls: int,
    ) -> VisibleFeedback:
        roles = {node.role for node in graph.nodes}
        has_checker = "ArithmeticChecker" in roles
        has_revision = "Rechecker" in roles
        has_alternative = "Adjudicator" in roles

        if has_revision:
            status = "revised"
            confidence = "high"
            issues: tuple[str, ...] = ()
        elif has_checker:
            status = "checked"
            confidence = "medium"
            issues = ("revision_not_attempted",)
        else:
            status = "draft"
            confidence = "low"
            issues = ("arithmetic_unchecked",)

        disagreement = "medium" if has_alternative else "none"
        return VisibleFeedback(
            status=status,
            confidence_bucket=confidence,
            disagreement_level=disagreement,
            issue_codes=issues,
            signals={
                "has_analysis": "MathDecomposer" in roles,
                "has_checker": has_checker,
                "has_revision": has_revision,
                "has_alternative": has_alternative,
                "answer_source": answer_source,
                "node_count": len(graph.nodes),
                "executed_llm_calls": llm_calls,
            },
        )
