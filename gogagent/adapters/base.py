"""Domain adapter interface for the shared Organization GoG runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

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
from gogagent.core.graphagent_library import (
    build_graphagent,
    downgrade_to_atomic,
    with_model_tier,
)
from gogagent.llm.base import LLMBackend


class DomainAdapter(ABC):
    """Compile shared actions into domain-specific executable DAG edits."""

    name: str

    @abstractmethod
    def base_graph(self, task: Mapping[str, Any]) -> OrgGraphSnapshot:
        """Return the minimum executable graph for a task."""

    @abstractmethod
    def task_features(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return label-blind features available during inference."""

    @abstractmethod
    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        """Compile a macro action into deterministic nodes and typed edges."""

    @abstractmethod
    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        llm: LLMBackend,
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        """Execute or simulate a candidate graph without consulting gold labels."""

    @abstractmethod
    def signature(self, graph: OrgGraphSnapshot) -> GraphSignature:
        """Build a lightweight structure-only graph signature."""


def compile_common_module_edit(
    graph: OrgGraphSnapshot,
    action: MacroAction,
    *,
    domain: str,
    module_type: str | None = None,
    fallback_source: str | None = None,
) -> CompiledEdit | None:
    """Compile domain-independent GraphAgent upgrade/downgrade edits."""

    if action is MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT:
        target = _first_atomic(graph)
        inferred_module = module_type or _default_module_for_domain(domain)
        graph_node = build_graphagent(
            node_id=target.node_id,
            module_type=inferred_module,
            domain=domain,
            profile=f"{domain}.{inferred_module}.expanded",
            metadata={"expanded_from": target.role},
        )
        return CompiledEdit(
            updated_nodes=(graph_node,),
            invalidated_nodes=_downstream_from(graph, target.node_id),
            metadata={
                "edit_type": "node_upgrade",
                "module_type": inferred_module,
                "target": target.node_id,
            },
        )

    if action is MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC:
        target = _last_graphagent(graph)
        return CompiledEdit(
            updated_nodes=(downgrade_to_atomic(target),),
            invalidated_nodes=_downstream_from(graph, target.node_id),
            metadata={
                "edit_type": "node_downgrade",
                "module_type": target.module_type,
                "target": target.node_id,
            },
        )

    if action is MacroAction.PRUNE_GRAPHAGENT_MODULE:
        target = _last_graphagent(graph)
        if len(graph.nodes) <= 1:
            raise ValueError("cannot prune the only node in a graph")
        return CompiledEdit(
            added_nodes=(),
            added_edges=(),
            removed_nodes=(target.node_id,),
            invalidated_nodes=_downstream_from(graph, target.node_id),
            metadata={
                "edit_type": "module_prune",
                "module_type": target.module_type,
                "target": target.node_id,
            },
        )

    if action is MacroAction.UPGRADE_NODE_MODEL:
        target = _last_non_large_node(graph)
        return CompiledEdit(
            updated_nodes=(with_model_tier(target, "large"),),
            invalidated_nodes=_downstream_from(graph, target.node_id),
            metadata={
                "edit_type": "model_upgrade",
                "target": target.node_id,
                "model_tier": "large",
            },
        )

    if action is MacroAction.DOWNGRADE_NODE_MODEL:
        target = _last_large_node(graph)
        return CompiledEdit(
            updated_nodes=(with_model_tier(target, "standard"),),
            invalidated_nodes=_downstream_from(graph, target.node_id),
            metadata={
                "edit_type": "model_downgrade",
                "target": target.node_id,
                "model_tier": "standard",
            },
        )

    if module_type is None:
        return None

    node_id = _module_node_id(module_type)
    if any(node.node_id == node_id for node in graph.nodes):
        raise ValueError(f"graph already contains module node {node_id}")
    source = fallback_source or _answer_tail(graph)
    graph_node = build_graphagent(
        node_id=node_id,
        module_type=module_type,
        domain=domain,
        metadata={"added_by": action.value},
    )
    return CompiledEdit(
        added_nodes=(graph_node,),
        added_edges=(EdgeSpec(source, node_id, "candidate_or_context"),),
        invalidated_nodes=(node_id,),
        metadata={
            "edit_type": "module_add",
            "module_type": module_type,
            "target": node_id,
        },
    )


def _module_node_id(module_type: str) -> str:
    stem = []
    for index, character in enumerate(module_type):
        if character.isupper() and index > 0:
            stem.append("_")
        stem.append(character.lower())
    return "".join(stem)


def _first_atomic(graph: OrgGraphSnapshot) -> NodeSpec:
    for node in graph.nodes:
        if node.node_kind != "graph":
            return node
    raise ValueError("graph has no atomic node to upgrade")


def _last_graphagent(graph: OrgGraphSnapshot) -> NodeSpec:
    for node in reversed(graph.nodes):
        if node.node_kind == "graph":
            return node
    raise ValueError("graph has no GraphAgent node")


def _last_non_large_node(graph: OrgGraphSnapshot) -> NodeSpec:
    for node in reversed(graph.nodes):
        if node.model_tier != "large":
            return node
    raise ValueError("graph has no node that can be upgraded")


def _last_large_node(graph: OrgGraphSnapshot) -> NodeSpec:
    for node in reversed(graph.nodes):
        if node.model_tier == "large":
            return node
    raise ValueError("graph has no large node that can be downgraded")


def _default_module_for_domain(domain: str) -> str:
    if domain == "mmlu":
        return "OptionEliminationGraph"
    if domain == "gsm8k":
        return "DecomposeSolveVerifyGraph"
    if domain == "humaneval":
        return "SpecAnalyzeCodeGraph"
    return "CritiqueReviseGraph"


def _answer_tail(graph: OrgGraphSnapshot) -> str:
    outgoing = {edge.src for edge in graph.edges}
    candidates = [node.node_id for node in graph.nodes if node.node_id not in outgoing]
    return candidates[-1] if candidates else graph.nodes[-1].node_id


def _downstream_from(graph: OrgGraphSnapshot, source: str) -> tuple[str, ...]:
    adjacency = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency.setdefault(edge.src, []).append(edge.dst)
    seen = {source}
    pending = [source]
    while pending:
        current = pending.pop()
        for child in adjacency.get(current, ()):
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return tuple(sorted(seen))
