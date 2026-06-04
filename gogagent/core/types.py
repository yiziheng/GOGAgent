"""Serializable types used across the GOGAgent runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from gogagent.core.actions import MacroAction


JsonMap = dict[str, Any]


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    role: str
    runner: str = "openai_compatible"
    profile: str = ""
    node_kind: str = "atomic"
    module_type: str = ""
    model_tier: str = "standard"
    input_ports: tuple[str, ...] = ()
    output_ports: tuple[str, ...] = ()
    internal_nodes: tuple["NodeSpec", ...] = ()
    internal_edges: tuple["EdgeSpec", ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "runner": self.runner,
            "profile": self.profile,
            "node_kind": self.node_kind,
            "module_type": self.module_type,
            "model_tier": self.model_tier,
            "input_ports": list(self.input_ports),
            "output_ports": list(self.output_ports),
            "internal_nodes": [node.to_dict() for node in self.internal_nodes],
            "internal_edges": [edge.to_dict() for edge in self.internal_edges],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    payload: str = "default"
    edge_kind: str = "exec"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "src": self.src,
            "dst": self.dst,
            "payload": self.payload,
            "edge_kind": self.edge_kind,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphSignature:
    roles: tuple[str, ...]
    node_count: int
    edge_count: int
    depth: int
    payload_modes: tuple[str, ...]
    graph_agent_count: int = 0
    atomic_agent_count: int = 0
    module_types: tuple[str, ...] = ()
    max_graphagent_internal_nodes: int = 0

    def to_dict(self) -> JsonMap:
        data = asdict(self)
        data["roles"] = list(self.roles)
        data["payload_modes"] = list(self.payload_modes)
        data["module_types"] = list(self.module_types)
        return data


@dataclass(frozen=True)
class OrgGraphSnapshot:
    """An immutable hierarchical executable GoG for one task episode."""

    graph_id: str
    domain: str
    step: int
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    parent_graph_id: str | None = None
    created_by: MacroAction | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "graph_id": self.graph_id,
            "domain": self.domain,
            "step": self.step,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "parent_graph_id": self.parent_graph_id,
            "created_by": self.created_by.value if self.created_by else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VisibleFeedback:
    """Label-blind execution signals visible to policy and supervisor."""

    status: str = "unknown"
    confidence_bucket: str = "medium"
    disagreement_level: str = "none"
    issue_codes: tuple[str, ...] = ()
    signals: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        data = asdict(self)
        data["issue_codes"] = list(self.issue_codes)
        return data


@dataclass(frozen=True)
class SupervisorFeedback:
    """A label-blind control-plane summary."""

    status: str
    confidence_bucket: str
    disagreement_level: str
    unresolved_issue_codes: tuple[str, ...]
    budget_risk: str
    stop_advice: str

    def to_dict(self) -> JsonMap:
        data = asdict(self)
        data["unresolved_issue_codes"] = list(self.unresolved_issue_codes)
        return data


@dataclass(frozen=True)
class ExecutionResult:
    graph_id: str
    final_output: str
    node_outputs: Mapping[str, str]
    visible_feedback: VisibleFeedback
    token_cost: int
    llm_calls: int
    cache: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "graph_id": self.graph_id,
            "final_output": self.final_output,
            "node_outputs": dict(self.node_outputs),
            "visible_feedback": self.visible_feedback.to_dict(),
            "token_cost": self.token_cost,
            "llm_calls": self.llm_calls,
            "cache": dict(self.cache),
        }


@dataclass(frozen=True)
class CompiledEdit:
    added_nodes: tuple[NodeSpec, ...] = ()
    added_edges: tuple[EdgeSpec, ...] = ()
    removed_nodes: tuple[str, ...] = ()
    removed_edges: tuple[EdgeSpec, ...] = ()
    updated_nodes: tuple[NodeSpec, ...] = ()
    invalidated_nodes: tuple[str, ...] = ()
    reusable_cache_keys: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroCandidate:
    action: MacroAction
    reason: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class TransitionEdge:
    src_graph_id: str
    dst_graph_id: str
    action: MacroAction

    def to_dict(self) -> JsonMap:
        return {
            "src_graph_id": self.src_graph_id,
            "dst_graph_id": self.dst_graph_id,
            "action": self.action.value,
        }


@dataclass(frozen=True)
class ExperienceRecord:
    graph_id: str
    domain: str
    task_features: Mapping[str, Any]
    feedback_type: str
    cost_bucket: str
    action: MacroAction
    return_value: float
    success: bool

    def to_dict(self) -> JsonMap:
        data = asdict(self)
        data["action"] = self.action.value
        return data


@dataclass(frozen=True)
class PolicyDecision:
    action: MacroAction
    scores: Mapping[str, float]
    candidates: Sequence[MacroCandidate]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "action": self.action.value,
            "scores": dict(self.scores),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }
