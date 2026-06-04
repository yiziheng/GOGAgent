"""Persistent Graph-of-Graphs archive for training traces."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from gogagent.core.actions import MacroAction
from gogagent.core.types import (
    EdgeSpec,
    ExperienceRecord,
    GraphSignature,
    NodeSpec,
    OrgGraphSnapshot,
    TransitionEdge,
)


class OrganizationGoG:
    """Store generated DAG snapshots and edit transitions for audit/training."""

    def __init__(self) -> None:
        self.snapshots: dict[str, OrgGraphSnapshot] = {}
        self.signatures: dict[str, GraphSignature] = {}
        self.transitions: list[TransitionEdge] = []
        self.experiences: list[ExperienceRecord] = []

    def add_snapshot(
        self,
        snapshot: OrgGraphSnapshot,
        signature: GraphSignature,
        transition: TransitionEdge | None = None,
    ) -> None:
        self.snapshots[snapshot.graph_id] = snapshot
        self.signatures[snapshot.graph_id] = signature
        if transition is not None:
            self.transitions.append(transition)

    def add_experience(self, experience: ExperienceRecord) -> None:
        self.experiences.append(experience)

    def fork_for_rollout(self) -> OrganizationGoG:
        """Return a mutable episode view while leaving frozen training memory intact."""

        fork = OrganizationGoG()
        fork.snapshots = dict(self.snapshots)
        fork.signatures = dict(self.signatures)
        fork.transitions = list(self.transitions)
        fork.experiences = list(self.experiences)
        return fork

    def save(self, path: str | Path) -> Path:
        """Persist frozen training memory without serializing raw labels."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> OrganizationGoG:
        """Load training memory for read-only rollout seeding."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls()
        memory.snapshots = {
            item["graph_id"]: _snapshot_from_dict(item)
            for item in payload.get("snapshots", ())
        }
        memory.signatures = {
            graph_id: _signature_from_dict(signature)
            for graph_id, signature in payload.get("signatures", {}).items()
        }
        memory.transitions = [
            TransitionEdge(
                src_graph_id=item["src_graph_id"],
                dst_graph_id=item["dst_graph_id"],
                action=MacroAction(item["action"]),
            )
            for item in payload.get("transitions", ())
        ]
        memory.experiences = [
            ExperienceRecord(
                graph_id=item["graph_id"],
                domain=item["domain"],
                task_features=item["task_features"],
                feedback_type=item["feedback_type"],
                cost_bucket=item["cost_bucket"],
                action=MacroAction(item["action"]),
                return_value=float(item["return_value"]),
                success=bool(item["success"]),
            )
            for item in payload.get("experiences", ())
        ]
        return memory

    def to_dict(self) -> dict[str, Any]:
        """Return a replayable, label-free memory checkpoint."""

        return {
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots.values()],
            "signatures": {
                graph_id: signature.to_dict()
                for graph_id, signature in self.signatures.items()
            },
            "transitions": [transition.to_dict() for transition in self.transitions],
            "experiences": [experience.to_dict() for experience in self.experiences],
        }


def _snapshot_from_dict(data: dict[str, Any]) -> OrgGraphSnapshot:
    created_by = data.get("created_by")
    return OrgGraphSnapshot(
        graph_id=data["graph_id"],
        domain=data["domain"],
        step=int(data["step"]),
        nodes=tuple(_node_from_dict(node) for node in data.get("nodes", ())),
        edges=tuple(_edge_from_dict(edge) for edge in data.get("edges", ())),
        parent_graph_id=data.get("parent_graph_id"),
        created_by=MacroAction(created_by) if created_by else None,
        metadata=data.get("metadata", {}),
    )


def _signature_from_dict(data: dict[str, Any]) -> GraphSignature:
    return GraphSignature(
        roles=tuple(data.get("roles", ())),
        node_count=int(data["node_count"]),
        edge_count=int(data["edge_count"]),
        depth=int(data["depth"]),
        payload_modes=tuple(data.get("payload_modes", ())),
        graph_agent_count=int(data.get("graph_agent_count", 0)),
        atomic_agent_count=int(data.get("atomic_agent_count", 0)),
        module_types=tuple(data.get("module_types", ())),
        max_graphagent_internal_nodes=int(data.get("max_graphagent_internal_nodes", 0)),
    )


def _node_from_dict(node: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        node_id=node["node_id"],
        role=node["role"],
        runner=node.get("runner", "openai_compatible"),
        profile=node.get("profile", ""),
        node_kind=node.get("node_kind", node.get("node_type", "atomic")),
        module_type=node.get("module_type", ""),
        model_tier=node.get("model_tier", "standard"),
        input_ports=tuple(node.get("input_ports", ())),
        output_ports=tuple(node.get("output_ports", ())),
        internal_nodes=tuple(_node_from_dict(child) for child in node.get("internal_nodes", ())),
        internal_edges=tuple(_edge_from_dict(edge) for edge in node.get("internal_edges", ())),
        metadata=node.get("metadata", {}),
    )


def _edge_from_dict(edge: dict[str, Any]) -> EdgeSpec:
    return EdgeSpec(
        src=edge["src"],
        dst=edge["dst"],
        payload=edge.get("payload", "default"),
        edge_kind=edge.get("edge_kind", "exec"),
        metadata=edge.get("metadata", {}),
    )
