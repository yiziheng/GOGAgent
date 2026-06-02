"""Persistent outer Graph-of-Graphs state and neighbor statistics."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from gogagent.core.actions import MacroAction
from gogagent.core.types import (
    EdgeSpec,
    ExperienceRecord,
    GraphSignature,
    NodeSpec,
    OrgGraphSnapshot,
    SimilarityEdge,
    TransitionEdge,
)
from gogagent.gog.similarity import signature_similarity


class OrganizationGoG:
    """Store inner DAG snapshots as outer nodes and make neighbors policy-visible."""

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k
        self.snapshots: dict[str, OrgGraphSnapshot] = {}
        self.signatures: dict[str, GraphSignature] = {}
        self.transitions: list[TransitionEdge] = []
        self.similarities: list[SimilarityEdge] = []
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
        self._refresh_similarity_edges()

    def add_experience(self, experience: ExperienceRecord) -> None:
        self.experiences.append(experience)

    def fork_for_rollout(self) -> OrganizationGoG:
        """Return a mutable episode view while leaving frozen training memory intact."""

        fork = OrganizationGoG(self.top_k)
        fork.snapshots = dict(self.snapshots)
        fork.signatures = dict(self.signatures)
        fork.transitions = list(self.transitions)
        fork.similarities = list(self.similarities)
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
        memory = cls(top_k=int(payload.get("top_k", 3)))
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
        memory.similarities = [
            SimilarityEdge(
                src_graph_id=item["src_graph_id"],
                dst_graph_id=item["dst_graph_id"],
                similarity=float(item["similarity"]),
            )
            for item in payload.get("similarities", ())
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
            "top_k": self.top_k,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots.values()],
            "signatures": {
                graph_id: signature.to_dict()
                for graph_id, signature in self.signatures.items()
            },
            "transitions": [transition.to_dict() for transition in self.transitions],
            "similarities": [edge.to_dict() for edge in self.similarities],
            "experiences": [experience.to_dict() for experience in self.experiences],
        }

    def similar_neighbors(self, graph_id: str) -> tuple[SimilarityEdge, ...]:
        edges = [
            edge
            for edge in self.similarities
            if edge.src_graph_id == graph_id or edge.dst_graph_id == graph_id
        ]
        return tuple(sorted(edges, key=lambda edge: edge.similarity, reverse=True)[: self.top_k])

    def neighbor_stats(self, graph_id: str, action: MacroAction) -> dict[str, float]:
        neighbor_ids = {graph_id}
        for edge in self.similar_neighbors(graph_id):
            neighbor_ids.add(edge.src_graph_id)
            neighbor_ids.add(edge.dst_graph_id)
        matching = [
            item for item in self.experiences if item.graph_id in neighbor_ids and item.action is action
        ]
        if not matching:
            return {"mean_return": 0.0, "success_rate": 0.0, "count": 0.0}
        return {
            "mean_return": mean(item.return_value for item in matching),
            "success_rate": mean(float(item.success) for item in matching),
            "count": float(len(matching)),
        }

    def _refresh_similarity_edges(self) -> None:
        graph_ids = sorted(self.snapshots)
        edges: list[SimilarityEdge] = []
        for index, src in enumerate(graph_ids):
            scored = []
            for dst in graph_ids[index + 1 :]:
                score = signature_similarity(self.signatures[src], self.signatures[dst])
                if score > 0:
                    scored.append(SimilarityEdge(src, dst, score))
            edges.extend(sorted(scored, key=lambda edge: edge.similarity, reverse=True)[: self.top_k])
        self.similarities = edges


def _snapshot_from_dict(data: dict[str, Any]) -> OrgGraphSnapshot:
    created_by = data.get("created_by")
    return OrgGraphSnapshot(
        graph_id=data["graph_id"],
        domain=data["domain"],
        step=int(data["step"]),
        nodes=tuple(
            NodeSpec(
                node_id=node["node_id"],
                role=node["role"],
                runner=node.get("runner", "openai_compatible"),
                profile=node.get("profile", ""),
                metadata=node.get("metadata", {}),
            )
            for node in data.get("nodes", ())
        ),
        edges=tuple(
            EdgeSpec(src=edge["src"], dst=edge["dst"], payload=edge.get("payload", "default"))
            for edge in data.get("edges", ())
        ),
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
    )
