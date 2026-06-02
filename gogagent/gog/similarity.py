"""Lightweight structure similarity for outer GoG edges."""

from __future__ import annotations

from gogagent.core.types import GraphSignature


def signature_similarity(left: GraphSignature, right: GraphSignature) -> float:
    """Return a deterministic [0, 1] similarity score without a learned GNN."""

    left_roles, right_roles = set(left.roles), set(right.roles)
    role_union = left_roles | right_roles
    role_score = len(left_roles & right_roles) / len(role_union) if role_union else 1.0
    payload_union = set(left.payload_modes) | set(right.payload_modes)
    payload_score = (
        len(set(left.payload_modes) & set(right.payload_modes)) / len(payload_union)
        if payload_union
        else 1.0
    )
    count_score = 1.0 / (1.0 + abs(left.node_count - right.node_count))
    depth_score = 1.0 / (1.0 + abs(left.depth - right.depth))
    return round(0.55 * role_score + 0.15 * payload_score + 0.15 * count_score + 0.15 * depth_score, 6)
