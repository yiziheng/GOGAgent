"""Deterministic legality masks for bounded hierarchical GoG construction."""

from __future__ import annotations

from gogagent.core.actions import MacroAction
from gogagent.core.graph_ops import topological_order
from gogagent.core.types import MacroCandidate, OrgGraphSnapshot, VisibleFeedback


class ConstraintEngine:
    """Keep the learned policy inside a small executable GoG grammar."""

    def __init__(
        self,
        max_steps: int = 4,
        max_nodes: int = 8,
        max_depth: int = 2,
        max_candidates: int = 10,
    ) -> None:
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_candidates = max_candidates

    def validate(self, graph: OrgGraphSnapshot) -> None:
        if len(graph.nodes) > self.max_nodes:
            raise ValueError(f"graph exceeds max_nodes={self.max_nodes}")
        topological_order(graph)
        for node in graph.nodes:
            if node.node_kind == "graph":
                if not node.internal_nodes:
                    raise ValueError(f"GraphAgent {node.node_id} has no internal nodes")
                if any(child.node_kind == "graph" for child in node.internal_nodes):
                    raise ValueError("nested GraphAgent depth > 2 is not supported")
                _validate_internal_dag(node)

    def legal_candidates(
        self,
        graph: OrgGraphSnapshot,
        feedback: VisibleFeedback,
    ) -> tuple[MacroCandidate, ...]:
        """Return label-blind legal module edits; STOP is always available."""

        candidates = [MacroCandidate(MacroAction.STOP, "finish with the current graph")]
        if graph.step >= self.max_steps:
            return tuple(candidates)

        modules = {node.module_type for node in graph.nodes if node.node_kind == "graph"}
        node_ids = {node.node_id for node in graph.nodes}
        atomic_nodes = tuple(node for node in graph.nodes if node.node_kind != "graph")
        graph_nodes = tuple(node for node in graph.nodes if node.node_kind == "graph")

        if atomic_nodes and not graph_nodes:
            candidates.append(
                MacroCandidate(
                    MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT,
                    "upgrade an atomic solver into an accuracy-oriented GraphAgent",
                    {"target": atomic_nodes[0].node_id},
                )
            )

        if graph.domain == "mmlu":
            _append_once(
                candidates,
                modules,
                node_ids,
                "SubjectExpertGraph",
                MacroAction.ADD_SUBJECT_EXPERT_GRAPH,
                prior_score=0.55,
                prior_reason="MMLU subject expertise is usually useful before option choice",
            )
            _append_once(
                candidates,
                modules,
                node_ids,
                "OptionEliminationGraph",
                MacroAction.ADD_OPTION_ELIMINATION_GRAPH,
                prior_score=0.8,
                prior_reason="MMLU multiple-choice accuracy benefits from explicit elimination",
            )
            _append_once(
                candidates,
                modules,
                node_ids,
                "DecomposeSolveVerifyGraph",
                MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH,
                prior_score=0.35,
                prior_reason="use structured solve-verify on reasoning-heavy questions",
            )
            if graph_nodes:
                _append_once(
                    candidates,
                    modules,
                    node_ids,
                    "AdversarialBestAnswerGraph",
                    MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH,
                    prior_score=0.2,
                    prior_reason=(
                        "label-blind risk-aware arbitration can challenge "
                        "high-consensus but under-tested answers"
                    ),
                )
            if _uncertain(feedback):
                _append_once(
                    candidates,
                    modules,
                    node_ids,
                    "SecondOpinionDebateGraph",
                    MacroAction.ADD_SECOND_OPINION_DEBATE_GRAPH,
                    prior_score=0.3,
                    prior_reason="debate helps when current evidence is uncertain",
                )
            if feedback.issue_codes:
                _append_once(
                    candidates,
                    modules,
                    node_ids,
                    "CritiqueReviseGraph",
                    MacroAction.ADD_CRITIQUE_REVISE_GRAPH,
                    prior_score=0.65,
                    prior_reason="critique-revise is useful after visible issues",
                )
        elif graph.domain == "gsm8k":
            _append_once(candidates, modules, node_ids, "DecomposeSolveVerifyGraph", MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH)
            _append_once(candidates, modules, node_ids, "ArithmeticUnitCheckGraph", MacroAction.ADD_ARITHMETIC_UNIT_CHECK_GRAPH)
            if feedback.issue_codes or feedback.status not in {"ready", "passed"}:
                _append_once(candidates, modules, node_ids, "CritiqueReviseGraph", MacroAction.ADD_CRITIQUE_REVISE_GRAPH)
        elif graph.domain == "humaneval":
            _append_once(candidates, modules, node_ids, "SpecAnalyzeCodeGraph", MacroAction.ADD_SPEC_ANALYZE_CODE_GRAPH)
            if feedback.issue_codes or feedback.status not in {"ready", "passed"}:
                _append_once(candidates, modules, node_ids, "TestDebugRetestGraph", MacroAction.ADD_TEST_DEBUG_RETEST_GRAPH)
            if _uncertain(feedback):
                _append_once(
                    candidates,
                    modules,
                    node_ids,
                    "AlternativeImplementationGraph",
                    MacroAction.ADD_ALTERNATIVE_IMPLEMENTATION_GRAPH,
                )

        if graph_nodes:
            target = graph_nodes[-1].node_id
            candidates.append(
                MacroCandidate(
                    MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC,
                    "downgrade the latest GraphAgent when complexity is too costly",
                    {"target": target},
                )
            )
            if len(graph.nodes) > 2:
                candidates.append(
                    MacroCandidate(
                        MacroAction.PRUNE_GRAPHAGENT_MODULE,
                        "remove an optional GraphAgent module",
                        {"target": target},
                    )
                )

        tier_candidates = [
            node for node in graph.nodes if node.model_tier not in {"large", "small"}
        ]
        if tier_candidates and _uncertain(feedback):
            candidates.append(
                MacroCandidate(
                    MacroAction.UPGRADE_NODE_MODEL,
                    "upgrade a high-value node when uncertainty remains",
                    {"target": tier_candidates[-1].node_id, "model_tier": "large"},
                )
            )
        large_nodes = [node for node in graph.nodes if node.model_tier == "large"]
        if large_nodes and feedback.status in {"ready", "passed"}:
            candidates.append(
                MacroCandidate(
                    MacroAction.DOWNGRADE_NODE_MODEL,
                    "downgrade large model usage after clean feedback",
                    {"target": large_nodes[-1].node_id, "model_tier": "standard"},
                )
            )
        return tuple(candidates[: self.max_candidates])

    def action_mask(
        self,
        graph: OrgGraphSnapshot,
        feedback: VisibleFeedback,
    ) -> dict[MacroAction, bool]:
        legal = {candidate.action for candidate in self.legal_candidates(graph, feedback)}
        return {action: action in legal for action in MacroAction}


def _capability(role: str) -> str:
    lowered = role.lower()
    if any(word in lowered for word in ("analyst", "planner", "decomposer")):
        return "ANALYST"
    if any(word in lowered for word in ("checker", "critic", "inspector", "tester")):
        return "CHECKER"
    if any(word in lowered for word in ("reviser", "resolver", "debugger", "fixer")):
        return "REVISER"
    if any(word in lowered for word in ("alternative", "independent", "secondopinion")):
        return "ALTERNATIVE"
    return "SOLVER"


def _append_once(
    candidates: list[MacroCandidate],
    existing_modules: set[str],
    existing_node_ids: set[str],
    module_type: str,
    action: MacroAction,
    *,
    prior_score: float = 0.0,
    prior_reason: str = "",
) -> None:
    if module_type not in existing_modules and _module_node_id(module_type) not in existing_node_ids:
        candidates.append(
            MacroCandidate(
                action,
                f"add {module_type} for accuracy-oriented reasoning",
                {
                    "module_type": module_type,
                    "template_id": module_type,
                    "prior_score": float(prior_score),
                    "prior_reason": prior_reason,
                },
            )
        )


def _module_node_id(module_type: str) -> str:
    stem = []
    for index, character in enumerate(module_type):
        if character.isupper() and index > 0:
            stem.append("_")
        stem.append(character.lower())
    return "".join(stem)


def _uncertain(feedback: VisibleFeedback) -> bool:
    return (
        feedback.confidence_bucket == "low"
        or feedback.disagreement_level != "none"
        or feedback.status not in {"ready", "passed"}
    )


def _validate_internal_dag(node: object) -> None:
    internal_nodes = getattr(node, "internal_nodes")
    node_ids = {child.node_id for child in internal_nodes}
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: [] for node_id in node_ids}
    for edge in getattr(node, "internal_edges"):
        if edge.src not in node_ids or edge.dst not in node_ids:
            raise ValueError(f"GraphAgent internal edge references missing node: {edge}")
        outgoing[edge.src].append(edge.dst)
        indegree[edge.dst] += 1
    pending = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        for child in outgoing[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    if visited != len(node_ids):
        raise ValueError(f"GraphAgent {getattr(node, 'node_id')} internal graph contains a cycle")
