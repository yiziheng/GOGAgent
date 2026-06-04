"""Reusable accuracy-oriented GraphAgent module templates."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from gogagent.core.types import EdgeSpec, NodeSpec


MODULE_INTERNALS: dict[str, tuple[tuple[str, str], tuple[tuple[str, str, str], ...]]] = {
    "SubjectExpertGraph": (
        (
            ("subject_router", "SubjectRouter"),
            ("concept_expert", "ConceptExpert"),
            ("concept_summarizer", "ConceptSummarizer"),
        ),
        (
            ("subject_router", "concept_expert", "subject_profile"),
            ("concept_expert", "concept_summarizer", "concept_notes"),
        ),
    ),
    "OptionEliminationGraph": (
        (
            ("concept_expert", "ConceptExpert"),
            ("option_eliminator", "OptionEliminator"),
            ("distractor_critic", "DistractorCritic"),
            ("choice_verifier", "ChoiceVerifier"),
        ),
        (
            ("concept_expert", "option_eliminator", "concept_notes"),
            ("option_eliminator", "distractor_critic", "remaining_options"),
            ("distractor_critic", "choice_verifier", "distractor_report"),
        ),
    ),
    "SecondOpinionDebateGraph": (
        (
            ("solver_a", "PrimaryOptionSolver"),
            ("solver_b", "IndependentOptionSolver"),
            ("adjudicator", "Adjudicator"),
        ),
        (
            ("solver_a", "adjudicator", "primary_candidate"),
            ("solver_b", "adjudicator", "alternative_candidate"),
        ),
    ),
    "DecomposeSolveVerifyGraph": (
        (
            ("decomposer", "MathDecomposer"),
            ("solver", "MathSolver"),
            ("verifier", "ArithmeticVerifier"),
        ),
        (
            ("decomposer", "solver", "math_plan"),
            ("solver", "verifier", "candidate_answer"),
        ),
    ),
    "AdversarialBestAnswerGraph": (
        (
            ("premise_challenger", "PremiseChallenger"),
            ("option_granularity_judge", "OptionGranularityJudge"),
            ("best_answer_arbitrator", "BestAnswerArbitrator"),
        ),
        (
            ("premise_challenger", "option_granularity_judge", "challenged_premises"),
            ("option_granularity_judge", "best_answer_arbitrator", "option_risk_report"),
        ),
    ),
    "ArithmeticUnitCheckGraph": (
        (
            ("equation_checker", "EquationChecker"),
            ("unit_checker", "UnitChecker"),
            ("numeric_rechecker", "NumericRechecker"),
        ),
        (
            ("equation_checker", "unit_checker", "equation_report"),
            ("unit_checker", "numeric_rechecker", "unit_report"),
        ),
    ),
    "CritiqueReviseGraph": (
        (
            ("critic", "Critic"),
            ("reviser", "Reviser"),
            ("rechecker", "Rechecker"),
        ),
        (
            ("critic", "reviser", "issue_report"),
            ("reviser", "rechecker", "revised_candidate"),
        ),
    ),
    "SpecAnalyzeCodeGraph": (
        (
            ("spec_analyzer", "SpecAnalyzer"),
            ("code_synthesizer", "CodeSynthesizer"),
            ("static_reviewer", "StaticReviewer"),
        ),
        (
            ("spec_analyzer", "code_synthesizer", "spec_summary"),
            ("code_synthesizer", "static_reviewer", "candidate_code"),
        ),
    ),
    "TestDebugRetestGraph": (
        (
            ("test_runner", "TestRunner"),
            ("error_localizer", "ErrorLocalizer"),
            ("debugger", "Debugger"),
            ("retester", "Retester"),
        ),
        (
            ("test_runner", "error_localizer", "execution_report"),
            ("error_localizer", "debugger", "localized_error"),
            ("debugger", "retester", "patched_code"),
        ),
    ),
    "AlternativeImplementationGraph": (
        (
            ("implementation_a", "PrimaryImplementation"),
            ("implementation_b", "AlternativeImplementation"),
            ("adjudicator", "CodeAdjudicator"),
        ),
        (
            ("implementation_a", "adjudicator", "primary_code"),
            ("implementation_b", "adjudicator", "alternative_code"),
        ),
    ),
}


MODULE_TO_ATOMIC_ROLE = {
    "SubjectExpertGraph": "DomainAnalyst",
    "OptionEliminationGraph": "Solver",
    "SecondOpinionDebateGraph": "Adjudicator",
    "DecomposeSolveVerifyGraph": "Solver",
    "AdversarialBestAnswerGraph": "Adjudicator",
    "ArithmeticUnitCheckGraph": "ArithmeticChecker",
    "CritiqueReviseGraph": "Rechecker",
    "SpecAnalyzeCodeGraph": "Solver",
    "TestDebugRetestGraph": "CodeChecker",
    "AlternativeImplementationGraph": "Adjudicator",
}


def build_graphagent(
    *,
    node_id: str,
    module_type: str,
    domain: str,
    profile: str = "",
    model_tier: str = "standard",
    metadata: Mapping[str, object] | None = None,
) -> NodeSpec:
    """Create one top-level GraphAgent node with a depth-1 internal DAG."""

    if module_type not in MODULE_INTERNALS:
        raise ValueError(f"unknown GraphAgent module: {module_type}")
    internal_node_specs, internal_edge_specs = MODULE_INTERNALS[module_type]
    internal_nodes = tuple(
        NodeSpec(
            node_id=f"{node_id}.{internal_id}",
            role=role,
            profile=f"{domain}.{module_type}.{role}",
            node_kind="atomic",
            model_tier=model_tier,
            metadata={"domain": domain, "parent_module": module_type},
        )
        for internal_id, role in internal_node_specs
    )
    internal_edges = tuple(
        EdgeSpec(
            src=f"{node_id}.{src}",
            dst=f"{node_id}.{dst}",
            payload=payload,
            edge_kind="internal",
        )
        for src, dst, payload in internal_edge_specs
    )
    return NodeSpec(
        node_id=node_id,
        role=module_type,
        profile=profile or f"{domain}.{module_type}",
        node_kind="graph",
        module_type=module_type,
        model_tier=model_tier,
        input_ports=("task", "candidate", "feedback"),
        output_ports=("candidate", "issue_report", "decision"),
        internal_nodes=internal_nodes,
        internal_edges=internal_edges,
        metadata={
            "domain": domain,
            "aggregation_rule": "last_internal_output",
            "expected_gain_type": _expected_gain(module_type),
            **dict(metadata or {}),
        },
    )


def downgrade_to_atomic(node: NodeSpec, *, node_id: str | None = None) -> NodeSpec:
    """Collapse a GraphAgent node into a cheaper atomic approximation."""

    if node.node_kind != "graph":
        raise ValueError(f"cannot downgrade non-GraphAgent node {node.node_id}")
    role = MODULE_TO_ATOMIC_ROLE.get(node.module_type or node.role, "Solver")
    return NodeSpec(
        node_id=node_id or node.node_id,
        role=role,
        runner=node.runner,
        profile=f"{node.profile}.downgraded",
        node_kind="atomic",
        model_tier="small" if node.model_tier != "small" else node.model_tier,
        metadata={
            "downgraded_from": node.module_type or node.role,
            "domain": node.metadata.get("domain", ""),
        },
    )


def with_model_tier(node: NodeSpec, model_tier: str) -> NodeSpec:
    return replace(
        node,
        model_tier=model_tier,
        internal_nodes=tuple(
            replace(child, model_tier=model_tier) for child in node.internal_nodes
        ),
    )


def _expected_gain(module_type: str) -> str:
    if module_type in {
        "OptionEliminationGraph",
        "SecondOpinionDebateGraph",
        "AdversarialBestAnswerGraph",
    }:
        return "mmlu_accuracy"
    if module_type in {"ArithmeticUnitCheckGraph", "DecomposeSolveVerifyGraph"}:
        return "gsm8k_accuracy"
    if module_type in {"SpecAnalyzeCodeGraph", "TestDebugRetestGraph"}:
        return "humaneval_pass_at_1"
    return "general_accuracy"
