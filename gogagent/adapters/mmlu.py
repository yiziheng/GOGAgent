"""Label-blind MMLU adapter for typed Organization GoG construction."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from gogagent.adapters.base import DomainAdapter
from gogagent.core.actions import MacroAction
from gogagent.core.graph_ops import default_signature, make_graph_id, topological_order
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


_OPTION_LABELS = ("A", "B", "C", "D")

_SUBJECT_PROFILES = {
    "stem": {
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    },
    "humanities": {
        "business_ethics",
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "world_religions",
    },
    "social_sciences": {
        "econometrics",
        "global_facts",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    },
    "professional": {
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "international_law",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "virology",
    },
}


class MMLUAdapter(DomainAdapter):
    """Compile MMLU reasoning capabilities without reading any gold label."""

    name = "mmlu"

    def base_graph(self, task: Mapping[str, Any]) -> OrgGraphSnapshot:
        profile = subject_profile(_public_subject(task))
        return OrgGraphSnapshot(
            graph_id=make_graph_id(),
            domain=self.name,
            step=0,
            nodes=(
                NodeSpec(
                    node_id="solver",
                    role="Solver",
                    profile=f"{profile}.solver",
                    metadata={"subject_profile": profile},
                ),
            ),
            edges=(),
            metadata={"subject_profile": profile},
        )

    def task_features(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return public, label-blind features only."""

        subject = _public_subject(task)
        options = _public_options(task)
        return {
            "dataset": self.name,
            "subject": subject,
            "subject_profile": subject_profile(subject),
            "question_length": len(_public_question(task)),
            "option_count": len(options),
        }

    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        del feedback
        profile = str(graph.metadata.get("subject_profile", "general"))
        node_ids = {node.node_id for node in graph.nodes}

        if action is MacroAction.STOP:
            return CompiledEdit(added_nodes=(), added_edges=(), metadata={"terminal": True})

        if action is MacroAction.ATTACH_ANALYST:
            _require_absent(node_ids, "domain_analyst")
            return CompiledEdit(
                added_nodes=(
                    NodeSpec(
                        "domain_analyst",
                        "DomainAnalyst",
                        profile=f"{profile}.analyst",
                    ),
                ),
                added_edges=(EdgeSpec("domain_analyst", "solver", "domain_analysis"),),
                invalidated_nodes=_downstream_from(graph, "solver"),
                metadata={"last_capability": "ANALYST"},
            )

        if action is MacroAction.ATTACH_CHECKER:
            _require_absent(node_ids, "option_critic")
            return CompiledEdit(
                added_nodes=(
                    NodeSpec(
                        "option_critic",
                        "OptionCritic",
                        profile=f"{profile}.checker",
                    ),
                ),
                added_edges=(EdgeSpec(_answer_tail(graph), "option_critic", "candidate_answer"),),
                metadata={"last_capability": "CHECKER"},
            )

        if action is MacroAction.ATTACH_REVISER:
            _require_present(node_ids, "option_critic", action)
            _require_absent(node_ids, "resolver", "rechecker")
            return CompiledEdit(
                added_nodes=(
                    NodeSpec("resolver", "Resolver", profile=f"{profile}.reviser"),
                    NodeSpec("rechecker", "Rechecker", profile=f"{profile}.rechecker"),
                ),
                added_edges=(
                    EdgeSpec("option_critic", "resolver", "option_critique"),
                    EdgeSpec("resolver", "rechecker", "revised_answer"),
                ),
                metadata={"last_capability": "REVISER"},
            )

        if action is MacroAction.ATTACH_ALTERNATIVE:
            _require_absent(node_ids, "second_opinion_solver", "adjudicator")
            return CompiledEdit(
                added_nodes=(
                    NodeSpec(
                        "second_opinion_solver",
                        "SecondOpinionSolver",
                        profile=f"{profile}.alternative",
                    ),
                    NodeSpec("adjudicator", "Adjudicator", profile=f"{profile}.adjudicator"),
                ),
                added_edges=(
                    EdgeSpec(_answer_tail(graph), "adjudicator", "primary_answer"),
                    EdgeSpec("second_opinion_solver", "adjudicator", "alternative_answer"),
                ),
                metadata={"last_capability": "ALTERNATIVE"},
            )

        raise ValueError(f"unsupported MMLU macro action: {action.value}")

    def execute(
        self,
        graph: OrgGraphSnapshot,
        task: Mapping[str, Any],
        llm: LLMBackend,
        previous: ExecutionResult | None = None,
    ) -> ExecutionResult:
        """Execute a graph using public question text and options only."""

        question = _public_question(task)
        options = _public_options(task)
        subject = _public_subject(task)
        profile = subject_profile(subject)
        nodes = {node.node_id: node for node in graph.nodes}
        incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for edge in graph.edges:
            incoming[edge.dst].append(edge.src)

        prior_cache = dict(previous.cache) if previous else {}
        cache = dict(prior_cache)
        node_outputs: dict[str, str] = {}
        generated_outputs: list[str] = []
        llm_calls = 0
        for node_id in topological_order(graph):
            node = nodes[node_id]
            context = {source: node_outputs[source] for source in sorted(incoming[node_id])}
            prompt = _node_prompt(node.role, profile, question, options)
            cache_key = _cache_key(node, prompt, context)
            output = prior_cache.get(cache_key)
            if output is None:
                output = llm.generate(node.role, prompt, context)
                cache[cache_key] = output
                generated_outputs.append(output)
                llm_calls += 1
            node_outputs[node_id] = output

        option_scores = _option_scores(question, options, subject, graph, node_outputs)
        predicted_option = max(option_scores, key=option_scores.get)
        sorted_scores = sorted(option_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        base_option = _base_option(question, options, subject)
        checker_present = "option_critic" in nodes
        revised = "rechecker" in nodes
        alternative_present = "second_opinion_solver" in nodes
        checker_disagreement = checker_present and predicted_option != base_option
        second_opinion_disagreement = alternative_present and (
            _salted_option(question, options, subject, "second-opinion") != predicted_option
        )
        disagreement = (
            "high"
            if checker_disagreement and second_opinion_disagreement
            else "medium"
            if checker_disagreement or second_opinion_disagreement
            else "none"
        )
        confidence = "high" if margin >= 0.16 else "medium" if margin >= 0.07 else "low"
        issue_codes = _issue_codes(
            checker_present=checker_present,
            revised=revised,
            margin=margin,
            disagreement=disagreement,
        )
        status = "ready" if checker_present and not issue_codes else "needs_review"
        feedback = VisibleFeedback(
            status=status,
            confidence_bucket=confidence,
            disagreement_level=disagreement,
            issue_codes=issue_codes,
            signals={
                "predicted_option": predicted_option,
                "option_scores": {key: round(value, 6) for key, value in option_scores.items()},
                "option_score_margin": round(margin, 6),
                "contradiction_count": int(checker_disagreement),
                "evidence_coverage": _evidence_coverage(graph),
                "checker_disagreement": checker_disagreement,
                "answer_changed": bool(previous and previous.final_output != predicted_option),
                "subject_profile": profile,
            },
        )
        return ExecutionResult(
            graph_id=graph.graph_id,
            final_output=predicted_option,
            node_outputs=node_outputs,
            visible_feedback=feedback,
            token_cost=sum(max(len(output) // 4, 1) for output in generated_outputs),
            llm_calls=llm_calls,
            cache=cache,
        )

    def signature(self, graph: OrgGraphSnapshot) -> GraphSignature:
        return default_signature(graph)


def subject_profile(subject: str) -> str:
    """Map a public MMLU subject to one coarse, fixed prompt profile."""

    normalized = subject.strip().lower().replace(" ", "_").replace("-", "_")
    for profile, subjects in _SUBJECT_PROFILES.items():
        if normalized in subjects:
            return profile
    return "general"


def _public_question(task: Mapping[str, Any]) -> str:
    return str(task.get("question", task.get("prompt", "")))


def _public_subject(task: Mapping[str, Any]) -> str:
    return str(task.get("subject", task.get("category", "unknown")))


def _public_options(task: Mapping[str, Any]) -> tuple[str, ...]:
    raw_options = task.get("options", task.get("choices", ()))
    if isinstance(raw_options, Mapping):
        options = tuple(str(raw_options.get(label, "")) for label in _OPTION_LABELS)
    elif isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        options = tuple(str(option) for option in raw_options)
    else:
        options = ()
    if len(options) != len(_OPTION_LABELS):
        raise ValueError("MMLU tasks must expose exactly four public answer options")
    return options


def _node_prompt(role: str, profile: str, question: str, options: Sequence[str]) -> str:
    option_text = " ".join(f"{label}. {value}" for label, value in zip(_OPTION_LABELS, options))
    return (
        f"MMLU profile={profile}. Act as {role}. "
        f"Question: {question} Options: {option_text}. "
        "Use only the provided question, options, and predecessor outputs."
    )


def _cache_key(node: NodeSpec, prompt: str, context: Mapping[str, str]) -> str:
    material = repr((node.node_id, node.role, node.profile, prompt, sorted(context.items())))
    return f"mmlu:{node.node_id}:{sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _option_scores(
    question: str,
    options: Sequence[str],
    subject: str,
    graph: OrgGraphSnapshot,
    node_outputs: Mapping[str, str],
) -> dict[str, float]:
    structure = tuple(sorted(node.role for node in graph.nodes))
    evidence = tuple(sorted(node_outputs.items()))
    raw_scores = []
    for label, option in zip(_OPTION_LABELS, options):
        material = repr((question, tuple(options), subject, structure, evidence, label, option))
        raw_scores.append(1 + int(sha256(material.encode("utf-8")).hexdigest()[:8], 16) % 1000)
    total = float(sum(raw_scores))
    return {label: raw / total for label, raw in zip(_OPTION_LABELS, raw_scores)}


def _base_option(question: str, options: Sequence[str], subject: str) -> str:
    return _salted_option(question, options, subject, "base-solver")


def _salted_option(question: str, options: Sequence[str], subject: str, salt: str) -> str:
    material = repr((question, tuple(options), subject, salt))
    index = int(sha256(material.encode("utf-8")).hexdigest()[:8], 16) % len(_OPTION_LABELS)
    return _OPTION_LABELS[index]


def _issue_codes(
    checker_present: bool,
    revised: bool,
    margin: float,
    disagreement: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    if not checker_present:
        issues.append("unchecked_answer")
    if margin < 0.07:
        issues.append("low_option_margin")
    if disagreement != "none" and not revised:
        issues.append("unresolved_option_conflict")
    return tuple(issues)


def _evidence_coverage(graph: OrgGraphSnapshot) -> float:
    roles = {node.role for node in graph.nodes}
    coverage = 0.25
    coverage += 0.2 if "DomainAnalyst" in roles else 0.0
    coverage += 0.25 if "OptionCritic" in roles else 0.0
    coverage += 0.2 if "Rechecker" in roles else 0.0
    coverage += 0.1 if "Adjudicator" in roles else 0.0
    return round(min(coverage, 1.0), 6)


def _answer_tail(graph: OrgGraphSnapshot) -> str:
    outgoing = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        outgoing[edge.src] += 1
    sinks = {node_id for node_id, count in outgoing.items() if count == 0}
    for preferred in ("adjudicator", "rechecker", "resolver", "option_critic", "solver"):
        if preferred in sinks:
            return preferred
    raise ValueError("MMLU graph does not contain an answer-producing sink")


def _downstream_from(graph: OrgGraphSnapshot, source: str) -> tuple[str, ...]:
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.src].append(edge.dst)
    seen = {source}
    pending = [source]
    while pending:
        current = pending.pop()
        for child in adjacency[current]:
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return tuple(sorted(seen))


def _require_absent(node_ids: set[str], *required_absent: str) -> None:
    duplicates = sorted(node_id for node_id in required_absent if node_id in node_ids)
    if duplicates:
        raise ValueError(f"MMLU graph already contains nodes: {duplicates}")


def _require_present(node_ids: set[str], required: str, action: MacroAction) -> None:
    if required not in node_ids:
        raise ValueError(f"{action.value} requires existing node {required}")
