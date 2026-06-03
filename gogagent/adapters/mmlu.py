"""Label-blind MMLU adapter for typed Organization GoG construction."""

from __future__ import annotations

import re
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
_ANSWER_ROLES = {
    "Solver",
    "Resolver",
    "Rechecker",
    "SecondOpinionSolver",
    "Adjudicator",
}
_REVIEW_ROLES = {"OptionCritic", "Rechecker"}

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
        llm_calls = 0
        token_cost = 0
        for node_id in topological_order(graph):
            node = nodes[node_id]
            context = {source: node_outputs[source] for source in sorted(incoming[node_id])}
            prompt = _node_prompt(node.role, profile, question, options)
            cache_key = _cache_key(node, prompt, context)
            output = prior_cache.get(cache_key)
            if output is None:
                response = llm.generate(node.role, prompt, context)
                output = response.text
                cache[cache_key] = output
                llm_calls += 1
                token_cost += response.total_tokens
            node_outputs[node_id] = output

        parsed_options = {
            node_id: _parse_final_option(node_outputs[node_id])
            for node_id, node in nodes.items()
            if node.role in _ANSWER_ROLES
        }
        failed_option_nodes = tuple(
            sorted(node_id for node_id, option in parsed_options.items() if option is None)
        )
        answer_source, predicted_option = _select_answer(graph, parsed_options)
        final_output = predicted_option or ""
        checker_present = "option_critic" in nodes
        revised = "rechecker" in nodes
        alternative_present = "second_opinion_solver" in nodes
        solver_option = parsed_options.get("solver")
        checker_disagreement = (
            checker_present
            and solver_option is not None
            and predicted_option is not None
            and solver_option != predicted_option
        )
        second_opinion_option = parsed_options.get("second_opinion_solver")
        second_opinion_disagreement = (
            alternative_present
            and second_opinion_option is not None
            and predicted_option is not None
            and second_opinion_option != predicted_option
        )
        disagreement = (
            "high"
            if checker_disagreement and second_opinion_disagreement
            else "medium"
            if checker_disagreement or second_opinion_disagreement
            else "none"
        )
        option_parse_failed = predicted_option is None
        confidence = (
            "low"
            if option_parse_failed
            else "medium"
            if disagreement != "none" or not checker_present
            else "high"
        )
        issue_codes = _issue_codes(
            checker_present=checker_present,
            revised=revised,
            disagreement=disagreement,
            option_parse_failed=option_parse_failed,
        )
        status = "ready" if checker_present and not issue_codes else "needs_review"
        feedback = VisibleFeedback(
            status=status,
            confidence_bucket=confidence,
            disagreement_level=disagreement,
            issue_codes=issue_codes,
            signals={
                "predicted_option": predicted_option,
                "answer_source": answer_source,
                "parsed_options": parsed_options,
                "option_parse_failed_nodes": failed_option_nodes,
                "contradiction_count": int(checker_disagreement),
                "evidence_coverage": _evidence_coverage(graph),
                "checker_disagreement": checker_disagreement,
                "second_opinion_disagreement": second_opinion_disagreement,
                "answer_changed": bool(previous and previous.final_output != final_output),
                "subject_profile": profile,
                "label_blind": True,
            },
        )
        return ExecutionResult(
            graph_id=graph.graph_id,
            final_output=final_output,
            node_outputs=node_outputs,
            visible_feedback=feedback,
            token_cost=token_cost,
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
    prompt = (
        f"MMLU profile={profile}. Act as {role}. "
        f"Question: {question} Options: {option_text}. "
        "Use only the provided question, options, and predecessor outputs."
    )
    if role in _REVIEW_ROLES:
        prompt += " Report a structured review line: ISSUES: <none|comma-separated issue codes>."
    if role in _ANSWER_ROLES:
        prompt += " End with exactly one line: FINAL: <A|B|C|D>."
    else:
        prompt += " Provide analysis only. Do not output a FINAL line."
    return prompt


def _cache_key(node: NodeSpec, prompt: str, context: Mapping[str, str]) -> str:
    material = repr((node.node_id, node.role, node.profile, prompt, sorted(context.items())))
    return f"mmlu:{node.node_id}:{sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _parse_final_option(output: str) -> str | None:
    options = re.findall(r"(?im)^\s*FINAL\s*:\s*([A-D])\s*$", output)
    if len(options) == 1:
        return options[0].upper()
    return None


def _issue_codes(
    checker_present: bool,
    revised: bool,
    disagreement: str,
    option_parse_failed: bool,
) -> tuple[str, ...]:
    issues: list[str] = []
    if option_parse_failed:
        issues.append("option_parse_failed")
    if not checker_present:
        issues.append("unchecked_answer")
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
    node_ids = {node.node_id for node in graph.nodes}
    for preferred in ("adjudicator", "rechecker", "resolver", "solver", "second_opinion_solver"):
        if preferred in node_ids:
            return preferred
    raise ValueError("MMLU graph does not contain an answer-producing node")


def _select_answer(
    graph: OrgGraphSnapshot,
    parsed_options: Mapping[str, str | None],
) -> tuple[str, str | None]:
    """Prefer the current answer tail, but fall back to a valid upstream answer.

    A late reviewer can be truncated before emitting ``FINAL`` even when its
    predecessor already produced a valid option. Treating every older parse
    miss as unresolved makes the graph keep growing, so policy feedback should
    focus on whether the final usable answer chain has a single option.
    """

    preferred_tail = _answer_tail(graph)
    if parsed_options.get(preferred_tail) is not None:
        return preferred_tail, parsed_options[preferred_tail]
    for candidate in ("adjudicator", "rechecker", "resolver", "second_opinion_solver", "solver"):
        option = parsed_options.get(candidate)
        if option is not None:
            return candidate, option
    return preferred_tail, None


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
