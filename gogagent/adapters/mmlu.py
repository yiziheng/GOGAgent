"""Label-blind MMLU adapter for typed Organization GoG construction."""

from __future__ import annotations

import re
import math
from hashlib import sha256
from typing import Any, Mapping, Sequence

from gogagent.adapters.base import DomainAdapter, compile_common_module_edit
from gogagent.adapters.mmlu_subjects import subject_profile
from gogagent.adapters.mmlu_task_encoder import encode_mmlu_task
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
    "OptionEliminationGraph",
    "SecondOpinionDebateGraph",
    "DecomposeSolveVerifyGraph",
    "AdversarialBestAnswerGraph",
    "CritiqueReviseGraph",
    "Solver",
    "Resolver",
    "Rechecker",
    "SecondOpinionSolver",
    "Adjudicator",
}
_REVIEW_ROLES = {"OptionCritic", "Rechecker"}
_MODULE_ACTIONS = {
    MacroAction.ADD_SUBJECT_EXPERT_GRAPH: "SubjectExpertGraph",
    MacroAction.ADD_OPTION_ELIMINATION_GRAPH: "OptionEliminationGraph",
    MacroAction.ADD_SECOND_OPINION_DEBATE_GRAPH: "SecondOpinionDebateGraph",
    MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH: "DecomposeSolveVerifyGraph",
    MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH: "AdversarialBestAnswerGraph",
    MacroAction.ADD_CRITIQUE_REVISE_GRAPH: "CritiqueReviseGraph",
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

        encoded = encode_mmlu_task(task)
        return {
            "dataset": self.name,
            **encoded,
        }

    def compile(
        self,
        graph: OrgGraphSnapshot,
        action: MacroAction,
        feedback: VisibleFeedback,
    ) -> CompiledEdit:
        del feedback
        common = compile_common_module_edit(
            graph,
            action,
            domain=self.name,
            module_type=_MODULE_ACTIONS.get(action),
            fallback_source=_answer_tail(graph),
        )
        if common is not None:
            return common
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
        checker_present = "option_critic" in nodes or "option_elimination_graph" in nodes
        revised = "rechecker" in nodes or "critique_revise_graph" in nodes
        alternative_present = (
            "second_opinion_solver" in nodes or "second_opinion_debate_graph" in nodes
        )
        solver_option = parsed_options.get("solver")
        checker_disagreement = (
            checker_present
            and solver_option is not None
            and predicted_option is not None
            and solver_option != predicted_option
        )
        second_opinion_option = parsed_options.get(
            "second_opinion_debate_graph",
            parsed_options.get("second_opinion_solver"),
        )
        second_opinion_disagreement = (
            alternative_present
            and second_opinion_option is not None
            and predicted_option is not None
            and second_opinion_option != predicted_option
        )
        adversarial_option = parsed_options.get("adversarial_best_answer_graph")
        adversarial_disagreement = (
            adversarial_option is not None
            and predicted_option is not None
            and adversarial_option != predicted_option
        )
        disagreement = (
            "high"
            if sum(
                bool(value)
                for value in (
                    checker_disagreement,
                    second_opinion_disagreement,
                    adversarial_disagreement,
                )
            )
            >= 2
            else "medium"
            if checker_disagreement or second_opinion_disagreement or adversarial_disagreement
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
                "parse_failed_count": len(failed_option_nodes),
                "final_tail_parse_failed": parsed_options.get(_answer_tail(graph)) is None,
                "vote_distribution": _vote_distribution(parsed_options),
                "answer_vote_entropy": _answer_vote_entropy(parsed_options),
                "majority_margin": _majority_margin(parsed_options),
                "contradiction_count": sum(
                    bool(value)
                    for value in (
                        checker_disagreement,
                        second_opinion_disagreement,
                        adversarial_disagreement,
                    )
                ),
                "answer_module_count": _answer_module_count(parsed_options),
                "consensus_fraction": _consensus_fraction(parsed_options),
                "rationale_similarity": _rationale_similarity(graph, node_outputs),
                "evidence_coverage": _evidence_coverage(graph),
                "checker_disagreement": checker_disagreement,
                "second_opinion_disagreement": second_opinion_disagreement,
                "adversarial_disagreement": adversarial_disagreement,
                "adversarial_checked": "adversarial_best_answer_graph" in nodes,
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
    if role == "SubjectExpertGraph":
        prompt += (
            " Run the internal subject-router/concept-expert/summarizer graph. "
            "Return concise concept notes and option risk hints."
        )
    elif role == "OptionEliminationGraph":
        prompt += (
            " Run concept expert, option elimination, distractor critique, and choice verification. "
            "Eliminate wrong options before choosing."
        )
    elif role == "SecondOpinionDebateGraph":
        prompt += (
            " Run two independent option solvers and adjudicate disagreements."
        )
    elif role == "DecomposeSolveVerifyGraph":
        prompt += (
            " Decompose the question, solve step by step, verify the selected option, "
            "and prefer the option that survives verification."
        )
    elif role == "AdversarialBestAnswerGraph":
        prompt += (
            " Do not simply agree with the predecessor answer. Run a premise challenger, "
            "an option granularity judge, and a best-answer arbitrator. Specifically test "
            "whether the current answer confuses cause with effect, a broad concept with a "
            "narrow example, a symptom with a label, or a merely plausible option with the "
            "best exam answer. Choose the option that survives this adversarial check."
        )
    elif role == "CritiqueReviseGraph":
        prompt += (
            " Critique the current candidate, revise if needed, and recheck the final choice."
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
    coverage += 0.2 if "DomainAnalyst" in roles or "SubjectExpertGraph" in roles else 0.0
    coverage += 0.25 if "OptionCritic" in roles or "OptionEliminationGraph" in roles else 0.0
    coverage += 0.15 if "DecomposeSolveVerifyGraph" in roles else 0.0
    coverage += 0.2 if "AdversarialBestAnswerGraph" in roles else 0.0
    coverage += 0.2 if "Rechecker" in roles or "CritiqueReviseGraph" in roles else 0.0
    coverage += 0.1 if "Adjudicator" in roles or "SecondOpinionDebateGraph" in roles else 0.0
    return round(min(coverage, 1.0), 6)


def _answer_tail(graph: OrgGraphSnapshot) -> str:
    node_ids = {node.node_id for node in graph.nodes}
    for preferred in (
        "critique_revise_graph",
        "adversarial_best_answer_graph",
        "second_opinion_debate_graph",
        "decompose_solve_verify_graph",
        "option_elimination_graph",
        "adjudicator",
        "rechecker",
        "resolver",
        "solver",
        "second_opinion_solver",
    ):
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
    for candidate in (
        "critique_revise_graph",
        "adversarial_best_answer_graph",
        "second_opinion_debate_graph",
        "decompose_solve_verify_graph",
        "option_elimination_graph",
        "adjudicator",
        "rechecker",
        "resolver",
        "second_opinion_solver",
        "solver",
    ):
        option = parsed_options.get(candidate)
        if option is not None:
            return candidate, option
    return preferred_tail, None


def _answer_module_count(parsed_options: Mapping[str, str | None]) -> int:
    return sum(1 for option in parsed_options.values() if option is not None)


def _vote_distribution(parsed_options: Mapping[str, str | None]) -> dict[str, int]:
    votes = [option for option in parsed_options.values() if option is not None]
    return {label: votes.count(label) for label in _OPTION_LABELS}


def _answer_vote_entropy(parsed_options: Mapping[str, str | None]) -> float:
    votes = [option for option in parsed_options.values() if option is not None]
    if not votes:
        return 0.0
    entropy = 0.0
    for label in _OPTION_LABELS:
        probability = votes.count(label) / len(votes)
        if probability:
            entropy -= probability * math.log(probability, 2)
    return round(entropy / 2.0, 6)


def _majority_margin(parsed_options: Mapping[str, str | None]) -> float:
    votes = [option for option in parsed_options.values() if option is not None]
    if len(votes) < 2:
        return 0.0
    counts = sorted((votes.count(label) for label in _OPTION_LABELS), reverse=True)
    return round((counts[0] - counts[1]) / len(votes), 6)


def _consensus_fraction(parsed_options: Mapping[str, str | None]) -> float:
    votes = [option for option in parsed_options.values() if option is not None]
    if not votes:
        return 0.0
    return round(max(votes.count(label) for label in _OPTION_LABELS) / len(votes), 6)


def _rationale_similarity(
    graph: OrgGraphSnapshot,
    node_outputs: Mapping[str, str],
) -> float:
    answer_node_ids = [
        node.node_id
        for node in graph.nodes
        if node.role in _ANSWER_ROLES and node.node_id in node_outputs
    ]
    if len(answer_node_ids) < 2:
        return 0.0
    tokens = [_content_tokens(node_outputs[node_id]) for node_id in answer_node_ids]
    pairs = []
    for left_index, left in enumerate(tokens):
        for right in tokens[left_index + 1 :]:
            if not left or not right:
                continue
            pairs.append(len(left & right) / len(left | right))
    if not pairs:
        return 0.0
    return round(sum(pairs) / len(pairs), 6)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z_'-]{2,}", text.lower())
        if token
        not in {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "final",
            "option",
            "answer",
            "correct",
            "incorrect",
        }
    }


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
