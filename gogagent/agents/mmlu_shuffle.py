"""MMLU option-shuffle self-consistency agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import random
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import (
    Agent,
    latest_answer,
    llm_audit_metadata,
    llm_metadata,
    parent_context_inputs,
)
from gogagent.datasets.prompt_specs import (
    mmlu_pro_choice_labels,
    mmlu_pro_options,
    parse_mmlu_answer_like,
    parse_mmlu_pro_answer_like,
)
from gogagent.graph.schema import GraphMessage
from gogagent.llm import AgentContext
from gogagent.prompt import MMLU_SOLVER_SYSTEM_PROMPT, agent_system_prompt


_SHUFFLE_RNG = random.Random(int(os.environ.get("GOGAGENT_MMLU_SHUFFLE_SEED", "18")))


@dataclass
class ShuffledMMLUSolverAgent(Agent):
    """UP-only MMLU solver that sees a shuffled option order."""

    agent_type: ClassVar[str] = "ShuffledMMLUSolverAgent"
    role: ClassVar[str] = "mmlu_shuffle_vote"
    description: ClassVar[str] = (
        "Solves an MMLU question with shuffled answer labels and maps the vote back."
    )
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "solver"
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Return one independent shuffled vote mapped to original A/B/C/D labels."""

        del inputs
        return execute_mmlu_shuffle_vote(
            self,
            problem,
            context=context,
            metadata={"up_internal_agent": True},
        )


@dataclass
class MMLUMajorityVoteAgent(Agent):
    """UP-only deterministic voter for anchor + shuffled MMLU votes."""

    agent_type: ClassVar[str] = "MMLUMajorityVoteAgent"
    role: ClassVar[str] = "mmlu_majority_vote"
    description: ClassVar[str] = (
        "Chooses a 2-vote majority among anchor and shuffled MMLU votes; otherwise "
        "falls back to the anchor answer."
    )
    standalone: ClassVar[bool] = False
    prompt_key: ClassVar[str] = "format_verifier"
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Vote deterministically; no extra LLM call is needed."""

        del context
        require_mmlu_choice_problem(problem)
        parent_inputs = parent_context_inputs(problem)
        anchor_answer = latest_answer(parent_inputs)
        if anchor_answer is None:
            raise RuntimeError(f"{self.agent_type} requires an upstream anchor answer")
        anchor_answer = parse_choice_answer(anchor_answer, problem)

        shuffled_votes = [
            parse_choice_answer(message.answer or message.content, problem)
            for message in inputs.values()
            if message.role == ShuffledMMLUSolverAgent.role
        ]
        labels = choice_labels_for_problem(problem)
        answer = choose_mmlu_vote(
            [anchor_answer, *shuffled_votes],
            fallback=anchor_answer,
            labels=labels,
        )
        return self.make_message(
            content=answer,
            answer=answer,
            notes={
                "anchor_answer": anchor_answer,
                "shuffled_votes": shuffled_votes,
                "votes": [anchor_answer, *shuffled_votes],
                "vote_counts": dict(Counter([anchor_answer, *shuffled_votes])),
                "fallback_used": answer == anchor_answer
                and Counter([anchor_answer, *shuffled_votes]).most_common(1)[0][1] < 2,
            },
            metadata={
                "deterministic_vote": True,
                "mmlu_shuffled_self_consistency": True,
            },
        )


def execute_mmlu_shuffle_vote(
    agent: Agent,
    problem: Mapping[str, Any],
    *,
    context: AgentContext | None,
    metadata: Mapping[str, Any] | None = None,
) -> GraphMessage:
    """Run one DeepSeek MMLU vote with shuffled option labels."""

    require_mmlu_choice_problem(problem)
    if context is None or context.llm_client is None:
        raise RuntimeError(f"{agent.agent_type}.execute requires AgentContext with llm_client")

    prompt, displayed_to_original = format_mmlu_shuffled_prompt(problem)
    response = context.llm_client.chat_text(
        role=ShuffledMMLUSolverAgent.role,
        prompt=prompt,
        system_prompt=shuffle_solver_system_prompt(problem),
    )
    displayed_answer = parse_displayed_answer(response.text, displayed_to_original)
    if displayed_answer not in displayed_to_original:
        raise RuntimeError(f"invalid displayed MMLU answer: {displayed_answer!r}")
    answer = displayed_to_original[displayed_answer]
    return agent.make_message(
        content=answer,
        answer=answer,
        notes={
            "displayed_answer": displayed_answer,
            "displayed_to_original": dict(displayed_to_original),
        },
        metadata={
            "raw_output": response.text,
            "mmlu_shuffled_self_consistency": True,
            "shuffled_options": True,
            **dict(metadata or {}),
            "llm": llm_metadata(response),
            "llm_audit": [llm_audit_metadata(response, phase="shuffled_vote")],
        },
    )


def format_mmlu_shuffled_prompt(
    problem: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Format an MMLU/MMLU-Pro prompt exactly like the shuffled baseline."""

    if dataset_name(problem) == "mmlu_pro":
        return format_mmlu_pro_shuffled_prompt(problem)
    options = problem.get("options")
    if not isinstance(options, Mapping) or not all(letter in options for letter in "ABCD"):
        raise ValueError("MMLU shuffled vote requires A/B/C/D options")
    original_letters = list("ABCD")
    _SHUFFLE_RNG.shuffle(original_letters)
    displayed_to_original = dict(zip("ABCD", original_letters, strict=True))
    displayed_lines = [
        f"{displayed_letter}. {options[original_letter]}"
        for displayed_letter, original_letter in displayed_to_original.items()
    ]
    subject = str(problem.get("subject", "unknown")).replace("_", " ")
    prompt = (
        f"Subject: {subject}\n\n"
        f"Question:\n{problem['question']}\n\n"
        "Options:\n"
        + "\n".join(displayed_lines)
        + "\n\nAnswer:"
    )
    return prompt, displayed_to_original


def format_mmlu_pro_shuffled_prompt(
    problem: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Format a shuffled MMLU-Pro prompt with dynamic option labels."""

    options = mmlu_pro_options(problem)
    labels = list(mmlu_pro_choice_labels(problem))
    if len(labels) < 2:
        raise ValueError("MMLU-Pro shuffled vote requires at least two options")
    original_letters = list(labels)
    _SHUFFLE_RNG.shuffle(original_letters)
    displayed_to_original = dict(zip(labels, original_letters, strict=True))
    displayed_lines = [
        f"{displayed_letter}. {options[original_letter]}"
        for displayed_letter, original_letter in displayed_to_original.items()
    ]
    subject = str(problem.get("subject", "unknown")).replace("_", " ")
    prompt = (
        f"Subject: {subject}\n\n"
        f"Question:\n{problem['question']}\n\n"
        "Options:\n"
        + "\n".join(displayed_lines)
        + "\n\nAnswer:"
    )
    return prompt, displayed_to_original


def choose_mmlu_vote(
    votes: list[str | None],
    *,
    fallback: str | None,
    labels: tuple[str, ...] = ("A", "B", "C", "D"),
) -> str:
    """Choose a 2-vote majority, otherwise return the anchor fallback."""

    legal = set(labels)
    counts = Counter(vote for vote in votes if vote in legal)
    if not counts:
        if fallback is None:
            raise RuntimeError("MMLU vote has no valid candidates and no fallback")
        return fallback
    answer, count = counts.most_common(1)[0]
    if count >= 2:
        return answer
    if fallback is None:
        raise RuntimeError("MMLU vote has no majority and no fallback")
    return fallback


def require_mmlu_choice_problem(problem: Mapping[str, Any]) -> None:
    """Fail explicitly if shuffled self-consistency is used elsewhere."""

    dataset = dataset_name(problem)
    if dataset not in {"mmlu", "mmlu_pro"}:
        raise RuntimeError(
            "MMLU shuffled self-consistency is only defined for "
            "dataset='mmlu' or dataset='mmlu_pro'"
        )


def dataset_name(problem: Mapping[str, Any]) -> str:
    """Return the normalized dataset name carried by a problem mapping."""

    return str(problem.get("dataset", "")).strip().lower()


def choice_labels_for_problem(problem: Mapping[str, Any]) -> tuple[str, ...]:
    """Return legal option labels for MMLU-style choice tasks."""

    if dataset_name(problem) == "mmlu_pro":
        return mmlu_pro_choice_labels(problem)
    return ("A", "B", "C", "D")


def parse_choice_answer(value: Any, problem: Mapping[str, Any]) -> str:
    """Parse an original-label answer for MMLU or MMLU-Pro."""

    if dataset_name(problem) == "mmlu_pro":
        return parse_mmlu_pro_answer_like(str(value), problem)
    return parse_mmlu_answer_like(str(value))


def parse_displayed_answer(
    value: Any,
    displayed_to_original: Mapping[str, str],
) -> str:
    """Parse an answer expressed in the displayed shuffled label space."""

    labels = tuple(displayed_to_original)
    if len(labels) == 4 and set(labels) == {"A", "B", "C", "D"}:
        return parse_mmlu_answer_like(str(value))
    return parse_mmlu_pro_answer_like(str(value), labels)


def shuffle_solver_system_prompt(problem: Mapping[str, Any]) -> str:
    """Return the dataset-specific system prompt for shuffled solver calls."""

    if dataset_name(problem) == "mmlu_pro":
        return agent_system_prompt(
            problem,
            "solver",
            role=ShuffledMMLUSolverAgent.role,
        )
    return MMLU_SOLVER_SYSTEM_PROMPT
