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
from gogagent.datasets.prompt_specs import parse_mmlu_answer_like
from gogagent.graph.schema import GraphMessage
from gogagent.llm import AgentContext
from gogagent.prompt import MMLU_SOLVER_SYSTEM_PROMPT


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
        require_mmlu(problem)
        parent_inputs = parent_context_inputs(problem)
        anchor_answer = latest_answer(parent_inputs)
        if anchor_answer is None:
            raise RuntimeError(f"{self.agent_type} requires an upstream anchor answer")
        anchor_answer = parse_mmlu_answer_like(anchor_answer)

        shuffled_votes = [
            parse_mmlu_answer_like(message.answer or message.content)
            for message in inputs.values()
            if message.role == ShuffledMMLUSolverAgent.role
        ]
        answer = choose_mmlu_vote([anchor_answer, *shuffled_votes], fallback=anchor_answer)
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

    require_mmlu(problem)
    if context is None or context.llm_client is None:
        raise RuntimeError(f"{agent.agent_type}.execute requires AgentContext with llm_client")

    prompt, displayed_to_original = format_mmlu_shuffled_prompt(problem)
    response = context.llm_client.chat_text(
        role=ShuffledMMLUSolverAgent.role,
        prompt=prompt,
        system_prompt=MMLU_SOLVER_SYSTEM_PROMPT,
    )
    displayed_answer = parse_mmlu_answer_like(response.text)
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
    """Format an MMLU prompt exactly like the standalone shuffled baseline."""

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


def choose_mmlu_vote(votes: list[str | None], *, fallback: str | None) -> str:
    """Choose a 2-vote majority, otherwise return the anchor fallback."""

    counts = Counter(vote for vote in votes if vote in {"A", "B", "C", "D"})
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


def require_mmlu(problem: Mapping[str, Any]) -> None:
    """Fail explicitly if the shuffled self-consistency module is used elsewhere."""

    dataset = str(problem.get("dataset", "")).strip().lower()
    if dataset != "mmlu":
        raise RuntimeError("MMLU shuffled self-consistency is only defined for dataset='mmlu'")
