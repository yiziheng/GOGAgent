"""AdversarialJudgeAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.solver import (
    MMLU_BRIEF_RATIONALE_OUTPUT_INSTRUCTION,
    mmlu_output_style,
)
from gogagent.agents.base import (
    Agent,
    aggregate_llm_metadata,
    answer_instruction,
    format_task_with_context,
    format_upstream_context,
    format_problem,
    latest_answer,
    llm_metadata,
    llm_audit_metadata,
    parse_answer_text,
)
from gogagent.datasets.prompt_specs import format_mmlu_direct_task, parse_mmlu_answer_like
from gogagent.graph.schema import GraphMessage
from gogagent.llm.client import AgentContext
from gogagent.prompt import agent_system_prompt, context_instruction
from gogagent.agents.mmlu_shuffle import (
    choice_labels_for_problem,
    choose_mmlu_vote,
    execute_mmlu_shuffle_vote,
)


@dataclass
class AdversarialJudgeAgent(Agent):
    """Independent second-opinion plus disagreement arbitration wrapper."""

    agent_type: ClassVar[str] = "AdversarialJudgeAgent"
    role: ClassVar[str] = "adversarial_judge"
    description: ClassVar[str] = (
        "Answers independently and arbitrates only when it disagrees with the upstream solver."
    )
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = "adversarial_judge"
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Use a blind second answer, then arbitrate only on disagreement."""

        if context is None or context.llm_client is None:
            raise RuntimeError(
                f"{self.agent_type}.execute requires AgentContext with llm_client"
            )
        if _is_mmlu_choice_problem(problem):
            return self._execute_mmlu_shuffled_vote(problem, inputs, context)

        upstream_raw = latest_answer(inputs)
        if upstream_raw is None:
            raise RuntimeError(f"{self.agent_type} requires an upstream answer")
        upstream_answer = normalize_candidate_answer(upstream_raw, problem)
        upstream_message = latest_message(inputs)
        independent_context = prior_messages(inputs)

        independent_response = context.llm_client.chat_text(
            role=self.role,
            prompt=build_independent_prompt(problem, independent_context),
            system_prompt=agent_system_prompt(
                problem,
                (
                    "adversarial_judge_brief_rationale"
                    if mmlu_output_style(problem) == "brief_rationale"
                    else self.prompt_key
                ),
                role=self.role,
            ),
        )
        independent_answer = normalize_candidate_answer(independent_response.text, problem)

        responses = [("independent_second_answer", independent_response)]
        final_response = independent_response
        arbitration_used = False
        if independent_answer == upstream_answer:
            answer = upstream_answer
        else:
            arbitration_used = True
            arbitration_response = context.llm_client.chat_text(
                role=self.role,
                prompt=build_arbitration_prompt(
                    problem,
                    context_inputs=inputs,
                    upstream_answer=upstream_answer,
                    upstream_content=(
                        upstream_message.content if upstream_message is not None else None
                    ),
                    independent_answer=independent_answer,
                    independent_content=independent_response.text,
                ),
                system_prompt=agent_system_prompt(
                    problem,
                    "adversarial_arbitrator",
                    role=self.role,
                ),
            )
            responses.append(("arbitration", arbitration_response))
            final_response = arbitration_response
            answer = normalize_candidate_answer(arbitration_response.text, problem)

        llm_calls = [
            {"phase": phase, **llm_metadata(response)}
            for phase, response in responses
        ]
        return self.make_message(
            content=answer,
            answer=answer,
            notes={
                "upstream_answer": upstream_answer,
                "upstream_raw_answer": str(upstream_raw),
                "upstream_content": upstream_message.content if upstream_message else None,
                "independent_answer": independent_answer,
                "independent_content": independent_response.text.strip(),
                "agreement": independent_answer == upstream_answer,
                "arbitration_used": arbitration_used,
                "changed_answer": answer != upstream_answer,
            },
            metadata={
                "raw_output": final_response.text,
                "independent_raw_output": independent_response.text,
                "llm": aggregate_llm_metadata(responses),
                "llm_calls": llm_calls,
                "llm_audit": [
                    llm_audit_metadata(response, phase=phase)
                    for phase, response in responses
                ],
            },
        )

    def _execute_mmlu_shuffled_vote(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext,
    ) -> GraphMessage:
        """Run one shuffled MMLU vote and fall back to the anchor on disagreement."""

        upstream_raw = latest_answer(inputs)
        if upstream_raw is None:
            raise RuntimeError(f"{self.agent_type} requires an upstream answer")
        anchor_answer = normalize_candidate_answer(upstream_raw, problem)
        shuffled = execute_mmlu_shuffle_vote(
            self,
            problem,
            context=context,
            metadata={"standalone_adversarial_shuffle": True},
        )
        shuffled_answer = normalize_candidate_answer(shuffled.answer, problem)
        answer = choose_mmlu_vote(
            [anchor_answer, shuffled_answer],
            fallback=anchor_answer,
            labels=choice_labels_for_problem(problem),
        )
        return self.make_message(
            content=answer,
            answer=answer,
            notes={
                "anchor_answer": anchor_answer,
                "shuffled_answer": shuffled_answer,
                "agreement": anchor_answer == shuffled_answer,
                "fallback_used": answer == anchor_answer and anchor_answer != shuffled_answer,
                "displayed_answer": shuffled.notes.get("displayed_answer"),
                "displayed_to_original": shuffled.notes.get("displayed_to_original"),
            },
            metadata={
                **dict(shuffled.metadata),
                "mmlu_adversarial_mode": "single_shuffle_anchor_fallback",
            },
        )


def build_independent_prompt(
    problem: Mapping[str, Any],
    context_inputs: Mapping[str, GraphMessage] | None = None,
) -> str:
    """Build the second-answer prompt.

    The latest solver output is intentionally excluded by the caller to keep the
    second answer independent, but earlier planning/briefing context is allowed.
    """

    output_instruction = (
        MMLU_BRIEF_RATIONALE_OUTPUT_INSTRUCTION
        if mmlu_output_style(problem) == "brief_rationale"
        else answer_instruction(problem)
    )
    task = format_task_with_context(
        format_direct_task(problem),
        context_inputs or {},
        instruction=context_instruction(problem),
    )
    return "\n\n".join(
        section
        for section in (
            task,
            output_instruction,
        )
        if section.strip()
    )


def build_arbitration_prompt(
    problem: Mapping[str, Any],
    *,
    context_inputs: Mapping[str, GraphMessage],
    upstream_answer: str,
    upstream_content: str | None,
    independent_answer: str,
    independent_content: str | None,
) -> str:
    """Build the disagreement arbitration prompt."""

    use_brief = mmlu_output_style(problem) == "brief_rationale"
    upstream_section = (
        f"Upstream solver output:\n{str(upstream_content).strip()}"
        if use_brief and upstream_content
        else f"Upstream solver answer: {upstream_answer}"
    )
    independent_section = (
        f"Independent second output:\n{str(independent_content).strip()}"
        if use_brief and independent_content
        else f"Independent second answer: {independent_answer}"
    )
    return "\n\n".join(
        section
        for section in (
            format_direct_task(problem),
            format_upstream_context(context_inputs),
            "Candidate answers:",
            upstream_section,
            independent_section,
            f"Parsed upstream answer: {upstream_answer}",
            f"Parsed independent answer: {independent_answer}",
            "Resolve the disagreement by solving the task from the original request. "
            "Do not favor either candidate by default.",
            answer_instruction(problem),
        )
        if section.strip()
    )


def format_direct_task(problem: Mapping[str, Any]) -> str:
    """Use the MMLU direct baseline format when possible."""

    if str(problem.get("dataset", "")).strip().lower() == "mmlu":
        return format_mmlu_direct_task(problem)
    return format_problem(problem)


def _is_mmlu_choice_problem(problem: Mapping[str, Any]) -> bool:
    return str(problem.get("dataset", "")).strip().lower() in {"mmlu", "mmlu_pro"}


def normalize_candidate_answer(value: Any, problem: Mapping[str, Any]) -> str:
    """Normalize an upstream candidate answer for internal comparison."""

    text = str(value).strip()
    if not text:
        raise RuntimeError("upstream answer must not be empty")
    if str(problem.get("dataset", "")).strip().lower() == "mmlu":
        return parse_mmlu_answer_like(text)
    return parse_answer_text(text, problem)


def latest_message(inputs: Mapping[str, GraphMessage]) -> GraphMessage | None:
    """Return the last upstream message, preserving content for arbitration."""

    if not inputs:
        return None
    return GraphMessage.from_dict(list(inputs.values())[-1])


def prior_messages(inputs: Mapping[str, GraphMessage]) -> dict[str, GraphMessage]:
    """Return all context messages before the latest answer-bearing message."""

    items = list(inputs.items())
    if not items:
        return {}
    return {
        str(node_id): GraphMessage.from_dict(message)
        for node_id, message in items[:-1]
    }
