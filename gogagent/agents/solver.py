"""SolverAgent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gogagent.agents.base import (
    Agent,
    effective_context_inputs,
    format_task_with_context,
    llm_audit_metadata,
)
from gogagent.datasets.prompt_specs import (
    format_mmlu_direct_task,
    format_mmlu_fewshot_task,
    parse_mmlu_answer_like,
)
from gogagent.graph.schema import GraphMessage
from gogagent.llm import AgentContext
from gogagent.prompt import agent_system_prompt


@dataclass
class SolverAgent(Agent):
    """Initial task solver prompt wrapper."""

    agent_type: ClassVar[str] = "SolverAgent"
    role: ClassVar[str] = "solver"
    description: ClassVar[str] = "Solves the task and produces an initial answer."
    standalone: ClassVar[bool] = True
    prompt_key: ClassVar[str] = "solver"
    output_mode: ClassVar[str] = "answer"

    def execute(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None = None,
    ) -> GraphMessage:
        """Execute MMLU solver calls with the direct DeepSeek baseline prompt."""

        if _is_mmlu_problem(problem):
            return self._execute_mmlu_direct(problem, inputs, context)
        return super().execute(problem, inputs, context=context)

    def _execute_mmlu_direct(
        self,
        problem: Mapping[str, Any],
        inputs: Mapping[str, GraphMessage],
        context: AgentContext | None,
    ) -> GraphMessage:
        if context is None or context.llm_client is None:
            raise RuntimeError(
                f"{self.agent_type}.execute requires AgentContext with llm_client"
            )
        fewshot_examples = problem.get("mmlu_fewshot_examples")
        if fewshot_examples:
            task_prompt = format_mmlu_fewshot_task(problem, _validate_fewshot(fewshot_examples))
        else:
            task_prompt = format_mmlu_direct_task(problem)
        context_inputs = effective_context_inputs(problem, inputs)
        output_style = mmlu_output_style(problem)
        prompt = format_task_with_context(
            task_prompt,
            context_inputs,
            instruction=self.context_instruction_for(problem),
        )
        if output_style == "brief_rationale":
            prompt = f"{prompt}\n\n{MMLU_BRIEF_RATIONALE_OUTPUT_INSTRUCTION}"
        system_prompt = agent_system_prompt(
            problem,
            "solver_brief_rationale" if output_style == "brief_rationale" else self.prompt_key,
            role=self.role,
        )
        response = context.llm_client.chat_text(
            role=self.role,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        raw_output = response.text.strip()
        if not raw_output:
            raise RuntimeError(f"{self.agent_type} LLM response must not be empty")
        answer = (
            parse_mmlu_answer_like(raw_output)
            if output_style == "brief_rationale"
            else raw_output
        )
        return self.make_message(
            content=raw_output if output_style == "brief_rationale" else answer,
            answer=answer,
            metadata={
                "raw_output": response.text,
                "direct_mmlu_solver": True,
                "mmlu_output_style": output_style,
                "mmlu_fewshot_count": len(fewshot_examples) if fewshot_examples else 0,
                "contextual_prompt": bool(context_inputs),
                "exact_letter": answer in {"A", "B", "C", "D"},
                "llm": {
                    "model": response.model,
                    "usage": response.usage.to_dict(),
                    "latency_seconds": response.latency_seconds,
                },
                "llm_audit": [llm_audit_metadata(response)],
            },
        )


def _is_mmlu_problem(
    problem: Mapping[str, Any],
) -> bool:
    return str(problem.get("dataset", "")).strip().lower() == "mmlu"


MMLU_BRIEF_RATIONALE_OUTPUT_INSTRUCTION = (
    "Return exactly three lines:\n"
    "Answer: <A, B, C, or D>\n"
    "Reason: <one short sentence using the decisive clue or rule>\n"
    "Risk: <one short sentence naming the most tempting distractor or uncertainty>"
)


def mmlu_output_style(problem: Mapping[str, Any]) -> str:
    """Return the optional MMLU agent output style."""

    style = str(problem.get("mmlu_agent_output_style", "answer_only")).strip().lower()
    if style in {"", "answer_only", "answer-only"}:
        return "answer_only"
    if style in {"brief_rationale", "brief-rationale"}:
        return "brief_rationale"
    raise ValueError(f"unsupported mmlu_agent_output_style: {style!r}")


def _validate_fewshot(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("mmlu_fewshot_examples must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError("mmlu_fewshot_examples must contain mapping items")
    return value
