"""Dataset-specific prompt formatting and answer parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Mapping


_MMLU_ANSWER_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|option|choice)?\s*[:\-]?\s*\(?([ABCD])\)?\s*[\.\。]?\s*$",
    re.IGNORECASE,
)


class DatasetPromptSpec(ABC):
    """Dataset adapter for LLM-facing prompts and final-answer parsing."""

    @abstractmethod
    def format_problem(self, problem: Mapping[str, Any]) -> str:
        """Format a public task as a natural-language benchmark prompt."""

    @abstractmethod
    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        """Return the answer-only output instruction."""

    @abstractmethod
    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        """Parse an answer-only model response into the internal answer string."""

    def matches(self, problem: Mapping[str, Any]) -> bool:
        """Return whether this spec should handle a problem without dataset key."""

        return False


@dataclass(frozen=True)
class MMLUPromptSpec(DatasetPromptSpec):
    """MMLU multiple-choice prompt and A/B/C/D parser."""

    def format_problem(self, problem: Mapping[str, Any]) -> str:
        options = _mmlu_options(problem) or {}
        lines = ["The following is a multiple-choice question."]
        subject = str(problem.get("subject", "")).strip()
        if subject:
            lines.append(f"Subject: {subject}")
        lines.extend(
            [
                "",
                "Question:",
                str(problem.get("question", _problem_statement(problem))).strip(),
                "",
                "Choices:",
            ]
        )
        for label in ("A", "B", "C", "D"):
            lines.append(f"{label}. {options.get(label, options.get(label.lower(), ''))}")
        return "\n".join(lines)

    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        del problem
        return "Answer with exactly one letter: A, B, C, or D. Do not include explanation."

    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        del problem
        raw = text.strip()
        if not raw:
            raise RuntimeError("answer-only LLM response is empty")
        match = _MMLU_ANSWER_RE.fullmatch(raw)
        if not match:
            raise RuntimeError(
                "MMLU answer-only response must be exactly one option letter; "
                f"got {text!r}"
            )
        return match.group(1).upper()

    def matches(self, problem: Mapping[str, Any]) -> bool:
        return _mmlu_options(problem) is not None


@dataclass(frozen=True)
class GSM8KPromptSpec(DatasetPromptSpec):
    """GSM8K prompt and final-answer parser."""

    def format_problem(self, problem: Mapping[str, Any]) -> str:
        return (
            "Solve the following grade-school math problem.\n\n"
            f"Question:\n{_problem_statement(problem)}"
        )

    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        del problem
        return "Return only the final numeric answer. Do not include explanation."

    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        del problem
        raw = text.strip()
        if not raw:
            raise RuntimeError("answer-only LLM response is empty")
        return raw


@dataclass(frozen=True)
class HumanEvalPromptSpec(DatasetPromptSpec):
    """HumanEval prompt and code-answer parser."""

    def format_problem(self, problem: Mapping[str, Any]) -> str:
        prompt = problem.get("prompt", _problem_statement(problem))
        entry_point = str(problem.get("entry_point", "")).strip()
        lines = ["Complete the following Python programming task."]
        if entry_point:
            lines.append(f"Entry point: {entry_point}")
        lines.extend(["", str(prompt)])
        return "\n".join(lines)

    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        del problem
        return "Return only the completed Python code. Do not include markdown fences."

    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        del problem
        raw = text.strip()
        if not raw:
            raise RuntimeError("answer-only LLM response is empty")
        return raw


@dataclass(frozen=True)
class GenericPromptSpec(DatasetPromptSpec):
    """Fallback prompt for exact-match or unknown tasks."""

    def format_problem(self, problem: Mapping[str, Any]) -> str:
        return f"Task:\n{_problem_statement(problem)}"

    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        del problem
        return "Return only the final answer. Do not include explanation."

    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        del problem
        raw = text.strip()
        if not raw:
            raise RuntimeError("answer-only LLM response is empty")
        return raw


PROMPT_SPECS: dict[str, DatasetPromptSpec] = {
    "mmlu": MMLUPromptSpec(),
    "gsm8k": GSM8KPromptSpec(),
    "humaneval": HumanEvalPromptSpec(),
}
GENERIC_PROMPT_SPEC = GenericPromptSpec()


def get_prompt_spec(problem: Mapping[str, Any]) -> DatasetPromptSpec:
    """Return the dataset prompt spec for a public task."""

    dataset = str(problem.get("dataset", "")).strip().lower()
    if dataset in PROMPT_SPECS:
        return PROMPT_SPECS[dataset]
    if dataset:
        raise ValueError(
            f"unsupported dataset prompt spec {dataset!r}; "
            f"registered datasets: {sorted(PROMPT_SPECS)}"
        )
    for spec in PROMPT_SPECS.values():
        if spec.matches(problem):
            return spec
    return GENERIC_PROMPT_SPEC


def format_problem(problem: Mapping[str, Any]) -> str:
    """Format a task using the registered dataset prompt spec."""

    return get_prompt_spec(problem).format_problem(problem)


def answer_instruction(problem: Mapping[str, Any]) -> str:
    """Return the registered answer-only instruction for a task."""

    return get_prompt_spec(problem).answer_instruction(problem)


def parse_answer_text(text: str, problem: Mapping[str, Any]) -> str:
    """Parse answer-only model output with the registered dataset spec."""

    return get_prompt_spec(problem).parse_answer(text, problem)


def _problem_statement(problem: Mapping[str, Any]) -> str:
    for key in ("question", "prompt", "problem", "task", "input"):
        value = problem.get(key)
        if value:
            return str(value)
    return str(problem)


def _mmlu_options(problem: Mapping[str, Any]) -> Mapping[str, Any] | None:
    options = problem.get("options") or problem.get("choices")
    if not isinstance(options, Mapping):
        return None
    labels = {str(label).upper() for label in options}
    return options if {"A", "B", "C", "D"}.issubset(labels) else None
