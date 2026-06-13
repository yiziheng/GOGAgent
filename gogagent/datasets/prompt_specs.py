"""Dataset-specific prompt formatting and answer parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping

from gogagent.prompt import MMLU_SOLVER_SYSTEM_PROMPT


_MMLU_ANSWER_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|option|choice)?\s*[:\-]?\s*\(?([ABCD])\)?\s*[\.\。]?\s*$",
    re.IGNORECASE,
)
_MMLU_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer|option|choice)\s*[:\-]?\s*\(?([ABCD])\)?",
    re.IGNORECASE,
)
_MMLU_STANDALONE_OPTION_RE = re.compile(
    r"^\s*\(?([ABCD])\)?\s*[\.\。]?\s*$",
    re.IGNORECASE,
)
_MMLU_INLINE_OPTION_RE = re.compile(
    r"(?:^|[\s\(\[\{\"'`])(?:correct\s+answer|answer|option|choice|letter)?"
    r"\s*(?:is|:|\-)?\s*([A-D])(?:[\s\)\]\}.:,;\"'`]|$)",
    re.IGNORECASE,
)

MMLU_DIRECT_SYSTEM_PROMPT = MMLU_SOLVER_SYSTEM_PROMPT


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

    def format_direct_task(self, problem: Mapping[str, Any]) -> str:
        return format_mmlu_direct_task(problem)

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
class MultiAgentBenchPromptSpec(DatasetPromptSpec):
    """MultiAgentBench/MARBLE task prompt and generic answer parser."""

    def format_problem(self, problem: Mapping[str, Any]) -> str:
        lines = ["MultiAgentBench task."]
        scenario = str(problem.get("scenario", "")).strip()
        if scenario:
            lines.append(f"Scenario: {scenario}")

        lines.extend(["", "Task:", str(problem.get("task", _problem_statement(problem))).strip()])

        options = _options(problem)
        if options:
            lines.extend(["", "Options:"])
            for label, value in options.items():
                lines.append(f"{label}. {value}")

        output_format = str(problem.get("output_format") or "").strip()
        if output_format:
            lines.extend(["", "Requested output format:", output_format])

        for title, key, limit in (
            ("Agent profiles", "agents", 1800),
            ("Relationships", "relationships", 1200),
            ("Environment/context", "environment", 2200),
            ("Additional context", "context", 1200),
            ("Metrics", "metrics", 1000),
        ):
            value = problem.get(key)
            if value:
                lines.extend(["", f"{title}:", _compact_value(value, limit=limit)])
        return "\n".join(lines)

    def answer_instruction(self, problem: Mapping[str, Any]) -> str:
        if _mmlu_options(problem) is not None:
            return "Answer with exactly one option letter. Do not include explanation."
        output_format = str(problem.get("output_format") or "").strip()
        if output_format:
            return f"Return only the requested final deliverable. Output format: {output_format}"
        return "Return only the final answer or deliverable. Do not include unnecessary commentary."

    def parse_answer(self, text: str, problem: Mapping[str, Any]) -> str:
        raw = text.strip()
        if not raw:
            raise RuntimeError("answer-only LLM response is empty")
        if _mmlu_options(problem) is not None:
            return parse_mmlu_answer_like(raw)
        return raw

    def matches(self, problem: Mapping[str, Any]) -> bool:
        return str(problem.get("dataset_protocol", "")).startswith("multiagentbench")


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
    "multiagentbench": MultiAgentBenchPromptSpec(),
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


def parse_mmlu_answer_like(text: str) -> str:
    """Extract an MMLU option from answer-first short text.

    This is intentionally looser than ``parse_answer_text`` and is only for
    internal agent outputs that may include short advisory context such as
    ``Reason`` and ``Risk`` lines.
    """

    raw = text.strip()
    if not raw:
        raise RuntimeError("MMLU answer-like response is empty")
    if raw.upper() in {"A", "B", "C", "D"}:
        return raw.upper()

    explicit_matches = _MMLU_EXPLICIT_ANSWER_RE.findall(raw)
    if explicit_matches:
        return explicit_matches[-1].upper()

    standalone_matches = [
        match.group(1).upper()
        for line in raw.splitlines()
        if (match := _MMLU_STANDALONE_OPTION_RE.fullmatch(line.strip()))
    ]
    if standalone_matches:
        return standalone_matches[-1]

    inline_matches = _MMLU_INLINE_OPTION_RE.findall(raw)
    if inline_matches:
        return inline_matches[-1].upper()

    raise RuntimeError(f"cannot extract MMLU option from {text!r}")


def format_mmlu_direct_task(problem: Mapping[str, Any]) -> str:
    """Format MMLU exactly like the standalone DeepSeek baseline."""

    options = _mmlu_options(problem)
    if options is None:
        raise ValueError("MMLU direct task requires A/B/C/D options")
    subject = str(problem.get("subject", "unknown")).replace("_", " ")
    question = str(problem.get("question", _problem_statement(problem))).strip()
    return (
        f"Subject: {subject}\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A. {options.get('A', options.get('a', ''))}\n"
        f"B. {options.get('B', options.get('b', ''))}\n"
        f"C. {options.get('C', options.get('c', ''))}\n"
        f"D. {options.get('D', options.get('d', ''))}\n\n"
        "Answer:"
    )


def format_mmlu_fewshot_task(
    problem: Mapping[str, Any],
    fewshot_examples: Iterable[Mapping[str, Any]],
) -> str:
    """Format MMLU with same-subject dev examples followed by the test question."""

    subject = str(problem.get("subject", "")).strip()
    subject_text = subject.replace("_", " ") if subject else "the subject"
    lines = [
        f"The following are multiple choice questions (with answers) about {subject_text}.",
        "",
    ]
    for example in fewshot_examples:
        options = _mmlu_options(example)
        if options is None:
            raise ValueError("MMLU few-shot examples require A/B/C/D options")
        answer = str(example.get("answer", "")).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"MMLU few-shot answer must be A/B/C/D, got {answer!r}")
        lines.extend(_format_mmlu_question_block(example, options))
        lines.append(f"Answer: {answer}")
        lines.append("")

    test_options = _mmlu_options(problem)
    if test_options is None:
        raise ValueError("MMLU few-shot task requires A/B/C/D options")
    lines.extend(_format_mmlu_question_block(problem, test_options))
    lines.append("Answer:")
    return "\n".join(lines)


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


def _options(problem: Mapping[str, Any]) -> Mapping[str, Any] | None:
    options = problem.get("options") or problem.get("choices")
    if not isinstance(options, Mapping):
        return None
    return {str(label).upper(): value for label, value in options.items()}


def _compact_value(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_mmlu_question_block(
    problem: Mapping[str, Any],
    options: Mapping[str, Any],
) -> list[str]:
    return [
        str(problem.get("question", _problem_statement(problem))).strip(),
        f"A. {options.get('A', options.get('a', ''))}",
        f"B. {options.get('B', options.get('b', ''))}",
        f"C. {options.get('C', options.get('c', ''))}",
        f"D. {options.get('D', options.get('d', ''))}",
    ]
