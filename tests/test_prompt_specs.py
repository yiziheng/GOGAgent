#!/usr/bin/env python3
"""Dataset prompt-spec registry checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.datasets import (
    PROMPT_SPECS,
    answer_instruction,
    format_problem,
    get_prompt_spec,
    parse_answer_text,
)


def main() -> None:
    mmlu = {
        "dataset": "mmlu",
        "subject": "college_chemistry",
        "question": "Which option is correct?",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
    }
    assert sorted(PROMPT_SPECS) == ["gsm8k", "humaneval", "mmlu"]
    assert get_prompt_spec(mmlu).__class__.__name__ == "MMLUPromptSpec"
    assert "Choices:" in format_problem(mmlu)
    assert parse_answer_text("Answer: C", mmlu) == "C"
    assert "A, B, C, or D" in answer_instruction(mmlu)

    no_dataset_mmlu = {
        "question": "Which option is correct?",
        "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
    }
    assert get_prompt_spec(no_dataset_mmlu).__class__.__name__ == "MMLUPromptSpec"

    gsm8k = {"dataset": "gsm8k", "question": "What is 2 + 2?"}
    assert "numeric answer" in answer_instruction(gsm8k)
    assert parse_answer_text("4", gsm8k) == "4"

    humaneval = {"dataset": "humaneval", "prompt": "def f():\n    pass", "entry_point": "f"}
    assert "Python programming task" in format_problem(humaneval)

    try:
        get_prompt_spec({"dataset": "mmmlu", "question": "typo"})
    except ValueError as error:
        assert "unsupported dataset prompt spec" in str(error)
        assert "mmlu" in str(error)
    else:
        raise AssertionError("unknown explicit dataset must not use generic prompt spec")

    print("Prompt spec registry smoke passed")


if __name__ == "__main__":
    main()
