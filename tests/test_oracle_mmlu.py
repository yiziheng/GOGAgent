#!/usr/bin/env python3
"""MMLU oracle checks aligned with OpenG-MAS option-letter scoring."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.graph.schema import GraphMessage
from gogagent.reward.oracle import score_answer


def main() -> None:
    example = {
        "task_id": "abstract_algebra-test-2",
        "question": "Find the maximum possible order for an element of S_n for n = 10.",
        "options": {
            "A": "6",
            "B": "12",
            "C": "30",
            "D": "105",
        },
    }
    assert not score_answer(
        "mmlu",
        example,
        GraphMessage(role="solver", content="LCM gives 30.", answer="30"),
        gold="C",
    ).correct
    assert score_answer(
        "mmlu",
        example,
        GraphMessage(role="solver", content="The answer is C.", answer="C"),
        gold="C",
    ).correct
    assert not score_answer(
        "mmlu",
        example,
        GraphMessage(role="solver", content="LCM gives 12.", answer="12"),
        gold="C",
    ).correct

    chemistry = {
        "task_id": "college_chemistry-test-1",
        "question": "What is the chemical shift when the pH equals pKa?",
        "options": {
            "A": "3.41 ppm",
            "B": "3.98 ppm",
            "C": "4.33 ppm",
            "D": "4.62 ppm",
        },
    }
    assert not score_answer(
        "mmlu",
        chemistry,
        GraphMessage(role="solver", content="Average is 4.62 ppm.", answer="4.62 ppm"),
        gold="D",
    ).correct
    assert score_answer(
        "mmlu",
        chemistry,
        GraphMessage(role="solver", content="Average is 4.62 ppm.", answer="D. 4.62 ppm"),
        gold="D",
    ).correct
    print("MMLU oracle smoke passed")


if __name__ == "__main__":
    main()
