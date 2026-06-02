"""Check the three dataset file adapters without downloading benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gogagent.datasets import (
    load_gsm8k_jsonl,
    load_humaneval_jsonl,
    load_mmlu_directory,
)


def main() -> None:
    with TemporaryDirectory() as temp_directory:
        root = Path(temp_directory)
        gsm8k = root / "gsm8k.jsonl"
        humaneval = root / "humaneval.jsonl"
        mmlu = root / "abstract_algebra_test.csv"
        gsm8k.write_text(
            json.dumps({"question": "What is 2 + 3?", "answer": "#### 5"}) + "\n",
            encoding="utf-8",
        )
        humaneval.write_text(
            json.dumps(
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n",
                    "entry_point": "add",
                    "canonical_solution": "    return a + b\n",
                    "test": "assert add(1, 2) == 3",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        mmlu.write_text("What is 2 + 3?,3,4,5,6,C\n", encoding="utf-8")

        examples = [
            next(load_gsm8k_jsonl(gsm8k)),
            next(load_mmlu_directory(root)),
            next(load_humaneval_jsonl(humaneval)),
        ]
        assert [example.dataset for example in examples] == [
            "gsm8k",
            "mmlu",
            "humaneval",
        ]
        assert examples[1].public_task["options"]["C"] == "5"
        assert "test" not in examples[2].public_task
        print("dataset loaders: ok")


if __name__ == "__main__":
    main()
