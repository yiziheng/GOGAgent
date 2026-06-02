"""Command line entry point for offline mock rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gogagent.adapters.registry import get_adapter
from gogagent.core.rollout import RolloutEngine
from gogagent.llm.mock import MockLLM


SMOKE_TASKS = {
    "gsm8k": {
        "question": "A shop has 3 boxes with 4 pencils each. How many pencils are there?",
    },
    "mmlu": {
        "question": "Which option is the capital of France?",
        "options": {"A": "Berlin", "B": "Paris", "C": "Rome", "D": "Madrid"},
        "subject": "world_history",
    },
    "humaneval": {
        "prompt": "def add(a, b):\\n    \"\"\"Return the sum of a and b.\"\"\"",
        "entry_point": "add",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a label-blind GOGAgent mock rollout")
    parser.add_argument("--domain", choices=sorted(SMOKE_TASKS), required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--episode-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.domain)
    result = RolloutEngine(adapter, MockLLM(), args.artifact_root).run(
        SMOKE_TASKS[args.domain],
        episode_id=args.episode_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
