"""Run one network-free rollout per supported dataset."""

from __future__ import annotations

import json
from pathlib import Path

from gogagent.adapters.registry import get_adapter
from gogagent.cli import SMOKE_TASKS
from gogagent.core.rollout import RolloutEngine
from gogagent.llm.mock import MockLLM


def main() -> None:
    root = Path("artifacts/runs")
    results = []
    for domain, task in SMOKE_TASKS.items():
        engine = RolloutEngine(get_adapter(domain), MockLLM(), root)
        results.append(engine.run(task, episode_id=f"smoke-{domain}"))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
