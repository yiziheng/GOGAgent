"""Offline smoke checks for all three domain adapters."""

from pathlib import Path

from gogagent.adapters.registry import get_adapter
from gogagent.cli import SMOKE_TASKS
from gogagent.core.rollout import RolloutEngine
from gogagent.llm.mock import MockLLM


def test_mock_rollouts_export_visible_graphs(tmp_path: Path) -> None:
    for domain, task in SMOKE_TASKS.items():
        result = RolloutEngine(get_adapter(domain), MockLLM(), tmp_path).run(task, f"test-{domain}")
        directory = Path(result["artifact_directory"])
        assert (directory / "trace.jsonl").exists()
        assert (directory / "gog.json").exists()
        assert (directory / "gog.svg").exists()
        assert list((directory / "snapshots").glob("*.svg"))
