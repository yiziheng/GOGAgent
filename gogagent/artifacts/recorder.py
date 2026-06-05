"""Run artifact recorder for Round 1 rollouts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from gogagent.artifacts.visualize import save_graph_json, save_graph_svg


class RunRecorder:
    """Write rollout traces, summaries, and graph visualizations."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.graph_json_path = self.run_dir / "gog.json"
        self.graph_svg_path = self.run_dir / "gog.svg"

    def record_trace(self, event: Mapping[str, Any] | Any) -> None:
        """Append one rollout event to ``trace.jsonl``."""

        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(event), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def record_step(self, event: Mapping[str, Any] | Any) -> None:
        """Alias for callers that name trace events as steps."""

        self.record_trace(event)

    def save_summary(self, summary: Mapping[str, Any] | Any) -> Path:
        """Save compact run metrics to ``summary.json``."""

        self.summary_path.write_text(
            json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.summary_path

    def save_graph(self, graph: Any) -> tuple[Path, Path]:
        """Save both ``gog.json`` and ``gog.svg``."""

        json_path = save_graph_json(graph, self.graph_json_path)
        svg_path = save_graph_svg(graph, self.graph_svg_path)
        return json_path, svg_path

    def paths(self) -> dict[str, str]:
        """Return generated artifact paths for summaries/tests."""

        return {
            "run_dir": str(self.run_dir),
            "trace": str(self.trace_path),
            "summary": str(self.summary_path),
            "gog_json": str(self.graph_json_path),
            "gog_svg": str(self.graph_svg_path),
        }


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and GraphMessage-like values to JSON-safe objects."""

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
