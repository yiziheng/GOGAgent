"""Verify train-only dense credit reaches GoG neighbor statistics."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gogagent.adapters.registry import get_adapter
from gogagent.cli import SMOKE_TASKS
from gogagent.core.actions import MacroAction
from gogagent.core.compiler import MacroCompiler
from gogagent.core.constraint_engine import ConstraintEngine
from gogagent.core.executor import IncrementalExecutor
from gogagent.core.types import TransitionEdge
from gogagent.core.rollout import RolloutEngine
from gogagent.gog.memory import OrganizationGoG
from gogagent.llm.mock import MockLLM
from gogagent.oracle.registry import get_oracle
from gogagent.training import TrainingEpisodeRecorder, TransitionCreditInput


def main() -> None:
    adapter = get_adapter("mmlu")
    task = SMOKE_TASKS["mmlu"]
    constraints = ConstraintEngine()
    compiler = MacroCompiler(adapter, constraints)
    executor = IncrementalExecutor(adapter, MockLLM())
    gog = OrganizationGoG()

    base = adapter.base_graph(task)
    first = executor.execute(base, task)
    gog.add_snapshot(base, adapter.signature(base))
    edited = compiler.compile(base, MacroAction.ATTACH_ANALYST, first.visible_feedback)
    second = executor.execute(edited, task, first)
    gog.add_snapshot(
        edited,
        adapter.signature(edited),
        TransitionEdge(base.graph_id, edited.graph_id, MacroAction.ATTACH_ANALYST),
    )

    summary = TrainingEpisodeRecorder(get_oracle("mmlu")).record(
        gog=gog,
        task=task,
        task_features=adapter.task_features(task),
        output="Answer: B",
        gold="B",
        steps=(
            TransitionCreditInput(
                graph_id=base.graph_id,
                action=MacroAction.ATTACH_ANALYST,
                token_cost=second.token_cost,
                feedback_type=second.visible_feedback.status,
                visible_delta=20.0,
            ),
        ),
    )
    stats = gog.neighbor_stats(base.graph_id, MacroAction.ATTACH_ANALYST)
    payload = json.dumps(
        {
            "summary": summary.to_dict(),
            "experiences": [record.to_dict() for record in gog.experiences],
            "neighbor_stats": stats,
        },
        sort_keys=True,
    )
    assert stats["count"] == 1.0
    assert stats["mean_return"] > 0.0
    assert '"gold"' not in payload

    with TemporaryDirectory() as temp_directory:
        root = Path(temp_directory)
        memory_path = gog.save(root / "training-memory.json")
        assert '"gold"' not in memory_path.read_text(encoding="utf-8")
        loaded = OrganizationGoG.load(memory_path)
        result = RolloutEngine(
            adapter,
            MockLLM(),
            root / "runs",
            gog_memory=loaded,
        ).run(task, "seeded-inference")
        trace_path = Path(result["artifact_directory"]) / "trace.jsonl"
        decisions = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "policy_decision"
        ]
        first_decision = decisions[0]
        assert first_decision["decision"]["action"] == "ATTACH_ANALYST"
        assert first_decision["neighbor_stats"]["ATTACH_ANALYST"]["count"] >= 1.0
    print("training credit: ok")


if __name__ == "__main__":
    main()
