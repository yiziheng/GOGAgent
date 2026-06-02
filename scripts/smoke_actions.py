"""Compile and execute every shared macro branch for every adapter."""

from __future__ import annotations

from dataclasses import replace

from gogagent.adapters.registry import get_adapter
from gogagent.cli import SMOKE_TASKS
from gogagent.core.actions import MacroAction
from gogagent.core.compiler import MacroCompiler
from gogagent.core.constraint_engine import ConstraintEngine
from gogagent.core.executor import IncrementalExecutor
from gogagent.llm.mock import MockLLM


def main() -> None:
    constraints = ConstraintEngine()
    for domain, task in SMOKE_TASKS.items():
        adapter = get_adapter(domain)
        compiler = MacroCompiler(adapter, constraints)
        executor = IncrementalExecutor(adapter, MockLLM())
        base = adapter.base_graph(task)
        first = executor.execute(base, task)

        linear_graph = base
        linear_result = first
        for action in (
            MacroAction.ATTACH_ANALYST,
            MacroAction.ATTACH_CHECKER,
            MacroAction.ATTACH_REVISER,
        ):
            feedback = linear_result.visible_feedback
            if action is MacroAction.ATTACH_REVISER:
                feedback = replace(
                    feedback,
                    status="needs_review",
                    issue_codes=(*feedback.issue_codes, "mock_visible_issue"),
                )
            linear_graph = compiler.compile(
                linear_graph,
                action,
                feedback,
            )
            linear_result = executor.execute(linear_graph, task, linear_result)

        alternative_graph = compiler.compile(
            base,
            MacroAction.ATTACH_ALTERNATIVE,
            replace(first.visible_feedback, confidence_bucket="low"),
        )
        executor.execute(alternative_graph, task, first)
        print(f"{domain}: shared macro branches ok")


if __name__ == "__main__":
    main()
