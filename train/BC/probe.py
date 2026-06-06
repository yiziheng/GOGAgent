"""Single-solver probe helpers for failure-driven BC trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gogagent.datasets import DatasetExample
from gogagent.graph.schema import Graph, GraphMessage
from gogagent.llm.client import AgentContext, LLMClient
from gogagent.reward.oracle import OracleResult, score_answer


@dataclass(frozen=True)
class SolverProbeResult:
    """The initial Solver-only attempt used to decide whether repair is needed."""

    output: GraphMessage
    oracle_result: OracleResult
    llm_calls: tuple[Mapping[str, Any], ...]

    @property
    def correct(self) -> bool:
        """Return whether the Solver-only answer matched the gold answer."""

        return self.oracle_result.correct

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable probe record."""

        return {
            "correct": self.correct,
            "output": self.output.to_dict(),
            "oracle_result": self.oracle_result.to_dict(),
            "llm_calls": [dict(call) for call in self.llm_calls],
        }


def run_solver_probe(
    *,
    example: DatasetExample,
    initial_graph: Graph,
    llm_client: LLMClient,
) -> SolverProbeResult:
    """Execute the initial Solver-only graph and score it against train-time gold."""

    context = AgentContext(llm_client=llm_client)
    output = initial_graph.execute(example.public_task, context=context)
    oracle_result = score_answer(
        example.dataset,
        example.public_task,
        output,
        gold=example.gold,
    )
    return SolverProbeResult(
        output=output,
        oracle_result=oracle_result,
        llm_calls=tuple(context.llm_calls),
    )
