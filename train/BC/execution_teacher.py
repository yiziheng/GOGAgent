"""Execution-verified BC trajectory teacher.

This teacher does not ask an LLM to invent graph actions. It tries a curated
list of meaningful construction methods from cheap to expensive and returns the
first method whose executed graph answers the train-time example correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.graph.factory import make_initial_graph
from gogagent.graph.schema import Graph, GraphMessage
from gogagent.llm import AgentContext, LLMClient
from gogagent.reward.oracle import OracleResult, score_answer
from train.BC.teacher import TeacherActionProposal
from train.BC.trajectory import TrajectoryBuildResult, build_trajectory


ActionMethod = tuple[ActionName, ...]

DEFAULT_EXECUTION_METHODS: tuple[ActionMethod, ...] = (
    (ActionName.STOP,),
    (ActionName.ADD_TASK_BRIEF, ActionName.STOP),
    (ActionName.ADD_PLAN_SKETCH, ActionName.STOP),
    (ActionName.ADD_ADVERSARIAL_JUDGE, ActionName.STOP),
    (ActionName.ADD_ADVERSARIAL_JUDGE, ActionName.UP, ActionName.STOP),
    (ActionName.ADD_PLAN_SKETCH, ActionName.ADD_ADVERSARIAL_JUDGE, ActionName.STOP),
    (
        ActionName.ADD_TASK_BRIEF,
        ActionName.ADD_PLAN_SKETCH,
        ActionName.ADD_ADVERSARIAL_JUDGE,
        ActionName.STOP,
    ),
)


@dataclass(frozen=True)
class ExecutionAttempt:
    """One tried graph-construction method and its execution result."""

    index: int
    actions: ActionMethod
    build: TrajectoryBuildResult
    output: GraphMessage | None
    oracle_result: OracleResult | None
    error: str | None = None

    @property
    def correct(self) -> bool:
        """Return whether this attempted method answered correctly."""

        return bool(self.oracle_result and self.oracle_result.correct)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable attempt summary."""

        return {
            "index": self.index,
            "actions": [action.value for action in self.actions],
            "valid": self.build.valid,
            "correct": self.correct,
            "output": self.output.to_dict() if self.output is not None else None,
            "oracle_result": (
                self.oracle_result.to_dict()
                if self.oracle_result is not None
                else None
            ),
            "error": self.error,
            "invalid_steps": [dict(step) for step in self.build.invalid_steps],
        }


@dataclass(frozen=True)
class ExecutionTeacherProposal:
    """First-success proposal plus diagnostics from all tried methods."""

    proposal: TeacherActionProposal
    build: TrajectoryBuildResult
    attempts: tuple[ExecutionAttempt, ...]
    output: GraphMessage
    oracle_result: OracleResult

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable proposal summary."""

        return {
            "proposal": self.proposal.to_dict(),
            "build": self.build.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "output": self.output.to_dict(),
            "oracle_result": self.oracle_result.to_dict(),
        }


class ExecutionTeacherError(RuntimeError):
    """Raised when no curated method answers an example correctly."""

    def __init__(self, message: str, attempts: Sequence[ExecutionAttempt]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class ExecutionVerifiedTeacher:
    """Try curated graph methods and choose the first correct trajectory."""

    llm_client: LLMClient
    methods: tuple[ActionMethod, ...] = DEFAULT_EXECUTION_METHODS
    require_stop: bool = True

    def propose(
        self,
        *,
        task: Mapping[str, Any],
        dataset: str,
        gold: Any,
        constraints: ActionConstraints,
        graph_id_prefix: str = "execution_teacher",
    ) -> ExecutionTeacherProposal:
        """Return the first action method whose graph executes correctly."""

        attempts: list[ExecutionAttempt] = []
        for index, actions in enumerate(self.methods, start=1):
            initial_graph = make_initial_graph(
                graph_id=f"{graph_id_prefix}_method_{index}"
            )
            build = build_trajectory(
                initial_graph=initial_graph,
                actions=actions,
                constraints=constraints,
                require_stop=self.require_stop,
            )
            if not build.valid:
                attempts.append(
                    ExecutionAttempt(
                        index=index,
                        actions=actions,
                        build=build,
                        output=None,
                        oracle_result=None,
                        error="invalid construction trajectory",
                    )
                )
                continue

            try:
                final_graph = Graph.from_dict(build.final_graph)
                output = final_graph.execute(
                    task,
                    context=AgentContext(llm_client=self.llm_client),
                )
                oracle_result = score_answer(dataset, task, output, gold=gold)
            except Exception as error:  # noqa: BLE001 - failed methods are simply not labels.
                attempts.append(
                    ExecutionAttempt(
                        index=index,
                        actions=actions,
                        build=build,
                        output=None,
                        oracle_result=None,
                        error=str(error),
                    )
                )
                continue

            attempt = ExecutionAttempt(
                index=index,
                actions=actions,
                build=build,
                output=output,
                oracle_result=oracle_result,
            )
            attempts.append(attempt)
            if oracle_result.correct:
                return ExecutionTeacherProposal(
                    proposal=_proposal_from_attempt(attempt),
                    build=build,
                    attempts=tuple(attempts),
                    output=output,
                    oracle_result=oracle_result,
                )

        raise ExecutionTeacherError(
            (
                "no execution-verified BC method answered correctly; "
                f"tried {len(attempts)} methods"
            ),
            attempts,
        )


def normalize_methods(methods: Sequence[Sequence[ActionName | str]]) -> tuple[ActionMethod, ...]:
    """Normalize a method array like [[STOP], [ADD_PLAN_SKETCH, STOP]]."""

    normalized: list[ActionMethod] = []
    for method in methods:
        actions = tuple(
            action if isinstance(action, ActionName) else ActionName(str(action))
            for action in method
        )
        if not actions:
            raise ValueError("execution teacher methods must not contain empty sequences")
        normalized.append(actions)
    if not normalized:
        raise ValueError("execution teacher requires at least one method")
    return tuple(normalized)


def _proposal_from_attempt(attempt: ExecutionAttempt) -> TeacherActionProposal:
    if attempt.oracle_result is None:
        raise ValueError("cannot build proposal from an unscored attempt")
    return TeacherActionProposal(
        actions=attempt.actions,
        reason=(
            "First execution-verified method that matched the train-time gold "
            f"answer at method index {attempt.index}."
        ),
        difficulty="execution_verified",
        failure_type=None,
        expected_graph_shape=" -> ".join(action.value for action in attempt.actions),
        raw_response={
            "source": "execution_verified_teacher",
            "method_index": attempt.index,
            "oracle_result": attempt.oracle_result.to_dict(),
        },
    )
