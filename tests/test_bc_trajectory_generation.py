#!/usr/bin/env python3
"""BC teacher trajectory generation smoke checks without live LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.datasets import DatasetExample
from gogagent.graph.schema import GraphMessage
from gogagent.llm import LLMClient, LLMJsonResponse, LLMUsage
from gogagent.reward.oracle import OracleResult
from train.BC.generate_trajectories import generate_one, step_rows_from_trajectory
from gogagent.graph.factory import make_initial_graph
from train.BC.probe import SolverProbeResult
from train.BC.teacher import TeacherTrajectoryClient
from train.BC.trajectory import build_trajectory


class ScriptedTeacherClient(LLMClient):
    """Scripted JSON teacher for BC smoke checks."""

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        assert role == "bc_teacher"
        assert "ADD_PLAN_SKETCH" in payload["action_descriptions"]
        assert "UP acts on the last top-level atomic node" in prompt
        assert "Solver-only graph" in prompt
        assert "reward_guide" not in payload
        assert payload["solver_probe"]["correct"] is False
        assert "SolverAgent answer is wrong" in payload["bc_decision_rule"]
        assert "Do not return STOP-only" in payload["bc_decision_rule"]
        assert "ADD_PLAN_SKETCH" in payload["action_rescue_guide"]
        assert "tempting distractors" in payload["action_rescue_guide"]
        assert "UP is illegal if the target node is already a subgraph" in payload["legality_guide"]
        assert "multi_step_reasoning" in payload["failure_types"]
        assert payload["style"] in payload["style_guide"]
        if payload["style"] == "hard_case_adversarial":
            assert "tempting distractor" in payload["style_guide"]
        assert payload["constraints"]["max_actions"] == 6
        assert response_schema is not None
        assert "actions" in response_schema
        assert "failure_type" in response_schema
        assert "role" not in response_schema
        assert instruction is not None
        data = {
            "actions": ["ADD_PLAN_SKETCH", "STOP"],
            "reason": "The solver jumped to an answer; add a plan sketch before solving.",
            "difficulty": "medium",
            "failure_type": "multi_step_reasoning",
            "expected_graph_shape": "planner -> solver",
        }
        return LLMJsonResponse(
            data=data,
            raw_text=json.dumps(data),
            model="scripted-bc-teacher",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_seconds=0.0,
        )


class FailingTeacherClient(LLMClient):
    """Teacher that simulates a provider/schema failure."""

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        del role, prompt, payload, response_schema, instruction
        raise RuntimeError("simulated teacher failure")


class StopOnlyRepairClient(LLMClient):
    """Teacher that returns an invalid STOP-only repair for a wrong probe."""

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        del role, prompt, payload, response_schema, instruction
        data = {
            "actions": ["STOP"],
            "reason": "invalid stop-only repair",
            "difficulty": "easy",
            "failure_type": "unknown",
            "expected_graph_shape": "solver only",
        }
        return LLMJsonResponse(
            data=data,
            raw_text=json.dumps(data),
            model="scripted-stop-only-teacher",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_seconds=0.0,
        )


class StopFirstRepairClient(LLMClient):
    """Teacher that returns STOP followed by another action."""

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        del role, prompt, payload, response_schema, instruction
        data = {
            "actions": ["STOP", "ADD_PLAN_SKETCH"],
            "reason": "invalid stop-first repair",
            "difficulty": "easy",
            "failure_type": "unknown",
            "expected_graph_shape": "solver only then impossible tail",
        }
        return LLMJsonResponse(
            data=data,
            raw_text=json.dumps(data),
            model="scripted-stop-first-teacher",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_seconds=0.0,
        )


def make_probe(*, correct: bool) -> SolverProbeResult:
    prediction = "A" if correct else "B"
    return SolverProbeResult(
        output=GraphMessage(
            sender="solver",
            role="solver",
            content=f"Scripted solver answer: {prediction}",
            answer=prediction,
        ),
        oracle_result=OracleResult(
            correct=correct,
            dataset="mmlu",
            prediction=prediction,
            gold="A",
            reason="correct" if correct else "wrong",
        ),
        llm_calls=(
            {
                "node_id": "solver",
                "node_name": "SolverAgent",
                "role": "solver",
                "llm": {"model": "scripted-solver"},
            },
        ),
    )


def main() -> None:
    constraints = ActionConstraints(max_depth=2, max_nodes=8)
    initial_graph = make_initial_graph()
    teacher = TeacherTrajectoryClient(llm_client=ScriptedTeacherClient(), max_actions=6)
    wrong_probe = make_probe(correct=False)
    correct_probe = make_probe(correct=True)
    proposal = teacher.propose(
        task={"task_id": "toy", "question": "Which option is best?"},
        style="accuracy_first",
        initial_graph=initial_graph,
        constraints=constraints,
        solver_probe=wrong_probe.to_dict(),
    )
    assert proposal.actions == (
        ActionName.ADD_PLAN_SKETCH,
        ActionName.STOP,
    )
    assert proposal.failure_type == "multi_step_reasoning"

    cost_proposal = teacher.propose(
        task={"task_id": "toy-cost", "question": "Which option is best?"},
        style="cost_aware",
        initial_graph=initial_graph,
        constraints=constraints,
        solver_probe=wrong_probe.to_dict(),
    )
    assert cost_proposal.actions == proposal.actions

    hard_case_proposal = teacher.propose(
        task={"task_id": "toy-hard", "question": "Which option is best?"},
        style="hard_case_adversarial",
        initial_graph=initial_graph,
        constraints=constraints,
        solver_probe=wrong_probe.to_dict(),
    )
    assert hard_case_proposal.actions == proposal.actions

    build = build_trajectory(
        initial_graph=initial_graph,
        actions=proposal.actions,
        constraints=constraints,
    )
    assert build.valid
    assert len(build.steps) == 2
    assert build.steps[0].target_action == ActionName.ADD_PLAN_SKETCH

    stop_tail_build = build_trajectory(
        initial_graph=initial_graph,
        actions=(ActionName.STOP, ActionName.ADD_PLAN_SKETCH),
        constraints=constraints,
    )
    assert not stop_tail_build.valid
    assert stop_tail_build.invalid_steps[0]["reason"] == "trajectory contains actions after STOP"

    row = {
        "trajectory_id": "toy::accuracy_first",
        "task_id": "toy",
        "dataset": "mmlu",
        "task": {"task_id": "toy", "question": "Which option is best?"},
        "style": "accuracy_first",
        "solver_probe": wrong_probe.to_dict(),
        "steps": [step.to_dict() for step in build.steps],
    }
    step_rows = step_rows_from_trajectory(row)
    assert len(step_rows) == 2
    assert step_rows[0]["target_action"] == "ADD_PLAN_SKETCH"
    assert step_rows[0]["solver_probe_output"]["answer"] == "B"
    assert "oracle_result" not in step_rows[0]["solver_probe_output"]

    correct_row = generate_one(
        example=DatasetExample(
            dataset="mmlu",
            public_task={"task_id": "toy-correct", "question": "Which option is best?"},
            gold="A",
        ),
        example_index=1,
        style="solver_probe_correct",
        initial_graph=make_initial_graph(),
        constraints=constraints,
        teacher=TeacherTrajectoryClient(llm_client=FailingTeacherClient(), max_actions=6),
        solver_probe=correct_probe,
        require_stop=True,
    )
    assert correct_row["actions"] == ["STOP"]
    assert correct_row["valid"]
    assert correct_row["proposal"]["raw_response"]["source"] == "solver_probe_correct"

    failing_teacher = TeacherTrajectoryClient(
        llm_client=FailingTeacherClient(), max_actions=6
    )
    failed = False
    try:
        generate_one(
            example=DatasetExample(
                dataset="mmlu",
                public_task={"task_id": "toy-fail", "question": "Which option is best?"},
                gold="A",
            ),
            example_index=1,
            style="accuracy_first",
            initial_graph=make_initial_graph(),
            constraints=constraints,
            teacher=failing_teacher,
            solver_probe=wrong_probe,
            require_stop=True,
        )
    except RuntimeError as exc:
        failed = "simulated teacher failure" in str(exc)
    assert failed, "teacher failures must abort instead of becoming invalid rows"

    stop_only_teacher = TeacherTrajectoryClient(
        llm_client=StopOnlyRepairClient(), max_actions=6
    )
    failed = False
    try:
        stop_only_teacher.propose(
            task={"task_id": "toy-stop", "question": "Which option is best?"},
            style="failure_repair",
            initial_graph=initial_graph,
            constraints=constraints,
            solver_probe=wrong_probe.to_dict(),
        )
    except ValueError as exc:
        failed = "STOP-first" in str(exc)
    assert failed, "wrong solver probes must not accept STOP-only repair trajectories"

    stop_first_teacher = TeacherTrajectoryClient(
        llm_client=StopFirstRepairClient(), max_actions=6
    )
    failed = False
    try:
        stop_first_teacher.propose(
            task={"task_id": "toy-stop-first", "question": "Which option is best?"},
            style="failure_repair",
            initial_graph=initial_graph,
            constraints=constraints,
            solver_probe=wrong_probe.to_dict(),
        )
    except ValueError as exc:
        failed = "STOP-first" in str(exc)
    assert failed, "wrong solver probes must not accept STOP-first repair trajectories"
    print("BC trajectory generation smoke passed")


if __name__ == "__main__":
    main()
