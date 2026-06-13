"""DeepSeek teacher client for BC action trajectory proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.actions.registry import ACTION_ORDER, teacher_action_descriptions
from gogagent.graph.schema import Graph
from gogagent.llm import LLMClient


DEFAULT_TEACHER_STYLES: tuple[str, ...] = (
    "failure_repair",
)

TEACHER_PROMPT = (
    "You are a behavior-cloning teacher for a Graph-of-Graphs multi-agent "
    "construction policy. Given a task, choose only graph-construction action "
    "names from the provided action space. Do not output graph JSON. Do not "
    "invent actions. Return exactly one JSON object with keys: actions, reason, "
    "difficulty, expected_graph_shape. The actions value must be a list of "
    "strings and should end with STOP. The initial graph is already executable "
    "and contains one SolverAgent, so STOP immediately means using the "
    "Solver-only graph. In the failure-repair stage, the SolverAgent answer is "
    "known wrong, so do not output STOP-only. Respect all legality constraints. "
    "UP acts on the last top-level atomic node by default."
)

BC_DECISION_RULE = (
    "BC decision rule:\n"
    "- The initial SolverAgent has already attempted the problem.\n"
    "- If that SolverAgent answer is correct, the pipeline emits [STOP] without "
    "calling this teacher.\n"
    "- This teacher is called only when the SolverAgent answer is wrong.\n"
    "- For a wrong SolverAgent answer, choose the simplest repair graph that is "
    "likely to correct this specific failure. Do not return STOP-only."
)

ACTION_RESCUE_GUIDE = (
    "Action rescue guide for known wrong SolverAgent answers:\n"
    "- ADD_TASK_BRIEF: helps when the question is long, dense, or likely misread; "
    "use it to clarify what the problem is asking.\n"
    "- ADD_PLAN_SKETCH: helps with multi-step reasoning, mathematical/logical "
    "decomposition, physics/legal reasoning, or cases where the solver jumped "
    "directly to an answer without a plan.\n"
    "- ADD_FORMAT_VERIFIER: helps when the reasoning may be mostly right but the "
    "final answer label or required output format may be wrong. It usually does "
    "not fix deep conceptual mistakes by itself.\n"
    "- UP: upgrades the last atomic node into a type-specific subgraph. Solver UP "
    "adds a light PlanSketch before Solver. Use it only when the task likely needs "
    "explicit multi-step reasoning; for factual or definition questions, prefer "
    "STOP or a simpler context action.\n"
    "- ADD_ADVERSARIAL_JUDGE: helps with tempting distractors, ambiguity, subtle "
    "misconceptions, or a confident but wrong SolverAgent answer. It is not the "
    "best first repair for simple arithmetic, pure format errors, or simple "
    "misreading."
)

LEGALITY_GUIDE = (
    "Legality constraints:\n"
    "- max_actions is the maximum number of actions in the returned trajectory.\n"
    "- max_nodes is the maximum total number of top-level plus nested graph nodes.\n"
    "- max_depth is the maximum graph nesting depth.\n"
    "- UP targets the last top-level atomic node by default.\n"
    "- UP is illegal if the target node is already a subgraph.\n"
    "- ADD_PLAN_SKETCH is illegal if a planner/plan sketch already exists.\n"
    "- ADD_FORMAT_VERIFIER is illegal if a verifier already exists.\n"
    "- Any expansion is illegal if max_depth or max_nodes would be exceeded.\n"
    "- STOP is always legal and should be used once the current graph is sufficient."
)

FAILURE_TYPES: tuple[str, ...] = (
    "misread_question",
    "multi_step_reasoning",
    "format_or_label_error",
    "conceptual_confusion",
    "tempting_distractor",
    "overconfident_wrong_answer",
    "unknown",
)

STYLE_GUIDES: dict[str, str] = {
    "failure_repair": (
        "Style: failure_repair. Choose the repair trajectory that most directly "
        "targets the observed SolverAgent failure. Prefer a compact repair graph "
        "over a broad or decorative one."
    ),
    "accuracy_first": (
        "Style: accuracy_first. For the known wrong SolverAgent answer, prefer "
        "the repair most likely to recover the correct answer, even if it needs "
        "more structure than the minimal repair."
    ),
    "cost_aware": (
        "Style: cost_aware. For the known wrong SolverAgent answer, prefer the "
        "smallest repair that plausibly targets the failure. Do not choose "
        "STOP-only, because the SolverAgent answer is already known wrong."
    ),
    "hard_case_adversarial": (
        "Style: hard_case_adversarial. For the known wrong SolverAgent answer, "
        "use ADD_ADVERSARIAL_JUDGE only if the failure looks like a tempting "
        "distractor, ambiguity, subtle misconception, or overconfident wrong "
        "answer. Otherwise choose a planner, format verifier, or UP according "
        "to the observed failure."
    ),
}


@dataclass(frozen=True)
class TeacherActionProposal:
    """One teacher-proposed action sequence before local replay validation."""

    actions: tuple[ActionName, ...]
    reason: str
    difficulty: str | None
    failure_type: str | None
    expected_graph_shape: str | None
    raw_response: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable proposal."""

        return {
            "actions": [action.value for action in self.actions],
            "reason": self.reason,
            "difficulty": self.difficulty,
            "failure_type": self.failure_type,
            "expected_graph_shape": self.expected_graph_shape,
            "raw_response": dict(self.raw_response),
        }


@dataclass(frozen=True)
class TeacherTrajectoryClient:
    """Request action-only BC trajectories from a strict JSON LLM client."""

    llm_client: LLMClient
    max_actions: int = 6

    def propose(
        self,
        *,
        task: Mapping[str, Any],
        style: str,
        initial_graph: Graph,
        constraints: ActionConstraints,
        solver_probe: Mapping[str, Any],
    ) -> TeacherActionProposal:
        """Ask the teacher for one action sequence."""

        if bool(solver_probe.get("correct")):
            raise ValueError("repair teacher should not be called for a correct solver probe")
        response = self.llm_client.chat_json(
            role="bc_teacher",
            prompt=TEACHER_PROMPT,
            payload={
                "task": dict(task),
                "style": style,
                "max_actions": self.max_actions,
                "action_space": [action.value for action in ACTION_ORDER],
                "action_descriptions": teacher_action_descriptions(),
                "bc_decision_rule": BC_DECISION_RULE,
                "solver_probe": dict(solver_probe),
                "action_rescue_guide": ACTION_RESCUE_GUIDE,
                "legality_guide": LEGALITY_GUIDE,
                "failure_types": list(FAILURE_TYPES),
                "style_guide": style_guide(style),
                "constraints": {
                    "max_depth": constraints.max_depth,
                    "max_nodes": constraints.max_nodes,
                    "max_actions": self.max_actions,
                    "up_target": "last top-level atomic node",
                    "stop_semantics": "STOP terminates graph construction",
                },
                "initial_graph": initial_graph.to_dict(),
            },
            response_schema=teacher_response_schema(),
            instruction=(
                "Return exactly one JSON object with the requested action trajectory. "
                "Do not wrap it in markdown. Do not include GraphMessage fields unless "
                "they are part of the requested schema."
            ),
        )
        data = dict(response.data)
        actions = _parse_actions(data.get("actions"), max_actions=self.max_actions)
        if actions[0] == ActionName.STOP:
            raise ValueError(
                "repair teacher returned a STOP-first trajectory for a known wrong solver answer"
            )
        failure_type = _optional_str(data.get("failure_type"))
        if failure_type is not None and failure_type not in FAILURE_TYPES:
            raise ValueError(
                f"unknown failure_type {failure_type!r}; allowed: {', '.join(FAILURE_TYPES)}"
            )
        return TeacherActionProposal(
            actions=actions,
            reason=str(data.get("reason", "")),
            difficulty=_optional_str(data.get("difficulty")),
            failure_type=failure_type,
            expected_graph_shape=_optional_str(data.get("expected_graph_shape")),
            raw_response={
                **data,
                "_llm": response.to_dict(),
            },
        )


def _parse_actions(value: Any, *, max_actions: int) -> tuple[ActionName, ...]:
    if not isinstance(value, list):
        raise ValueError("teacher response must include actions as a list")
    if not value:
        raise ValueError("teacher response actions must not be empty")
    if len(value) > max_actions:
        raise ValueError(f"teacher returned too many actions: {len(value)}>{max_actions}")
    actions: list[ActionName] = []
    for index, item in enumerate(value, start=1):
        try:
            actions.append(ActionName(str(item)))
        except ValueError as exc:
            allowed = ", ".join(action.value for action in ACTION_ORDER)
            raise ValueError(
                f"teacher action #{index} is unknown: {item!r}; allowed: {allowed}"
            ) from exc
    return tuple(actions)


def teacher_response_schema() -> dict[str, Any]:
    """Return the strict JSON schema described to the teacher model."""

    return {
        "actions": ["ACTION_NAME", "..."],
        "reason": "short explanation",
        "difficulty": "easy | medium | hard",
        "failure_type": "one of the provided failure_types",
        "expected_graph_shape": "short natural-language graph summary",
    }


def style_guide(style: str) -> str:
    """Return the teacher policy guide for a named style."""

    try:
        return STYLE_GUIDES[style]
    except KeyError as exc:
        allowed = ", ".join(sorted(STYLE_GUIDES))
        raise ValueError(f"unknown BC teacher style {style!r}; allowed: {allowed}") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
