"""Small RL-ready candidate scorer with heuristic cold-start weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gogagent.core.actions import MacroAction
from gogagent.core.types import MacroCandidate, PolicyDecision
from gogagent.gog.memory import OrganizationGoG


class QScorer:
    """Score legal macro actions; replace heuristic weights during RL training."""

    def decide(
        self,
        state: Mapping[str, Any],
        graph_id: str,
        candidates: Sequence[MacroCandidate],
        gog: OrganizationGoG,
    ) -> PolicyDecision:
        feedback = state["observable_feedback"]
        supervisor = state["supervisor_feedback"]
        scores: dict[str, float] = {}
        for candidate in candidates:
            action = candidate.action
            score = self._cold_start_score(action, feedback, supervisor)
            stats = gog.neighbor_stats(graph_id, action)
            score += 0.25 * stats["mean_return"] + 0.2 * stats["success_rate"]
            scores[action.value] = round(score, 6)
        selected = max(candidates, key=lambda candidate: scores[candidate.action.value])
        return PolicyDecision(selected.action, scores, tuple(candidates))

    @staticmethod
    def _cold_start_score(
        action: MacroAction,
        feedback: Mapping[str, Any],
        supervisor: Mapping[str, Any],
    ) -> float:
        issues = feedback.get("issue_codes", ())
        status = feedback.get("status", "unknown")
        confidence = feedback.get("confidence_bucket", "medium")
        disagreement = feedback.get("disagreement_level", "none")
        if action is MacroAction.STOP:
            return 8.0 if status in {"ready", "passed"} and confidence == "high" and not issues else 0.1
        if action is MacroAction.ATTACH_REVISER:
            return 6.0 if issues or disagreement != "none" else 1.0
        if action is MacroAction.ATTACH_CHECKER:
            return 5.0
        if action is MacroAction.ATTACH_ANALYST:
            return 3.0
        if action is MacroAction.ATTACH_ALTERNATIVE:
            return 2.0 if confidence == "low" or disagreement != "none" else 0.5
        return 0.0
