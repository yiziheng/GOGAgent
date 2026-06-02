"""Fixed label-blind supervisor control plane."""

from __future__ import annotations

from gogagent.core.types import ExecutionResult, SupervisorFeedback


class SupervisorAgent:
    """Summarize observable execution state without consulting any gold label."""

    def summarize(
        self,
        execution: ExecutionResult,
        used_tokens: int,
        token_budget: int,
    ) -> SupervisorFeedback:
        feedback = execution.visible_feedback
        remaining_ratio = max(token_budget - used_tokens, 0) / max(token_budget, 1)
        budget_risk = "high" if remaining_ratio < 0.2 else "low"
        should_stop = (
            feedback.status in {"ready", "passed"}
            and feedback.confidence_bucket == "high"
            and not feedback.issue_codes
        )
        return SupervisorFeedback(
            status=feedback.status,
            confidence_bucket=feedback.confidence_bucket,
            disagreement_level=feedback.disagreement_level,
            unresolved_issue_codes=feedback.issue_codes,
            budget_risk=budget_risk,
            stop_advice="stop" if should_stop else "continue",
        )
