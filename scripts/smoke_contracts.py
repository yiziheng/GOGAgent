"""Local contract checks for the GOGAgent research MVP."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gogagent.adapters.mmlu import MMLUAdapter
from gogagent.adapters.gsm8k import GSM8KAdapter
from gogagent.adapters.humaneval import HumanEvalAdapter
from gogagent.core.actions import MacroAction
from gogagent.core.compiler import MacroCompiler
from gogagent.core.constraint_engine import ConstraintEngine
from gogagent.core.graph_ops import topological_order
from gogagent.core.rollout import RolloutEngine
from gogagent.core.types import MacroCandidate, PolicyDecision, VisibleFeedback
from gogagent.llm.base import LLMBackend, LLMResponse
from gogagent.policy.hierarchical_gnn import HierarchicalGNNPolicy
from gogagent.policy.hierarchical_gnn import TASK_VECTOR_DIM
from gogagent.training.learner import DQNStyleLearner
from gogagent.training.mmlu_runner import _terminal_action_bonus
from gogagent.training.mmlu_runner import _stop_transition_from_trace
from gogagent.training.replay import DenseConstructionReward
from gogagent.training.replay import ReplayTransition


class FakeLLM(LLMBackend):
    name = "fake_llm"

    def generate(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        del prompt, context
        if "Code" in role or role in {"Solver", "SpecAnalyzeCodeGraph"}:
            text = "```python\ndef add(a, b):\n    return a + b\n```"
        elif role in {"DomainAnalyst", "SubjectExpertGraph"}:
            text = "Concept notes: eliminate unsupported options."
        elif role in {"OptionCritic", "Rechecker"}:
            text = "ISSUES: none"
        elif role in {
            "OptionEliminationGraph",
            "SecondOpinionDebateGraph",
            "CritiqueReviseGraph",
            "AdversarialBestAnswerGraph",
            "Adjudicator",
            "Resolver",
            "SecondOpinionSolver",
        }:
            text = "FINAL: A"
        else:
            text = "FINAL: A"
        return LLMResponse(
            text=text,
            prompt_tokens=8,
            completion_tokens=4,
            total_tokens=12,
            model="fake",
            latency_seconds=0.0,
        )


class FirstEditPolicy:
    def decide(self, state, graph, candidates):  # noqa: ANN001
        del state, graph
        chosen = next(
            (candidate for candidate in candidates if candidate.action is not MacroAction.STOP),
            candidates[0],
        )
        return PolicyDecision(
            chosen.action,
            {candidate.action.value: float(index) for index, candidate in enumerate(candidates)},
            tuple(candidates),
            metadata={"policy": "first_non_stop_test_policy"},
        )


def main() -> None:
    adapter = MMLUAdapter()
    constraints = ConstraintEngine(max_steps=3, max_nodes=8)
    compiler = MacroCompiler(adapter, constraints)
    task = {
        "task_id": "smoke-mmlu",
        "subject": "machine_learning",
        "question": "Which option is correct?",
        "options": ["A is correct", "B is wrong", "C is wrong", "D is wrong"],
    }
    task_features = adapter.task_features(task)
    assert len(task_features["task_vector"]) == TASK_VECTOR_DIM, "MMLU TaskEncoder is not policy-visible"
    assert "keyword_flags" in task_features, "MMLU TaskEncoder did not emit semantic flags"
    feedback = VisibleFeedback(
        status="needs_review",
        confidence_bucket="low",
        disagreement_level="medium",
        issue_codes=("unchecked_answer",),
    )

    base = adapter.base_graph(task)
    upgraded = compiler.compile(base, MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT, feedback)
    constraints.validate(upgraded)
    assert any(node.node_kind == "graph" for node in upgraded.nodes), "upgrade did not create GraphAgent"
    graph_node = next(node for node in upgraded.nodes if node.node_kind == "graph")
    assert graph_node.internal_nodes and graph_node.internal_edges, "GraphAgent is missing internals"
    assert topological_order(upgraded), "upgraded GoG is not topologically sortable"

    downgraded = compiler.compile(upgraded, MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC, feedback)
    assert all(node.node_kind == "atomic" for node in downgraded.nodes), "downgrade left GraphAgent behind"

    candidates = constraints.legal_candidates(base, feedback)
    assert any(
        candidate.action is MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT for candidate in candidates
    ), "upgrade action is not legal"
    assert constraints.legal_candidates(upgraded, feedback), "upgraded graph has no legal actions"
    assert any(
        candidate.action is MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC
        for candidate in constraints.legal_candidates(upgraded, feedback)
    ), "downgrade action is not legal"

    policy = HierarchicalGNNPolicy(epsilon=0.0)
    state = {
        "task_features": task_features,
        "observable_feedback": feedback.to_dict(),
        "used_tokens": 0,
        "remaining_tokens": 256,
    }
    decision = policy.decide(state, base, candidates)
    assert decision.metadata["policy"] == "torch_hierarchical_gcn_dqn", "policy is not torch GCN/RL"
    assert decision.metadata["framework"] == "torch", "policy did not report torch framework"
    assert "q_values" in decision.metadata and "legal_mask" in decision.metadata
    assert "action_priors" in decision.metadata, "MMLU template prior is not exposed"
    assert any(decision.metadata["action_priors"].values()), "MMLU template prior is inactive"
    assert MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH.value in decision.metadata["action_space"], (
        "adversarial best-answer action is missing from GNN action space"
    )
    embedding = policy.state_embedding(base, state)
    before = policy.action_head.weight.detach().clone()
    loss = policy.td_update(state_embedding=embedding, action=decision.action, target=1.0)
    after = policy.action_head.weight.detach().clone()
    assert loss >= 0.0 and not before.equal(after), "TD update did not change torch policy weights"

    llm = FakeLLM()
    before_execution = adapter.execute(base, task, llm)
    after_execution = adapter.execute(upgraded, task, llm, before_execution)
    dense = DenseConstructionReward().score(base, upgraded, before_execution, after_execution)
    assert isinstance(dense.reward, float), "dense reward did not produce a scalar"
    transition = ReplayTransition(
        graph_id=base.graph_id,
        next_graph_id=upgraded.graph_id,
        action=MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT,
        reward=dense.reward,
        done=False,
        state=state,
        next_state=state,
        action_mask={candidate.action.value: True for candidate in candidates},
        next_action_mask={candidate.action.value: True for candidate in candidates},
        dense_reward=dense,
    )
    learner_result = DQNStyleLearner(policy).train_one(
        graph=base,
        next_graph=upgraded,
        transition=transition,
    )
    assert learner_result.loss >= 0.0, "learner did not return a valid TD loss"
    assert _terminal_action_bonus(1.0, 0, 2) > 0.0, "correct terminal reward did not boost actions"
    assert _terminal_action_bonus(0.0, 0, 2) < 0.0, "incorrect terminal reward did not penalize actions"

    option_graph = compiler.compile(base, MacroAction.ADD_OPTION_ELIMINATION_GRAPH, feedback)
    adversarial_candidates = constraints.legal_candidates(option_graph, after_execution.visible_feedback)
    assert any(
        candidate.action is MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH
        for candidate in adversarial_candidates
    ), "adversarial best-answer graph is not legal after an MMLU graph module"
    risky_state = {
        **state,
        "observable_feedback": {
            **feedback.to_dict(),
            "signals": {
                "parsed_options": {
                    "solver": "A",
                    "option_elimination_graph": "A",
                    "decompose_solve_verify_graph": "A",
                },
                "parse_failed_count": 0,
                "consensus_fraction": 1.0,
                "majority_margin": 1.0,
                "answer_vote_entropy": 0.0,
                "rationale_similarity": 0.45,
                "contradiction_count": 0,
                "evidence_coverage": 0.85,
                "answer_changed": False,
                "adversarial_checked": False,
            },
        },
    }
    risky_decision = policy.decide(risky_state, option_graph, adversarial_candidates)
    risky_priors = risky_decision.metadata["action_priors"]
    assert (
        risky_priors[MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH.value]
        > risky_priors[MacroAction.STOP.value]
    ), "label-blind consensus risk did not raise adversarial prior above STOP"
    adversarial_graph = compiler.compile(
        option_graph,
        MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH,
        after_execution.visible_feedback,
    )
    adversarial_execution = adapter.execute(adversarial_graph, task, llm, after_execution)
    assert "adversarial_best_answer_graph" in adversarial_execution.node_outputs, (
        "adversarial graph did not execute"
    )
    assert adversarial_execution.visible_feedback.signals["adversarial_checked"] is True

    stop_transition = _stop_transition_from_trace(
        [
            {"event": "snapshot", "graph": base.to_dict(), "execution": before_execution.to_dict()},
            {
                "event": "policy_decision",
                "state": state,
                "decision": {"action": MacroAction.STOP.value},
                "legal_action_mask": {MacroAction.STOP.value: True},
            },
            {"event": "terminal", "src_graph_id": base.graph_id, "action": "STOP"},
        ],
        terminal_reward=0.0,
    )
    assert stop_transition is not None and stop_transition.action is MacroAction.STOP
    assert stop_transition.done and stop_transition.reward < 0.0, (
        "wrong terminal STOP did not become a negative training transition"
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = RolloutEngine(
            adapter,
            llm,
            artifact_root=Path(tmp),
            constraints=ConstraintEngine(max_steps=2, max_nodes=8),
            policy=FirstEditPolicy(),
        ).run(task, episode_id="contract", artifact_directory=Path(tmp) / "rollout")
        trace_path = Path(result["artifact_directory"]) / "trace.jsonl"
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(record.get("event") == "replay_transition" for record in records), (
            "rollout did not emit replay_transition"
        )
        snapshot_paths = list((Path(result["artifact_directory"]) / "snapshots").glob("*.json"))
        assert snapshot_paths, "rollout did not export visible graph snapshots"
        assert any(
            any(node.get("node_kind") == "graph" for node in json.loads(path.read_text())["nodes"])
            for path in snapshot_paths
        ), "visible snapshots do not contain GraphAgent nodes"

    _check_domain_graphagent_compile(
        GSM8KAdapter(),
        {
            "task_id": "smoke-gsm8k",
            "question": "Tom has 2 apples and buys 3 more. How many apples?",
        },
        MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH,
    )
    _check_domain_graphagent_compile(
        HumanEvalAdapter(),
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"",
            "entry_point": "add",
        },
        MacroAction.ADD_SPEC_ANALYZE_CODE_GRAPH,
    )


def _check_domain_graphagent_compile(adapter, task, action):  # noqa: ANN001
    constraints = ConstraintEngine(max_steps=3, max_nodes=8)
    compiler = MacroCompiler(adapter, constraints)
    feedback = VisibleFeedback(
        status="needs_review",
        confidence_bucket="low",
        disagreement_level="medium",
        issue_codes=("unchecked_answer",),
    )
    base = adapter.base_graph(task)
    graph = compiler.compile(base, action, feedback)
    constraints.validate(graph)
    assert any(node.node_kind == "graph" for node in graph.nodes), (
        f"{adapter.name} action {action.value} did not create GraphAgent"
    )
    downgraded = compiler.compile(graph, MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC, feedback)
    constraints.validate(downgraded)
    assert all(node.node_kind == "atomic" for node in downgraded.nodes), (
        f"{adapter.name} downgrade did not collapse GraphAgent"
    )


if __name__ == "__main__":
    main()
