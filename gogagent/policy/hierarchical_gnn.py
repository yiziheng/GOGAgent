"""Torch hierarchical GCN + RL action policy."""

from __future__ import annotations

from hashlib import sha256
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from gogagent.core.actions import MacroAction
from gogagent.core.graph_ops import graph_depth
from gogagent.core.types import MacroCandidate, OrgGraphSnapshot, PolicyDecision


TASK_VECTOR_DIM = 25
RISK_FEATURE_DIM = 9
GRAPH_FEATURE_DIM = 8 + TASK_VECTOR_DIM + RISK_FEATURE_DIM
MMLU_PRIOR_WEIGHT = 0.75

ACTION_SPACE: tuple[MacroAction, ...] = (
    MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT,
    MacroAction.ADD_SUBJECT_EXPERT_GRAPH,
    MacroAction.ADD_OPTION_ELIMINATION_GRAPH,
    MacroAction.ADD_SECOND_OPINION_DEBATE_GRAPH,
    MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH,
    MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH,
    MacroAction.ADD_ARITHMETIC_UNIT_CHECK_GRAPH,
    MacroAction.ADD_CRITIQUE_REVISE_GRAPH,
    MacroAction.ADD_SPEC_ANALYZE_CODE_GRAPH,
    MacroAction.ADD_TEST_DEBUG_RETEST_GRAPH,
    MacroAction.ADD_ALTERNATIVE_IMPLEMENTATION_GRAPH,
    MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC,
    MacroAction.PRUNE_GRAPHAGENT_MODULE,
    MacroAction.SET_PAYLOAD_MODE,
    MacroAction.UPGRADE_NODE_MODEL,
    MacroAction.DOWNGRADE_NODE_MODEL,
    MacroAction.STOP,
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_SPACE)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}


class HierarchicalGCNEncoder(nn.Module):
    """Encode AtomicAgent and GraphAgent nodes with a torch GCN pass."""

    def __init__(
        self,
        *,
        input_dim: int = 32,
        hidden_dim: int = 32,
        graph_feature_dim: int = GRAPH_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.graph_feature_dim = graph_feature_dim
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_feature_dim, hidden_dim)

    def forward(self, graph: OrgGraphSnapshot, state: Mapping[str, Any]) -> torch.Tensor:
        node_features = self._node_feature_tensor(graph)
        if node_features.numel() == 0:
            pooled = torch.zeros(self.hidden_dim, dtype=torch.float32)
        else:
            hidden = torch.tanh(self.node_proj(node_features))
            adjacency = self._normalized_adjacency(graph, hidden.shape[0])
            neighbor_hidden = adjacency @ hidden
            convolved = torch.tanh(
                self.self_proj(hidden) + self.neighbor_proj(neighbor_hidden)
            )
            pooled = convolved.mean(dim=0)
        graph_features = self._graph_features(graph, state)
        return torch.tanh(pooled + self.graph_proj(graph_features))

    def _node_feature_tensor(self, graph: OrgGraphSnapshot) -> torch.Tensor:
        if not graph.nodes:
            return torch.empty((0, self.input_dim), dtype=torch.float32)
        return torch.tensor(
            [self._node_features(node, graph.domain) for node in graph.nodes],
            dtype=torch.float32,
        )

    def _node_features(self, node: Any, domain: str) -> list[float]:
        base = [
            _signed_hash("node", domain, node.role, node.node_kind, node.module_type, str(i))
            for i in range(self.input_dim)
        ]
        kind_bonus = 0.25 if node.node_kind == "graph" else -0.05
        tier_bonus = {"small": -0.1, "standard": 0.0, "large": 0.1}.get(
            node.model_tier,
            0.0,
        )
        internal_scale = min(len(node.internal_nodes), 8) / 8.0
        for index in range(self.input_dim):
            base[index] = math.tanh(base[index] + kind_bonus + tier_bonus)
        base[0] = 1.0 if node.node_kind == "graph" else 0.0
        base[1] = internal_scale
        base[2] = tier_bonus
        base[3] = _signed_hash("role", node.role)
        base[4] = _signed_hash("module", node.module_type or node.role)
        return base

    def _normalized_adjacency(
        self,
        graph: OrgGraphSnapshot,
        node_count: int,
    ) -> torch.Tensor:
        adjacency = torch.eye(node_count, dtype=torch.float32)
        index_by_id = {node.node_id: index for index, node in enumerate(graph.nodes)}
        for edge in graph.edges:
            if edge.src not in index_by_id or edge.dst not in index_by_id:
                continue
            src = index_by_id[edge.src]
            dst = index_by_id[edge.dst]
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(1) * adjacency * inv_sqrt.unsqueeze(0)

    def _graph_features(
        self,
        graph: OrgGraphSnapshot,
        state: Mapping[str, Any],
    ) -> torch.Tensor:
        graph_agents = sum(1 for node in graph.nodes if node.node_kind == "graph")
        internal_nodes = sum(len(node.internal_nodes) for node in graph.nodes)
        visible = state.get("observable_feedback", {})
        issue_count = (
            len(visible.get("issue_codes", ()))
            if isinstance(visible, Mapping)
            else 0
        )
        remaining = float(state.get("remaining_tokens", 0))
        used = float(state.get("used_tokens", 0))
        budget_ratio = remaining / max(remaining + used, 1.0)
        values = [
            len(graph.nodes) / 10.0,
            len(graph.edges) / 12.0,
            graph_depth(graph) / 8.0,
            graph_agents / 4.0,
            internal_nodes / 16.0,
            issue_count / 4.0,
            budget_ratio,
            _signed_hash("domain", graph.domain),
        ]
        values.extend(_task_vector(state))
        values.extend(_risk_vector(state))
        return torch.tensor(values, dtype=torch.float32)


class HierarchicalGNNPolicy(nn.Module):
    """Masked DQN-style policy over module-level graph-edit actions."""

    def __init__(
        self,
        *,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        seed: int = 13,
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        torch.manual_seed(seed)
        self.encoder = HierarchicalGCNEncoder(hidden_dim=hidden_dim)
        self.action_head = nn.Linear(hidden_dim, len(ACTION_SPACE))
        self._init_action_biases()
        self.epsilon = epsilon
        self.temperature = temperature
        self._rng = random.Random(seed)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, graph: OrgGraphSnapshot, state: Mapping[str, Any]) -> torch.Tensor:
        embedding = self.encoder(graph, state)
        return self.action_head(embedding)

    def decide(
        self,
        state: Mapping[str, Any],
        graph: OrgGraphSnapshot,
        candidates: Sequence[MacroCandidate],
    ) -> PolicyDecision:
        legal = tuple(candidate.action for candidate in candidates)
        if not legal:
            raise ValueError("policy requires at least one legal candidate")
        with torch.no_grad():
            q_values = self.forward(graph, state)
            action_priors = _action_prior_scores(graph, state, candidates)
            biased_q_values = q_values + action_priors
            mask = torch.tensor(
                [action in legal for action in ACTION_SPACE],
                dtype=torch.bool,
            )
            masked_q = biased_q_values.masked_fill(~mask, float("-inf"))
            legal_logits = masked_q[mask] / self.temperature
            probs = torch.softmax(legal_logits, dim=0)
            legal_actions = [action for action in ACTION_SPACE if action in legal]
            explored = False
            if self.epsilon and self._rng.random() < self.epsilon:
                selected = self._rng.choice(legal_actions)
                explored = True
            else:
                selected = legal_actions[int(torch.argmax(probs).item())]
            embedding = self.encoder(graph, state)
        scores = {
            action.value: round(float(biased_q_values[ACTION_TO_INDEX[action]].item()), 6)
            for action in legal
        }
        metadata = {
            "policy": "torch_hierarchical_gcn_dqn",
            "framework": "torch",
            "action_space": [action.value for action in ACTION_SPACE],
            "legal_mask": [bool(action in legal) for action in ACTION_SPACE],
            "q_values": {
                action.value: round(float(q_values[index].item()), 6)
                for index, action in enumerate(ACTION_SPACE)
            },
            "action_priors": {
                action.value: round(float(action_priors[index].item()), 6)
                for index, action in enumerate(ACTION_SPACE)
            },
            "biased_q_values": {
                action.value: round(float(biased_q_values[index].item()), 6)
                for index, action in enumerate(ACTION_SPACE)
            },
            "probabilities": {
                action.value: round(float(prob.item()), 6)
                for action, prob in zip(legal_actions, probs, strict=True)
            },
            "explored": explored,
            "graph_embedding": [round(float(value.item()), 6) for value in embedding],
        }
        return PolicyDecision(selected, scores, tuple(candidates), metadata=metadata)

    def td_update(
        self,
        *,
        state_embedding: torch.Tensor,
        action: MacroAction,
        target: float,
        learning_rate: float | None = None,
    ) -> float:
        """Run one torch optimizer step against a scalar TD target."""

        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        self.train()
        q_value = self.action_head(state_embedding)[ACTION_TO_INDEX[action]]
        target_tensor = torch.tensor(float(target), dtype=torch.float32)
        loss = F.smooth_l1_loss(q_value, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return round(float(loss.detach().item()), 6)

    def state_embedding(
        self,
        graph: OrgGraphSnapshot,
        state: Mapping[str, Any],
    ) -> torch.Tensor:
        return self.encoder(graph, state)

    def q_value(self, embedding: torch.Tensor, action: MacroAction) -> torch.Tensor:
        return self.action_head(embedding)[ACTION_TO_INDEX[action]]

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "epsilon": self.epsilon,
                "temperature": self.temperature,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> HierarchicalGNNPolicy:
        payload = torch.load(path, map_location="cpu")
        policy = cls(
            epsilon=float(payload.get("epsilon", 0.0)),
            temperature=float(payload.get("temperature", 1.0)),
            **kwargs,
        )
        state_dict = payload["state_dict"]
        current = policy.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        policy.load_state_dict(compatible, strict=False)
        return policy

    def _init_action_biases(self) -> None:
        with torch.no_grad():
            self.action_head.bias.zero_()
            for action, index in ACTION_TO_INDEX.items():
                self.action_head.bias[index] = _initial_action_bias(action)


def _signed_hash(*parts: str) -> float:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()
    integer = int(digest[:8], 16)
    return (integer / 0xFFFFFFFF) * 2.0 - 1.0


def _task_vector(state: Mapping[str, Any]) -> list[float]:
    features = state.get("task_features", {})
    if not isinstance(features, Mapping):
        return [0.0] * TASK_VECTOR_DIM
    raw = features.get("task_vector", ())
    vector = [float(value) for value in raw] if isinstance(raw, Sequence) else []
    if len(vector) < TASK_VECTOR_DIM:
        vector.extend([0.0] * (TASK_VECTOR_DIM - len(vector)))
    return vector[:TASK_VECTOR_DIM]


def _risk_vector(state: Mapping[str, Any]) -> list[float]:
    visible = state.get("observable_feedback", {})
    signals = visible.get("signals", {}) if isinstance(visible, Mapping) else {}
    if not isinstance(signals, Mapping):
        signals = {}
    parsed_options = signals.get("parsed_options", {})
    parsed_count = (
        sum(1 for value in parsed_options.values() if value)
        if isinstance(parsed_options, Mapping)
        else 0
    )
    parse_failed_count = float(signals.get("parse_failed_count", 0.0) or 0.0)
    consensus_fraction = float(signals.get("consensus_fraction", 0.0) or 0.0)
    majority_margin = float(signals.get("majority_margin", 0.0) or 0.0)
    entropy = float(signals.get("answer_vote_entropy", 0.0) or 0.0)
    rationale_similarity = float(signals.get("rationale_similarity", 0.0) or 0.0)
    contradiction_count = float(signals.get("contradiction_count", 0.0) or 0.0)
    evidence_coverage = float(signals.get("evidence_coverage", 0.0) or 0.0)
    answer_changed = 1.0 if signals.get("answer_changed") else 0.0
    adversarial_checked = 1.0 if signals.get("adversarial_checked") else 0.0
    return [
        min(parsed_count / 4.0, 1.0),
        min(parse_failed_count / 4.0, 1.0),
        max(0.0, min(consensus_fraction, 1.0)),
        max(0.0, min(majority_margin, 1.0)),
        max(0.0, min(entropy, 1.0)),
        max(0.0, min(rationale_similarity, 1.0)),
        min(contradiction_count / 3.0, 1.0),
        max(0.0, min(evidence_coverage, 1.0)),
        max(0.0, min(answer_changed + adversarial_checked, 1.0)),
    ]


def _consensus_stop_risk(
    state: Mapping[str, Any],
    graph: OrgGraphSnapshot,
) -> float:
    """Estimate label-blind value-of-computation risk for STOP.

    This is not a subject router. It uses execution-state signals that the GNN
    also receives, so terminal RL updates can learn when these patterns make
    STOP or adversarial construction valuable.
    """

    vector = _risk_vector(state)
    (
        parsed_count,
        parse_failures,
        consensus,
        majority_margin,
        entropy,
        rationale_similarity,
        contradiction,
        evidence_coverage,
        changed_or_checked,
    ) = vector
    modules = {node.module_type for node in graph.nodes if node.node_kind == "graph"}
    has_adversarial = "AdversarialBestAnswerGraph" in modules
    high_consensus_same_premise = (
        1.0 if parsed_count >= 0.5 and consensus >= 0.99 and rationale_similarity >= 0.18 else 0.0
    )
    low_diversity_consensus = max(0.0, consensus - entropy)
    risk = (
        0.35 * high_consensus_same_premise
        + 0.25 * low_diversity_consensus
        + 0.25 * parse_failures
        + 0.12 * majority_margin
        + 0.08 * rationale_similarity
        - 0.18 * contradiction
        - 0.18 * changed_or_checked
        - (0.25 if has_adversarial else 0.0)
        - 0.05 * evidence_coverage
    )
    return max(0.0, min(float(risk), 1.0))


def _action_prior_scores(
    graph: OrgGraphSnapshot,
    state: Mapping[str, Any],
    candidates: Sequence[MacroCandidate],
) -> torch.Tensor:
    priors = torch.zeros(len(ACTION_SPACE), dtype=torch.float32)
    if graph.domain != "mmlu":
        return priors
    features = state.get("task_features", {})
    task_prior = _mmlu_task_action_prior(features if isinstance(features, Mapping) else {})
    graph_prior = _mmlu_graph_action_prior(graph, state)
    for candidate in candidates:
        base_prior = float(candidate.parameters.get("prior_score", 0.0))
        score = base_prior + task_prior.get(candidate.action, 0.0) + graph_prior.get(candidate.action, 0.0)
        priors[ACTION_TO_INDEX[candidate.action]] = MMLU_PRIOR_WEIGHT * score
    return priors


def _mmlu_task_action_prior(features: Mapping[str, Any]) -> dict[MacroAction, float]:
    keyword_flags = _mapping_float_dict(features.get("keyword_flags", {}))
    type_flags = _mapping_float_dict(features.get("question_type_flags", {}))
    profile = str(features.get("subject_profile", "general"))
    option_overlap = float(features.get("option_overlap_mean", 0.0) or 0.0)
    option_variance = float(features.get("option_length_variance", 0.0) or 0.0)
    question_words = float(features.get("question_word_count", 0.0) or 0.0)
    math_like = max(
        keyword_flags.get("math", 0.0),
        keyword_flags.get("economics", 0.0),
        1.0 if profile == "stem" else 0.0,
    )
    expert_like = max(
        keyword_flags.get("medicine", 0.0),
        keyword_flags.get("law", 0.0),
        keyword_flags.get("history", 0.0),
        1.0 if profile in {"professional", "humanities"} else 0.0,
    )
    ambiguous_options = max(option_overlap, min(option_variance / 12.0, 1.0))
    reasoning_like = max(
        math_like,
        type_flags.get("causal", 0.0),
        type_flags.get("compare", 0.0),
        1.0 if question_words > 45 else 0.0,
    )
    critique_like = max(
        type_flags.get("exception", 0.0),
        keyword_flags.get("philosophy", 0.0),
        ambiguous_options,
    )
    return {
        MacroAction.ADD_SUBJECT_EXPERT_GRAPH: 0.45 * expert_like + 0.15,
        MacroAction.ADD_OPTION_ELIMINATION_GRAPH: 0.5 * ambiguous_options + 0.35,
        MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH: 0.55 * reasoning_like,
        MacroAction.ADD_SECOND_OPINION_DEBATE_GRAPH: 0.35 * critique_like,
        MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH: 0.05,
        MacroAction.ADD_CRITIQUE_REVISE_GRAPH: 0.4 * critique_like,
        MacroAction.STOP: -0.15,
        MacroAction.PRUNE_GRAPHAGENT_MODULE: 0.1 if question_words < 18 else 0.0,
    }


def _mapping_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _mmlu_graph_action_prior(
    graph: OrgGraphSnapshot,
    state: Mapping[str, Any],
) -> dict[MacroAction, float]:
    modules = {node.module_type for node in graph.nodes if node.node_kind == "graph"}
    useful_modules = modules & {
        "SubjectExpertGraph",
        "OptionEliminationGraph",
        "DecomposeSolveVerifyGraph",
        "SecondOpinionDebateGraph",
        "CritiqueReviseGraph",
        "AdversarialBestAnswerGraph",
    }
    risk = _consensus_stop_risk(state, graph)
    has_adversarial = "AdversarialBestAnswerGraph" in modules
    priors: dict[MacroAction, float] = {
        MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC: -0.8,
        MacroAction.PRUNE_GRAPHAGENT_MODULE: -0.9,
    }
    if not has_adversarial and graph.step >= 1:
        priors[MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH] = 1.15 * risk - 0.05
    if len(useful_modules) >= 2:
        priors[MacroAction.STOP] = 0.45 - 0.75 * risk
    if len(useful_modules) >= 3 or graph.step >= 3:
        priors[MacroAction.STOP] = 1.2 - 1.1 * risk
        priors[MacroAction.UPGRADE_NODE_MODEL] = -0.25
    if has_adversarial and len(useful_modules) >= 3:
        priors[MacroAction.STOP] = max(priors.get(MacroAction.STOP, 0.0), 0.8)
    return priors


def _initial_action_bias(action: MacroAction) -> float:
    if action is MacroAction.STOP:
        return 0.1
    if action in {
        MacroAction.EXPAND_ATOMIC_TO_GRAPHAGENT,
        MacroAction.ADD_OPTION_ELIMINATION_GRAPH,
        MacroAction.ADD_DECOMPOSE_SOLVE_VERIFY_GRAPH,
        MacroAction.ADD_ADVERSARIAL_BEST_ANSWER_GRAPH,
        MacroAction.ADD_SPEC_ANALYZE_CODE_GRAPH,
    }:
        return 0.35
    if action in {
        MacroAction.DOWNGRADE_GRAPHAGENT_TO_ATOMIC,
        MacroAction.PRUNE_GRAPHAGENT_MODULE,
    }:
        return 0.15
    return 0.25
