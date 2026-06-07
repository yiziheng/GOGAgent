"""Runtime policy runner for graph-construction action selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.graph.schema import Graph
from gogagent.policy.graph_encoder import GraphEncoder
from gogagent.policy.network import PolicyNetwork
from gogagent.policy.selector import select_action as select_masked_action
from gogagent.policy.task_encoder import SentenceTransformerTaskEncoder


@dataclass
class PolicyRunner:
    """Load a trained policy and select legal graph-construction actions."""

    graph_encoder: GraphEncoder
    policy_network: PolicyNetwork
    task_encoder: SentenceTransformerTaskEncoder
    device: torch.device
    constraints: ActionConstraints = field(default_factory=ActionConstraints)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        task_encoder_device: str | None = None,
    ) -> "PolicyRunner":
        """Create a runner from a saved policy checkpoint."""

        from train.checkpoint import load_policy_components

        target_device = torch.device(device)
        graph_encoder, policy_network, checkpoint = load_policy_components(
            checkpoint_path,
            map_location=target_device,
            device=target_device,
        )
        config = checkpoint["config"]
        task_config = dict(config.get("task_encoder") or {})
        task_encoder = SentenceTransformerTaskEncoder(
            model_name=task_config.get("model_name"),
            device=task_encoder_device,
            normalize_embeddings=bool(task_config.get("normalize_embeddings", True)),
        )
        constraint_config = dict(config.get("action_constraints") or {})
        constraints = (
            ActionConstraints(**constraint_config)
            if constraint_config
            else ActionConstraints()
        )
        graph_encoder.eval()
        policy_network.eval()
        return cls(
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            task_encoder=task_encoder,
            device=target_device,
            constraints=constraints,
        )

    def logits(self, graph: Graph, task: Mapping[str, Any]) -> torch.Tensor:
        """Return raw action logits for one graph/task pair."""

        with torch.no_grad():
            graph_embedding = self.graph_encoder(graph)
            task_embedding = self.task_encoder.encode_task(task, device=self.device)
            return self.policy_network(graph_embedding, task_embedding)

    def select_action(
        self,
        graph: Graph,
        task: Mapping[str, Any],
        *,
        constraints: ActionConstraints | None = None,
        mode: str = "argmax",
        temperature: float = 1.0,
    ) -> ActionName:
        """Return a legal action selected from masked policy logits."""

        logits = self.logits(graph, task)
        return select_masked_action(
            logits,
            graph,
            constraints or self.constraints,
            mode=mode,  # type: ignore[arg-type]
            temperature=temperature,
        )


def main() -> None:
    args = parse_args()
    runner = PolicyRunner.from_checkpoint(
        args.checkpoint,
        device=args.device,
        task_encoder_device=args.task_encoder_device,
    )
    graph = Graph.from_dict(_load_json_object(args.graph))
    task = _load_json_object(args.task)
    action = runner.select_action(graph, task)
    print(json.dumps({"predicted_action": action.value}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True, help="Graph JSON path")
    parser.add_argument("--task", type=Path, required=True, help="Task JSON path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-encoder-device", default=None)
    return parser.parse_args()


def _load_json_object(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"expected JSON object in {path}")
    return data


if __name__ == "__main__":
    main()


__all__ = ["PolicyRunner"]
