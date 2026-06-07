#!/usr/bin/env python3
"""Refine a graph-construction policy with GRPO-style RL."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.artifacts import prepare_run_dir  # noqa: E402
from gogagent.actions.base import ActionConstraints  # noqa: E402
from gogagent.config import llm_client_from_env  # noqa: E402
from gogagent.datasets import DatasetExample, enrich_example, load_examples, load_selection  # noqa: E402
from gogagent.llm import AgentContext  # noqa: E402
from gogagent.policy import (  # noqa: E402
    ACTION_SPACE,
    GraphEncoder,
    PolicyNetwork,
    SentenceTransformerTaskEncoder,
    task_to_text,
)
from train.RL.logging import append_jsonl, write_json  # noqa: E402
from train.RL.losses import compute_group_advantages, grpo_rollout_loss  # noqa: E402
from train.RL.rollout import rollout_group  # noqa: E402
from train.checkpoint import (  # noqa: E402
    load_policy_components,
    module_parameter_count,
    save_policy_checkpoint,
)


@dataclass(frozen=True)
class PreparedRLExample:
    """One training problem with a cached task embedding."""

    index: int
    example: DatasetExample
    task_embedding: torch.Tensor


def main() -> None:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="mmlu", choices=("mmlu", "gsm8k", "humaneval"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection-jsonl", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--rl-output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "rl",
    )
    parser.add_argument(
        "--policy-output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "policies",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-encoder-device", default=None)
    parser.add_argument("--no-item-artifacts", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run grouped rollout RL and save the refined policy checkpoint."""

    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rl_run_dir = prepare_run_dir(args.rl_output_dir, run_id, overwrite=args.overwrite)
    policy_run_dir = prepare_run_dir(args.policy_output_dir, run_id, overwrite=args.overwrite)
    rollouts_path = rl_run_dir / "rollouts.jsonl"
    metrics_path = rl_run_dir / "metrics.jsonl"

    device = torch.device(args.device)
    graph_encoder, policy_network, checkpoint = load_policy_components(
        args.checkpoint,
        map_location=device,
        device=device,
    )
    reference_graph_encoder, reference_policy_network, _ = load_policy_components(
        args.checkpoint,
        map_location=device,
        device=device,
    )
    freeze_reference_policy(reference_graph_encoder, reference_policy_network)
    graph_encoder.eval()
    policy_network.eval()

    checkpoint_config = checkpoint["config"]
    task_encoder = make_task_encoder(
        checkpoint_config,
        task_encoder_device=args.task_encoder_device,
    )
    examples = load_examples(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
    )
    selection = load_selection(args.selection_jsonl)
    enriched_examples = [enrich_example(example, selection) for example in examples]
    prepared_examples = prepare_examples(
        enriched_examples,
        task_encoder=task_encoder,
        device=device,
        show_progress=not args.no_progress,
    )

    client = llm_client_from_env(args.env)
    context = AgentContext(llm_client=client)
    constraints = ActionConstraints(max_depth=args.max_depth, max_nodes=args.max_nodes)
    optimizer = torch.optim.AdamW(
        [*graph_encoder.parameters(), *policy_network.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    global_group_index = 0
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = train_one_epoch(
            epoch=epoch,
            prepared_examples=prepared_examples,
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            reference_graph_encoder=reference_graph_encoder,
            reference_policy_network=reference_policy_network,
            optimizer=optimizer,
            context=context,
            client_description=dict(client.describe()),
            run_dir=rl_run_dir,
            constraints=constraints,
            group_size=args.group_size,
            max_actions=args.max_actions,
            temperature=args.temperature,
            kl_beta=args.kl_beta,
            grad_clip=args.grad_clip,
            seed=args.seed,
            global_group_start=global_group_index,
            rollouts_path=rollouts_path,
            metrics_path=metrics_path,
            save_item_artifacts=not args.no_item_artifacts,
            show_progress=not args.no_progress,
        )
        global_group_index += len(prepared_examples)
        history.append(epoch_metrics)

    final_metrics = history[-1]
    checkpoint_path = policy_run_dir / "policy.pt"
    output_config = make_output_config(
        checkpoint_config,
        args=args,
        run_id=run_id,
        task_encoder=task_encoder,
    )
    save_policy_checkpoint(
        checkpoint_path,
        graph_encoder=graph_encoder,
        policy_network=policy_network,
        optimizer=optimizer,
        config=output_config,
        metrics={"history": history, "final": final_metrics},
        extra={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_checkpoint": str(args.checkpoint),
            "trainable_parameters": {
                "graph_encoder": module_parameter_count(graph_encoder),
                "policy_network": module_parameter_count(policy_network),
            },
        },
    )

    summary = {
        "run_id": run_id,
        "rl_run_dir": str(rl_run_dir),
        "policy_run_dir": str(policy_run_dir),
        "checkpoint": str(checkpoint_path),
        "base_checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "split": args.split,
        "selection_jsonl": str(args.selection_jsonl) if args.selection_jsonl else None,
        "num_examples": len(prepared_examples),
        "epochs": args.epochs,
        "group_size": args.group_size,
        "device": str(device),
        "task_encoder_model": task_encoder.model_name,
        "action_space": [action.value for action in ACTION_SPACE],
        "rollouts_jsonl": str(rollouts_path),
        "metrics_jsonl": str(metrics_path),
        "config": output_config,
        "history": history,
        "final": final_metrics,
    }
    summary_path = rl_run_dir / "summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    write_json(policy_run_dir / "train_summary.json", summary)
    return summary


def train_one_epoch(
    *,
    epoch: int,
    prepared_examples: list[PreparedRLExample],
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    reference_graph_encoder: GraphEncoder,
    reference_policy_network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    context: AgentContext,
    client_description: dict[str, Any],
    run_dir: Path,
    constraints: ActionConstraints,
    group_size: int,
    max_actions: int,
    temperature: float,
    kl_beta: float,
    grad_clip: float,
    seed: int,
    global_group_start: int,
    rollouts_path: Path,
    metrics_path: Path,
    save_item_artifacts: bool,
    show_progress: bool,
) -> dict[str, Any]:
    """Train one epoch over grouped problem rollouts."""

    graph_encoder.eval()
    policy_network.eval()
    reference_graph_encoder.eval()
    reference_policy_network.eval()

    order = list(prepared_examples)
    random.Random(seed + epoch).shuffle(order)
    processed_groups = 0
    total_loss = 0.0
    total_reward = 0.0
    total_rollouts = 0
    correct_rollouts = 0
    format_valid_rollouts = 0
    status_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}

    iterator = tqdm(
        order,
        desc=f"RL epoch {epoch}",
        unit="problem",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for local_group_index, prepared in enumerate(iterator, start=1):
        group_index = global_group_start + local_group_index
        rollouts = rollout_group(
            epoch=epoch,
            example_index=prepared.index,
            group_index=group_index,
            example=prepared.example,
            task_embedding=prepared.task_embedding,
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            context=context,
            client_description=client_description,
            run_dir=run_dir,
            constraints=constraints,
            group_size=group_size,
            max_actions=max_actions,
            temperature=temperature,
            save_item_artifacts=save_item_artifacts,
        )
        rewards = [rollout.reward for rollout in rollouts]
        advantages = compute_group_advantages(rewards)

        optimizer.zero_grad(set_to_none=True)
        losses = []
        rollout_loss_metrics = []
        for rollout, advantage in zip(rollouts, advantages, strict=True):
            loss, loss_metrics = grpo_rollout_loss(
                rollout=rollout,
                advantage=advantage,
                task_embedding=prepared.task_embedding,
                graph_encoder=graph_encoder,
                policy_network=policy_network,
                reference_graph_encoder=reference_graph_encoder,
                reference_policy_network=reference_policy_network,
                constraints=constraints,
                temperature=temperature,
                kl_beta=kl_beta,
            )
            losses.append(loss)
            rollout_loss_metrics.append(loss_metrics)
            rollout.advantage = float(advantage)
            rollout.loss = float(loss.detach().cpu().item())
            rollout.logprob_sum = loss_metrics["logprob_sum"]
            rollout.kl = loss_metrics["kl"]

        group_loss = torch.stack(losses).mean()
        group_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [*graph_encoder.parameters(), *policy_network.parameters()],
                grad_clip,
            )
        optimizer.step()

        group_loss_value = float(group_loss.detach().cpu().item())
        reward_mean = sum(rewards) / len(rewards)
        processed_groups += 1
        total_loss += group_loss_value
        total_reward += sum(rewards)
        total_rollouts += len(rollouts)

        for rollout in rollouts:
            if rollout.correct is True:
                correct_rollouts += 1
            if rollout.format_valid is True:
                format_valid_rollouts += 1
            status_counts[rollout.status] = status_counts.get(rollout.status, 0) + 1
            for action in rollout.action_sequence:
                action_counts[action] = action_counts.get(action, 0) + 1
            append_jsonl(rollouts_path, rollout.to_dict())

        metrics_row = {
            "epoch": epoch,
            "group_index": group_index,
            "task_id": rollouts[0].task_id,
            "dataset": rollouts[0].dataset,
            "reward_mean": reward_mean,
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "advantage_abs_mean": (
                sum(abs(advantage) for advantage in advantages) / len(advantages)
            ),
            "loss": group_loss_value,
            "kl_mean": (
                sum(metric["kl"] for metric in rollout_loss_metrics)
                / len(rollout_loss_metrics)
            ),
            "correct_count": sum(1 for rollout in rollouts if rollout.correct is True),
            "status_counts": dict(sorted(status_counts.items())),
            "action_counts": dict(sorted(action_counts.items())),
        }
        append_jsonl(metrics_path, metrics_row)
        iterator.set_postfix(
            {
                "loss": f"{total_loss / processed_groups:.4f}",
                "reward": f"{total_reward / total_rollouts:.3f}",
                "acc": f"{correct_rollouts / total_rollouts:.3f}",
            }
        )

    denominator = max(1, total_rollouts)
    return {
        "epoch": epoch,
        "groups": processed_groups,
        "rollouts": total_rollouts,
        "loss": total_loss / max(1, processed_groups),
        "reward": total_reward / denominator,
        "accuracy": correct_rollouts / denominator,
        "format_valid_rate": format_valid_rollouts / denominator,
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }


def prepare_examples(
    examples: list[DatasetExample],
    *,
    task_encoder: SentenceTransformerTaskEncoder,
    device: torch.device,
    show_progress: bool,
) -> list[PreparedRLExample]:
    """Cache frozen task embeddings for policy inputs."""

    task_cache: dict[str, torch.Tensor] = {}
    prepared: list[PreparedRLExample] = []
    iterator = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc="Encoding RL tasks",
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for index, example in iterator:
            task = dict(example.public_task)
            text = task_to_text(task)
            embedding = task_cache.get(text)
            if embedding is None:
                embedding = task_encoder.encode_task(task, device=device).detach()
                task_cache[text] = embedding
            prepared.append(
                PreparedRLExample(index=index, example=example, task_embedding=embedding)
            )
            iterator.set_postfix({"unique_tasks": len(task_cache)})
    return prepared


def make_task_encoder(
    checkpoint_config: dict[str, Any],
    *,
    task_encoder_device: str | None,
) -> SentenceTransformerTaskEncoder:
    task_config = dict(checkpoint_config.get("task_encoder") or {})
    return SentenceTransformerTaskEncoder(
        model_name=task_config.get("model_name"),
        device=task_encoder_device,
        normalize_embeddings=bool(task_config.get("normalize_embeddings", True)),
    )


def make_output_config(
    checkpoint_config: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    task_encoder: SentenceTransformerTaskEncoder,
) -> dict[str, Any]:
    output_config = deepcopy(checkpoint_config)
    output_config["task_encoder"] = {
        **dict(output_config.get("task_encoder") or {}),
        "model_name": task_encoder.model_name,
        "normalize_embeddings": task_encoder.normalize_embeddings,
    }
    output_config["training"] = {
        "method": "grpo_rl",
        "run_id": run_id,
        "base_checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "split": args.split,
        "selection_jsonl": str(args.selection_jsonl) if args.selection_jsonl else None,
        "epochs": args.epochs,
        "group_size": args.group_size,
        "max_actions": args.max_actions,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "temperature": args.temperature,
        "kl_beta": args.kl_beta,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "uses_solver_probe_output_as_input": False,
    }
    return output_config


def freeze_reference_policy(
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
) -> None:
    graph_encoder.eval()
    policy_network.eval()
    for parameter in graph_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in policy_network.parameters():
        parameter.requires_grad_(False)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.group_size < 2:
        raise ValueError("group_size must be at least 2 for within-question GRPO")
    if args.max_actions <= 0:
        raise ValueError("max_actions must be positive")
    if args.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if args.max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")


if __name__ == "__main__":
    main()
