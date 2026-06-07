#!/usr/bin/env python3
"""Train the GOG graph-construction policy with behavior cloning."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gogagent.policy import (  # noqa: E402
    ACTION_SPACE,
    GraphEncoder,
    PolicyNetwork,
    SentenceTransformerTaskEncoder,
    action_count,
    feature_dim,
    task_to_text,
)
from train.BC.step_dataset import BCStepDataset, BCStepExample  # noqa: E402
from train.checkpoint import (  # noqa: E402
    module_parameter_count,
    save_policy_checkpoint,
)


MASK_VALUE = -1e9


@dataclass(frozen=True)
class PreparedBCStep:
    """One BC example with a cached task embedding."""

    source: BCStepExample
    task_embedding: torch.Tensor


def main() -> None:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, required=True, help="BC steps.jsonl path")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "policies",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-encoder-device", default=None)
    parser.add_argument("--task-encoder-model", default=None)
    parser.add_argument("--no-normalize-task-embeddings", action="store_true")
    parser.add_argument("--graph-embedding-dim", type=int, default=64)
    parser.add_argument("--graph-hidden-dim", type=int, default=None)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--max-subgraph-depth", type=int, default=2)
    parser.add_argument("--graph-dropout", type=float, default=0.0)
    parser.add_argument("--graph-pooling", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--policy-hidden-dim", type=int, default=None)
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="balanced",
        help="Apply per-action class weights to the BC loss. Default: balanced",
    )
    parser.add_argument(
        "--class-weight-alpha",
        type=float,
        default=0.5,
        help="Smoothing exponent for balanced class weights. Default: 0.5",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train BC policy and save a checkpoint."""

    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    run_dir = make_run_dir(args.output_dir, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = BCStepDataset(args.steps, limit=args.limit)
    task_encoder = SentenceTransformerTaskEncoder(
        model_name=args.task_encoder_model,
        device=args.task_encoder_device,
        normalize_embeddings=not args.no_normalize_task_embeddings,
    )
    task_embedding_dim = task_encoder.embedding_dim
    prepared_steps = prepare_steps(
        dataset,
        task_encoder=task_encoder,
        device=device,
        show_progress=not args.no_progress,
    )

    graph_encoder_config = {
        "input_dim": feature_dim(),
        "embedding_dim": args.graph_embedding_dim,
        "hidden_dim": args.graph_hidden_dim,
        "num_layers": args.graph_layers,
        "max_subgraph_depth": args.max_subgraph_depth,
        "dropout": args.graph_dropout,
        "pooling": args.graph_pooling,
    }
    policy_network_config = {
        "graph_embedding_dim": args.graph_embedding_dim,
        "task_embedding_dim": task_embedding_dim,
        "num_actions": action_count(),
        "hidden_dim": args.policy_hidden_dim,
        "dropout": args.policy_dropout,
    }
    graph_encoder = GraphEncoder(**graph_encoder_config).to(device)
    policy_network = PolicyNetwork(**policy_network_config).to(device)
    class_weights = make_class_weights(
        prepared_steps,
        mode=args.class_weight,
        alpha=args.class_weight_alpha,
        device=device,
    )
    class_weight_summary = summarize_class_weights(class_weights)
    optimizer = torch.optim.AdamW(
        [*graph_encoder.parameters(), *policy_network.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            prepared_steps,
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            grad_clip=args.grad_clip,
            class_weights=class_weights,
            show_progress=not args.no_progress,
        )
        history.append(metrics)

    checkpoint_config = {
        "graph_encoder": graph_encoder_config,
        "policy_network": policy_network_config,
        "task_encoder": {
            "model_name": task_encoder.model_name,
            "normalize_embeddings": task_encoder.normalize_embeddings,
        },
        "training": {
            "method": "behavior_cloning",
            "steps": str(args.steps),
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
            "class_weight": {
                "mode": args.class_weight,
                "alpha": args.class_weight_alpha,
                "weights": class_weight_summary,
            },
            "uses_solver_probe_output_as_input": False,
        },
    }
    final_metrics = history[-1]
    checkpoint_path = run_dir / "policy.pt"
    save_policy_checkpoint(
        checkpoint_path,
        graph_encoder=graph_encoder,
        policy_network=policy_network,
        optimizer=optimizer,
        config=checkpoint_config,
        metrics={"history": history, "final": final_metrics},
        extra={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trainable_parameters": {
                "graph_encoder": module_parameter_count(graph_encoder),
                "policy_network": module_parameter_count(policy_network),
            },
        },
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "steps": str(args.steps),
        "num_examples": len(dataset),
        "epochs": args.epochs,
        "device": str(device),
        "task_encoder_model": task_encoder.model_name,
        "action_space": [action.value for action in ACTION_SPACE],
        "class_weight": {
            "mode": args.class_weight,
            "alpha": args.class_weight_alpha,
            "weights": class_weight_summary,
        },
        "config": checkpoint_config,
        "history": history,
        "final": final_metrics,
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def prepare_steps(
    dataset: BCStepDataset,
    *,
    task_encoder: SentenceTransformerTaskEncoder,
    device: torch.device,
    show_progress: bool,
) -> list[PreparedBCStep]:
    """Precompute frozen task embeddings for every step."""

    task_cache: dict[str, torch.Tensor] = {}
    prepared: list[PreparedBCStep] = []
    iterator = tqdm(
        range(len(dataset)),
        desc="Encoding BC tasks",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for index in iterator:
            example = dataset[index]
            text = task_to_text(example.task)
            embedding = task_cache.get(text)
            if embedding is None:
                embedding = task_encoder.encode_task(example.task, device=device).detach()
                task_cache[text] = embedding
            prepared.append(PreparedBCStep(source=example, task_embedding=embedding))
            iterator.set_postfix({"unique_tasks": len(task_cache)})
    return prepared


def train_one_epoch(
    steps: list[PreparedBCStep],
    *,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    shuffle: bool,
    grad_clip: float,
    class_weights: torch.Tensor | None,
    show_progress: bool,
) -> dict[str, Any]:
    """Train one epoch over unbatched BC steps."""

    graph_encoder.train()
    policy_network.train()
    indices = list(range(len(steps)))
    if shuffle:
        random.Random(seed + epoch).shuffle(indices)

    total_loss = 0.0
    correct = 0
    processed = 0
    counts: dict[str, int] = {}
    iterator = tqdm(
        indices,
        desc=f"BC epoch {epoch}",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for index in iterator:
        prepared = steps[index]
        example = prepared.source
        optimizer.zero_grad(set_to_none=True)
        graph_embedding = graph_encoder(example.graph_before)
        logits = policy_network(graph_embedding, prepared.task_embedding)
        masked_logits = apply_legal_action_mask(
            logits,
            example.legal_action_mask(device=logits.device),
        )
        target = torch.tensor([example.target_index], dtype=torch.long, device=device)
        loss = action_cross_entropy(
            masked_logits,
            target,
            class_weights=class_weights,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [*graph_encoder.parameters(), *policy_network.parameters()],
                grad_clip,
            )
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        processed += 1
        prediction = int(torch.argmax(masked_logits.detach()).cpu().item())
        if prediction == example.target_index:
            correct += 1
        counts[example.target_action.value] = counts.get(example.target_action.value, 0) + 1
        iterator.set_postfix(
            {
                "loss": f"{total_loss / processed:.4f}",
                "acc": f"{correct / processed:.3f}",
            }
        )

    denominator = max(1, len(steps))
    return {
        "epoch": epoch,
        "loss": total_loss / denominator,
        "accuracy": correct / denominator,
        "examples": len(steps),
        "target_action_distribution": dict(sorted(counts.items())),
    }


def apply_legal_action_mask(
    logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    *,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """Apply an output-side legality mask to action logits."""

    if logits.dim() != 1:
        raise ValueError("BC training expects one unbatched logits vector")
    if legal_action_mask.shape != logits.shape:
        raise ValueError(
            "legal_action_mask shape must match logits shape: "
            f"{tuple(legal_action_mask.shape)} != {tuple(logits.shape)}"
        )
    if not bool(legal_action_mask.any().item()):
        raise ValueError("legal_action_mask must contain at least one legal action")
    return logits.masked_fill(~legal_action_mask.to(device=logits.device), mask_value)


def action_cross_entropy(
    masked_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Return one weighted action-classification loss."""

    loss = F.cross_entropy(masked_logits.unsqueeze(0), target, reduction="none")
    if class_weights is None:
        return loss.mean()
    if class_weights.dim() != 1 or class_weights.numel() != masked_logits.numel():
        raise ValueError("class_weights must have shape [num_actions]")
    return (loss * class_weights[target]).mean()


def make_class_weights(
    steps: list[PreparedBCStep],
    *,
    mode: str,
    alpha: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Build smoothed inverse-frequency action weights."""

    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError("class weight mode must be 'none' or 'balanced'")
    if alpha <= 0:
        raise ValueError("class_weight_alpha must be positive")

    counts = Counter(step.source.target_index for step in steps)
    observed = {index: count for index, count in counts.items() if count > 0}
    if not observed:
        raise ValueError("cannot build class weights without observed target actions")

    total = sum(observed.values())
    observed_class_count = len(observed)
    weights = torch.zeros(action_count(), dtype=torch.float32, device=device)
    for index, count in observed.items():
        base_weight = total / (observed_class_count * count)
        weights[index] = float(base_weight**alpha)

    observed_indices = torch.tensor(sorted(observed), dtype=torch.long, device=device)
    observed_mean = weights[observed_indices].mean().clamp_min(1e-12)
    weights[observed_indices] = weights[observed_indices] / observed_mean
    return weights


def summarize_class_weights(class_weights: torch.Tensor | None) -> dict[str, float]:
    """Return JSON-friendly class weights keyed by action name."""

    if class_weights is None:
        return {action.value: 1.0 for action in ACTION_SPACE}
    values = class_weights.detach().cpu().tolist()
    return {
        action.value: float(values[index])
        for index, action in enumerate(ACTION_SPACE)
    }


def make_run_dir(output_dir: Path, run_id: str | None) -> Path:
    """Return the BC training artifact directory."""

    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / run_name


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    if args.class_weight_alpha <= 0:
        raise ValueError("class_weight_alpha must be positive")


if __name__ == "__main__":
    main()
