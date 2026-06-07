"""Shared policy checkpoint helpers for BC and future RL training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from gogagent.policy import ACTION_SPACE, GraphEncoder, PolicyNetwork


CHECKPOINT_TYPE = "gog_policy"
CHECKPOINT_VERSION = 1


def save_policy_checkpoint(
    path: str | Path,
    *,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    optimizer: torch.optim.Optimizer | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a shared GOG policy checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "version": CHECKPOINT_VERSION,
        "action_space": [action.value for action in ACTION_SPACE],
        "graph_encoder_state_dict": graph_encoder.state_dict(),
        "policy_state_dict": policy_network.state_dict(),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = dict(extra)
    torch.save(payload, target)
    return target


def load_policy_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a shared GOG policy checkpoint."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("policy checkpoint must be a dict payload")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != CHECKPOINT_TYPE:
        raise ValueError(
            f"unsupported checkpoint_type {checkpoint_type!r}; expected {CHECKPOINT_TYPE!r}"
        )
    action_space = checkpoint.get("action_space")
    expected_action_space = [action.value for action in ACTION_SPACE]
    if action_space != expected_action_space:
        raise ValueError(
            "checkpoint action_space does not match current ACTION_SPACE: "
            f"{action_space!r} != {expected_action_space!r}"
        )
    for key in ("graph_encoder_state_dict", "policy_state_dict", "config"):
        if key not in checkpoint:
            raise ValueError(f"policy checkpoint missing required key {key!r}")
    return checkpoint


def build_policy_components(
    checkpoint: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[GraphEncoder, PolicyNetwork]:
    """Instantiate graph encoder and policy network from a checkpoint payload."""

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")

    graph_config = _mapping_config(config, "graph_encoder")
    policy_config = _mapping_config(config, "policy_network")
    graph_encoder = GraphEncoder(**graph_config)
    policy_network = PolicyNetwork(**policy_config)
    graph_encoder.load_state_dict(checkpoint["graph_encoder_state_dict"])
    policy_network.load_state_dict(checkpoint["policy_state_dict"])
    graph_encoder.to(device)
    policy_network.to(device)
    return graph_encoder, policy_network


def load_policy_components(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
) -> tuple[GraphEncoder, PolicyNetwork, dict[str, Any]]:
    """Load a checkpoint and return initialized policy components."""

    checkpoint = load_policy_checkpoint(path, map_location=map_location)
    target_device = device if device is not None else map_location
    graph_encoder, policy_network = build_policy_components(
        checkpoint,
        device=target_device,
    )
    return graph_encoder, policy_network, checkpoint


def module_parameter_count(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _mapping_config(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint config missing mapping {key!r}")
    return dict(value)


__all__ = [
    "CHECKPOINT_TYPE",
    "CHECKPOINT_VERSION",
    "build_policy_components",
    "load_policy_checkpoint",
    "load_policy_components",
    "module_parameter_count",
    "save_policy_checkpoint",
]
