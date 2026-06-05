"""Masked action selection for policy logits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from gogagent.actions.base import ActionConstraints, ActionName
from gogagent.actions.mask import compute_action_mask
from gogagent.policy.action_space import ACTION_SPACE


SelectionMode = Literal["argmax", "sample"]


@dataclass(frozen=True)
class MaskedActionSelector:
    """Select only legal actions from policy logits."""

    mode: SelectionMode = "argmax"
    temperature: float = 1.0

    def select(
        self,
        logits: torch.Tensor,
        graph: Any,
        constraints: ActionConstraints | None = None,
        *,
        mode: SelectionMode | None = None,
        generator: torch.Generator | None = None,
    ) -> ActionName:
        """Return a legal ActionName."""

        return select_action(
            logits,
            graph,
            constraints,
            mode=mode or self.mode,
            temperature=self.temperature,
            generator=generator,
        )


def select_action(
    logits: torch.Tensor,
    graph: Any,
    constraints: ActionConstraints | None = None,
    *,
    mode: SelectionMode = "argmax",
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> ActionName:
    """Mask logits by legality and select an ActionName."""

    action_space = ACTION_SPACE
    masked_logits, legal_actions = mask_action_logits(
        logits,
        graph,
        constraints,
        action_space=action_space,
        temperature=temperature,
    )
    if not legal_actions:
        raise ValueError("no legal actions available")

    if mode == "argmax":
        selected_index = int(torch.argmax(masked_logits).item())
    elif mode == "sample":
        probabilities = torch.softmax(masked_logits, dim=-1)
        selected_index = int(
            torch.multinomial(probabilities, num_samples=1, generator=generator).item()
        )
    else:
        raise ValueError("mode must be 'argmax' or 'sample'")

    selected = action_space[selected_index]
    if selected not in legal_actions:
        raise RuntimeError(f"selected illegal action after masking: {selected.value}")
    return selected


def mask_action_logits(
    logits: torch.Tensor,
    graph: Any,
    constraints: ActionConstraints | None = None,
    *,
    action_space: Sequence[ActionName] | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, tuple[ActionName, ...]]:
    """Return logits with illegal actions set to -inf plus legal actions."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    logits = logits.float()
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits.squeeze(0)
    if logits.dim() != 1:
        raise ValueError("selector expects logits with shape [num_actions]")

    action_space = tuple(action_space or ACTION_SPACE)
    if logits.numel() != len(action_space):
        raise ValueError(
            f"logits length {logits.numel()} does not match action space size "
            f"{len(action_space)}"
        )

    legality = compute_action_mask(graph, constraints)
    legal_flags = torch.tensor(
        [_is_action_legal(legality, action) for action in action_space],
        device=logits.device,
        dtype=torch.bool,
    )
    legal_actions = tuple(
        action for action, is_legal in zip(action_space, legal_flags.tolist()) if is_legal
    )
    masked_logits = logits / temperature
    masked_logits = masked_logits.masked_fill(~legal_flags, float("-inf"))
    return masked_logits, legal_actions


def _is_action_legal(mask: Mapping[Any, bool], action: ActionName) -> bool:
    if action in mask:
        return bool(mask[action])
    if action.value in mask:
        return bool(mask[action.value])
    return bool(mask.get(action.value.lower(), False))


__all__ = ["MaskedActionSelector", "mask_action_logits", "select_action"]
