"""Policy network heads for graph-construction decisions."""

from __future__ import annotations

import torch
from torch import nn

from gogagent.policy.action_space import action_count


class PolicyNetwork(nn.Module):
    """Map graph and task embeddings to flat action logits."""

    def __init__(
        self,
        graph_embedding_dim: int,
        task_embedding_dim: int,
        *,
        num_actions: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if graph_embedding_dim <= 0:
            raise ValueError("graph_embedding_dim must be positive")
        if task_embedding_dim <= 0:
            raise ValueError("task_embedding_dim must be positive")

        self.graph_embedding_dim = graph_embedding_dim
        self.task_embedding_dim = task_embedding_dim
        self.num_actions = action_count() if num_actions is None else num_actions
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")

        input_dim = graph_embedding_dim + task_embedding_dim
        hidden_dim = hidden_dim or input_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(
        self,
        graph_embedding: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return action logits with shape [num_actions] or [batch, num_actions]."""

        graph_embedding, task_embedding, squeezed = _align_embeddings(
            graph_embedding,
            task_embedding,
        )
        device, dtype = self._parameter_device_dtype()
        graph_embedding = graph_embedding.to(device=device, dtype=dtype)
        task_embedding = task_embedding.to(device=device, dtype=dtype)
        if graph_embedding.size(-1) != self.graph_embedding_dim:
            raise ValueError(
                "graph_embedding last dimension does not match graph_embedding_dim"
            )
        if task_embedding.size(-1) != self.task_embedding_dim:
            raise ValueError("task_embedding last dimension does not match task_embedding_dim")

        logits = self.head(torch.cat((graph_embedding, task_embedding), dim=-1))
        return logits.squeeze(0) if squeezed else logits

    def _parameter_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype


def _align_embeddings(
    graph_embedding: torch.Tensor,
    task_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if not isinstance(graph_embedding, torch.Tensor):
        graph_embedding = torch.as_tensor(graph_embedding, dtype=torch.float32)
    if not isinstance(task_embedding, torch.Tensor):
        task_embedding = torch.as_tensor(task_embedding, dtype=torch.float32)

    if graph_embedding.dim() == 1:
        graph_embedding = graph_embedding.unsqueeze(0)
        graph_squeezed = True
    elif graph_embedding.dim() == 2:
        graph_squeezed = False
    else:
        raise ValueError("graph_embedding must have shape [D] or [B, D]")

    if task_embedding.dim() == 1:
        task_embedding = task_embedding.unsqueeze(0).to(device=graph_embedding.device)
        task_squeezed = True
    elif task_embedding.dim() == 2:
        task_embedding = task_embedding.to(device=graph_embedding.device)
        task_squeezed = False
    else:
        raise ValueError("task_embedding must have shape [D] or [B, D]")

    if graph_embedding.size(0) != task_embedding.size(0):
        if graph_embedding.size(0) == 1:
            graph_embedding = graph_embedding.expand(task_embedding.size(0), -1)
            graph_squeezed = False
        elif task_embedding.size(0) == 1:
            task_embedding = task_embedding.expand(graph_embedding.size(0), -1)
            task_squeezed = False
        else:
            raise ValueError("graph_embedding and task_embedding batch sizes must match")

    return graph_embedding, task_embedding, graph_squeezed and task_squeezed


__all__ = ["PolicyNetwork"]
