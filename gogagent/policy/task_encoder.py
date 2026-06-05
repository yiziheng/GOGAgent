"""SentenceTransformer task/text encoder for policy inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_ENV_VAR = "GOGAGENT_TASK_ENCODER_MODEL"


@dataclass
class SentenceTransformerTaskEncoder:
    """Encode task text with a local SentenceTransformer model."""

    model_name: str | None = None
    device: str | None = None
    normalize_embeddings: bool = True
    _model: SentenceTransformer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        model_name = self.model_name or default_model_name()
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=self.device)

    @property
    def embedding_dim(self) -> int:
        """Return the SentenceTransformer output width."""

        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            probe = self.encode_text("dimension probe")
            return int(probe.numel())
        return int(dimension)

    def encode_text(
        self,
        text: str,
        *,
        device: torch.device | str | None = None,
    ) -> torch.FloatTensor:
        """Encode text into one torch FloatTensor with model-native dimension."""

        embedding = self._model.encode(
            str(text),
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
        embedding = embedding.detach().float().flatten()
        if device is not None:
            embedding = embedding.to(device=device)
        return embedding

    def encode_task(
        self,
        task: Any,
        *,
        device: torch.device | str | None = None,
    ) -> torch.FloatTensor:
        """Encode any JSON-like task object through its generic text form."""

        return self.encode_text(task_to_text(task), device=device)


def default_model_name() -> str:
    """Return the configured task encoder model name."""

    return os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)


@lru_cache(maxsize=8)
def get_task_encoder(
    model_name: str | None = None,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> SentenceTransformerTaskEncoder:
    """Return a cached SentenceTransformerTaskEncoder."""

    return SentenceTransformerTaskEncoder(
        model_name=model_name or default_model_name(),
        device=device,
        normalize_embeddings=normalize_embeddings,
    )


def encode_text(
    text: str,
    *,
    model_name: str | None = None,
    device: torch.device | str | None = None,
    normalize_embeddings: bool = True,
) -> torch.FloatTensor:
    """Encode text using the cached default SentenceTransformer encoder."""

    encoder_device = _cache_device(device)
    encoder = get_task_encoder(
        model_name=model_name,
        device=encoder_device,
        normalize_embeddings=normalize_embeddings,
    )
    return encoder.encode_text(text, device=device)


def encode_task(
    task: Any,
    *,
    model_name: str | None = None,
    device: torch.device | str | None = None,
    normalize_embeddings: bool = True,
) -> torch.FloatTensor:
    """Encode a task object using the cached default SentenceTransformer encoder."""

    encoder_device = _cache_device(device)
    encoder = get_task_encoder(
        model_name=model_name,
        device=encoder_device,
        normalize_embeddings=normalize_embeddings,
    )
    return encoder.encode_task(task, device=device)


def task_embedding_dim(
    *,
    model_name: str | None = None,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> int:
    """Return the configured SentenceTransformer embedding dimension."""

    return get_task_encoder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
    ).embedding_dim


def task_to_text(task: Any) -> str:
    """Return a stable, generic text representation for task-like objects."""

    if isinstance(task, str):
        return task
    return json.dumps(task, sort_keys=True, default=str, ensure_ascii=False)


def _cache_device(device: torch.device | str | None) -> str | None:
    if device is None:
        return None
    return str(device)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "MODEL_ENV_VAR",
    "SentenceTransformerTaskEncoder",
    "default_model_name",
    "encode_task",
    "encode_text",
    "get_task_encoder",
    "task_embedding_dim",
    "task_to_text",
]
