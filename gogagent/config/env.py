"""Load project configuration from a repository-local .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from gogagent.llm import OpenAICompatibleClient


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_THINKING = "disabled"


class EnvConfigError(RuntimeError):
    """Raised when .env-based runtime configuration is invalid."""


def project_root() -> Path:
    """Return the repository root for the local project layout."""

    return Path(__file__).resolve().parents[2]


def load_project_env(env_path: str | Path | None = None) -> Path:
    """Load environment variables from ``.env`` and return the loaded path."""

    path = Path(env_path) if env_path is not None else project_root() / ".env"
    if not path.exists():
        raise EnvConfigError(f"missing .env file: {path}")
    loaded = load_dotenv(path, override=True)
    if not loaded:
        raise EnvConfigError(f"failed to load .env file: {path}")
    return path


def require_env(name: str, env: Mapping[str, str] | None = None) -> str:
    """Return a required environment variable after .env has been loaded."""

    source = env if env is not None else os.environ
    value = source.get(name, "")
    if not value.strip():
        raise EnvConfigError(f"missing required environment variable {name!r}")
    return value.strip()


def llm_client_from_env(
    env_path: str | Path | None = None,
    *,
    prefix: str = "GOGAGENT",
) -> OpenAICompatibleClient:
    """Create the strict OpenAI-compatible client from .env variables."""

    load_project_env(env_path)
    return OpenAICompatibleClient(
        base_url=_env_or_default(f"{prefix}_BASE_URL", DEFAULT_BASE_URL),
        model=_env_or_default(f"{prefix}_MODEL", DEFAULT_MODEL),
        api_key=require_env(f"{prefix}_API_KEY"),
        timeout=_float_env(f"{prefix}_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_retries=_int_env(f"{prefix}_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        temperature=_float_env(f"{prefix}_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_optional_int_env(f"{prefix}_MAX_TOKENS"),
        thinking=_optional_choice_env(
            f"{prefix}_THINKING",
            allowed={None, "enabled", "disabled"},
            default=DEFAULT_THINKING,
        ),
    )


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be a float, got {value!r}") from exc


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be an integer, got {value!r}") from exc


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be an integer, got {value!r}") from exc


def _optional_choice_env(
    name: str,
    *,
    allowed: set[str | None],
    default: str | None = None,
) -> str | None:
    value = os.environ.get(name, "").strip()
    selected = value or default
    if selected not in allowed:
        choices = ", ".join(str(choice) for choice in sorted(allowed, key=str))
        raise EnvConfigError(f"{name} must be one of: {choices}")
    return selected
