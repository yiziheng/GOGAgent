"""Strict LLM client for GraphMessage-producing agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping
from urllib import parse


try:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
except ImportError as exc:  # pragma: no cover - exercised by environment setup.
    raise RuntimeError(
        "The openai SDK is required. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


@dataclass(frozen=True)
class LLMUsage:
    """Provider-reported token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMJsonResponse:
    """One strict JSON model response."""

    data: dict[str, Any]
    raw_text: str
    model: str
    usage: LLMUsage
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "raw_text": self.raw_text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class AgentContext:
    """Execution context passed from graph runtime to every Agent."""

    llm_client: "LLMClient"
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_llm_call(self, event: Mapping[str, Any]) -> None:
        """Record one node-level LLM call for audit artifacts."""

        self.llm_calls.append(dict(event))


class LLMClient(ABC):
    """Strict JSON LLM client used by production Agent execution."""

    @abstractmethod
    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        """Return one parsed JSON object or raise an explicit error."""

    def describe(self) -> Mapping[str, Any]:
        """Return credential-free metadata for artifacts."""

        return {"name": self.__class__.__name__}


class LLMClientError(RuntimeError):
    """Base class for strict LLM client errors."""


class LLMJsonError(LLMClientError):
    """Raised when the provider response is not a JSON object."""


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible chat completions client with strict JSON parsing."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking: str | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = _require_non_empty("model", model)
        api_key = _require_non_empty("api_key", api_key)
        self.timeout = _require_positive("timeout", timeout)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.temperature = temperature
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        self.max_tokens = max_tokens
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be 'enabled', 'disabled', or None")
        self.thinking = thinking
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def chat_json(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> LLMJsonResponse:
        started_at = time.monotonic()
        request_payload = self._build_payload(
            role=role,
            prompt=prompt,
            payload=payload,
            response_schema=response_schema,
            instruction=instruction,
        )
        try:
            response = self._client.chat.completions.create(**request_payload)
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            raise LLMClientError(
                "chat completions request failed through openai SDK: "
                f"{type(exc).__name__}: {_safe_error_message(exc)}"
            ) from exc
        return _parse_json_response(
            response.model_dump(mode="json"),
            default_model=self.model,
            latency_seconds=time.monotonic() - started_at,
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "api_key_configured": True,
        }

    def _build_payload(
        self,
        *,
        role: str,
        prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        schema = response_schema or {
            "sender": "string optional",
            "role": "string",
            "content": "string",
            "answer": "string or null",
            "confidence": "number or null",
            "notes": "object",
            "metadata": "object",
        }
        user_payload = {
            "agent_prompt": prompt,
            "payload": dict(payload),
            "required_output_schema": dict(schema),
            "instruction": instruction
            or "Return exactly one JSON object. Do not wrap it in markdown.",
        }
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are the {role} agent in a graph-of-graphs multi-agent system. "
                        "You must return a strict JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_payload["max_tokens"] = self.max_tokens
        if self.thinking is not None:
            request_payload["extra_body"] = {"thinking": {"type": self.thinking}}
        return request_payload


def _parse_json_response(
    payload: Mapping[str, Any],
    *,
    default_model: str,
    latency_seconds: float,
) -> LLMJsonResponse:
    try:
        choices = payload["choices"]
        text = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("chat completions response is missing message content") from exc
    if not isinstance(text, str):
        raise LLMClientError("chat completions message content must be a string")
    data = _extract_json_object(text)
    usage_payload = payload.get("usage", {})
    if not isinstance(usage_payload, Mapping):
        usage_payload = {}
    usage = LLMUsage(
        prompt_tokens=_usage_value(usage_payload, "prompt_tokens"),
        completion_tokens=_usage_value(usage_payload, "completion_tokens"),
        total_tokens=_usage_value(usage_payload, "total_tokens"),
    )
    model = payload.get("model", default_model)
    return LLMJsonResponse(
        data=data,
        raw_text=text,
        model=model if isinstance(model, str) else default_model,
        usage=usage,
        latency_seconds=latency_seconds,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMJsonError("provider response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMJsonError("provider response JSON must be an object")
    return parsed


def _normalize_base_url(base_url: str) -> str:
    normalized = _require_non_empty("base_url", base_url).rstrip("/")
    parsed = parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http or https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not include credentials, a query, or a fragment")
    return normalized


def _require_non_empty(name: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _safe_error_message(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    message = str(reason if reason is not None else exc).strip()
    return message or "<empty>"


def _usage_value(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
