"""OpenAI-compatible chat completions backend using only the standard library."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Mapping
from urllib import error, parse, request

from gogagent.llm.base import LLMBackend, LLMResponse


_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatibleLLM(LLMBackend):
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = _require_non_empty("model", model)
        self._api_key = api_key
        self.timeout = _require_positive("timeout", timeout)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.temperature = temperature
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        self.max_tokens = max_tokens

    def generate(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        """Generate text and return provider-reported token usage."""

        started_at = time.monotonic()
        payload = self._build_payload(role, prompt, context)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            try:
                response_payload = self._post(body)
                return _parse_response(
                    response_payload,
                    fallback_model=self.model,
                    latency_seconds=time.monotonic() - started_at,
                )
            except error.HTTPError as exc:
                if exc.code not in _RETRYABLE_HTTP_STATUSES or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"chat completions request failed with HTTP status {exc.code}"
                    ) from exc
            except (error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("chat completions request failed after retries") from exc
            time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable")

    def describe(self) -> Mapping[str, Any]:
        """Return artifact-safe backend settings without the API key."""

        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self._api_key),
        }

    def _build_payload(
        self,
        role: str,
        prompt: str,
        context: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        user_content = prompt
        if context:
            serialized_context = json.dumps(dict(context), ensure_ascii=False, sort_keys=True)
            user_content = f"Context:\n{serialized_context}\n\nPrompt:\n{prompt}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are the {role} agent."},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _post(self, body: bytes) -> Mapping[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        chat_request = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(chat_request, timeout=self.timeout) as response:
            response_body = response.read()
        try:
            payload = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("chat completions endpoint returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("chat completions endpoint returned a non-object response")
        return payload


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


def _parse_response(
    payload: Mapping[str, Any],
    *,
    fallback_model: str,
    latency_seconds: float,
) -> LLMResponse:
    try:
        choices = payload["choices"]
        text = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("chat completions response is missing message content") from exc
    if not isinstance(text, str):
        raise RuntimeError("chat completions message content must be a string")

    usage = payload.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens", prompt_tokens + completion_tokens)
    model = payload.get("model", fallback_model)
    return LLMResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model if isinstance(model, str) else fallback_model,
        latency_seconds=latency_seconds,
    )


def _usage_value(usage: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = usage.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default
