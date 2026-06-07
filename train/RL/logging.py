"""Backward-compatible JSONL/JSON logging helpers for RL runs."""

from __future__ import annotations

from gogagent.artifacts import append_jsonl, write_json

__all__ = ["append_jsonl", "write_json"]
