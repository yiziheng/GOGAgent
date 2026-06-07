"""Run-directory and path helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil


def prepare_run_dir(output_dir: str | Path, run_id: str | None, *, overwrite: bool) -> Path:
    """Create or replace one run directory."""

    root = Path(output_dir)
    run_dir = root / (run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"run directory already exists: {run_dir}; pass --overwrite or choose --run-id"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_path(value: str) -> str:
    """Return a compact filesystem-safe path component."""

    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return cleaned[:96] or "item"
