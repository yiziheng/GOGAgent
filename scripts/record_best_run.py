"""Record best benchmark runs with enough version state to reproduce them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any


SOURCE_PATTERNS = (
    "gogagent/**/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "requirements.txt",
    "requirements-*.txt",
    "environment.yml",
    "pyproject.toml",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a new best benchmark run")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--eval-split", default="")
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--train-run-id", default="")
    parser.add_argument("--train-data", default="")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--registry-dir", type=Path, default=Path("artifacts/best_runs"))
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-tokens", default="")
    parser.add_argument("--thinking", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    summary_path = args.summary.resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    summary = _read_json(summary_path)
    metric_value = summary.get(args.metric)
    if metric_value is None:
        raise ValueError(f"summary missing metric {args.metric!r}: {summary_path}")
    score = float(metric_value)
    registry_dir = args.registry_dir
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "registry.json"
    registry = _read_json(registry_path) if registry_path.exists() else {"benchmarks": {}}

    benchmark_key = _benchmark_key(args.dataset, args.eval_data, args.eval_split)
    benchmarks = registry.setdefault("benchmarks", {})
    current = benchmarks.get(benchmark_key)
    previous_best = float(current["best_score"]) if current else None
    is_best = previous_best is None or score > previous_best
    run_record = _run_record(args, summary_path, score, previous_best, is_best)
    history = registry.setdefault("history", [])
    history.append(run_record)

    if is_best:
        record_dir = registry_dir / benchmark_key / run_record["record_id"]
        record_dir.mkdir(parents=True, exist_ok=True)
        run_record["record_directory"] = str(record_dir)
        _write_json(record_dir / "metadata.json", run_record)
        shutil.copy2(summary_path, record_dir / "summary.json")
        if args.policy_checkpoint is not None and args.policy_checkpoint.exists():
            shutil.copy2(args.policy_checkpoint, record_dir / "policy_checkpoint.pt")
        _write_json(record_dir / "source_manifest.json", _source_manifest(project_root))
        _write_source_snapshot(project_root, record_dir / "source_snapshot.tar.gz")
        _write_text(record_dir / "git_status.txt", _git("status", "--short") or "")
        _write_text(record_dir / "git_diff.patch", _git("diff", "--binary") or "")
        benchmarks[benchmark_key] = {
            "dataset": args.dataset,
            "eval_data": str(args.eval_data),
            "eval_split": args.eval_split,
            "metric": args.metric,
            "best_score": score,
            "record_id": run_record["record_id"],
            "record_directory": str(record_dir),
            "eval_run_id": args.eval_run_id,
            "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else "",
            "updated_at": run_record["recorded_at"],
        }
        print(
            f"[best-run] new best for {benchmark_key}: "
            f"{score:.6f} (previous={previous_best})"
        )
        print(f"[best-run] record directory: {record_dir}")
    else:
        print(
            f"[best-run] no new best for {benchmark_key}: "
            f"{score:.6f} <= {previous_best:.6f}"
        )

    _write_json(registry_path, registry)
    _write_leaderboard(registry_dir / "leaderboard.md", registry)


def _run_record(
    args: argparse.Namespace,
    summary_path: Path,
    score: float,
    previous_best: float | None,
    is_best: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "record_id": f"{now}-{_slug(args.eval_run_id)}-{score:.6f}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "is_best": is_best,
        "previous_best": previous_best,
        "metric": args.metric,
        "score": score,
        "dataset": args.dataset,
        "eval_data": str(args.eval_data),
        "eval_split": args.eval_split,
        "eval_run_id": args.eval_run_id,
        "summary_path": str(summary_path),
        "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else "",
        "train_run_id": args.train_run_id,
        "train_data": args.train_data,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "git": _git_metadata(),
    }


def _benchmark_key(dataset: str, eval_data: Path, split: str) -> str:
    data_name = eval_data.name
    parent_name = eval_data.parent.name
    if data_name in {"val", "test", "dev", "train"}:
        data_name = parent_name
    return _slug(f"{dataset}_{data_name}_{split or 'split'}")


def _git_metadata() -> dict[str, Any]:
    status = _git("status", "--short") or ""
    return {
        "commit": _git("rev-parse", "HEAD") or "",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def _source_manifest(project_root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in _source_paths(project_root):
        data = path.read_bytes()
        entries.append(
            {
                "path": str(path.relative_to(project_root)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    return entries


def _write_source_snapshot(project_root: Path, output_path: Path) -> None:
    with tarfile.open(output_path, "w:gz") as archive:
        for path in _source_paths(project_root):
            archive.add(path, arcname=str(path.relative_to(project_root)))


def _source_paths(project_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        paths.update(path for path in project_root.glob(pattern) if path.is_file())
    return sorted(paths)


def _write_leaderboard(path: Path, registry: dict[str, Any]) -> None:
    lines = [
        "# GOGAgent Best Runs",
        "",
        "| Benchmark | Metric | Best | Eval Run | Policy | Record | Updated |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for key, record in sorted(registry.get("benchmarks", {}).items()):
        lines.append(
            "| {key} | {metric} | {score:.6f} | {run} | {policy} | {record_id} | {updated} |".format(
                key=key,
                metric=record.get("metric", ""),
                score=float(record.get("best_score", 0.0)),
                run=record.get("eval_run_id", ""),
                policy=record.get("policy_checkpoint", ""),
                record_id=record.get("record_id", ""),
                updated=record.get("updated_at", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-")


if __name__ == "__main__":
    main()
