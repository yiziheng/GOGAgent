#!/usr/bin/env python3
"""Sample a normalized MMLU-Pro JSONL subset for eval or policy training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    args = parse_args()

    from gogagent.datasets.mmlu_pro import load_mmlu_pro_jsonl

    examples = list(load_mmlu_pro_jsonl(args.source))
    if args.total <= 0:
        raise ValueError("--total must be positive")
    if args.total > len(examples):
        raise RuntimeError(
            f"requested {args.total} examples, but source only has {len(examples)}"
        )

    selected = (
        sample_balanced_by_subject(examples, total=args.total, seed=args.seed)
        if args.balanced_by_subject
        else random.Random(args.seed).sample(examples, args.total)
    )
    selected = sorted(selected, key=lambda example: str(example.public_task.get("task_id", "")))

    output_root = args.output_root.resolve()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "output_root": str(output_root),
        "output_split": args.output_split,
        "seed": args.seed,
        "requested_total": args.total,
        "actual_total": len(selected),
        "source_total": len(examples),
        "balanced_by_subject": args.balanced_by_subject,
        "per_subject": subject_counts(selected),
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    prepare_output_root(output_root, overwrite=args.overwrite)
    subset_path = output_root / f"{args.output_split}.jsonl"
    with subset_path.open("w", encoding="utf-8") as handle:
        for example in selected:
            handle.write(json.dumps(row_for_output(example), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    manifest["subset_jsonl"] = str(subset_path)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Full MMLU-Pro JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Subset root; <output-split>.jsonl and manifest.json are written here.",
    )
    parser.add_argument("--output-split", default="test")
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--balanced-by-subject",
        action="store_true",
        help="Spread samples as evenly as possible across subject/category.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sample_balanced_by_subject(examples: list[Any], *, total: int, seed: int) -> list[Any]:
    """Sample approximately evenly across subject/category buckets."""

    rng = random.Random(seed)
    buckets: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        buckets[subject_key(example)].append(example)
    subjects = sorted(buckets)
    if not subjects:
        raise RuntimeError("no MMLU-Pro subjects found")

    selected: list[Any] = []
    base = total // len(subjects)
    remainder = total % len(subjects)
    shuffled_subjects = list(subjects)
    rng.shuffle(shuffled_subjects)
    target_counts = {
        subject: min(len(buckets[subject]), base)
        for subject in subjects
    }
    for subject in shuffled_subjects[:remainder]:
        if target_counts[subject] < len(buckets[subject]):
            target_counts[subject] += 1

    leftover = total - sum(target_counts.values())
    while leftover > 0:
        candidates = [
            subject
            for subject in subjects
            if target_counts[subject] < len(buckets[subject])
        ]
        if not candidates:
            raise RuntimeError("could not allocate all requested MMLU-Pro samples")
        rng.shuffle(candidates)
        for subject in candidates:
            if leftover == 0:
                break
            target_counts[subject] += 1
            leftover -= 1

    for subject in subjects:
        selected.extend(rng.sample(buckets[subject], target_counts[subject]))
    return selected


def row_for_output(example: Any) -> dict[str, Any]:
    task = dict(example.public_task)
    return {
        "task_id": task.get("task_id"),
        "subject": task.get("subject"),
        "category": task.get("category"),
        "question": task.get("question"),
        "options": task.get("options"),
        "answer": example.gold,
        "source_file": task.get("source_file"),
        "source_row": task.get("source_row"),
    }


def subject_counts(examples: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[subject_key(example)] += 1
    return dict(sorted(counts.items()))


def subject_key(example: Any) -> str:
    task = example.public_task
    return str(task.get("subject") or task.get("category") or "unknown")


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"output root already exists: {output_root}")
        if output_root in {Path("/"), output_root.parent}:
            raise RuntimeError(f"refusing to overwrite unsafe output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


if __name__ == "__main__":
    main()
