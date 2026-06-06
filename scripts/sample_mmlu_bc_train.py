#!/usr/bin/env python3
"""Sample a balanced MMLU subset for BC teacher trajectory generation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    source_files = sorted(source_dir.glob(f"*_{args.split}.csv"))
    if not source_files:
        raise RuntimeError(f"no MMLU files found in {source_dir} for split={args.split!r}")

    subject_rows = {subject_name(path, args.split): read_csv_rows(path) for path in source_files}
    total_capacity = sum(len(rows) for rows in subject_rows.values())
    if args.total > total_capacity:
        raise RuntimeError(
            f"requested {args.total} samples, but only {total_capacity} rows are available"
        )

    counts = balanced_counts(
        subjects=tuple(sorted(subject_rows)),
        capacities={subject: len(rows) for subject, rows in subject_rows.items()},
        total=args.total,
        seed=args.seed,
    )
    sampled = sample_rows(subject_rows, counts=counts, seed=args.seed)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "seed": args.seed,
        "requested_total": args.total,
        "actual_total": sum(len(rows) for rows in sampled.values()),
        "subject_count": len(subject_rows),
        "per_subject": {subject: len(rows) for subject, rows in sorted(sampled.items())},
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    for subject, rows in sorted(sampled.items()):
        write_csv_rows(output_dir / f"{subject}_{args.split}.csv", rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU" / "data" / "test",
        help="directory containing original MMLU *_test.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "bc_train_test150" / "test",
        help="directory to write sampled MMLU CSV files",
    )
    parser.add_argument("--split", default="test", help="MMLU split suffix to sample")
    parser.add_argument("--total", type=int, default=150, help="total rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="deterministic sampling seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output directory if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sampling plan without writing files",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle)]


def write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def subject_name(path: Path, split: str) -> str:
    suffix = f"_{split}.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected MMLU filename for split={split!r}: {path.name}")
    return path.name[: -len(suffix)]


def balanced_counts(
    *,
    subjects: tuple[str, ...],
    capacities: dict[str, int],
    total: int,
    seed: int,
) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not subjects:
        raise ValueError("subjects must not be empty")

    rng = random.Random(seed)
    base = total // len(subjects)
    remainder = total % len(subjects)
    counts = {subject: min(base, capacities[subject]) for subject in subjects}

    shuffled_subjects = list(subjects)
    rng.shuffle(shuffled_subjects)
    for subject in shuffled_subjects[:remainder]:
        if counts[subject] < capacities[subject]:
            counts[subject] += 1

    leftover = total - sum(counts.values())
    while leftover > 0:
        candidates = [subject for subject in subjects if counts[subject] < capacities[subject]]
        if not candidates:
            raise RuntimeError("could not allocate all requested samples")
        rng.shuffle(candidates)
        for subject in candidates:
            if leftover == 0:
                break
            counts[subject] += 1
            leftover -= 1
    return counts


def sample_rows(
    subject_rows: dict[str, list[list[str]]],
    *,
    counts: dict[str, int],
    seed: int,
) -> dict[str, list[list[str]]]:
    rng = random.Random(seed)
    output: dict[str, list[list[str]]] = {}
    for subject in sorted(subject_rows):
        rows = subject_rows[subject]
        count = counts[subject]
        indices = sorted(rng.sample(range(len(rows)), count))
        output[subject] = [rows[index] for index in indices]
    return output


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise RuntimeError(f"output directory already exists: {output_dir}")
        if output_dir in {Path("/"), output_dir.parent}:
            raise RuntimeError(f"refusing to overwrite unsafe output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


if __name__ == "__main__":
    main()
