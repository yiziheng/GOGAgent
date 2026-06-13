#!/usr/bin/env python3
"""Sample MMLU val rows after the GPTSwarm153 prefix.

The GPTSwarm-compatible evaluation subset is produced by sorting MMLU val CSV
files, concatenating all rows, applying ``numpy.default_rng(888).permutation``,
and taking the first 153 rows. This script uses the same shuffled order, skips
that prefix by default, and writes the following rows as a BC training subset.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 888
DEFAULT_OFFSET = 153


@dataclass(frozen=True)
class MMLURow:
    """One canonical MMLU CSV row with source metadata."""

    subject: str
    split: str
    source_file: Path
    source_row: int
    values: tuple[str, str, str, str, str, str]

    @property
    def question(self) -> str:
        return self.values[0]

    @property
    def options(self) -> dict[str, str]:
        return dict(zip(("A", "B", "C", "D"), self.values[1:5], strict=True))

    @property
    def answer(self) -> str:
        return self.values[5]

    @property
    def task_id(self) -> str:
        return f"{self.subject}-{self.split}-{self.source_row}"


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source_dir, args.split)
    output_root = args.output_root.resolve()
    output_split_dir = output_root / args.output_split

    rows = load_ordered_rows(source_dir, split=args.split)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.total <= 0:
        raise ValueError("total must be positive")
    if args.offset + args.total > len(rows):
        raise RuntimeError(
            f"requested offset+total={args.offset + args.total}, "
            f"but source only has {len(rows)} rows"
        )

    shuffled_indices = shuffled_order(len(rows), seed=args.seed)
    excluded_content_keys = load_excluded_content_keys(args.exclude_selection_jsonl)
    selected_indices = select_indices(
        rows,
        shuffled_indices,
        offset=args.offset,
        total=args.total,
        excluded_content_keys=excluded_content_keys,
    )
    excluded_indices = shuffled_indices[: args.offset]
    selected_rows = [rows[index] for index in selected_indices]
    excluded_rows = [rows[index] for index in excluded_indices]
    check_disjoint(selected_rows, excluded_rows)

    manifest = build_manifest(
        source_dir=source_dir,
        output_root=output_root,
        output_split=args.output_split,
        split=args.split,
        seed=args.seed,
        offset=args.offset,
        total=args.total,
        source_total=len(rows),
        selected_rows=selected_rows,
        excluded_rows=excluded_rows,
        excluded_content_count=len(excluded_content_keys),
    )

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    prepare_output_root(output_root, overwrite=args.overwrite)
    write_subset_csvs(output_split_dir, selected_rows, output_split=args.output_split)
    selection_path = output_root / "selection.jsonl"
    write_selection_jsonl(selection_path, selected_rows, offset=args.offset)
    manifest["selection_jsonl_sha256"] = sha256_file(selection_path)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing original MMLU *_val.csv files. If omitted, "
            "common local layouts under data/MMLU are tried."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "bc_val_after_gptswarm153",
        help="Subset root; manifest.json, selection.jsonl, and <output-split>/ are written here.",
    )
    parser.add_argument("--split", default="val", help="Source MMLU split suffix")
    parser.add_argument("--output-split", default="val", help="Output CSV split suffix")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_OFFSET,
        help="How many shuffled rows to skip; 153 skips GPTSwarm153.",
    )
    parser.add_argument("--total", type=int, default=300, help="Number of rows to sample")
    parser.add_argument(
        "--exclude-selection-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "MMLU_subsets" / "gptswarm153" / "selection.jsonl",
        help=(
            "Optional selection.jsonl whose exact question/options/answer content "
            "is excluded while scanning after offset."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output root")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest only")
    return parser.parse_args()


def resolve_source_dir(source_dir: Path | None, split: str) -> Path:
    """Return a source directory containing ``*_<split>.csv`` files."""

    candidates = [source_dir] if source_dir is not None else [
        REPO_ROOT / "data" / "MMLU" / "data" / split,
        REPO_ROOT / "data" / "MMLU" / split,
        REPO_ROOT / "data" / "MMLU" / "data" / "validation",
        REPO_ROOT / "data" / "MMLU" / "validation",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if list(resolved.glob(f"*_{split}.csv")):
            return resolved
    searched = "\n".join(str(path) for path in candidates if path is not None)
    raise RuntimeError(f"could not find MMLU *_{split}.csv files. Searched:\n{searched}")


def load_ordered_rows(source_dir: Path, *, split: str) -> list[MMLURow]:
    """Load all rows by sorted CSV path, matching the GPTSwarm153 algorithm."""

    rows: list[MMLURow] = []
    suffix = f"_{split}.csv"
    source_files = sorted(source_dir.glob(f"*{suffix}"))
    if not source_files:
        raise RuntimeError(f"no MMLU files found in {source_dir} for split={split!r}")
    for path in source_files:
        subject = subject_name(path, split)
        with path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle)):
                if len(row) < 6:
                    raise ValueError(f"{path}:{row_index + 1} expected 6 columns, got {len(row)}")
                rows.append(
                    MMLURow(
                        subject=subject,
                        split=split,
                        source_file=path,
                        source_row=row_index + 1,
                        values=tuple(str(value) for value in row[:6]),  # type: ignore[arg-type]
                    )
                )
    return rows


def shuffled_order(size: int, *, seed: int) -> list[int]:
    """Return numpy.default_rng(seed).permutation(size) as Python ints."""

    return [int(index) for index in np.random.default_rng(seed).permutation(size)]


def select_indices(
    rows: list[MMLURow],
    shuffled_indices: list[int],
    *,
    offset: int,
    total: int,
    excluded_content_keys: set[tuple[str, str, str, str, str, str]],
) -> list[int]:
    """Select rows after offset, skipping exact content exclusions."""

    selected: list[int] = []
    for index in shuffled_indices[offset:]:
        if content_key(rows[index]) in excluded_content_keys:
            continue
        selected.append(index)
        if len(selected) == total:
            return selected
    raise RuntimeError(
        f"could only select {len(selected)} rows after offset={offset}; requested {total}"
    )


def check_disjoint(selected_rows: Iterable[MMLURow], excluded_rows: Iterable[MMLURow]) -> None:
    """Ensure selected rows do not overlap the skipped GPTSwarm prefix."""

    selected_ids = {(row.subject, row.source_row) for row in selected_rows}
    excluded_ids = {(row.subject, row.source_row) for row in excluded_rows}
    overlap = selected_ids & excluded_ids
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(f"sample overlaps excluded prefix: {preview}")


def build_manifest(
    *,
    source_dir: Path,
    output_root: Path,
    output_split: str,
    split: str,
    seed: int,
    offset: int,
    total: int,
    source_total: int,
    selected_rows: list[MMLURow],
    excluded_rows: list[MMLURow],
    excluded_content_count: int,
) -> dict[str, Any]:
    per_subject: dict[str, int] = {}
    for row in selected_rows:
        per_subject[row.subject] = per_subject.get(row.subject, 0) + 1
    return {
        "algorithm": (
            "Sort MMLU val CSV paths, concatenate rows, apply "
            f"numpy.default_rng({seed}).permutation, skip first {offset}, "
            f"then take the next {total} rows after excluding exact content "
            "matches from the configured exclusion selection."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": output_root.name,
        "purpose": "BC training subset after the GPTSwarm153 validation prefix.",
        "project_root": str(REPO_ROOT),
        "source_dir": str(source_dir),
        "source_split": split,
        "source_total": source_total,
        "output_root": str(output_root),
        "output_split": output_split,
        "seed": seed,
        "offset": offset,
        "excluded_prefix_count": len(excluded_rows),
        "excluded_content_count": excluded_content_count,
        "requested_total": total,
        "total": len(selected_rows),
        "subject_count": len(per_subject),
        "per_subject": dict(sorted(per_subject.items())),
        "selected_task_ids": [row.task_id for row in selected_rows],
        "excluded_prefix_task_ids": [row.task_id for row in excluded_rows],
    }


def write_subset_csvs(output_split_dir: Path, rows: list[MMLURow], *, output_split: str) -> None:
    grouped: dict[str, list[MMLURow]] = {}
    for row in rows:
        grouped.setdefault(row.subject, []).append(row)
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for subject, subject_rows in sorted(grouped.items()):
        output_path = output_split_dir / f"{subject}_{output_split}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(row.values for row in subject_rows)


def write_selection_jsonl(path: Path, rows: list[MMLURow], *, offset: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for rank, row in enumerate(rows):
            record = {
                "rank": rank,
                "shuffle_rank": offset + rank,
                "task_id": row.task_id,
                "subject": row.subject,
                "source_split": row.split,
                "source_file": relative_path(row.source_file),
                "source_row": row.source_row,
                "question": row.question,
                "options": row.options,
                "answer": row.answer,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_excluded_content_keys(path: Path | None) -> set[tuple[str, str, str, str, str, str]]:
    """Load exact question/options/answer keys to exclude from an existing selection."""

    if path is None or not path.exists():
        return set()
    keys: set[tuple[str, str, str, str, str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} expected JSON object")
            options = row.get("options") or {}
            if not isinstance(options, dict):
                raise ValueError(f"{path}:{line_number} expected options object")
            keys.add(
                (
                    str(row.get("question", "")),
                    str(row.get("answer", "")),
                    str(options.get("A", "")),
                    str(options.get("B", "")),
                    str(options.get("C", "")),
                    str(options.get("D", "")),
                )
            )
    return keys


def content_key(row: MMLURow) -> tuple[str, str, str, str, str, str]:
    return (
        row.question,
        row.answer,
        row.options["A"],
        row.options["B"],
        row.options["C"],
        row.options["D"],
    )


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"output root already exists: {output_root}")
        if output_root == REPO_ROOT or REPO_ROOT not in output_root.parents:
            raise RuntimeError(f"refusing to overwrite unsafe output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_name(path: Path, split: str) -> str:
    suffix = f"_{split}.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected MMLU filename for split={split!r}: {path.name}")
    return path.name[: -len(suffix)]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
