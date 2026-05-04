#!/usr/bin/env python3
"""Generate a CSV of shuffled SAM-DD samples with subject-aware metadata."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten the SAM-DD folder hierarchy into a labels CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing Tester*/<class>/<view> folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path (e.g., Dataset/samdd_labels.csv).",
    )
    parser.add_argument(
        "--subject-glob",
        type=str,
        default="Tester*",
        help="Glob pattern (relative to --data-dir) used to find subject folders.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of allowed image extensions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling the exported rows.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before writing the CSV.",
    )
    return parser.parse_args()


def _collect_rows(
    root: Path,
    subject_glob: str,
    extensions: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []
    for subject_dir in sorted(root.glob(subject_glob)):
        if not subject_dir.is_dir():
            continue
        subject_name = subject_dir.name
        for class_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            classname = class_dir.name
            for view_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
                view_name = view_dir.name
                for img_path in sorted(view_dir.iterdir()):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in extensions:
                        continue
                    rel_path = img_path.relative_to(root).as_posix()
                    rows.append(
                        {
                            "subject": subject_name,
                            "classname": classname,
                            "view": view_name,
                            "img": img_path.name,
                            "filepath": rel_path,
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext.strip())
    if not extensions:
        raise ValueError("At least one file extension must be provided.")

    rows = _collect_rows(data_dir, args.subject_glob, extensions)
    if not rows:
        raise RuntimeError(f"No images found under '{data_dir}' using glob '{args.subject_glob}'.")

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    fieldnames = ["subject", "classname", "view", "img", "filepath"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to '{output_path}'.")


if __name__ == "__main__":
    main()
