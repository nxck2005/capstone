#!/usr/bin/env python3
"""Write or byte-check deterministic committed dataset split manifests."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from data.manifests import (  # noqa: E402
    check_manifest,
    manifest_path,
    validate_manifest_bytes,
    write_manifest,
)
from data.provenance import configured_datasets  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and compare exact bytes with committed files",
    )
    return parser


def _counts(dataset: str) -> str:
    rows = validate_manifest_bytes(dataset, manifest_path(dataset, REPO).read_bytes())
    counts = Counter(row.split for row in rows)
    return " ".join(
        f"{split}={counts[split]}" for split in ("train", "val", "test")
    )


def main() -> int:
    args = _parser().parse_args()
    for dataset in configured_datasets():
        if args.check:
            sha256 = check_manifest(dataset, REPO)
            path = manifest_path(dataset, REPO)
        else:
            path, sha256 = write_manifest(dataset, REPO)
        print(
            f"{dataset}: path={path.relative_to(REPO)} {_counts(dataset)} "
            f"sha256={sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
