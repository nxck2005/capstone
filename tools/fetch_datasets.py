#!/usr/bin/env python3
"""Fetch, measure, verify, and extract the three normative dataset archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from data.provenance import (  # noqa: E402
    configured_datasets,
    fetch_archive,
    provision_archives,
    verify_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--measure",
        action="store_true",
        help="fetch and measure archives even while configured hashes are pending",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify existing archives without network access or extraction",
    )
    return parser


def _print(provenance) -> None:
    print(
        f"{provenance.dataset}: url={provenance.url} "
        f"filename={provenance.filename} bytes={provenance.byte_length} "
        f"sha256={provenance.sha256}"
    )


def main() -> int:
    args = _parser().parse_args()
    if args.check:
        for dataset in configured_datasets():
            _print(verify_archive(dataset, REPO))
        return 0
    if args.measure:
        for dataset in configured_datasets():
            _print(fetch_archive(dataset, REPO))
        return 0
    for provenance in provision_archives(REPO, measure_only=False):
        _print(provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
