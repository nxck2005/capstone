#!/usr/bin/env python3
"""Classify a change set for the deliberately small hosted CI budget.

The evidence lane is an optimisation for committed G8_C evidence only.  It is
closed over an explicit path allowlist; an empty change set, an unknown path,
or any code/configuration/documentation change selects the software lane.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


EVIDENCE_FILE_PATHS = frozenset(
    {
        "results/baseline/g8/campaign_state.json",
        "instructions/RESUME.md",
        "worklogs/w4-classical-baseline-progress.md",
    }
)
EVIDENCE_DIRECTORY_PREFIXES = ("results/baseline/g8/work_units/",)


def _normalise(path: str) -> str:
    value = str(PurePosixPath(path))
    if value == ".":
        return ""
    return value.removeprefix("./")


def classify_paths(paths: list[str] | tuple[str, ...]) -> str:
    """Return ``evidence`` only for a non-empty, wholly allowlisted change."""

    normalised = tuple(_normalise(path) for path in paths if _normalise(path))
    if not normalised:
        return "software"
    for path in normalised:
        if path in EVIDENCE_FILE_PATHS:
            continue
        if any(path.startswith(prefix) for prefix in EVIDENCE_DIRECTORY_PREFIXES):
            continue
        return "software"
    return "evidence"


def changed_paths(*, base: str, head: str) -> list[str]:
    """Read names from a three-dot diff without interpreting file contents."""

    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMRTUXB", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paths", nargs="+", help="paths to classify")
    source.add_argument("--base", help="base commit for a three-dot git diff")
    parser.add_argument("--head", help="head commit for --base; defaults to HEAD")
    args = parser.parse_args(argv)
    if args.base is not None and args.head is None:
        args.head = "HEAD"
    paths = args.paths if args.paths is not None else changed_paths(base=args.base, head=args.head)
    print(classify_paths(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
