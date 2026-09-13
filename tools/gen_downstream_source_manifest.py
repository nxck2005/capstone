#!/usr/bin/env python3
"""Freeze the final post-search execution-source successor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from runtime.source_guard import build_downstream_manifest  # noqa: E402

TARGET = REPO / "results/learned/w9/downstream_source_manifest_v4.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    source_commit = args.source_commit or head
    value = build_downstream_manifest(REPO, source_commit=source_commit)
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if TARGET.exists() or TARGET.is_symlink():
        if TARGET.is_symlink() or TARGET.read_bytes() != raw:
            raise SystemExit(f"immutable downstream source manifest differs: {TARGET}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(raw)
    print(f"downstream source manifest: {value['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
