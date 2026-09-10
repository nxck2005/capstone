#!/usr/bin/env python3
"""Create the full-tree W9 v4 scientific source manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from runtime.source_guard import build_manifest  # noqa: E402


TARGET = REPO / "results/learned/er9/er_execution_source_manifest_v4.json"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", default=None, help="full source commit; defaults to HEAD")
    args = parser.parse_args(argv)
    source_commit = args.source_commit or _git("rev-parse", "HEAD")
    if len(source_commit) != 40:
        raise SystemExit("--source-commit must be a full Git commit SHA")
    manifest = build_manifest(
        REPO,
        source_commit=source_commit,
        relevant_config_paths=(
            "configs/er9-digital-pascal-v4.yaml",
            "configs/learned-er2-randomized-pascal-v4.yaml",
            "spec/params.generated.yaml",
        ),
    )
    body = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if TARGET.exists():
        if TARGET.read_text(encoding="utf-8") != body:
            raise SystemExit(f"immutable v4 source manifest differs: {TARGET}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(body, encoding="ascii")
    print(f"W9 v4 source manifest: {manifest['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
