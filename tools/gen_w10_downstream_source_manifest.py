#!/usr/bin/env python3
"""Build the AM-98 successor downstream source manifest at the exact clean HEAD."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write  # noqa: E402
from runtime.source_epochs import W10_SOURCE_PATH, build_w10_manifest  # noqa: E402


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    value = build_w10_manifest(REPO, source_commit=head)
    immutable_write(REPO / W10_SOURCE_PATH, value)
    print(f"W10 downstream source manifest: {value['manifest_id']} @ {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
