#!/usr/bin/env python3
"""Build one W10 downstream source manifest epoch at the exact clean HEAD.

``--epoch v1`` reproduces the historical AM-98 freeze; the default ``v2`` builds
the corrected AM-99 successor over the existing v1 bytes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write  # noqa: E402
from runtime.source_epochs import (  # noqa: E402
    W10_SOURCE_PATH,
    W10_V2_SOURCE_PATH,
    build_w10_manifest,
    build_w10_manifest_v2,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", choices=("v1", "v2"), default="v2")
    args = parser.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    if args.epoch == "v1":
        value = build_w10_manifest(REPO, source_commit=head)
        target = REPO / W10_SOURCE_PATH
    else:
        value = build_w10_manifest_v2(REPO, source_commit=head)
        target = REPO / W10_V2_SOURCE_PATH
    immutable_write(target, value)
    print(f"W10 downstream source manifest {args.epoch}: {value['manifest_id']} @ {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
