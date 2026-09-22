#!/usr/bin/env python3
"""Build one W10 downstream source manifest epoch at the exact clean HEAD.

``--epoch v1`` reproduces the historical AM-98 freeze; ``v2`` is the historical
AM-99 successor, ``v3`` is the repaired pre-science successor, ``v4`` builds the
authority-generation repair over the exact v3 bytes, ``v5`` builds the
post-PAPR/pre-selection repair over the exact immutable v4 bytes, and ``v6``
freezes the completed JPEG evidence and post-JPEG/pre-ER-12 frontier.
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
    W10_V3_SOURCE_PATH,
    W10_V4_SOURCE_PATH,
    W10_V5_SOURCE_PATH,
    W10_V6_SOURCE_PATH,
    build_w10_manifest,
    build_w10_manifest_v2,
    build_w10_manifest_v3,
    build_w10_manifest_v4,
    build_w10_manifest_v5,
    build_w10_manifest_v6,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", choices=("v1", "v2", "v3", "v4", "v5", "v6"), default="v4")
    args = parser.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    if args.epoch == "v1":
        value = build_w10_manifest(REPO, source_commit=head)
        target = REPO / W10_SOURCE_PATH
    elif args.epoch == "v2":
        value = build_w10_manifest_v2(REPO, source_commit=head)
        target = REPO / W10_V2_SOURCE_PATH
    elif args.epoch == "v3":
        value = build_w10_manifest_v3(REPO, source_commit=head)
        target = REPO / W10_V3_SOURCE_PATH
    elif args.epoch == "v4":
        value = build_w10_manifest_v4(REPO, source_commit=head)
        target = REPO / W10_V4_SOURCE_PATH
    elif args.epoch == "v5":
        value = build_w10_manifest_v5(REPO, source_commit=head)
        target = REPO / W10_V5_SOURCE_PATH
    else:
        value = build_w10_manifest_v6(REPO, source_commit=head)
        target = REPO / W10_V6_SOURCE_PATH
    immutable_write(target, value)
    print(f"W10 downstream source manifest {args.epoch}: {value['manifest_id']} @ {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
