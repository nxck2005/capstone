#!/usr/bin/env python3
"""Owner-authorized sole writer for the exact W9-A/G-10 matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.g10_protocol import AUTHORIZATION_PATH, G10ProtocolHold  # noqa: E402
from evaluation.g10_runner import G10ExecutionHold, run_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--authorization", default=str(AUTHORIZATION_PATH), type=Path)
    args = parser.parse_args()
    try:
        manifest = run_campaign(
            authorization_path=args.authorization,
            runtime_root=args.runtime_root,
            root=REPO,
        )
    except (G10ProtocolHold, G10ExecutionHold, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"G-10 EXECUTION HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "G-10 3x21 matrix PASS: "
        f"{manifest['runtime_manifest_id']} cells={manifest['matrix_shape']['cells']} "
        f"runtime={manifest['runtime_root']} test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
