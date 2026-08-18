#!/usr/bin/env python3
"""E2 execution boundary; intentionally closed during the E0/E1 checkpoint."""

from __future__ import annotations

import argparse
import sys


def refuse_e2_execution() -> None:
    raise RuntimeError(
        "G8_E E2 is closed during the pre-data freeze; owner execution authorization "
        "and the E2 implementation checkpoint are required. No measurement was run."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args
    try:
        refuse_e2_execution()
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
