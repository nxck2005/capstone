#!/usr/bin/env python3
"""Verify frozen Pascal G8_C evidence through portable scientific bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_merge import SUCCESSOR_RUNTIME_ROOT  # noqa: E402
from baseline.g8_pascal_portable import verify_portable_successor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=SUCCESSOR_RUNTIME_ROOT)
    args = parser.parse_args()
    try:
        result = verify_portable_successor(runtime_root=args.runtime_root)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    print("PASS — CLEAN-CHECKOUT PASCAL SCIENTIFIC RUNTIME VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
