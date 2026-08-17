#!/usr/bin/env python3
"""Generate or check the checkout-portable Pascal scientific runtime manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_portable import (  # noqa: E402
    PORTABLE_MANIFEST_PATH,
    build_scientific_manifest,
)
from baseline.g8_pascal_merge import SUCCESSOR_RUNTIME_ROOT  # noqa: E402
from baseline.g8_campaign import rendered_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=SUCCESSOR_RUNTIME_ROOT)
    args = parser.parse_args()
    expected = rendered_json(build_scientific_manifest(args.runtime_root))
    if args.check:
        try:
            actual = PORTABLE_MANIFEST_PATH.read_bytes()
        except OSError as exc:
            print(f"FAIL: cannot read {PORTABLE_MANIFEST_PATH}: {exc}", file=sys.stderr)
            return 1
        if actual != expected:
            print(f"FAIL: {PORTABLE_MANIFEST_PATH} is stale", file=sys.stderr)
            return 1
        print(f"PASS: {PORTABLE_MANIFEST_PATH.relative_to(REPO)} is current")
        return 0
    PORTABLE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTABLE_MANIFEST_PATH.write_bytes(expected)
    print(f"PASS: wrote {PORTABLE_MANIFEST_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
