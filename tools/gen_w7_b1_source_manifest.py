#!/usr/bin/env python3
"""Generate the additive W7-B1 scientific source authority."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from training.deterministic_core import canonical_bytes  # noqa: E402
from verify_w7_b1 import B1_SOURCE_PATH, build_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=B1_SOURCE_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_source(args.source_commit)
    raw = canonical_bytes(value)
    if args.check:
        if args.output.read_bytes() != raw:
            raise SystemExit("W7-B1 source manifest is stale")
        print(f"W7-B1 source manifest PASS: {value['manifest_id']}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"wrote {args.output}: {value['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
