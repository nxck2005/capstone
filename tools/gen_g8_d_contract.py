#!/usr/bin/env python3
"""Generate or check the deterministic G8_D D4 pre-data contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_d import build_g8_d_contract, rendered_json  # noqa: E402


OUT = REPO / "results/baseline/g8_d/measurement_contract.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered_json(build_g8_d_contract(REPO))
    if args.check:
        try:
            actual = OUT.read_bytes()
        except OSError as exc:
            print(f"FAIL: cannot read {OUT}: {exc}", file=sys.stderr)
            return 1
        if actual != expected:
            print(f"FAIL: {OUT} is stale", file=sys.stderr)
            return 1
        print(f"PASS: {OUT.relative_to(REPO)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(expected)
    print(f"PASS: wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
