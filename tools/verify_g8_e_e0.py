#!/usr/bin/env python3
"""Independently verify the G8_E E0 opening witness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_e import E0_PATH, G8EContractError, verify_e0_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=E0_PATH)
    args = parser.parse_args()
    try:
        value = verify_e0_file(args.path)
    except G8EContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print({"status": "PASS", "artifact_id": value["artifact_id"], "coverage": value["safety"]["g8_e_measurement_coverage"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
