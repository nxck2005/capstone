#!/usr/bin/env python3
"""Verify the immutable AM-96 ER-9 semantic freeze."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from evaluation.am96_spec_compatibility import (  # noqa: E402
    AM96SpecCompatibilityError,
    load,
)
from verify_g10_w9 import verify as verify_g10_terminal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-downstream",
        action="store_true",
        help="also authenticate the frozen contract after downstream evidence exists",
    )
    args = parser.parse_args()
    try:
        value = load(REPO, allow_downstream=args.allow_downstream)
        terminal = verify_g10_terminal(REPO)
        evidence = value["g10_terminal_evidence"]
        adjudication = terminal["adjudication"]
        if adjudication["classification"] != evidence["classification"]:
            raise AM96SpecCompatibilityError("AM-96/G-10 classification binding differs")
        if adjudication["events"] != [
            {
                "direction": "positive_to_negative",
                "first_snr_db": -5,
                "last_snr_db": -4,
                "location_kind": "measured_bracket",
            }
        ]:
            raise AM96SpecCompatibilityError("AM-96/G-10 headline bracket binding differs")
    except (AM96SpecCompatibilityError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"AM-96 HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "AM-96 semantic freeze PASS: "
        f"{value['freeze_id']} closed_items={value['audit']['closed_items']} "
        "er9_training=0 er2_randomized_training=0 test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
