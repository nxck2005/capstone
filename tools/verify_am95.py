#!/usr/bin/env python3
"""Verify the immutable AM-95 ER-2 semantic freeze and G-10 binding."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from evaluation.am95_spec_compatibility import (  # noqa: E402
    AM95SpecCompatibilityError,
    load,
)
from verify_g10_w9 import verify as verify_g10_terminal  # noqa: E402


def main() -> int:
    try:
        value = load(REPO)
        terminal = verify_g10_terminal(REPO)
        if terminal["adjudication"]["classification"] != value["g10_terminal_evidence"]["classification"]:
            raise AM95SpecCompatibilityError("AM-95/G-10 classification binding differs")
        if terminal["adjudication"]["events"] != [
            {"direction": "positive_to_negative", "first_snr_db": -5, "last_snr_db": -4, "location_kind": "measured_bracket"}
        ]:
            raise AM95SpecCompatibilityError("AM-95/G-10 headline bracket binding differs")
    except (AM95SpecCompatibilityError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"AM-95 HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "AM-95 semantic freeze PASS: "
        f"{value['freeze_id']} domain={value['sampling_contract']['domain_db']} "
        "distribution=discrete_uniform unit=per_sample_per_epoch "
        "er2_randomized_training=0 er9_training=0 test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
