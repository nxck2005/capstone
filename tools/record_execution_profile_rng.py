#!/usr/bin/env python3
"""Record the cross-profile RNG hash comparison as qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402
from verify_execution_profile_rng import compare_reports, _read  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--pascal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local = _read(args.local)
    pascal = _read(args.pascal)
    comparison = compare_reports(local, pascal)
    if comparison["status"] != "PASS":
        raise SystemExit("RNG stimulus hashes differ")
    payload = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_rng_equivalence_summary",
        "scientific_status": "NON-SCIENTIFIC",
        "left_report_sha256": hashlib.sha256(args.local.read_bytes()).hexdigest(),
        "right_report_sha256": hashlib.sha256(args.pascal.read_bytes()).hexdigest(),
        "comparison": comparison,
        "interpretation": "exact generated information-bit and AWGN array bytes are equal on the preregistered cells; no decoder-invariance claim is inferred",
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }
    payload["summary_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    args.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps({"status": "PASS", "summary_sha256": payload["summary_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
