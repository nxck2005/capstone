#!/usr/bin/env python3
"""Fail-closed verifier for the qualification-only RNG stream audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402


class RngAuditVerificationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RngAuditVerificationError(f"cannot read RNG audit {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RngAuditVerificationError("RNG audit is not an object")
    return payload


def verify_report(report: Mapping[str, Any], *, expected_profile: str | None = None) -> None:
    if report.get("schema_version") != 1 or report.get("artifact_kind") != "execution_profile_rng_audit":
        raise RngAuditVerificationError("unsupported RNG audit schema")
    if report.get("scientific_status") != "NON-SCIENTIFIC":
        raise RngAuditVerificationError("RNG audit claims scientific status")
    if expected_profile is not None and report.get("execution_profile_id") != expected_profile:
        raise RngAuditVerificationError("RNG audit profile differs")
    for key in ("g8_coverage", "test_access", "validation_decoding", "training"):
        if report.get(key) != 0:
            raise RngAuditVerificationError(f"RNG audit {key} is nonzero")
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise RngAuditVerificationError("RNG audit digest differs")
    cells = report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RngAuditVerificationError("RNG audit has no selected cells")
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("streams"), Mapping):
            raise RngAuditVerificationError("RNG cell schema differs")
        streams = cell["streams"]
        if set(streams) != {"information_bits", "awgn_real", "awgn_imag"}:
            raise RngAuditVerificationError("RNG purpose set differs")
        for purpose, stream in streams.items():
            if not isinstance(stream, Mapping) or not isinstance(stream.get("sha256"), str) or len(stream["sha256"]) != 64:
                raise RngAuditVerificationError(f"RNG {purpose} hash is missing")


def compare_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    verify_report(left)
    verify_report(right)
    if left.get("campaign_id") != right.get("campaign_id") or left.get("required_identity_sha256") != right.get("required_identity_sha256") or left.get("parity_plan_sha256") != right.get("parity_plan_sha256") or left.get("trials_per_selected_identity") != right.get("trials_per_selected_identity"):
        raise RngAuditVerificationError("RNG reports do not bind the same frozen audit")
    left_cells = left["cells"]
    right_cells = right["cells"]
    if [cell["ordinal"] for cell in left_cells] != [cell["ordinal"] for cell in right_cells]:
        raise RngAuditVerificationError("RNG selected-cell order differs")
    mismatches: list[dict[str, Any]] = []
    for a, b in zip(left_cells, right_cells, strict=True):
        for purpose in ("information_bits", "awgn_real", "awgn_imag"):
            if a["streams"][purpose]["sha256"] != b["streams"][purpose]["sha256"]:
                mismatches.append({"ordinal": a["ordinal"], "purpose": purpose})
    return {
        "status": "PASS" if not mismatches else "MISMATCH",
        "left_profile": left["execution_profile_id"],
        "right_profile": right["execution_profile_id"],
        "selected_cells": len(left_cells),
        "mismatches": mismatches,
        "information_bits_equal": not any(item["purpose"] == "information_bits" for item in mismatches),
        "awgn_real_equal": not any(item["purpose"] == "awgn_real" for item in mismatches),
        "awgn_imag_equal": not any(item["purpose"] == "awgn_imag" for item in mismatches),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare_reports(_read(args.left), _read(args.right))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RngAuditVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
