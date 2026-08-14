#!/usr/bin/env python3
"""Verify a synthetic execution-profile performance artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.execution_profiles import canonical_json_bytes


class PerformanceVerificationError(RuntimeError):
    pass


def verify(report: Mapping[str, Any], *, expected_profile: str | None = None) -> None:
    if report.get("schema_version") != 1 or report.get("artifact_kind") != "execution_profile_synthetic_performance":
        raise PerformanceVerificationError("unsupported performance artifact")
    if report.get("scientific_status") != "NON-SCIENTIFIC" or report.get("synthetic_data_only") is not True or report.get("training_campaign") is not False:
        raise PerformanceVerificationError("performance artifact is not non-scientific")
    if expected_profile is not None and report.get("execution_profile_id") != expected_profile:
        raise PerformanceVerificationError("performance profile differs")
    for key in ("g8_coverage", "test_access", "validation_decoding", "inference"):
        if report.get(key) != 0:
            raise PerformanceVerificationError(f"performance {key} is nonzero")
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise PerformanceVerificationError("performance artifact digest differs")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise PerformanceVerificationError("performance measurements are missing")
    passed = [item for item in measurements if item.get("status") == "PASS"]
    if not passed or report.get("safe_batch_size") != max(item["batch_size"] for item in passed):
        raise PerformanceVerificationError("safe batch size is not measured")
    if any(item.get("peak_gpu_memory_bytes", 0) <= 0 or item.get("throughput_images_per_s", 0) <= 0 or item.get("all_finite") is not True for item in passed):
        raise PerformanceVerificationError("performance pass is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.report.read_bytes()), expected_profile=args.profile)
    print(json.dumps({"status": "PASS", "profile": args.profile}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PerformanceVerificationError as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
