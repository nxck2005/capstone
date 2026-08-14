#!/usr/bin/env python3
"""Fail-closed verifier for non-scientific execution-profile qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes, profile_definition  # noqa: E402
from config.params import get  # noqa: E402


class QualificationVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationVerificationError(message)


def verify_report(report: Mapping[str, object], *, expected_profile: str) -> dict[str, object]:
    _require(report.get("schema_version") == 1, "unsupported qualification schema")
    _require(report.get("artifact_kind") == "execution_profile_qualification", "wrong artifact kind")
    _require(report.get("scientific_status") == "NON-SCIENTIFIC", "qualification claims scientific status")
    for counter in (
        "g8_coverage",
        "test_access_count",
        "validation_count",
        "selection_count",
        "training_campaign_count",
    ):
        _require(report.get(counter) == 0, f"qualification {counter} is nonzero")
    _require(report.get("synthetic_data_only") is True, "qualification was not synthetic-only")
    _require(report.get("execution_profile_id") == expected_profile, "qualification profile differs")
    recorded_hash = report.get("report_sha256")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    _require(
        recorded_hash == hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest(),
        "qualification report hash differs",
    )

    environment = report.get("environment")
    _require(isinstance(environment, Mapping), "qualification environment is missing")
    fields = set(get("environment.execution_profile_record_fields"))
    _require(fields <= set(environment), "qualification environment record is incomplete")
    _require(environment.get("execution_profile_id") == expected_profile, "environment profile differs")
    profile = profile_definition(expected_profile)
    _require(environment.get("lock_file") == profile["lock_file"], "environment lock path differs")
    _require(environment.get("lock_file_sha256") == profile["lock_file_sha256"], "environment lock hash differs")
    _require(environment.get("gpu_uuid") in profile["allowed_gpu_uuids"], "environment GPU UUID differs")

    checks = report.get("checks")
    _require(isinstance(checks, Mapping), "qualification checks are missing")
    _require(checks.get("real_cuda_tensor_operations") is True, "CUDA tensor operation failed")
    ldpc = checks.get("ldpc")
    _require(isinstance(ldpc, Mapping) and ldpc.get("encode_pass") is True, "LDPC encode failed")
    _require(ldpc.get("decode_pass") is True, "LDPC decode failed")
    modulation = checks.get("project_modulation_demodulation")
    _require(
        isinstance(modulation, Mapping)
        and set(modulation) == {"bpsk", "qpsk", "qam16"}
        and all(value is True for value in modulation.values()),
        "project modulation/demodulation failed",
    )
    djscc = checks.get("djscc")
    _require(isinstance(djscc, Mapping), "DJSCC checks are missing")
    for key in (
        "forward_pass",
        "backward_pass",
        "finite_loss",
        "finite_gradients",
        "optimizer_step",
        "checkpoint_save",
        "checkpoint_reload",
        "same_seed_repeatability",
    ):
        _require(djscc.get(key) is True, f"DJSCC {key} failed")
    _require(int(djscc.get("safe_batch_size", 0)) > 0, "safe batch size is absent")
    _require(int(djscc.get("peak_gpu_memory_bytes", 0)) > 0, "peak GPU memory is absent")
    _require(float(djscc.get("throughput_images_per_s", 0)) > 0, "throughput is absent")
    crash = checks.get("crash_kill_restart")
    _require(isinstance(crash, Mapping) and all(value is True for value in crash.values()), "crash/restart probe failed")
    return dict(environment)


def verify_files(paths: list[Path], *, expected_profile: str) -> list[dict[str, object]]:
    _require(paths, "no qualification reports supplied")
    environments = [verify_report(json.loads(path.read_bytes()), expected_profile=expected_profile) for path in paths]
    indices = [int(item["gpu_index"]) for item in environments]
    uuids = [str(item["gpu_uuid"]) for item in environments]
    _require(len(indices) == len(set(indices)), "duplicate GPU index across qualification reports")
    _require(len(uuids) == len(set(uuids)), "duplicate GPU UUID across qualification reports")
    if expected_profile == "confessor_pascal_cu126":
        _require(set(indices) == {0, 1}, "Pascal qualification must cover cuda:0 and cuda:1")
        _require(set(uuids) == set(profile_definition(expected_profile)["allowed_gpu_uuids"]), "Pascal qualification does not cover both registered GPUs")
    return environments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    environments = verify_files(args.reports, expected_profile=args.profile)
    print(json.dumps({"status": "PASS", "profile": args.profile, "gpu_uuids": [item["gpu_uuid"] for item in environments]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
