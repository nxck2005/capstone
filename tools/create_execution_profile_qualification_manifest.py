#!/usr/bin/env python3
"""Create the content-addressed, zero-coverage qualification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for root in (REPO / "src", REPO / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from config.execution_profiles import canonical_json_bytes  # noqa: E402
from config.params import get  # noqa: E402
from verify_execution_profile_qualification import verify_files  # noqa: E402
from verify_execution_profile_performance import verify  # noqa: E402
from verify_openjpeg_profile_parity import read as read_openjpeg, verify as verify_openjpeg  # noqa: E402
from verify_execution_profile_parity import read_report, verify_report  # noqa: E402
from verify_execution_profile_rng import _read as read_rng, verify_report as verify_rng  # noqa: E402


def binding(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {"path": str(path.relative_to(REPO)), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def create(output: Path) -> dict[str, Any]:
    qualification_dir = REPO / "results/execution_profiles/qualification"
    pascal_reports = [qualification_dir / "confessor_pascal_cu126/gpu0.json", qualification_dir / "confessor_pascal_cu126/gpu1.json"]
    local_reports = [qualification_dir / "local_4060_cu130/gpu0.json"]
    verify_files(pascal_reports, expected_profile="confessor_pascal_cu126")
    verify_files(local_reports, expected_profile="local_4060_cu130")
    rng_summary = qualification_dir / "rng/summary.json"
    rng = read_rng(rng_summary)
    if rng.get("artifact_kind") != "execution_profile_rng_equivalence_summary" or rng.get("comparison", {}).get("status") != "PASS":
        raise RuntimeError("RNG summary is not a passing equivalence artifact")
    parity_summary = qualification_dir / "parity/summary.json"
    parity = json.loads(parity_summary.read_bytes())
    if parity.get("artifact_kind") != "execution_profile_three_device_parity_summary" or any(item.get("criterion_pass") is not True for item in parity.get("comparisons", {}).values()):
        raise RuntimeError("paired parity summary is not passing")
    openjpeg_comparison = qualification_dir / "openjpeg/comparison.json"
    openjpeg = read_openjpeg(openjpeg_comparison)
    if openjpeg.get("codestream_and_decoded_pixels_equal") is not True:
        raise RuntimeError("OpenJPEG comparison is not passing")
    performance_reports = [qualification_dir / "performance/confessor_pascal_cu126/cuda0_titan_xp.json", qualification_dir / "performance/confessor_pascal_cu126/cuda1_gtx_1080ti.json"]
    for report in performance_reports:
        verify(json.loads(report.read_bytes()), expected_profile="confessor_pascal_cu126")
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    payload = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_qualification_manifest",
        "scientific_status": "NON-SCIENTIFIC_ZERO_COVERAGE",
        "eligibility_status": "eligible_production_execution_profile",
        "execution_profile_id": "confessor_pascal_cu126",
        "lock_file": profile["lock_file"],
        "lock_file_sha256": profile["lock_file_sha256"],
        "gpu_uuids": list(profile["allowed_gpu_uuids"]),
        "qualification_reports": [binding(path) for path in local_reports + pascal_reports],
        "rng_equivalence_summary": binding(rng_summary),
        "paired_parity_summary": binding(parity_summary),
        "openjpeg_parity_summary": binding(openjpeg_comparison),
        "performance_reports": [binding(path) for path in performance_reports],
        "qualification_criteria": {
            "per_cell_disagreement_rate_max": 0.02,
            "aggregate_disagreement_rate_max": 0.01,
            "waterfall_displacement_db_max": 0.5,
            "bit_identical_gpu_decoding_required": False,
        },
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "selection": 0,
        "training_campaign": 0,
        "old_result_ingest": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = create(args.output)
    print(json.dumps({"status": "PASS", "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
