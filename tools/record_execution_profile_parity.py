#!/usr/bin/env python3
"""Build the immutable three-device qualification parity summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402
from verify_execution_profile_parity import compare, read_report  # noqa: E402


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(local: Path, titan: Path, gtx: Path, output: Path) -> dict[str, Any]:
    reports = {path: read_report(path) for path in (local, titan, gtx)}
    pairs = [
        ("local_4060_cu130_vs_titan_xp", local, titan),
        ("local_4060_cu130_vs_gtx_1080ti", local, gtx),
        ("titan_xp_vs_gtx_1080ti", titan, gtx),
    ]
    comparisons = {}
    for name, left, right in pairs:
        result = compare(reports[left], reports[right])
        if result["criterion_pass"] is not True:
            raise RuntimeError(f"paired criterion failed: {name}")
        result["comparison_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
        comparisons[name] = result
    output_payload = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_three_device_parity_summary",
        "scientific_status": "NON-SCIENTIFIC",
        "qualified_profile_decision": "confessor_pascal_cu126_unified_for_both_registered_pascal_gpus",
        "source_reports": {
            "local_4060_cu130": {"path": str(local), "sha256": _file_sha(local)},
            "confessor_pascal_cu126_cuda0_titan_xp": {"path": str(titan), "sha256": _file_sha(titan)},
            "confessor_pascal_cu126_cuda1_gtx_1080ti": {"path": str(gtx), "sha256": _file_sha(gtx)},
        },
        "comparisons": comparisons,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }
    output_payload["summary_sha256"] = hashlib.sha256(canonical_json_bytes(output_payload)).hexdigest()
    output.write_bytes(canonical_json_bytes(output_payload))
    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--titan", type=Path, required=True)
    parser.add_argument("--gtx1080ti", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = record(args.local, args.titan, args.gtx1080ti, args.output)
    print(json.dumps({"status": "PASS", "summary_sha256": result["summary_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
