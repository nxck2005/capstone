#!/usr/bin/env python3
"""Fail-closed verifier for the qualification manifest and its bound bytes."""

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

from config.execution_profiles import canonical_json_bytes  # noqa: E402
from config.params import get  # noqa: E402


class QualificationManifestError(RuntimeError):
    pass


def verify(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or payload.get("artifact_kind") != "execution_profile_qualification_manifest":
        raise QualificationManifestError("unsupported qualification manifest")
    if payload.get("scientific_status") != "NON-SCIENTIFIC_ZERO_COVERAGE" or payload.get("eligibility_status") != "eligible_production_execution_profile":
        raise QualificationManifestError("qualification manifest status differs")
    if payload.get("execution_profile_id") != "confessor_pascal_cu126":
        raise QualificationManifestError("qualification manifest profile differs")
    if payload.get("lock_file_sha256") != get("environment.execution_profiles.confessor_pascal_cu126.lock_file_sha256"):
        raise QualificationManifestError("qualification manifest lock differs")
    for key in ("g8_coverage", "test_access", "validation_decoding", "selection", "training_campaign"):
        if payload.get(key) != 0:
            raise QualificationManifestError(f"qualification manifest {key} is nonzero")
    digest = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise QualificationManifestError("qualification manifest digest differs")
    bindings = []
    for key in ("qualification_reports", "performance_reports"):
        values = payload.get(key)
        if not isinstance(values, list):
            raise QualificationManifestError(f"qualification manifest {key} is missing")
        bindings.extend(values)
    for key in ("rng_equivalence_summary", "paired_parity_summary", "openjpeg_parity_summary"):
        bindings.append(payload.get(key))
    for item in bindings:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise QualificationManifestError("qualification artifact binding schema differs")
        artifact = REPO / str(item["path"])
        if not artifact.is_file() or len(artifact.read_bytes()) != item["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
            raise QualificationManifestError(f"qualification artifact bytes differ: {item.get('path')}")
    return dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps({"status": "PASS", "profile": result["execution_profile_id"], "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
