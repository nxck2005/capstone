#!/usr/bin/env python3
"""Freeze the additive corrected-v3 E1 contract without opening E2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    v3._atomic_publish(path, v3.rendered_json(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if not args.check and args.source_commit != _head():
            raise v3.G8EV3Error("--source-commit must equal the current code-bearing HEAD")
        if any(path.exists() for path in (v3.V3_AUTHORIZATION_PATH, v3.V3_RUNTIME_ROOT)):
            raise v3.G8EV3Error("v3 authorization/runtime exists; pre-data freeze is forbidden")
        source = v3.build_source_manifest(args.source_commit)
        data = v3.build_scientific_data_identity(verify_archive_bytes=True)
        storage = (
            v3._rendered_object(v3.V3_STORAGE_PLAN_PATH, "v3 frozen storage plan")[0]
            if args.check
            else v3.build_storage_plan()
        )
        v3.storage_preflight(storage)
        correction = v3.build_correction_provenance()
        contract = v3.build_contract(source, data, storage)
        artifacts = (
            (v3.V3_SOURCE_MANIFEST_PATH, source),
            (v3.V3_DATA_IDENTITY_PATH, data),
            (v3.V3_STORAGE_PLAN_PATH, storage),
            (v3.V3_CORRECTION_PATH, correction),
            (v3.V3_CONTRACT_PATH, contract),
        )
        if args.check:
            for path, value in artifacts:
                observed, raw = v3._rendered_object(path, f"v3 frozen artifact {path.name}")
                if observed != value or raw != v3.rendered_json(value):
                    raise v3.G8EV3Error(f"v3 frozen artifact differs: {path}")
        else:
            for path, value in artifacts:
                _publish(path, value)
        print({
            "status": "PASS",
            "mode": "check" if args.check else "freeze",
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "source_manifest_id": source["source_manifest_id"],
            "data_identity_id": data["data_identity_id"],
            "validation_count": data["validation_count"],
            "production_e2_records": 0,
        })
        return 0
    except (OSError, subprocess.CalledProcessError, v3.G8EV3Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
