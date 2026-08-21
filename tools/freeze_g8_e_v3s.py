#!/usr/bin/env python3
"""Freeze the corrected-v3 confessor worker-successor epoch without opening E2.

Run ON the worker host (confessor) so the storage plan and relocation
provenance describe the worker filesystem and the frozen worker device.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3s as v3s  # noqa: E402


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", required=True, help="frozen worker CUDA device, e.g. cuda:0")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if not args.check and args.source_commit != _head():
            raise v3s.G8EV3SError("--source-commit must equal the current code-bearing HEAD")
        if any(path.exists() for path in (v3s.V3S_AUTHORIZATION_PATH, v3s.V3S_RUNTIME_ROOT)):
            raise v3s.G8EV3SError("v3s authorization/runtime exists; pre-data freeze is forbidden")
        if not v3s.V3S_RELOCATION_PROVENANCE_PATH.is_file():
            raise v3s.G8EV3SError(
                "v3s relocation provenance must be frozen on the predecessor host and committed first"
            )
        stored_relocation, _ = v3s._rendered_object(
            v3s.V3S_RELOCATION_PROVENANCE_PATH, "v3s relocation provenance"
        )
        if stored_relocation.get("worker", {}).get("device") != args.device:
            raise v3s.G8EV3SError("--device differs from the frozen relocation provenance worker binding")
        if args.check:
            if v3s.verify_frozen_contract(verify_live_sources=True, verify_live_data=False) is None:
                raise v3s.G8EV3SError("unreachable")
            print({"status": "PASS", "mode": "check", "production_e2_records": 0})
            return 0
        source = v3s.build_source_manifest(args.source_commit)
        storage = v3s.build_storage_plan()
        v3s.storage_preflight(storage, v3s.V3S_RUNTIME_ROOT)
        contract = v3s.build_contract(source, storage)
        for path, value in (
            (v3s.V3S_SOURCE_MANIFEST_PATH, source),
            (v3s.V3S_STORAGE_PLAN_PATH, storage),
            (v3s.V3S_CONTRACT_PATH, contract),
        ):
            v3s._atomic_publish(path, v3s.rendered_json(value))
        print({
            "status": "PASS",
            "mode": "freeze",
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "profile_id": contract["execution_profile"]["profile_id"],
            "device": contract["execution_profile"]["device"],
            "source_manifest_id": source["source_manifest_id"],
            "production_e2_records": 0,
        })
        return 0
    except (OSError, subprocess.CalledProcessError, v3s.G8EV3SError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
