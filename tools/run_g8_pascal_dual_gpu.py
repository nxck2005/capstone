#!/usr/bin/env python3
"""Plan the explicit two-GPU Pascal G8_C successor.

The default and only permitted pre-merge mode is ``--plan-only``.  Execution
is deliberately gated until the merged main commit, final profile
qualification, successor marker, and authenticated writer checks are all
present; this tool cannot accidentally turn a structural plan into G8 data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_successor import (  # noqa: E402
    REQUIRED_COUNT,
    SUCCESSOR_COORDINATOR_CONTRACT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    SUCCESSOR_STATE,
    authority_shard,
    load_json,
    validate_coordinator_contract,
    validate_successor_manifest,
    validate_successor_state,
)
from config.execution_profiles import profile_definition  # noqa: E402


def _inventory() -> list[dict[str, str]]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,  # literal-ok: subprocess safety timeout
    ).stdout
    result = []
    for line in output.splitlines():
        index, uuid, name = [field.strip() for field in line.split(",", 2)]
        result.append({"gpu_index": int(index), "gpu_uuid": uuid, "gpu_name": name})
    return result


def build_plan() -> dict[str, object]:
    manifest = validate_successor_manifest(load_json(SUCCESSOR_MANIFEST))
    state = validate_successor_state(load_json(SUCCESSOR_STATE))
    contract = validate_coordinator_contract(load_json(SUCCESSOR_COORDINATOR_CONTRACT))
    profile = profile_definition(SUCCESSOR_PROFILE_ID)
    inventory = _inventory()
    by_index = {item["gpu_index"]: item for item in inventory}
    workers = []
    for worker in contract["workers"]:
        gpu = by_index.get(worker["gpu_index"])
        if gpu is None:
            raise RuntimeError(f"required GPU index is missing: {worker['gpu_index']}")
        if gpu["gpu_uuid"] != worker["gpu_uuid"]:
            raise RuntimeError(f"GPU UUID mismatch on {worker['device']}")
        if gpu["gpu_uuid"] not in profile["allowed_gpu_uuids"]:
            raise RuntimeError(f"GPU UUID is not allowed by {SUCCESSOR_PROFILE_ID}")
        workers.append({**dict(worker), **gpu, "assigned_authority_ordinals": [ordinal for ordinal in range(REQUIRED_COUNT) if authority_shard(ordinal) == worker["shard_index"]]})
    return {
        "status": "NON-SCIENTIFIC_PLAN_ONLY",
        "campaign_id": manifest["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "successor_root": str(SUCCESSOR_ROOT.relative_to(REPO)),
        "old_root": contract["old_root"],
        "required_identity_count": REQUIRED_COUNT,
        "accepted_count": state["completed_authority_ordinals"].__len__(),
        "workers": workers,
        "duplicate_work_units": 0,
        "missing_work_units": 0,
        "test_access": 0,
        "launch_gate": "MERGED_MAIN_FINAL_QUALIFICATION_WRITER_AUTH_AND_ZERO_COVERAGE_MARKER_REQUIRED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("G8_C PASCAL HOLD — scientific execution is gated until merged-main qualification and launch marker")
    if not args.plan_only:
        parser.error("only --plan-only is permitted before the explicit launch gate")
    print(json.dumps(build_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
