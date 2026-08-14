#!/usr/bin/env python3
"""Verify G8_C supersession and zero-coverage Pascal successor artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_successor import (  # noqa: E402
    PARITY_PLAN,
    SUCCESSOR_COORDINATOR_CONTRACT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    SUCCESSOR_STATE,
    SUPERSESSION_ARTIFACT,
    SuccessorContractError,
    canonical_json,
    load_json,
    sha256_bytes,
    validate_coordinator_contract,
    validate_successor_manifest,
    validate_successor_state,
)


def verify() -> dict[str, object]:
    manifest = validate_successor_manifest(load_json(SUCCESSOR_MANIFEST))
    state = validate_successor_state(load_json(SUCCESSOR_STATE))
    coordinator = validate_coordinator_contract(load_json(SUCCESSOR_COORDINATOR_CONTRACT))
    supersession = load_json(SUPERSESSION_ARTIFACT)
    required_supersession = {
        "schema_version", "artifact_role", "supersession_id", "old_campaign_id",
        "old_campaign_manifest_sha256", "old_campaign_state_sha256", "old_accepted_work_unit_count",
        "old_accepted_authority_prefix", "old_next_incomplete", "old_source_epoch_ids",
        "old_execution_profile_id", "reason", "scientific_validity", "continuation_status",
        "successor_bler_table_eligibility", "successor_campaign_id", "old_results_may_be_used_for",
        "old_tree_aggregate_sha256",
    }
    if set(supersession) != required_supersession:
        raise SuccessorContractError("supersession artifact schema differs")
    if supersession["schema_version"] != 1 or supersession["artifact_role"] != "g8_c_campaign_supersession":
        raise SuccessorContractError("unsupported supersession artifact")
    if supersession["old_accepted_work_unit_count"] != 748 or supersession["old_accepted_authority_prefix"] != [0, 747]:
        raise SuccessorContractError("old accepted boundary differs from authenticated live state")
    next_unit = supersession["old_next_incomplete"]
    if not isinstance(next_unit, dict) or next_unit["authority_ordinal"] != 748 or next_unit["request_only"] is not True or next_unit["result_published"] is not False:
        raise SuccessorContractError("old request-only trailing attempt was not preserved")
    if supersession["old_execution_profile_id"] != "local_4060_cu130" or supersession["continuation_status"] != "superseded":
        raise SuccessorContractError("old campaign profile/continuation status differs")
    if supersession["successor_bler_table_eligibility"] != "none":
        raise SuccessorContractError("old results can contribute to successor table")
    if manifest["campaign_id"] != state["campaign_id"] or supersession["successor_campaign_id"] not in {manifest["campaign_id"], "pending_manifest_digest"}:
        raise SuccessorContractError("successor IDs are not bound")
    parity = load_json(PARITY_PLAN)
    if parity.get("scientific_status") != "NON-SCIENTIFIC" or parity.get("paired_trial_count_per_cell") != 512:
        raise SuccessorContractError("parity plan is not preregistered non-scientific diagnostics")
    if parity.get("test_access") != 0 or parity.get("validation_decoding") != 0 or parity.get("training") != 0:
        raise SuccessorContractError("parity plan claims protected access")
    # This aggregate was authenticated during the live-state reconstruction
    # before the additive records existed.  The predecessor's own read-only
    # verifier remains the authority for the complete work-unit tree; this
    # check prevents silently replacing the recorded historical root.
    if supersession["old_tree_aggregate_sha256"] != "aee60ece7dc9d0ea7b6b0ed7769c2e35cf59bada99e56bf3d38f36447527b48c":
        raise SuccessorContractError("old G8 evidence aggregate is not the authenticated root")
    for relative, expected in {
        "results/baseline/g8/campaign_manifest.json": supersession["old_campaign_manifest_sha256"],
        "results/baseline/g8/campaign_state.json": supersession["old_campaign_state_sha256"],
    }.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise SuccessorContractError(f"old G8 evidence bytes changed: {relative}")
    successor_results = [path for path in SUCCESSOR_ROOT.rglob("*.result.json") if path.is_file()]
    if successor_results:
        raise SuccessorContractError("successor namespace contains result evidence before launch")
    return {
        "status": "PASS",
        "successor_campaign_id": manifest["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "accepted": 0,
        "required": manifest["required_identity_count"],
        "old_accepted": supersession["old_accepted_work_unit_count"],
        "old_tree_sha256": supersession["old_tree_aggregate_sha256"],
        "test_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(verify(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuccessorContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
