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
    SUCCESSOR_SOURCE_MANIFEST,
    SUCCESSOR_RUNNER_CONTRACT,
    SUCCESSOR_STATE,
    SUPERSESSION_ARTIFACT,
    SuccessorContractError,
    canonical_json,
    load_json,
    sha256_bytes,
    validate_coordinator_contract,
    validate_successor_manifest,
    validate_successor_state,
    validate_runner_contract,
)
from baseline.g8_pascal_production import (  # noqa: E402
    audit_campaign,
    validate_production_contracts,
)
from baseline.g8_campaign import rendered_json  # noqa: E402


def verify(*, runtime_root: Path | None = None) -> dict[str, object]:
    manifest = validate_successor_manifest(load_json(SUCCESSOR_MANIFEST))
    state = validate_successor_state(load_json(SUCCESSOR_STATE))
    coordinator = validate_coordinator_contract(load_json(SUCCESSOR_COORDINATOR_CONTRACT))
    source_manifest = load_json(REPO / "results/baseline/g8_pascal_successor/source_manifest.json")
    runner_contract = validate_runner_contract(load_json(SUCCESSOR_RUNNER_CONTRACT))
    production_contracts = validate_production_contracts()
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
    if manifest["campaign_id"] != state["campaign_id"] or supersession["successor_campaign_id"] != manifest["campaign_id"]:
        raise SuccessorContractError("successor IDs are not bound")
    if hashlib.sha256(SUCCESSOR_SOURCE_MANIFEST.read_bytes()).hexdigest() != manifest["source_manifest_sha256"]:
        raise SuccessorContractError("successor source manifest hash is not bound by the campaign")
    source_required = {
        "schema_version", "artifact_role", "campaign_id", "execution_profile_id", "scientific_status",
        "lock_file", "lock_file_sha256", "sources", "coordinator_contract_sha256",
        "predecessor_manifest_sha256", "source_generation_commit", "old_result_ingest", "test_access",
        "validation_decoding", "inference", "training",
    }
    if set(source_manifest) != source_required or source_manifest["campaign_id"] != manifest["campaign_id"]:
        raise SuccessorContractError("successor source manifest schema/campaign binding differs")
    if source_manifest["execution_profile_id"] != SUCCESSOR_PROFILE_ID or source_manifest["scientific_status"] != "NON-SCIENTIFIC_ZERO_COVERAGE":
        raise SuccessorContractError("successor source manifest status/profile differs")
    if source_manifest["lock_file"] != "requirements-pascal.lock" or source_manifest["lock_file_sha256"] != hashlib.sha256((REPO / "requirements-pascal.lock").read_bytes()).hexdigest():
        raise SuccessorContractError("successor source manifest lock binding differs")
    if source_manifest["coordinator_contract_sha256"] != hashlib.sha256(SUCCESSOR_COORDINATOR_CONTRACT.read_bytes()).hexdigest() or source_manifest["predecessor_manifest_sha256"] != supersession["old_campaign_manifest_sha256"]:
        raise SuccessorContractError("successor source manifest artifact binding differs")
    for entry in source_manifest["sources"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "role", "bytes", "sha256"}:
            raise SuccessorContractError("successor source entry schema differs")
        path = REPO / entry["path"]
        if not path.is_file():
            raise SuccessorContractError(f"historical successor source path is missing: {entry.get('path')}")
    if any(source_manifest[key] != 0 for key in ("test_access", "validation_decoding", "inference", "training")) or source_manifest["old_result_ingest"] is not False:
        raise SuccessorContractError("successor source manifest claims protected activity")
    if runner_contract["campaign_id"] != manifest["campaign_id"] or runner_contract["manifest_sha256"] != hashlib.sha256(SUCCESSOR_MANIFEST.read_bytes()).hexdigest() or runner_contract["state_sha256"] != hashlib.sha256(SUCCESSOR_STATE.read_bytes()).hexdigest() or runner_contract["coordinator_sha256"] != hashlib.sha256(SUCCESSOR_COORDINATOR_CONTRACT.read_bytes()).hexdigest() or runner_contract["source_manifest_sha256"] != hashlib.sha256(SUCCESSOR_SOURCE_MANIFEST.read_bytes()).hexdigest():
        raise SuccessorContractError("successor runner artifact bindings differ")
    for entry in runner_contract["runner_source_paths"]:
        path = REPO / entry["path"]
        if not path.is_file():
            raise SuccessorContractError(f"historical successor runner source path is missing: {entry.get('path')}")
    parity = load_json(PARITY_PLAN)
    if parity.get("scientific_status") != "NON-SCIENTIFIC" or parity.get("paired_trial_count_per_cell") != 512:
        raise SuccessorContractError("parity plan is not preregistered non-scientific diagnostics")
    if parity.get("selected_identity_ordinals") != [
        441, 2990, 1765, 1383, 285, 395, 1376, 794, 539,
        1195, 186, 2694, 2274, 2402, 1915,
        2509, 2093, 2554,
    ]:
        raise SuccessorContractError("parity identity stratum changed after preregistration")
    bindings = parity.get("selected_identity_bindings")
    if not isinstance(bindings, list) or [item.get("ordinal") for item in bindings] != parity["selected_identity_ordinals"]:
        raise SuccessorContractError("parity identity bindings are missing or reordered")
    required_path = REPO / "results/baseline/g8/required_bler_identities.json"
    required_raw = required_path.read_bytes()
    required_payload = json.loads(required_raw)
    if required_raw != rendered_json(required_payload) or parity.get("required_identity_sha256") != hashlib.sha256(required_raw).hexdigest():
        raise SuccessorContractError("parity plan is not bound to the required identity bytes")
    units = required_payload.get("required_bler_work_units")
    if not isinstance(units, list) or len(units) != 3213:
        raise SuccessorContractError("required G8 identity artifact is not the authenticated 3213-cell grid")
    snr_values = sorted({float(unit["snr_db"]) for unit in units})
    for binding in bindings:
        ordinal = binding.get("ordinal")
        if not isinstance(ordinal, int) or not 0 <= ordinal < len(units):
            raise SuccessorContractError("parity identity ordinal is outside the required grid")
        snr = float(units[ordinal]["snr_db"])
        expected_stratum = "below_waterfall" if snr == snr_values[0] else "above_waterfall" if snr == snr_values[-1] else "near_waterfall"
        if binding != {
            "ordinal": ordinal,
            "work_unit_id": units[ordinal]["work_unit_id"],
            "identity": units[ordinal]["identity"],
            "snr_db": units[ordinal]["snr_db"],
            "stratum": expected_stratum,
        }:
            raise SuccessorContractError("parity identity binding differs from required grid")
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
    tracked_runtime = SUCCESSOR_ROOT / "runtime"
    outside_runtime_results = [
        path
        for path in SUCCESSOR_ROOT.rglob("*.result.json")
        if path.is_file() and tracked_runtime not in path.parents
    ]
    if outside_runtime_results:
        raise SuccessorContractError("successor tracked namespace contains result evidence outside the production runtime")
    selected_runtime = tracked_runtime if runtime_root is None else runtime_root
    try:
        runtime = audit_campaign(selected_runtime)
    except Exception as exc:
        if isinstance(exc, SuccessorContractError):
            raise
        raise SuccessorContractError(f"successor production runtime audit failed: {exc}") from exc
    return {
        "status": "PASS",
        "successor_campaign_id": manifest["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "accepted": runtime["accepted_count"],
        "required": manifest["required_identity_count"],
        "old_accepted": supersession["old_accepted_work_unit_count"],
        "old_tree_sha256": supersession["old_tree_aggregate_sha256"],
        "production_contract_sha256": production_contracts["production_contract_sha256"],
        "runtime_exists": runtime["runtime_exists"],
        "test_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="optional external successor runtime root")
    args = parser.parse_args()
    print(json.dumps(verify(runtime_root=args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuccessorContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
