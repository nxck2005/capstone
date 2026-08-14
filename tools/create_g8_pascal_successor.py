#!/usr/bin/env python3
"""Create the zero-coverage G8_C Pascal successor and supersession records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Authenticated before this migration from the live G8 checkout.  It covers
# only the predecessor campaign's tracked request/result/state files; the
# additive supersession record is intentionally outside that historical tree.
OLD_TRACKED_TREE_AGGREGATE = "aee60ece7dc9d0ea7b6b0ed7769c2e35cf59bada99e56bf3d38f36447527b48c"

from baseline.g8_pascal_successor import (  # noqa: E402
    GPU_ASSIGNMENTS,
    PARITY_PLAN,
    REQUIRED_COUNT,
    SUCCESSOR_COORDINATOR_CONTRACT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    SUCCESSOR_SOURCE_MANIFEST,
    SUCCESSOR_STATE,
    SUPERSESSION_ARTIFACT,
    TRIALS_PER_IDENTITY,
    WORK_UNIT_PARTITION,
    canonical_json,
    rendered_json,
    sha256_bytes,
)


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = rendered_json(payload)
    path.write_bytes(body)
    return sha256_bytes(body)


def _old_tree_digest() -> str:
    paths = subprocess.run(
        ["git", "ls-files", "results/baseline/g8"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    entries = []
    additive = {
        "results/baseline/g8/g8_c_supersession.json",
        "results/baseline/g8/execution_profile_parity_plan.json",
    }
    for relative in paths:
        if relative in additive:
            continue
        body = (REPO / relative).read_bytes()
        entries.append({"path": relative, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    return sha256_bytes(canonical_json(entries))


def _old_state() -> dict:
    return json.loads((REPO / "results/baseline/g8/campaign_state.json").read_bytes())


def build() -> None:
    old_manifest = json.loads((REPO / "results/baseline/g8/campaign_manifest.json").read_bytes())
    old_state = _old_state()
    old_campaign_id = old_manifest["campaign_id"]
    old_manifest_sha = hashlib.sha256((REPO / "results/baseline/g8/campaign_manifest.json").read_bytes()).hexdigest()
    old_state_sha = hashlib.sha256((REPO / "results/baseline/g8/campaign_state.json").read_bytes()).hexdigest()
    old_source_epoch_ids = [
        {"path": "results/baseline/g8/bler_characterization_source_manifest.json", "sha256": hashlib.sha256((REPO / "results/baseline/g8/bler_characterization_source_manifest.json").read_bytes()).hexdigest()},
        {"path": "results/baseline/g8/bler_characterization_source_manifest_v2.json", "sha256": hashlib.sha256((REPO / "results/baseline/g8/bler_characterization_source_manifest_v2.json").read_bytes()).hexdigest()},
    ]
    supersession = {
        "schema_version": 1,
        "artifact_role": "g8_c_campaign_supersession",
        "supersession_id": "g8csup-" + "0" * 64,
        "old_campaign_id": old_campaign_id,
        "old_campaign_manifest_sha256": old_manifest_sha,
        "old_campaign_state_sha256": old_state_sha,
        "old_accepted_work_unit_count": 748,
        "old_accepted_authority_prefix": [0, 747],
        "old_next_incomplete": {
            "authority_ordinal": 748,
            "work_unit_id": "bler-3d67593f9deb3cfaab668644",
            "attempt": 1,
            "status": "claimed_request_published",
            "request_only": True,
            "result_published": False,
        },
        "old_source_epoch_ids": old_source_epoch_ids,
        "old_execution_profile_id": "local_4060_cu130",
        "reason": "owner-directed compute-host migration before G8_C BlerTable freeze",
        "scientific_validity": "valid_authenticated_immutable_historical_evidence",
        "continuation_status": "superseded",
        "successor_bler_table_eligibility": "none",
        "successor_campaign_id": "pending_manifest_digest",
        "old_results_may_be_used_for": ["provenance", "diagnostics", "cross_profile_equivalence_checks"],
        "old_tree_aggregate_sha256": OLD_TRACKED_TREE_AGGREGATE,
    }
    coordinator = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_dual_gpu_coordinator_contract",
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "worker_count": 2,
        "partition_rule": WORK_UNIT_PARTITION,
        "workers": [
            # CUDA's logical enumeration on confessor is the reverse of the
            # nvidia-smi index order; bind the UUID, not a guessed ordinal.
            {"shard_index": 0, "shard_count": 2, "device": "cuda:0", "gpu_index": 0, "gpu_uuid": "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a"},
            {"shard_index": 1, "shard_count": 2, "device": "cuda:1", "gpu_index": 1, "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"},
        ],
        "generic_cuda_device_permitted": False,
        "old_root": "results/baseline/g8/work_units",
        "successor_root": "results/baseline/g8_pascal_successor",
        "duplicate_assignment_policy": "fail_closed_if_ordinal_or_gpu_repeats",
        "recovery_policy": "authenticated_durable_state_only; request_result_state_atomic; no_old_result_ingest",
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
    }
    coordinator["contract_sha256"] = sha256_bytes(canonical_json(coordinator))
    coordinator_sha = _write(SUCCESSOR_COORDINATOR_CONTRACT, coordinator)

    source_manifest = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_source_manifest",
        "campaign_id": "pending_manifest_digest",
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "scientific_status": "NON-SCIENTIFIC_ZERO_COVERAGE",
        "sources": [
            {"path": "src/baseline/g8_pascal_successor.py", "role": "successor_contract"},
            {"path": "tools/run_g8_pascal_dual_gpu.py", "role": "coordinator"},
            {"path": "src/config/execution_profiles.py", "role": "profile_authentication"},
            {"path": "requirements-pascal.lock", "role": "environment_lock"},
        ],
        "old_result_ingest": False,
        "test_access": 0,
        "validation_decoding": 0,
        "inference": 0,
        "training": 0,
    }
    source_sha = _write(SUCCESSOR_SOURCE_MANIFEST, source_manifest)

    physical_contract = {
        "required_identities": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "grid_source": "results/baseline/g8/required_bler_identities.json exact physical identities only",
        "decoder_and_demapper": "predecessor_frozen_contract_unchanged",
        "count_and_interval_semantics": "predecessor_frozen_contract_unchanged",
        "seed_semantics": "campaign-bound successor namespace with same Philox/NumPy stream contract",
    }
    manifest = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_manifest",
        "campaign_id": "pending_manifest_digest",
        "status": "successor_open",
        "predecessor_campaign_id": old_campaign_id,
        "predecessor_manifest_sha256": old_manifest_sha,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "accepted_count": 0,
        "accepted_authority_ordinals": [],
        "successor_table_contribution": "successor_only_no_predecessor_results",
        "scientific_execution_performed": False,
        "test_access": 0,
        "physical_contract": physical_contract,
        "coordinator_contract_sha256": coordinator_sha,
        "source_manifest_sha256": source_sha,
    }
    manifest["campaign_id"] = "g8p-" + sha256_bytes(canonical_json({k: v for k, v in manifest.items() if k != "campaign_id"}))
    _write(SUCCESSOR_MANIFEST, manifest)

    # The supersession record is written only after the successor identity is
    # known.  A pending placeholder would leave the old/new campaign binding
    # weaker than the rest of the authenticated record.
    supersession["successor_campaign_id"] = manifest["campaign_id"]
    supersession["supersession_id"] = "g8csup-" + sha256_bytes(
        canonical_json({k: v for k, v in supersession.items() if k != "supersession_id"})
    )
    _write(SUPERSESSION_ARTIFACT, supersession)

    state = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_state",
        "campaign_id": manifest["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "completed_authority_ordinals": [],
        "in_progress_authority_ordinals": [],
        "failed_authority_ordinals": [],
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "test_access": 0,
        "scientific_execution_performed": False,
        "old_result_ingest_permitted": False,
    }
    state["state_sha256"] = sha256_bytes(canonical_json(state))
    _write(SUCCESSOR_STATE, state)

    selected_ordinals = [
        441, 2990, 1765, 1383, 285, 395, 1376, 794, 539,
        1195, 186, 2694, 2274, 2402, 1915,
        2509, 2093, 2554,
    ]
    required_units = json.loads((REPO / "results/baseline/g8/required_bler_identities.json").read_bytes())["required_bler_work_units"]
    snr_values = sorted({float(row["snr_db"]) for row in required_units})
    selected_bindings = []
    for ordinal in selected_ordinals:
        row = required_units[ordinal]
        snr = float(row["snr_db"])
        stratum = "below_waterfall" if snr == snr_values[0] else "above_waterfall" if snr == snr_values[-1] else "near_waterfall"
        selected_bindings.append({
            "ordinal": ordinal,
            "work_unit_id": row["work_unit_id"],
            "identity": row["identity"],
            "snr_db": row["snr_db"],
            "stratum": stratum,
        })
    parity = {
        "schema_version": 1,
        "artifact_role": "execution_profile_paired_parity_plan",
        "status": "preregistered_before_inspection",
        "scientific_status": "NON-SCIENTIFIC",
        "stimulus_rule": "exact old G8 contract seed material; no result/state publication",
        "rng_hashes": ["information_bits_bytes", "awgn_real_float64_bytes", "awgn_imag_float64_bytes"],
        "paired_trial_count_per_cell": 512,
        "disagreement_rule": "qualification-only hold if any cell >2 percent or aggregate >1 percent; no hypothesis test",
        "bler_ber_reporting": True,
        "waterfall_displacement_rule": "qualification-only hold if paired diagnostic displacement exceeds 0.5 dB",
        "gpu_pairs": ["local_4060_cu130↔confessor_pascal_cu126/cuda:0", "local_4060_cu130↔confessor_pascal_cu126/cuda:1", "cuda:0↔cuda:1"],
        "selected_identity_rule": "fixed stratified identities from the committed required-BLER grid; selected ordinals are recorded before parity inspection",
        "selected_identity_ordinals": selected_ordinals,
        "selected_identity_bindings": selected_bindings,
        "required_identity_sha256": hashlib.sha256(
            (REPO / "results/baseline/g8/required_bler_identities.json").read_bytes()
        ).hexdigest(),
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }
    parity["plan_sha256"] = sha256_bytes(canonical_json(parity))
    _write(PARITY_PLAN, parity)
    print(json.dumps({"campaign_id": manifest["campaign_id"], "state_sha256": state["state_sha256"], "supersession_id": supersession["supersession_id"], "source_manifest_sha256": source_sha, "parity_plan_sha256": parity["plan_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()
