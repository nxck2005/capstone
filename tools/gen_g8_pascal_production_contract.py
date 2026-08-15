#!/usr/bin/env python3
"""Generate the additive production contracts for the Pascal G8_C successor.

The schema-1 successor files are the historical zero-coverage readiness
marker.  These version-1 production files are a separate, immutable closure:
they bind the child-worker coordinator, the successor transaction module and
the exact frozen PHY import closure used after the owner opens science.
"""

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
    REQUIRED_COUNT,
    SUCCESSOR_COORDINATOR_CONTRACT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_RUNNER_CONTRACT,
    SUCCESSOR_ROOT,
    SUCCESSOR_STATE,
    canonical_json,
    load_json,
    rendered_json,
    sha256_bytes,
)
from baseline.g8_pascal_production import (  # noqa: E402
    FAILED_WORK_UNIT_POLICY,
    MAX_UNITS_POLICY,
    PASCAL_SUCCESSOR_CUSTODY_POLICY,
)
from config.params import get  # noqa: E402


PRODUCTION_CONTRACT = SUCCESSOR_ROOT / "production_contract.json"
PRODUCTION_SOURCE_MANIFEST = SUCCESSOR_ROOT / "production_source_manifest.json"
PRODUCTION_RUNNER_CONTRACT = SUCCESSOR_ROOT / "production_runner_contract.json"
PRODUCTION_COORDINATOR_CONTRACT = SUCCESSOR_ROOT / "production_coordinator_contract.json"
LOCK_PATH = REPO / "requirements-pascal.lock"
REQUIRED_IDENTITIES = REPO / "results/baseline/g8/required_bler_identities.json"

WORKERS = [
    {
        "shard_index": 0,
        "shard_count": 2,
        "device": "cuda:0",
        "gpu_name": "NVIDIA TITAN Xp",
        "gpu_uuid": "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
        "gpu_compute_capability": "6.1",
    },
    {
        "shard_index": 1,
        "shard_count": 2,
        "device": "cuda:1",
        "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
        "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
        "gpu_compute_capability": "6.1",
    },
]

# This is the import/runtime closure traced from the successor child worker:
# coordinator -> successor transaction -> frozen G8 runner/contract -> LDPC
# adapter and modulation.  The source manifest is intentionally explicit;
# adding a result-affecting import requires regenerating this closure.
SOURCE_CLOSURE = [
    ("src/baseline/g8_pascal_successor.py", "successor_contract"),
    ("src/baseline/g8_pascal_production.py", "successor_transaction_state_worker"),
    ("tools/run_g8_pascal_dual_gpu.py", "dual_gpu_child_process_coordinator"),
    ("src/baseline/g8_bler_runner.py", "frozen_measurement_runner"),
    ("src/baseline/g8_bler_resume.py", "frozen_runner_import_resume_contract"),
    ("src/baseline/g8_bler_work_units.py", "frozen_runner_import_transaction_contract"),
    ("src/baseline/g8_bler_contract.py", "frozen_seed_count_interval_contract"),
    ("src/baseline/g8_campaign.py", "frozen_campaign_identity_contract"),
    ("src/baseline/classical/composition.py", "bler_identity_type"),
    ("src/baseline/classical/outage.py", "composition_import_closure"),
    ("src/artifacts/rng.py", "composition_import_rng_contract"),
    ("src/data/manifests.py", "composition_import_manifest_contract"),
    ("src/data/adapters.py", "manifest_import_dataset_adapter_contract"),
    ("src/data/identity.py", "manifest_import_identity_contract"),
    ("src/data/provenance.py", "manifest_import_provenance_contract"),
    ("src/baseline/ldpc/__init__.py", "ldpc_package_initialization"),
    ("src/baseline/ldpc/adapter.py", "sionna_ldpc_adapter"),
    ("src/baseline/ldpc/modulation.py", "modulation_and_max_log_llr"),
    ("src/baseline/ldpc/transport.py", "campaign_import_closure_packet_transport"),
    ("src/baseline/ldpc/rate_matching.py", "transport_import_closure_rate_matching"),
    ("src/baseline/ldpc/segmentation.py", "transport_import_closure_segmentation"),
    ("src/baseline/ldpc/crc.py", "transport_import_closure_crc"),
    ("src/config/execution_profiles.py", "execution_profile_authentication"),
    ("src/config/params.py", "generated_parameter_loader"),
    ("src/config/run_config.py", "profile_authentication_import_closure"),
    ("src/env.py", "deterministic_backend_and_openjpeg_authentication"),
    ("spec/params.generated.yaml", "generated_parameters_consumed_at_runtime"),
    ("requirements-pascal.lock", "authenticated_python_environment"),
    ("results/baseline/g8/required_bler_identities.json", "exact_required_physical_grid"),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_entries() -> list[dict[str, object]]:
    entries = []
    for relative, role in SOURCE_CLOSURE:
        path = REPO / relative
        body = path.read_bytes()
        entries.append({"path": relative, "role": role, "bytes": len(body), "sha256": sha256_bytes(body)})
    return entries


def _write(path: Path, payload: dict[str, object], *, check: bool) -> str:
    body = rendered_json(payload)
    if check:
        if path.read_bytes() != body:
            raise RuntimeError(f"production contract is stale: {path}")
    else:
        path.write_bytes(body)
    return sha256_bytes(body)


def build(*, check: bool = False) -> dict[str, str]:
    manifest = load_json(SUCCESSOR_MANIFEST)
    state = load_json(SUCCESSOR_STATE)
    old_manifest_sha = _sha(REPO / "results/baseline/g8/campaign_manifest.json")
    old_state_sha = _sha(SUCCESSOR_STATE)
    required_sha = _sha(REQUIRED_IDENTITIES)
    lock_sha = _sha(LOCK_PATH)
    campaign_id = manifest["campaign_id"]
    coordinator_sha = _sha(SUCCESSOR_COORDINATOR_CONTRACT)

    coordinator = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_production_coordinator_contract",
        "campaign_id": campaign_id,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": _sha(SUCCESSOR_MANIFEST),
        "worker_count": 2,
        "partition_rule": "authority_ordinal % 2 == shard_index",
        "workers": WORKERS,
        "generic_cuda_device_permitted": False,
        "child_process_model": "two_independent_child_processes_one_per_explicit_cuda_device",
        "cross_machine_ddp": False,
        "old_root": "results/baseline/g8/work_units",
        "successor_runtime_root": "results/baseline/g8_pascal_successor/runtime",
        "successor_work_unit_root": "results/baseline/g8_pascal_successor/runtime",
        "duplicate_assignment_policy": "fail_closed_on_overlap_or_omission",
        "failure_isolation": "published_evidence_is_immutable_and_other_worker_continues",
        "max_units_policy": MAX_UNITS_POLICY,
        "failed_work_unit_policy": FAILED_WORK_UNIT_POLICY,
        "evidence_custody_policy": PASCAL_SUCCESSOR_CUSTODY_POLICY,
        "recovery_policy": "reconcile_durable_requests_results_states_after_restart",
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "old_result_ingest": False,
    }
    coordinator["contract_sha256"] = sha256_bytes(canonical_json(coordinator))
    production_coordinator_sha = _write(PRODUCTION_COORDINATOR_CONTRACT, coordinator, check=check)

    source_manifest = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_production_source_manifest",
        "campaign_id": campaign_id,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": _sha(SUCCESSOR_MANIFEST),
        "predecessor_manifest_sha256": old_manifest_sha,
        "readiness_state_sha256": old_state_sha,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": lock_sha,
        "required_bler_artifact": "results/baseline/g8/required_bler_identities.json",
        "required_bler_artifact_sha256": required_sha,
        "coordinator_contract_sha256": production_coordinator_sha,
        "sources": _source_entries(),
        "old_result_ingest": False,
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
    }
    production_source_sha = _write(PRODUCTION_SOURCE_MANIFEST, source_manifest, check=check)

    runner = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_production_runner_contract",
        "campaign_id": campaign_id,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": _sha(SUCCESSOR_MANIFEST),
        "readiness_state_sha256": old_state_sha,
        "production_source_manifest_sha256": production_source_sha,
        "coordinator_contract_sha256": production_coordinator_sha,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": lock_sha,
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": int(get("baseline.bler_characterisation_trials")),
        "workers": WORKERS,
        "entrypoint": "baseline.g8_pascal_production.run_unit",
        "measurement_entrypoint": "baseline.g8_bler_runner._execute_measurement",
        "request_schema": "g8_c_pascal_successor_work_unit_request:v1",
        "result_schema": "g8_c_pascal_successor_work_unit_result:v1",
        "state_schema": "g8_c_pascal_successor_production_state:v1",
        "max_units_policy": MAX_UNITS_POLICY,
        "failed_work_unit_policy": FAILED_WORK_UNIT_POLICY,
        "runner_source_paths": [entry["path"] for entry in _source_entries()],
        "driver_version_required": True,
        "old_result_ingest": False,
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
    }
    runner["contract_sha256"] = sha256_bytes(canonical_json(runner))
    runner_sha = _write(PRODUCTION_RUNNER_CONTRACT, runner, check=check)

    production_contract = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_production_contract",
        "campaign_id": campaign_id,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": _sha(SUCCESSOR_MANIFEST),
        "readiness_state_sha256": old_state_sha,
        "production_source_manifest_sha256": production_source_sha,
        "source_manifest_sha256": production_source_sha,
        "runner_contract_sha256": runner_sha,
        "coordinator_contract_sha256": production_coordinator_sha,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": lock_sha,
        "required_bler_artifact_sha256": required_sha,
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": int(get("baseline.bler_characterisation_trials")),
        "execution_profile_provenance_fields": [
            "execution_profile_id", "lock_file", "lock_file_sha256", "python_version",
            "torch_version", "torch_cuda_build", "torchvision_version", "numpy_version",
            "sionna_version", "openjpeg_version", "deterministic_backend", "amp", "gpu_name",
            "gpu_uuid", "gpu_vram_mib", "gpu_compute_capability", "gpu_index", "nvidia_smi_index",
            "driver_version", "device", "git_commit", "git_dirty", "config_hash",
        ],
        "physical_contract": {
            "identity_source": "required_bler_identities.json exact array authority",
            "seed_contract": "g8_bler_contract.stream_seed_records with successor campaign ID",
            "trial_count_source": "params.baseline.bler_characterisation_trials",
            "measurement": "frozen g8_bler_runner._execute_measurement",
            "decoder_and_demapper": "frozen LDPC adapter and max-log-LLR path",
            "count_semantics": "frozen g8_bler_contract.recompute_measurements",
            "resume_granularity": "one complete work unit; restart incomplete unit from trial zero",
        },
        "worker_batch_policy": {
            "max_units": MAX_UNITS_POLICY,
            "failed_work_unit": FAILED_WORK_UNIT_POLICY,
        },
        "evidence_custody_policy": PASCAL_SUCCESSOR_CUSTODY_POLICY,
        "source_closure_role": "all result-affecting local imports and authenticated environment bytes are listed in production_source_manifest.json",
        "driver_version_required": True,
        "old_result_ingest": False,
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
    }
    production_contract["contract_sha256"] = sha256_bytes(canonical_json(production_contract))
    production_sha = _write(PRODUCTION_CONTRACT, production_contract, check=check)
    return {
        "production_contract_sha256": production_sha,
        "production_source_manifest_sha256": production_source_sha,
        "production_runner_contract_sha256": runner_sha,
        "production_coordinator_contract_sha256": production_coordinator_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(check=args.check), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
