from __future__ import annotations

import copy
import json

import pytest

from baseline.g8_pascal_successor import (
    REQUIRED_COUNT,
    TRIALS_PER_IDENTITY,
    SuccessorContractError,
    authority_shard,
    validate_coordinator_contract,
    validate_successor_manifest,
    validate_successor_state,
)


def _manifest():
    return {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_manifest",
        "campaign_id": "g8p-test",
        "status": "successor_open",
        "predecessor_campaign_id": "g8-old",
        "predecessor_manifest_sha256": "a" * 64,
        "execution_profile_id": "confessor_pascal_cu126",
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "accepted_count": 0,
        "accepted_authority_ordinals": [],
        "successor_table_contribution": "successor_only_no_predecessor_results",
        "scientific_execution_performed": False,
        "test_access": 0,
        "physical_contract": {"seed": "successor"},
        "coordinator_contract_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
    }


def _state():
    body = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_successor_state",
        "campaign_id": "g8p-test",
        "execution_profile_id": "confessor_pascal_cu126",
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
    from baseline.g8_pascal_successor import digest_without_field

    body["state_sha256"] = digest_without_field(body, "state_sha256")
    return body


def _coordinator():
    body = {
        "schema_version": 1,
        "artifact_role": "g8_c_pascal_dual_gpu_coordinator_contract",
        "execution_profile_id": "confessor_pascal_cu126",
        "worker_count": 2,
        "partition_rule": "authority_ordinal_modulo_2",
        "workers": [
            {"shard_index": 0, "shard_count": 2, "device": "cuda:0", "gpu_index": 0, "gpu_uuid": "GPU-a"},
            {"shard_index": 1, "shard_count": 2, "device": "cuda:1", "gpu_index": 1, "gpu_uuid": "GPU-b"},
        ],
        "generic_cuda_device_permitted": False,
        "old_root": "results/baseline/g8/work_units",
        "successor_root": "results/baseline/g8_pascal_successor",
        "duplicate_assignment_policy": "fail",
        "recovery_policy": "durable",
        "scientific_status": "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE",
    }
    from baseline.g8_pascal_successor import sha256_bytes, canonical_json

    body["contract_sha256"] = sha256_bytes(canonical_json(body))
    return body


def test_successor_starts_clean_and_two_shards_cover_exactly():
    validate_successor_manifest(_manifest())
    validate_successor_state(_state())
    validate_coordinator_contract(_coordinator())
    assert [authority_shard(i) for i in range(6)] == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("field", ["accepted_count", "execution_profile_id", "test_access", "successor_table_contribution"])
def test_successor_manifest_mutations_fail(field):
    body = _manifest()
    body[field] = 1 if field == "accepted_count" else ("local_4060_cu130" if field == "execution_profile_id" else (1 if field == "test_access" else "mixed"))
    with pytest.raises(SuccessorContractError):
        validate_successor_manifest(body)


def test_successor_state_old_result_ingest_mutation_fails():
    body = _state()
    body["old_result_ingest_permitted"] = True
    with pytest.raises(SuccessorContractError):
        validate_successor_state(body)


def test_coordinator_generic_cuda_and_duplicate_mutations_fail():
    body = _coordinator()
    body["generic_cuda_device_permitted"] = True
    from baseline.g8_pascal_successor import SuccessorContractError
    with pytest.raises(SuccessorContractError):
        validate_coordinator_contract(body)
