"""Clean G8_C Pascal successor contracts and dual-GPU topology.

This module intentionally has no measurement entry point.  It authenticates
the zero-coverage successor namespace, the owner-directed supersession record,
and the explicit two-GPU shard mapping used by the eventual worker command.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT


SUCCESSOR_ROOT = REPO_ROOT / "results/baseline/g8_pascal_successor"
SUCCESSOR_PROFILE_ID = "confessor_pascal_cu126"
SUCCESSOR_MANIFEST = SUCCESSOR_ROOT / "campaign_manifest.json"
SUCCESSOR_STATE = SUCCESSOR_ROOT / "campaign_state.json"
SUCCESSOR_COORDINATOR_CONTRACT = SUCCESSOR_ROOT / "dual_gpu_coordinator_contract.json"
SUCCESSOR_SOURCE_MANIFEST = SUCCESSOR_ROOT / "source_manifest.json"
SUPERSESSION_ARTIFACT = REPO_ROOT / "results/baseline/g8/g8_c_supersession.json"
PARITY_PLAN = REPO_ROOT / "results/baseline/g8/execution_profile_parity_plan.json"

REQUIRED_COUNT = 3213
TRIALS_PER_IDENTITY = 5000
WORK_UNIT_PARTITION = "authority_ordinal_modulo_2"
GPU_ASSIGNMENTS = (
    {"shard_index": 0, "shard_count": 2, "device": "cuda:0", "gpu_index": 0},
    {"shard_index": 1, "shard_count": 2, "device": "cuda:1", "gpu_index": 1},
)


class SuccessorContractError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def rendered_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_without_field(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha256_bytes(canonical_json(body))


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessorContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or raw != rendered_json(payload):
        raise SuccessorContractError(f"{path} is not canonical rendered JSON")
    return payload


def validate_successor_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "campaign_id", "status", "predecessor_campaign_id",
        "predecessor_manifest_sha256", "execution_profile_id", "required_identity_count",
        "trials_per_identity", "accepted_count", "accepted_authority_ordinals",
        "successor_table_contribution", "scientific_execution_performed", "test_access",
        "physical_contract", "coordinator_contract_sha256", "source_manifest_sha256",
    }
    if set(payload) != required:
        raise SuccessorContractError("successor manifest schema differs")
    if payload["schema_version"] != 1 or payload["artifact_role"] != "g8_c_pascal_successor_manifest":
        raise SuccessorContractError("unsupported successor manifest")
    if payload["status"] != "successor_open" or payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID:
        raise SuccessorContractError("successor status/profile is not frozen")
    if payload["required_identity_count"] != REQUIRED_COUNT or payload["trials_per_identity"] != TRIALS_PER_IDENTITY:
        raise SuccessorContractError("successor physical coverage is not 3213 x 5000")
    if payload["accepted_count"] != 0 or payload["accepted_authority_ordinals"] != []:
        raise SuccessorContractError("successor does not start at zero coverage")
    if payload["successor_table_contribution"] != "successor_only_no_predecessor_results":
        raise SuccessorContractError("successor table isolation policy differs")
    if payload["scientific_execution_performed"] is not False or payload["test_access"] != 0:
        raise SuccessorContractError("successor marker claims scientific execution")
    if not isinstance(payload["physical_contract"], Mapping):
        raise SuccessorContractError("successor physical contract is missing")
    return dict(payload)


def validate_successor_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "campaign_id", "execution_profile_id", "required_identity_count",
        "trials_per_identity", "completed_authority_ordinals", "in_progress_authority_ordinals",
        "failed_authority_ordinals", "protected_counters", "test_access", "scientific_execution_performed",
        "old_result_ingest_permitted", "state_sha256",
    }
    if set(payload) != required:
        raise SuccessorContractError("successor state schema differs")
    if payload["schema_version"] != 1 or payload["artifact_role"] != "g8_c_pascal_successor_state":
        raise SuccessorContractError("unsupported successor state")
    if payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID:
        raise SuccessorContractError("successor state profile differs")
    if payload["required_identity_count"] != REQUIRED_COUNT or payload["trials_per_identity"] != TRIALS_PER_IDENTITY:
        raise SuccessorContractError("successor state physical coverage differs")
    for key in ("completed_authority_ordinals", "in_progress_authority_ordinals", "failed_authority_ordinals"):
        values = payload[key]
        if not isinstance(values, list) or any(not isinstance(item, int) or item < 0 or item >= REQUIRED_COUNT for item in values):
            raise SuccessorContractError(f"invalid successor {key}")
    if payload["completed_authority_ordinals"] or payload["in_progress_authority_ordinals"] or payload["failed_authority_ordinals"]:
        raise SuccessorContractError("successor state is not zero coverage")
    if payload["protected_counters"] != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise SuccessorContractError("successor protected counters are nonzero")
    if payload["test_access"] != 0 or payload["scientific_execution_performed"] is not False:
        raise SuccessorContractError("successor state claims scientific access")
    if payload["old_result_ingest_permitted"] is not False:
        raise SuccessorContractError("old result ingestion is permitted")
    if payload["state_sha256"] != digest_without_field(payload, "state_sha256"):
        raise SuccessorContractError("successor state digest differs")
    return dict(payload)


def validate_coordinator_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "execution_profile_id", "worker_count", "partition_rule",
        "workers", "generic_cuda_device_permitted", "old_root", "successor_root", "duplicate_assignment_policy",
        "recovery_policy", "scientific_status", "contract_sha256",
    }
    if set(payload) != required:
        raise SuccessorContractError("coordinator contract schema differs")
    if payload["schema_version"] != 1 or payload["artifact_role"] != "g8_c_pascal_dual_gpu_coordinator_contract":
        raise SuccessorContractError("unsupported coordinator contract")
    if payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID or payload["worker_count"] != 2:
        raise SuccessorContractError("coordinator profile/worker count differs")
    if payload["partition_rule"] != WORK_UNIT_PARTITION or payload["generic_cuda_device_permitted"] is not False:
        raise SuccessorContractError("coordinator does not bind explicit modulo shards/devices")
    workers = payload["workers"]
    if not isinstance(workers, list) or len(workers) != 2:
        raise SuccessorContractError("coordinator worker list differs")
    seen_devices: set[str] = set()
    seen_shards: set[int] = set()
    for worker in workers:
        if not isinstance(worker, Mapping) or set(worker) != {"shard_index", "shard_count", "device", "gpu_index", "gpu_uuid"}:
            raise SuccessorContractError("coordinator worker schema differs")
        if worker["shard_count"] != 2 or worker["device"] not in {"cuda:0", "cuda:1"}:
            raise SuccessorContractError("coordinator worker device is not explicit")
        if worker["device"] in seen_devices or worker["shard_index"] in seen_shards:
            raise SuccessorContractError("duplicate coordinator worker assignment")
        seen_devices.add(worker["device"])
        seen_shards.add(worker["shard_index"])
    if seen_devices != {"cuda:0", "cuda:1"} or seen_shards != {0, 1}:
        raise SuccessorContractError("coordinator does not cover both shards")
    if payload["old_root"] == payload["successor_root"] or payload["successor_root"] != str(SUCCESSOR_ROOT.relative_to(REPO_ROOT)):
        raise SuccessorContractError("coordinator root isolation is missing")
    if payload["scientific_status"] != "NON-SCIENTIFIC_UNTIL_EXPLICIT_LAUNCH_GATE":
        raise SuccessorContractError("coordinator launch status differs")
    if payload["contract_sha256"] != digest_without_field(payload, "contract_sha256"):
        raise SuccessorContractError("coordinator contract digest differs")
    return dict(payload)


def authority_shard(authority_ordinal: int, *, shard_count: int = 2) -> int:
    if not isinstance(authority_ordinal, int) or authority_ordinal < 0:
        raise ValueError("authority ordinal must be non-negative")
    if shard_count != 2:
        raise ValueError("successor coordinator is frozen to two shards")
    return authority_ordinal % shard_count
