"""Production transaction primitives for the G8_C Pascal successor.

The tracked ``campaign_state.json`` in the successor directory is a readiness
marker and intentionally remains a zero-coverage artifact.  This module owns
the *mutable* production runtime state under an explicit, separate successor
runtime root.  It reuses the frozen G8 request identity, seed, count and PHY
measurement implementation, but gives those records a successor-specific
campaign binding and artifact role so predecessor evidence cannot be admitted.

The state protocol is deliberately small and reachable:

    absent/available -> claimed -> request_published -> result_published
    -> accepted

``failed`` is retryable only through the next clean attempt.  ``terminal_invalid``
is non-mergeable and terminal.  Requests and results are immutable per-attempt
history.  State publication uses a per-unit exclusive flock, staged bytes,
file fsync, and a directory fsync; first publication is no-replace and
replacement is compare-and-swap while holding that lock.

No selection, validation decoding, classifier, training or test access belongs
to this module.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from baseline import g8_bler_contract as frozen_bler
from baseline import g8_bler_runner as frozen_runner
from baseline.g8_pascal_successor import (
    REQUIRED_COUNT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    SUCCESSOR_STATE,
    SUCCESSOR_SOURCE_MANIFEST,
    TRIALS_PER_IDENTITY,
    canonical_json,
    digest_without_field,
    load_json,
    sha256_bytes,
    successor_campaign_identifier,
    authority_shard,
    validate_successor_manifest,
    validate_successor_state,
)
from config.execution_profiles import authenticate_execution_profile
from config.params import REPO_ROOT


PRODUCTION_SCHEMA_VERSION = 1
PRODUCTION_CONTRACT_ARTIFACT_ROLE = "g8_c_pascal_successor_production_contract"
PRODUCTION_CONTRACT = SUCCESSOR_ROOT / "production_contract.json"
PRODUCTION_SOURCE_MANIFEST = SUCCESSOR_ROOT / "production_source_manifest.json"
PRODUCTION_RUNNER_CONTRACT = SUCCESSOR_ROOT / "production_runner_contract.json"
PRODUCTION_COORDINATOR_CONTRACT = SUCCESSOR_ROOT / "production_coordinator_contract.json"
PRE_MEASUREMENT_REPAIR_POLICY = SUCCESSOR_ROOT / "pre_measurement_repair.json"
POST_CAMPAIGN_SOURCE_COMPATIBILITY = REPO_ROOT / "results/baseline/g8_f/am87_post_campaign_source_compatibility.json"
AM88_POST_CAMPAIGN_SOURCE_COMPATIBILITY = REPO_ROOT / "results/baseline/g8_f/am88_post_campaign_source_compatibility.json"
AM87_FINAL_COMMIT = "6ea39f6e5e7744175ed1b367a6368b44ad3909a6"
PRODUCTION_STATE_ARTIFACT_ROLE = "g8_c_pascal_successor_production_state"
PRODUCTION_STATE_FILENAME = "campaign_state.json"
OLD_WORK_UNIT_ROOT = REPO_ROOT / "results/baseline/g8/work_units"
SUCCESSOR_LOGICAL_RUNTIME_ROOT = "results/baseline/g8_pascal_successor/runtime"

PRODUCTION_WORKERS = (
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
)

REQUIRED_PRODUCTION_SOURCE_PATHS = (
    "src/baseline/g8_pascal_successor.py",
    "src/baseline/g8_pascal_production.py",
    "tools/run_g8_pascal_dual_gpu.py",
    "src/baseline/g8_bler_runner.py",
    "src/baseline/g8_bler_resume.py",
    "src/baseline/g8_bler_work_units.py",
    "src/baseline/g8_bler_contract.py",
    "src/baseline/g8_campaign.py",
    "src/baseline/classical/composition.py",
    "src/baseline/classical/outage.py",
    "src/artifacts/rng.py",
    "src/data/manifests.py",
    "src/data/adapters.py",
    "src/data/identity.py",
    "src/data/provenance.py",
    "src/baseline/ldpc/__init__.py",
    "src/baseline/ldpc/adapter.py",
    "src/baseline/ldpc/modulation.py",
    "src/baseline/ldpc/transport.py",
    "src/baseline/ldpc/rate_matching.py",
    "src/baseline/ldpc/segmentation.py",
    "src/baseline/ldpc/crc.py",
    "src/config/execution_profiles.py",
    "src/config/params.py",
    "src/config/run_config.py",
    "src/env.py",
    "spec/params.generated.yaml",
    "requirements-pascal.lock",
    "results/baseline/g8/required_bler_identities.json",
    "results/baseline/g8_pascal_successor/pre_measurement_repair.json",
)

PRODUCTION_PROVENANCE_FIELDS = (
    "execution_profile_id",
    "lock_file",
    "lock_file_sha256",
    "python_version",
    "torch_version",
    "torch_cuda_build",
    "torchvision_version",
    "numpy_version",
    "sionna_version",
    "openjpeg_version",
    "deterministic_backend",
    "amp",
    "gpu_name",
    "gpu_uuid",
    "gpu_vram_mib",
    "gpu_compute_capability",
    "gpu_index",
    "nvidia_smi_index",
    "driver_version",
    "device",
    "git_commit",
    "git_dirty",
    "config_hash",
)

REQUEST_ARTIFACT_ROLE = "g8_c_pascal_successor_work_unit_request"
RESULT_ARTIFACT_ROLE = "g8_c_pascal_successor_work_unit_result"
REQUEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1

MAX_UNITS_POLICY = "maximum_attempted_work_units_per_worker_invocation"
FAILED_WORK_UNIT_POLICY = "failed_work_unit_counts_toward_cap_and_terminates_worker_batch"
PASCAL_SUCCESSOR_CUSTODY_POLICY = {
    "scope": "owner_authorized_confessor_pascal_cu126_g8_c_successor_only",
    "local_evidence_accumulation": "continuous_authenticated_per_unit_evidence_on_sole_writer_permitted",
    "git_publication_timing": "after_unattended_campaign_or_owner_selected_manual_checkpoint",
    "prepublication_loss_risk": "explicitly_accepted_by_owner",
    "scientific_validity_basis": "authenticated_per_unit_evidence_and_final_complete_coverage_verification",
    "final_handoff": "reconcile_commit_authenticated_https_push_fetch_parity_required_before_bler_table_freeze_or_g8_d",
}

STATUS_CLAIMED = "claimed"
STATUS_REQUEST_PUBLISHED = "request_published"
STATUS_RESULT_PUBLISHED = "result_published"
STATUS_ACCEPTED = "accepted"
STATUS_FAILED = "failed"
STATUS_TERMINAL_INVALID = "terminal_invalid"
STATE_STATUSES = (
    STATUS_CLAIMED,
    STATUS_REQUEST_PUBLISHED,
    STATUS_RESULT_PUBLISHED,
    STATUS_ACCEPTED,
    STATUS_FAILED,
    STATUS_TERMINAL_INVALID,
)

REQUEST_FILENAME_RE = re.compile(r"^(?P<digest>[0-9a-f]{64})\.attempt-(?P<attempt>[1-9][0-9]*)\.request\.json$")
RESULT_FILENAME_RE = re.compile(r"^(?P<digest>[0-9a-f]{64})\.attempt-(?P<attempt>[1-9][0-9]*)\.result\.json$")
STATE_FILENAME_RE = re.compile(r"^(?P<digest>[0-9a-f]{64})\.state\.json$")
BUCKET_RE = re.compile(r"^[0-9a-f]{2}$")
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


class SuccessorProductionError(RuntimeError):
    """Base class for successor production transaction failures."""


class ProductionContractError(SuccessorProductionError):
    """A successor contract, request, result or profile binding is invalid."""


class PublicationConflict(SuccessorProductionError):
    """An immutable artifact or state CAS publication conflicts with bytes on disk."""


class StaleStateError(PublicationConflict):
    """The state predecessor digest no longer matches under the unit lock."""


class RuntimeRootError(SuccessorProductionError):
    """A runtime root is missing, aliased, malformed or points at predecessor data."""


class RecoveryError(SuccessorProductionError):
    """Durable evidence cannot be reconciled safely."""


def _exact_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ProductionContractError(f"{name} must be an exact integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _exact_int(value, name)
    if value <= 0:
        raise ProductionContractError(f"{name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    value = _exact_int(value, name)
    if value < 0:
        raise ProductionContractError(f"{name} must be non-negative")
    return value


def _digest(value: Any, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or HEX_DIGEST_RE.fullmatch(value) is None:
        raise ProductionContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionContractError(f"{name} must be a nonblank string")
    return value


def _canonical(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionContractError(f"{name} must be a mapping")
    try:
        body = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionContractError(f"{name} is not finite canonical JSON") from exc
    if not isinstance(body, dict):  # pragma: no cover - guarded by Mapping above
        raise ProductionContractError(f"{name} is not a JSON object")
    return body


def _same_keys(value: Any, expected: Sequence[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ProductionContractError(f"{name} has missing or unexpected fields")


def _load_pre_measurement_repair_policy() -> dict[str, Any]:
    """Load and validate the one exact predecessor-failure exception.

    This record is deliberately a snapshot, not an old-contract allow-list.
    The runtime may use the compatibility path only for the two byte-pinned
    attempt-1 transactions recorded here.
    """

    try:
        payload = load_json(PRE_MEASUREMENT_REPAIR_POLICY)
    except Exception as exc:
        raise ProductionContractError("pre-measurement repair policy cannot be loaded") from exc
    _same_keys(
        payload,
        (
            "artifact_role", "campaign_id", "execution_profile_id", "observed_failure",
            "defect", "predecessor", "schema_version", "state_snapshot", "transition", "units",
        ),
        "pre-measurement repair policy",
    )
    if payload["schema_version"] != 1 or payload["artifact_role"] != "g8_c_pascal_successor_pre_measurement_repair_policy":
        raise ProductionContractError("unsupported pre-measurement repair policy")
    if payload["campaign_id"] != successor_campaign_identifier(load_json(SUCCESSOR_MANIFEST)) or payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID:
        raise ProductionContractError("pre-measurement repair policy campaign/profile differs")

    if payload["defect"] != {
        "class": "successor_nested_identity_not_adapted_for_frozen_runner",
        "frozen_runner_lookup": "request['bler_identity']",
        "lookup_precedes": ["SionnaLDPCAdapter construction", "Philox RNG construction", "trial loop"],
        "observed_failure": "KeyError: 'bler_identity'",
        "repair": "explicit nested successor identity to frozen-runner request view; frozen runner unchanged",
    }:
        raise ProductionContractError("pre-measurement defect record differs")

    observed = payload["observed_failure"]
    _same_keys(
        observed,
        ("bit_errors", "block_errors", "coverage_contribution", "protected_counters", "result_status", "test_access", "trials_completed"),
        "pre-measurement observed failure",
    )
    if observed != {
        "bit_errors": 0,
        "block_errors": 0,
        "coverage_contribution": 0,
        "protected_counters": {"inference": 0, "test_access": 0, "training": 0, "validation_decoding": 0},
        "result_status": "failed",
        "test_access": 0,
        "trials_completed": 0,
    }:
        raise ProductionContractError("pre-measurement observed failure is not zero-trial failed evidence")

    predecessor = payload["predecessor"]
    _same_keys(
        predecessor,
        (
            "campaign_manifest_sha256", "coordinator_contract_sha256", "lock_file", "lock_file_sha256",
            "production_contract_sha256", "required_bler_artifact_sha256", "runner_contract_sha256",
            "source_manifest_sha256",
        ),
        "pre-measurement predecessor binding",
    )
    for key in (
        "campaign_manifest_sha256", "coordinator_contract_sha256", "lock_file_sha256",
        "production_contract_sha256", "required_bler_artifact_sha256", "runner_contract_sha256",
        "source_manifest_sha256",
    ):
        _digest(predecessor[key], f"pre-measurement predecessor {key}")
    if predecessor["lock_file"] != "requirements-pascal.lock":
        raise ProductionContractError("pre-measurement predecessor lock filename differs")
    if predecessor["campaign_manifest_sha256"] != hashlib.sha256(SUCCESSOR_MANIFEST.read_bytes()).hexdigest():
        raise ProductionContractError("pre-measurement predecessor campaign manifest differs")
    lock_path = REPO_ROOT / "requirements-pascal.lock"
    if predecessor["lock_file_sha256"] != hashlib.sha256(lock_path.read_bytes()).hexdigest():
        raise ProductionContractError("pre-measurement predecessor lock bytes differ")
    required_path = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
    if predecessor["required_bler_artifact_sha256"] != hashlib.sha256(required_path.read_bytes()).hexdigest():
        raise ProductionContractError("pre-measurement predecessor required grid differs")

    snapshot = payload["state_snapshot"]
    _same_keys(
        snapshot,
        (
            "accepted_authority_ordinals", "campaign_state_file_sha256", "failed_authority_ordinals",
            "in_progress_authority_ordinals", "scientific_execution_performed", "terminal_invalid_authority_ordinals",
        ),
        "pre-measurement campaign state snapshot",
    )
    _digest(snapshot["campaign_state_file_sha256"], "pre-measurement campaign state file SHA-256")
    if snapshot != {
        "accepted_authority_ordinals": [],
        "campaign_state_file_sha256": snapshot["campaign_state_file_sha256"],
        "failed_authority_ordinals": [0, 1],
        "in_progress_authority_ordinals": [],
        "scientific_execution_performed": True,
        "terminal_invalid_authority_ordinals": [],
    }:
        raise ProductionContractError("pre-measurement campaign state snapshot is not the observed [0,1] failure")

    transition = payload["transition"]
    _same_keys(
        transition,
        (
            "current_contract_required_for_new_execution", "from", "no_general_source_switch",
            "refreshed_owner_authorization_required", "retry_starts_from_trial_zero", "to",
        ),
        "pre-measurement retry transition",
    )
    if transition != {
        "current_contract_required_for_new_execution": True,
        "from": {"attempt": 1, "production_contract_epoch": "predecessor", "status": "failed", "trials_completed": 0},
        "no_general_source_switch": True,
        "refreshed_owner_authorization_required": True,
        "retry_starts_from_trial_zero": True,
        "to": {"attempt": "predecessor_attempt_plus_one", "production_contract_epoch": "current", "status": "claimed", "trials_completed": 0},
    }:
        raise ProductionContractError("pre-measurement retry transition policy differs")

    units = payload["units"]
    if not isinstance(units, list) or len(units) != 2:
        raise ProductionContractError("pre-measurement repair policy must bind exactly two units")
    for expected_ordinal, record in enumerate(units):
        _same_keys(
            record,
            ("attempt", "authority_ordinal", "device", "gpu_uuid", "request_file_sha256", "result_file_sha256", "state_file_sha256", "work_unit_id"),
            f"pre-measurement unit {expected_ordinal}",
        )
        if record["authority_ordinal"] != expected_ordinal or record["attempt"] != 1:
            raise ProductionContractError("pre-measurement unit ordinal/attempt differs")
        _digest(record["request_file_sha256"], f"pre-measurement unit {expected_ordinal} request file SHA-256")
        _digest(record["result_file_sha256"], f"pre-measurement unit {expected_ordinal} result file SHA-256")
        _digest(record["state_file_sha256"], f"pre-measurement unit {expected_ordinal} state file SHA-256")
        unit = _required_unit_by_ordinal(expected_ordinal)
        if record["work_unit_id"] != unit["work_unit_id"]:
            raise ProductionContractError("pre-measurement unit identity differs")
        worker = next((item for item in PRODUCTION_WORKERS if item["shard_index"] == expected_ordinal % 2), None)
        if worker is None or record["device"] != worker["device"] or record["gpu_uuid"] != worker["gpu_uuid"]:
            raise ProductionContractError("pre-measurement unit GPU binding differs")
    return payload


def _repair_unit_record(ordinal: int) -> dict[str, Any]:
    policy = _load_pre_measurement_repair_policy()
    for record in policy["units"]:
        if record["authority_ordinal"] == ordinal:
            return dict(record)
    raise ProductionContractError("authority ordinal is not in the exact pre-measurement repair record")


def _legacy_bindings() -> dict[str, Any]:
    """Return predecessor scalar bindings only for the exact repair record."""

    current = successor_bindings()
    predecessor = _load_pre_measurement_repair_policy()["predecessor"]
    legacy = dict(current)
    legacy.update({
        "campaign_manifest_sha256": predecessor["campaign_manifest_sha256"],
        "coordinator_contract_sha256": predecessor["coordinator_contract_sha256"],
        "lock_file": predecessor["lock_file"],
        "lock_file_sha256": predecessor["lock_file_sha256"],
        "production_contract_sha256": predecessor["production_contract_sha256"],
        "required_bler_artifact_sha256": predecessor["required_bler_artifact_sha256"],
        "runner_contract_sha256": predecessor["runner_contract_sha256"],
        "source_manifest_sha256": predecessor["source_manifest_sha256"],
    })
    return legacy


def _legacy_state_snapshot(state: Mapping[str, Any], raw: bytes | None = None, *, ordinal: int | None = None) -> dict[str, Any]:
    """Validate one byte-pinned old failed state, never an arbitrary old epoch."""

    payload = _canonical(state, "pre-measurement predecessor state")
    if ordinal is None:
        ordinal = int(payload.get("identity", {}).get("authority_ordinal", -1))
    record = _repair_unit_record(ordinal)
    if raw is None:
        raw = canonical_json(payload)
    if sha256_bytes(raw) != record["state_file_sha256"]:
        raise ProductionContractError("pre-measurement predecessor state bytes are not the recorded failure")
    validated = validate_production_state_snapshot(payload, bindings=_legacy_bindings())
    identity = validated["identity"]
    if identity["attempt"] != 1 or identity["status"] != STATUS_FAILED or identity["trials_completed"] != 0 or identity["result_status"] != "failed":
        raise ProductionContractError("pre-measurement predecessor state is not a zero-trial failed attempt 1")
    return validated


def _legacy_attempt_evidence(
    ordinal: int,
    attempt: int,
    request: Mapping[str, Any],
    request_raw: bytes,
    result: Mapping[str, Any],
    result_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact old request/result pair admitted by the repair policy."""

    if attempt != 1:
        raise ProductionContractError("pre-measurement compatibility is limited to attempt 1")
    record = _repair_unit_record(ordinal)
    if sha256_bytes(request_raw) != record["request_file_sha256"] or sha256_bytes(result_raw) != record["result_file_sha256"]:
        raise ProductionContractError("pre-measurement predecessor request/result bytes are not the recorded failure")
    legacy = _legacy_bindings()
    validated_request = validate_request(request, bindings=legacy)
    validated_result = validate_result(result, bindings=legacy, request=validated_request, attempt=1)
    observed = _load_pre_measurement_repair_policy()["observed_failure"]
    if (
        validated_result["status"] != observed["result_status"]
        or validated_result["measurement"]["trials_completed"] != observed["trials_completed"]
        or validated_result["measurement"]["bit_errors"] != observed["bit_errors"]
        or validated_result["measurement"]["block_errors"] != observed["block_errors"]
        or validated_result["disposition"]["required_coverage_contribution"] != observed["coverage_contribution"]
        or validated_result["disposition"]["protected_counters"] != observed["protected_counters"]
        or validated_result["disposition"]["test_access"] != observed["test_access"]
    ):
        raise ProductionContractError("pre-measurement predecessor result is not zero-trial non-coverage evidence")
    return validated_request, validated_result


@lru_cache(maxsize=1)
def _successor_bindings_json() -> bytes:
    """Read the live successor artifacts and return immutable scalar bindings."""

    manifest = load_json(SUCCESSOR_MANIFEST)
    campaign_id = manifest.get("campaign_id")
    if campaign_id != successor_campaign_identifier(manifest):
        raise ProductionContractError("successor campaign identity does not reproduce")
    if manifest.get("execution_profile_id") != SUCCESSOR_PROFILE_ID:
        raise ProductionContractError("successor profile binding differs")
    if manifest.get("required_identity_count") != REQUIRED_COUNT or manifest.get("trials_per_identity") != TRIALS_PER_IDENTITY:
        raise ProductionContractError("successor physical grid binding differs")
    source_sha = hashlib.sha256(SUCCESSOR_SOURCE_MANIFEST.read_bytes()).hexdigest()
    if manifest.get("source_manifest_sha256") != source_sha:
        raise ProductionContractError("successor source manifest digest differs from campaign")
    # The schema-1 files are the immutable zero-coverage readiness marker.
    # Production has a separate additive contract family so that opening the
    # transaction protocol never rewrites that historical marker.
    runner_sha = hashlib.sha256(PRODUCTION_RUNNER_CONTRACT.read_bytes()).hexdigest()
    coordinator_sha = hashlib.sha256(PRODUCTION_COORDINATOR_CONTRACT.read_bytes()).hexdigest()
    source = load_json(PRODUCTION_SOURCE_MANIFEST)
    if source.get("campaign_id") != campaign_id:
        raise ProductionContractError("successor production source campaign binding differs")
    if source.get("coordinator_contract_sha256") != coordinator_sha:
        raise ProductionContractError("successor production source/coordinator campaign binding differs")
    contract = load_json(PRODUCTION_CONTRACT)
    contract_sha = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()
    production_source_sha = hashlib.sha256(PRODUCTION_SOURCE_MANIFEST.read_bytes()).hexdigest()
    repair_policy = _load_pre_measurement_repair_policy()
    repair_policy_sha = hashlib.sha256(PRE_MEASUREMENT_REPAIR_POLICY.read_bytes()).hexdigest()
    if contract.get("campaign_id") != campaign_id or contract.get("source_manifest_sha256") != production_source_sha:
        raise ProductionContractError("successor production contract/source binding differs")
    if contract.get("runner_contract_sha256") != runner_sha or contract.get("coordinator_contract_sha256") != coordinator_sha:
        raise ProductionContractError("successor production contract execution binding differs")
    if contract.get("pre_measurement_repair_policy_sha256") != repair_policy_sha or contract.get("pre_measurement_retry_compatibility") != repair_policy:
        raise ProductionContractError("successor production contract repair policy binding differs")
    lock_path = REPO_ROOT / "requirements-pascal.lock"
    required_raw = (REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes()
    return canonical_json({
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(SUCCESSOR_MANIFEST.read_bytes()).hexdigest(),
        "source_manifest_sha256": production_source_sha,
        "readiness_source_manifest_sha256": source_sha,
        "runner_contract_sha256": runner_sha,
        "coordinator_contract_sha256": coordinator_sha,
        "required_bler_artifact_sha256": hashlib.sha256(required_raw).hexdigest(),
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "production_contract_sha256": contract_sha,
        "pre_measurement_repair_policy_sha256": repair_policy_sha,
    })


def successor_bindings() -> dict[str, Any]:
    """Return a fresh binding mapping while caching immutable contract bytes."""

    return json.loads(_successor_bindings_json())


@lru_cache(maxsize=1)
def _required_unit_bytes() -> tuple[bytes, ...]:
    path = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
    try:
        payload = json.loads(path.read_bytes())
        units = payload["required_bler_work_units"]
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise ProductionContractError("required BLER identity artifact cannot be read") from exc
    if not isinstance(units, list) or len(units) != REQUIRED_COUNT:
        raise ProductionContractError("required BLER identity artifact does not contain 3213 units")
    result: list[bytes] = []
    seen: set[str] = set()
    for ordinal, unit in enumerate(units):
        if not isinstance(unit, Mapping) or not isinstance(unit.get("work_unit_id"), str):
            raise ProductionContractError("required identity record is malformed")
        if unit["work_unit_id"] in seen:
            raise ProductionContractError("required identity work-unit IDs are duplicated")
        seen.add(unit["work_unit_id"])
        if "ordinal" in unit and unit["ordinal"] != ordinal:
            raise ProductionContractError("required identity ordinal does not match array authority")
        result.append(canonical_json(dict(unit)))
    return tuple(result)


@lru_cache(maxsize=1)
def _required_unit_ordinals() -> dict[str, int]:
    return {
        json.loads(raw)["work_unit_id"]: ordinal
        for ordinal, raw in enumerate(_required_unit_bytes())
    }


def _required_unit_by_ordinal(ordinal: int) -> dict[str, Any]:
    ordinal = _nonnegative_int(ordinal, "authority ordinal")
    if ordinal >= REQUIRED_COUNT:
        raise ProductionContractError("authority ordinal is outside the successor grid")
    return json.loads(_required_unit_bytes()[ordinal])


def _required_unit_by_id(work_unit_id: str) -> tuple[int, dict[str, Any]]:
    work_unit_id = _nonblank(work_unit_id, "work_unit_id")
    ordinal = _required_unit_ordinals().get(work_unit_id)
    if ordinal is None:
        raise ProductionContractError("work unit is not an exact required successor identity")
    return ordinal, _required_unit_by_ordinal(ordinal)


def unit_digest(campaign_id: str, work_unit_id: str) -> str:
    _nonblank(campaign_id, "campaign_id")
    _nonblank(work_unit_id, "work_unit_id")
    return sha256_bytes(canonical_json({"campaign_id": campaign_id, "work_unit_id": work_unit_id}))


def _runtime_root(root: Path | str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        raise RuntimeRootError("successor runtime root must be absolute")
    try:
        resolved = path.resolve(strict=False)
        old_resolved = OLD_WORK_UNIT_ROOT.resolve(strict=False)
        successor_tracked = SUCCESSOR_ROOT.resolve(strict=False)
    except OSError as exc:
        raise RuntimeRootError(f"cannot resolve successor runtime root: {exc}") from exc
    if (
        resolved == old_resolved
        or old_resolved in resolved.parents
        or resolved == successor_tracked
    ):
        raise RuntimeRootError("successor runtime root aliases old data or the tracked successor root")
    if resolved == Path("/"):
        raise RuntimeRootError("filesystem root is not a successor runtime root")
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        entry = None
    except OSError as exc:
        raise RuntimeRootError(f"cannot inspect successor runtime root: {exc}") from exc
    if entry is not None and (stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode)):
        raise RuntimeRootError("successor runtime root is not a real directory")
    return path


def ensure_runtime_root(root: Path | str) -> Path:
    path = _runtime_root(root)
    try:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise RuntimeRootError("successor runtime root is not a real directory")
        else:
            if not path.parent.is_dir():
                raise RuntimeRootError("successor runtime-root parent must already exist")
            path.mkdir(mode=0o700)
        locks = path / ".locks"
        if locks.exists() and (locks.is_symlink() or not locks.is_dir()):
            raise RuntimeRootError("successor runtime lock directory is not real")
        locks.mkdir(mode=0o700, exist_ok=True)
        return path
    except OSError as exc:
        raise RuntimeRootError(f"cannot prepare successor runtime root: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SuccessorProductionError(f"directory fsync failed for {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _ensure_real_bucket(path: Path) -> None:
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise RuntimeRootError(f"cannot create successor artifact bucket: {exc}") from exc
        entry = os.lstat(path)
    except OSError as exc:
        raise RuntimeRootError(f"cannot inspect successor artifact bucket: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise RuntimeRootError(f"successor artifact bucket is not a real directory: {path}")


@contextlib.contextmanager
def unit_lock(root: Path | str, digest: str) -> Iterator[None]:
    root_path = ensure_runtime_root(root)
    if HEX_DIGEST_RE.fullmatch(digest) is None:
        raise RuntimeRootError("unit lock digest is malformed")
    lock_path = root_path / ".locks" / f"{digest}.lock"
    try:
        try:
            entry = os.lstat(lock_path)
        except FileNotFoundError:
            entry = None
        if entry is not None and (stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1):
            raise RuntimeRootError("successor unit lock is not a private regular file")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, _FILE_MODE)
        if os.fstat(descriptor).st_nlink != 1:
            os.close(descriptor)
            raise RuntimeRootError("successor unit lock is a hard-link alias")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as exc:
        raise RuntimeRootError(f"cannot acquire successor unit lock: {exc}") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def campaign_lock(root: Path | str) -> Iterator[None]:
    root_path = ensure_runtime_root(root)
    lock_path = root_path / ".campaign.lock"
    try:
        try:
            entry = os.lstat(lock_path)
        except FileNotFoundError:
            entry = None
        if entry is not None and (stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1):
            raise RuntimeRootError("successor campaign lock is not a private regular file")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, _FILE_MODE)
        if os.fstat(descriptor).st_nlink != 1:
            os.close(descriptor)
            raise RuntimeRootError("successor campaign lock is a hard-link alias")
    except OSError as exc:
        raise RuntimeRootError(f"cannot open successor campaign lock: {exc}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_regular(path: Path, label: str) -> bytes | None:
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeRootError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
        raise RuntimeRootError(f"{label} is not a single regular file")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read()
    except OSError as exc:
        raise RuntimeRootError(f"cannot read {label}: {exc}") from exc


def _stage_and_publish(path: Path, body: bytes, *, replace: bool) -> str:
    _ensure_real_bucket(path.parent)
    staging = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.staging"  # literal-ok: cryptographic staging-name entropy; filesystem uniqueness only.
    descriptor = None
    parent_descriptor = None
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            _FILE_MODE,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        parent_descriptor = os.open(path.parent, _DIRECTORY_FLAGS)
        if replace:
            os.replace(
                staging.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        else:
            os.link(
                staging.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(staging.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except FileExistsError as exc:
        raise PublicationConflict(f"publication target already exists: {path}") from exc
    except OSError as exc:
        raise SuccessorProductionError(f"cannot publish {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
    return sha256_bytes(body)


def publish_immutable_json(path: Path, payload: Mapping[str, Any], *, root: Path) -> str:
    """Publish request/result bytes without replacing an existing target."""

    body = canonical_json(dict(payload))
    existing = _read_regular(path, "immutable artifact")
    if existing is not None:
        if existing == body:
            return sha256_bytes(body)
        raise PublicationConflict(f"immutable artifact conflicts at {path}")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeRootError("immutable successor artifact is outside its runtime root") from exc
    return _stage_and_publish(path, body, replace=False)


def _read_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    if raw is None:
        raise RecoveryError(f"missing {label}: {path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"{label} is malformed JSON: {path}") from exc
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise RecoveryError(f"{label} is not canonical JSON: {path}")
    return payload, raw


def _write_state_cas(
    root: Path,
    state: Mapping[str, Any],
    *,
    expected_sha256: str | None,
) -> str:
    payload = validate_production_state_snapshot(state)
    digest = state_sha256(payload)
    path = state_path(root, payload["identity"]["authority_ordinal"], payload["identity"]["work_unit_id"])
    current = _read_regular(path, "unit state")
    if current is None:
        if expected_sha256 is not None:
            raise StaleStateError("state target disappeared before compare-and-swap")
        return _stage_and_publish(path, canonical_json(payload), replace=False)
    current_sha = sha256_bytes(current)
    if current_sha != expected_sha256:
        raise StaleStateError("unit state predecessor digest differs")
    if current == canonical_json(payload):
        return digest
    return _stage_and_publish(path, canonical_json(payload), replace=True)


def _state_identity(
    bindings: Mapping[str, Any],
    *,
    ordinal: int,
    unit: Mapping[str, Any],
    attempt: int,
    status: str,
    shard_index: int,
    shard_count: int,
    device: str,
    gpu_uuid: str,
    request_sha256: str | None,
    result_sha256: str | None,
    result_status: str | None,
    scientific_execution_performed: bool,
    trials_completed: int,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "artifact_role": PRODUCTION_STATE_ARTIFACT_ROLE,
        "campaign_id": bindings["campaign_id"],
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "runner_contract_sha256": bindings["runner_contract_sha256"],
        "production_contract_sha256": bindings["production_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "work_unit_id": unit["work_unit_id"],
        "authority_ordinal": ordinal,
        "required_work_unit_record_sha256": sha256_bytes(canonical_json(dict(unit))),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "attempt": attempt,
        "status": status,
        "request_sha256": request_sha256,
        "result_sha256": result_sha256,
        "result_status": result_status,
        "scientific_execution_performed": scientific_execution_performed,
        "trials_completed": trials_completed,
        "test_access": 0,
        "reason": reason,
    }


def build_state(
    bindings: Mapping[str, Any],
    *,
    ordinal: int,
    unit: Mapping[str, Any],
    attempt: int,
    status: str,
    shard_index: int,
    shard_count: int,
    device: str,
    gpu_uuid: str,
    request_sha256: str | None = None,
    result_sha256: str | None = None,
    result_status: str | None = None,
    scientific_execution_performed: bool = False,
    trials_completed: int = 0,
    reason: str | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _state_identity(
        bindings,
        ordinal=ordinal,
        unit=unit,
        attempt=attempt,
        status=status,
        shard_index=shard_index,
        shard_count=shard_count,
        device=device,
        gpu_uuid=gpu_uuid,
        request_sha256=request_sha256,
        result_sha256=result_sha256,
        result_status=result_status,
        scientific_execution_performed=scientific_execution_performed,
        trials_completed=trials_completed,
        reason=reason,
    )
    payload = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "artifact_role": PRODUCTION_STATE_ARTIFACT_ROLE,
        "identity": identity,
        "runtime_provenance": dict(runtime_provenance or {}),
        "identity_sha256": sha256_bytes(canonical_json(identity)),
        "state_sha256": None,
    }
    payload["state_sha256"] = state_sha256(payload)
    return validate_production_state_snapshot(payload)


def state_sha256(state: Mapping[str, Any]) -> str:
    body = dict(state)
    body.pop("state_sha256", None)
    return sha256_bytes(canonical_json(body))


def state_path(root: Path | str, ordinal: int, work_unit_id: str) -> Path:
    bindings = successor_bindings()
    digest = unit_digest(bindings["campaign_id"], work_unit_id)
    if ordinal != _required_unit_by_id(work_unit_id)[0]:
        raise ProductionContractError("state ordinal does not reproduce from work-unit authority")
    return _runtime_root(root) / digest[:2] / f"{digest}.state.json"


def request_path(root: Path | str, work_unit_id: str, attempt: int) -> Path:
    bindings = successor_bindings()
    digest = unit_digest(bindings["campaign_id"], work_unit_id)
    ordinal, _ = _required_unit_by_id(work_unit_id)
    _ = ordinal
    return _runtime_root(root) / digest[:2] / f"{digest}.attempt-{_positive_int(attempt, 'attempt')}.request.json"


def result_path(root: Path | str, work_unit_id: str, attempt: int) -> Path:
    bindings = successor_bindings()
    digest = unit_digest(bindings["campaign_id"], work_unit_id)
    _required_unit_by_id(work_unit_id)
    return _runtime_root(root) / digest[:2] / f"{digest}.attempt-{_positive_int(attempt, 'attempt')}.result.json"


def _validate_profile_provenance(
    value: Any,
    *,
    device: str,
    gpu_uuid: str,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionContractError("execution profile provenance is missing")
    payload = dict(value)
    required = set(PRODUCTION_PROVENANCE_FIELDS)
    if set(payload) != required:
        raise ProductionContractError("execution profile provenance fields differ")
    if payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID or payload["device"] != device or payload["gpu_uuid"] != gpu_uuid:
        raise ProductionContractError("execution profile/device provenance differs")
    worker = next((item for item in PRODUCTION_WORKERS if item["device"] == device), None)
    if worker is None or worker["gpu_uuid"] != gpu_uuid:
        raise ProductionContractError("execution profile is not an authenticated registered worker mapping")
    if payload["gpu_name"] != worker["gpu_name"] or payload["gpu_compute_capability"] != worker["gpu_compute_capability"]:
        raise ProductionContractError("execution profile GPU identity differs from the registered worker mapping")
    _digest(payload["lock_file_sha256"], "lock_file_sha256")
    _nonblank(payload["driver_version"], "driver_version")
    if payload["lock_file"] != "requirements-pascal.lock":
        raise ProductionContractError("successor result does not bind the Pascal lock")
    if type(payload["git_dirty"]) is not bool or type(payload["amp"]) is not bool:
        raise ProductionContractError("profile boolean provenance is malformed")
    _nonblank(payload["git_commit"], "git_commit")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ProductionContractError("execution profile config hash differs from the production contract")
    return json.loads(canonical_json(payload))


def _request_identity(bindings: Mapping[str, Any], ordinal: int, unit: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(unit["identity"])
    return {
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": bindings["lock_file_sha256"],
        "campaign_id": bindings["campaign_id"],
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "runner_contract_sha256": bindings["runner_contract_sha256"],
        "production_contract_sha256": bindings["production_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "authority_ordinal": ordinal,
        "work_unit_id": unit["work_unit_id"],
        "required_work_unit_record_sha256": sha256_bytes(canonical_json(dict(unit))),
        "bler_identity": identity,
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "trials_requested": TRIALS_PER_IDENTITY,
        "trial_count_source": "params.baseline.bler_characterisation_trials",
        "seed_derivation_identity": frozen_bler.SEED_DERIVATION_IDENTITY,
        "seed_domain_separator": frozen_bler.SEED_DOMAIN_SEPARATOR,
        "stream_seeds": frozen_bler.stream_seed_records(bindings["campaign_id"], unit["work_unit_id"]),
        "implementation": {
            **frozen_bler.implementation_binding(),
            "successor_request_schema_version": REQUEST_SCHEMA_VERSION,
            "successor_result_schema_version": RESULT_SCHEMA_VERSION,
            "measurement_implementation": "frozen_g8_bler_runner._execute_measurement",
        },
        "profile_provenance": dict(profile),
    }


def build_request(
    bindings: Mapping[str, Any],
    *,
    ordinal: int,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    unit = _required_unit_by_ordinal(ordinal)
    profile_record = _validate_profile_provenance(
        profile,
        device=str(profile["device"]),
        gpu_uuid=str(profile["gpu_uuid"]),
        expected_config_hash=bindings["production_contract_sha256"],
    )
    body = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "artifact_role": REQUEST_ARTIFACT_ROLE,
        "identity": _request_identity(bindings, ordinal, unit, profile_record),
        "scientific_evidence": True,
        "merge_eligible": False,
        "test_access": 0,
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
    }
    return validate_request(body, bindings=bindings)


def validate_request(request: Mapping[str, Any], *, bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = _canonical(request, "successor request")
    _same_keys(payload, ("schema_version", "artifact_role", "identity", "scientific_evidence", "merge_eligible", "test_access", "protected_counters"), "successor request")
    if payload["schema_version"] != REQUEST_SCHEMA_VERSION or payload["artifact_role"] != REQUEST_ARTIFACT_ROLE:
        raise ProductionContractError("unsupported successor request")
    identity = payload["identity"]
    if not isinstance(identity, Mapping):
        raise ProductionContractError("successor request identity is missing")
    expected = successor_bindings() if bindings is None else bindings
    required = {
        "execution_profile_id", "lock_file", "lock_file_sha256", "campaign_id", "campaign_manifest_sha256",
        "source_manifest_sha256", "runner_contract_sha256", "production_contract_sha256",
        "required_bler_artifact_sha256", "authority_ordinal", "work_unit_id",
        "required_work_unit_record_sha256", "bler_identity", "snr_db", "source_packet_config_ids",
        "trials_requested", "trial_count_source", "seed_derivation_identity", "seed_domain_separator",
        "stream_seeds", "implementation", "profile_provenance",
    }
    if set(identity) != required:
        raise ProductionContractError("successor request identity schema differs")
    if any(identity[key] != expected[key] for key in (
        "campaign_id", "campaign_manifest_sha256", "source_manifest_sha256", "runner_contract_sha256",
        "production_contract_sha256", "required_bler_artifact_sha256", "execution_profile_id",
    )):
        raise ProductionContractError("successor request campaign/profile binding differs")
    if identity["lock_file"] != "requirements-pascal.lock" or identity["lock_file_sha256"] != expected["lock_file_sha256"]:
        raise ProductionContractError("successor request lock binding differs")
    ordinal = _nonnegative_int(identity["authority_ordinal"], "authority_ordinal")
    if ordinal >= REQUIRED_COUNT:
        raise ProductionContractError("successor request ordinal is outside the grid")
    ordinal_expected, unit = _required_unit_by_id(_nonblank(identity["work_unit_id"], "work_unit_id"))
    if ordinal != ordinal_expected:
        raise ProductionContractError("successor request ordinal/work-unit binding differs")
    if identity["required_work_unit_record_sha256"] != sha256_bytes(canonical_json(unit)):
        raise ProductionContractError("successor request required-identity digest differs")
    if identity["bler_identity"] != unit["identity"] or identity["snr_db"] != unit["snr_db"] or identity["source_packet_config_ids"] != unit["source_packet_config_ids"]:
        raise ProductionContractError("successor request physical identity differs")
    if identity["trials_requested"] != TRIALS_PER_IDENTITY or identity["trial_count_source"] != "params.baseline.bler_characterisation_trials":
        raise ProductionContractError("successor request trial count differs")
    if identity["seed_derivation_identity"] != frozen_bler.SEED_DERIVATION_IDENTITY or identity["seed_domain_separator"] != frozen_bler.SEED_DOMAIN_SEPARATOR:
        raise ProductionContractError("successor request seed contract differs")
    expected_seeds = frozen_bler.stream_seed_records(expected["campaign_id"], unit["work_unit_id"])
    if identity["stream_seeds"] != expected_seeds:
        raise ProductionContractError("successor request seeds do not reproduce")
    profile = identity["profile_provenance"]
    if not isinstance(profile, Mapping):
        raise ProductionContractError("successor request profile provenance is missing")
    profile_record = _validate_profile_provenance(
        profile,
        device=str(profile.get("device")),
        gpu_uuid=str(profile.get("gpu_uuid")),
        expected_config_hash=expected["production_contract_sha256"],
    )
    if profile_record["lock_file_sha256"] != expected["lock_file_sha256"]:
        raise ProductionContractError("successor request profile lock provenance differs")
    worker = next(item for item in PRODUCTION_WORKERS if item["device"] == profile_record["device"])
    if worker["shard_index"] != authority_shard(ordinal):
        raise ProductionContractError("successor request profile/device mapping is swapped across shards")
    if payload["scientific_evidence"] is not True or payload["merge_eligible"] is not False or payload["test_access"] != 0:
        raise ProductionContractError("successor request disposition is unsafe")
    if payload["protected_counters"] != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise ProductionContractError("successor request protected counters are nonzero")
    return payload


def build_result(
    request: Mapping[str, Any],
    measurement: Mapping[str, Any],
    *,
    attempt: int,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_request(request)
    identity = request["identity"]
    status = str(measurement.get("status"))
    if status not in {"complete", "failed"}:
        raise ProductionContractError("successor measurement status is not complete/failed")
    completed = _nonnegative_int(measurement.get("trials_completed"), "trials_completed")
    bit_errors = _nonnegative_int(measurement.get("bit_errors"), "bit_errors")
    block_errors = _nonnegative_int(measurement.get("block_errors"), "block_errors")
    k = int(identity["bler_identity"]["k_and_n"][0])
    try:
        derived = frozen_bler.recompute_measurements(
            trials_completed=completed,
            information_bits=completed * k,
            bit_errors=bit_errors,
            block_errors=block_errors,
            information_length=k,
        )
    except Exception as exc:
        raise ProductionContractError(f"successor count semantics failed: {exc}") from exc
    if status == "complete" and completed != TRIALS_PER_IDENTITY:
        raise ProductionContractError("a complete successor result must contain exactly 5000 trials")
    profile_record = _validate_profile_provenance(
        profile,
        device=str(profile["device"]),
        gpu_uuid=str(profile["gpu_uuid"]),
        expected_config_hash=identity["production_contract_sha256"],
    )
    if profile_record != identity["profile_provenance"]:
        raise ProductionContractError("successor result provenance differs from its request provenance")
    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_role": RESULT_ARTIFACT_ROLE,
        "status": status,
        "request_sha256": sha256_bytes(canonical_json(request)),
        "identity": dict(identity),
        "measurement": {
            "trials_completed": completed,
            "information_bits": completed * k,
            "bit_errors": bit_errors,
            "block_errors": block_errors,
            **derived,
            "confidence_interval_method": frozen_bler.CONFIDENCE_INTERVAL_METHOD,
            "confidence_interval_percent": frozen_bler.CONFIDENCE_INTERVAL_PERCENT,
            "confidence_interval_role": frozen_bler.CONFIDENCE_INTERVAL_ROLE,
        },
        "execution_provenance": profile_record,
        "attempt": _positive_int(attempt, "attempt"),
        "disposition": {
            "scientific_evidence": True,
            "merge_eligible": status == "complete",
            "required_coverage_contribution": 1 if status == "complete" else 0,
            "test_access": 0,
            "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        },
        "result_sha256": None,
    }
    body["result_sha256"] = sha256_bytes(canonical_json({key: value for key, value in body.items() if key != "result_sha256"}))
    return validate_result(body, bindings=successor_bindings())


def validate_result(
    result: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    payload = _canonical(result, "successor result")
    _same_keys(payload, ("schema_version", "artifact_role", "status", "request_sha256", "identity", "measurement", "execution_provenance", "attempt", "disposition", "result_sha256"), "successor result")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION or payload["artifact_role"] != RESULT_ARTIFACT_ROLE:
        raise ProductionContractError("unsupported successor result")
    if payload["status"] not in {"complete", "failed"}:
        raise ProductionContractError("successor result status is invalid")
    result_attempt = _positive_int(payload["attempt"], "result attempt")
    expected = successor_bindings() if bindings is None else bindings
    _digest(payload["request_sha256"], "request_sha256")
    if request is not None and sha256_bytes(canonical_json(dict(request))) != payload["request_sha256"]:
        raise ProductionContractError("successor result request digest differs")
    if attempt is not None and result_attempt != _positive_int(attempt, "attempt"):
        raise ProductionContractError("successor result attempt differs from its immutable filename")
    request_identity = payload["identity"]
    request_body = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "artifact_role": REQUEST_ARTIFACT_ROLE,
        "identity": request_identity,
        "scientific_evidence": True,
        "merge_eligible": False,
        "test_access": 0,
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
    }
    validate_request(request_body, bindings=expected)
    measurement = payload["measurement"]
    if not isinstance(measurement, Mapping):
        raise ProductionContractError("successor result measurement is missing")
    for key in ("trials_completed", "information_bits", "bit_errors", "block_errors"):
        _nonnegative_int(measurement.get(key), f"measurement.{key}")
    k = int(request_identity["bler_identity"]["k_and_n"][0])
    if measurement["information_bits"] != measurement["trials_completed"] * k:
        raise ProductionContractError("successor information-bit count does not reproduce")
    expected_measurement = frozen_bler.recompute_measurements(
        trials_completed=measurement["trials_completed"],
        information_bits=measurement["information_bits"],
        bit_errors=measurement["bit_errors"],
        block_errors=measurement["block_errors"],
        information_length=k,
    )
    for key, value in expected_measurement.items():
        if measurement.get(key) != value:
            raise ProductionContractError(f"successor derived measurement differs: {key}")
    if payload["status"] == "complete" and measurement["trials_completed"] != TRIALS_PER_IDENTITY:
        raise ProductionContractError("complete successor result does not contain 5000 trials")
    if payload["status"] == "failed" and payload["disposition"]["required_coverage_contribution"] != 0:
        raise ProductionContractError("failed successor result contributes coverage")
    execution_profile = _validate_profile_provenance(
        payload["execution_provenance"],
        device=str(payload["execution_provenance"].get("device")),
        gpu_uuid=str(payload["execution_provenance"].get("gpu_uuid")),
        expected_config_hash=request_identity["production_contract_sha256"],
    )
    if execution_profile != request_identity["profile_provenance"]:
        raise ProductionContractError("successor result provenance differs from its request")
    disposition = payload["disposition"]
    if disposition.get("scientific_evidence") is not True or disposition.get("test_access") != 0:
        raise ProductionContractError("successor result disposition is unsafe")
    if disposition.get("merge_eligible") != (payload["status"] == "complete"):
        raise ProductionContractError("successor result merge disposition differs")
    if disposition.get("protected_counters") != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise ProductionContractError("successor result protected counters are nonzero")
    supplied = payload["result_sha256"]
    _digest(supplied, "result_sha256")
    body = dict(payload)
    body.pop("result_sha256")
    if supplied != sha256_bytes(canonical_json(body)):
        raise ProductionContractError("successor result digest does not reproduce")
    return payload


def validate_production_state_snapshot(
    state: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _canonical(state, "successor production state")
    _same_keys(payload, ("schema_version", "artifact_role", "identity", "runtime_provenance", "identity_sha256", "state_sha256"), "successor production state")
    if payload["schema_version"] != PRODUCTION_SCHEMA_VERSION or payload["artifact_role"] != PRODUCTION_STATE_ARTIFACT_ROLE:
        raise ProductionContractError("unsupported successor production state")
    identity = payload["identity"]
    if not isinstance(identity, Mapping):
        raise ProductionContractError("successor production state identity is missing")
    required = {
        "schema_version", "artifact_role", "campaign_id", "campaign_manifest_sha256", "source_manifest_sha256",
        "runner_contract_sha256", "production_contract_sha256", "required_bler_artifact_sha256", "execution_profile_id",
        "work_unit_id", "authority_ordinal", "required_work_unit_record_sha256", "shard_index", "shard_count",
        "device", "gpu_uuid", "attempt", "status", "request_sha256", "result_sha256", "result_status",
        "scientific_execution_performed", "trials_completed", "test_access", "reason",
    }
    if set(identity) != required:
        raise ProductionContractError("successor unit-state identity schema differs")
    bindings = successor_bindings() if bindings is None else bindings
    for key in ("campaign_id", "campaign_manifest_sha256", "source_manifest_sha256", "runner_contract_sha256", "production_contract_sha256", "required_bler_artifact_sha256", "execution_profile_id"):
        if identity[key] != bindings[key]:
            raise ProductionContractError(f"successor unit state binding differs: {key}")
    ordinal = _nonnegative_int(identity["authority_ordinal"], "authority_ordinal")
    if ordinal >= REQUIRED_COUNT:
        raise ProductionContractError("successor unit state ordinal is outside the grid")
    unit = _required_unit_by_ordinal(ordinal)
    if identity["work_unit_id"] != unit["work_unit_id"] or identity["required_work_unit_record_sha256"] != sha256_bytes(canonical_json(unit)):
        raise ProductionContractError("successor unit state required identity differs")
    shard_count = _positive_int(identity["shard_count"], "shard_count")
    shard_index = _nonnegative_int(identity["shard_index"], "shard_index")
    if shard_count != 2 or shard_index >= shard_count or authority_shard(ordinal) != shard_index:
        raise ProductionContractError("successor unit state shard binding differs")
    _nonblank(identity["device"], "device")
    _nonblank(identity["gpu_uuid"], "gpu_uuid")
    if not isinstance(identity["scientific_execution_performed"], bool) or identity["test_access"] != 0:
        raise ProductionContractError("successor unit state protected flags are invalid")
    _positive_int(identity["attempt"], "attempt")
    if identity["status"] not in STATE_STATUSES:
        raise ProductionContractError("successor unit state status is invalid")
    _digest(identity["request_sha256"], "request_sha256", allow_none=True)
    _digest(identity["result_sha256"], "result_sha256", allow_none=True)
    if identity["status"] == STATUS_CLAIMED and (identity["request_sha256"], identity["result_sha256"], identity["result_status"], identity["scientific_execution_performed"], identity["trials_completed"]) != (None, None, None, False, 0):
        raise ProductionContractError("claimed successor state is not clean")
    if identity["status"] == STATUS_REQUEST_PUBLISHED and identity["request_sha256"] is None:
        raise ProductionContractError("request-published state lacks request binding")
    if identity["status"] == STATUS_REQUEST_PUBLISHED and (identity["result_sha256"], identity["result_status"], identity["scientific_execution_performed"], identity["trials_completed"]) != (None, None, False, 0):
        raise ProductionContractError("request-published successor state is not clean")
    if identity["status"] in {STATUS_RESULT_PUBLISHED, STATUS_ACCEPTED} and (identity["request_sha256"] is None or identity["result_sha256"] is None or identity["result_status"] is None):
        raise ProductionContractError("result state lacks request/result binding")
    if identity["status"] in {STATUS_RESULT_PUBLISHED, STATUS_ACCEPTED} and (identity["result_status"] != "complete" or identity["scientific_execution_performed"] is not True or identity["trials_completed"] != TRIALS_PER_IDENTITY):
        raise ProductionContractError("complete successor result state lacks exactly 5000 executed trials")
    if identity["status"] == STATUS_FAILED and (identity["request_sha256"] is None or identity["result_sha256"] is None or identity["result_status"] != "failed" or identity["scientific_execution_performed"] is not True):
        raise ProductionContractError("failed successor state lacks its failed result binding")
    if identity["status"] in {STATUS_CLAIMED, STATUS_REQUEST_PUBLISHED, STATUS_RESULT_PUBLISHED, STATUS_ACCEPTED} and identity["reason"] is not None:
        raise ProductionContractError("non-failed successor state carries a failure reason")
    if identity["status"] == STATUS_FAILED:
        _nonblank(identity["reason"], "failed successor state reason")
    if identity["status"] == STATUS_TERMINAL_INVALID:
        _nonblank(identity["reason"], "terminal invalid successor state reason")
    if payload["identity_sha256"] != sha256_bytes(canonical_json(dict(identity))):
        raise ProductionContractError("successor unit-state identity digest differs")
    _digest(payload["state_sha256"], "state_sha256")
    if payload["state_sha256"] != state_sha256(payload):
        raise ProductionContractError("successor unit-state digest differs")
    if not isinstance(payload["runtime_provenance"], Mapping):
        raise ProductionContractError("successor unit-state runtime provenance is malformed")
    profile = _validate_profile_provenance(
        payload["runtime_provenance"],
        device=str(identity["device"]),
        gpu_uuid=str(identity["gpu_uuid"]),
        expected_config_hash=identity["production_contract_sha256"],
    )
    if profile["lock_file_sha256"] != bindings["lock_file_sha256"]:
        raise ProductionContractError("successor unit-state lock provenance differs")
    worker = next(item for item in PRODUCTION_WORKERS if item["device"] == profile["device"])
    if worker["shard_index"] != shard_index:
        raise ProductionContractError("successor unit-state profile/device mapping is swapped across shards")
    return payload


def validate_state_transition(previous: Mapping[str, Any], proposed: Mapping[str, Any]) -> None:
    try:
        old_payload = validate_production_state_snapshot(previous)
        old_is_legacy_repair = False
    except ProductionContractError:
        old_payload = _legacy_state_snapshot(previous)
        old_is_legacy_repair = True
    old = old_payload["identity"]
    new = validate_production_state_snapshot(proposed)["identity"]
    if old_is_legacy_repair:
        if old["status"] != STATUS_FAILED or old["attempt"] != 1 or old["trials_completed"] != 0:
            raise PublicationConflict("only the recorded zero-trial failed attempt may cross the repair epoch")
        if new["status"] != STATUS_CLAIMED or new["attempt"] != 2:
            raise PublicationConflict("pre-measurement repair must create exactly attempt-2 clean claim")
        if (new["request_sha256"], new["result_sha256"], new["result_status"], new["scientific_execution_performed"], new["trials_completed"]) != (None, None, None, False, 0):
            raise PublicationConflict("pre-measurement repair retry must begin clean")
        for field in ("schema_version", "artifact_role", "campaign_id", "campaign_manifest_sha256", "required_bler_artifact_sha256", "execution_profile_id", "work_unit_id", "authority_ordinal", "required_work_unit_record_sha256"):
            if old[field] != new[field]:
                raise PublicationConflict(f"pre-measurement repair changed immutable field: {field}")
        if (old["shard_index"], old["shard_count"], old["device"], old["gpu_uuid"]) != (new["shard_index"], new["shard_count"], new["device"], new["gpu_uuid"]):
            raise PublicationConflict("pre-measurement repair changed shard/device/GPU identity")
        for field in PRODUCTION_PROVENANCE_FIELDS:
            if field not in {"config_hash", "git_commit"} and old_payload["runtime_provenance"][field] != proposed["runtime_provenance"][field]:
                raise PublicationConflict(f"pre-measurement repair changed authenticated profile field: {field}")
        return
    permanent = (
        "schema_version", "artifact_role", "campaign_id", "campaign_manifest_sha256", "source_manifest_sha256",
        "runner_contract_sha256", "production_contract_sha256", "required_bler_artifact_sha256", "execution_profile_id",
        "work_unit_id", "authority_ordinal", "required_work_unit_record_sha256",
    )
    for field in permanent:
        if old[field] != new[field]:
            raise PublicationConflict(f"successor unit-state immutable field changed: {field}")
    if old["status"] in {STATUS_ACCEPTED, STATUS_TERMINAL_INVALID}:
        if canonical_json(previous) != canonical_json(proposed):
            raise PublicationConflict("terminal successor state is immutable")
        return
    if new["attempt"] == old["attempt"] + 1:
        if old["status"] != STATUS_FAILED or new["status"] != STATUS_CLAIMED:
            raise PublicationConflict("only failed state may advance to a clean next attempt")
        if (new["request_sha256"], new["result_sha256"], new["result_status"], new["scientific_execution_performed"], new["trials_completed"]) != (None, None, None, False, 0):
            raise PublicationConflict("retry attempt must begin clean")
        return
    if new["attempt"] != old["attempt"]:
        raise PublicationConflict("successor attempt must remain equal or advance by one")
    if (old["shard_index"], old["shard_count"], old["device"], old["gpu_uuid"]) != (new["shard_index"], new["shard_count"], new["device"], new["gpu_uuid"]):
        raise PublicationConflict("successor device/shard mapping may not change within an attempt")
    if canonical_json(previous["runtime_provenance"]) != canonical_json(proposed["runtime_provenance"]):
        raise PublicationConflict("successor execution-profile provenance may not change within an attempt")
    allowed = {
        STATUS_CLAIMED: {STATUS_REQUEST_PUBLISHED, STATUS_TERMINAL_INVALID},
        STATUS_REQUEST_PUBLISHED: {STATUS_RESULT_PUBLISHED, STATUS_FAILED, STATUS_TERMINAL_INVALID},
        STATUS_RESULT_PUBLISHED: {STATUS_ACCEPTED, STATUS_TERMINAL_INVALID},
        STATUS_FAILED: set(),
    }
    if new["status"] not in allowed.get(old["status"], set()):
        raise PublicationConflict(f"illegal successor state transition {old['status']} -> {new['status']}")
    if new["trials_completed"] < old["trials_completed"]:
        raise PublicationConflict("successor trials_completed may not decrease")
    if old["scientific_execution_performed"] and not new["scientific_execution_performed"]:
        raise PublicationConflict("successor scientific execution flag may not clear")
    if old["request_sha256"] is not None and new["request_sha256"] != old["request_sha256"]:
        raise PublicationConflict("successor request digest may not change")


def read_state(root: Path | str, ordinal: int, work_unit_id: str) -> tuple[dict[str, Any], str] | None:
    path = state_path(root, ordinal, work_unit_id)
    raw = _read_regular(path, "successor unit state")
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"successor unit state is malformed: {path}") from exc
    if not isinstance(payload, Mapping) or canonical_json(payload) != raw:
        raise RecoveryError(f"successor unit state is not canonical: {path}")
    try:
        validated = validate_production_state_snapshot(payload)
    except ProductionContractError as current_error:
        try:
            validated = _legacy_state_snapshot(payload, raw, ordinal=ordinal)
        except ProductionContractError as legacy_error:
            raise RecoveryError(f"successor unit state is not bound to the current or exact repair epoch: {path}") from current_error
    return validated, sha256_bytes(raw)


def publish_state(
    root: Path | str,
    proposed: Mapping[str, Any],
    *,
    expected_sha256: str | None,
) -> str:
    validated = validate_production_state_snapshot(proposed)
    digest = unit_digest(validated["identity"]["campaign_id"], validated["identity"]["work_unit_id"])
    with unit_lock(root, digest):
        current = read_state(root, validated["identity"]["authority_ordinal"], validated["identity"]["work_unit_id"])
        if current is not None:
            if expected_sha256 != current[1]:
                raise StaleStateError("successor state CAS predecessor differs")
            validate_state_transition(current[0], validated)
        elif expected_sha256 is not None:
            raise StaleStateError("successor state disappeared before CAS")
        return _write_state_cas(Path(root), validated, expected_sha256=expected_sha256)


def _profile_bindings(profile: Mapping[str, Any], *, expected_device: str, expected_uuid: str) -> dict[str, Any]:
    if profile.get("device") != expected_device or profile.get("gpu_uuid") != expected_uuid:
        raise ProductionContractError("authenticated profile is not bound to the worker mapping")
    return dict(profile)


def _frozen_measurement_request_view(request: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the nested successor identity to the frozen runner's view.

    The successor request remains nested and immutable.  The frozen runner's
    validator interface predates that schema and consumes only these four
    physical fields at top level, so this function is the explicit boundary
    between the two contracts.
    """

    successor = validate_request(request)
    identity = successor["identity"]
    return {
        "bler_identity": dict(identity["bler_identity"]),
        "snr_db": identity["snr_db"],
        "trials_requested": identity["trials_requested"],
        "stream_seeds": dict(identity["stream_seeds"]),
    }


def _measurement_context() -> Any:
    """Return a validator shim so the frozen runner can execute successor requests."""

    class _Context:
        @staticmethod
        def validate_request(request: Mapping[str, Any], **_: Any) -> dict[str, Any]:
            return _frozen_measurement_request_view(request)

    return _Context()


def execute_frozen_measurement(request: Mapping[str, Any], *, device: str, batch_size: int) -> dict[str, Any]:
    """Call the exact predecessor-frozen PHY implementation for a successor request."""

    request = validate_request(request)
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise ProductionContractError("successor worker requires explicit cuda:N")
    try:
        return frozen_runner._execute_measurement(
            request,
            device=device,
            batch_size=_positive_int(batch_size, "batch_size"),
            context=_measurement_context(),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "trials_completed": 0,
            "bit_errors": 0,
            "block_errors": 0,
            "error": str(exc),
        }


def run_unit(
    root: Path | str,
    *,
    ordinal: int,
    shard_index: int,
    shard_count: int,
    device: str,
    gpu_uuid: str,
    profile: Mapping[str, Any],
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run one successor unit with crash-safe claim/request/result/state order."""

    root_path = ensure_runtime_root(root)
    if shard_count != 2 or shard_index not in {0, 1} or authority_shard(ordinal) != shard_index:
        raise ProductionContractError("successor worker shard assignment is invalid")
    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ProductionContractError("successor worker requires explicit cuda:N")
    bindings = successor_bindings()
    profile = _profile_bindings(profile, expected_device=device, expected_uuid=gpu_uuid)
    _validate_profile_provenance(
        profile,
        device=device,
        gpu_uuid=gpu_uuid,
        expected_config_hash=bindings["production_contract_sha256"],
    )
    unit = _required_unit_by_ordinal(ordinal)
    digest = unit_digest(bindings["campaign_id"], unit["work_unit_id"])
    with unit_lock(root_path, digest):
        existing = read_state(root_path, ordinal, unit["work_unit_id"])
        if existing is None:
            attempt = 1
            claim = build_state(
                bindings,
                ordinal=ordinal,
                unit=unit,
                attempt=attempt,
                status=STATUS_CLAIMED,
                shard_index=shard_index,
                shard_count=shard_count,
                device=device,
                gpu_uuid=gpu_uuid,
                runtime_provenance=profile,
            )
            claim_sha = _write_state_cas(root_path, claim, expected_sha256=None)
        else:
            current, current_sha = existing
            if current["identity"]["status"] == STATUS_ACCEPTED:
                return {"status": STATUS_ACCEPTED, "ordinal": ordinal, "attempt": current["identity"]["attempt"]}
            if current["identity"]["status"] == STATUS_FAILED:
                attempt = current["identity"]["attempt"] + 1
                claim = build_state(bindings, ordinal=ordinal, unit=unit, attempt=attempt, status=STATUS_CLAIMED, shard_index=shard_index, shard_count=shard_count, device=device, gpu_uuid=gpu_uuid, runtime_provenance=profile)
                validate_state_transition(current, claim)
                claim_sha = _write_state_cas(root_path, claim, expected_sha256=current_sha)
            else:
                if (current["identity"]["shard_index"], current["identity"]["shard_count"], current["identity"]["device"], current["identity"]["gpu_uuid"]) != (shard_index, shard_count, device, gpu_uuid):
                    raise PublicationConflict("successor unit worker mapping differs from durable claim")
                claim, claim_sha, attempt = current, current_sha, current["identity"]["attempt"]
        request_file = request_path(root_path, unit["work_unit_id"], attempt)
        existing_request = _read_regular(request_file, "successor request")
        if existing_request is None:
            if claim["identity"]["status"] != STATUS_CLAIMED:
                raise RecoveryError("successor state claims a request that is absent from immutable history")
            request = build_request(bindings, ordinal=ordinal, profile=profile)
            request_sha = publish_immutable_json(request_file, request, root=root_path)
        else:
            try:
                request = validate_request(json.loads(existing_request), bindings=bindings)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RecoveryError("successor request is malformed") from exc
            request_sha = sha256_bytes(existing_request)
            if claim["identity"]["request_sha256"] is not None and claim["identity"]["request_sha256"] != request_sha:
                raise RecoveryError("successor request digest differs from durable state")
        if claim["identity"]["status"] == STATUS_CLAIMED:
            request_state = build_state(bindings, ordinal=ordinal, unit=unit, attempt=attempt, status=STATUS_REQUEST_PUBLISHED, shard_index=shard_index, shard_count=shard_count, device=device, gpu_uuid=gpu_uuid, request_sha256=request_sha, runtime_provenance=profile)
            validate_state_transition(claim, request_state)
            claim_sha = _write_state_cas(root_path, request_state, expected_sha256=claim_sha)
            claim = request_state
        elif claim["identity"]["status"] not in {STATUS_REQUEST_PUBLISHED, STATUS_RESULT_PUBLISHED}:
            raise RecoveryError(f"cannot execute successor unit from {claim['identity']['status']}")
        result_file = result_path(root_path, unit["work_unit_id"], attempt)
        existing_result = _read_regular(result_file, "successor result")
        if existing_result is None:
            if claim["identity"]["status"] == STATUS_RESULT_PUBLISHED:
                raise RecoveryError("successor result-published state lacks immutable result history")
            measurement = execute_frozen_measurement(request, device=device, batch_size=batch_size)
            result = build_result(request, measurement, attempt=attempt, profile=profile)
            result_sha = publish_immutable_json(result_file, result, root=root_path)
        else:
            result, _ = _read_json_file(result_file, "successor result")
            validate_result(result, bindings=bindings, request=request, attempt=attempt)
            result_sha = sha256_bytes(existing_result)
            if claim["identity"]["result_sha256"] is not None and claim["identity"]["result_sha256"] != result_sha:
                raise RecoveryError("successor result digest differs from durable state")
        result_status = result["status"]
        if result_status == "complete":
            linked = build_state(bindings, ordinal=ordinal, unit=unit, attempt=attempt, status=STATUS_RESULT_PUBLISHED, shard_index=shard_index, shard_count=shard_count, device=device, gpu_uuid=gpu_uuid, request_sha256=request_sha, result_sha256=result_sha, result_status="complete", scientific_execution_performed=True, trials_completed=result["measurement"]["trials_completed"], runtime_provenance=profile)
            if claim["identity"]["status"] == STATUS_REQUEST_PUBLISHED:
                validate_state_transition(claim, linked)
                linked_sha = _write_state_cas(root_path, linked, expected_sha256=claim_sha)
            else:
                if claim["identity"]["status"] != STATUS_RESULT_PUBLISHED:
                    raise RecoveryError("complete successor result is at an unreachable state")
                linked_sha = claim_sha
            accepted = build_state(bindings, ordinal=ordinal, unit=unit, attempt=attempt, status=STATUS_ACCEPTED, shard_index=shard_index, shard_count=shard_count, device=device, gpu_uuid=gpu_uuid, request_sha256=request_sha, result_sha256=result_sha, result_status="complete", scientific_execution_performed=True, trials_completed=result["measurement"]["trials_completed"], runtime_provenance=profile)
            validate_state_transition(linked, accepted)
            final_sha = _write_state_cas(root_path, accepted, expected_sha256=linked_sha)
            return {"status": STATUS_ACCEPTED, "ordinal": ordinal, "attempt": attempt, "request_sha256": request_sha, "result_sha256": result_sha, "state_sha256": final_sha}
        failed = build_state(bindings, ordinal=ordinal, unit=unit, attempt=attempt, status=STATUS_FAILED, shard_index=shard_index, shard_count=shard_count, device=device, gpu_uuid=gpu_uuid, request_sha256=request_sha, result_sha256=result_sha, result_status="failed", scientific_execution_performed=True, trials_completed=result["measurement"]["trials_completed"], reason="frozen PHY execution failed; retryable next attempt", runtime_provenance=profile)
        if claim["identity"]["status"] == STATUS_REQUEST_PUBLISHED:
            validate_state_transition(claim, failed)
            final_sha = _write_state_cas(root_path, failed, expected_sha256=claim_sha)
        else:
            raise RecoveryError("failed successor result is at an unreachable state")
        return {"status": STATUS_FAILED, "ordinal": ordinal, "attempt": attempt, "request_sha256": request_sha, "result_sha256": result_sha, "state_sha256": final_sha}


def _attempts(root: Path, digest: str, kind: str) -> list[int]:
    bucket = root / digest[:2]
    if not bucket.exists():
        return []
    pattern = REQUEST_FILENAME_RE if kind == "request" else RESULT_FILENAME_RE
    values: list[int] = []
    for entry in bucket.iterdir():
        match = pattern.fullmatch(entry.name)
        if match and match.group("digest") == digest:
            values.append(int(match.group("attempt")))
    return sorted(values)


def inspect_unit(root: Path | str, ordinal: int) -> dict[str, Any]:
    root_path = _runtime_root(root)
    bindings = successor_bindings()
    unit = _required_unit_by_ordinal(ordinal)
    digest = unit_digest(bindings["campaign_id"], unit["work_unit_id"])
    request_attempts = _attempts(root_path, digest, "request")
    result_attempts = _attempts(root_path, digest, "result")
    if request_attempts and request_attempts != list(range(1, max(request_attempts) + 1)):
        raise RecoveryError("successor request attempts contain a gap")
    if result_attempts and result_attempts != list(range(1, max(result_attempts) + 1)):
        raise RecoveryError("successor result attempts contain a gap")
    if any(attempt not in request_attempts for attempt in result_attempts):
        raise RecoveryError("successor result has no matching request")
    state_record = read_state(root_path, ordinal, unit["work_unit_id"])
    state = None if state_record is None else state_record[0]
    validated_requests: dict[int, dict[str, Any]] = {}
    validated_results: dict[int, dict[str, Any]] = {}
    for attempt in request_attempts:
        request, raw = _read_json_file(request_path(root_path, unit["work_unit_id"], attempt), "successor request")
        try:
            validated_request = validate_request(request, bindings=bindings)
        except ProductionContractError as current_error:
            try:
                if attempt != 1:
                    raise ProductionContractError("pre-measurement compatibility is limited to attempt 1")
                record = _repair_unit_record(ordinal)
                if sha256_bytes(raw) != record["request_file_sha256"]:
                    raise ProductionContractError("pre-measurement predecessor request bytes differ")
                validated_request = validate_request(request, bindings=_legacy_bindings())
            except ProductionContractError as legacy_error:
                raise RecoveryError("successor request is not bound to the current or exact repair epoch") from legacy_error
        validated_requests[attempt] = {"request": validated_request, "raw": raw}
    for attempt in result_attempts:
        request_record = validated_requests.get(attempt)
        if request_record is None:
            raise RecoveryError("successor result has no validated request history")
        request = request_record["request"]
        request_raw = request_record["raw"]
        result, raw = _read_json_file(result_path(root_path, unit["work_unit_id"], attempt), "successor result")
        try:
            validate_result(result, bindings=bindings, request=request, attempt=attempt)
        except ProductionContractError as current_error:
            try:
                _legacy_attempt_evidence(ordinal, attempt, request, request_raw, result, raw)
            except ProductionContractError as legacy_error:
                raise RecoveryError("successor result is not bound to the current or exact repair epoch") from legacy_error
        validated_results[attempt] = {"result": result, "sha256": sha256_bytes(raw)}
    if state is not None:
        if state["identity"]["status"] == STATUS_ACCEPTED:
            attempt = state["identity"]["attempt"]
            if attempt not in validated_results or validated_results[attempt]["result"]["status"] != "complete":
                raise RecoveryError("accepted successor state lacks its complete result")
        if state["identity"]["attempt"] not in request_attempts and request_attempts:
            raise RecoveryError("successor state attempt is absent from request history")
        state_identity = state["identity"]
        state_attempt = state_identity["attempt"]
        if state_identity["request_sha256"] is not None:
            request_raw = _read_regular(request_path(root_path, unit["work_unit_id"], state_attempt), "successor request")
            if request_raw is None or sha256_bytes(request_raw) != state_identity["request_sha256"]:
                raise RecoveryError("successor state request digest does not match immutable history")
        if state_identity["result_sha256"] is not None:
            result_record = validated_results.get(state_attempt)
            if result_record is None or result_record["sha256"] != state_identity["result_sha256"]:
                raise RecoveryError("successor state result digest does not match immutable history")
            if state_identity["status"] in {STATUS_RESULT_PUBLISHED, STATUS_ACCEPTED} and result_record["result"]["status"] != "complete":
                raise RecoveryError("successor complete-result state points at a failed result")
            if state_identity["status"] == STATUS_FAILED and result_record["result"]["status"] != "failed":
                raise RecoveryError("successor failed state points at a complete result")
    classification = "available"
    if state is not None:
        classification = state["identity"]["status"]
    elif result_attempts:
        classification = STATUS_RESULT_PUBLISHED if validated_results[max(result_attempts)]["result"]["status"] == "complete" else STATUS_FAILED
    elif request_attempts:
        classification = STATUS_REQUEST_PUBLISHED
    return {
        "ordinal": ordinal,
        "work_unit_id": unit["work_unit_id"],
        "classification": classification,
        "request_attempts": request_attempts,
        "result_attempts": result_attempts,
        "state": state,
        "validated_results": validated_results,
        "accepted": classification == STATUS_ACCEPTED,
        "complete_evidence": any(item["result"]["status"] == "complete" for item in validated_results.values()),
        "retryable": classification == STATUS_FAILED,
        "test_access": 0,
    }


def _initial_campaign_state(bindings: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "artifact_role": "g8_c_pascal_successor_campaign_state",
        "campaign_id": bindings["campaign_id"],
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "runner_contract_sha256": bindings["runner_contract_sha256"],
        "production_contract_sha256": bindings["production_contract_sha256"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "accepted_authority_ordinals": [],
        "in_progress_authority_ordinals": [],
        "failed_authority_ordinals": [],
        "terminal_invalid_authority_ordinals": [],
        "protected_counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        "old_result_ingest_permitted": False,
        "scientific_execution_performed": False,
        "test_access": 0,
        "state_sha256": None,
    }
    body["state_sha256"] = digest_without_field(body, "state_sha256")
    return body


def validate_campaign_state(state: Mapping[str, Any], *, bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = _canonical(state, "successor campaign state")
    _same_keys(payload, ("schema_version", "artifact_role", "campaign_id", "campaign_manifest_sha256", "source_manifest_sha256", "runner_contract_sha256", "production_contract_sha256", "execution_profile_id", "required_identity_count", "trials_per_identity", "accepted_authority_ordinals", "in_progress_authority_ordinals", "failed_authority_ordinals", "terminal_invalid_authority_ordinals", "protected_counters", "old_result_ingest_permitted", "scientific_execution_performed", "test_access", "state_sha256"), "successor campaign state")
    expected = successor_bindings() if bindings is None else bindings
    if payload["schema_version"] != PRODUCTION_SCHEMA_VERSION or payload["artifact_role"] != "g8_c_pascal_successor_campaign_state":
        raise ProductionContractError("unsupported successor campaign state")
    for key in ("campaign_id", "campaign_manifest_sha256", "source_manifest_sha256", "runner_contract_sha256", "production_contract_sha256", "execution_profile_id"):
        if payload[key] != expected[key]:
            raise ProductionContractError(f"successor campaign state binding differs: {key}")
    if payload["required_identity_count"] != REQUIRED_COUNT or payload["trials_per_identity"] != TRIALS_PER_IDENTITY:
        raise ProductionContractError("successor campaign state grid differs")
    lists = []
    for key in ("accepted_authority_ordinals", "in_progress_authority_ordinals", "failed_authority_ordinals", "terminal_invalid_authority_ordinals"):
        values = payload[key]
        if not isinstance(values, list) or any(type(value) is not int or not 0 <= value < REQUIRED_COUNT for value in values) or values != sorted(set(values)):
            raise ProductionContractError(f"successor campaign state {key} is malformed")
        lists.append(set(values))
    if any(lists[i] & lists[j] for i in range(len(lists)) for j in range(i)):
        raise ProductionContractError("successor campaign state classifications overlap")
    evidence_present = any(lists)
    if payload["scientific_execution_performed"] is not evidence_present:
        raise ProductionContractError("successor campaign scientific flag does not match durable classifications")
    if payload["protected_counters"] != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0} or payload["test_access"] != 0 or payload["old_result_ingest_permitted"] is not False:
        raise ProductionContractError("successor campaign protected counters are unsafe")
    if type(payload["scientific_execution_performed"]) is not bool:
        raise ProductionContractError("successor campaign scientific flag is malformed")
    _digest(payload["state_sha256"], "campaign state_sha256")
    if payload["state_sha256"] != digest_without_field(payload, "state_sha256"):
        raise ProductionContractError("successor campaign state digest does not reproduce")
    return payload


def _write_campaign_state(root: Path, state: Mapping[str, Any], *, expected_sha256: str | None) -> str:
    path = root / PRODUCTION_STATE_FILENAME
    body = canonical_json(validate_campaign_state(state))
    current = _read_regular(path, "successor campaign state")
    if current is None:
        if expected_sha256 is not None:
            raise StaleStateError("successor campaign state disappeared")
        return _stage_and_publish(path, body, replace=False)
    if sha256_bytes(current) != expected_sha256:
        raise StaleStateError("successor campaign state predecessor differs")
    if current == body:
        return sha256_bytes(body)
    return _stage_and_publish(path, body, replace=True)


def _validate_runtime_campaign_state(raw: bytes, *, bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate current aggregate state or the one recorded predecessor snapshot."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryError("successor campaign state is malformed") from exc
    if not isinstance(payload, Mapping) or canonical_json(payload) != raw:
        raise RecoveryError("successor campaign state is not canonical")
    try:
        return validate_campaign_state(payload, bindings=bindings)
    except ProductionContractError as current_error:
        policy = _load_pre_measurement_repair_policy()
        snapshot = policy["state_snapshot"]
        if sha256_bytes(raw) != snapshot["campaign_state_file_sha256"]:
            raise RecoveryError("successor campaign state is not bound to the current or exact repair epoch") from current_error
        try:
            legacy = validate_campaign_state(payload, bindings=_legacy_bindings())
        except ProductionContractError as legacy_error:
            raise RecoveryError("pre-measurement predecessor campaign state is invalid") from legacy_error
        if any(legacy[key] != snapshot[key] for key in (
            "accepted_authority_ordinals", "failed_authority_ordinals", "in_progress_authority_ordinals",
            "terminal_invalid_authority_ordinals", "scientific_execution_performed",
        )):
            raise RecoveryError("pre-measurement predecessor campaign state snapshot differs")
        return legacy


def _repair_complete_unit(root: Path, ordinal: int, report: Mapping[str, Any]) -> None:
    """Finish a durable request/result transaction without re-running PHY code."""

    complete_attempts = [
        attempt
        for attempt, item in report["validated_results"].items()
        if item["result"]["status"] == "complete"
    ]
    if len(complete_attempts) != 1:
        raise RecoveryError("successor unit has zero or multiple complete result attempts")
    attempt = complete_attempts[0]
    unit = _required_unit_by_ordinal(ordinal)
    bindings = successor_bindings()
    request, request_raw = _read_json_file(request_path(root, unit["work_unit_id"], attempt), "successor request")
    result, result_raw = _read_json_file(result_path(root, unit["work_unit_id"], attempt), "successor result")
    validate_request(request, bindings=bindings)
    validate_result(result, bindings=bindings, request=request, attempt=attempt)
    profile = request["identity"]["profile_provenance"]
    digest = unit_digest(bindings["campaign_id"], unit["work_unit_id"])
    with unit_lock(root, digest):
        current = read_state(root, ordinal, unit["work_unit_id"])
        if current is None:
            claim = build_state(
                bindings,
                ordinal=ordinal,
                unit=unit,
                attempt=attempt,
                status=STATUS_CLAIMED,
                shard_index=authority_shard(ordinal),
                shard_count=2,
                device=profile["device"],
                gpu_uuid=profile["gpu_uuid"],
                runtime_provenance=profile,
            )
            state_sha = _write_state_cas(root, claim, expected_sha256=None)
            current = (claim, state_sha)
        state, state_sha = current
        state_identity = state["identity"]
        if state_identity["attempt"] != attempt:
            raise RecoveryError("complete successor result belongs to a non-current attempt")
        if state_identity["status"] == STATUS_ACCEPTED:
            return
        if state_identity["status"] == STATUS_TERMINAL_INVALID:
            raise RecoveryError("terminal-invalid successor unit has complete result evidence")
        request_sha = sha256_bytes(request_raw)
        result_sha = sha256_bytes(result_raw)
        if state_identity["status"] == STATUS_CLAIMED:
            request_state = build_state(
                bindings,
                ordinal=ordinal,
                unit=unit,
                attempt=attempt,
                status=STATUS_REQUEST_PUBLISHED,
                shard_index=state_identity["shard_index"],
                shard_count=state_identity["shard_count"],
                device=state_identity["device"],
                gpu_uuid=state_identity["gpu_uuid"],
                request_sha256=request_sha,
                runtime_provenance=profile,
            )
            validate_state_transition(state, request_state)
            state_sha = _write_state_cas(root, request_state, expected_sha256=state_sha)
            state = request_state
        elif state_identity["status"] == STATUS_REQUEST_PUBLISHED:
            if state_identity["request_sha256"] != request_sha:
                raise RecoveryError("successor request digest differs from durable state")
        elif state_identity["status"] == STATUS_RESULT_PUBLISHED:
            if state_identity["request_sha256"] != request_sha or state_identity["result_sha256"] != result_sha:
                raise RecoveryError("successor result binding differs from durable state")
        else:
            raise RecoveryError(f"cannot reconcile complete result from {state_identity['status']}")
        linked = build_state(
            bindings,
            ordinal=ordinal,
            unit=unit,
            attempt=attempt,
            status=STATUS_RESULT_PUBLISHED,
            shard_index=state["identity"]["shard_index"],
            shard_count=state["identity"]["shard_count"],
            device=state["identity"]["device"],
            gpu_uuid=state["identity"]["gpu_uuid"],
            request_sha256=request_sha,
            result_sha256=result_sha,
            result_status="complete",
            scientific_execution_performed=True,
            trials_completed=result["measurement"]["trials_completed"],
            runtime_provenance=profile,
        )
        if state["identity"]["status"] != STATUS_RESULT_PUBLISHED:
            validate_state_transition(state, linked)
            state_sha = _write_state_cas(root, linked, expected_sha256=state_sha)
        accepted = build_state(
            bindings,
            ordinal=ordinal,
            unit=unit,
            attempt=attempt,
            status=STATUS_ACCEPTED,
            shard_index=linked["identity"]["shard_index"],
            shard_count=linked["identity"]["shard_count"],
            device=linked["identity"]["device"],
            gpu_uuid=linked["identity"]["gpu_uuid"],
            request_sha256=request_sha,
            result_sha256=result_sha,
            result_status="complete",
            scientific_execution_performed=True,
            trials_completed=result["measurement"]["trials_completed"],
            runtime_provenance=profile,
        )
        if linked["identity"]["status"] != STATUS_ACCEPTED:
            validate_state_transition(linked, accepted)
            _write_state_cas(root, accepted, expected_sha256=state_sha)


def reconcile_campaign(root: Path | str) -> dict[str, Any]:
    """Rebuild exact accepted coverage from successor evidence under a lock."""

    root_path = ensure_runtime_root(root)
    bindings = successor_bindings()
    with campaign_lock(root_path):
        current_raw = _read_regular(root_path / PRODUCTION_STATE_FILENAME, "successor campaign state")
        if current_raw is None:
            current = _initial_campaign_state(bindings)
            current_sha = None
        else:
            current = _validate_runtime_campaign_state(current_raw, bindings=bindings)
            current_sha = sha256_bytes(current_raw)
        reports = [inspect_unit(root_path, ordinal) for ordinal in range(REQUIRED_COUNT)]
        for report in reports:
            if report["complete_evidence"] and not report["accepted"]:
                _repair_complete_unit(root_path, report["ordinal"], report)
        reports = [inspect_unit(root_path, ordinal) for ordinal in range(REQUIRED_COUNT)]
        accepted = [report["ordinal"] for report in reports if report["accepted"]]
        failed = [report["ordinal"] for report in reports if report["retryable"]]
        terminal_invalid = [report["ordinal"] for report in reports if report["classification"] == STATUS_TERMINAL_INVALID]
        in_progress = [report["ordinal"] for report in reports if report["classification"] in {STATUS_CLAIMED, STATUS_REQUEST_PUBLISHED, STATUS_RESULT_PUBLISHED}]
        candidate = dict(current)
        candidate.update({
            "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
            "source_manifest_sha256": bindings["source_manifest_sha256"],
            "runner_contract_sha256": bindings["runner_contract_sha256"],
            "production_contract_sha256": bindings["production_contract_sha256"],
            "execution_profile_id": bindings["execution_profile_id"],
            "accepted_authority_ordinals": accepted,
            "in_progress_authority_ordinals": in_progress,
            "failed_authority_ordinals": failed,
            "terminal_invalid_authority_ordinals": terminal_invalid,
            "scientific_execution_performed": bool(accepted or failed or in_progress or terminal_invalid),
        })
        candidate["state_sha256"] = digest_without_field(candidate, "state_sha256")
        validate_campaign_state(candidate, bindings=bindings)
        installed_sha = _write_campaign_state(root_path, candidate, expected_sha256=current_sha)
        return {
            "status": "PASS",
            "campaign_id": bindings["campaign_id"],
            "accepted_authority_ordinals": accepted,
            "accepted_count": len(accepted),
            "required_identity_count": REQUIRED_COUNT,
            "in_progress_authority_ordinals": in_progress,
            "failed_authority_ordinals": failed,
            "terminal_invalid_authority_ordinals": terminal_invalid,
            "installed_state_sha256": installed_sha,
            "test_access": 0,
            "old_result_ingest": False,
        }


def _runtime_entries(root: Path) -> Iterator[Path]:
    try:
        yield from root.iterdir()
    except OSError as exc:
        raise RecoveryError(f"cannot enumerate successor runtime root: {exc}") from exc


def validate_runtime_namespace(root: Path | str) -> None:
    """Reject aliases, foreign campaign files and malformed partial names."""

    root_path = _runtime_root(root)
    if not root_path.exists():
        return
    bindings = successor_bindings()
    known_digests = {
        unit_digest(bindings["campaign_id"], _required_unit_by_ordinal(ordinal)["work_unit_id"])
        for ordinal in range(REQUIRED_COUNT)
    }
    for entry in _runtime_entries(root_path):
        if entry.name == PRODUCTION_STATE_FILENAME or entry.name == ".campaign.lock":
            if entry.is_symlink() or not entry.is_file():
                raise RecoveryError(f"successor runtime control entry is not a regular file: {entry}")
            continue
        if entry.name == ".locks":
            if entry.is_symlink() or not entry.is_dir():
                raise RecoveryError("successor runtime lock directory is not real")
            for lock in entry.iterdir():
                if lock.is_symlink() or not lock.is_file() or lock.name != f"{lock.stem}.lock" or lock.stem not in known_digests:
                    raise RecoveryError(f"foreign or malformed successor lock entry: {lock}")
            continue
        if BUCKET_RE.fullmatch(entry.name) is None or entry.is_symlink() or not entry.is_dir():
            raise RecoveryError(f"foreign or malformed successor runtime entry: {entry}")
        for artifact in entry.iterdir():
            if artifact.is_symlink() or not artifact.is_file():
                raise RecoveryError(f"successor runtime artifact is an alias: {artifact}")
            match = STATE_FILENAME_RE.fullmatch(artifact.name)
            if match:
                if match.group("digest") not in known_digests or entry.name != match.group("digest")[:2]:
                    raise RecoveryError(f"foreign successor state artifact: {artifact}")
                continue
            match = REQUEST_FILENAME_RE.fullmatch(artifact.name) or RESULT_FILENAME_RE.fullmatch(artifact.name)
            if match is None or match.group("digest") not in known_digests or entry.name != match.group("digest")[:2]:
                raise RecoveryError(f"foreign or malformed successor artifact: {artifact}")


def audit_campaign(root: Path | str) -> dict[str, Any]:
    """Audit durable successor evidence without changing it."""

    root_path = _runtime_root(root)
    bindings = successor_bindings()
    if not root_path.exists():
        return {
            "status": "PASS",
            "campaign_id": bindings["campaign_id"],
            "accepted_authority_ordinals": [],
            "accepted_count": 0,
            "required_identity_count": REQUIRED_COUNT,
            "terminal_invalid_authority_ordinals": [],
            "runtime_exists": False,
            "test_access": 0,
        }
    validate_runtime_namespace(root_path)
    reports = [inspect_unit(root_path, ordinal) for ordinal in range(REQUIRED_COUNT)]
    accepted = [report["ordinal"] for report in reports if report["accepted"]]
    failed = [report["ordinal"] for report in reports if report["retryable"]]
    in_progress = [report["ordinal"] for report in reports if report["classification"] in {STATUS_CLAIMED, STATUS_REQUEST_PUBLISHED, STATUS_RESULT_PUBLISHED}]
    terminal_invalid = [report["ordinal"] for report in reports if report["classification"] == STATUS_TERMINAL_INVALID]
    state_raw = _read_regular(root_path / PRODUCTION_STATE_FILENAME, "successor campaign state")
    if state_raw is None:
        if accepted or failed or in_progress or terminal_invalid:
            raise RecoveryError("successor evidence exists without a durable campaign state")
    else:
        state = _validate_runtime_campaign_state(state_raw, bindings=bindings)
        expected = {
            "accepted_authority_ordinals": accepted,
            "failed_authority_ordinals": failed,
            "in_progress_authority_ordinals": in_progress,
            "terminal_invalid_authority_ordinals": terminal_invalid,
        }
        if any(state[key] != value for key, value in expected.items()):
            raise RecoveryError("successor campaign state is stale relative to durable evidence")
    return {
        "status": "PASS",
        "campaign_id": bindings["campaign_id"],
        "accepted_authority_ordinals": accepted,
        "accepted_count": len(accepted),
        "required_identity_count": REQUIRED_COUNT,
        "in_progress_authority_ordinals": in_progress,
        "failed_authority_ordinals": failed,
        "terminal_invalid_authority_ordinals": terminal_invalid,
        "runtime_exists": True,
        "test_access": 0,
        "old_result_ingest": False,
    }


def initial_campaign_state(bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _initial_campaign_state(successor_bindings() if bindings is None else bindings)


def authenticate_worker_profile(*, device: str, expected_gpu_uuid: str, config_hash: str) -> dict[str, Any]:
    """Authenticate exact profile facts immediately before a worker starts."""

    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ProductionContractError("successor worker device must be explicit cuda:N")
    try:
        environment = authenticate_execution_profile(
            SUCCESSOR_PROFILE_ID,
            device=device,
            config_hash=config_hash,
            require_openjpeg=False,
        )
    except Exception as exc:
        raise ProductionContractError(f"successor execution profile authentication failed: {exc}") from exc
    if environment.get("gpu_uuid") != expected_gpu_uuid:
        raise ProductionContractError("successor worker GPU UUID does not match coordinator mapping")
    if environment.get("driver_version") in {None, ""}:
        raise ProductionContractError("NVIDIA driver_version is required for new scientific provenance")
    return {**environment, "device": device}


def exact_shard_partition() -> dict[int, list[int]]:
    partitions = {0: [], 1: []}
    for ordinal in range(REQUIRED_COUNT):
        partitions[authority_shard(ordinal)].append(ordinal)
    if set(partitions[0]) & set(partitions[1]) or set(partitions[0]) | set(partitions[1]) != set(range(REQUIRED_COUNT)):
        raise ProductionContractError("successor two-shard partition has overlap or omission")
    return partitions


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProductionContractError(f"cannot hash successor contract file {path}: {exc}") from exc


def _validate_custody_policy(value: Any) -> None:
    if value != PASCAL_SUCCESSOR_CUSTODY_POLICY:
        raise ProductionContractError("successor production evidence custody policy differs")


def _leaf_difference_paths(old: Any, new: Any, prefix: str = "") -> set[str]:
    """Return exact dotted YAML leaf paths changed by an additive amendment."""

    if isinstance(old, Mapping) and isinstance(new, Mapping):
        result: set[str] = set()
        for key in set(old) | set(new):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.add(child)
            else:
                result.update(_leaf_difference_paths(old[key], new[key], child))
        return result
    return set() if old == new else {prefix}


def _load_post_campaign_source_compatibility(
    source_entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Authenticate AM-87's exact, off-measurement-path source compatibility.

    Completed G8_C evidence remains bound to its original source bytes.  This
    additive record permits only the current verifier implementation itself,
    the historical-campaign compatibility verifier and generated parameters
    whose exact semantic diff is the new G8_F corpus rule. It is not a general
    source-drift allowlist.
    """

    try:
        raw = POST_CAMPAIGN_SOURCE_COMPATIBILITY.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionContractError(f"cannot load AM-87 source compatibility: {exc}") from None
    required = {
        "schema_version", "artifact_role", "amendment", "discovery_date", "timing",
        "classification", "measurement_source_commit", "allowed_parameter_paths",
        "entries", "protected_boundary", "compatibility_id",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ProductionContractError("AM-87 source-compatibility schema differs")
    if (
        value["schema_version"] != 1
        or value["artifact_role"] != "g8_c_g8_e_am87_post_campaign_source_compatibility"
        or value["amendment"] != "AM-87"
        or value["discovery_date"] != "2026-08-23"
        or value["timing"] != "post_g8c_post_g8e_pass_one_pre_g8f_execution"
        or value["classification"] != "off_measurement_path_protocol_parameters_and_post_campaign_verifier"
    ):
        raise ProductionContractError("AM-87 source-compatibility header differs")
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    if value["compatibility_id"] != "g8postsource-" + sha256_bytes(canonical_json(body)):
        raise ProductionContractError("AM-87 source-compatibility ID differs")
    boundary = value["protected_boundary"]
    if boundary != {
        "g8_c_changed": False,
        "g8_e_changed": False,
        "g8_f_execution": 0,
        "pass_one_rerun": False,
        "pass_two": 0,
        "test_access": 0,
        "training": 0,
    }:
        raise ProductionContractError("AM-87 source-compatibility boundary differs")
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != 3:  # literal-ok: exact params/two-verifier compatibility set
        raise ProductionContractError("AM-87 source-compatibility entries differ")
    by_path: dict[str, dict[str, Any]] = {}
    frozen_by_path = {str(entry.get("path")): entry for entry in source_entries}
    for item in entries:
        fields = {
            "path", "kind", "archived_bytes", "archived_sha256", "current_bytes",
            "current_sha256", "measurement_path_reachable", "justification",
        }
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ProductionContractError("AM-87 source-compatibility entry schema differs")
        path_text = item["path"]
        if path_text in by_path or path_text not in frozen_by_path:
            raise ProductionContractError("AM-87 source-compatibility path is duplicate or foreign")
        frozen = frozen_by_path[path_text]
        if item["archived_bytes"] != frozen.get("bytes") or item["archived_sha256"] != frozen.get("sha256"):
            raise ProductionContractError("AM-87 compatibility does not bind frozen source entry")
        path = REPO_ROOT / path_text
        if (
            not path.is_file()
            or item["current_bytes"] != path.stat().st_size
            or item["current_sha256"] != _file_sha256(path)
        ):
            raise ProductionContractError("AM-87 compatibility current bytes differ")
        if item["measurement_path_reachable"] is not False or not isinstance(item["justification"], str) or not item["justification"]:
            raise ProductionContractError("AM-87 compatibility reachability/justification differs")
        by_path[path_text] = dict(item)
    if set(by_path) != {
        "spec/params.generated.yaml",
        "src/baseline/g8_campaign.py",
        "src/baseline/g8_pascal_production.py",
    }:
        raise ProductionContractError("AM-87 compatibility permits an unexpected source path")
    if by_path["spec/params.generated.yaml"]["kind"] != "am87_g8f_only_generated_parameter_change":
        raise ProductionContractError("AM-87 parameter compatibility kind differs")
    if by_path["src/baseline/g8_pascal_production.py"]["kind"] != "post_campaign_verifier_compatibility_only":
        raise ProductionContractError("AM-87 verifier compatibility kind differs")
    if by_path["src/baseline/g8_campaign.py"]["kind"] != "historical_campaign_verifier_compatibility_only":
        raise ProductionContractError("AM-87 historical-campaign compatibility kind differs")

    commit = value["measurement_source_commit"]
    if commit != "426110b05161e73e4d819bdc01f4857c012d6d59":
        raise ProductionContractError("AM-87 compatibility measurement source commit differs")
    try:
        historical_params = subprocess.run(
            ["git", "show", f"{commit}:spec/params.generated.yaml"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=15,  # literal-ok: bounded local historical-byte query
        ).stdout
        old = yaml.safe_load(historical_params)
        new = yaml.safe_load((REPO_ROOT / "spec/params.generated.yaml").read_bytes())
    except (OSError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        raise ProductionContractError(f"cannot reproduce AM-87 parameter diff: {exc}") from None
    params_entry = by_path["spec/params.generated.yaml"]
    if len(historical_params) != params_entry["archived_bytes"] or sha256_bytes(historical_params) != params_entry["archived_sha256"]:
        raise ProductionContractError("AM-87 historical parameter bytes differ")
    allowed_paths = value["allowed_parameter_paths"]
    if not isinstance(allowed_paths, list) or allowed_paths != sorted(set(allowed_paths)):
        raise ProductionContractError("AM-87 allowed parameter paths are not sorted unique")
    if _leaf_difference_paths(old, new) != set(allowed_paths):
        raise ProductionContractError("current parameter drift exceeds exact AM-87 G8_F paths")
    if not all(path.startswith("reference_classifier.artifact_finetune_") for path in allowed_paths):
        raise ProductionContractError("AM-87 compatibility reaches outside G8_F corpus parameters")
    return by_path


def _load_am88_post_campaign_source_compatibility(
    source_entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Authenticate the exact AM-87 → AM-88 off-measurement-path chain."""

    try:
        am87_raw = POST_CAMPAIGN_SOURCE_COMPATIBILITY.read_bytes()
        am87 = json.loads(am87_raw)
        raw = AM88_POST_CAMPAIGN_SOURCE_COMPATIBILITY.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionContractError(f"cannot load AM-88 source compatibility: {exc}") from None
    am87_body = {key: child for key, child in am87.items() if key != "compatibility_id"}
    if am87.get("compatibility_id") != "g8postsource-" + sha256_bytes(canonical_json(am87_body)):
        raise ProductionContractError("AM-87 source-compatibility identity differs in AM-88 chain")
    required = {
        "schema_version", "artifact_role", "amendment", "discovery_date", "timing",
        "classification", "prior_compatibility", "allowed_parameter_paths", "entries",
        "protected_boundary", "compatibility_id",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ProductionContractError("AM-88 source-compatibility schema differs")
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    if value["compatibility_id"] != "g8postsource-" + sha256_bytes(canonical_json(body)):
        raise ProductionContractError("AM-88 source-compatibility ID differs")
    if (
        value["schema_version"] != 1
        or value["artifact_role"] != "g8_c_g8_e_am88_post_campaign_source_compatibility"
        or value["amendment"] != "AM-88"
        or value["discovery_date"] != "2026-08-24"
        or value["timing"] != "post_am87_pre_f0_execution_zero"
        or value["classification"] != "off_measurement_path_sampler_protocol_and_post_campaign_verifiers"
        or value["protected_boundary"] != {
            "g8_c_changed": False, "g8_d_changed": False, "g8_e_changed": False,
            "g8_f_execution": 0, "pass_one_rerun": False, "pass_two": 0,
            "test_access": 0, "training": 0,
        }
    ):
        raise ProductionContractError("AM-88 source-compatibility header/boundary differs")
    if value["prior_compatibility"] != {
        "path": str(POST_CAMPAIGN_SOURCE_COMPATIBILITY.relative_to(REPO_ROOT)),
        "compatibility_id": am87["compatibility_id"],
        "sha256": sha256_bytes(am87_raw),
    }:
        raise ProductionContractError("AM-88 prior compatibility binding differs")
    frozen = {str(entry.get("path")): entry for entry in source_entries}
    am87_entries = {str(entry.get("path")): entry for entry in am87.get("entries", []) if isinstance(entry, Mapping)}
    expected_paths = {
        "spec/params.generated.yaml", "src/baseline/g8_campaign.py",
        "src/baseline/g8_pascal_production.py",
    }
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected_paths):
        raise ProductionContractError("AM-88 source-compatibility entries differ")
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        fields = {
            "path", "kind", "archived_bytes", "archived_sha256", "current_bytes",
            "current_sha256", "measurement_path_reachable", "justification",
        }
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ProductionContractError("AM-88 source-compatibility entry schema differs")
        path_text = str(item["path"])
        prior = am87_entries.get(path_text)
        original = frozen.get(path_text)
        path = REPO_ROOT / path_text
        if path_text in result or path_text not in expected_paths or prior is None or original is None:
            raise ProductionContractError("AM-88 source-compatibility path is duplicate or foreign")
        if prior.get("archived_sha256") != original.get("sha256") or prior.get("archived_bytes") != original.get("bytes"):
            raise ProductionContractError("AM-87 frozen source chain differs under AM-88")
        archived = subprocess.run(
            ["git", "show", f"{AM87_FINAL_COMMIT}:{path_text}"], cwd=REPO_ROOT,
            check=False, capture_output=True, timeout=15,  # literal-ok: bounded local historical-byte query
        )
        if (
            archived.returncode != 0
            or len(archived.stdout) != item["archived_bytes"]
            or sha256_bytes(archived.stdout) != item["archived_sha256"]
            or item["archived_bytes"] != prior.get("current_bytes")
            or item["archived_sha256"] != prior.get("current_sha256")
            or not path.is_file()
            or item["current_bytes"] != path.stat().st_size
            or item["current_sha256"] != _file_sha256(path)
            or item["measurement_path_reachable"] is not False
            or not isinstance(item["justification"], str) or not item["justification"]
        ):
            raise ProductionContractError("AM-88 source compatibility byte/reachability chain differs")
        result[path_text] = dict(item)
    if set(result) != expected_paths:
        raise ProductionContractError("AM-88 source compatibility is incomplete")
    previous_params = subprocess.run(
        ["git", "show", f"{AM87_FINAL_COMMIT}:spec/params.generated.yaml"], cwd=REPO_ROOT,
        check=True, capture_output=True, timeout=15,  # literal-ok: bounded local historical-byte query
    ).stdout
    try:
        old = yaml.safe_load(previous_params)
        new = yaml.safe_load((REPO_ROOT / "spec/params.generated.yaml").read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise ProductionContractError(f"cannot reproduce AM-88 parameter diff: {exc}") from None
    allowed = value["allowed_parameter_paths"]
    if not isinstance(allowed, list) or allowed != sorted(set(allowed)):
        raise ProductionContractError("AM-88 allowed parameter paths differ")
    if _leaf_difference_paths(old, new) != set(allowed):
        raise ProductionContractError("current parameter drift exceeds exact AM-88 G8_F paths")
    if not all(path.startswith("reference_classifier.artifact_finetune_") for path in allowed):
        raise ProductionContractError("AM-88 compatibility reaches outside G8_F sampler parameters")
    return result


def validate_production_contracts() -> dict[str, Any]:
    """Validate the complete additive contract family and live source closure."""

    try:
        manifest = load_json(SUCCESSOR_MANIFEST)
        readiness_state = load_json(SUCCESSOR_STATE)
        source = load_json(PRODUCTION_SOURCE_MANIFEST)
        runner = load_json(PRODUCTION_RUNNER_CONTRACT)
        coordinator = load_json(PRODUCTION_COORDINATOR_CONTRACT)
        contract = load_json(PRODUCTION_CONTRACT)
        repair_policy = _load_pre_measurement_repair_policy()
    except Exception as exc:
        if isinstance(exc, ProductionContractError):
            raise
        raise ProductionContractError(f"successor production contract cannot be loaded: {exc}") from exc

    campaign_id = manifest.get("campaign_id")
    if campaign_id != successor_campaign_identifier(manifest):
        raise ProductionContractError("successor production contract campaign identity differs")
    if source.get("campaign_id") != campaign_id or runner.get("campaign_id") != campaign_id or coordinator.get("campaign_id") != campaign_id or contract.get("campaign_id") != campaign_id:
        raise ProductionContractError("successor production contracts are not campaign-bound")
    if source.get("execution_profile_id") != SUCCESSOR_PROFILE_ID or runner.get("execution_profile_id") != SUCCESSOR_PROFILE_ID or coordinator.get("execution_profile_id") != SUCCESSOR_PROFILE_ID or contract.get("execution_profile_id") != SUCCESSOR_PROFILE_ID:
        raise ProductionContractError("successor production contracts are not profile-bound")
    if manifest.get("source_manifest_sha256") != _file_sha256(SUCCESSOR_SOURCE_MANIFEST):
        raise ProductionContractError("zero-coverage source marker no longer matches the campaign")
    if readiness_state.get("campaign_id") != campaign_id:
        raise ProductionContractError("zero-coverage readiness state is not campaign-bound")
    try:
        validate_successor_manifest(manifest)
        validate_successor_state(readiness_state)
    except Exception as exc:
        raise ProductionContractError(f"zero-coverage successor readiness marker is invalid: {exc}") from exc
    campaign_sha = _file_sha256(SUCCESSOR_MANIFEST)
    readiness_sha = _file_sha256(SUCCESSOR_STATE)
    repair_policy_sha = _file_sha256(PRE_MEASUREMENT_REPAIR_POLICY)

    if source.get("artifact_role") != "g8_c_pascal_successor_production_source_manifest":
        raise ProductionContractError("unsupported successor production source manifest")
    if source.get("lock_file") != "requirements-pascal.lock" or source.get("lock_file_sha256") != _file_sha256(REPO_ROOT / "requirements-pascal.lock"):
        raise ProductionContractError("successor production source lock binding differs")
    required_path = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
    if source.get("required_bler_artifact") != str(required_path.relative_to(REPO_ROOT)) or source.get("required_bler_artifact_sha256") != _file_sha256(required_path):
        raise ProductionContractError("successor production required-identity binding differs")
    source_entries = source.get("sources")
    if not isinstance(source_entries, list) or not source_entries:
        raise ProductionContractError("successor production source closure is empty")
    compatibility: dict[str, dict[str, Any]] | None = None
    source_paths: list[str] = []
    for entry in source_entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "bytes", "sha256"}:
            raise ProductionContractError("successor production source entry schema differs")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ProductionContractError("successor production source path is unsafe")
        path = REPO_ROOT / relative
        exact = path.is_file() and entry["bytes"] == path.stat().st_size and entry["sha256"] == _file_sha256(path)
        if not exact:
            if compatibility is None:
                compatibility = _load_am88_post_campaign_source_compatibility(source_entries)
            if relative not in compatibility:
                raise ProductionContractError(f"successor production source bytes differ: {relative}")
        source_paths.append(relative)
    if len(source_paths) != len(set(source_paths)):
        raise ProductionContractError("successor production source closure contains duplicates")
    if tuple(source_paths) != REQUIRED_PRODUCTION_SOURCE_PATHS:
        raise ProductionContractError("successor production source closure is incomplete or reordered")
    production_source_sha = _file_sha256(PRODUCTION_SOURCE_MANIFEST)
    if source.get("campaign_manifest_sha256") != campaign_sha or source.get("readiness_state_sha256") != readiness_sha:
        raise ProductionContractError("successor production source readiness binding differs")
    if source.get("coordinator_contract_sha256") != _file_sha256(PRODUCTION_COORDINATOR_CONTRACT):
        raise ProductionContractError("successor production source coordinator binding differs")
    if source.get("pre_measurement_repair_policy_sha256") != repair_policy_sha:
        raise ProductionContractError("successor production source repair policy binding differs")

    expected_workers = [dict(worker) for worker in PRODUCTION_WORKERS]
    if coordinator.get("artifact_role") != "g8_c_pascal_successor_production_coordinator_contract" or coordinator.get("workers") != expected_workers:
        raise ProductionContractError("successor production coordinator topology differs")
    if coordinator.get("worker_count") != 2 or coordinator.get("partition_rule") != "authority_ordinal % 2 == shard_index" or coordinator.get("generic_cuda_device_permitted") is not False or coordinator.get("child_process_model") != "two_independent_child_processes_one_per_explicit_cuda_device":
        raise ProductionContractError("successor production coordinator process contract differs")
    if coordinator.get("max_units_policy") != MAX_UNITS_POLICY or coordinator.get("failed_work_unit_policy") != FAILED_WORK_UNIT_POLICY:
        raise ProductionContractError("successor production coordinator batch policy differs")
    _validate_custody_policy(coordinator.get("evidence_custody_policy"))
    if (
        coordinator.get("old_root") != "results/baseline/g8/work_units"
        or coordinator.get("successor_runtime_root") != SUCCESSOR_LOGICAL_RUNTIME_ROOT
        or coordinator.get("successor_work_unit_root") != SUCCESSOR_LOGICAL_RUNTIME_ROOT
        or coordinator.get("old_root") == coordinator.get("successor_runtime_root")
        or coordinator.get("old_result_ingest") is not False
    ):
        raise ProductionContractError("successor production coordinator root isolation differs")
    if coordinator.get("protected_counters") != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise ProductionContractError("successor production coordinator protected counters are nonzero")
    if coordinator.get("pre_measurement_repair_policy_sha256") != repair_policy_sha or coordinator.get("pre_measurement_retry_compatibility") != repair_policy:
        raise ProductionContractError("successor production coordinator repair policy differs")
    if coordinator.get("contract_sha256") != digest_without_field(coordinator, "contract_sha256"):
        raise ProductionContractError("successor production coordinator digest differs")

    runner_sha = _file_sha256(PRODUCTION_RUNNER_CONTRACT)
    if runner.get("artifact_role") != "g8_c_pascal_successor_production_runner_contract" or runner.get("campaign_manifest_sha256") != campaign_sha or runner.get("readiness_state_sha256") != readiness_sha or runner.get("production_source_manifest_sha256") != production_source_sha or runner.get("coordinator_contract_sha256") != _file_sha256(PRODUCTION_COORDINATOR_CONTRACT):
        raise ProductionContractError("successor production runner artifact binding differs")
    if runner.get("workers") != expected_workers or runner.get("runner_source_paths") != source_paths:
        raise ProductionContractError("successor production runner closure/topology differs")
    if runner.get("max_units_policy") != MAX_UNITS_POLICY or runner.get("failed_work_unit_policy") != FAILED_WORK_UNIT_POLICY:
        raise ProductionContractError("successor production runner batch policy differs")
    if runner.get("required_identity_count") != REQUIRED_COUNT or runner.get("trials_per_identity") != TRIALS_PER_IDENTITY or runner.get("lock_file_sha256") != _file_sha256(REPO_ROOT / "requirements-pascal.lock"):
        raise ProductionContractError("successor production runner physical/lock binding differs")
    if runner.get("driver_version_required") is not True or runner.get("old_result_ingest") is not False or runner.get("protected_counters") != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise ProductionContractError("successor production runner provenance/safety binding differs")
    if runner.get("pre_measurement_repair_policy_sha256") != repair_policy_sha or runner.get("pre_measurement_retry_compatibility") != repair_policy:
        raise ProductionContractError("successor production runner repair policy differs")
    if runner.get("contract_sha256") != digest_without_field(runner, "contract_sha256"):
        raise ProductionContractError("successor production runner digest differs")

    contract_sha = _file_sha256(PRODUCTION_CONTRACT)
    if contract.get("artifact_role") != PRODUCTION_CONTRACT_ARTIFACT_ROLE or contract.get("campaign_manifest_sha256") != campaign_sha or contract.get("readiness_state_sha256") != readiness_sha or contract.get("source_manifest_sha256") != production_source_sha or contract.get("production_source_manifest_sha256") != production_source_sha or contract.get("runner_contract_sha256") != runner_sha or contract.get("coordinator_contract_sha256") != _file_sha256(PRODUCTION_COORDINATOR_CONTRACT):
        raise ProductionContractError("successor production execution closure binding differs")
    if contract.get("required_identity_count") != REQUIRED_COUNT or contract.get("trials_per_identity") != TRIALS_PER_IDENTITY or contract.get("lock_file_sha256") != _file_sha256(REPO_ROOT / "requirements-pascal.lock"):
        raise ProductionContractError("successor production contract physical/lock binding differs")
    if contract.get("execution_profile_provenance_fields") != list(PRODUCTION_PROVENANCE_FIELDS):
        raise ProductionContractError("successor production profile provenance field closure differs")
    if contract.get("pre_measurement_repair_policy_sha256") != repair_policy_sha or contract.get("pre_measurement_retry_compatibility") != repair_policy:
        raise ProductionContractError("successor production contract repair policy differs")
    if contract.get("worker_batch_policy") != {"max_units": MAX_UNITS_POLICY, "failed_work_unit": FAILED_WORK_UNIT_POLICY}:
        raise ProductionContractError("successor production batch policy differs")
    _validate_custody_policy(contract.get("evidence_custody_policy"))
    if contract.get("driver_version_required") is not True or contract.get("old_result_ingest") is not False or contract.get("protected_counters") != {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0}:
        raise ProductionContractError("successor production contract safety/provenance binding differs")
    if contract.get("contract_sha256") != digest_without_field(contract, "contract_sha256"):
        raise ProductionContractError("successor production contract digest differs")
    return {
        "campaign_id": campaign_id,
        "production_contract_sha256": contract_sha,
        "production_source_manifest_sha256": production_source_sha,
        "production_runner_contract_sha256": runner_sha,
        "production_coordinator_contract_sha256": _file_sha256(PRODUCTION_COORDINATOR_CONTRACT),
        "source_paths": source_paths,
        "workers": expected_workers,
    }


__all__ = [
    "PRODUCTION_CONTRACT",
    "PRODUCTION_SOURCE_MANIFEST",
    "PRODUCTION_RUNNER_CONTRACT",
    "PRODUCTION_COORDINATOR_CONTRACT",
    "PRE_MEASUREMENT_REPAIR_POLICY",
    "PRODUCTION_WORKERS",
    "REQUIRED_PRODUCTION_SOURCE_PATHS",
    "PRODUCTION_PROVENANCE_FIELDS",
    "PRODUCTION_SCHEMA_VERSION",
    "PRODUCTION_STATE_ARTIFACT_ROLE",
    "REQUEST_ARTIFACT_ROLE",
    "RESULT_ARTIFACT_ROLE",
    "STATUS_ACCEPTED",
    "STATUS_CLAIMED",
    "STATUS_FAILED",
    "STATUS_REQUEST_PUBLISHED",
    "STATUS_RESULT_PUBLISHED",
    "STATUS_TERMINAL_INVALID",
    "MAX_UNITS_POLICY",
    "FAILED_WORK_UNIT_POLICY",
    "PASCAL_SUCCESSOR_CUSTODY_POLICY",
    "SuccessorProductionError",
    "ProductionContractError",
    "PublicationConflict",
    "StaleStateError",
    "RuntimeRootError",
    "RecoveryError",
    "successor_bindings",
    "unit_digest",
    "ensure_runtime_root",
    "state_path",
    "request_path",
    "result_path",
    "build_request",
    "validate_request",
    "build_result",
    "validate_result",
    "build_state",
    "validate_production_state_snapshot",
    "validate_state_transition",
    "read_state",
    "publish_state",
    "execute_frozen_measurement",
    "_frozen_measurement_request_view",
    "run_unit",
    "inspect_unit",
    "validate_runtime_namespace",
    "audit_campaign",
    "validate_campaign_state",
    "reconcile_campaign",
    "authenticate_worker_profile",
    "exact_shard_partition",
    "validate_production_contracts",
]
