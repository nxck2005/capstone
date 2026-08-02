#!/usr/bin/env python3
"""Independently verify the generated G8_B B3 resume contract.

Independence is intentional.  This file does not import the B3 runtime module
or the generator and does not obtain expected B3 constants from either one.
It reconstructs the contract from immutable campaign artifacts, literal B3
checkpoint truth, and the bytes of the three contract sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline.g8_campaign import rendered_json  # noqa: E402


class G8BlerResumeContractError(RuntimeError):
    """An independently checked B3 resume-contract invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8BlerResumeContractError(message)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise G8BlerResumeContractError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


EXPECTED_PHASE = "G8_B"
EXPECTED_CHECKPOINT = "B3"
EXPECTED_CAMPAIGN = "G-8"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_ARTIFACT_ROLE = "g8_bler_resume_merge_contract"
EXPECTED_ID_PREFIX = "g8resume"
EXPECTED_SOURCE_ROLE = "g8b_b3_resume_contract_source"
EXPECTED_SOURCE_PATHS = (
    "src/baseline/g8_bler_resume.py",
    "tools/gen_g8_bler_resume_contract.py",
    "tools/verify_g8_bler_resume_contract.py",
)
EXPECTED_OUTPUT_PATH = "results/baseline/g8/bler_resume_contract.json"

EXPECTED_CAMPAIGN_ID = (
    "g8-8acd86ad87ef223187b69a2caf6ab8d29de3700dac9d5a60bb421cb228d8900a"
)
EXPECTED_MANIFEST_SHA256 = (
    "0e9504abdc79e90e07044a12a26aea10d5d3ef2cfc645ee4ee2a2bbe4f0722d1"
)
EXPECTED_REQUIRED_SHA256 = (
    "b8f7540af2dcc34f3e2f070bbc651ccbd3af99fbbb335dc3988264216cc32b77"
)
EXPECTED_SELECTION_SHA256 = (
    "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e"
)
EXPECTED_B1C_ID = (
    "g8bler-fa7b64abd2b20078668b5f251ea75dd3264f9293941657f70d2979bc83907975"
)
EXPECTED_B1C_SHA256 = (
    "bc0db0a8ffe7b62238fa13e83ac9e82dec257816c98edce9a15cd3b226132866"
)
EXPECTED_B2C_ID = (
    "g8state-a36b37f3c21d4254a50ffe5e893237ee4738c68c7b3e9d76b473856ca7605deb"
)
EXPECTED_B2C_SHA256 = (
    "cac1dcf803d435de7b483db04d12afc30bea4180a835d8c0476de65540fbf583"
)
EXPECTED_REQUIRED_COUNT = 3213
EXPECTED_B1C_SCHEMA_VERSION = 2
EXPECTED_REQUEST_SCHEMA_VERSION = 2
EXPECTED_RESULT_SCHEMA_VERSION = 2
EXPECTED_UNIT_STATE_SCHEMA_VERSION = 2
EXPECTED_STATE_ARTIFACT_ROLE = "g8_bler_work_unit_state"

EXPECTED_ROOT = "results/baseline/g8/work_units"
EXPECTED_STATE_SUFFIX = ".state.json"
EXPECTED_REQUEST_SUFFIX = ".request.json"
EXPECTED_RESULT_SUFFIX = ".result.json"
EXPECTED_ATTEMPT_PREFIX = "attempt-"
EXPECTED_LOCK_DIRECTORY = ".locks"
EXPECTED_LOCK_SUFFIX = ".lock"
EXPECTED_STAGING_SUFFIX = ".staging"
EXPECTED_B4_COMMAND = (
    'rg -n "bounded_smoke|run_g8_bler|SionnaLDPCAdapter|information_bit_stream|'
    'normal_stream|map_bits|max_log_llr" src/baseline tools tests'
)

EXPECTED_CLASSIFICATIONS = (
    "absent",
    "claimed_unbound",
    "claimed_request_published",
    "recoverable_failed_result",
    "recoverable_complete_result",
    "failed_retryable",
    "completed_full_strength",
    "terminal_nonmergeable",
)
EXPECTED_REMAINING = (
    "absent",
    "claimed_unbound",
    "claimed_request_published",
    "failed_retryable",
)
EXPECTED_RECOVERABLE = ("recoverable_failed_result", "recoverable_complete_result")
EXPECTED_TERMINAL = ("completed_full_strength", "terminal_nonmergeable")
EXPECTED_PLAN_FIELDS = (
    "schema_version",
    "artifact_role",
    "bler_resume_contract_id",
    "bler_resume_contract_sha256",
    "bler_state_contract_id",
    "bler_state_contract_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "request_schema_version",
    "result_schema_version",
    "unit_state_schema_version",
    "required_work_unit_count",
    "shard_count",
    "shard_index",
    "shard_plan_digest",
    "assigned_work_unit_ids",
    "assigned_unit_records",
    "completed_work_unit_ids",
    "recoverable_work_unit_ids",
    "remaining_work_unit_ids",
    "terminal_nonmergeable_work_unit_ids",
    "proposed_attempts",
    "logical_root",
    "scan_mode",
    "ignored_staging_count",
    "test_split_access",
    "plan_digest",
)
EXPECTED_REPORT_FIELDS = (
    "schema_version",
    "artifact_role",
    "bler_resume_contract_id",
    "bler_resume_contract_sha256",
    "bler_state_contract_id",
    "bler_state_contract_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "request_schema_version",
    "result_schema_version",
    "unit_state_schema_version",
    "required_work_unit_count",
    "required_work_unit_ids",
    "validated_complete_work_unit_ids",
    "missing_work_unit_ids",
    "remaining_work_unit_ids",
    "recoverable_work_unit_ids",
    "failed_work_unit_ids",
    "bounded_nonmergeable_work_unit_ids",
    "duplicate_count",
    "unknown_count",
    "exact_coverage_count",
    "valid_request_count",
    "valid_result_count",
    "valid_complete_result_count",
    "total_required_coverage_contribution",
    "coverage_complete",
    "merge_ready",
    "logical_root",
    "scan_mode",
    "ignored_staging_count",
    "test_split_access",
    "report_digest",
)


def _immutable_artifact(path: Path, expected_sha: str, expected_id: str, label: str) -> dict[str, Any]:
    payload, raw = _read_json(path, label)
    _require(_sha256(raw) == expected_sha, f"{label} SHA-256 changed")
    _require(payload.get("contract_id") == expected_id, f"{label} ID changed")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload


def _source_bindings() -> list[dict[str, Any]]:
    result = []
    for relative in EXPECTED_SOURCE_PATHS:
        try:
            body = (REPO_ROOT / relative).read_bytes()
        except OSError as exc:
            raise G8BlerResumeContractError(f"cannot read bound B3 source {relative}: {exc}") from exc
        result.append(
            {
                "path": relative,
                "role": EXPECTED_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": _sha256(body),
            }
        )
    return result


def _expected_without_id(b1c: dict[str, Any], b2c: dict[str, Any]) -> dict[str, Any]:
    request = b1c["request_schema"]
    result = b1c["result_schema"]
    unit = b2c["unit_state_schema"]
    authority = {
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "required_bler_artifact_sha256": EXPECTED_REQUIRED_SHA256,
        "selection_policy_sha256": EXPECTED_SELECTION_SHA256,
        "bler_tooling_contract_id": EXPECTED_B1C_ID,
        "bler_tooling_contract_sha256": EXPECTED_B1C_SHA256,
        "tooling_schema_version": EXPECTED_B1C_SCHEMA_VERSION,
        "request_schema_version": EXPECTED_REQUEST_SCHEMA_VERSION,
        "result_schema_version": EXPECTED_RESULT_SCHEMA_VERSION,
        "required_work_unit_count": EXPECTED_REQUIRED_COUNT,
    }
    state_binding = {
        "bler_state_contract_id": EXPECTED_B2C_ID,
        "bler_state_contract_sha256": EXPECTED_B2C_SHA256,
    }
    schemas = {
        "request": {
            "schema_version": request["version"],
            "artifact_role": request["artifact_role"],
            "fields": request["fields"],
            "unknown_fields_rejected": True,
            "omitted_field_defaults_permitted": False,
            "request_is_never_merge_eligible": True,
            "test_split_access": 0,
        },
        "result": {
            "schema_version": result["version"],
            "artifact_role": result["artifact_role"],
            "fields": result["sections"],
            "identity_fields": result["identity_fields"],
            "measurement_fields": result["measurement_fields"],
            "execution_metadata_fields": result["execution_metadata_fields"],
            "disposition_fields": result["disposition_fields"],
            "implementation_fields": result["implementation_fields"],
            "statuses": result["statuses"],
            "status_rules": result["status_rules"],
            "execution_metadata_rules": result["execution_metadata_rules"],
            "test_split_access": 0,
        },
        "unit_state": {
            "schema_version": unit["schema_version"],
            "artifact_role": unit["artifact_role"],
            "fields": unit["top_level_fields"],
            "identity_fields": unit["identity_fields"],
            "runtime_metadata_fields": unit["runtime_metadata_fields"],
            "statuses": unit["all_statuses"],
            "canonical_file_encoding": "compact sorted-key JSON bytes, ensure_ascii=true, allow_nan=false, separators (\",\", \":\"), no trailing newline",
            "test_split_access": 0,
        },
        "resume_plan": {
            "schema_version": 1,
            "artifact_role": "g8_bler_resume_plan",
            "fields": list(EXPECTED_PLAN_FIELDS),
            "digest_field": "plan_digest",
            "digest_rule": "sha256(canonical JSON over the complete plan identity excluding plan_digest)",
        },
        "merge_report": {
            "schema_version": 1,
            "artifact_role": "g8_bler_merge_validation_report",
            "fields": list(EXPECTED_REPORT_FIELDS),
            "digest_field": "report_digest",
            "digest_rule": "sha256(canonical JSON over the complete report identity excluding report_digest)",
        },
        "campaign_reconciliation": {
            "schema_version": 1,
            "artifact_role": "g8_campaign_reconciliation_proposal",
            "digest_field": "proposal_digest",
            "digest_rule": "sha256(canonical JSON over the complete proposal identity excluding proposal_digest)",
        },
    }

    rules = {
        "absent": "no state, request, or result: remaining; proposed attempt 1",
        "claimed_unbound": "claimed state without request: remaining; old attempt plus 1",
        "claimed_request_published": "claimed state with immutable request history: remaining; old attempt plus 1",
        "recoverable_failed_result": "claimed state plus exact failed result: recoverable; no proposed attempt",
        "recoverable_complete_result": "claimed state plus exact complete result: recoverable; no proposed attempt",
        "failed_retryable": "failed state: remaining; old attempt plus 1",
        "completed_full_strength": "result_linked exact complete merge-eligible result: completed; no proposed attempt",
        "terminal_nonmergeable": "result_linked exact complete non-mergeable result only in explicit isolated bounded mode",
    }
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "campaign": EXPECTED_CAMPAIGN,
        "artifact_role": EXPECTED_ARTIFACT_ROLE,
        "phase": EXPECTED_PHASE,
        "checkpoint": EXPECTED_CHECKPOINT,
        "scientific_execution_performed": False,
        "characterization_started": False,
        "bounded_smoke_started": False,
        "contract_sources": _source_bindings(),
        "authority_bindings": authority,
        "state_contract_binding": state_binding,
        "schemas": schemas,
        "lock": {
            "target": "the existing physical parent directory inode of the runtime root",
            "parent_must_already_exist": True,
            "inspection_creates_parent": False,
            "inspection_creates_root": False,
            "inspection_creates_lock_file": False,
            "open_flags": ["O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW"],
            "flock_function": "fcntl.flock",
            "exclusive_mode": "LOCK_EX",
            "shared_mode": "LOCK_SH",
            "exclusive_required_for": ["inspection", "recovery", "reconciliation", "merge"],
            "shared_required_for": ["future complete per-unit runner transaction"],
            "absent_root_never_bypasses_lock": True,
            "unsupported_directory_locking": "fail closed; no fallback lock or unlocked continuation",
            "lock_order": ["global parent-directory lock", "B2C per-unit lock"],
        },
        "lock_lease": {
            "opaque": True,
            "fields": [
                "canonical physical root",
                "mode",
                "owner PID",
                "active state",
                "parent st_dev",
                "parent st_ino",
            ],
            "rejects": [
                "missing lease",
                "inactive lease",
                "wrong-root lease",
                "shared lease where exclusive is required",
                "lease inherited across fork without reacquisition",
                "changed locked-parent inode",
            ],
            "public_repair_acquires_exclusive": True,
            "batch_inspection_reuses_one_exclusive_lease": True,
            "manual_nested_b2c_lock_for_replace": False,
        },
        "runtime_paths": {
            "logical_root": EXPECTED_ROOT,
            "state_filename_suffix": EXPECTED_STATE_SUFFIX,
            "request_filename_suffix": EXPECTED_REQUEST_SUFFIX,
            "result_filename_suffix": EXPECTED_RESULT_SUFFIX,
            "attempt_token_prefix": EXPECTED_ATTEMPT_PREFIX,
            "lock_directory_name": EXPECTED_LOCK_DIRECTORY,
            "lock_filename_suffix": EXPECTED_LOCK_SUFFIX,
            "staging_filename_suffix": EXPECTED_STAGING_SUFFIX,
            "allowed_root_entries": ["two-lowercase-hex bucket directory", ".locks directory"],
            "allowed_bucket_entries": [
                "<digest>.state.json",
                "<digest>.attempt-<attempt>.request.json",
                "<digest>.attempt-<attempt>.result.json",
                ".<final-name>.<pid>.<random>.staging (ignored orphan staging)",
            ],
            "absolute_paths_in_artifacts": False,
            "hostname_pid_inode_mtime_in_digests": False,
        },
        "census": {
            "read_only_default": True,
            "opens_descriptors_and_flocks": True,
            "root_absence_is_a_valid_empty_census": True,
            "root_absence_is_locked": True,
            "ignored_orphan_staging_is_counted": True,
            "unknown_entries_hold": True,
            "symlinks_and_dangling_symlinks_hold": True,
            "hard_linked_authoritative_files_hold": True,
            "filesystem_enumeration_order_is_ignored": True,
            "rejections": [
                "symlinks and dangling symlinks at every level",
                "unknown top-level entries",
                "wrong-case buckets",
                "non-directory buckets",
                "non-regular authoritative files",
                "files in the wrong bucket",
                "unknown work-unit digests",
                "malformed attempt names",
                "hard-linked authoritative aliases",
                "duplicate semantic artifacts",
                "request or result files for a future attempt",
                "filenames whose digest does not map to the embedded work-unit ID",
                "unrecognized temporary files",
                "nested directories not defined by the contract",
            ],
            "read_only_mutations": [
                "create lock file",
                "create runtime root",
                "create .locks",
                "create bucket",
                "create state",
                "write, rename, or unlink",
                "update campaign state",
            ],
        },
        "history_validation": {
            "validate_every_attempt_through_state_attempt": True,
            "exact_same_attempt_request_for_every_result": True,
            "all_request_bytes_canonical": True,
            "all_requests_validate_against_exact_authority": True,
            "full_strength_retry_requests_byte_identical": True,
            "no_request_or_result_beyond_state_attempt": True,
            "malformed_or_foreign_older_request_only_holds": True,
            "older_failed_results_contribute_zero": True,
            "older_complete_merge_eligible_result_while_state_advanced_holds": True,
            "incomplete_final_result_holds": True,
            "incomplete_result_is_invalid_in_production_and_bounded_inspection": True,
            "state_result_binding": {
                "result_linked_request_sha256_exact": True,
                "result_linked_result_sha256_exact": True,
                "result_linked_result_path_exact_current_attempt": True,
                "result_linked_trials_equal_result_measurement": True,
                "result_linked_scientific_execution_performed": True,
                "failed_persisted_request_sha256_exact": True,
                "failed_persisted_trials_equal_result_measurement": True,
                "failed_without_result_reports_state_trials": True,
                "failed_without_request_permitted_only_without_current_request_or_result": True,
            },
        },
        "classifications": {
            "ordered": list(EXPECTED_CLASSIFICATIONS),
            "rules": rules,
            "remaining": list(EXPECTED_REMAINING),
            "recoverable": list(EXPECTED_RECOVERABLE),
            "terminal": list(EXPECTED_TERMINAL),
            "proposed_attempt_policy": {
                "absent": "attempt_1",
                "claimed_unbound": "old_attempt_plus_1",
                "claimed_request_published": "old_attempt_plus_1",
                "failed_retryable": "old_attempt_plus_1",
                "recoverable_failed_result": None,
                "recoverable_complete_result": None,
                "completed_full_strength": None,
                "terminal_nonmergeable": None,
            },
            "unknown_classification_holds": True,
        },
        "recovery": {
            "matrix_rows": [
                {
                    "classification": "recoverable_failed_result",
                    "result_status": "failed",
                    "successor_state_status": "failed",
                    "result_reference_in_successor": False,
                },
                {
                    "classification": "recoverable_complete_result",
                    "result_status": "complete",
                    "successor_state_status": "result_linked",
                    "result_reference_in_successor": True,
                },
            ],
            "exactly_two_rows": True,
            "request_only_recovery": False,
            "repair_requires_exclusive_global_lease": True,
            "per_unit_compare_and_swap_owned_by_b2c": True,
            "no_manual_nested_b2c_lock": True,
        },
        "publication": {
            "canonical_bytes_before_destination_open": True,
            "staging_same_directory_unique": True,
            "staging_open_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
            "staging_mode": "0600",
            "flush_and_file_fsync": True,
            "no_replace_publication": "descriptor-relative hard link or equivalently proven no-replace primitive",
            "directory_fsync": True,
            "final_path_never_opened_for_writing": True,
            "symlink_dangling_symlink_and_hard_link_alias_rejected": True,
            "exact_existing_bytes_are_idempotent": True,
            "different_existing_bytes_are_conflict": True,
            "uncertain_publication": "reread installed artifact and accept only exact proposed canonical bytes and exact SHA-256; matching status alone is insufficient",
            "directory_fsync_failure_is_not_swallowed": True,
        },
        "resume_plan": {
            "schema_version": 1,
            "artifact_role": "g8_bler_resume_plan",
            "digest_field": "plan_digest",
            "digest_excludes_only_self": True,
            "fields": list(EXPECTED_PLAN_FIELDS),
            "binds": [
                "B3 contract ID and SHA",
                "B2C contract ID and SHA",
                "B1C contract ID and SHA",
                "campaign ID",
                "campaign manifest SHA",
                "required identities SHA",
                "selection-policy SHA",
                "required count",
                "shard count and index",
                "exact B2 shard-plan digest",
                "logical root only",
                "scan mode",
                "ignored staging count",
                "assigned unit records",
                "completed IDs",
                "recoverable IDs",
                "remaining IDs",
                "proposed attempt for every remaining unit",
                "test access",
            ],
            "forbidden_fields": [
                "absolute path",
                "hostname",
                "PID",
                "timestamp",
                "inode",
                "mtime",
                "filesystem order",
                "worker name",
                "completion order",
                "wall time",
            ],
            "order_rule": "exact frozen shard-plan order",
            "repeat_identical_bytes": True,
            "reshard_changes_membership_only": True,
        },
        "merge_report": {
            "schema_version": 1,
            "artifact_role": "g8_bler_merge_validation_report",
            "digest_field": "report_digest",
            "digest_excludes_only_self": True,
            "fields": list(EXPECTED_REPORT_FIELDS),
            "validation_only": True,
            "constructs_bler_table": False,
            "coverage_complete_requires": [
                "3213 required IDs in canonical authority order",
                "3213 unique completed_full_strength units",
                "3213 valid requests",
                "3213 valid complete results",
                "sum(required_coverage_contribution) == 3213",
                "no unknown IDs",
                "no duplicates",
                "no omissions",
                "test_split_access == 0",
            ],
            "partial_state_is_not_merge_ready": True,
        },
        "campaign_reconciliation": {
            "schema_version": 1,
            "artifact_role": "g8_campaign_reconciliation_proposal",
            "campaign_state_logical_path": "results/baseline/g8/campaign_state.json",
            "derive_completed_only_from": "completed_full_strength",
            "lag_allowed": True,
            "lead_holds": True,
            "apply_adds_only": True,
            "apply_never_removes": True,
            "bounded_and_recoverable_contribute": False,
            "failed_and_claimed_contribute": False,
            "in_progress_is_not_execution_authority": True,
            "counters_unchanged": True,
            "atomic_writer": "baseline.g8_campaign.write_campaign_state_atomically",
            "reread_exact_installed_bytes": True,
            "uncertain_publication_requires_exact_bytes": True,
            "live_b3_closeout_expected_change": "none",
        },
        "bounded_mode": {
            "requires_explicit_root": True,
            "requires_nonproduction_root": True,
            "rejects_none": True,
            "rejects_default_root": True,
            "rejects_lexical_resolved_symlink_and_same_inode_alias": True,
            "terminal_nonmergeable_permitted_only_here": True,
            "required_coverage_contribution": 0,
            "test_split_access": 0,
        },
        "no_science_boundary": {
            "phase": EXPECTED_PHASE,
            "stage": "tooling_open",
            "scientific_execution_performed": False,
            "validation_decoding": 0,
            "inference": 0,
            "training": 0,
            "test_access": 0,
            "blertable_exists": False,
            "runner_exists": False,
            "full_strength_requests_or_results": False,
            "production_runtime_root": False,
        },
        "b4_handoff": {
            "exact_restart_command": EXPECTED_B4_COMMAND,
            "runner_may_consume": [
                "authenticated B1C requests",
                "authenticated B2C state",
                "authenticated B3 plans and reconciliation",
                "exact G8_A work-unit authority",
            ],
            "runner_must_not_invent_work_unit_identities": True,
            "full_strength_not_authorized_by_b3": True,
        },
    }


def verify(path: Path, *, require_registered: bool = False) -> dict[str, Any]:
    payload, raw = _read_json(path, "B3 resume contract")
    _require(raw == rendered_json(payload), "B3 resume contract is not canonical rendered JSON")
    b1c = _immutable_artifact(
        REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json",
        EXPECTED_B1C_SHA256,
        EXPECTED_B1C_ID,
        "B1C tooling contract",
    )
    b2c = _immutable_artifact(
        REPO_ROOT / "results/baseline/g8/bler_state_contract.json",
        EXPECTED_B2C_SHA256,
        EXPECTED_B2C_ID,
        "B2C state contract",
    )
    expected = _expected_without_id(b1c, b2c)
    _require(set(payload) == set(expected) | {"contract_id"}, "B3 contract top-level schema changed")
    _require(payload["contract_id"].startswith(f"{EXPECTED_ID_PREFIX}-"), "B3 contract ID prefix changed")
    basis = dict(payload)
    supplied_id = basis.pop("contract_id")
    _require(supplied_id == f"{EXPECTED_ID_PREFIX}-{_sha256(_canonical(basis))}", "B3 contract ID does not reproduce")
    _require(payload["contract_sources"] == _source_bindings(), "B3 bound source bytes or SHA-256 changed")
    _require(
        [entry["path"] for entry in payload["contract_sources"]] == list(EXPECTED_SOURCE_PATHS),
        "B3 source path order changed",
    )
    _require(
        all(not Path(entry["path"]).is_absolute() for entry in payload["contract_sources"]),
        "B3 contract binds an absolute source path",
    )
    _require(
        all(entry["path"] != EXPECTED_OUTPUT_PATH for entry in payload["contract_sources"]),
        "B3 contract binds its own output path",
    )
    _require(_sha256(raw).encode("ascii") not in raw, "B3 contract binds its own SHA-256")
    _require(
        {key: value for key, value in payload.items() if key != "contract_id"} == expected,
        "B3 contract content differs from independently reconstructed truth",
    )

    manifest, manifest_raw = _read_json(
        REPO_ROOT / "results/baseline/g8/campaign_manifest.json", "campaign manifest"
    )
    required_path = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
    required_raw = required_path.read_bytes()
    _require(_sha256(manifest_raw) == EXPECTED_MANIFEST_SHA256, "campaign manifest bytes changed")
    _require(_sha256(required_raw) == EXPECTED_REQUIRED_SHA256, "required identities bytes changed")
    _require(manifest.get("campaign_id") == EXPECTED_CAMPAIGN_ID, "campaign ID changed")
    _require(payload["authority_bindings"]["campaign_id"] == manifest["campaign_id"], "authority campaign ID mismatch")
    _require(payload["authority_bindings"]["required_work_unit_count"] == EXPECTED_REQUIRED_COUNT, "required count changed")

    if require_registered:
        state, _state_raw = _read_json(
            REPO_ROOT / "results/baseline/g8/campaign_state.json", "campaign state"
        )
        artifacts = state.get("identity", {}).get("produced_artifacts", [])
        matches = [entry for entry in artifacts if entry.get("path") == EXPECTED_OUTPUT_PATH]
        _require(len(matches) == 1, "campaign state does not register exactly one B3 contract")
        _require(
            matches[0] == {"path": EXPECTED_OUTPUT_PATH, "sha256": _sha256(raw), "bytes": len(raw)},
            "registered B3 contract binding does not match installed bytes",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=REPO_ROOT / EXPECTED_OUTPUT_PATH)
    parser.add_argument("--require-registered", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = verify(args.path, require_registered=args.require_registered)
    except (G8BlerResumeContractError, OSError) as exc:
        raise SystemExit(f"G8 B3 resume contract HOLD: {exc}") from exc
    print(
        "G8 B3 resume contract PASS: "
        f"contract_id={payload['contract_id']} bytes={args.path.stat().st_size} "
        f"registered={str(args.require_registered).lower()} science=false test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
