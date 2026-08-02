#!/usr/bin/env python3
"""Generate the deterministic G8_B B3 resume/merge contract.

This artifact describes the already-frozen resume layer.  It is generated
before registration: the B3 runtime can authenticate the B2C state contract
and the B1C authority while the candidate B3 artifact is still unregistered.
The artifact deliberately does not contain its own SHA-256 or bind its own
output path as a source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import g8_bler_contract as bler_contract  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    canonical_json,
    rendered_json,
    sha256_bytes,
)
from baseline.g8_bler_resume import AuthenticatedResumeContext  # noqa: E402
from baseline.classical.outage import write_json_atomically  # noqa: E402


CONTRACT_PATH = REPO_ROOT / resume.RESUME_CONTRACT_REPO_RELATIVE_PATH
CONTRACT_SCHEMA_VERSION = resume.RESUME_CONTRACT_SCHEMA_VERSION
CONTRACT_ARTIFACT_ROLE = resume.RESUME_CONTRACT_ARTIFACT_ROLE
CONTRACT_ID_PREFIX = resume.RESUME_CONTRACT_ID_PREFIX
CONTRACT_SOURCE_PATHS = resume.RESUME_CONTRACT_SOURCE_PATHS
CONTRACT_SOURCE_ROLE = resume.RESUME_CONTRACT_SOURCE_ROLE

RESUME_PLAN_FIELDS = (
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

MERGE_REPORT_FIELDS = (
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

# Closed B3 classifications (the verifier and contract must name all eight):
# absent, claimed_unbound, claimed_request_published,
# recoverable_failed_result, recoverable_complete_result, failed_retryable,
# completed_full_strength, terminal_nonmergeable.


def _binding(path: str) -> dict[str, Any]:
    body = (REPO_ROOT / path).read_bytes()
    return {
        "path": path,
        "role": CONTRACT_SOURCE_ROLE,
        "bytes": len(body),
        "sha256": sha256_bytes(body),
    }


def contract_identifier(payload: dict[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


def _authority(context: AuthenticatedResumeContext) -> dict[str, Any]:
    authority = context.authority_binding()
    return {
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "tooling_schema_version": authority["tooling_schema_version"],
        "request_schema_version": authority["request_schema_version"],
        "result_schema_version": authority["result_schema_version"],
        "required_work_unit_count": authority["required_work_unit_count"],
    }


def _schema_bindings() -> dict[str, Any]:
    return {
        "request": {
            "schema_version": bler_contract.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
            "artifact_role": bler_contract.REQUEST_ARTIFACT_ROLE,
            "fields": list(bler_contract.REQUEST_FIELDS),
            "unknown_fields_rejected": True,
            "omitted_field_defaults_permitted": False,
            "request_is_never_merge_eligible": True,
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        },
        "result": {
            "schema_version": bler_contract.BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
            "artifact_role": bler_contract.RESULT_ARTIFACT_ROLE,
            "fields": list(bler_contract.RESULT_FIELDS),
            "identity_fields": list(bler_contract.RESULT_IDENTITY_FIELDS),
            "measurement_fields": list(bler_contract.RESULT_MEASUREMENT_FIELDS),
            "execution_metadata_fields": list(bler_contract.RESULT_EXECUTION_METADATA_FIELDS),
            "disposition_fields": list(bler_contract.RESULT_DISPOSITION_FIELDS),
            "implementation_fields": list(bler_contract.IMPLEMENTATION_FIELDS),
            "statuses": list(bler_contract.RESULT_STATUSES),
            "status_rules": dict(bler_contract.RESULT_STATUS_RULES),
            "execution_metadata_rules": dict(bler_contract.EXECUTION_METADATA_RULES),
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        },
        "unit_state": {
            "schema_version": work_units.UNIT_STATE_SCHEMA_VERSION,
            "artifact_role": work_units.UNIT_STATE_ARTIFACT_ROLE,
            "fields": list(work_units.UNIT_STATE_FIELDS),
            "identity_fields": list(work_units.UNIT_STATE_IDENTITY_FIELDS),
            "runtime_metadata_fields": list(work_units.UNIT_STATE_RUNTIME_METADATA_FIELDS),
            "statuses": list(work_units.STATE_STATUSES),
            "canonical_file_encoding": resume.CANONICAL_FILE_ENCODING,
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        },
        "resume_plan": {
            "schema_version": resume.RESUME_PLAN_SCHEMA_VERSION,
            "artifact_role": resume.RESUME_PLAN_ARTIFACT_ROLE,
            "fields": list(RESUME_PLAN_FIELDS),
            "digest_field": resume.PLAN_DIGEST_FIELD,
            "digest_rule": "sha256(canonical JSON over the complete plan identity excluding plan_digest)",
        },
        "merge_report": {
            "schema_version": resume.MERGE_REPORT_SCHEMA_VERSION,
            "artifact_role": resume.MERGE_REPORT_ARTIFACT_ROLE,
            "fields": list(MERGE_REPORT_FIELDS),
            "digest_field": resume.MERGE_REPORT_DIGEST_FIELD,
            "digest_rule": "sha256(canonical JSON over the complete report identity excluding report_digest)",
        },
        "campaign_reconciliation": {
            "schema_version": resume.CAMPAIGN_RECONCILIATION_SCHEMA_VERSION,
            "artifact_role": resume.CAMPAIGN_RECONCILIATION_ARTIFACT_ROLE,
            "digest_field": "proposal_digest",
            "digest_rule": "sha256(canonical JSON over the complete proposal identity excluding proposal_digest)",
        },
    }


def _build_without_id(context: AuthenticatedResumeContext) -> dict[str, Any]:
    authority = _authority(context)
    state_binding = context.state_contract_binding()
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "campaign": resume.CAMPAIGN_ROLE,
        "artifact_role": CONTRACT_ARTIFACT_ROLE,
        "phase": resume.PHASE,
        "checkpoint": resume.CHECKPOINT,
        "scientific_execution_performed": False,
        "characterization_started": False,
        "bounded_smoke_started": False,
        "contract_sources": [_binding(path) for path in CONTRACT_SOURCE_PATHS],
        "authority_bindings": authority,
        "state_contract_binding": dict(state_binding),
        "schemas": _schema_bindings(),
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
            "logical_root": resume.WORK_UNIT_ROOT_LOGICAL_PREFIX,
            "state_filename_suffix": resume.STATE_FILENAME_SUFFIX,
            "request_filename_suffix": resume.REQUEST_FILENAME_SUFFIX,
            "result_filename_suffix": resume.RESULT_FILENAME_SUFFIX,
            "attempt_token_prefix": resume.ATTEMPT_TOKEN_PREFIX,
            "lock_directory_name": resume.LOCK_DIRECTORY_NAME,
            "lock_filename_suffix": resume.LOCK_FILENAME_SUFFIX,
            "staging_filename_suffix": resume.STAGING_FILENAME_SUFFIX,
            "allowed_root_entries": list(resume.ALLOWED_ROOT_ENTRIES),
            "allowed_bucket_entries": list(resume.ALLOWED_BUCKET_ENTRIES),
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
            "rejections": list(resume.CENSUS_REJECTIONS),
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
            "ordered": list(resume.CLASSIFICATIONS),
            "rules": {
                resume.CLASSIFICATION_ABSENT: "no state, request, or result: remaining; proposed attempt 1",
                resume.CLASSIFICATION_CLAIMED_UNBOUND: "claimed state without request: remaining; old attempt plus 1",
                resume.CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED: "claimed state with immutable request history: remaining; old attempt plus 1",
                resume.CLASSIFICATION_RECOVERABLE_FAILED_RESULT: "claimed state plus exact failed result: recoverable; no proposed attempt",
                resume.CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT: "claimed state plus exact complete result: recoverable; no proposed attempt",
                resume.CLASSIFICATION_FAILED_RETRYABLE: "failed state: remaining; old attempt plus 1",
                resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH: "result_linked exact complete merge-eligible result: completed; no proposed attempt",
                resume.CLASSIFICATION_TERMINAL_NONMERGEABLE: "result_linked exact complete non-mergeable result only in explicit isolated bounded mode",
            },
            "remaining": list(resume.REMAINING_CLASSIFICATIONS),
            "recoverable": list(resume.RECOVERABLE_CLASSIFICATIONS),
            "terminal": list(resume.TERMINAL_CLASSIFICATIONS),
            "proposed_attempt_policy": dict(resume.PROPOSED_ATTEMPT_POLICY),
            "unknown_classification_holds": True,
        },
        "recovery": {
            "matrix_rows": [
                {
                    "classification": resume.CLASSIFICATION_RECOVERABLE_FAILED_RESULT,
                    "result_status": bler_contract.STATUS_FAILED,
                    "successor_state_status": work_units.STATUS_FAILED,
                    "result_reference_in_successor": False,
                },
                {
                    "classification": resume.CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT,
                    "result_status": bler_contract.STATUS_COMPLETE,
                    "successor_state_status": work_units.STATUS_RESULT_LINKED,
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
            "schema_version": resume.RESUME_PLAN_SCHEMA_VERSION,
            "artifact_role": resume.RESUME_PLAN_ARTIFACT_ROLE,
            "digest_field": resume.PLAN_DIGEST_FIELD,
            "digest_excludes_only_self": True,
            "fields": list(RESUME_PLAN_FIELDS),
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
            "schema_version": resume.MERGE_REPORT_SCHEMA_VERSION,
            "artifact_role": resume.MERGE_REPORT_ARTIFACT_ROLE,
            "digest_field": resume.MERGE_REPORT_DIGEST_FIELD,
            "digest_excludes_only_self": True,
            "fields": list(MERGE_REPORT_FIELDS),
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
            "schema_version": resume.CAMPAIGN_RECONCILIATION_SCHEMA_VERSION,
            "artifact_role": resume.CAMPAIGN_RECONCILIATION_ARTIFACT_ROLE,
            "campaign_state_logical_path": resume.CAMPAIGN_STATE_LOGICAL_PATH,
            "derive_completed_only_from": resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH,
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
            "phase": resume.PHASE,
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
            "exact_restart_command": resume.B4_RESTART_COMMAND,
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


def build() -> dict[str, Any]:
    context = AuthenticatedResumeContext(require_resume_contract=False)
    payload = _build_without_id(context)
    payload["contract_id"] = contract_identifier(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    expected = rendered_json(payload)
    if args.check:
        try:
            actual = CONTRACT_PATH.read_bytes()
        except OSError as exc:
            raise SystemExit(f"missing {CONTRACT_PATH.relative_to(REPO_ROOT)}: {exc}") from exc
        if actual != expected:
            raise SystemExit("bler_resume_contract.json is stale; regenerate it")
        print(
            "ok: G8 B3 resume/merge contract matches regenerated artifact "
            f"contract_id={payload['contract_id']}"
        )
        return 0
    digest = write_json_atomically(CONTRACT_PATH, payload)
    print(
        f"wrote {CONTRACT_PATH.relative_to(REPO_ROOT)} "
        f"contract_id={payload['contract_id']} sha256={digest} "
        f"bytes={len(expected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
