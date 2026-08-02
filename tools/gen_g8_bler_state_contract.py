#!/usr/bin/env python3
"""Generate the deterministic G8_B B2C unit-state contract artifact.

The contract is reproducible from the bound sources and the immutable B1C
authority alone.  It deliberately does **not** bind its own artifact SHA-256,
so it can be built from an :class:`AuthenticatedExecutionContext` before the
corrected contract has been installed and registered.  The external artifact
SHA-256 that every unit state binds is obtained separately, from the
authenticated campaign-state artifact binding, by
:class:`AuthenticatedUnitStateContext`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import g8_bler_contract  # noqa: E402
from baseline.g8_bler_work_units import (  # noqa: E402
    B1C_REQUEST_SCHEMA_VERSION,
    B1C_RESULT_SCHEMA_VERSION,
    B1C_TOOLING_SCHEMA_VERSION,
    B3_RESTART_COMMAND,
    CHECKPOINT,
    LOCK_DIRECTORY_NAME,
    LOCK_FILENAME_SUFFIX,
    PHASE,
    SHARD_FORMULA,
    SHARD_PLAN_ARTIFACT_ROLE,
    SHARD_PLAN_DIGEST_RULE,
    SHARD_PLAN_FIELDS,
    SHARD_PLAN_SCHEMA_VERSION,
    SHARDING_ALGORITHM,
    STAGING_FILENAME_SUFFIX,
    STATE_CONTRACT_ARTIFACT_ROLE,
    STATE_CONTRACT_ID_PREFIX,
    STATE_CONTRACT_REPO_RELATIVE_PATH,
    STATE_CONTRACT_SCHEMA_VERSION,
    STATE_CONTRACT_SOURCE_PATHS,
    STATE_CONTRACT_SOURCE_ROLE,
    STATE_STATUSES,
    STATUS_CLAIMED,
    STATUS_FAILED,
    STATUS_RESULT_LINKED,
    SUPERSEDED_CHECKPOINT,
    SUPERSEDED_STATE_CONTRACT_BYTES,
    SUPERSEDED_STATE_CONTRACT_ID,
    SUPERSEDED_STATE_CONTRACT_SHA256,
    UNIT_STATE_ARTIFACT_ROLE,
    UNIT_STATE_FIELDS,
    UNIT_STATE_IDENTITY_FIELDS,
    UNIT_STATE_PERMANENT_FIELDS,
    UNIT_STATE_RUNTIME_METADATA_FIELDS,
    UNIT_STATE_SCHEMA_VERSION,
    UNIT_STATE_SHARD_FIELDS,
    AuthenticatedExecutionContext,
)
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes


CONTRACT_SCHEMA_VERSION = STATE_CONTRACT_SCHEMA_VERSION
CONTRACT_ARTIFACT_ROLE = STATE_CONTRACT_ARTIFACT_ROLE
CONTRACT_ID_PREFIX = STATE_CONTRACT_ID_PREFIX
CONTRACT_PATH = REPO_ROOT / STATE_CONTRACT_REPO_RELATIVE_PATH
CONTRACT_SOURCE_PATHS = STATE_CONTRACT_SOURCE_PATHS
CONTRACT_SOURCE_ROLE = STATE_CONTRACT_SOURCE_ROLE


def contract_identifier(payload: dict[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


def _source_bindings() -> list[dict[str, Any]]:
    bindings = []
    for relative in CONTRACT_SOURCE_PATHS:
        body = (REPO_ROOT / relative).read_bytes()
        bindings.append(
            {
                "path": relative,
                "role": CONTRACT_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            }
        )
    return bindings


def _authority(context: AuthenticatedExecutionContext) -> dict[str, Any]:
    binding = context.authority_binding()
    return {
        "campaign_id": binding["campaign_id"],
        "campaign_manifest_sha256": binding["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": binding["required_bler_artifact_sha256"],
        "selection_policy_sha256": binding["selection_policy_sha256"],
        "bler_tooling_contract_id": binding["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": binding["bler_tooling_contract_sha256"],
        "tooling_schema_version": binding["tooling_schema_version"],
        "request_schema_version": binding["request_schema_version"],
        "result_schema_version": binding["result_schema_version"],
        "required_work_unit_count": binding["required_work_unit_count"],
    }


def _build_without_id(context: AuthenticatedExecutionContext) -> dict[str, Any]:
    authority = _authority(context)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifact_role": CONTRACT_ARTIFACT_ROLE,
        "campaign": "G-8",
        "phase": PHASE,
        "checkpoint": CHECKPOINT,
        "supersedes": {
            "checkpoint": SUPERSEDED_CHECKPOINT,
            "contract_id": SUPERSEDED_STATE_CONTRACT_ID,
            "contract_sha256": SUPERSEDED_STATE_CONTRACT_SHA256,
            "contract_bytes": SUPERSEDED_STATE_CONTRACT_BYTES,
            "reason": (
                "B2 first publication opened the final pathname directly, its "
                "replacement performed an unlocked read-compare-replace that it "
                "described as compare-and-swap, its failure path could close one "
                "descriptor twice, its symlink guards followed links so dangling "
                "links passed, directory fsync failures were swallowed, results "
                "were neither request-bound nor terminal, unit states did not bind "
                "their own state contract, and its verifier imported every expected "
                "value from the module under test"
            ),
            "states_written_under_the_superseded_contract": 0,
            "per_unit_migration_required": False,
        },
        "contract_sources": _source_bindings(),
        "authority_bindings": authority,
        "adversary_model": {
            "assumes": "a trusted local user and a single trusted repository checkout",
            "defends_against": [
                "two cooperating workers racing on the same unit",
                "a hard process kill at any point during publication",
                "a stale writer resuming with an obsolete predecessor digest",
                "an orphaned staging artifact from an earlier killed writer",
                "a symlink, dangling symlink, or alias left by a careless tool or partial restore",
                "a parent-directory rename between validation and publication",
            ],
            "does_not_claim": (
                "defeat of a local attacker who already holds write access to the "
                "work-unit root and can win an unbounded time-of-check/time-of-use "
                "race; descriptor-relative publication narrows that window rather "
                "than closing it"
            ),
            "filesystem_requirements": [
                "Linux or POSIX",
                "descriptor-relative openat/linkat/renameat",
                "hard links within one directory",
                "working directory fsync",
                "fcntl.flock",
            ],
            "unavailable_primitive_policy": (
                "fail closed with AtomicStateError stating that crash-durable "
                "publication is unavailable; never fall back to writing the final "
                "pathname directly"
            ),
        },
        "execution_context": {
            "layers": {
                "AuthenticatedExecutionContext": (
                    "authenticates the exact B1C contract, campaign manifest, "
                    "required-identity artifact, selection policy, schemas, count, "
                    "and every complete ordered work-unit record once at "
                    "construction; sufficient for contract generation and shard "
                    "planning"
                ),
                "AuthenticatedUnitStateContext": (
                    "wraps an authenticated execution context and additionally "
                    "authenticates the registered B2C state-contract artifact "
                    "against campaign state — path, byte count, SHA-256, contract "
                    "ID, schema version, checkpoint, supersession and source "
                    "bindings; required by every unit-state build, validate, read, "
                    "create and replace operation"
                ),
            },
            "plain_execution_context_rejected_for_unit_state": True,
            "explicit_campaign_state_and_contract_paths_allowed": (
                "for isolated tests and staged migration verification; production "
                "defaults remain the committed paths"
            ),
            "immutable_internal_storage": (
                "canonical work-unit bytes, scalar bindings, tuples, and read-only mappings; "
                "public record and index lookups return fresh decoded copies"
            ),
            "authority_cache_rule": (
                "the required-identity artifact may be loaded and hashed once per context; "
                "record lookup and shard construction never invoke an uncached artifact loader per unit"
            ),
            "seed_derivation_identity": g8_bler_contract.SEED_DERIVATION_IDENTITY,
            "seed_forbidden_inputs": list(g8_bler_contract.SEED_FORBIDDEN_INPUTS),
            "shard_assignment_does_not_enter_seed_derivation": True,
            "request_or_result_construction": False,
            "test_split_access": 0,
        },
        "circular_dependency_rule": {
            "artifact_binds_its_own_sha256": False,
            "artifact_binds_its_own_path_as_a_source": False,
            "unit_state_binds_external_artifact_sha256": True,
            "external_sha256_source": (
                "the authenticated campaign-state produced-artifact binding for "
                f"{STATE_CONTRACT_REPO_RELATIVE_PATH}"
            ),
            "generator_requires_unit_state_context": False,
        },
        "sharding": {
            "schema_version": SHARD_PLAN_SCHEMA_VERSION,
            "artifact_role": SHARD_PLAN_ARTIFACT_ROLE,
            "algorithm": SHARDING_ALGORITHM,
            "ordinal_definition": "zero-based position in the exact ordered required_bler_work_units sequence",
            "formula": SHARD_FORMULA,
            "order_rule": "preserve canonical artifact order within every shard",
            "forbidden_order_sources": [
                "set order",
                "dictionary order",
                "filesystem order",
                "completion order",
                "hostname",
                "process ID",
                "worker ID",
                "runtime duration",
                "python hash",
            ],
            "argument_rule": "shard_count is an exact positive integer; shard_index is an exact integer in range",
            "plan_fields": list(SHARD_PLAN_FIELDS),
            "plan_digest_rule": SHARD_PLAN_DIGEST_RULE,
            "seed_independence": "changing shard_count or shard_index changes membership only, never seeds or request identity",
        },
        "unit_state_path": {
            "root_relative_layout": "<first-two-lowercase-hex>/<sha256-utf8-work-unit-id>.state.json",
            "digest_input": "exact UTF-8 bytes of the exact work-unit ID",
            "path_is_not_authority": "the payload carries the original ID and validation recomputes the digest",
            "lock_layout": f"{LOCK_DIRECTORY_NAME}/<sha256-utf8-work-unit-id>{LOCK_FILENAME_SUFFIX}",
            "staging_layout": f".<final-name>.<pid>.<random>{STAGING_FILENAME_SUFFIX}",
            "staging_is_never_state": True,
            "no_follow_inspection_rule": (
                "os.lstat, never exists()+is_symlink(), so a dangling symlink is "
                "detected as present rather than reported absent"
            ),
            "rejections": [
                "absolute or relative aliases",
                "dot or dot-dot traversal",
                "outside-root paths",
                "final-path symlinks including dangling symlinks",
                "root symlinks",
                "bucket-directory symlinks",
                "lock-file symlinks",
                "staging-file symlinks",
                "non-directory parent components",
                "a final path that is not a regular file",
                "wrong extension",
                "uppercase or malformed digest",
                "unknown IDs",
                "a path whose digest does not match its payload ID",
            ],
        },
        "unit_state_schema": {
            "schema_version": UNIT_STATE_SCHEMA_VERSION,
            "artifact_role": UNIT_STATE_ARTIFACT_ROLE,
            "top_level_fields": list(UNIT_STATE_FIELDS),
            "identity_fields": list(UNIT_STATE_IDENTITY_FIELDS),
            "runtime_metadata_fields": list(UNIT_STATE_RUNTIME_METADATA_FIELDS),
            "state_contract_binding_fields": [
                "bler_state_contract_id",
                "bler_state_contract_sha256",
            ],
            "identity_digest_rule": "identity_sha256 = sha256(canonical JSON of identity only); runtime_metadata is excluded",
            "canonical_file_encoding": "compact sorted-key JSON bytes, ensure_ascii=true, allow_nan=false, no trailing newline",
            "statuses": {
                STATUS_CLAIMED: {
                    "pre_execution_claim": True,
                    "scientific_execution_performed": False,
                    "trials_completed": 0,
                    "request_sha256": None,
                    "result_path": None,
                    "result_sha256": None,
                },
                STATUS_FAILED: {
                    "characterized_evidence": False,
                    "result_path": None,
                    "result_sha256": None,
                    "merge_decision": "not implemented in B2C",
                },
                STATUS_RESULT_LINKED: {
                    "request_sha256": "required lowercase SHA-256; no result may exist without a request binding",
                    "result_path": "required nonblank canonical repository-relative path",
                    "result_sha256": "required lowercase SHA-256",
                    "scientific_execution_performed": True,
                    "trials_completed": "strictly positive",
                    "test_split_access": 0,
                    "terminal": True,
                    "result_validation": (
                        "B2C validates the reference fields only; B3 validates the "
                        "actual request and result files"
                    ),
                },
            },
            "all_statuses": list(STATE_STATUSES),
            "global_invariants": [
                "bindings match the authenticated context exactly",
                "every state binds the registered B2C state-contract ID and SHA-256",
                "a state binding the superseded B2 contract is rejected",
                "canonical ordinal and required-record hash reproduce",
                "shard ownership and shard-plan digest reproduce",
                "attempt is a positive exact integer",
                "counters are exact non-negative integers and booleans are rejected",
                "test_split_access is exactly zero",
                "no result reference may exist without a bound request SHA-256",
                "unknown and omitted fields fail closed",
                "NaN and infinity fail through canonical JSON and numeric metadata validation",
            ],
        },
        "transitions": {
            "permanently_immutable_fields": list(UNIT_STATE_PERMANENT_FIELDS),
            "shard_fields_immutable_within_an_attempt": list(UNIT_STATE_SHARD_FIELDS),
            "same_attempt_rules": [
                "trials_completed may never decrease",
                "scientific_execution_performed may never change from true to false",
                "a non-null request_sha256 may never change or become null",
                "shard_count, shard_index, sharding_algorithm and shard_plan_digest are immutable",
                "failed -> claimed is forbidden",
                "failed -> result_linked is forbidden",
            ],
            "terminal_result_rule": (
                "a valid result_linked state is terminal; the only permitted "
                "operation is exact canonical-byte idempotence, and a replacement "
                "containing different bytes raises StateConflictError even when the "
                "writer supplies the current SHA-256"
            ),
            "retry_rule": {
                "new_attempt": "old_attempt + 1",
                "new_status": STATUS_CLAIMED,
                "request_sha256": None,
                "result_path": None,
                "result_sha256": None,
                "scientific_execution_performed": False,
                "trials_completed": 0,
                "test_split_access": 0,
                "reshard_permitted": True,
                "reshard_rule": (
                    "on that exact new-attempt transition the shard assignment may "
                    "change to another valid shard plan for the same work unit; this "
                    "is the only legal resharding path"
                ),
                "attempt_skip_rejected": True,
                "attempt_regression_rejected": True,
                "result_linked_reassignment_rejected": True,
            },
        },
        "publication": {
            "exclusive_creation": {
                "operation": "crash-atomic staged no-replace publication",
                "final_pathname_opened_for_writing": False,
                "steps": [
                    "validate the complete proposed state",
                    "render complete canonical bytes before any final-path publication",
                    "open the root descriptor-relative with O_DIRECTORY|O_NOFOLLOW",
                    "open the bucket descriptor-relative with O_DIRECTORY|O_NOFOLLOW",
                    "reject any object already occupying the final name via no-follow lstat",
                    "create a unique same-directory staging file with O_CREAT|O_EXCL|O_NOFOLLOW mode 0600",
                    "write all canonical bytes",
                    "flush and fsync the staging file",
                    "publish with descriptor-relative os.link(follow_symlinks=False), which cannot replace",
                    "fsync the containing directory",
                    "remove the staging name safely",
                    "reread and validate the installed final bytes",
                ],
                "no_replace_primitive": "descriptor-relative hard-link publication",
                "winner_rule": "exactly one simultaneous creator succeeds; every loser receives StateConflictError",
                "conflicting_objects": [
                    "pre-existing regular file",
                    "symlink",
                    "dangling symlink",
                    "directory",
                    "any other filesystem object",
                ],
                "file_fsync": True,
                "directory_fsync": "required; any failure becomes AtomicStateError",
                "silent_overwrite": False,
                "fallback_to_direct_final_path_write": False,
            },
            "atomic_replacement": {
                "operation": "linearizable compare-and-swap under an exclusive per-unit lock",
                "expected_previous_sha256_required": True,
                "critical_section": (
                    "the expected-previous-SHA check and the publication occur "
                    "inside one exclusive per-unit critical section"
                ),
                "process_lock": "fcntl.flock(LOCK_EX) on a canonical lock file",
                "thread_lock": "a process-local keyed threading.Lock",
                "lock_release": "on normal exit, on exception, and on process death",
                "lock_files_are_not_state": True,
                "steps": [
                    "acquire the process-local and process-safe per-unit locks",
                    "reread and validate the current canonical state",
                    "recompute its SHA-256",
                    "compare with expected_previous_sha256",
                    "reject a mismatch with StaleWriterError",
                    "enforce terminal result semantics and exact-byte idempotence",
                    "validate the complete transition",
                    "write and fsync a same-directory staging file",
                    "atomically replace the final state descriptor-relative",
                    "fsync the containing directory",
                    "reread and validate the installed state",
                    "return the installed SHA-256",
                ],
                "two_writers_same_predecessor": "exactly one succeeds; every loser raises StaleWriterError",
                "stale_writer_error": "StaleWriterError",
                "malformed_state_repair": False,
                "partial_staging_cleanup": True,
                "post_replace_failure": "installed old-or-new canonical state is recoverable by reread",
            },
            "descriptor_discipline": {
                "closed_exactly_once": True,
                "ownership_transfer_rule": "after ownership transfers to a stream the raw descriptor variable is set to None",
                "double_close_in_except_and_finally": False,
                "original_exception_preserved_as_cause": True,
                "secondary_ebadf_masking": False,
                "cleanup_failure_masks_publication_failure": False,
            },
            "hard_exit_semantics": {
                "before_publication": "an ignored staging artifact may remain; the final path is absent",
                "after_publication": "the final path contains complete canonical old-or-new bytes, never partial JSON",
                "retry_after_hard_exit": "a later valid retry or valid read always remains possible",
                "staging_artifacts_interpreted_as_state": False,
            },
        },
        "scope": {
            "tracked_live_unit_state_files": [],
            "b2c_state_writes_only_in_isolated_temporary_tests": True,
            "live_state_tree_required_absent_for_contract_verification": False,
            "live_state_tree_absence_is_an_explicit_verifier_option": "--require-no-live-state",
            "tracked_unit_state_or_lock_files_always_rejected": True,
            "exact_resume_and_merge_checkpoint": "B3",
            "runner_exists": False,
            "simulation_started": False,
            "bounded_smoke_started": False,
            "characterization_started": False,
            "scientific_execution_performed": False,
            "test_split_access": 0,
            "exact_b3_restart_command": B3_RESTART_COMMAND,
        },
    }


def build() -> dict[str, Any]:
    context = AuthenticatedExecutionContext()
    payload = _build_without_id(context)
    payload["contract_id"] = contract_identifier(payload)
    return payload


def write(path: Path = CONTRACT_PATH) -> bytes:
    payload = build()
    body = rendered_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = rendered_json(build())
    if args.check:
        try:
            actual = CONTRACT_PATH.read_bytes()
        except OSError as exc:
            raise SystemExit(f"G8 B2C state contract is missing: {exc}") from exc
        if actual != expected:
            raise SystemExit("G8 B2C state contract is stale; regenerate it")
        print(
            "ok: G8 B2C state contract matches regenerated artifact "
            f"contract_id={json.loads(expected)['contract_id']}"
        )
        return 0
    CONTRACT_PATH.write_bytes(expected)
    print(
        "generated G8 B2C state contract "
        f"contract_id={json.loads(expected)['contract_id']} bytes={len(expected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
