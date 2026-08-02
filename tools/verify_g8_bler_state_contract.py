#!/usr/bin/env python3
"""Independently verify the generated G8_B B2C state contract.

Independence boundary.  This verifier imports ``baseline.g8_bler_work_units``
**only as the system under test** — to invoke its implementation — and never to
learn what the right answer is.  Every expected value used here is defined or
derived locally: the immutable campaign and B1C values, the superseded B2
values, the expected field sets and schema versions, the status and transition
rules, the shard formula and canonical ordering, the path derivation, the
publication guarantees, and the state-contract binding requirement.  It also
reads the campaign manifest, the required-identity artifact, the B1C tooling
contract, the campaign state, and the state contract for itself, with its own
canonical-JSON implementation.  It never imports anything from
``tools/gen_g8_bler_state_contract.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The module under test.  Deliberately imported as a whole module and never
# with `from baseline.g8_bler_work_units import <expected constant>`.
from baseline import g8_bler_work_units as units  # noqa: E402


# ---------------------------------------------------------------------------
# Independently defined expected truth
# ---------------------------------------------------------------------------

EXPECTED_PHASE = "G8_B"
EXPECTED_CHECKPOINT = "B2C"
EXPECTED_CAMPAIGN_ROLE = "G-8"

EXPECTED_CAMPAIGN_ID = (
    "g8-8acd86ad87ef223187b69a2caf6ab8d29de3700dac9d5a60bb421cb228d8900a"
)
EXPECTED_CAMPAIGN_MANIFEST_SHA256 = (
    "0e9504abdc79e90e07044a12a26aea10d5d3ef2cfc645ee4ee2a2bbe4f0722d1"
)
EXPECTED_REQUIRED_IDENTITIES_SHA256 = (
    "b8f7540af2dcc34f3e2f070bbc651ccbd3af99fbbb335dc3988264216cc32b77"
)
EXPECTED_SELECTION_POLICY_SHA256 = (
    "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e"
)
EXPECTED_B1C_CONTRACT_ID = (
    "g8bler-fa7b64abd2b20078668b5f251ea75dd3264f9293941657f70d2979bc83907975"
)
EXPECTED_B1C_CONTRACT_SHA256 = (
    "bc0db0a8ffe7b62238fa13e83ac9e82dec257816c98edce9a15cd3b226132866"
)
EXPECTED_REQUIRED_WORK_UNIT_COUNT = 3213  # literal-ok: frozen B1C required-identity count
EXPECTED_TOOLING_SCHEMA_VERSION = 2
EXPECTED_REQUEST_SCHEMA_VERSION = 2
EXPECTED_RESULT_SCHEMA_VERSION = 2

EXPECTED_SUPERSEDED_CHECKPOINT = "B2"
EXPECTED_SUPERSEDED_CONTRACT_ID = (
    "g8state-77ff45564fbe282179a860d70f2cc509264d06e1855d7360a50994a4fabaaa7c"
)
EXPECTED_SUPERSEDED_CONTRACT_SHA256 = (
    "2422c4c2a019c2a901cfd8732747555262dfca5601b28b0e700ff33743d4d939"
)
EXPECTED_SUPERSEDED_CONTRACT_BYTES = 9390  # literal-ok: committed B2 artifact length

EXPECTED_CONTRACT_SCHEMA_VERSION = 2
EXPECTED_CONTRACT_ARTIFACT_ROLE = "g8_bler_state_contract"
EXPECTED_CONTRACT_ID_PREFIX = "g8state"
EXPECTED_CONTRACT_SOURCE_ROLE = "g8b_b2c_contract_source"
EXPECTED_CONTRACT_SOURCE_PATHS = (
    "src/baseline/g8_bler_work_units.py",
    "tools/gen_g8_bler_state_contract.py",
    "tools/verify_g8_bler_state_contract.py",
)

EXPECTED_UNIT_STATE_SCHEMA_VERSION = 2
EXPECTED_UNIT_STATE_ARTIFACT_ROLE = "g8_bler_work_unit_state"
EXPECTED_UNIT_STATE_FIELDS = (
    "schema_version",
    "artifact_role",
    "identity",
    "runtime_metadata",
    "identity_sha256",
)
EXPECTED_UNIT_STATE_IDENTITY_FIELDS = (
    "schema_version",
    "artifact_role",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "bler_state_contract_id",
    "bler_state_contract_sha256",
    "request_schema_version",
    "result_schema_version",
    "work_unit_id",
    "canonical_ordinal",
    "required_work_unit_record_sha256",
    "sharding_algorithm",
    "shard_index",
    "shard_count",
    "shard_plan_digest",
    "attempt",
    "status",
    "request_sha256",
    "result_path",
    "result_sha256",
    "scientific_execution_performed",
    "trials_completed",
    "test_split_access",
)
EXPECTED_UNIT_STATE_RUNTIME_METADATA_FIELDS = (
    "hostname",
    "process_id",
    "device",
    "wall_clock_annotation",
    "update_annotation",
)
EXPECTED_UNIT_STATE_PERMANENT_FIELDS = (
    "schema_version",
    "artifact_role",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "bler_state_contract_id",
    "bler_state_contract_sha256",
    "request_schema_version",
    "result_schema_version",
    "work_unit_id",
    "canonical_ordinal",
    "required_work_unit_record_sha256",
    "sharding_algorithm",
)
EXPECTED_UNIT_STATE_SHARD_FIELDS = ("shard_index", "shard_count", "shard_plan_digest")

EXPECTED_STATUS_CLAIMED = "claimed"
EXPECTED_STATUS_FAILED = "failed"
EXPECTED_STATUS_RESULT_LINKED = "result_linked"
EXPECTED_STATUSES = (
    EXPECTED_STATUS_CLAIMED,
    EXPECTED_STATUS_FAILED,
    EXPECTED_STATUS_RESULT_LINKED,
)

EXPECTED_SHARDING_ALGORITHM = "canonical_ordinal_modulo_v1"
EXPECTED_SHARD_PLAN_SCHEMA_VERSION = 1
EXPECTED_SHARD_PLAN_ARTIFACT_ROLE = "g8_bler_shard_plan"
EXPECTED_SHARD_FORMULA = "ordinal % shard_count == shard_index"
EXPECTED_SHARD_PLAN_DIGEST_RULE = (
    "sha256(canonical JSON over the complete shard plan identity excluding plan_digest)"
)
EXPECTED_SHARD_PLAN_FIELDS = (
    "schema_version",
    "artifact_role",
    "sharding_algorithm",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "total_required_work_unit_count",
    "shard_index",
    "shard_count",
    "assigned_work_unit_ids",
    "assigned_work_unit_count",
    "plan_digest",
)

EXPECTED_STATE_FILENAME_SUFFIX = ".state.json"
EXPECTED_LOCK_DIRECTORY_NAME = ".locks"
EXPECTED_LOCK_FILENAME_SUFFIX = ".lock"
EXPECTED_STAGING_FILENAME_SUFFIX = ".staging"

CAMPAIGN_MANIFEST_PATH = REPO_ROOT / "results/baseline/g8/campaign_manifest.json"
REQUIRED_IDENTITIES_PATH = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
TOOLING_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json"
CAMPAIGN_STATE_PATH = REPO_ROOT / "results/baseline/g8/campaign_state.json"
CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_state_contract.json"
CONTRACT_REPO_RELATIVE_PATH = "results/baseline/g8/bler_state_contract.json"
LIVE_WORK_UNIT_ROOT = REPO_ROOT / "results/baseline/g8/work_units"
# Deliberately a separate name from the path-derivation root above: the
# optional closeout check inspects the runtime tree, while path derivation
# must keep reproducing the canonical layout regardless.
LIVE_STATE_TREE_PATH = LIVE_WORK_UNIT_ROOT

EXPECTED_B3_RESTART_COMMAND = (
    'rg -n "resume|remaining_work|merge|request_sha256|result_sha256|'
    'completed_work_unit_ids|in_progress_work_unit_id|unit_state|shard_plan" '
    "src/baseline tools tests"
)

_SHARD_COUNTS_UNDER_TEST = (1, 2, 3, 7, 11, 32, 64, 127, EXPECTED_REQUIRED_WORK_UNIT_COUNT)
_RACING_CREATORS = 8


class G8BlerStateContractError(RuntimeError):
    """The generated B2C contract is stale, malformed, or incomplete."""


def _fail(message: str) -> None:
    raise G8BlerStateContractError(message)


# ---------------------------------------------------------------------------
# Independent canonical JSON and digests
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _rendered(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"cannot independently read {label} at {path}: {exc}")
    return payload, raw


# ---------------------------------------------------------------------------
# Independent reconstruction of the expected contract payload
# ---------------------------------------------------------------------------


def _independent_source_bindings() -> list[dict[str, Any]]:
    bindings = []
    for relative in EXPECTED_CONTRACT_SOURCE_PATHS:
        target = REPO_ROOT / relative
        try:
            body = target.read_bytes()
        except OSError as exc:
            _fail(f"cannot read bound B2C source {relative}: {exc}")
        bindings.append(
            {
                "path": relative,
                "role": EXPECTED_CONTRACT_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": _sha256(body),
            }
        )
    return bindings


def _independent_authority() -> dict[str, Any]:
    return {
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": EXPECTED_CAMPAIGN_MANIFEST_SHA256,
        "required_bler_artifact_sha256": EXPECTED_REQUIRED_IDENTITIES_SHA256,
        "selection_policy_sha256": EXPECTED_SELECTION_POLICY_SHA256,
        "bler_tooling_contract_id": EXPECTED_B1C_CONTRACT_ID,
        "bler_tooling_contract_sha256": EXPECTED_B1C_CONTRACT_SHA256,
        "tooling_schema_version": EXPECTED_TOOLING_SCHEMA_VERSION,
        "request_schema_version": EXPECTED_REQUEST_SCHEMA_VERSION,
        "result_schema_version": EXPECTED_RESULT_SCHEMA_VERSION,
        "required_work_unit_count": EXPECTED_REQUIRED_WORK_UNIT_COUNT,
    }


def _independent_seed_binding() -> tuple[str, list[str]]:
    """Read the seed identity from the B1C artifact, not from the SUT."""

    payload, raw = _read_json(TOOLING_CONTRACT_PATH, "the B1C tooling contract")
    if _sha256(raw) != EXPECTED_B1C_CONTRACT_SHA256:
        _fail("B1C tooling-contract artifact SHA-256 is not the immutable authority")
    if payload.get("contract_id") != EXPECTED_B1C_CONTRACT_ID:
        _fail("B1C tooling-contract ID is not the immutable authority")
    seed = payload.get("seed")
    if not isinstance(seed, dict):
        _fail("B1C tooling contract has no seed block")
    identity = seed.get("derivation_identity")
    forbidden = seed.get("forbidden_inputs")
    if not isinstance(identity, str) or not isinstance(forbidden, list):
        _fail("B1C seed derivation identity or forbidden inputs are malformed")
    return identity, list(forbidden)


def _expected_without_id() -> dict[str, Any]:
    """Independent reconstruction; this never calls or imports the generator."""

    seed_identity, seed_forbidden = _independent_seed_binding()
    return {
        "schema_version": EXPECTED_CONTRACT_SCHEMA_VERSION,
        "artifact_role": EXPECTED_CONTRACT_ARTIFACT_ROLE,
        "campaign": EXPECTED_CAMPAIGN_ROLE,
        "phase": EXPECTED_PHASE,
        "checkpoint": EXPECTED_CHECKPOINT,
        "supersedes": {
            "checkpoint": EXPECTED_SUPERSEDED_CHECKPOINT,
            "contract_id": EXPECTED_SUPERSEDED_CONTRACT_ID,
            "contract_sha256": EXPECTED_SUPERSEDED_CONTRACT_SHA256,
            "contract_bytes": EXPECTED_SUPERSEDED_CONTRACT_BYTES,
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
        "contract_sources": _independent_source_bindings(),
        "authority_bindings": _independent_authority(),
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
            "seed_derivation_identity": seed_identity,
            "seed_forbidden_inputs": seed_forbidden,
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
                f"{CONTRACT_REPO_RELATIVE_PATH}"
            ),
            "generator_requires_unit_state_context": False,
        },
        "sharding": {
            "schema_version": EXPECTED_SHARD_PLAN_SCHEMA_VERSION,
            "artifact_role": EXPECTED_SHARD_PLAN_ARTIFACT_ROLE,
            "algorithm": EXPECTED_SHARDING_ALGORITHM,
            "ordinal_definition": "zero-based position in the exact ordered required_bler_work_units sequence",
            "formula": EXPECTED_SHARD_FORMULA,
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
            "plan_fields": list(EXPECTED_SHARD_PLAN_FIELDS),
            "plan_digest_rule": EXPECTED_SHARD_PLAN_DIGEST_RULE,
            "seed_independence": "changing shard_count or shard_index changes membership only, never seeds or request identity",
        },
        "unit_state_path": {
            "root_relative_layout": "<first-two-lowercase-hex>/<sha256-utf8-work-unit-id>.state.json",
            "digest_input": "exact UTF-8 bytes of the exact work-unit ID",
            "path_is_not_authority": "the payload carries the original ID and validation recomputes the digest",
            "lock_layout": (
                f"{EXPECTED_LOCK_DIRECTORY_NAME}/<sha256-utf8-work-unit-id>"
                f"{EXPECTED_LOCK_FILENAME_SUFFIX}"
            ),
            "staging_layout": f".<final-name>.<pid>.<random>{EXPECTED_STAGING_FILENAME_SUFFIX}",
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
            "schema_version": EXPECTED_UNIT_STATE_SCHEMA_VERSION,
            "artifact_role": EXPECTED_UNIT_STATE_ARTIFACT_ROLE,
            "top_level_fields": list(EXPECTED_UNIT_STATE_FIELDS),
            "identity_fields": list(EXPECTED_UNIT_STATE_IDENTITY_FIELDS),
            "runtime_metadata_fields": list(EXPECTED_UNIT_STATE_RUNTIME_METADATA_FIELDS),
            "state_contract_binding_fields": [
                "bler_state_contract_id",
                "bler_state_contract_sha256",
            ],
            "identity_digest_rule": "identity_sha256 = sha256(canonical JSON of identity only); runtime_metadata is excluded",
            "canonical_file_encoding": "compact sorted-key JSON bytes, ensure_ascii=true, allow_nan=false, no trailing newline",
            "statuses": {
                EXPECTED_STATUS_CLAIMED: {
                    "pre_execution_claim": True,
                    "scientific_execution_performed": False,
                    "trials_completed": 0,
                    "request_sha256": None,
                    "result_path": None,
                    "result_sha256": None,
                },
                EXPECTED_STATUS_FAILED: {
                    "characterized_evidence": False,
                    "result_path": None,
                    "result_sha256": None,
                    "merge_decision": "not implemented in B2C",
                },
                EXPECTED_STATUS_RESULT_LINKED: {
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
            "all_statuses": list(EXPECTED_STATUSES),
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
            "permanently_immutable_fields": list(EXPECTED_UNIT_STATE_PERMANENT_FIELDS),
            "shard_fields_immutable_within_an_attempt": list(EXPECTED_UNIT_STATE_SHARD_FIELDS),
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
                "new_status": EXPECTED_STATUS_CLAIMED,
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
            "exact_b3_restart_command": EXPECTED_B3_RESTART_COMMAND,
        },
    }


def _independent_contract_identifier(payload: dict[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{EXPECTED_CONTRACT_ID_PREFIX}-{_sha256(_canonical(basis))}"


# ---------------------------------------------------------------------------
# Independent checks against the committed artifacts
# ---------------------------------------------------------------------------


def _verify_campaign_manifest() -> None:
    payload, raw = _read_json(CAMPAIGN_MANIFEST_PATH, "the G8 campaign manifest")
    if _sha256(raw) != EXPECTED_CAMPAIGN_MANIFEST_SHA256:
        _fail("campaign-manifest SHA-256 is not the immutable authority")
    if not isinstance(payload, dict) or payload.get("campaign_id") != EXPECTED_CAMPAIGN_ID:
        _fail("campaign manifest does not carry the immutable campaign ID")


def _verify_required_identity_order(context: Any) -> list[str]:
    payload, raw = _read_json(REQUIRED_IDENTITIES_PATH, "the required-BLER identities")
    if _sha256(raw) != EXPECTED_REQUIRED_IDENTITIES_SHA256:
        _fail("required-BLER artifact SHA-256 is not the corrected B1C authority")
    work_units = payload.get("required_bler_work_units") if isinstance(payload, dict) else None
    if not isinstance(work_units, list) or len(work_units) != EXPECTED_REQUIRED_WORK_UNIT_COUNT:
        _fail(
            "required-BLER artifact does not contain exactly "
            f"{EXPECTED_REQUIRED_WORK_UNIT_COUNT} work units"
        )
    ids: list[str] = []
    for unit in work_units:
        if not isinstance(unit, dict) or not isinstance(unit.get("work_unit_id"), str):
            _fail("required-BLER artifact contains a malformed work-unit record")
        ids.append(unit["work_unit_id"])
        if _sha256(_canonical(unit)) != context.work_unit_record_sha256(unit["work_unit_id"]):
            _fail("context record bytes do not reproduce an artifact work-unit record")
    if ids != list(context.ordered_work_unit_ids):
        _fail("context order differs from the exact required_bler_work_units artifact order")
    return ids


def _verify_registered_state_contract_binding(
    raw_contract: bytes, campaign_state_path: Path
) -> dict[str, Any]:
    payload, _raw = _read_json(campaign_state_path, "the G8 campaign state")
    identity = payload.get("identity") if isinstance(payload, dict) else None
    if not isinstance(identity, dict):
        _fail("campaign state has no identity block")
    artifacts = identity.get("produced_artifacts")
    if not isinstance(artifacts, list):
        _fail("campaign state has no produced-artifact list")
    matches = [
        entry
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("path") == CONTRACT_REPO_RELATIVE_PATH
    ]
    if len(matches) != 1:
        _fail("campaign state must register exactly one B2C state-contract artifact")
    entry = matches[0]
    if entry.get("sha256") != _sha256(raw_contract) or entry.get("bytes") != len(raw_contract):
        _fail("registered state-contract binding does not match the artifact on disk")
    if entry.get("sha256") == EXPECTED_SUPERSEDED_CONTRACT_SHA256:
        _fail("campaign state still registers the superseded B2 state contract")
    if identity.get("completed_work_unit_ids") != []:
        _fail("campaign state records completed scientific work units")
    if identity.get("in_progress_work_unit_id") is not None:
        _fail("campaign state records an in-progress scientific work unit")
    counters = identity.get("counters")
    if not isinstance(counters, dict) or set(counters) != {
        "validation_decoding",
        "inference",
        "training",
        "test_access",
    }:
        _fail("campaign counters have the wrong schema")
    if any(value != 0 for value in counters.values()):
        _fail("campaign state reports nonzero scientific counters")
    return entry


def _verify_sharding_independently(context: Any, ids: list[str]) -> None:
    for count in _SHARD_COUNTS_UNDER_TEST:
        seen: list[str] = []
        for index in range(count):
            manual = [
                work_unit_id
                for ordinal, work_unit_id in enumerate(ids)
                if ordinal % count == index
            ]
            plan = units.build_shard_plan(context, count, index)
            if plan["assigned_work_unit_ids"] != manual:
                _fail(f"ordinal-modulo partition mismatch at {count}/{index}")
            if set(plan) != set(EXPECTED_SHARD_PLAN_FIELDS):
                _fail("shard plan field set changed")
            if plan["sharding_algorithm"] != EXPECTED_SHARDING_ALGORITHM:
                _fail("shard plan algorithm changed")
            if units.validate_shard_plan(context, plan) != plan:
                _fail("generated shard plan does not validate")
            body = dict(plan)
            digest = body.pop("plan_digest")
            if digest != _sha256(_canonical(body)):
                _fail("shard plan digest rule does not reproduce")
            seen.extend(manual)
        if len(seen) != len(set(seen)) or set(seen) != set(ids):
            _fail(f"shard partition is not complete and disjoint at count {count}")


def _verify_path_derivation(context: Any, ids: list[str]) -> None:
    midpoint = len(ids) // 2
    for work_unit_id in (ids[0], ids[midpoint], ids[-1]):
        digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
        relative = Path(digest[:2]) / f"{digest}{EXPECTED_STATE_FILENAME_SUFFIX}"
        path = units.unit_state_path(context, work_unit_id)
        if path.relative_to(LIVE_WORK_UNIT_ROOT) != relative:
            _fail("unit-state path does not reproduce the SHA-256 digest layout")


def _independent_identity_digest(state: dict[str, Any]) -> str:
    return _sha256(_canonical(state["identity"]))


def _verify_state_contract_binding(state_context: Any, contract_payload: dict[str, Any],
                                   raw_contract: bytes, ids: list[str]) -> None:
    plan = units.build_shard_plan(state_context, 7, 0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(state_context, work_unit_id, plan)
    identity = claim["identity"]
    if set(identity) != set(EXPECTED_UNIT_STATE_IDENTITY_FIELDS):
        _fail("unit-state identity field set is not the corrected B2C field set")
    if set(claim) != set(EXPECTED_UNIT_STATE_FIELDS):
        _fail("unit-state top-level field set changed")
    if claim["schema_version"] != EXPECTED_UNIT_STATE_SCHEMA_VERSION:
        _fail("unit-state schema version was not incremented for B2C")
    if identity["bler_state_contract_id"] != contract_payload["contract_id"]:
        _fail("unit state does not bind the B2C state-contract ID")
    if identity["bler_state_contract_sha256"] != _sha256(raw_contract):
        _fail("unit state does not bind the external B2C state-contract SHA-256")
    if identity["bler_state_contract_id"] == EXPECTED_SUPERSEDED_CONTRACT_ID:
        _fail("unit state binds the superseded B2 contract ID")
    if _independent_identity_digest(claim) != claim["identity_sha256"]:
        _fail("state identity digest does not reproduce independently")
    if identity["canonical_ordinal"] != ids.index(work_unit_id):
        _fail("unit-state canonical ordinal does not reproduce independently")

    # A state carrying the superseded contract must fail closed.
    forged = json.loads(json.dumps(claim))
    forged["identity"]["bler_state_contract_id"] = EXPECTED_SUPERSEDED_CONTRACT_ID
    forged["identity_sha256"] = _independent_identity_digest(forged)
    try:
        units.validate_unit_state(state_context, forged)
    except units.UnitStateError:
        pass
    else:
        _fail("a unit state binding the superseded B2 contract was accepted")

    # A plain execution context must not be accepted where the state layer is
    # required.
    try:
        units.build_unit_state(state_context.execution_context, work_unit_id, plan)
    except units.UnitStateError:
        pass
    else:
        _fail("a plain execution context was accepted for a unit-state build")


def _result_linked(state_context: Any, work_unit_id: str, plan: dict[str, Any], **overrides: Any):
    payload = {
        "attempt": 1,
        "status": EXPECTED_STATUS_RESULT_LINKED,
        "request_sha256": _sha256(b"request"),
        "result_path": "results/baseline/g8/results/unit.json",
        "result_sha256": _sha256(b"result"),
        "scientific_execution_performed": True,
        "trials_completed": 5,
        "test_split_access": 0,
    }
    payload.update(overrides)
    return units.build_unit_state(state_context, work_unit_id, plan, **payload)


def _verify_state_invariants(state_context: Any) -> None:
    plan = units.build_shard_plan(state_context, 7, 0)
    work_unit_id = plan["assigned_work_unit_ids"][0]

    # A result may never exist without a request binding.
    try:
        _result_linked(state_context, work_unit_id, plan, request_sha256=None)
    except units.UnitStateError:
        pass
    else:
        _fail("a result-linked state with a null request SHA-256 was accepted")

    claim = units.build_unit_state(state_context, work_unit_id, plan)
    failed = units.build_unit_state(
        state_context, work_unit_id, plan, status=EXPECTED_STATUS_FAILED
    )
    linked = _result_linked(state_context, work_unit_id, plan)

    # Terminal result semantics.
    changed = _result_linked(state_context, work_unit_id, plan, trials_completed=6)
    for previous, proposed, label in (
        (linked, changed, "a changed result-linked state"),
        (linked, claim, "a result-linked regression to a claim"),
        (failed, claim, "a same-attempt failed -> claimed transition"),
        (failed, linked, "a same-attempt failed -> result_linked transition"),
    ):
        try:
            units.validate_state_transition(previous, proposed)
        except units.StateConflictError:
            pass
        else:
            _fail(f"{label} was accepted")

    # Monotonicity within one attempt.
    def _partial(**overrides: Any) -> dict[str, Any]:
        return _result_linked(
            state_context,
            work_unit_id,
            plan,
            status=EXPECTED_STATUS_FAILED,
            result_path=None,
            result_sha256=None,
            **overrides,
        )

    def _forced(state: dict[str, Any], **identity_overrides: Any) -> dict[str, Any]:
        """Bypass build-time validation to isolate one transition rule."""

        forced = json.loads(json.dumps(state))
        forced["identity"].update(identity_overrides)
        forced["identity_sha256"] = _independent_identity_digest(forced)
        return forced

    high = _partial(trials_completed=9)
    for proposed, label in (
        (_partial(trials_completed=3), "a decrease in trials_completed"),
        (
            _forced(high, scientific_execution_performed=False),
            "a true -> false scientific execution flag",
        ),
        (_partial(request_sha256=_sha256(b"other"), trials_completed=9), "a changed request SHA-256"),
        (_forced(high, request_sha256=None), "a request SHA-256 becoming null"),
    ):
        try:
            units.validate_state_transition(high, proposed)
        except units.StateConflictError:
            pass
        else:
            _fail(f"{label} was accepted")

    # Retry: exactly the next attempt, clean, and only then may the shard move.
    shard_index = next(
        index
        for index in range(11)
        if work_unit_id in units.build_shard_plan(state_context, 11, index)["assigned_work_unit_ids"]
    )
    other_plan = units.build_shard_plan(state_context, 11, shard_index)
    retry = units.build_unit_state(state_context, work_unit_id, other_plan, attempt=2)
    units.validate_state_transition(failed, retry)

    skipped = units.build_unit_state(state_context, work_unit_id, other_plan, attempt=3)
    try:
        units.validate_state_transition(failed, skipped)
    except units.StateConflictError:
        pass
    else:
        _fail("an attempt skip was accepted")

    same_attempt_reshard = units.build_unit_state(state_context, work_unit_id, other_plan)
    try:
        units.validate_state_transition(claim, same_attempt_reshard)
    except units.StateConflictError:
        pass
    else:
        _fail("a same-attempt reshard was accepted")

    try:
        units.validate_state_transition(linked, retry)
    except units.StateConflictError:
        pass
    else:
        _fail("a result-linked reassignment was accepted")


def _verify_publication_and_locking(state_context: Any) -> None:
    with tempfile.TemporaryDirectory(prefix=".g8-b2c-contract-verify-") as raw_root:
        root = Path(raw_root) / "work_units"
        plan = units.build_shard_plan(state_context, 7, 0)
        work_unit_id = plan["assigned_work_unit_ids"][0]
        claim = units.build_unit_state(state_context, work_unit_id, plan)
        path = units.unit_state_path(state_context, work_unit_id, root=root)

        first_sha = units.create_unit_state_exclusive(state_context, claim, root=root)
        if first_sha != _sha256(path.read_bytes()):
            _fail("exclusive creation returned the wrong installed SHA-256")
        if not path.read_bytes().endswith(b"}"):
            _fail("installed state is not complete canonical JSON")
        try:
            units.create_unit_state_exclusive(state_context, claim, root=root)
        except units.StateConflictError:
            pass
        else:
            _fail("exclusive create accepted a second creator")

        # No staging artifact may survive as state.
        leftovers = [
            entry.name
            for entry in path.parent.iterdir()
            if entry.name.endswith(EXPECTED_STAGING_FILENAME_SUFFIX)
        ]
        if leftovers:
            _fail(f"staging artifacts survived a successful publication: {leftovers}")

        failed = units.build_unit_state(
            state_context, work_unit_id, plan, status=EXPECTED_STATUS_FAILED
        )
        second_sha = units.replace_unit_state(state_context, path, failed, first_sha, root=root)
        if second_sha != _sha256(_canonical(failed)):
            _fail("atomic replacement returned the wrong installed SHA-256")
        try:
            units.replace_unit_state(state_context, path, claim, first_sha, root=root)
        except units.StaleWriterError:
            pass
        else:
            _fail("a stale writer was accepted")
        if units.read_unit_state(state_context, path, root=root) != failed:
            _fail("installed state cannot be reread as one canonical snapshot")

        # Lock files must never be mistaken for state.
        lock_dir = root / EXPECTED_LOCK_DIRECTORY_NAME
        if not lock_dir.is_dir():
            _fail("no per-unit lock directory was created for a replacement")

        # A failed unit is reassigned by the one legal path: the exact next
        # attempt, clean, optionally on another valid shard plan.
        shard_index = next(
            index
            for index in range(11)
            if work_unit_id
            in units.build_shard_plan(state_context, 11, index)["assigned_work_unit_ids"]
        )
        other_plan = units.build_shard_plan(state_context, 11, shard_index)
        retry = units.build_unit_state(state_context, work_unit_id, other_plan, attempt=2)
        retry_sha = units.replace_unit_state(state_context, path, retry, second_sha, root=root)

        # Terminal result semantics through the publication boundary.
        linked = _result_linked(state_context, work_unit_id, other_plan, attempt=2)
        linked_sha = units.replace_unit_state(state_context, path, linked, retry_sha, root=root)
        if units.replace_unit_state(state_context, path, linked, linked_sha, root=root) != linked_sha:
            _fail("exact result-linked idempotence was not honoured")
        changed = _result_linked(state_context, work_unit_id, other_plan, attempt=2, trials_completed=6)
        try:
            units.replace_unit_state(state_context, path, changed, linked_sha, root=root)
        except units.StateConflictError:
            pass
        else:
            _fail("a result-linked state was changed with the current SHA-256")

        _verify_dangling_symlink_rejection(state_context, plan, Path(raw_root))
        _verify_hard_exit_recovery(state_context, plan, Path(raw_root))
        _verify_creation_race(state_context, plan, Path(raw_root))


def _verify_dangling_symlink_rejection(state_context: Any, plan: dict[str, Any], base: Path) -> None:
    root = base / "dangling"
    work_unit_id = plan["assigned_work_unit_ids"][1]
    state = units.build_unit_state(state_context, work_unit_id, plan)
    path = units.unit_state_path(state_context, work_unit_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(base / "does-not-exist", path)
    try:
        units.create_unit_state_exclusive(state_context, state, root=root)
    except (units.UnsafeUnitStatePathError, units.StateConflictError):
        pass
    else:
        _fail("a dangling symlink at the final unit-state name was accepted")


@contextlib.contextmanager
def _forking() -> Any:
    """Fork real child processes for the publication drills.

    Process lifetime is the property under test, so threads cannot substitute:
    only a real ``os._exit`` proves what survives a hard kill.  CPython warns
    that forking a possibly multi-threaded process can deadlock; that hazard
    needs a child which acquires a lock another thread held at fork time.
    These children only perform filesystem syscalls on an isolated temporary
    root and then ``os._exit``, never returning to interpreter shutdown, so
    the warning is suppressed deliberately and narrowly.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _fork_child(work: Any) -> int:
    """Run ``work`` in a real child process and return its exit status."""

    with _forking():
        pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        status = 0
        try:
            work()
        except BaseException:
            status = 1
        os._exit(status)
    _pid, raw_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(raw_status)


def _verify_hard_exit_recovery(state_context: Any, plan: dict[str, Any], base: Path) -> None:
    root = base / "hard-exit"
    work_unit_id = plan["assigned_work_unit_ids"][2]
    state = units.build_unit_state(state_context, work_unit_id, plan)
    path = units.unit_state_path(state_context, work_unit_id, root=root)

    def die_before_publication() -> None:  # pragma: no cover - child process
        units._publish_without_replace = lambda *args, **kwargs: os._exit(9)
        units.create_unit_state_exclusive(state_context, state, root=root)

    if _fork_child(die_before_publication) != 9:
        _fail("the hard-exit-before-publication drill did not exit as instructed")
    if path.exists() or path.is_symlink():
        _fail("a hard exit before publication left the final unit-state path present")

    def die_after_publication() -> None:  # pragma: no cover - child process
        original = units._fsync_directory_descriptor
        units._fsync_directory_descriptor = lambda *args, **kwargs: os._exit(9)
        try:
            units.create_unit_state_exclusive(state_context, state, root=root)
        finally:
            units._fsync_directory_descriptor = original

    if _fork_child(die_after_publication) != 9:
        _fail("the hard-exit-after-publication drill did not exit as instructed")
    installed = units.read_unit_state(state_context, path, root=root)
    if installed != state:
        _fail("a hard exit after publication did not leave complete canonical bytes")

    # An orphan staging artifact must neither be state nor block a retry.
    orphan_root = base / "orphan"
    orphan_path = units.unit_state_path(state_context, work_unit_id, root=orphan_root)
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    (orphan_path.parent / f".{orphan_path.name}.1.deadbeef{EXPECTED_STAGING_FILENAME_SUFFIX}").write_bytes(
        b'{"partial":'
    )
    units.create_unit_state_exclusive(state_context, state, root=orphan_root)
    if units.read_unit_state(state_context, orphan_path, root=orphan_root) != state:
        _fail("an orphan staging artifact blocked a valid retry")


def _verify_creation_race(state_context: Any, plan: dict[str, Any], base: Path) -> None:
    root = base / "race"
    work_unit_id = plan["assigned_work_unit_ids"][3]
    state = units.build_unit_state(state_context, work_unit_id, plan)
    path = units.unit_state_path(state_context, work_unit_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    read_fd, write_fd = os.pipe()

    def creator() -> None:  # pragma: no cover - child process
        os.read(read_fd, 1)
        units.create_unit_state_exclusive(state_context, state, root=root)

    children = []
    for _ in range(_RACING_CREATORS):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            status = 0
            try:
                creator()
            except BaseException:
                status = 1
            os._exit(status)
        children.append(pid)
    os.write(write_fd, b"g" * _RACING_CREATORS)
    winners = 0
    for pid in children:
        _pid, raw_status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(raw_status) == 0:
            winners += 1
    os.close(read_fd)
    os.close(write_fd)
    if winners != 1:
        _fail(f"{winners} of {_RACING_CREATORS} simultaneous creators succeeded; exactly one must win")
    if units.read_unit_state(state_context, path, root=root) != state:
        _fail("the creation race did not install one complete canonical state")


def _verify_no_live_state() -> None:
    if LIVE_STATE_TREE_PATH.exists() or LIVE_STATE_TREE_PATH.is_symlink():
        _fail("B2C closeout verification found a live work-unit state tree")


def _verify_no_tracked_state() -> None:
    result = subprocess.run(
        ["git", "ls-files", "results/baseline/g8/work_units"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        _fail("tracked per-unit state or lock files exist")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify(
    path: Path = CONTRACT_PATH,
    *,
    campaign_state_path: Path = CAMPAIGN_STATE_PATH,
    require_no_live_state: bool = False,
) -> dict[str, Any]:
    payload, raw = _read_json(path, "the B2C state contract")
    if not isinstance(payload, dict):
        _fail("B2C state contract is not a JSON object")
    if raw != _rendered(payload):
        _fail("B2C state contract is not canonical rendered JSON")

    expected = _expected_without_id()
    if set(payload) != set(expected) | {"contract_id"}:
        _fail("B2C state contract has missing or unknown top-level fields")
    if payload["schema_version"] != EXPECTED_CONTRACT_SCHEMA_VERSION:
        _fail("B2C state contract schema version was not incremented")
    if payload["artifact_role"] != EXPECTED_CONTRACT_ARTIFACT_ROLE:
        _fail("B2C state contract artifact role is wrong")
    if (
        payload["campaign"] != EXPECTED_CAMPAIGN_ROLE
        or payload["phase"] != EXPECTED_PHASE
        or payload["checkpoint"] != EXPECTED_CHECKPOINT
    ):
        _fail("B2C state contract campaign/phase/checkpoint binding is wrong")
    if payload["contract_id"] == EXPECTED_SUPERSEDED_CONTRACT_ID:
        _fail("B2C state contract still carries the superseded B2 contract ID")
    if payload["contract_id"] != _independent_contract_identifier(payload):
        _fail("B2C state contract ID does not reproduce")

    supersedes = payload["supersedes"]
    if (
        supersedes.get("checkpoint") != EXPECTED_SUPERSEDED_CHECKPOINT
        or supersedes.get("contract_id") != EXPECTED_SUPERSEDED_CONTRACT_ID
        or supersedes.get("contract_sha256") != EXPECTED_SUPERSEDED_CONTRACT_SHA256
        or supersedes.get("contract_bytes") != EXPECTED_SUPERSEDED_CONTRACT_BYTES
    ):
        _fail("B2C state contract does not supersede the exact B2 contract")

    sources = payload["contract_sources"]
    if not isinstance(sources, list) or [entry.get("path") for entry in sources] != list(
        EXPECTED_CONTRACT_SOURCE_PATHS
    ):
        _fail("B2C contract source path list changed")
    if sources != _independent_source_bindings():
        _fail("B2C bound source bytes or SHA-256 values changed")
    if any(Path(entry["path"]).is_absolute() for entry in sources):
        _fail("B2C contract binds an absolute source path")
    if any(entry["path"] == CONTRACT_REPO_RELATIVE_PATH for entry in sources):
        _fail("B2C contract binds its own output path")
    if _sha256(raw).encode("ascii") in raw:
        _fail("B2C contract binds its own artifact SHA-256")

    if {key: value for key, value in payload.items() if key != "contract_id"} != expected:
        _fail("B2C contract nested schema/content differs from independent reconstruction")

    _verify_campaign_manifest()
    registered = _verify_registered_state_contract_binding(raw, campaign_state_path)

    context = units.AuthenticatedExecutionContext()
    ids = _verify_required_identity_order(context)
    _verify_sharding_independently(context, ids)
    _verify_path_derivation(context, ids)

    state_context = units.AuthenticatedUnitStateContext(
        context, campaign_state_path=campaign_state_path, state_contract_path=path
    )
    _verify_state_contract_binding(state_context, payload, raw, ids)
    _verify_state_invariants(state_context)
    _verify_publication_and_locking(state_context)

    _verify_no_tracked_state()
    if require_no_live_state:
        _verify_no_live_state()

    scope = payload["scope"]
    if scope["scientific_execution_performed"] is not False or scope["test_split_access"] != 0:
        _fail("B2C contract scope claims scientific execution or test access")
    if scope["tracked_live_unit_state_files"] != [] or scope["exact_resume_and_merge_checkpoint"] != "B3":
        _fail("B2C contract scope boundary changed")
    if scope["exact_b3_restart_command"] != EXPECTED_B3_RESTART_COMMAND:
        _fail("B2C contract does not carry the exact B3 restart command")
    if registered["sha256"] != _sha256(raw):
        _fail("registered state-contract binding drifted during verification")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--campaign-state-path", type=Path, default=CAMPAIGN_STATE_PATH)
    parser.add_argument(
        "--require-no-live-state",
        action="store_true",
        help=(
            "additionally require that no runtime work-unit tree exists. Use this "
            "when closing B2C; a legitimate untracked runtime tree during B3, B4 or "
            "G8_C must not fail ordinary contract verification."
        ),
    )
    args = parser.parse_args(argv)
    try:
        payload = verify(
            args.path,
            campaign_state_path=args.campaign_state_path,
            require_no_live_state=args.require_no_live_state,
        )
    except (
        G8BlerStateContractError,
        units.G8BlerWorkUnitError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        raise SystemExit(f"G8 B2C state contract HOLD: {exc}") from exc
    print(
        "G8 B2C state contract PASS: "
        f"contract_id={payload['contract_id']}, supersedes={EXPECTED_SUPERSEDED_CONTRACT_ID}, "
        f"required_work_units={payload['authority_bindings']['required_work_unit_count']}, "
        f"unit_state_schema_version={payload['unit_state_schema']['schema_version']}, "
        "algorithm=canonical_ordinal_modulo_v1, publication=staged-no-replace, "
        "replacement=locked-cas, science=false, test_split_access=0, "
        f"require_no_live_state={str(args.require_no_live_state).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
