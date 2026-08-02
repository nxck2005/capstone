#!/usr/bin/env python3
"""Generate the deterministic G8_B B2 unit-state contract artifact."""

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
    EXPECTED_B1C_CONTRACT_ID,
    EXPECTED_B1C_CONTRACT_SHA256,
    EXPECTED_CAMPAIGN_ID,
    EXPECTED_CAMPAIGN_MANIFEST_SHA256,
    EXPECTED_REQUIRED_IDENTITIES_SHA256,
    EXPECTED_REQUIRED_WORK_UNIT_COUNT,
    EXPECTED_SELECTION_POLICY_SHA256,
    PHASE,
    SHARDING_ALGORITHM,
    SHARD_PLAN_ARTIFACT_ROLE,
    SHARD_PLAN_DIGEST_RULE,
    SHARD_PLAN_FIELDS,
    SHARD_PLAN_SCHEMA_VERSION,
    SHARD_FORMULA,
    STATE_STATUSES,
    STATUS_CLAIMED,
    STATUS_FAILED,
    STATUS_RESULT_LINKED,
    UNIT_STATE_ARTIFACT_ROLE,
    UNIT_STATE_FIELDS,
    UNIT_STATE_IDENTITY_FIELDS,
    UNIT_STATE_RUNTIME_METADATA_FIELDS,
    UNIT_STATE_SCHEMA_VERSION,
    AuthenticatedExecutionContext,
)
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_ARTIFACT_ROLE = "g8_bler_state_contract"
CONTRACT_ID_PREFIX = "g8state"
CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_state_contract.json"
CONTRACT_SOURCE_PATHS = (
    "src/baseline/g8_bler_work_units.py",
    "tools/gen_g8_bler_state_contract.py",
    "tools/verify_g8_bler_state_contract.py",
)
CONTRACT_SOURCE_ROLE = "g8b_b2_contract_source"


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
        "contract_sources": _source_bindings(),
        "authority_bindings": authority,
        "execution_context": {
            "authentication": (
                "AuthenticatedExecutionContext verifies the exact B1C contract, campaign manifest, "
                "required-identity artifact, selection policy, schemas, count, and every complete "
                "ordered work-unit record once at construction."
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
            "rejections": [
                "absolute or relative aliases",
                "dot or dot-dot traversal",
                "outside-root paths",
                "symlink escape",
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
                    "merge_decision": "not implemented in B2",
                },
                STATUS_RESULT_LINKED: {
                    "result_path": "required nonblank canonical repository-relative path",
                    "result_sha256": "required lowercase SHA-256",
                    "result_validation": "B2 validates the reference fields only; it does not read, merge, or prefer a result",
                },
            },
            "all_statuses": list(STATE_STATUSES),
            "global_invariants": [
                "bindings match the authenticated context exactly",
                "canonical ordinal and required-record hash reproduce",
                "shard ownership and shard-plan digest reproduce",
                "attempt is a positive exact integer and cannot regress on replacement",
                "counters are exact non-negative integers and booleans are rejected",
                "test_split_access is exactly zero",
                "unknown and omitted fields fail closed",
                "NaN and infinity fail through canonical JSON and numeric metadata validation",
            ],
        },
        "publication": {
            "exclusive_creation": {
                "filesystem_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW when supported"],
                "winner_rule": "exactly one simultaneous creator succeeds; losers receive StateConflictError",
                "publication_bytes_prepared_before_open": True,
                "file_fsync": True,
                "directory_fsync": "after publication where supported; unsupported directory fsync is explicit",
                "silent_overwrite": False,
            },
            "atomic_replacement": {
                "operation": "optimistic compare-and-swap",
                "expected_previous_sha256_required": True,
                "steps": [
                    "read and validate current canonical state",
                    "compare actual SHA-256 with expected previous SHA-256",
                    "validate proposed state and monotonic attempt",
                    "write same-directory temporary",
                    "flush and fsync temporary file",
                    "os.replace destination atomically",
                    "fsync containing directory where supported",
                    "reread and validate installed canonical state",
                ],
                "stale_writer_error": "StaleWriterError",
                "malformed_state_repair": False,
                "partial_temporary_cleanup": True,
                "post_replace_failure": "installed old-or-new canonical state is recoverable by reread",
            },
        },
        "scope": {
            "tracked_live_unit_state_files": [],
            "b2_state_writes_only_in_isolated_temporary_tests": True,
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
            raise SystemExit(f"G8 B2 state contract is missing: {exc}") from exc
        if actual != expected:
            raise SystemExit("G8 B2 state contract is stale; regenerate it")
        print(
            "ok: G8 B2 state contract matches regenerated artifact "
            f"contract_id={json.loads(expected)['contract_id']}"
        )
        return 0
    CONTRACT_PATH.write_bytes(expected)
    print(
        "generated G8 B2 state contract "
        f"contract_id={json.loads(expected)['contract_id']} bytes={len(expected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
