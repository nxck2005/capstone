#!/usr/bin/env python3
"""Independently verify the generated G8_B B2 state contract.

The verifier has its own expected payload construction and exercises the
filesystem boundaries in isolated temporary directories.  It does not import
the generator or inspect a directory of unit states as execution history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
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
    G8BlerWorkUnitError,
    StateConflictError,
    StaleWriterError,
    build_shard_plan,
    build_unit_state,
    canonical_state_bytes,
    create_unit_state_exclusive,
    read_unit_state,
    replace_unit_state,
    state_identity_digest,
    unit_state_path,
    validate_shard_plan,
    validate_unit_state,
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


class G8BlerStateContractError(RuntimeError):
    """The generated B2 contract is stale, malformed, or incomplete."""


def _fail(message: str) -> None:
    raise G8BlerStateContractError(message)


def _source_bindings() -> list[dict[str, Any]]:
    bindings = []
    for relative in CONTRACT_SOURCE_PATHS:
        target = REPO_ROOT / relative
        try:
            body = target.read_bytes()
        except OSError as exc:
            _fail(f"cannot read bound B2 source {relative}: {exc}")
        bindings.append(
            {
                "path": relative,
                "role": CONTRACT_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
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


def _expected_without_id(context: AuthenticatedExecutionContext) -> dict[str, Any]:
    """Independent reconstruction; this intentionally does not call the generator."""

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


def _contract_identifier(payload: dict[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"cannot read B2 state contract {path}: {exc}")
    if not isinstance(payload, dict):
        _fail("B2 state contract is not a JSON object")
    if raw != rendered_json(payload):
        _fail("B2 state contract is not canonical rendered JSON")
    return payload, raw


def _verify_required_identity_order(context: AuthenticatedExecutionContext) -> None:
    try:
        raw = (REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes()
        artifact = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"cannot independently read required-BLER identities: {exc}")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_REQUIRED_IDENTITIES_SHA256:
        _fail("required-BLER artifact SHA-256 is not the corrected B1C authority")
    work_units = artifact.get("required_bler_work_units") if isinstance(artifact, dict) else None
    if not isinstance(work_units, list) or len(work_units) != EXPECTED_REQUIRED_WORK_UNIT_COUNT:
        _fail("required-BLER artifact does not contain exactly 3,213 work units")
    ids = []
    for unit in work_units:
        if not isinstance(unit, dict) or not isinstance(unit.get("work_unit_id"), str):
            _fail("required-BLER artifact contains a malformed work-unit record")
        ids.append(unit["work_unit_id"])
        if sha256_bytes(canonical_json(unit)) != context.work_unit_record_sha256(unit["work_unit_id"]):
            _fail("context record bytes do not reproduce an artifact work-unit record")
    if ids != list(context.ordered_work_unit_ids):
        _fail("context order differs from the exact required_bler_work_units artifact order")


def _verify_sharding_independently(context: AuthenticatedExecutionContext) -> None:
    ids = list(context.ordered_work_unit_ids)
    for count in (1, 2, 3, 7, 11, 32, 64, 127, 3213):
        seen: list[str] = []
        for index in range(count):
            manual = [work_unit_id for ordinal, work_unit_id in enumerate(ids) if ordinal % count == index]
            plan = build_shard_plan(context, count, index)
            if plan["assigned_work_unit_ids"] != manual:
                _fail(f"ordinal-modulo partition mismatch at {count}/{index}")
            if validate_shard_plan(context, plan) != plan:
                _fail("generated shard plan does not validate")
            body = dict(plan)
            digest = body.pop("plan_digest")
            if digest != hashlib.sha256(canonical_json(body)).hexdigest():
                _fail("shard plan digest rule does not reproduce")
            seen.extend(manual)
        if len(seen) != len(set(seen)) or set(seen) != set(ids):
            _fail(f"shard partition is not complete and disjoint at count {count}")


def _verify_path_derivation(context: AuthenticatedExecutionContext) -> None:
    for work_unit_id in (context.work_unit_ids[0], context.work_unit_ids[1606], context.work_unit_ids[-1]):
        digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
        relative = Path(digest[:2]) / f"{digest}.state.json"
        path = unit_state_path(context, work_unit_id)
        if path.relative_to(REPO_ROOT / "results/baseline/g8/work_units") != relative:
            _fail("unit-state path does not reproduce the SHA-256 digest layout")


def _verify_state_rules_and_publication(context: AuthenticatedExecutionContext) -> None:
    with tempfile.TemporaryDirectory(prefix=".g8-b2-contract-verify-") as raw_root:
        root = Path(raw_root)
        plan = build_shard_plan(context, 7, 0)
        work_unit_id = plan["assigned_work_unit_ids"][0]
        claim = build_unit_state(context, work_unit_id, plan)
        if validate_unit_state(context, claim) != claim:
            _fail("valid pre-execution claim does not validate")
        if state_identity_digest(claim) != claim["identity_sha256"]:
            _fail("state identity digest does not reproduce independently")
        annotated = build_unit_state(
            context,
            work_unit_id,
            plan,
            runtime_metadata={
                "hostname": "verifier",
                "process_id": 1,
                "device": "cpu",
                "wall_clock_annotation": "annotation",
                "update_annotation": "annotation",
            },
        )
        if state_identity_digest(annotated) != state_identity_digest(claim):
            _fail("runtime metadata changed the state identity digest")
        path = unit_state_path(context, work_unit_id, root=root)
        first_sha = create_unit_state_exclusive(context, claim, root=root)
        try:
            create_unit_state_exclusive(context, claim, root=root)
        except StateConflictError:
            pass
        else:
            _fail("exclusive create accepted two creators")
        failed = build_unit_state(context, work_unit_id, plan, status=STATUS_FAILED)
        second_sha = replace_unit_state(context, path, failed, first_sha, root=root)
        if second_sha != hashlib.sha256(canonical_state_bytes(context, failed)).hexdigest():
            _fail("atomic replacement returned the wrong installed SHA-256")
        try:
            replace_unit_state(context, path, claim, first_sha, root=root)
        except StaleWriterError:
            pass
        else:
            _fail("stale writer was accepted")
        if read_unit_state(context, path, root=root) != failed:
            _fail("installed state cannot be reread as one canonical snapshot")


def _verify_no_live_state() -> None:
    live_root = REPO_ROOT / "results/baseline/g8/work_units"
    if live_root.exists():
        _fail("B2 contract verification found a live work-unit state tree")
    result = subprocess.run(
        ["git", "ls-files", "results/baseline/g8/work_units"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        _fail("tracked per-unit state files exist during B2")


def verify(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    payload, _raw = _load(path)
    if set(payload) != {
        "schema_version",
        "artifact_role",
        "campaign",
        "phase",
        "checkpoint",
        "contract_id",
        "contract_sources",
        "authority_bindings",
        "execution_context",
        "sharding",
        "unit_state_path",
        "unit_state_schema",
        "publication",
        "scope",
    }:
        _fail("B2 state contract has missing or unknown top-level fields")
    if payload["schema_version"] != CONTRACT_SCHEMA_VERSION or payload["artifact_role"] != CONTRACT_ARTIFACT_ROLE:
        _fail("B2 state contract schema or artifact role is wrong")
    if payload["campaign"] != "G-8" or payload["phase"] != PHASE or payload["checkpoint"] != CHECKPOINT:
        _fail("B2 state contract campaign/phase/checkpoint binding is wrong")
    if payload["contract_id"] != _contract_identifier(payload):
        _fail("B2 state contract ID does not reproduce")

    sources = payload["contract_sources"]
    if not isinstance(sources, list) or [entry.get("path") for entry in sources] != list(CONTRACT_SOURCE_PATHS):
        _fail("B2 contract source path list changed")
    if sources != _source_bindings():
        _fail("B2 bound source bytes or SHA-256 values changed")
    if any(Path(entry["path"]).is_absolute() for entry in sources):
        _fail("B2 contract binds an absolute source path")
    if str(path) == str(CONTRACT_PATH) and any(entry["path"] == str(path.relative_to(REPO_ROOT)) for entry in sources):
        _fail("B2 contract binds its own output")

    context = AuthenticatedExecutionContext()
    expected_authority = _authority(context)
    if payload["authority_bindings"] != expected_authority:
        _fail("B2 authority bindings do not match the authenticated context")
    if expected_authority != {
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": EXPECTED_CAMPAIGN_MANIFEST_SHA256,
        "required_bler_artifact_sha256": EXPECTED_REQUIRED_IDENTITIES_SHA256,
        "selection_policy_sha256": EXPECTED_SELECTION_POLICY_SHA256,
        "bler_tooling_contract_id": EXPECTED_B1C_CONTRACT_ID,
        "bler_tooling_contract_sha256": EXPECTED_B1C_CONTRACT_SHA256,
        "tooling_schema_version": B1C_TOOLING_SCHEMA_VERSION,
        "request_schema_version": B1C_REQUEST_SCHEMA_VERSION,
        "result_schema_version": B1C_RESULT_SCHEMA_VERSION,
        "required_work_unit_count": EXPECTED_REQUIRED_WORK_UNIT_COUNT,
    }:
        _fail("B2 authority bindings are not the immutable B1C values")
    if g8_bler_contract.TOOLING_CONTRACT_ARTIFACT.read_bytes() is None:  # pragma: no cover
        _fail("unreachable tooling artifact check")

    expected = _expected_without_id(context)
    if {key: value for key, value in payload.items() if key != "contract_id"} != expected:
        _fail("B2 contract nested schema/content differs from independent reconstruction")

    _verify_required_identity_order(context)
    _verify_sharding_independently(context)
    _verify_path_derivation(context)
    _verify_state_rules_and_publication(context)
    _verify_no_live_state()

    scope = payload["scope"]
    if scope["scientific_execution_performed"] is not False or scope["test_split_access"] != 0:
        _fail("B2 contract scope claims scientific execution or test access")
    if scope["tracked_live_unit_state_files"] != [] or scope["exact_resume_and_merge_checkpoint"] != "B3":
        _fail("B2 contract scope boundary changed")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    try:
        payload = verify(args.path)
    except (G8BlerStateContractError, G8BlerWorkUnitError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"G8 B2 state contract HOLD: {exc}") from exc
    print(
        "G8 B2 state contract PASS: "
        f"contract_id={payload['contract_id']}, required_work_units="
        f"{payload['authority_bindings']['required_work_unit_count']}, "
        "algorithm=canonical_ordinal_modulo_v1, science=false, test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
