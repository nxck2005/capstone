"""Authenticated G8_B work-unit, shard-plan, and unit-state primitives.

This module is deliberately infrastructure-only.  It authenticates the frozen
B1C authority, partitions its canonical work-unit sequence, and provides local
filesystem primitives for one state snapshot.  It does not execute a work
unit, create a request or result, inspect a state directory, or decide resume
or merge policy; those decisions belong to later G8 checkpoints.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from baseline import g8_bler_contract as bler_contract
from baseline.g8_campaign import (
    CAMPAIGN,
    CAMPAIGN_MANIFEST,
    REQUIRED_BLER_IDENTITIES,
    canonical_json,
    rendered_json,
    sha256_bytes,
    sha256_file,
)
from config.params import REPO_ROOT


# ---------------------------------------------------------------------------
# Frozen B1C bindings and local schema constants
# ---------------------------------------------------------------------------

PHASE = "G8_B"
CHECKPOINT = "B2"
CAMPAIGN_ROLE = "G-8"

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
EXPECTED_REQUIRED_WORK_UNIT_COUNT = 3213

B1C_TOOLING_SCHEMA_VERSION = 2
B1C_REQUEST_SCHEMA_VERSION = 2
B1C_RESULT_SCHEMA_VERSION = 2

SHARDING_ALGORITHM = "canonical_ordinal_modulo_v1"
SHARD_PLAN_SCHEMA_VERSION = 1
SHARD_PLAN_ARTIFACT_ROLE = "g8_bler_shard_plan"
SHARD_PLAN_FIELDS = (
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
SHARD_FORMULA = "ordinal % shard_count == shard_index"
SHARD_PLAN_DIGEST_RULE = (
    "sha256(canonical JSON over the complete shard plan identity excluding plan_digest)"
)

UNIT_STATE_SCHEMA_VERSION = 1
UNIT_STATE_ARTIFACT_ROLE = "g8_bler_work_unit_state"
UNIT_STATE_FIELDS = (
    "schema_version",
    "artifact_role",
    "identity",
    "runtime_metadata",
    "identity_sha256",
)
UNIT_STATE_IDENTITY_FIELDS = (
    "schema_version",
    "artifact_role",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
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
UNIT_STATE_RUNTIME_METADATA_FIELDS = (
    "hostname",
    "process_id",
    "device",
    "wall_clock_annotation",
    "update_annotation",
)

STATUS_CLAIMED = "claimed"
STATUS_FAILED = "failed"
STATUS_RESULT_LINKED = "result_linked"
STATE_STATUSES = (STATUS_CLAIMED, STATUS_FAILED, STATUS_RESULT_LINKED)

DEFAULT_WORK_UNIT_ROOT = REPO_ROOT / "results/baseline/g8/work_units"
STATE_FILENAME_SUFFIX = ".state.json"
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

B3_RESTART_COMMAND = (
    'rg -n "resume|remaining_work|merge|request_sha256|result_sha256|'
    'completed_work_unit_ids|in_progress_work_unit_id|unit_state|shard_plan" '
    "src/baseline tools tests"
)


class G8BlerWorkUnitError(RuntimeError):
    """Base class for fail-closed B2 authority and state errors."""


class AuthorityAuthenticationError(G8BlerWorkUnitError):
    """The frozen campaign or B1C authority could not be authenticated."""


class ShardPlanError(G8BlerWorkUnitError):
    """A shard argument or persisted shard plan is invalid."""


class UnsafeUnitStatePathError(G8BlerWorkUnitError):
    """A unit-state path is not the one safe canonical path for its ID."""


class UnitStateError(G8BlerWorkUnitError):
    """A unit-state snapshot is malformed or violates local invariants."""


class StateConflictError(UnitStateError):
    """Exclusive initial creation lost a race or found an existing state."""


class StateNotFoundError(UnitStateError):
    """An atomic replacement target does not exist."""


class StaleWriterError(StateConflictError):
    """An atomic replacement observed a different previous state digest."""


class AtomicStateError(UnitStateError):
    """A filesystem operation failed without silently repairing state."""


def _require(condition: bool, message: str, error: type[Exception] = UnitStateError) -> None:
    if not condition:
        raise error(message)


def _exact_int(value: Any, name: str, error: type[Exception] = UnitStateError) -> int:
    if type(value) is not int:  # bool is intentionally not an integer here.
        raise error(f"{name} must be an exact integer")
    return value


def _positive_int(value: Any, name: str, error: type[Exception] = UnitStateError) -> int:
    value = _exact_int(value, name, error)
    if value <= 0:
        raise error(f"{name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str, error: type[Exception] = UnitStateError) -> int:
    value = _exact_int(value, name, error)
    if value < 0:
        raise error(f"{name} must be non-negative")
    return value


def _nonblank_string(value: Any, name: str, error: type[Exception] = UnitStateError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a nonblank string")
    return value


def _digest(value: Any, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or HEX_DIGEST_RE.fullmatch(value) is None:
        raise UnitStateError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _canonical_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UnitStateError(f"{name} is not finite canonical JSON") from exc


def _same_keys(value: Any, expected: Sequence[str], name: str, error: type[Exception]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise error(f"{name} has missing or unknown fields")


def _authority_expected() -> dict[str, Any]:
    return {
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
    }


class AuthenticatedExecutionContext:
    """Process-scoped immutable access to the authenticated B1C authority.

    Construction deliberately clears the B1C byte caches before invoking its
    public loader.  That makes each context a fresh authentication boundary;
    the loader then populates its immutable canonical-byte cache exactly once.
    Subsequent record, ordinal, seed, and shard lookups never reread or
    rehash the 8.6 MB required-identity file.
    """

    __slots__ = (
        "_authority",
        "_record_bytes",
        "_ordered_work_unit_ids",
        "_ordinals",
        "_tooling_contract_path",
    )

    def __init__(self, *, tooling_contract_path: Path | None = None) -> None:
        try:
            campaign_cache_clear = getattr(bler_contract._campaign_binding_bytes, "cache_clear", None)
            if campaign_cache_clear is not None:
                campaign_cache_clear()
            required_cache_clear = getattr(bler_contract._required_work_unit_bytes, "cache_clear", None)
            if required_cache_clear is not None:
                required_cache_clear()

            artifact = (
                bler_contract.TOOLING_CONTRACT_ARTIFACT
                if tooling_contract_path is None
                else Path(tooling_contract_path)
            )
            tooling_payload = bler_contract.load_bler_tooling_contract(artifact)
            tooling_sha256 = sha256_file(artifact)
            campaign_binding = dict(bler_contract.campaign_bindings())
            record_bytes = bler_contract._required_work_unit_bytes()
        except Exception as exc:
            if isinstance(exc, G8BlerWorkUnitError):
                raise
            raise AuthorityAuthenticationError(
                f"cannot authenticate the corrected B1C authority: {exc}"
            ) from exc

        expected = _authority_expected()
        actual_campaign = {
            "campaign_id": campaign_binding.get("campaign_id"),
            "campaign_manifest_sha256": campaign_binding.get("campaign_manifest_sha256"),
            "required_bler_artifact_sha256": campaign_binding.get("required_bler_artifact_sha256"),
            "selection_policy_sha256": campaign_binding.get("selection_policy_sha256"),
        }
        if actual_campaign != {
            key: expected[key]
            for key in (
                "campaign_id",
                "campaign_manifest_sha256",
                "required_bler_artifact_sha256",
                "selection_policy_sha256",
            )
        }:
            raise AuthorityAuthenticationError("campaign or artifact binding differs from B1C authority")
        if tooling_payload.get("contract_id") != expected["bler_tooling_contract_id"]:
            raise AuthorityAuthenticationError("B1C tooling-contract ID differs from immutable authority")
        if tooling_sha256 != expected["bler_tooling_contract_sha256"]:
            raise AuthorityAuthenticationError("B1C tooling-contract SHA-256 differs from immutable authority")
        if tooling_payload.get("schema_version") != expected["tooling_schema_version"]:
            raise AuthorityAuthenticationError("B1C tooling-contract schema version is not 2")
        if tooling_payload.get("request_schema", {}).get("version") != expected["request_schema_version"]:
            raise AuthorityAuthenticationError("B1C request schema version is not 2")
        if tooling_payload.get("result_schema", {}).get("version") != expected["result_schema_version"]:
            raise AuthorityAuthenticationError("B1C result schema version is not 2")
        if tooling_payload.get("phase") != PHASE or tooling_payload.get("checkpoint") != "B1C":
            raise AuthorityAuthenticationError("B1C tooling contract is not the corrected G8_B/B1C contract")
        tooling_bindings = tooling_payload.get("campaign_bindings")
        if (
            not isinstance(tooling_bindings, Mapping)
            or tooling_bindings.get("required_work_unit_count")
            != EXPECTED_REQUIRED_WORK_UNIT_COUNT
        ):
            raise AuthorityAuthenticationError("B1C tooling contract required work-unit count changed")

        if not isinstance(record_bytes, Mapping):
            raise AuthorityAuthenticationError("B1C required-work-unit cache is not a mapping")
        if len(record_bytes) != EXPECTED_REQUIRED_WORK_UNIT_COUNT:
            raise AuthorityAuthenticationError("required work-unit count differs from B1C authority")

        ordered_ids: list[str] = []
        immutable_records: dict[str, bytes] = {}
        for work_unit_id, raw_record in record_bytes.items():
            if not isinstance(work_unit_id, str) or not work_unit_id:
                raise AuthorityAuthenticationError("required work-unit ID is malformed")
            if not isinstance(raw_record, bytes):
                raise AuthorityAuthenticationError("required work-unit cache is not canonical bytes")
            try:
                record = json.loads(raw_record)
                canonical_record = canonical_json(record)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise AuthorityAuthenticationError(
                    f"required work-unit {work_unit_id!r} is not canonical JSON"
                ) from exc
            if canonical_record != raw_record:
                raise AuthorityAuthenticationError(
                    f"required work-unit {work_unit_id!r} is not canonically represented"
                )
            if not isinstance(record, Mapping) or record.get("work_unit_id") != work_unit_id:
                raise AuthorityAuthenticationError(
                    f"required work-unit record does not bind its ID: {work_unit_id!r}"
                )
            ordered_ids.append(work_unit_id)
            immutable_records[work_unit_id] = bytes(raw_record)

        self._authority = MappingProxyType(
            {
                **expected,
                "required_work_unit_count": len(ordered_ids),
            }
        )
        self._record_bytes = MappingProxyType(immutable_records)
        self._ordered_work_unit_ids = tuple(ordered_ids)
        self._ordinals = MappingProxyType(
            {work_unit_id: ordinal for ordinal, work_unit_id in enumerate(ordered_ids)}
        )
        self._tooling_contract_path = artifact

    @property
    def campaign_id(self) -> str:
        return self._authority["campaign_id"]

    @property
    def required_work_unit_count(self) -> int:
        return self._authority["required_work_unit_count"]

    @property
    def tooling_contract_path(self) -> Path:
        return self._tooling_contract_path

    def authority_binding(self) -> dict[str, Any]:
        """Return a fresh scalar binding; internal authority is never exposed."""

        return dict(self._authority)

    def campaign_binding(self) -> dict[str, str]:
        return {
            key: self._authority[key]
            for key in (
                "campaign_id",
                "campaign_manifest_sha256",
                "required_bler_artifact_sha256",
                "selection_policy_sha256",
            )
        }

    def tooling_binding(self) -> dict[str, str]:
        return {
            "bler_tooling_contract_id": self._authority["bler_tooling_contract_id"],
            "bler_tooling_contract_sha256": self._authority["bler_tooling_contract_sha256"],
        }

    @property
    def ordered_work_unit_ids(self) -> tuple[str, ...]:
        return self._ordered_work_unit_ids

    @property
    def work_unit_ids(self) -> tuple[str, ...]:
        return self._ordered_work_unit_ids

    def ordinal(self, work_unit_id: str) -> int:
        _nonblank_string(work_unit_id, "work_unit_id", AuthorityAuthenticationError)
        try:
            return self._ordinals[work_unit_id]
        except KeyError as exc:
            raise AuthorityAuthenticationError(
                f"work unit {work_unit_id!r} is not an exact required BLER identity"
            ) from exc

    def work_unit_record_bytes(self, work_unit_id: str) -> bytes:
        self.ordinal(work_unit_id)
        return self._record_bytes[work_unit_id]

    def work_unit_record_sha256(self, work_unit_id: str) -> str:
        return sha256_bytes(self.work_unit_record_bytes(work_unit_id))

    def work_unit_record(self, work_unit_id: str) -> dict[str, Any]:
        """Return a fresh decoded record; mutation cannot affect later lookups."""

        try:
            return json.loads(self.work_unit_record_bytes(work_unit_id))
        except json.JSONDecodeError as exc:  # pragma: no cover - authenticated at construction
            raise AuthorityAuthenticationError("cached work-unit bytes became invalid") from exc

    def work_unit_index(self) -> dict[str, dict[str, Any]]:
        """Return a fresh index without rereading the required artifact."""

        return {
            work_unit_id: json.loads(raw)
            for work_unit_id, raw in self._record_bytes.items()
        }

    def seed(self, work_unit_id: str, purpose: str) -> int:
        self.ordinal(work_unit_id)
        return bler_contract.derive_seed(self.campaign_id, work_unit_id, purpose)

    def stream_seed_records(self, work_unit_id: str) -> dict[str, dict[str, Any]]:
        self.ordinal(work_unit_id)
        return json.loads(canonical_json(bler_contract.stream_seed_records(self.campaign_id, work_unit_id)))


def _context(value: AuthenticatedExecutionContext) -> AuthenticatedExecutionContext:
    if not isinstance(value, AuthenticatedExecutionContext):
        raise TypeError("context must be an AuthenticatedExecutionContext")
    return value


# ---------------------------------------------------------------------------
# Deterministic ordinal-modulo sharding
# ---------------------------------------------------------------------------


def validate_shard_arguments(shard_count: Any, shard_index: Any) -> tuple[int, int]:
    count = _positive_int(shard_count, "shard_count", ShardPlanError)
    index = _exact_int(shard_index, "shard_index", ShardPlanError)
    if index < 0 or index >= count:
        raise ShardPlanError("shard_index must satisfy 0 <= shard_index < shard_count")
    return count, index


def _shard_plan_body(
    context: AuthenticatedExecutionContext,
    shard_count: int,
    shard_index: int,
    assigned_ids: list[str],
) -> dict[str, Any]:
    authority = context.authority_binding()
    return {
        "schema_version": SHARD_PLAN_SCHEMA_VERSION,
        "artifact_role": SHARD_PLAN_ARTIFACT_ROLE,
        "sharding_algorithm": SHARDING_ALGORITHM,
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "total_required_work_unit_count": context.required_work_unit_count,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_work_unit_ids": list(assigned_ids),
        "assigned_work_unit_count": len(assigned_ids),
    }


def _with_plan_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["plan_digest"] = sha256_bytes(canonical_json(result))
    return json.loads(canonical_json(result))


def build_shard_plan(
    context: AuthenticatedExecutionContext,
    shard_count: Any,
    shard_index: Any,
) -> dict[str, Any]:
    context = _context(context)
    count, index = validate_shard_arguments(shard_count, shard_index)
    assigned = [
        work_unit_id
        for ordinal, work_unit_id in enumerate(context.ordered_work_unit_ids)
        if ordinal % count == index
    ]
    return _with_plan_digest(_shard_plan_body(context, count, index, assigned))


def validate_shard_plan(
    context: AuthenticatedExecutionContext,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    context = _context(context)
    if not isinstance(plan, Mapping):
        raise ShardPlanError("shard plan must be a mapping")
    payload = _canonical_copy(dict(plan), "shard plan")
    _same_keys(payload, SHARD_PLAN_FIELDS, "shard plan", ShardPlanError)
    if payload["schema_version"] != SHARD_PLAN_SCHEMA_VERSION:
        raise ShardPlanError("unsupported shard-plan schema_version")
    if payload["artifact_role"] != SHARD_PLAN_ARTIFACT_ROLE:
        raise ShardPlanError("shard plan has the wrong artifact role")
    if payload["sharding_algorithm"] != SHARDING_ALGORITHM:
        raise ShardPlanError("shard plan uses an unknown sharding algorithm")
    count, index = validate_shard_arguments(payload["shard_count"], payload["shard_index"])
    expected_authority = context.authority_binding()
    for field in (
        "campaign_id",
        "campaign_manifest_sha256",
        "required_bler_artifact_sha256",
        "selection_policy_sha256",
        "bler_tooling_contract_id",
        "bler_tooling_contract_sha256",
    ):
        if payload[field] != expected_authority[field]:
            raise ShardPlanError(f"shard plan binding changed: {field}")
    if payload["total_required_work_unit_count"] != context.required_work_unit_count:
        raise ShardPlanError("shard plan total work-unit count changed")
    assigned = payload["assigned_work_unit_ids"]
    if not isinstance(assigned, list) or any(not isinstance(item, str) for item in assigned):
        raise ShardPlanError("assigned work-unit IDs are malformed")
    expected_assigned = [
        work_unit_id
        for ordinal, work_unit_id in enumerate(context.ordered_work_unit_ids)
        if ordinal % count == index
    ]
    if assigned != expected_assigned:
        raise ShardPlanError("assigned IDs do not reproduce canonical ordinal-modulo membership")
    if payload["assigned_work_unit_count"] != len(assigned):
        raise ShardPlanError("assigned work-unit count does not reproduce")
    supplied_digest = _digest(payload["plan_digest"], "plan_digest")
    body = dict(payload)
    body.pop("plan_digest")
    if supplied_digest != sha256_bytes(canonical_json(body)):
        raise ShardPlanError("shard plan digest does not reproduce")
    return json.loads(canonical_json(payload))


def shard_plan_digest(plan: Mapping[str, Any]) -> str:
    if not isinstance(plan, Mapping):
        raise ShardPlanError("shard plan must be a mapping")
    body = dict(plan)
    supplied = body.pop("plan_digest", None)
    _digest(supplied, "plan_digest")
    return sha256_bytes(canonical_json(body))


def shard_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    return canonical_json(plan)


# ---------------------------------------------------------------------------
# Safe deterministic unit-state paths
# ---------------------------------------------------------------------------


def _root_path(root: Path | str | None) -> Path:
    value = DEFAULT_WORK_UNIT_ROOT if root is None else Path(root)
    if not value.is_absolute():
        raise UnsafeUnitStatePathError("work-unit root must be absolute")
    if value.exists() and value.is_symlink():
        raise UnsafeUnitStatePathError("work-unit root may not be a symlink")
    if value.exists() and not value.is_dir():
        raise UnsafeUnitStatePathError("work-unit root is not a directory")
    return value


def unit_state_relative_path(
    context: AuthenticatedExecutionContext,
    work_unit_id: str,
) -> Path:
    context = _context(context)
    context.ordinal(work_unit_id)
    digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
    return Path(digest[:2]) / f"{digest}{STATE_FILENAME_SUFFIX}"


def unit_state_path(
    context: AuthenticatedExecutionContext,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    return _root_path(root) / unit_state_relative_path(context, work_unit_id)


def state_path_for_work_unit(
    context: AuthenticatedExecutionContext,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    return unit_state_path(context, work_unit_id, root=root)


def _validate_candidate_path_shape(
    path: Path | str,
    *,
    root: Path | str | None,
) -> tuple[Path, Path]:
    root_path = _root_path(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise UnsafeUnitStatePathError("unit-state paths must be absolute canonical paths")
    raw = str(path)
    if "\\" in raw or os.path.normpath(raw) != raw:
        raise UnsafeUnitStatePathError("unit-state path is not normalized POSIX canonical form")
    if any(part in {".", ".."} for part in candidate.parts):
        raise UnsafeUnitStatePathError("unit-state path contains traversal or alias components")
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeUnitStatePathError("unit-state path is outside the G8 work-unit root") from exc
    parts = relative.parts
    if len(parts) != 2:
        raise UnsafeUnitStatePathError("unit-state path has the wrong directory layout")
    bucket, filename = parts
    if re.fullmatch(r"[0-9a-f]{2}", bucket) is None:
        raise UnsafeUnitStatePathError("unit-state bucket is not lowercase two-digit hex")
    if not filename.endswith(STATE_FILENAME_SUFFIX):
        raise UnsafeUnitStatePathError("unit-state path has the wrong extension")
    digest = filename[: -len(STATE_FILENAME_SUFFIX)]
    if HEX_DIGEST_RE.fullmatch(digest) is None or digest[:2] != bucket:
        raise UnsafeUnitStatePathError("unit-state filename is not a lowercase SHA-256 digest")
    root_resolved = root_path.resolve(strict=False)
    parent_resolved = candidate.parent.resolve(strict=False)
    try:
        if os.path.commonpath((str(root_resolved), str(parent_resolved))) != str(root_resolved):
            raise UnsafeUnitStatePathError("unit-state parent resolves outside the G8 work-unit root")
    except ValueError as exc:
        raise UnsafeUnitStatePathError("unit-state parent has an incompatible filesystem root") from exc
    if candidate.exists() and candidate.is_symlink():
        raise UnsafeUnitStatePathError("unit-state path may not be a symlink")
    if candidate.resolve(strict=False) != (root_resolved / relative).resolve(strict=False):
        raise UnsafeUnitStatePathError("unit-state path resolves through an unsafe alias")
    return root_path, candidate


def validate_unit_state_path(
    context: AuthenticatedExecutionContext,
    path: Path | str,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    context = _context(context)
    expected = unit_state_path(context, work_unit_id, root=root)
    root_path, candidate = _validate_candidate_path_shape(path, root=root)
    if str(candidate) != str(expected):
        raise UnsafeUnitStatePathError("unit-state path digest does not correspond to the work-unit ID")
    if root_path != _root_path(root):  # pragma: no cover - defensive Path subclass guard
        raise UnsafeUnitStatePathError("unit-state root changed during validation")
    expected_digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
    if candidate.name[: -len(STATE_FILENAME_SUFFIX)] != expected_digest:
        raise UnsafeUnitStatePathError("unit-state path carries a different work-unit digest")
    return candidate


# ---------------------------------------------------------------------------
# Closed per-unit state schema and identity digest
# ---------------------------------------------------------------------------


def _default_runtime_metadata() -> dict[str, Any]:
    return {field: None for field in UNIT_STATE_RUNTIME_METADATA_FIELDS}


def _validate_runtime_metadata(value: Any) -> dict[str, Any]:
    _same_keys(value, UNIT_STATE_RUNTIME_METADATA_FIELDS, "runtime metadata", UnitStateError)
    payload = dict(value)
    for field in ("hostname", "device", "wall_clock_annotation", "update_annotation"):
        item = payload[field]
        if item is not None:
            _nonblank_string(item, f"runtime_metadata.{field}")
    process_id = payload["process_id"]
    if process_id is not None:
        _positive_int(process_id, "runtime_metadata.process_id")
    return json.loads(canonical_json(payload))


def _validate_result_path(value: Any) -> str:
    _nonblank_string(value, "result_path")
    if Path(value).is_absolute() or "\\" in value:
        raise UnitStateError("result_path must be a canonical repository-relative POSIX path")
    if value != PurePosixPath(value).as_posix() or any(
        part in {"", ".", ".."} for part in PurePosixPath(value).parts
    ):
        raise UnitStateError("result_path is not canonical")
    return value


def _validate_state_shape(state: Any) -> dict[str, Any]:
    _same_keys(state, UNIT_STATE_FIELDS, "unit state", UnitStateError)
    payload = _canonical_copy(dict(state), "unit state")
    if payload["schema_version"] != UNIT_STATE_SCHEMA_VERSION:
        raise UnitStateError("unsupported unit-state schema_version")
    if payload["artifact_role"] != UNIT_STATE_ARTIFACT_ROLE:
        raise UnitStateError("unit state has the wrong artifact role")
    _same_keys(payload["identity"], UNIT_STATE_IDENTITY_FIELDS, "unit-state identity", UnitStateError)
    _validate_runtime_metadata(payload["runtime_metadata"])
    _digest(payload["identity_sha256"], "identity_sha256")
    return payload


def _recomputed_identity_digest(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(identity)))


def state_identity_digest(state: Mapping[str, Any]) -> str:
    payload = _validate_state_shape(state)
    return _recomputed_identity_digest(payload["identity"])


def _validate_state_against_context(
    context: AuthenticatedExecutionContext,
    payload: dict[str, Any],
) -> dict[str, Any]:
    authority = context.authority_binding()
    identity = payload["identity"]
    if identity["schema_version"] != UNIT_STATE_SCHEMA_VERSION:
        raise UnitStateError("unit-state identity schema version changed")
    if identity["artifact_role"] != UNIT_STATE_ARTIFACT_ROLE:
        raise UnitStateError("unit-state identity artifact role changed")
    for field in (
        "campaign_id",
        "campaign_manifest_sha256",
        "required_bler_artifact_sha256",
        "selection_policy_sha256",
        "bler_tooling_contract_id",
        "bler_tooling_contract_sha256",
    ):
        if identity[field] != authority[field]:
            raise UnitStateError(f"unit-state authority binding changed: {field}")
    if identity["request_schema_version"] != authority["request_schema_version"]:
        raise UnitStateError("unit-state request schema version is not the corrected v2 schema")
    if identity["result_schema_version"] != authority["result_schema_version"]:
        raise UnitStateError("unit-state result schema version is not the corrected v2 schema")
    work_unit_id = identity["work_unit_id"]
    _nonblank_string(work_unit_id, "identity.work_unit_id")
    ordinal = context.ordinal(work_unit_id)
    if identity["canonical_ordinal"] != ordinal:
        raise UnitStateError("unit-state canonical ordinal does not reproduce")
    if identity["required_work_unit_record_sha256"] != context.work_unit_record_sha256(work_unit_id):
        raise UnitStateError("unit-state required work-unit record hash does not reproduce")
    if identity["sharding_algorithm"] != SHARDING_ALGORITHM:
        raise UnitStateError("unit-state uses an unknown sharding algorithm")
    count, index = validate_shard_arguments(identity["shard_count"], identity["shard_index"])
    expected_plan = build_shard_plan(context, count, index)
    if work_unit_id not in expected_plan["assigned_work_unit_ids"]:
        raise UnitStateError("unit state is owned by a different shard")
    if identity["shard_plan_digest"] != expected_plan["plan_digest"]:
        raise UnitStateError("unit-state shard-plan digest does not reproduce")

    _positive_int(identity["attempt"], "identity.attempt")
    if identity["status"] not in STATE_STATUSES:
        raise UnitStateError("unit state has an unknown status")
    _digest(identity["request_sha256"], "identity.request_sha256", allow_none=True)
    _digest(identity["result_sha256"], "identity.result_sha256", allow_none=True)
    if not isinstance(identity["result_path"], (str, type(None))):
        raise UnitStateError("identity.result_path must be a string or null")
    if identity["result_path"] is not None:
        _validate_result_path(identity["result_path"])
    if type(identity["scientific_execution_performed"]) is not bool:
        raise UnitStateError("scientific_execution_performed must be a boolean")
    _nonnegative_int(identity["trials_completed"], "identity.trials_completed")
    if type(identity["test_split_access"]) is not int or identity["test_split_access"] != 0:
        raise UnitStateError("test_split_access must be the exact integer zero")

    status = identity["status"]
    request_sha = identity["request_sha256"]
    result_path = identity["result_path"]
    result_sha = identity["result_sha256"]
    scientific = identity["scientific_execution_performed"]
    trials = identity["trials_completed"]
    if status == STATUS_CLAIMED:
        if (request_sha, result_path, result_sha, scientific, trials) != (None, None, None, False, 0):
            raise UnitStateError("pre-execution claim must have no execution or result fields")
    elif status == STATUS_FAILED:
        if result_path is not None or result_sha is not None:
            raise UnitStateError("failed state may not carry a result reference")
        if not scientific and trials != 0:
            raise UnitStateError("failed state with completed trials must mark execution performed")
    elif status == STATUS_RESULT_LINKED:
        if result_path is None or result_sha is None:
            raise UnitStateError("result-linked state requires canonical result path and SHA-256")
        if not scientific or trials <= 0:
            raise UnitStateError("result-linked state requires scientific execution and positive trials")

    if payload["identity_sha256"] != _recomputed_identity_digest(identity):
        raise UnitStateError("unit-state identity digest does not reproduce")
    return payload


def validate_unit_state(
    context: AuthenticatedExecutionContext,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    context = _context(context)
    return _validate_state_against_context(context, _validate_state_shape(state))


def build_unit_state(
    context: AuthenticatedExecutionContext,
    work_unit_id: str,
    shard_plan: Mapping[str, Any],
    *,
    attempt: Any = 1,
    status: str = STATUS_CLAIMED,
    request_sha256: str | None = None,
    result_path: str | None = None,
    result_sha256: str | None = None,
    scientific_execution_performed: bool = False,
    trials_completed: Any = 0,
    test_split_access: Any = 0,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = _context(context)
    plan = validate_shard_plan(context, shard_plan)
    context.ordinal(work_unit_id)
    if work_unit_id not in plan["assigned_work_unit_ids"]:
        raise UnitStateError("work unit is not owned by the supplied shard plan")
    _positive_int(attempt, "attempt")
    if not isinstance(status, str) or status not in STATE_STATUSES:
        raise UnitStateError("status is not a supported unit-state status")
    _digest(request_sha256, "request_sha256", allow_none=True)
    _digest(result_sha256, "result_sha256", allow_none=True)
    if result_path is not None:
        _validate_result_path(result_path)
    if type(scientific_execution_performed) is not bool:
        raise UnitStateError("scientific_execution_performed must be a boolean")
    _nonnegative_int(trials_completed, "trials_completed")
    if type(test_split_access) is not int or test_split_access != 0:
        raise UnitStateError("test_split_access must be the exact integer zero")
    runtime = _default_runtime_metadata() if runtime_metadata is None else _validate_runtime_metadata(runtime_metadata)
    authority = context.authority_binding()
    identity = {
        "schema_version": UNIT_STATE_SCHEMA_VERSION,
        "artifact_role": UNIT_STATE_ARTIFACT_ROLE,
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "request_schema_version": authority["request_schema_version"],
        "result_schema_version": authority["result_schema_version"],
        "work_unit_id": work_unit_id,
        "canonical_ordinal": context.ordinal(work_unit_id),
        "required_work_unit_record_sha256": context.work_unit_record_sha256(work_unit_id),
        "sharding_algorithm": SHARDING_ALGORITHM,
        "shard_index": plan["shard_index"],
        "shard_count": plan["shard_count"],
        "shard_plan_digest": plan["plan_digest"],
        "attempt": attempt,
        "status": status,
        "request_sha256": request_sha256,
        "result_path": result_path,
        "result_sha256": result_sha256,
        "scientific_execution_performed": scientific_execution_performed,
        "trials_completed": trials_completed,
        "test_split_access": test_split_access,
    }
    state = {
        "schema_version": UNIT_STATE_SCHEMA_VERSION,
        "artifact_role": UNIT_STATE_ARTIFACT_ROLE,
        "identity": identity,
        "runtime_metadata": runtime,
        "identity_sha256": _recomputed_identity_digest(identity),
    }
    return validate_unit_state(context, state)


def canonical_state_bytes(
    context: AuthenticatedExecutionContext,
    state: Mapping[str, Any],
) -> bytes:
    validated = validate_unit_state(context, state)
    try:
        return canonical_json(validated)
    except (TypeError, ValueError) as exc:  # pragma: no cover - validation already rejects these
        raise UnitStateError("unit state cannot be rendered as canonical JSON") from exc


def unit_state_sha256(
    context: AuthenticatedExecutionContext,
    state: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_state_bytes(context, state))


def _validate_existing_state(
    context: AuthenticatedExecutionContext,
    path: Path,
    *,
    root: Path | str | None,
) -> tuple[dict[str, Any], bytes, str]:
    _validate_candidate_path_shape(path, root=root)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise StateNotFoundError(f"unit-state file does not exist: {path}") from exc
    except OSError as exc:
        raise AtomicStateError(f"cannot read unit-state file {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UnitStateError(f"existing unit state is malformed JSON: {path}") from exc
    state = validate_unit_state(context, parsed)
    work_unit_id = state["identity"]["work_unit_id"]
    validate_unit_state_path(context, path, work_unit_id, root=root)
    expected_raw = canonical_state_bytes(context, state)
    if raw != expected_raw:
        raise UnitStateError("existing unit state is not canonical JSON")
    return state, raw, sha256_bytes(raw)


def _fsync_directory(path: Path) -> bool:
    """Fsync a directory when the platform supports it; propagate real errors."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS, errno.EACCES}:
            return False
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _safe_parent_for_write(path: Path, *, root: Path | str | None) -> Path:
    root_path, candidate = _validate_candidate_path_shape(path, root=root)
    parent = candidate.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AtomicStateError(f"cannot create unit-state parent directory {parent}: {exc}") from exc
    if parent.is_symlink() or parent.resolve(strict=False) != (root_path / candidate.relative_to(root_path).parent).resolve(strict=False):
        raise UnsafeUnitStatePathError("unit-state parent directory is an unsafe symlink or alias")
    if parent.resolve(strict=False).is_relative_to(root_path.resolve(strict=False)) is False:
        raise UnsafeUnitStatePathError("unit-state parent directory escaped the work-unit root")
    return parent


def create_unit_state_exclusive(
    context: AuthenticatedExecutionContext,
    state: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    path: Path | str | None = None,
) -> str:
    """Create one canonical state file with filesystem-exclusive semantics."""

    context = _context(context)
    validated = validate_unit_state(context, state)
    work_unit_id = validated["identity"]["work_unit_id"]
    target = unit_state_path(context, work_unit_id, root=root) if path is None else Path(path)
    validate_unit_state_path(context, target, work_unit_id, root=root)
    parent = _safe_parent_for_write(target, root=root)
    body = canonical_state_bytes(context, validated)
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags, stat.S_IRUSR | stat.S_IWUSR)
        except FileExistsError as exc:
            raise StateConflictError(f"unit-state already exists: {target}") from exc
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(parent)
    except StateConflictError:
        raise
    except Exception as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, G8BlerWorkUnitError):
            raise
        raise AtomicStateError(f"exclusive unit-state creation failed: {target}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    installed, installed_body, installed_sha = _validate_existing_state(context, target, root=root)
    if installed != validated or installed_body != body:
        raise AtomicStateError("exclusive creation did not install the requested canonical state")
    return installed_sha


def _validate_state_transition(previous: Mapping[str, Any], proposed: Mapping[str, Any]) -> None:
    old = previous["identity"]
    new = proposed["identity"]
    immutable_fields = (
        "schema_version",
        "artifact_role",
        "campaign_id",
        "campaign_manifest_sha256",
        "required_bler_artifact_sha256",
        "selection_policy_sha256",
        "bler_tooling_contract_id",
        "bler_tooling_contract_sha256",
        "request_schema_version",
        "result_schema_version",
        "work_unit_id",
        "canonical_ordinal",
        "required_work_unit_record_sha256",
        "sharding_algorithm",
        "shard_index",
        "shard_count",
        "shard_plan_digest",
    )
    for field in immutable_fields:
        if old[field] != new[field]:
            raise StateConflictError(f"unit-state immutable binding changed: {field}")
    if new["attempt"] < old["attempt"]:
        raise StateConflictError("unit-state attempt regressed")
    if old["status"] == STATUS_RESULT_LINKED and new["status"] != STATUS_RESULT_LINKED:
        raise StateConflictError("result-linked unit state cannot regress to another status")
    if old["status"] == STATUS_FAILED and new["status"] == STATUS_CLAIMED and new["attempt"] == old["attempt"]:
        raise StateConflictError("failed unit state cannot become a same-attempt claim")


def replace_unit_state(
    context: AuthenticatedExecutionContext,
    path: Path | str,
    proposed_state: Mapping[str, Any],
    expected_previous_sha256: str,
    *,
    root: Path | str | None = None,
) -> str:
    """Compare-and-swap one state using an fsynced same-directory replacement."""

    context = _context(context)
    expected = _digest(expected_previous_sha256, "expected_previous_sha256")
    target = Path(path)
    current, _current_body, current_sha = _validate_existing_state(context, target, root=root)
    if current_sha != expected:
        raise StaleWriterError(
            f"stale unit-state writer: expected {expected}, observed {current_sha}"
        )
    proposed = validate_unit_state(context, proposed_state)
    validate_unit_state_path(context, target, proposed["identity"]["work_unit_id"], root=root)
    _validate_state_transition(current, proposed)
    body = canonical_state_bytes(context, proposed)
    parent = target.parent
    descriptor: int | None = None
    temporary: Path | None = None
    replaced = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".partial", dir=parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        replaced = True
        temporary = None
        _fsync_directory(parent)
    except Exception as exc:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, G8BlerWorkUnitError):
            raise
        if replaced:
            raise AtomicStateError(
                f"unit-state replacement installed bytes but post-replace durability failed: {target}: {exc}"
            ) from exc
        raise AtomicStateError(f"unit-state replacement failed before publication: {target}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    _installed, installed_body, installed_sha = _validate_existing_state(context, target, root=root)
    if installed_body != body:
        raise AtomicStateError("installed unit state does not match the proposed canonical bytes")
    return installed_sha


def read_unit_state(
    context: AuthenticatedExecutionContext,
    path: Path | str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one snapshot only; never interpret a directory as history."""

    context = _context(context)
    _validate_candidate_path_shape(Path(path), root=root)
    try:
        raw = Path(path).read_bytes()
        parsed = json.loads(raw)
    except FileNotFoundError as exc:
        raise StateNotFoundError(f"unit-state file does not exist: {path}") from exc
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UnitStateError(f"unit-state file is malformed: {path}") from exc
    state = validate_unit_state(context, parsed)
    validate_unit_state_path(context, path, state["identity"]["work_unit_id"], root=root)
    if raw != canonical_state_bytes(context, state):
        raise UnitStateError("unit-state file is not canonical JSON")
    return state


# Explicit aliases make the two publication boundaries discoverable without
# introducing a second implementation or a resume/merge policy.
exclusive_create_unit_state = create_unit_state_exclusive
atomic_replace_unit_state = replace_unit_state


__all__ = [
    "AuthenticatedExecutionContext",
    "AtomicStateError",
    "AuthorityAuthenticationError",
    "B1C_REQUEST_SCHEMA_VERSION",
    "B1C_RESULT_SCHEMA_VERSION",
    "B1C_TOOLING_SCHEMA_VERSION",
    "B3_RESTART_COMMAND",
    "CHECKPOINT",
    "DEFAULT_WORK_UNIT_ROOT",
    "EXPECTED_B1C_CONTRACT_ID",
    "EXPECTED_B1C_CONTRACT_SHA256",
    "EXPECTED_CAMPAIGN_ID",
    "EXPECTED_CAMPAIGN_MANIFEST_SHA256",
    "EXPECTED_REQUIRED_IDENTITIES_SHA256",
    "EXPECTED_REQUIRED_WORK_UNIT_COUNT",
    "EXPECTED_SELECTION_POLICY_SHA256",
    "G8BlerWorkUnitError",
    "STATUS_CLAIMED",
    "STATUS_FAILED",
    "STATUS_RESULT_LINKED",
    "SHARDING_ALGORITHM",
    "SHARD_PLAN_ARTIFACT_ROLE",
    "SHARD_PLAN_FIELDS",
    "SHARD_PLAN_SCHEMA_VERSION",
    "SHARD_FORMULA",
    "ShardPlanError",
    "StaleWriterError",
    "StateConflictError",
    "StateNotFoundError",
    "UNIT_STATE_ARTIFACT_ROLE",
    "UNIT_STATE_FIELDS",
    "UNIT_STATE_IDENTITY_FIELDS",
    "UNIT_STATE_RUNTIME_METADATA_FIELDS",
    "UNIT_STATE_SCHEMA_VERSION",
    "UnitStateError",
    "UnsafeUnitStatePathError",
    "atomic_replace_unit_state",
    "build_shard_plan",
    "build_unit_state",
    "canonical_state_bytes",
    "create_unit_state_exclusive",
    "exclusive_create_unit_state",
    "read_unit_state",
    "replace_unit_state",
    "shard_plan_bytes",
    "shard_plan_digest",
    "state_identity_digest",
    "state_path_for_work_unit",
    "unit_state_path",
    "unit_state_relative_path",
    "unit_state_sha256",
    "validate_shard_arguments",
    "validate_shard_plan",
    "validate_unit_state",
    "validate_unit_state_path",
]
