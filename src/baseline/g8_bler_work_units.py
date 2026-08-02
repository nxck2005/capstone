"""Authenticated G8_B work-unit, shard-plan, and unit-state primitives.

This module is deliberately infrastructure-only.  It authenticates the frozen
B1C authority, partitions its canonical work-unit sequence, and provides local
filesystem primitives for one state snapshot.  It does not execute a work
unit, create a request or result, inspect a state directory, or decide resume
or merge policy; those decisions belong to later G8 checkpoints.

B2C corrects the B2 publication mechanics.  First publication is crash-atomic
(stage, fsync, descriptor-relative no-replace link, directory fsync);
replacement is linearizable inside one exclusive per-unit critical section;
descriptors are closed exactly once and never mask a domain error; symlink
guards are no-follow so dangling links are detected; directory durability
fails closed; results are request-bound and terminal; and every unit state
binds the registered B2C state contract in addition to the B1C tooling
contract.

Adversary model.  These primitives assume a trusted local user and a single
trusted checkout.  They defend against accident and concurrency — racing
workers, hard process kills mid-write, stale writers, orphaned staging files,
and stray symlinks or aliases left by a careless tool or a partial restore.
They do not claim to defeat a local attacker who already holds write access to
the work-unit root and can win an arbitrary time-of-check/time-of-use race.
Publication is descriptor-relative in order to narrow that window, not to
close it against an unbounded adversary.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from baseline import g8_bler_contract as bler_contract
from baseline.g8_campaign import (
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from config.params import REPO_ROOT


# ---------------------------------------------------------------------------
# Frozen B1C bindings and local schema constants
# ---------------------------------------------------------------------------

PHASE = "G8_B"
CHECKPOINT = "B2C"
SUPERSEDED_CHECKPOINT = "B2"
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
EXPECTED_REQUIRED_WORK_UNIT_COUNT = 3213  # literal-ok: frozen B1C required-identity count

B1C_TOOLING_SCHEMA_VERSION = 2
B1C_REQUEST_SCHEMA_VERSION = 2
B1C_RESULT_SCHEMA_VERSION = 2

# The exact B2 state contract that B2C supersedes.  These three values are
# recorded so a state or artifact written under the superseded contract is
# rejected rather than silently accepted.
SUPERSEDED_STATE_CONTRACT_ID = (
    "g8state-77ff45564fbe282179a860d70f2cc509264d06e1855d7360a50994a4fabaaa7c"
)
SUPERSEDED_STATE_CONTRACT_SHA256 = (
    "2422c4c2a019c2a901cfd8732747555262dfca5601b28b0e700ff33743d4d939"
)
SUPERSEDED_STATE_CONTRACT_BYTES = 9390  # literal-ok: committed B2 artifact length

STATE_CONTRACT_SCHEMA_VERSION = 2
STATE_CONTRACT_ARTIFACT_ROLE = "g8_bler_state_contract"
STATE_CONTRACT_ID_PREFIX = "g8state"
STATE_CONTRACT_SOURCE_ROLE = "g8b_b2c_contract_source"
STATE_CONTRACT_SOURCE_PATHS = (
    "src/baseline/g8_bler_work_units.py",
    "tools/gen_g8_bler_state_contract.py",
    "tools/verify_g8_bler_state_contract.py",
)
STATE_CONTRACT_REPO_RELATIVE_PATH = "results/baseline/g8/bler_state_contract.json"
CAMPAIGN_STATE_REPO_RELATIVE_PATH = "results/baseline/g8/campaign_state.json"
DEFAULT_STATE_CONTRACT_PATH = REPO_ROOT / STATE_CONTRACT_REPO_RELATIVE_PATH
DEFAULT_CAMPAIGN_STATE_PATH = REPO_ROOT / CAMPAIGN_STATE_REPO_RELATIVE_PATH

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

UNIT_STATE_SCHEMA_VERSION = 2
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
UNIT_STATE_RUNTIME_METADATA_FIELDS = (
    "hostname",
    "process_id",
    "device",
    "wall_clock_annotation",
    "update_annotation",
)

# Bindings that can never change, in any transition, including a retry.
UNIT_STATE_PERMANENT_FIELDS = (
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
# Shard assignment is immutable within an attempt and may only change on the
# exact next-attempt clean-claim transition.
UNIT_STATE_SHARD_FIELDS = ("shard_index", "shard_count", "shard_plan_digest")

STATUS_CLAIMED = "claimed"
STATUS_FAILED = "failed"
STATUS_RESULT_LINKED = "result_linked"
STATE_STATUSES = (STATUS_CLAIMED, STATUS_FAILED, STATUS_RESULT_LINKED)

DEFAULT_WORK_UNIT_ROOT = REPO_ROOT / "results/baseline/g8/work_units"
STATE_FILENAME_SUFFIX = ".state.json"
STAGING_FILENAME_SUFFIX = ".staging"
LOCK_DIRECTORY_NAME = ".locks"
LOCK_FILENAME_SUFFIX = ".lock"
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
BUCKET_RE = re.compile(r"^[0-9a-f]{2}$")

_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_STATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_DIRECTORY_MODE = 0o700

B3_RESTART_COMMAND = (
    'rg -n "resume|remaining_work|merge|request_sha256|result_sha256|'
    'completed_work_unit_ids|in_progress_work_unit_id|unit_state|shard_plan" '
    "src/baseline tools tests"
)


class G8BlerWorkUnitError(RuntimeError):
    """Base class for fail-closed B2C authority and state errors."""


class AuthorityAuthenticationError(G8BlerWorkUnitError):
    """The frozen campaign or B1C authority could not be authenticated."""


class StateContractAuthenticationError(G8BlerWorkUnitError):
    """The registered B2C state contract could not be authenticated."""


class ShardPlanError(G8BlerWorkUnitError):
    """A shard argument or persisted shard plan is invalid."""


class UnsafeUnitStatePathError(G8BlerWorkUnitError):
    """A unit-state path is not the one safe canonical path for its ID."""


class UnitStateError(G8BlerWorkUnitError):
    """A unit-state snapshot is malformed or violates local invariants."""


class UnitStateContextRequiredError(UnitStateError, TypeError):
    """A unit-state operation was given a plain execution context.

    This inherits ``TypeError`` as well as ``UnitStateError`` so callers that
    treat a wrong context object as a programming error and callers that treat
    it as a domain failure both see the intended exception.
    """


class StateConflictError(UnitStateError):
    """Exclusive creation lost a race, or a terminal state was rewritten."""


class StateNotFoundError(UnitStateError):
    """An atomic replacement target does not exist."""


class StaleWriterError(StateConflictError):
    """An atomic replacement observed a different previous state digest."""


class AtomicStateError(UnitStateError):
    """A filesystem operation failed without silently repairing state."""


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


def _digest(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
    error: type[Exception] = UnitStateError,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or HEX_DIGEST_RE.fullmatch(value) is None:
        raise error(f"{name} must be lowercase hexadecimal SHA-256")
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

    This layer authenticates campaign, work-unit and sharding authority only.
    It is sufficient to generate the B2C contract and to plan shards; it is
    deliberately *not* sufficient to build, validate, read, create or replace
    a unit state, because a unit state must additionally bind the registered
    B2C state contract.  See :class:`AuthenticatedUnitStateContext`.
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


# ---------------------------------------------------------------------------
# Authenticated B2C state-contract layer
# ---------------------------------------------------------------------------


def _state_contract_identifier(payload: Mapping[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{STATE_CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


class AuthenticatedUnitStateContext:
    """Authenticated access to the registered B2C state contract.

    This wraps an :class:`AuthenticatedExecutionContext` and additionally
    authenticates the B2C state-contract *artifact* against the campaign
    state's produced-artifact binding: path, byte count, SHA-256, contract ID,
    schema version, checkpoint, supersession of the exact B2 contract, and the
    corrected production/generator/verifier source bindings.

    The layering avoids a circular artifact dependency.  The contract artifact
    never binds its own SHA-256, so the generator can build it from an
    execution context alone; the *external* SHA-256 is then obtained here from
    the authenticated campaign-state binding and is what every unit state
    binds.
    """

    __slots__ = (
        "_execution_context",
        "_binding",
        "_contract",
        "_contract_path",
        "_campaign_state_path",
    )

    def __init__(
        self,
        execution_context: AuthenticatedExecutionContext | None = None,
        *,
        campaign_state_path: Path | str | None = None,
        state_contract_path: Path | str | None = None,
    ) -> None:
        if execution_context is None:
            execution_context = AuthenticatedExecutionContext()
        if isinstance(execution_context, AuthenticatedUnitStateContext):
            execution_context = execution_context.execution_context
        if not isinstance(execution_context, AuthenticatedExecutionContext):
            raise TypeError("execution_context must be an AuthenticatedExecutionContext")

        contract_path = (
            DEFAULT_STATE_CONTRACT_PATH
            if state_contract_path is None
            else Path(state_contract_path)
        )
        state_path = (
            DEFAULT_CAMPAIGN_STATE_PATH
            if campaign_state_path is None
            else Path(campaign_state_path)
        )

        registered = self._registered_binding(state_path)
        payload, raw = self._authenticated_contract(contract_path, registered, execution_context)

        self._execution_context = execution_context
        self._contract = MappingProxyType(
            {
                "contract_id": payload["contract_id"],
                "schema_version": payload["schema_version"],
                "checkpoint": payload["checkpoint"],
                "phase": payload["phase"],
            }
        )
        self._binding = MappingProxyType(
            {
                "bler_state_contract_id": payload["contract_id"],
                "bler_state_contract_sha256": sha256_bytes(raw),
            }
        )
        self._contract_path = contract_path
        self._campaign_state_path = state_path

    # -- authentication helpers -------------------------------------------

    @staticmethod
    def _registered_binding(state_path: Path) -> dict[str, Any]:
        try:
            raw = state_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StateContractAuthenticationError(
                f"cannot read the campaign state {state_path}: {exc}"
            ) from exc
        identity = payload.get("identity") if isinstance(payload, Mapping) else None
        if not isinstance(identity, Mapping):
            raise StateContractAuthenticationError("campaign state has no identity block")
        artifacts = identity.get("produced_artifacts")
        if not isinstance(artifacts, list):
            raise StateContractAuthenticationError("campaign state has no produced-artifact list")
        matches = [
            entry
            for entry in artifacts
            if isinstance(entry, Mapping)
            and entry.get("path") == STATE_CONTRACT_REPO_RELATIVE_PATH
        ]
        if len(matches) != 1:
            raise StateContractAuthenticationError(
                "campaign state must register exactly one B2C state-contract artifact"
            )
        entry = dict(matches[0])
        if set(entry) != {"path", "sha256", "bytes"}:
            raise StateContractAuthenticationError(
                "registered state-contract binding has the wrong schema"
            )
        _digest(entry["sha256"], "registered state contract sha256", error=StateContractAuthenticationError)
        _nonnegative_int(entry["bytes"], "registered state contract bytes", StateContractAuthenticationError)
        return entry

    @staticmethod
    def _authenticated_contract(
        contract_path: Path,
        registered: Mapping[str, Any],
        execution_context: AuthenticatedExecutionContext,
    ) -> tuple[dict[str, Any], bytes]:
        try:
            raw = contract_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StateContractAuthenticationError(
                f"cannot read the B2C state contract {contract_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise StateContractAuthenticationError("B2C state contract is not a JSON object")

        actual_sha256 = sha256_bytes(raw)
        if len(raw) != registered["bytes"] or actual_sha256 != registered["sha256"]:
            raise StateContractAuthenticationError(
                "B2C state-contract artifact does not match its registered byte count and SHA-256"
            )

        if payload.get("artifact_role") != STATE_CONTRACT_ARTIFACT_ROLE:
            raise StateContractAuthenticationError("state contract has the wrong artifact role")
        if payload.get("schema_version") != STATE_CONTRACT_SCHEMA_VERSION:
            raise StateContractAuthenticationError(
                "state contract is not the incremented B2C contract schema version"
            )
        if payload.get("phase") != PHASE or payload.get("checkpoint") != CHECKPOINT:
            raise StateContractAuthenticationError("state contract is not the G8_B/B2C contract")
        contract_id = payload.get("contract_id")
        if not isinstance(contract_id, str) or not contract_id.startswith(
            f"{STATE_CONTRACT_ID_PREFIX}-"
        ):
            raise StateContractAuthenticationError("state contract ID has the wrong prefix")
        if contract_id == SUPERSEDED_STATE_CONTRACT_ID:
            raise StateContractAuthenticationError(
                "state contract is the superseded B2 contract"
            )
        if contract_id != _state_contract_identifier(payload):
            raise StateContractAuthenticationError("state contract ID does not reproduce")

        supersedes = payload.get("supersedes")
        if not isinstance(supersedes, Mapping) or {
            "checkpoint": supersedes.get("checkpoint"),
            "contract_id": supersedes.get("contract_id"),
            "contract_sha256": supersedes.get("contract_sha256"),
            "contract_bytes": supersedes.get("contract_bytes"),
        } != {
            "checkpoint": SUPERSEDED_CHECKPOINT,
            "contract_id": SUPERSEDED_STATE_CONTRACT_ID,
            "contract_sha256": SUPERSEDED_STATE_CONTRACT_SHA256,
            "contract_bytes": SUPERSEDED_STATE_CONTRACT_BYTES,
        }:
            raise StateContractAuthenticationError(
                "state contract does not supersede the exact B2 contract"
            )

        sources = payload.get("contract_sources")
        if not isinstance(sources, list) or [
            entry.get("path") if isinstance(entry, Mapping) else None for entry in sources
        ] != list(STATE_CONTRACT_SOURCE_PATHS):
            raise StateContractAuthenticationError("state contract source path list changed")
        for entry in sources:
            try:
                body = (REPO_ROOT / entry["path"]).read_bytes()
            except OSError as exc:
                raise StateContractAuthenticationError(
                    f"cannot read bound B2C source {entry['path']}: {exc}"
                ) from exc
            if (
                entry.get("role") != STATE_CONTRACT_SOURCE_ROLE
                or entry.get("bytes") != len(body)
                or entry.get("sha256") != sha256_bytes(body)
            ):
                raise StateContractAuthenticationError(
                    f"bound B2C source changed: {entry['path']}"
                )
            if entry["path"] == STATE_CONTRACT_REPO_RELATIVE_PATH:
                raise StateContractAuthenticationError("state contract binds its own output path")

        if actual_sha256.encode("ascii") in raw:
            raise StateContractAuthenticationError("state contract binds its own artifact SHA-256")

        unit_schema = payload.get("unit_state_schema")
        if (
            not isinstance(unit_schema, Mapping)
            or unit_schema.get("schema_version") != UNIT_STATE_SCHEMA_VERSION
            or set(unit_schema.get("identity_fields") or ()) != set(UNIT_STATE_IDENTITY_FIELDS)
        ):
            raise StateContractAuthenticationError(
                "state contract does not describe the corrected unit-state schema"
            )

        authority = payload.get("authority_bindings")
        expected_authority = execution_context.authority_binding()
        if not isinstance(authority, Mapping) or any(
            authority.get(field) != expected_authority[field]
            for field in (
                "campaign_id",
                "campaign_manifest_sha256",
                "required_bler_artifact_sha256",
                "selection_policy_sha256",
                "bler_tooling_contract_id",
                "bler_tooling_contract_sha256",
            )
        ):
            raise StateContractAuthenticationError(
                "state contract authority bindings differ from the authenticated context"
            )
        return payload, raw

    # -- accessors ---------------------------------------------------------

    @property
    def execution_context(self) -> AuthenticatedExecutionContext:
        return self._execution_context

    @property
    def state_contract_path(self) -> Path:
        return self._contract_path

    @property
    def campaign_state_path(self) -> Path:
        return self._campaign_state_path

    @property
    def state_contract_id(self) -> str:
        return self._binding["bler_state_contract_id"]

    @property
    def state_contract_sha256(self) -> str:
        return self._binding["bler_state_contract_sha256"]

    def state_contract_binding(self) -> dict[str, str]:
        return dict(self._binding)

    # Delegated execution-context authority.
    @property
    def campaign_id(self) -> str:
        return self._execution_context.campaign_id

    @property
    def required_work_unit_count(self) -> int:
        return self._execution_context.required_work_unit_count

    @property
    def ordered_work_unit_ids(self) -> tuple[str, ...]:
        return self._execution_context.ordered_work_unit_ids

    @property
    def work_unit_ids(self) -> tuple[str, ...]:
        return self._execution_context.work_unit_ids

    def authority_binding(self) -> dict[str, Any]:
        return self._execution_context.authority_binding()

    def ordinal(self, work_unit_id: str) -> int:
        return self._execution_context.ordinal(work_unit_id)

    def work_unit_record_sha256(self, work_unit_id: str) -> str:
        return self._execution_context.work_unit_record_sha256(work_unit_id)

    def work_unit_record(self, work_unit_id: str) -> dict[str, Any]:
        return self._execution_context.work_unit_record(work_unit_id)

    def seed(self, work_unit_id: str, purpose: str) -> int:
        return self._execution_context.seed(work_unit_id, purpose)


def _context(value: Any) -> AuthenticatedExecutionContext:
    """Accept either authority layer where only B1C authority is required."""

    if isinstance(value, AuthenticatedUnitStateContext):
        return value.execution_context
    if isinstance(value, AuthenticatedExecutionContext):
        return value
    raise TypeError(
        "context must be an AuthenticatedExecutionContext or AuthenticatedUnitStateContext"
    )


def _state_context(value: Any) -> AuthenticatedUnitStateContext:
    """Require the B2C state-contract layer; reject a plain execution context."""

    if isinstance(value, AuthenticatedUnitStateContext):
        return value
    if isinstance(value, AuthenticatedExecutionContext):
        raise UnitStateContextRequiredError(
            "unit-state operations require an AuthenticatedUnitStateContext; "
            "a plain AuthenticatedExecutionContext does not authenticate the "
            "registered B2C state contract"
        )
    raise UnitStateContextRequiredError("context must be an AuthenticatedUnitStateContext")


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
    context: Any,
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
    context: Any,
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
    supplied_digest = _digest(payload["plan_digest"], "plan_digest", error=ShardPlanError)
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
    _digest(supplied, "plan_digest", error=ShardPlanError)
    return sha256_bytes(canonical_json(body))


def shard_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    return canonical_json(plan)


# ---------------------------------------------------------------------------
# No-follow filesystem inspection and safe deterministic unit-state paths
# ---------------------------------------------------------------------------


def _lstat(path: Path | str) -> os.stat_result | None:
    """No-follow inspection; a dangling symlink is *present*, not absent."""

    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except NotADirectoryError as exc:
        raise UnsafeUnitStatePathError(
            f"unit-state path has a non-directory parent component: {path}"
        ) from exc
    except OSError as exc:
        raise UnsafeUnitStatePathError(f"cannot inspect unit-state path {path}: {exc}") from exc


def _reject_symlink_or_nondirectory(path: Path, label: str) -> os.stat_result | None:
    entry = _lstat(path)
    if entry is None:
        return None
    if stat.S_ISLNK(entry.st_mode):
        raise UnsafeUnitStatePathError(f"{label} may not be a symlink: {path}")
    if not stat.S_ISDIR(entry.st_mode):
        raise UnsafeUnitStatePathError(f"{label} is not a directory: {path}")
    return entry


def _root_path(root: Path | str | None) -> Path:
    value = DEFAULT_WORK_UNIT_ROOT if root is None else Path(root)
    if not value.is_absolute():
        raise UnsafeUnitStatePathError("work-unit root must be absolute")
    _reject_symlink_or_nondirectory(value, "work-unit root")
    return value


def unit_state_relative_path(
    context: Any,
    work_unit_id: str,
) -> Path:
    context = _context(context)
    context.ordinal(work_unit_id)
    digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
    return Path(digest[:2]) / f"{digest}{STATE_FILENAME_SUFFIX}"


def unit_state_path(
    context: Any,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    return _root_path(root) / unit_state_relative_path(context, work_unit_id)


def state_path_for_work_unit(
    context: Any,
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
    """Validate path *shape* and no-follow filesystem state.

    Shape validation alone is never treated as a security proof: the
    publication and replacement primitives reopen the root and bucket
    descriptor-relative with ``O_NOFOLLOW`` and perform every subsequent
    operation against those descriptors, so a rename between this check and
    the open cannot redirect a write outside the authenticated root.
    """

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
    if BUCKET_RE.fullmatch(bucket) is None:
        raise UnsafeUnitStatePathError("unit-state bucket is not lowercase two-digit hex")
    if not filename.endswith(STATE_FILENAME_SUFFIX):
        raise UnsafeUnitStatePathError("unit-state path has the wrong extension")
    digest = filename[: -len(STATE_FILENAME_SUFFIX)]
    if HEX_DIGEST_RE.fullmatch(digest) is None or digest[:2] != bucket:
        raise UnsafeUnitStatePathError("unit-state filename is not a lowercase SHA-256 digest")

    # Containment is established lexically by ``relative_to`` above and then
    # enforced physically by the no-follow component inspection below and by
    # the descriptor-relative publication primitives.  Comparing two
    # ``resolve()`` calls derived from this same candidate would prove
    # nothing an attacker-controlled parent had not already decided, so it is
    # deliberately not used as the safety argument.
    #
    # No-follow inspection: a dangling symlink at the bucket or the final name
    # is *present*, and both are rejected.
    _reject_symlink_or_nondirectory(candidate.parent, "unit-state bucket directory")
    final = _lstat(candidate)
    if final is not None and stat.S_ISLNK(final.st_mode):
        raise UnsafeUnitStatePathError("unit-state path may not be a symlink")
    if final is not None and not stat.S_ISREG(final.st_mode):
        raise UnsafeUnitStatePathError("unit-state path is not a regular file")
    return root_path, candidate


def validate_unit_state_path(
    context: Any,
    path: Path | str,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    context = _context(context)
    expected = unit_state_path(context, work_unit_id, root=root)
    _root_path_value, candidate = _validate_candidate_path_shape(path, root=root)
    if str(candidate) != str(expected):
        raise UnsafeUnitStatePathError("unit-state path digest does not correspond to the work-unit ID")
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
    context: AuthenticatedUnitStateContext,
    payload: dict[str, Any],
) -> dict[str, Any]:
    authority = context.authority_binding()
    state_binding = context.state_contract_binding()
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
    if identity["bler_state_contract_id"] == SUPERSEDED_STATE_CONTRACT_ID:
        raise UnitStateError("unit state binds the superseded B2 state contract")
    if identity["bler_state_contract_sha256"] == SUPERSEDED_STATE_CONTRACT_SHA256:
        raise UnitStateError("unit state binds the superseded B2 state-contract artifact")
    for field in ("bler_state_contract_id", "bler_state_contract_sha256"):
        if identity[field] != state_binding[field]:
            raise UnitStateError(
                f"unit state does not bind the registered B2C state contract: {field}"
            )
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

    # No result may exist without a request binding, in any status.
    if (result_path is not None or result_sha is not None) and request_sha is None:
        raise UnitStateError("a result reference requires a bound request SHA-256")

    if status == STATUS_CLAIMED:
        if (request_sha, result_path, result_sha, scientific, trials) != (None, None, None, False, 0):
            raise UnitStateError("pre-execution claim must have no execution or result fields")
    elif status == STATUS_FAILED:
        if result_path is not None or result_sha is not None:
            raise UnitStateError("failed state may not carry a result reference")
        if not scientific and trials != 0:
            raise UnitStateError("failed state with completed trials must mark execution performed")
    elif status == STATUS_RESULT_LINKED:
        if request_sha is None:
            raise UnitStateError("result-linked state requires a bound request SHA-256")
        if result_path is None or result_sha is None:
            raise UnitStateError("result-linked state requires canonical result path and SHA-256")
        if not scientific or trials <= 0:
            raise UnitStateError("result-linked state requires scientific execution and positive trials")

    if payload["identity_sha256"] != _recomputed_identity_digest(identity):
        raise UnitStateError("unit-state identity digest does not reproduce")
    return payload


def validate_unit_state(
    context: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    context = _state_context(context)
    return _validate_state_against_context(context, _validate_state_shape(state))


def build_unit_state(
    context: Any,
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
    context = _state_context(context)
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
    state_binding = context.state_contract_binding()
    identity = {
        "schema_version": UNIT_STATE_SCHEMA_VERSION,
        "artifact_role": UNIT_STATE_ARTIFACT_ROLE,
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "bler_state_contract_id": state_binding["bler_state_contract_id"],
        "bler_state_contract_sha256": state_binding["bler_state_contract_sha256"],
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
    context: Any,
    state: Mapping[str, Any],
) -> bytes:
    validated = validate_unit_state(context, state)
    try:
        return canonical_json(validated)
    except (TypeError, ValueError) as exc:  # pragma: no cover - validation already rejects these
        raise UnitStateError("unit state cannot be rendered as canonical JSON") from exc


def unit_state_sha256(
    context: Any,
    state: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_state_bytes(context, state))


# ---------------------------------------------------------------------------
# Transition legality
# ---------------------------------------------------------------------------


def _clean_claim_fields(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        identity["status"],
        identity["request_sha256"],
        identity["result_path"],
        identity["result_sha256"],
        identity["scientific_execution_performed"],
        identity["trials_completed"],
        identity["test_split_access"],
    )


_CLEAN_CLAIM = (STATUS_CLAIMED, None, None, None, False, 0, 0)


def validate_state_transition(
    previous: Mapping[str, Any],
    proposed: Mapping[str, Any],
) -> None:
    """Reject every illegal transition between two validated unit states.

    A valid ``result_linked`` state is terminal: this raises for *any*
    proposed successor.  Exact canonical-byte idempotence is handled by the
    replacement primitive before this function is consulted.
    """

    old = previous["identity"]
    new = proposed["identity"]
    for field in UNIT_STATE_PERMANENT_FIELDS:
        if old[field] != new[field]:
            raise StateConflictError(f"unit-state immutable binding changed: {field}")

    if old["status"] == STATUS_RESULT_LINKED:
        raise StateConflictError(
            "result-linked unit state is terminal; only exact canonical-byte "
            "idempotence is permitted"
        )

    old_attempt = old["attempt"]
    new_attempt = new["attempt"]
    if new_attempt == old_attempt + 1:
        # The only legal resharding path: a clean claim on the next attempt.
        if _clean_claim_fields(new) != _CLEAN_CLAIM:
            raise StateConflictError(
                "a new attempt must begin as a clean claim with no request, "
                "result, execution flag or completed trials"
            )
        return
    if new_attempt != old_attempt:
        raise StateConflictError(
            "unit-state attempt must stay equal or advance by exactly one"
        )

    # Same attempt.
    for field in UNIT_STATE_SHARD_FIELDS:
        if old[field] != new[field]:
            raise StateConflictError(
                f"shard assignment may only change on the next-attempt clean claim: {field}"
            )
    if new["trials_completed"] < old["trials_completed"]:
        raise StateConflictError("unit-state trials_completed may never decrease")
    if old["scientific_execution_performed"] and not new["scientific_execution_performed"]:
        raise StateConflictError(
            "scientific_execution_performed may never change from true to false"
        )
    if old["request_sha256"] is not None and new["request_sha256"] != old["request_sha256"]:
        raise StateConflictError("a bound request SHA-256 may never change or become null")
    if old["status"] == STATUS_FAILED and new["status"] == STATUS_CLAIMED:
        raise StateConflictError("failed unit state cannot become a same-attempt claim")
    if old["status"] == STATUS_FAILED and new["status"] == STATUS_RESULT_LINKED:
        raise StateConflictError("failed unit state cannot become a same-attempt result")


# Retained as the historical private name used by earlier B2 call sites.
_validate_state_transition = validate_state_transition


# ---------------------------------------------------------------------------
# Descriptor-relative crash-atomic publication
# ---------------------------------------------------------------------------


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _fsync_directory_descriptor(descriptor: int, label: str) -> None:
    """Fail closed.  ``EACCES`` is a permission failure, not "unsupported"."""

    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise AtomicStateError(
            f"crash-durable publication is unavailable: {label} directory fsync failed: {exc}"
        ) from exc


def _open_root_descriptor(root_path: Path) -> int:
    _reject_symlink_or_nondirectory(root_path, "work-unit root")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AtomicStateError(f"cannot create the work-unit root {root_path}: {exc}") from exc
    _reject_symlink_or_nondirectory(root_path, "work-unit root")
    try:
        return os.open(root_path, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafeUnitStatePathError(
                f"work-unit root may not be a symlink: {root_path}"
            ) from exc
        raise AtomicStateError(f"cannot open the work-unit root {root_path}: {exc}") from exc


def _open_subdirectory_descriptor(parent_fd: int, name: str, label: str) -> int:
    try:
        os.mkdir(name, _DIRECTORY_MODE, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise AtomicStateError(f"cannot create the {label} {name}: {exc}") from exc
    try:
        entry = os.lstat(name, dir_fd=parent_fd)
    except OSError as exc:  # pragma: no cover - the mkdir above guarantees presence
        raise AtomicStateError(f"cannot inspect the {label} {name}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode):
        raise UnsafeUnitStatePathError(f"{label} may not be a symlink: {name}")
    if not stat.S_ISDIR(entry.st_mode):
        raise UnsafeUnitStatePathError(f"{label} is not a directory: {name}")
    try:
        return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafeUnitStatePathError(f"{label} may not be a symlink: {name}") from exc
        raise AtomicStateError(f"cannot open the {label} {name}: {exc}") from exc


@contextmanager
def _bucket_descriptor(root_path: Path, bucket: str) -> Iterator[tuple[int, int]]:
    """Yield ``(root_fd, bucket_fd)`` opened no-follow, descriptor-relative."""

    root_fd = _open_root_descriptor(root_path)
    bucket_fd: int | None = None
    try:
        bucket_fd = _open_subdirectory_descriptor(root_fd, bucket, "unit-state bucket directory")
        yield root_fd, bucket_fd
    finally:
        _close_quietly(bucket_fd)
        _close_quietly(root_fd)


def _staging_name(final_name: str) -> str:
    token = os.urandom(12).hex()  # literal-ok: 96-bit staging-name uniqueness token
    return f".{final_name}.{os.getpid()}.{token}{STAGING_FILENAME_SUFFIX}"


def _write_staging_file(bucket_fd: int, name: str, body: bytes) -> None:
    """Create, write, flush and fsync one staging file, closing its fd once."""

    descriptor: int | None = None
    stream = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            _STATE_FILE_MODE,
            dir_fd=bucket_fd,
        )
        stream = os.fdopen(descriptor, "wb")
    except Exception as exc:
        raise AtomicStateError(f"cannot stage unit-state bytes in {name}: {exc}") from exc
    finally:
        # ``os.fdopen`` takes ownership of the descriptor on success; drop our
        # reference so the raw descriptor is closed exactly once, here or by
        # the stream, and never by both.
        if stream is not None:
            descriptor = None
        _close_quietly(descriptor)

    try:
        with stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as exc:
        raise AtomicStateError(f"cannot durably stage unit-state bytes in {name}: {exc}") from exc


def _publish_without_replace(bucket_fd: int, staging: str, final_name: str) -> None:
    """Publish with a Linux/POSIX no-replace primitive; never overwrite."""

    try:
        os.link(
            staging,
            final_name,
            src_dir_fd=bucket_fd,
            dst_dir_fd=bucket_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        # A regular file, symlink, dangling symlink, directory, or any other
        # object already occupies the final name.
        raise StateConflictError(f"unit-state already exists: {final_name}") from exc
    except NotImplementedError as exc:
        raise AtomicStateError(
            "crash-durable publication is unavailable: this platform does not "
            "support descriptor-relative no-follow hard-link publication"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV}:
            raise AtomicStateError(
                "crash-durable publication is unavailable: this filesystem "
                f"supplies no atomic no-replace primitive: {exc}"
            ) from exc
        raise AtomicStateError(f"unit-state publication failed: {final_name}: {exc}") from exc


def _remove_staging(bucket_fd: int, staging: str) -> None:
    """Best-effort staging removal.

    An orphan staging file is never interpreted as a unit state and never
    blocks a retry, so a cleanup failure after successful publication must not
    turn a published state into a reported failure.
    """

    try:
        os.unlink(staging, dir_fd=bucket_fd)
    except OSError:
        pass


def _read_state_bytes(path: Path) -> bytes:
    descriptor: int | None = None
    stream = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        stream = os.fdopen(descriptor, "rb")
    except FileNotFoundError as exc:
        raise StateNotFoundError(f"unit-state file does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafeUnitStatePathError(
                f"unit-state path may not be a symlink: {path}"
            ) from exc
        raise AtomicStateError(f"cannot read unit-state file {path}: {exc}") from exc
    except Exception as exc:
        raise AtomicStateError(f"cannot read unit-state file {path}: {exc}") from exc
    finally:
        if stream is not None:
            descriptor = None
        _close_quietly(descriptor)

    try:
        with stream:
            return stream.read()
    except Exception as exc:
        raise AtomicStateError(f"cannot read unit-state file {path}: {exc}") from exc


def _validate_existing_state(
    context: AuthenticatedUnitStateContext,
    path: Path,
    *,
    root: Path | str | None,
) -> tuple[dict[str, Any], bytes, str]:
    _validate_candidate_path_shape(path, root=root)
    raw = _read_state_bytes(path)
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


def _reread_installed(
    context: AuthenticatedUnitStateContext,
    target: Path,
    root: Path | str | None,
) -> tuple[dict[str, Any], bytes, str]:
    """Reread and validate installed bytes, always as a domain error.

    The post-install reread is the last line of the publication contract, so
    an unexpected failure here must still reach the caller as an
    :class:`AtomicStateError` rather than as a raw ``OSError``.
    """

    try:
        return _validate_existing_state(context, target, root=root)
    except G8BlerWorkUnitError:
        raise
    except Exception as exc:
        raise AtomicStateError(
            f"cannot reread and validate the installed unit state: {target}: {exc}"
        ) from exc


def create_unit_state_exclusive(
    context: Any,
    state: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    path: Path | str | None = None,
) -> str:
    """Publish one canonical state file crash-atomically and exclusively.

    Bytes are rendered in full, staged in a unique same-directory file with
    ``O_CREAT | O_EXCL | O_NOFOLLOW`` and mode ``0600``, written, flushed and
    fsynced, then published to the final name with a descriptor-relative
    no-follow hard link that cannot replace anything.  The final pathname is
    never opened for writing.  A hard exit before publication leaves the final
    path absent; a hard exit after publication leaves complete canonical
    bytes.
    """

    context = _state_context(context)
    validated = validate_unit_state(context, state)
    work_unit_id = validated["identity"]["work_unit_id"]
    root_path = _root_path(root)
    target = unit_state_path(context, work_unit_id, root=root) if path is None else Path(path)
    validate_unit_state_path(context, target, work_unit_id, root=root)
    body = canonical_state_bytes(context, validated)

    relative = target.relative_to(root_path)
    bucket, final_name = relative.parts
    with _bucket_descriptor(root_path, bucket) as (_root_fd, bucket_fd):
        # Reject anything already occupying the final name before staging, so
        # the common conflict does not leave an orphan behind.
        try:
            os.lstat(final_name, dir_fd=bucket_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - defensive
            raise AtomicStateError(f"cannot inspect {final_name}: {exc}") from exc
        else:
            raise StateConflictError(f"unit-state already exists: {target}")

        staging = _staging_name(final_name)
        try:
            _write_staging_file(bucket_fd, staging, body)
            _publish_without_replace(bucket_fd, staging, final_name)
        except BaseException:
            _remove_staging(bucket_fd, staging)
            raise
        _fsync_directory_descriptor(bucket_fd, "unit-state bucket")
        _remove_staging(bucket_fd, staging)

    installed, installed_body, installed_sha = _reread_installed(context, target, root)
    if installed != validated or installed_body != body:
        raise AtomicStateError("exclusive creation did not install the requested canonical state")
    return installed_sha


# ---------------------------------------------------------------------------
# Linearizable replacement
# ---------------------------------------------------------------------------

_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _process_local_lock(root_path: Path, digest: str) -> threading.Lock:
    key = (str(root_path.resolve(strict=False)), digest)
    with _PROCESS_LOCK_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


@contextmanager
def unit_state_lock(root_path: Path, work_unit_id_digest: str) -> Iterator[None]:
    """Hold the exclusive per-unit critical section.

    Two layers are required.  ``fcntl.flock(LOCK_EX)`` excludes other
    *processes* and is released by the kernel on normal exit, on exception and
    on process death.  A process-local keyed :class:`threading.Lock` excludes
    other *threads* in this process, which ``flock`` on a per-open-file-
    description lock would not reliably do.

    Lock files live in a canonical ``.locks`` directory that can never collide
    with a two-hex-digit state bucket, so a lock is never mistaken for state.
    """

    local = _process_local_lock(root_path, work_unit_id_digest)
    local.acquire()
    root_fd: int | None = None
    locks_fd: int | None = None
    lock_fd: int | None = None
    try:
        root_fd = _open_root_descriptor(root_path)
        locks_fd = _open_subdirectory_descriptor(root_fd, LOCK_DIRECTORY_NAME, "unit-state lock directory")
        name = f"{work_unit_id_digest}{LOCK_FILENAME_SUFFIX}"
        try:
            lock_fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                _STATE_FILE_MODE,
                dir_fd=locks_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise UnsafeUnitStatePathError(
                    f"unit-state lock file may not be a symlink: {name}"
                ) from exc
            raise AtomicStateError(f"cannot open the unit-state lock {name}: {exc}") from exc
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise AtomicStateError(f"cannot acquire the unit-state lock {name}: {exc}") from exc
        yield
    finally:
        # Closing the descriptor releases the flock; the kernel also releases
        # it if this process dies while holding it.
        _close_quietly(lock_fd)
        _close_quietly(locks_fd)
        _close_quietly(root_fd)
        local.release()


def replace_unit_state(
    context: Any,
    path: Path | str,
    proposed_state: Mapping[str, Any],
    expected_previous_sha256: str,
    *,
    root: Path | str | None = None,
) -> str:
    """Linearizably replace one state under an exclusive per-unit lock.

    The expected-previous-SHA comparison and the publication happen inside one
    critical section, so two writers that start from the same predecessor SHA
    can never both succeed: exactly one wins and every loser observes the
    winner and raises :class:`StaleWriterError`.
    """

    context = _state_context(context)
    expected = _digest(expected_previous_sha256, "expected_previous_sha256")
    target = Path(path)
    root_path = _root_path(root)
    proposed = validate_unit_state(context, proposed_state)
    work_unit_id = proposed["identity"]["work_unit_id"]
    validate_unit_state_path(context, target, work_unit_id, root=root)
    digest = target.name[: -len(STATE_FILENAME_SUFFIX)]
    body = canonical_state_bytes(context, proposed)

    with unit_state_lock(root_path, digest):
        current, current_body, current_sha = _validate_existing_state(context, target, root=root)
        if current_sha != expected:
            raise StaleWriterError(
                f"stale unit-state writer: expected {expected}, observed {current_sha}"
            )
        if current["identity"]["status"] == STATUS_RESULT_LINKED:
            # Terminal.  Exact canonical-byte idempotence only.
            if current_body == body:
                return current_sha
            raise StateConflictError(
                "result-linked unit state is terminal and cannot be changed"
            )
        validate_state_transition(current, proposed)
        if current_body == body:
            return current_sha

        relative = target.relative_to(root_path)
        bucket, final_name = relative.parts
        with _bucket_descriptor(root_path, bucket) as (_root_fd, bucket_fd):
            staging = _staging_name(final_name)
            try:
                _write_staging_file(bucket_fd, staging, body)
                try:
                    os.replace(staging, final_name, src_dir_fd=bucket_fd, dst_dir_fd=bucket_fd)
                except OSError as exc:
                    raise AtomicStateError(
                        f"unit-state replacement failed before publication: {target}: {exc}"
                    ) from exc
            except BaseException:
                _remove_staging(bucket_fd, staging)
                raise
            _fsync_directory_descriptor(bucket_fd, "unit-state bucket")

        _installed, installed_body, installed_sha = _reread_installed(context, target, root)
        if installed_body != body:
            raise AtomicStateError("installed unit state does not match the proposed canonical bytes")
        return installed_sha


def read_unit_state(
    context: Any,
    path: Path | str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one snapshot only; never interpret a directory as history."""

    context = _state_context(context)
    state, _raw, _sha = _validate_existing_state(context, Path(path), root=root)
    return state


# Explicit aliases make the two publication boundaries discoverable without
# introducing a second implementation or a resume/merge policy.
exclusive_create_unit_state = create_unit_state_exclusive
atomic_replace_unit_state = replace_unit_state


__all__ = [
    "AtomicStateError",
    "AuthenticatedExecutionContext",
    "AuthenticatedUnitStateContext",
    "AuthorityAuthenticationError",
    "B1C_REQUEST_SCHEMA_VERSION",
    "B1C_RESULT_SCHEMA_VERSION",
    "B1C_TOOLING_SCHEMA_VERSION",
    "B3_RESTART_COMMAND",
    "CAMPAIGN_STATE_REPO_RELATIVE_PATH",
    "CHECKPOINT",
    "DEFAULT_CAMPAIGN_STATE_PATH",
    "DEFAULT_STATE_CONTRACT_PATH",
    "DEFAULT_WORK_UNIT_ROOT",
    "EXPECTED_B1C_CONTRACT_ID",
    "EXPECTED_B1C_CONTRACT_SHA256",
    "EXPECTED_CAMPAIGN_ID",
    "EXPECTED_CAMPAIGN_MANIFEST_SHA256",
    "EXPECTED_REQUIRED_IDENTITIES_SHA256",
    "EXPECTED_REQUIRED_WORK_UNIT_COUNT",
    "EXPECTED_SELECTION_POLICY_SHA256",
    "G8BlerWorkUnitError",
    "LOCK_DIRECTORY_NAME",
    "LOCK_FILENAME_SUFFIX",
    "PHASE",
    "SHARDING_ALGORITHM",
    "SHARD_FORMULA",
    "SHARD_PLAN_ARTIFACT_ROLE",
    "SHARD_PLAN_DIGEST_RULE",
    "SHARD_PLAN_FIELDS",
    "SHARD_PLAN_SCHEMA_VERSION",
    "STAGING_FILENAME_SUFFIX",
    "STATE_CONTRACT_ARTIFACT_ROLE",
    "STATE_CONTRACT_ID_PREFIX",
    "STATE_CONTRACT_REPO_RELATIVE_PATH",
    "STATE_CONTRACT_SCHEMA_VERSION",
    "STATE_CONTRACT_SOURCE_PATHS",
    "STATE_CONTRACT_SOURCE_ROLE",
    "STATE_STATUSES",
    "STATUS_CLAIMED",
    "STATUS_FAILED",
    "STATUS_RESULT_LINKED",
    "SUPERSEDED_CHECKPOINT",
    "SUPERSEDED_STATE_CONTRACT_BYTES",
    "SUPERSEDED_STATE_CONTRACT_ID",
    "SUPERSEDED_STATE_CONTRACT_SHA256",
    "ShardPlanError",
    "StaleWriterError",
    "StateConflictError",
    "StateContractAuthenticationError",
    "StateNotFoundError",
    "UNIT_STATE_ARTIFACT_ROLE",
    "UNIT_STATE_FIELDS",
    "UNIT_STATE_IDENTITY_FIELDS",
    "UNIT_STATE_PERMANENT_FIELDS",
    "UNIT_STATE_RUNTIME_METADATA_FIELDS",
    "UNIT_STATE_SCHEMA_VERSION",
    "UNIT_STATE_SHARD_FIELDS",
    "UnitStateContextRequiredError",
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
    "unit_state_lock",
    "unit_state_path",
    "unit_state_relative_path",
    "unit_state_sha256",
    "validate_shard_arguments",
    "validate_shard_plan",
    "validate_state_transition",
    "validate_unit_state",
    "validate_unit_state_path",
]
