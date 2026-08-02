#!/usr/bin/env python3
"""G8_B B3 exact resume, recovery and merge validation.

This module is the read-only-by-default authority that a restart consults to
learn *exactly* what work remains.  It never runs a simulation, never writes a
request or a result, and never produces a scientific measurement.  It reads
authenticated authority (B1C tooling contract, B2C state contract, campaign
manifest, required-identity artifact) plus the on-disk per-unit state and
per-attempt request/result files, and it derives:

* a closed per-unit classification;
* an explicitly bounded recovery matrix;
* a byte-deterministic resume plan;
* a merge *validation* report;
* a campaign-state reconciliation proposal.

Two rules drive every design decision here.

**Nothing benign is inferred from silence.**  A malformed state, an unreadable
file, an unknown filename, a hard-linked alias, a dangling symlink or a
contradictory chain raises a typed HOLD.  It is never quietly reclassified as
``absent``, quarantined, deleted, downgraded or skipped, because each of those
would silently drop required scientific coverage.

**Filesystem order is never evidence.**  Every returned sequence is ordered by
the frozen required-work-unit order and by explicit numeric attempt.  Hostname,
PID, wall time, mtime, inode, worker name, completion order and absolute paths
never enter a record or a digest.
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
from baseline import g8_bler_work_units as work_units
from baseline.g8_campaign import canonical_json, sha256_bytes
from config.params import REPO_ROOT

# ---------------------------------------------------------------------------
# Checkpoint identity and the frozen B3 contract artifact
# ---------------------------------------------------------------------------

PHASE = "G8_B"
CHECKPOINT = "B3"
CAMPAIGN_ROLE = "G-8"

RESUME_CONTRACT_SCHEMA_VERSION = 1
RESUME_CONTRACT_ARTIFACT_ROLE = "g8_bler_resume_merge_contract"
RESUME_CONTRACT_ID_PREFIX = "g8resume"
RESUME_CONTRACT_SOURCE_ROLE = "g8b_b3_resume_contract_source"
RESUME_CONTRACT_REPO_RELATIVE_PATH = "results/baseline/g8/bler_resume_contract.json"
RESUME_CONTRACT_SOURCE_PATHS = (
    "src/baseline/g8_bler_resume.py",
    "tools/gen_g8_bler_resume_contract.py",
    "tools/verify_g8_bler_resume_contract.py",
)
DEFAULT_RESUME_CONTRACT_PATH = REPO_ROOT / RESUME_CONTRACT_REPO_RELATIVE_PATH

#: The exact command a B4 session runs first.  Stored in campaign state at
#: registration and bound into the contract.
B4_RESTART_COMMAND = (
    'rg -n "bounded_smoke|NON-SCIENTIFIC BOUNDED SMOKE|build_bounded_smoke_request|'
    'resume_plan|merge_report|result_linked" src/baseline tools tests'
)

# ---------------------------------------------------------------------------
# Immutable authority, restated locally so a drift is a loud failure
# ---------------------------------------------------------------------------

EXPECTED_CAMPAIGN_ID = work_units.EXPECTED_CAMPAIGN_ID
EXPECTED_CAMPAIGN_MANIFEST_SHA256 = work_units.EXPECTED_CAMPAIGN_MANIFEST_SHA256
EXPECTED_REQUIRED_IDENTITIES_SHA256 = work_units.EXPECTED_REQUIRED_IDENTITIES_SHA256
EXPECTED_SELECTION_POLICY_SHA256 = work_units.EXPECTED_SELECTION_POLICY_SHA256
EXPECTED_B1C_CONTRACT_ID = work_units.EXPECTED_B1C_CONTRACT_ID
EXPECTED_B1C_CONTRACT_SHA256 = work_units.EXPECTED_B1C_CONTRACT_SHA256
EXPECTED_B2C_CONTRACT_ID = (
    "g8state-a36b37f3c21d4254a50ffe5e893237ee4738c68c7b3e9d76b473856ca7605deb"
)
EXPECTED_B2C_CONTRACT_SHA256 = (
    "cac1dcf803d435de7b483db04d12afc30bea4180a835d8c0476de65540fbf583"
)
EXPECTED_REQUIRED_WORK_UNIT_COUNT = work_units.EXPECTED_REQUIRED_WORK_UNIT_COUNT
REQUEST_SCHEMA_VERSION = work_units.B1C_REQUEST_SCHEMA_VERSION
RESULT_SCHEMA_VERSION = work_units.B1C_RESULT_SCHEMA_VERSION
UNIT_STATE_SCHEMA_VERSION = work_units.UNIT_STATE_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Canonical runtime layout
# ---------------------------------------------------------------------------

#: The logical repository-relative prefix every persisted path is expressed in.
#: The *physical* root is caller supplied; its absolute location is deliberately
#: never recorded in a plan, a report or a digest.
WORK_UNIT_ROOT_LOGICAL_PREFIX = "results/baseline/g8/work_units"
DEFAULT_WORK_UNIT_ROOT = work_units.DEFAULT_WORK_UNIT_ROOT

STATE_FILENAME_SUFFIX = work_units.STATE_FILENAME_SUFFIX
REQUEST_FILENAME_SUFFIX = ".request.json"
RESULT_FILENAME_SUFFIX = ".result.json"
ATTEMPT_TOKEN_PREFIX = "attempt-"
STAGING_FILENAME_SUFFIX = work_units.STAGING_FILENAME_SUFFIX
LOCK_DIRECTORY_NAME = work_units.LOCK_DIRECTORY_NAME
LOCK_FILENAME_SUFFIX = work_units.LOCK_FILENAME_SUFFIX
RECONCILIATION_LOCK_NAME = ".reconciliation.lock"

#: ``attempt`` is a positive base-10 integer with no sign, whitespace, prefix,
#: suffix, decimal point or leading zero.  ``attempt-1`` is valid;
#: ``attempt-01``, ``attempt-0``, ``attempt-+1`` and ``attempt-1.0`` are not.
ATTEMPT_TOKEN_RE = re.compile(r"^[1-9][0-9]*$")
ATTEMPT_GRAMMAR = (
    "positive base-10 integer with no sign, whitespace, prefix, suffix, "
    "decimal point or leading zero"
)
HEX_DIGEST_RE = work_units.HEX_DIGEST_RE
BUCKET_RE = work_units.BUCKET_RE
_STAGING_NAME_RE = re.compile(
    r"^\.(?P<final>.+)\.(?P<pid>[1-9][0-9]*)\.(?P<token>[0-9a-f]{24})"
    + re.escape(STAGING_FILENAME_SUFFIX)
    + r"$"
)

CANONICAL_FILE_ENCODING = (
    "compact sorted-key JSON bytes, ensure_ascii=true, allow_nan=false, "
    'separators (",", ":"), no trailing newline'
)

#: Artifact kinds the census recognizes inside a bucket directory.
ARTIFACT_KIND_STATE = "state"
ARTIFACT_KIND_REQUEST = "request"
ARTIFACT_KIND_RESULT = "result"
ARTIFACT_KINDS = (ARTIFACT_KIND_STATE, ARTIFACT_KIND_REQUEST, ARTIFACT_KIND_RESULT)

#: The closed set of entries a conforming runtime root may contain.
ALLOWED_ROOT_ENTRIES = (
    "two-lowercase-hex bucket directory",
    f"{LOCK_DIRECTORY_NAME} directory",
    f"{RECONCILIATION_LOCK_NAME} regular file",
)
ALLOWED_BUCKET_ENTRIES = (
    f"<digest>{STATE_FILENAME_SUFFIX}",
    f"<digest>.{ATTEMPT_TOKEN_PREFIX}<attempt>{REQUEST_FILENAME_SUFFIX}",
    f"<digest>.{ATTEMPT_TOKEN_PREFIX}<attempt>{RESULT_FILENAME_SUFFIX}",
    f".<final-name>.<pid>.<random>{STAGING_FILENAME_SUFFIX} (ignored orphan staging)",
)
CENSUS_REJECTIONS = (
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
)
FORBIDDEN_ORDER_SOURCES = (
    "filesystem enumeration order",
    "completion order",
    "file mtime",
    "inode",
    "lexical filename order",
    "worker name",
    "hostname",
    "process ID",
    "wall time",
    "absolute path",
    "process-local memory",
)

# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------

#: The single documented lock order.  No code may acquire these in reverse.
LOCK_ORDER = ("global_reconciliation_lock", "per_unit_b2c_lock")
LOCK_MODE_EXCLUSIVE = "exclusive"
LOCK_MODE_SHARED = "shared"
LOCK_MODES = (LOCK_MODE_EXCLUSIVE, LOCK_MODE_SHARED)

# ---------------------------------------------------------------------------
# Scan modes
# ---------------------------------------------------------------------------

#: Production coverage.  A terminal non-mergeable unit is a HOLD here, because
#: a bounded-smoke result parked at a required unit's authoritative location
#: would permanently block that full-strength unit.
SCAN_MODE_PRODUCTION_MERGE = "production_merge"
#: Explicitly opt-in bounded-smoke inspection on an isolated non-production
#: root.  Terminal non-mergeable units are permitted and contribute zero.
SCAN_MODE_BOUNDED_SMOKE_INSPECTION = "bounded_smoke_inspection"
SCAN_MODES = (SCAN_MODE_PRODUCTION_MERGE, SCAN_MODE_BOUNDED_SMOKE_INSPECTION)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class G8BlerResumeError(RuntimeError):
    """A B3 resume/merge invariant was violated."""


class ResumeHoldError(G8BlerResumeError):
    """A contradiction that must stop the campaign rather than be classified.

    Every subclass is a HOLD.  Nothing in this module converts a HOLD into a
    benign classification, a skip, a quarantine or a deletion.
    """


class ResumeCensusError(ResumeHoldError):
    """The runtime root contains something the contract does not define."""


class ResumeChainError(ResumeHoldError):
    """A state/request/result chain does not form one exact binding."""


class ResumeContradictionError(ResumeHoldError):
    """Two authoritative sources disagree about the same work unit."""


class ResumeCampaignError(ResumeHoldError):
    """Campaign state disagrees with validated per-unit evidence."""


class ResumeContractAuthenticationError(G8BlerResumeError):
    """The registered B3 resume/merge contract failed authentication."""


class ResumeRepairError(G8BlerResumeError):
    """An explicit repair was refused or could not be completed."""


class ResumeLockError(G8BlerResumeError):
    """The global reconciliation lock could not be established."""


# ---------------------------------------------------------------------------
# Small strict helpers
# ---------------------------------------------------------------------------


def _exact_int(value: Any, name: str, error: type[Exception] = G8BlerResumeError) -> int:
    if type(value) is not int:
        raise error(f"{name} must be an exact integer, not {type(value).__name__}")
    return value


def _positive_int(value: Any, name: str, error: type[Exception] = G8BlerResumeError) -> int:
    number = _exact_int(value, name, error)
    if number <= 0:
        raise error(f"{name} must be positive")
    return number


def _nonblank_string(value: Any, name: str, error: type[Exception] = G8BlerResumeError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a non-blank string")
    return value


def _digest(value: Any, name: str, error: type[Exception] = G8BlerResumeError) -> str:
    _nonblank_string(value, name, error)
    if HEX_DIGEST_RE.fullmatch(value) is None:
        raise error(f"{name} must be a lowercase hex SHA-256 digest")
    return value


def _fresh(value: Any) -> Any:
    """Return a fresh decoded copy so no internal object can escape."""

    return json.loads(canonical_json(value))


# ---------------------------------------------------------------------------
# Attempt grammar
# ---------------------------------------------------------------------------


def parse_attempt_token(token: Any) -> int:
    """Parse one ``attempt-<n>`` token body under the exact frozen grammar."""

    _nonblank_string(token, "attempt token", ResumeCensusError)
    if ATTEMPT_TOKEN_RE.fullmatch(token) is None:
        raise ResumeCensusError(
            f"attempt token {token!r} is malformed; it must be a {ATTEMPT_GRAMMAR}"
        )
    # The regex above already fixed the grammar to unsigned base-10 digits.
    return int(token)


def format_attempt(attempt: Any) -> str:
    """Render one attempt as its exact canonical token body."""

    number = _positive_int(attempt, "attempt", ResumeCensusError)
    return str(number)


# ---------------------------------------------------------------------------
# Canonical runtime paths
# ---------------------------------------------------------------------------


def work_unit_digest(work_unit_id: str) -> str:
    """The exact SHA-256 of the UTF-8 work-unit ID bytes."""

    _nonblank_string(work_unit_id, "work_unit_id")
    return hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()


def _artifact_filename(digest: str, kind: str, attempt: Any) -> str:
    if kind == ARTIFACT_KIND_STATE:
        return f"{digest}{STATE_FILENAME_SUFFIX}"
    suffix = REQUEST_FILENAME_SUFFIX if kind == ARTIFACT_KIND_REQUEST else RESULT_FILENAME_SUFFIX
    return f"{digest}.{ATTEMPT_TOKEN_PREFIX}{format_attempt(attempt)}{suffix}"


def artifact_relative_path(
    context: Any,
    work_unit_id: str,
    kind: str,
    attempt: Any = None,
) -> PurePosixPath:
    """Return the exact ``<bucket>/<name>`` path for one artifact."""

    context = _resume_context(context)
    context.ordinal(work_unit_id)
    if kind not in ARTIFACT_KINDS:
        raise G8BlerResumeError(f"unknown artifact kind {kind!r}")
    digest = work_unit_digest(work_unit_id)
    return PurePosixPath(digest[:2]) / _artifact_filename(digest, kind, attempt)


def request_relative_path(context: Any, work_unit_id: str, attempt: Any) -> PurePosixPath:
    return artifact_relative_path(context, work_unit_id, ARTIFACT_KIND_REQUEST, attempt)


def result_relative_path(context: Any, work_unit_id: str, attempt: Any) -> PurePosixPath:
    return artifact_relative_path(context, work_unit_id, ARTIFACT_KIND_RESULT, attempt)


def logical_artifact_path(
    context: Any,
    work_unit_id: str,
    kind: str,
    attempt: Any = None,
) -> str:
    """The repository-relative logical path a unit state may record.

    This is deliberately independent of the physical root: a state written on
    one machine records the same logical path everywhere.
    """

    relative = artifact_relative_path(context, work_unit_id, kind, attempt)
    return str(PurePosixPath(WORK_UNIT_ROOT_LOGICAL_PREFIX) / relative)


def logical_result_path(context: Any, work_unit_id: str, attempt: Any) -> str:
    return logical_artifact_path(context, work_unit_id, ARTIFACT_KIND_RESULT, attempt)


def artifact_path(
    context: Any,
    work_unit_id: str,
    kind: str,
    attempt: Any = None,
    *,
    root: Path | str | None = None,
) -> Path:
    """The absolute physical path of one artifact under ``root``."""

    return _root_path(root) / artifact_relative_path(context, work_unit_id, kind, attempt)


def request_path(
    context: Any,
    work_unit_id: str,
    attempt: Any,
    *,
    root: Path | str | None = None,
) -> Path:
    return artifact_path(context, work_unit_id, ARTIFACT_KIND_REQUEST, attempt, root=root)


def result_path(
    context: Any,
    work_unit_id: str,
    attempt: Any,
    *,
    root: Path | str | None = None,
) -> Path:
    return artifact_path(context, work_unit_id, ARTIFACT_KIND_RESULT, attempt, root=root)


def state_path(
    context: Any,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    return artifact_path(context, work_unit_id, ARTIFACT_KIND_STATE, root=root)


# ---------------------------------------------------------------------------
# No-follow filesystem primitives
# ---------------------------------------------------------------------------


def _lstat(path: Path | str, *, dir_fd: int | None = None) -> os.stat_result | None:
    """No-follow inspection.  A dangling symlink is *present*, not absent.

    A read or permission error is never reported as "no state": it raises.
    """

    try:
        return os.lstat(path, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResumeCensusError(f"cannot inspect runtime path {path}: {exc}") from exc


def _root_path(root: Path | str | None) -> Path:
    value = DEFAULT_WORK_UNIT_ROOT if root is None else Path(root)
    if not value.is_absolute():
        raise ResumeCensusError("work-unit root must be an absolute path")
    entry = _lstat(value)
    if entry is not None:
        if stat.S_ISLNK(entry.st_mode):
            raise ResumeCensusError(f"work-unit root may not be a symlink: {value}")
        if not stat.S_ISDIR(entry.st_mode):
            raise ResumeCensusError(f"work-unit root is not a directory: {value}")
    return value


def _open_directory(path: Path) -> int:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ResumeCensusError(f"runtime directory may not be a symlink: {path}") from exc
        raise ResumeCensusError(f"cannot open runtime directory {path}: {exc}") from exc


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _read_canonical_file(path: Path) -> bytes:
    """Read one authoritative file no-follow, as exact bytes."""

    descriptor: int | None = None
    stream = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        stream = os.fdopen(descriptor, "rb")
    except FileNotFoundError as exc:
        raise ResumeCensusError(f"runtime file disappeared during the scan: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ResumeCensusError(f"runtime file may not be a symlink: {path}") from exc
        raise ResumeCensusError(f"cannot read runtime file {path}: {exc}") from exc
    finally:
        if stream is not None:
            descriptor = None
        _close_quietly(descriptor)
    try:
        with stream:
            return stream.read()
    except OSError as exc:
        raise ResumeCensusError(f"cannot read runtime file {path}: {exc}") from exc


def _require_regular_unaliased(entry: os.stat_result, path: Path, label: str) -> None:
    if stat.S_ISLNK(entry.st_mode):
        raise ResumeCensusError(f"{label} may not be a symlink: {path}")
    if not stat.S_ISREG(entry.st_mode):
        raise ResumeCensusError(f"{label} is not a regular file: {path}")
    if entry.st_nlink != 1:
        raise ResumeCensusError(
            f"{label} has {entry.st_nlink} hard links; an authoritative artifact "
            f"must have exactly one name: {path}"
        )


# ---------------------------------------------------------------------------
# Authenticated B3 context
# ---------------------------------------------------------------------------


def _resume_contract_identifier(payload: Mapping[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{RESUME_CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


class AuthenticatedResumeContext:
    """Authenticated B3 authority over resume, recovery and merge validation.

    Wraps an :class:`~baseline.g8_bler_work_units.AuthenticatedUnitStateContext`
    — which itself wraps the B1C
    :class:`~baseline.g8_bler_work_units.AuthenticatedExecutionContext` — and
    additionally authenticates the registered B3 resume/merge contract once
    that artifact exists.

    Authentication happens **once per context**.  The 8.6 MB required-identity
    artifact is loaded and hashed by the execution context at construction and
    never again; per-unit lookups here resolve through immutable in-memory
    tuples and read-only mappings.  Every public accessor returns a fresh
    decoded copy, so a caller mutating a returned record cannot poison a later
    call.

    ``require_resume_contract`` exists for exactly one bootstrap case: the
    generator must build the contract *before* it has been written and
    registered.  Every plan, merge report and reconciliation refuses to run
    without an authenticated registered contract.
    """

    __slots__ = (
        "_state_context",
        "_digest_index",
        "_resume_contract",
        "_resume_binding",
        "_resume_contract_path",
    )

    def __init__(
        self,
        state_context: Any = None,
        *,
        campaign_state_path: Path | str | None = None,
        state_contract_path: Path | str | None = None,
        resume_contract_path: Path | str | None = None,
        require_resume_contract: bool = False,
    ) -> None:
        if state_context is None:
            state_context = work_units.AuthenticatedUnitStateContext(
                campaign_state_path=campaign_state_path,
                state_contract_path=state_contract_path,
            )
        elif isinstance(state_context, AuthenticatedResumeContext):
            state_context = state_context.state_context
        elif isinstance(state_context, work_units.AuthenticatedExecutionContext):
            state_context = work_units.AuthenticatedUnitStateContext(
                state_context,
                campaign_state_path=campaign_state_path,
                state_contract_path=state_contract_path,
            )
        if not isinstance(state_context, work_units.AuthenticatedUnitStateContext):
            raise TypeError(
                "state_context must be an AuthenticatedUnitStateContext; a plain "
                "execution context cannot authenticate unit state"
            )

        authority = state_context.authority_binding()
        expected = {
            "campaign_id": EXPECTED_CAMPAIGN_ID,
            "campaign_manifest_sha256": EXPECTED_CAMPAIGN_MANIFEST_SHA256,
            "required_bler_artifact_sha256": EXPECTED_REQUIRED_IDENTITIES_SHA256,
            "selection_policy_sha256": EXPECTED_SELECTION_POLICY_SHA256,
            "bler_tooling_contract_id": EXPECTED_B1C_CONTRACT_ID,
            "bler_tooling_contract_sha256": EXPECTED_B1C_CONTRACT_SHA256,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "required_work_unit_count": EXPECTED_REQUIRED_WORK_UNIT_COUNT,
        }
        for field, value in expected.items():
            if authority.get(field) != value:
                raise ResumeContractAuthenticationError(
                    f"B3 authority binding differs from the immutable campaign authority: {field}"
                )
        state_binding = state_context.state_contract_binding()
        if state_binding["bler_state_contract_id"] != EXPECTED_B2C_CONTRACT_ID:
            raise ResumeContractAuthenticationError(
                "B3 requires the exact registered B2C state contract ID"
            )
        if state_binding["bler_state_contract_sha256"] != EXPECTED_B2C_CONTRACT_SHA256:
            raise ResumeContractAuthenticationError(
                "B3 requires the exact registered B2C state-contract artifact SHA-256"
            )

        self._state_context = state_context
        # One reverse index, built once from the authenticated ordered
        # authority.  A digest is otherwise a one-way function and a scanner
        # would have to rehash all 3213 IDs per file.
        self._digest_index = MappingProxyType(
            {
                work_unit_digest(work_unit_id): work_unit_id
                for work_unit_id in state_context.ordered_work_unit_ids
            }
        )
        if len(self._digest_index) != len(state_context.ordered_work_unit_ids):
            raise ResumeContractAuthenticationError(
                "two required work-unit IDs collide on the same SHA-256 digest"
            )

        contract_path = (
            DEFAULT_RESUME_CONTRACT_PATH
            if resume_contract_path is None
            else Path(resume_contract_path)
        )
        self._resume_contract_path = contract_path
        registered = self._registered_resume_binding(state_context.campaign_state_path)
        if registered is None:
            if require_resume_contract:
                raise ResumeContractAuthenticationError(
                    "campaign state does not register the B3 resume/merge contract"
                )
            self._resume_contract = None
            self._resume_binding = None
        else:
            payload, raw = self._authenticated_resume_contract(
                contract_path, registered, state_context
            )
            self._resume_contract = MappingProxyType(
                {
                    "contract_id": payload["contract_id"],
                    "schema_version": payload["schema_version"],
                    "artifact_role": payload["artifact_role"],
                    "phase": payload["phase"],
                    "checkpoint": payload["checkpoint"],
                }
            )
            self._resume_binding = MappingProxyType(
                {
                    "bler_resume_contract_id": payload["contract_id"],
                    "bler_resume_contract_sha256": sha256_bytes(raw),
                }
            )

    # -- authentication ----------------------------------------------------

    @staticmethod
    def _registered_resume_binding(state_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(state_path.read_bytes())
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResumeContractAuthenticationError(
                f"cannot read the campaign state {state_path}: {exc}"
            ) from exc
        identity = payload.get("identity") if isinstance(payload, Mapping) else None
        if not isinstance(identity, Mapping):
            raise ResumeContractAuthenticationError("campaign state has no identity block")
        artifacts = identity.get("produced_artifacts")
        if not isinstance(artifacts, list):
            raise ResumeContractAuthenticationError("campaign state has no produced-artifact list")
        matches = [
            entry
            for entry in artifacts
            if isinstance(entry, Mapping)
            and entry.get("path") == RESUME_CONTRACT_REPO_RELATIVE_PATH
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ResumeContractAuthenticationError(
                "campaign state must register at most one B3 resume/merge contract"
            )
        entry = dict(matches[0])
        if set(entry) != {"path", "sha256", "bytes"}:
            raise ResumeContractAuthenticationError(
                "registered resume-contract binding has the wrong schema"
            )
        _digest(entry["sha256"], "registered resume contract sha256", ResumeContractAuthenticationError)
        _positive_int(entry["bytes"], "registered resume contract bytes", ResumeContractAuthenticationError)
        return entry

    @staticmethod
    def _authenticated_resume_contract(
        contract_path: Path,
        registered: Mapping[str, Any],
        state_context: work_units.AuthenticatedUnitStateContext,
    ) -> tuple[dict[str, Any], bytes]:
        try:
            raw = contract_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResumeContractAuthenticationError(
                f"cannot read the B3 resume contract {contract_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ResumeContractAuthenticationError("B3 resume contract is not a JSON object")

        actual_sha256 = sha256_bytes(raw)
        if len(raw) != registered["bytes"] or actual_sha256 != registered["sha256"]:
            raise ResumeContractAuthenticationError(
                "B3 resume-contract artifact does not match its registered byte count and SHA-256"
            )
        if payload.get("artifact_role") != RESUME_CONTRACT_ARTIFACT_ROLE:
            raise ResumeContractAuthenticationError("resume contract has the wrong artifact role")
        if payload.get("schema_version") != RESUME_CONTRACT_SCHEMA_VERSION:
            raise ResumeContractAuthenticationError("resume contract has the wrong schema version")
        if payload.get("phase") != PHASE or payload.get("checkpoint") != CHECKPOINT:
            raise ResumeContractAuthenticationError("resume contract is not the G8_B/B3 contract")
        contract_id = payload.get("contract_id")
        if not isinstance(contract_id, str) or not contract_id.startswith(
            f"{RESUME_CONTRACT_ID_PREFIX}-"
        ):
            raise ResumeContractAuthenticationError("resume contract ID has the wrong prefix")
        if contract_id != _resume_contract_identifier(payload):
            raise ResumeContractAuthenticationError("resume contract ID does not reproduce")

        sources = payload.get("contract_sources")
        if not isinstance(sources, list) or [
            entry.get("path") if isinstance(entry, Mapping) else None for entry in sources
        ] != list(RESUME_CONTRACT_SOURCE_PATHS):
            raise ResumeContractAuthenticationError("resume contract source path list changed")
        for entry in sources:
            if entry["path"] == RESUME_CONTRACT_REPO_RELATIVE_PATH:
                raise ResumeContractAuthenticationError("resume contract binds its own output path")
            try:
                body = (REPO_ROOT / entry["path"]).read_bytes()
            except OSError as exc:
                raise ResumeContractAuthenticationError(
                    f"cannot read bound B3 source {entry['path']}: {exc}"
                ) from exc
            if (
                entry.get("role") != RESUME_CONTRACT_SOURCE_ROLE
                or entry.get("bytes") != len(body)
                or entry.get("sha256") != sha256_bytes(body)
            ):
                raise ResumeContractAuthenticationError(
                    f"bound B3 source changed: {entry['path']}"
                )
        if actual_sha256.encode("ascii") in raw:
            raise ResumeContractAuthenticationError("resume contract binds its own artifact SHA-256")

        authority = payload.get("authority_bindings")
        expected_authority = state_context.authority_binding()
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
            raise ResumeContractAuthenticationError(
                "resume contract authority bindings differ from the authenticated context"
            )
        state_binding = payload.get("state_contract_binding")
        expected_state = state_context.state_contract_binding()
        if not isinstance(state_binding, Mapping) or any(
            state_binding.get(field) != expected_state[field] for field in expected_state
        ):
            raise ResumeContractAuthenticationError(
                "resume contract does not bind the registered B2C state contract"
            )
        return payload, raw

    # -- accessors ---------------------------------------------------------

    @property
    def state_context(self) -> work_units.AuthenticatedUnitStateContext:
        return self._state_context

    @property
    def execution_context(self) -> work_units.AuthenticatedExecutionContext:
        return self._state_context.execution_context

    @property
    def campaign_state_path(self) -> Path:
        return self._state_context.campaign_state_path

    @property
    def resume_contract_path(self) -> Path:
        return self._resume_contract_path

    @property
    def campaign_id(self) -> str:
        return self._state_context.campaign_id

    @property
    def required_work_unit_count(self) -> int:
        return self._state_context.required_work_unit_count

    @property
    def ordered_work_unit_ids(self) -> tuple[str, ...]:
        return self._state_context.ordered_work_unit_ids

    def authority_binding(self) -> dict[str, Any]:
        return self._state_context.authority_binding()

    def state_contract_binding(self) -> dict[str, str]:
        return self._state_context.state_contract_binding()

    def resume_contract_binding(self) -> dict[str, str] | None:
        """The registered B3 binding, or ``None`` before registration."""

        if self._resume_binding is None:
            return None
        return dict(self._resume_binding)

    def require_resume_contract_binding(self) -> dict[str, str]:
        binding = self.resume_contract_binding()
        if binding is None:
            raise ResumeContractAuthenticationError(
                "this operation requires the registered B3 resume/merge contract; "
                f"campaign state does not bind {RESUME_CONTRACT_REPO_RELATIVE_PATH}"
            )
        return binding

    def ordinal(self, work_unit_id: str) -> int:
        return self._state_context.ordinal(work_unit_id)

    def work_unit_record(self, work_unit_id: str) -> dict[str, Any]:
        return self._state_context.work_unit_record(work_unit_id)

    def work_unit_record_sha256(self, work_unit_id: str) -> str:
        return self._state_context.work_unit_record_sha256(work_unit_id)

    def work_unit_id_for_digest(self, digest: str) -> str:
        """Resolve a filename digest, or raise; an unknown digest is a HOLD."""

        _digest(digest, "work-unit digest", ResumeCensusError)
        try:
            return self._digest_index[digest]
        except KeyError as exc:
            raise ResumeCensusError(
                f"runtime artifact carries an unknown work-unit digest: {digest}"
            ) from exc

    def known_digest(self, digest: str) -> bool:
        return digest in self._digest_index


def _resume_context(value: Any) -> AuthenticatedResumeContext:
    if isinstance(value, AuthenticatedResumeContext):
        return value
    raise TypeError(
        "an AuthenticatedResumeContext is required; B3 never operates on "
        "unauthenticated authority"
    )


# ---------------------------------------------------------------------------
# Global reconciliation lock
# ---------------------------------------------------------------------------

_THREAD_LOCK_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _root_thread_lock(root_path: Path) -> threading.Lock:
    key = str(root_path)
    with _THREAD_LOCK_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def reconciliation_lock(
    root: Path | str | None = None,
    *,
    mode: str = LOCK_MODE_EXCLUSIVE,
    create_missing_root: bool = False,
) -> Iterator[dict[str, Any]]:
    """Hold the global coordination lock at ``<root>/.reconciliation.lock``.

    ``exclusive`` excludes every conforming worker for the whole
    read/repair/reconcile/merge operation; ``shared`` is what a future B4+
    worker holds for its complete claim → request → execute → result →
    state-link transaction, so shared holders coexist while an exclusive
    reconciliation excludes all of them.

    Release happens on normal return, on exception and on process death: the
    kernel drops a ``flock`` when the last descriptor referring to that open
    file description closes, including at process exit.  There is no
    "continue without locking" fallback.

    An **absent root** is not an error and does not create anything.  A worker
    must create the root before it can do anything, and it creates it while
    taking this same lock, so an absent root means no worker can be mid-flight.
    That is what keeps read-only inspection from materialising the production
    runtime tree merely by looking at it.

    Lock order is fixed and never reversed: this global lock, then the per-unit
    B2C lock.  See :data:`LOCK_ORDER`.
    """

    if mode not in LOCK_MODES:
        raise ResumeLockError(f"unknown reconciliation lock mode {mode!r}")
    root_path = _root_path(root)
    if _lstat(root_path) is None:
        if not create_missing_root:
            yield {"held": False, "root_present": False, "mode": mode}
            return
        try:
            root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ResumeLockError(f"cannot create the work-unit root {root_path}: {exc}") from exc
        root_path = _root_path(root_path)

    thread_lock = _root_thread_lock(root_path) if mode == LOCK_MODE_EXCLUSIVE else None
    if thread_lock is not None:
        thread_lock.acquire()
    root_fd: int | None = None
    lock_fd: int | None = None
    try:
        root_fd = _open_directory(root_path)
        existing = _lstat(RECONCILIATION_LOCK_NAME, dir_fd=root_fd)
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise ResumeCensusError(
                    "the reconciliation lock path may not be a symlink: "
                    f"{root_path / RECONCILIATION_LOCK_NAME}"
                )
            if not stat.S_ISREG(existing.st_mode):
                raise ResumeCensusError(
                    "the reconciliation lock path is not a regular file: "
                    f"{root_path / RECONCILIATION_LOCK_NAME}"
                )
        try:
            lock_fd = os.open(
                RECONCILIATION_LOCK_NAME,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=root_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ResumeCensusError(
                    "the reconciliation lock path may not be a symlink: "
                    f"{root_path / RECONCILIATION_LOCK_NAME}"
                ) from exc
            raise ResumeLockError(f"cannot open the reconciliation lock: {exc}") from exc
        operation = fcntl.LOCK_EX if mode == LOCK_MODE_EXCLUSIVE else fcntl.LOCK_SH
        try:
            fcntl.flock(lock_fd, operation)
        except OSError as exc:
            raise ResumeLockError(f"cannot acquire the reconciliation lock: {exc}") from exc
        yield {"held": True, "root_present": True, "mode": mode}
    finally:
        # Closing the descriptor releases the flock; the kernel also releases
        # it if this process dies while holding it.
        _close_quietly(lock_fd)
        _close_quietly(root_fd)
        if thread_lock is not None:
            thread_lock.release()


# ---------------------------------------------------------------------------
# No-follow filesystem census
# ---------------------------------------------------------------------------


def _classify_bucket_filename(name: str) -> tuple[str, str, int | None]:
    """Parse one bucket filename into ``(kind, digest, attempt)``.

    Raises :class:`ResumeCensusError` for every name the contract does not
    define, including uppercase digests and malformed attempt tokens.
    """

    if name.endswith(STATE_FILENAME_SUFFIX):
        digest = name[: -len(STATE_FILENAME_SUFFIX)]
        if HEX_DIGEST_RE.fullmatch(digest) is None:
            raise ResumeCensusError(f"state filename is not a lowercase SHA-256 digest: {name}")
        return ARTIFACT_KIND_STATE, digest, None
    for suffix, kind in (
        (REQUEST_FILENAME_SUFFIX, ARTIFACT_KIND_REQUEST),
        (RESULT_FILENAME_SUFFIX, ARTIFACT_KIND_RESULT),
    ):
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        separator = stem.find(f".{ATTEMPT_TOKEN_PREFIX}")
        if separator < 0:
            raise ResumeCensusError(f"{kind} filename has no attempt token: {name}")
        digest = stem[:separator]
        token = stem[separator + 1 + len(ATTEMPT_TOKEN_PREFIX) :]
        if HEX_DIGEST_RE.fullmatch(digest) is None:
            raise ResumeCensusError(
                f"{kind} filename is not a lowercase SHA-256 digest: {name}"
            )
        return kind, digest, parse_attempt_token(token)
    raise ResumeCensusError(f"unrecognized runtime filename: {name}")


def _staging_final_name(name: str) -> str | None:
    """Return the final name a B2C staging artifact was staging, or ``None``."""

    match = _STAGING_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("final")


def census_runtime_root(
    context: Any,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Enumerate the runtime root no-follow and reject everything undefined.

    The returned record is ordered by the frozen required-work-unit order and
    by explicit numeric attempt.  Filesystem enumeration order never survives
    into it, so two scans of identical bytes on filesystems that enumerate
    differently produce identical output.

    An absent root is valid and simply means all work remains.
    """

    context = _resume_context(context)
    root_path = _root_path(root)
    ordered_ids = context.ordered_work_unit_ids

    states: dict[str, bool] = {}
    requests: dict[str, set[int]] = {}
    results: dict[str, set[int]] = {}
    ignored_staging = 0
    lock_files = 0
    reconciliation_lock_present = False
    lock_directory_present = False
    buckets: set[str] = set()

    if _lstat(root_path) is None:
        return _census_record(
            context,
            root_present=False,
            ordered_ids=ordered_ids,
            states=states,
            requests=requests,
            results=results,
            buckets=buckets,
            ignored_staging=ignored_staging,
            lock_files=lock_files,
            reconciliation_lock_present=False,
            lock_directory_present=False,
        )

    for entry in _scandir(root_path):
        name = entry.name
        entry_stat = _lstat(root_path / name)
        if entry_stat is None:
            # Vanished between enumeration and inspection; a concurrent worker
            # is excluded by the reconciliation lock, so this is a contradiction.
            raise ResumeCensusError(f"runtime entry disappeared during the census: {name}")
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ResumeCensusError(f"runtime root entry may not be a symlink: {name}")
        if name == RECONCILIATION_LOCK_NAME:
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ResumeCensusError("the reconciliation lock is not a regular file")
            reconciliation_lock_present = True
            continue
        if name == LOCK_DIRECTORY_NAME:
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise ResumeCensusError(f"{LOCK_DIRECTORY_NAME} is not a directory")
            lock_directory_present = True
            lock_files += _census_lock_directory(context, root_path / name)
            continue
        if BUCKET_RE.fullmatch(name) is None:
            raise ResumeCensusError(
                f"unknown runtime root entry {name!r}; only two-lowercase-hex buckets, "
                f"{LOCK_DIRECTORY_NAME} and {RECONCILIATION_LOCK_NAME} are defined"
            )
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise ResumeCensusError(f"bucket {name!r} is not a directory")
        buckets.add(name)
        ignored_staging += _census_bucket(
            context,
            root_path / name,
            name,
            states=states,
            requests=requests,
            results=results,
        )

    return _census_record(
        context,
        root_present=True,
        ordered_ids=ordered_ids,
        states=states,
        requests=requests,
        results=results,
        buckets=buckets,
        ignored_staging=ignored_staging,
        lock_files=lock_files,
        reconciliation_lock_present=reconciliation_lock_present,
        lock_directory_present=lock_directory_present,
    )


def _scandir(path: Path) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(path) as scan:
            return list(scan)
    except OSError as exc:
        raise ResumeCensusError(f"cannot enumerate runtime directory {path}: {exc}") from exc


def _census_lock_directory(context: AuthenticatedResumeContext, path: Path) -> int:
    count = 0
    for entry in _scandir(path):
        name = entry.name
        entry_stat = _lstat(path / name)
        if entry_stat is None:
            raise ResumeCensusError(f"lock entry disappeared during the census: {name}")
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ResumeCensusError(f"unit-state lock may not be a symlink: {name}")
        if not name.endswith(LOCK_FILENAME_SUFFIX):
            raise ResumeCensusError(f"unknown entry in {LOCK_DIRECTORY_NAME}: {name!r}")
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ResumeCensusError(f"unit-state lock is not a regular file: {name}")
        digest = name[: -len(LOCK_FILENAME_SUFFIX)]
        if HEX_DIGEST_RE.fullmatch(digest) is None:
            raise ResumeCensusError(f"unit-state lock name is not a lowercase digest: {name}")
        context.work_unit_id_for_digest(digest)
        count += 1
    return count


def _census_bucket(
    context: AuthenticatedResumeContext,
    path: Path,
    bucket: str,
    *,
    states: dict[str, bool],
    requests: dict[str, set[int]],
    results: dict[str, set[int]],
) -> int:
    ignored_staging = 0
    for entry in _scandir(path):
        name = entry.name
        entry_stat = _lstat(path / name)
        if entry_stat is None:
            raise ResumeCensusError(f"bucket entry disappeared during the census: {name}")
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ResumeCensusError(f"runtime bucket entry may not be a symlink: {bucket}/{name}")
        if stat.S_ISDIR(entry_stat.st_mode):
            raise ResumeCensusError(
                f"the contract defines no nested directory inside a bucket: {bucket}/{name}"
            )

        final_name = _staging_final_name(name)
        if final_name is not None:
            # An orphan staging artifact from a killed writer.  It is ignored
            # as evidence and counted, never read as state.
            kind, digest, _attempt = _classify_bucket_filename(final_name)
            if digest[:2] != bucket:
                raise ResumeCensusError(
                    f"staging artifact stages a foreign bucket: {bucket}/{name}"
                )
            context.work_unit_id_for_digest(digest)
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ResumeCensusError(f"staging artifact is not a regular file: {bucket}/{name}")
            ignored_staging += 1
            continue
        if name.startswith(".") or name.endswith(STAGING_FILENAME_SUFFIX):
            raise ResumeCensusError(
                f"unrecognized temporary object in the runtime tree: {bucket}/{name}"
            )

        kind, digest, attempt = _classify_bucket_filename(name)
        if digest[:2] != bucket:
            raise ResumeCensusError(
                f"runtime artifact is in the wrong bucket: expected {digest[:2]}, found {bucket}/{name}"
            )
        work_unit_id = context.work_unit_id_for_digest(digest)
        _require_regular_unaliased(entry_stat, path / name, f"{kind} artifact")

        if kind == ARTIFACT_KIND_STATE:
            if work_unit_id in states:
                raise ResumeCensusError(f"duplicate state artifact for {work_unit_id!r}")
            states[work_unit_id] = True
        elif kind == ARTIFACT_KIND_REQUEST:
            bucketed = requests.setdefault(work_unit_id, set())
            if attempt in bucketed:
                raise ResumeCensusError(
                    f"duplicate request artifact for {work_unit_id!r} attempt {attempt}"
                )
            bucketed.add(attempt)
        else:
            bucketed = results.setdefault(work_unit_id, set())
            if attempt in bucketed:
                raise ResumeCensusError(
                    f"duplicate result artifact for {work_unit_id!r} attempt {attempt}"
                )
            bucketed.add(attempt)
    return ignored_staging


def _census_record(
    context: AuthenticatedResumeContext,
    *,
    root_present: bool,
    ordered_ids: Sequence[str],
    states: Mapping[str, bool],
    requests: Mapping[str, set[int]],
    results: Mapping[str, set[int]],
    buckets: set[str],
    ignored_staging: int,
    lock_files: int,
    reconciliation_lock_present: bool,
    lock_directory_present: bool,
) -> dict[str, Any]:
    """Assemble the census in frozen authority order, never filesystem order."""

    record = {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "artifact_role": "g8_bler_runtime_census",
        "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
        "root_present": root_present,
        "reconciliation_lock_present": reconciliation_lock_present,
        "lock_directory_present": lock_directory_present,
        "lock_file_count": lock_files,
        "bucket_count": len(buckets),
        "ignored_orphan_staging_count": ignored_staging,
        "state_work_unit_ids": [
            work_unit_id for work_unit_id in ordered_ids if work_unit_id in states
        ],
        "request_attempts": {
            work_unit_id: sorted(requests[work_unit_id])
            for work_unit_id in ordered_ids
            if work_unit_id in requests
        },
        "result_attempts": {
            work_unit_id: sorted(results[work_unit_id])
            for work_unit_id in ordered_ids
            if work_unit_id in results
        },
        "required_work_unit_count": context.required_work_unit_count,
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }
    return _fresh(record)


__all__ = [
    "ALLOWED_BUCKET_ENTRIES",
    "ALLOWED_ROOT_ENTRIES",
    "ARTIFACT_KINDS",
    "ARTIFACT_KIND_REQUEST",
    "ARTIFACT_KIND_RESULT",
    "ARTIFACT_KIND_STATE",
    "ATTEMPT_GRAMMAR",
    "ATTEMPT_TOKEN_PREFIX",
    "AuthenticatedResumeContext",
    "B4_RESTART_COMMAND",
    "CAMPAIGN_ROLE",
    "CANONICAL_FILE_ENCODING",
    "CENSUS_REJECTIONS",
    "CHECKPOINT",
    "DEFAULT_RESUME_CONTRACT_PATH",
    "DEFAULT_WORK_UNIT_ROOT",
    "EXPECTED_B1C_CONTRACT_ID",
    "EXPECTED_B1C_CONTRACT_SHA256",
    "EXPECTED_B2C_CONTRACT_ID",
    "EXPECTED_B2C_CONTRACT_SHA256",
    "EXPECTED_CAMPAIGN_ID",
    "EXPECTED_CAMPAIGN_MANIFEST_SHA256",
    "EXPECTED_REQUIRED_IDENTITIES_SHA256",
    "EXPECTED_REQUIRED_WORK_UNIT_COUNT",
    "EXPECTED_SELECTION_POLICY_SHA256",
    "FORBIDDEN_ORDER_SOURCES",
    "G8BlerResumeError",
    "LOCK_MODES",
    "LOCK_MODE_EXCLUSIVE",
    "LOCK_MODE_SHARED",
    "LOCK_ORDER",
    "PHASE",
    "RECONCILIATION_LOCK_NAME",
    "REQUEST_FILENAME_SUFFIX",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_FILENAME_SUFFIX",
    "RESULT_SCHEMA_VERSION",
    "RESUME_CONTRACT_ARTIFACT_ROLE",
    "RESUME_CONTRACT_ID_PREFIX",
    "RESUME_CONTRACT_REPO_RELATIVE_PATH",
    "RESUME_CONTRACT_SCHEMA_VERSION",
    "RESUME_CONTRACT_SOURCE_PATHS",
    "RESUME_CONTRACT_SOURCE_ROLE",
    "ResumeCampaignError",
    "ResumeCensusError",
    "ResumeChainError",
    "ResumeContractAuthenticationError",
    "ResumeContradictionError",
    "ResumeHoldError",
    "ResumeLockError",
    "ResumeRepairError",
    "SCAN_MODES",
    "SCAN_MODE_BOUNDED_SMOKE_INSPECTION",
    "SCAN_MODE_PRODUCTION_MERGE",
    "STATE_FILENAME_SUFFIX",
    "UNIT_STATE_SCHEMA_VERSION",
    "WORK_UNIT_ROOT_LOGICAL_PREFIX",
    "artifact_path",
    "artifact_relative_path",
    "census_runtime_root",
    "format_attempt",
    "logical_artifact_path",
    "logical_result_path",
    "parse_attempt_token",
    "reconciliation_lock",
    "request_path",
    "request_relative_path",
    "result_path",
    "result_relative_path",
    "state_path",
    "work_unit_digest",
]
