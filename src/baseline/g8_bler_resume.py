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
# Closed per-unit classification (G8_B3 §13, as corrected by B3.H1 §29)
# ---------------------------------------------------------------------------

#: The frozen B2C ``claimed`` state.  A claim is a pre-execution reservation and
#: is *always* request-unbound; a published request file is immutable attempt
#: history, not a state transition.  B3 may never construct or accept a
#: ``claimed`` state whose ``request_sha256`` is non-null.
FROZEN_CLAIMED_STATE_FIELDS: dict[str, object] = {
    "status": work_units.STATUS_CLAIMED,
    "request_sha256": None,
    "result_path": None,
    "result_sha256": None,
    "scientific_execution_performed": False,
    "trials_completed": 0,
}

CLASSIFICATION_ABSENT = "absent"
CLASSIFICATION_CLAIMED_UNBOUND = "claimed_unbound"
CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED = "claimed_request_published"
CLASSIFICATION_RECOVERABLE_FAILED_RESULT = "recoverable_failed_result"
CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT = "recoverable_complete_result"
CLASSIFICATION_FAILED_RETRYABLE = "failed_retryable"
CLASSIFICATION_COMPLETED_FULL_STRENGTH = "completed_full_strength"
CLASSIFICATION_TERMINAL_NONMERGEABLE = "terminal_nonmergeable"

#: Exactly the eight reachable classes.  Anything outside this set is a typed
#: HOLD, never a benign classification.
CLASSIFICATIONS = (
    CLASSIFICATION_ABSENT,
    CLASSIFICATION_CLAIMED_UNBOUND,
    CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED,
    CLASSIFICATION_RECOVERABLE_FAILED_RESULT,
    CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT,
    CLASSIFICATION_FAILED_RETRYABLE,
    CLASSIFICATION_COMPLETED_FULL_STRENGTH,
    CLASSIFICATION_TERMINAL_NONMERGEABLE,
)

#: Rejected as unreachable by B3.H1.  These named a ``claimed`` state carrying a
#: bound ``request_sha256``, which the frozen B2C schema forbids.  They are
#: retained only so tests can assert their absence; they are live nowhere.
REJECTED_UNREACHABLE_CLASSIFICATIONS = (
    "claimed_request_bound",
    "recoverable_request_binding",
)

#: The exact two-row repair matrix.  There is no request-only state repair: a
#: published request does not bind itself into a claimed state.
REPAIR_MATRIX = (
    (CLASSIFICATION_RECOVERABLE_FAILED_RESULT, work_units.STATUS_FAILED),
    (CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT, work_units.STATUS_RESULT_LINKED),
)

#: Classes ``--repair-recoverable`` may transition.  Read-only inspection never
#: repairs, and no other class is ever modified by repair.
REPAIRABLE_CLASSIFICATIONS = tuple(name for name, _ in REPAIR_MATRIX)
NON_REPAIRABLE_CLASSIFICATIONS = tuple(
    name for name in CLASSIFICATIONS if name not in REPAIRABLE_CLASSIFICATIONS
)

#: Recoverable IDs in a resume plan are exactly the repairable classes.
RECOVERABLE_CLASSIFICATIONS = REPAIRABLE_CLASSIFICATIONS
#: Remaining work.  ``claimed_request_published`` is remaining work, not
#: recoverable evidence.
REMAINING_CLASSIFICATIONS = (
    CLASSIFICATION_ABSENT,
    CLASSIFICATION_CLAIMED_UNBOUND,
    CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED,
    CLASSIFICATION_FAILED_RETRYABLE,
)
TERMINAL_CLASSIFICATIONS = (
    CLASSIFICATION_COMPLETED_FULL_STRENGTH,
    CLASSIFICATION_TERMINAL_NONMERGEABLE,
)

#: Proposed next attempt per classification.  ``None`` means no attempt is
#: proposed: terminal, or awaiting explicit repair.
PROPOSED_ATTEMPT_POLICY: dict[str, str | None] = {
    CLASSIFICATION_ABSENT: "attempt_1",
    CLASSIFICATION_CLAIMED_UNBOUND: "old_attempt_plus_1",
    CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED: "old_attempt_plus_1",
    CLASSIFICATION_FAILED_RETRYABLE: "old_attempt_plus_1",
    CLASSIFICATION_RECOVERABLE_FAILED_RESULT: None,
    CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT: None,
    CLASSIFICATION_COMPLETED_FULL_STRENGTH: None,
    CLASSIFICATION_TERMINAL_NONMERGEABLE: None,
}

#: Where a repaired unit lands on the next scan.  A complete result becomes
#: ``completed_full_strength`` in production merge mode, or
#: ``terminal_nonmergeable`` for a valid bounded-smoke result under explicit
#: bounded-smoke inspection.
POST_REPAIR_CLASSIFICATIONS: dict[str, tuple[str, ...]] = {
    CLASSIFICATION_RECOVERABLE_FAILED_RESULT: (CLASSIFICATION_FAILED_RETRYABLE,),
    CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT: (
        CLASSIFICATION_COMPLETED_FULL_STRENGTH,
        CLASSIFICATION_TERMINAL_NONMERGEABLE,
    ),
}


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
# The frozen classification rule (G8_B3 §13, as corrected by B3.H1 §29)
# ---------------------------------------------------------------------------


def classification_for_shape(
    *,
    state_status: str | None,
    state_request_bound: bool,
    request_present: bool,
    result_status: str | None,
    result_merge_eligible: bool = True,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> str:
    """Map an already-validated per-unit shape onto the closed enum.

    This is the classification *rule* alone.  It assumes every byte-level check
    of §11 and §12 has already succeeded, and it decides nothing about
    filesystem layout, digests or merge eligibility.  B3.2's classifier
    validates the chain and then defers to exactly this rule, so the corrected
    model is frozen in one place rather than restated per call site.

    ``state_status`` is ``None`` when no state exists.  ``result_status`` is the
    current attempt's result status, or ``None`` when no result exists.
    """

    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")

    if state_status is None:
        if request_present or result_status is not None:
            raise ResumeContradictionError(
                "no unit state exists but a request or result artifact does"
            )
        return CLASSIFICATION_ABSENT

    if state_status == work_units.STATUS_CLAIMED:
        # B2C keeps a claim request-unbound.  A bound claim cannot exist, so
        # reaching here means a forged state slipped past validation.
        if state_request_bound:
            raise ResumeContradictionError(
                "claimed state carries a bound request_sha256, which B2C forbids"
            )
        if result_status is None:
            # A published request is immutable attempt history, not a state
            # binding: both of these are remaining work, not recoverable.
            return (
                CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED
                if request_present
                else CLASSIFICATION_CLAIMED_UNBOUND
            )
        if not request_present:
            raise ResumeContradictionError("a result exists without its exact request")
        if result_status == bler_contract.STATUS_FAILED:
            return CLASSIFICATION_RECOVERABLE_FAILED_RESULT
        if result_status == bler_contract.STATUS_COMPLETE:
            return CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT
        raise ResumeContradictionError(
            f"result status {result_status!r} cannot be recovered from a claimed state"
        )

    if state_status == work_units.STATUS_FAILED:
        if result_status == bler_contract.STATUS_COMPLETE:
            raise ResumeContradictionError("failed state with a complete result for that attempt")
        if result_status is not None and not request_present:
            raise ResumeContradictionError("a result exists without its exact request")
        return CLASSIFICATION_FAILED_RETRYABLE

    if state_status == work_units.STATUS_RESULT_LINKED:
        if not request_present or result_status != bler_contract.STATUS_COMPLETE:
            raise ResumeContradictionError(
                "result_linked state without its exact request and complete result"
            )
        if result_merge_eligible:
            return CLASSIFICATION_COMPLETED_FULL_STRENGTH
        # A terminal non-mergeable unit parked at a required production
        # location would permanently block that full-strength unit.
        if scan_mode != SCAN_MODE_BOUNDED_SMOKE_INSPECTION:
            raise ResumeContradictionError(
                "terminal non-mergeable result at a required production work unit"
            )
        return CLASSIFICATION_TERMINAL_NONMERGEABLE

    raise ResumeHoldError(f"unknown unit-state status {state_status!r}")


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


# ---------------------------------------------------------------------------
# Request / result chain validation (G8_B3 §11, §12)
# ---------------------------------------------------------------------------


def _read_exact_artifact(path: Path, label: str) -> tuple[bytes, str, dict[str, Any]]:
    """Read one authoritative artifact and require exact canonical JSON bytes.

    Returns the raw bytes, their SHA-256 and the decoded payload.  A file whose
    bytes are semantically valid JSON but not the exact canonical encoding is a
    HOLD: a digest computed over re-rendered bytes would silently disagree with
    the digest a peer computes over the file.
    """

    entry = _lstat(path)
    if entry is None:
        raise ResumeChainError(f"{label} is absent: {path}")
    _require_regular_unaliased(entry, path, label)
    raw = _read_canonical_file(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ResumeChainError(f"{label} is not decodable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ResumeChainError(f"{label} is not a JSON object: {path}")
    if canonical_json(payload) != raw:
        raise ResumeChainError(f"{label} is not exact canonical JSON bytes: {path}")
    return raw, sha256_bytes(raw), payload


def read_unit_state_snapshot(
    context: Any,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Read and fully validate one unit state, or return ``None`` if absent.

    ``None`` means *no state file exists*.  A malformed, unreadable, aliased or
    non-canonical state is never reported as absent; it raises.
    """

    context = _resume_context(context)
    path = state_path(context, work_unit_id, root=root)
    entry = _lstat(path)
    if entry is None:
        return None
    _require_regular_unaliased(entry, path, "unit state")
    raw, digest, _payload = _read_exact_artifact(path, "unit state")
    try:
        state = work_units.read_unit_state(context.state_context, path, root=_root_path(root))
    except work_units.G8BlerWorkUnitError as exc:
        raise ResumeChainError(f"unit state failed B2C validation: {path}: {exc}") from exc
    identity = state["identity"]
    if identity["work_unit_id"] != work_unit_id:
        raise ResumeContradictionError(
            "unit-state path digest does not correspond to the embedded work-unit ID"
        )
    record = _fresh(state)
    record["state_sha256"] = digest
    record["state_bytes"] = len(raw)
    return record


def validate_request_file(
    context: Any,
    work_unit_id: str,
    attempt: Any,
    *,
    root: Path | str | None = None,
    require_full_strength: bool = True,
) -> dict[str, Any]:
    """Validate one attempt's request file against the frozen B1C contract."""

    context = _resume_context(context)
    number = _positive_int(attempt, "attempt", ResumeChainError)
    path = request_path(context, work_unit_id, number, root=root)
    raw, digest, payload = _read_exact_artifact(path, "work-unit request")
    try:
        request = (
            bler_contract.require_full_strength_request(payload)
            if require_full_strength
            else bler_contract.validate_work_unit_request(payload)
        )
    except Exception as exc:  # noqa: BLE001 - the contract raises its own hierarchy
        raise ResumeChainError(f"work-unit request failed B1C validation: {path}: {exc}") from exc

    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ResumeChainError(
            f"work-unit request schema version {request['schema_version']} is not "
            f"{REQUEST_SCHEMA_VERSION}"
        )
    if request["work_unit_id"] != work_unit_id:
        raise ResumeContradictionError(
            "request filename digest does not correspond to the embedded work-unit ID"
        )
    authority = context.authority_binding()
    for field in (
        "campaign_id",
        "campaign_manifest_sha256",
        "required_bler_artifact_sha256",
        "selection_policy_sha256",
        "bler_tooling_contract_id",
        "bler_tooling_contract_sha256",
    ):
        if request[field] != authority[field]:
            raise ResumeChainError(f"work-unit request carries a foreign {field}")
    if request["test_split_access"] != bler_contract.TEST_SPLIT_ACCESS:
        raise ResumeChainError("work-unit request does not declare zero test-split access")

    # The request digest a peer computes over the *file* must equal the B1C
    # canonical digest of its content; otherwise two workers disagree.
    if bler_contract.request_digest(request) != digest:
        raise ResumeChainError("work-unit request digest does not reproduce from its file bytes")

    return _fresh(
        {
            "work_unit_id": work_unit_id,
            "attempt": number,
            "request": request,
            "request_sha256": digest,
            "request_bytes": len(raw),
            "logical_path": logical_artifact_path(
                context, work_unit_id, ARTIFACT_KIND_REQUEST, number
            ),
            "execution_class": request["execution_class"],
        }
    )


def validate_result_file(
    context: Any,
    work_unit_id: str,
    attempt: Any,
    *,
    root: Path | str | None = None,
    request_record: Mapping[str, Any] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Validate one attempt's result against its exact request.

    A result is never validated in isolation: §12 requires locating the exact
    request for the same work unit *and the same attempt* first, so a result
    can never be credited against a request it did not run.
    """

    context = _resume_context(context)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    number = _positive_int(attempt, "attempt", ResumeChainError)
    if request_record is None:
        request_record = validate_request_file(
            context,
            work_unit_id,
            number,
            root=root,
            require_full_strength=scan_mode == SCAN_MODE_PRODUCTION_MERGE,
        )
    if request_record["attempt"] != number:
        raise ResumeContradictionError("a result was offered against another attempt's request")

    path = result_path(context, work_unit_id, number, root=root)
    raw, digest, payload = _read_exact_artifact(path, "work-unit result")
    try:
        result = bler_contract.validate_work_unit_result(
            payload, request=request_record["request"]
        )
    except Exception as exc:  # noqa: BLE001 - the contract raises its own hierarchy
        raise ResumeChainError(f"work-unit result failed B1C validation: {path}: {exc}") from exc

    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResumeChainError(
            f"work-unit result schema version {result['schema_version']} is not "
            f"{RESULT_SCHEMA_VERSION}"
        )
    identity = result["identity"]
    if identity["work_unit_id"] != work_unit_id:
        raise ResumeContradictionError(
            "result filename digest does not correspond to the embedded work-unit ID"
        )
    if identity["request_sha256"] != request_record["request_sha256"]:
        raise ResumeContradictionError(
            "work-unit result binds a request SHA-256 other than its exact request file"
        )

    metadata = result["execution_metadata"]
    disposition = result["disposition"]
    status = result["status"]
    merge_candidate = (
        status == bler_contract.STATUS_COMPLETE
        and scan_mode == SCAN_MODE_PRODUCTION_MERGE
        and bool(disposition["merge_eligible"])
    )
    if merge_candidate:
        if metadata["attempt"] is None:
            raise ResumeChainError("a production merge candidate must record its attempt")
        if metadata["attempt"] != number:
            raise ResumeContradictionError(
                "result execution metadata records a different attempt than its path"
            )
        if metadata["shard_index"] is None or metadata["shard_count"] is None:
            raise ResumeChainError("a production merge candidate must record its shard assignment")
        if shard_index is not None and metadata["shard_index"] != shard_index:
            raise ResumeContradictionError(
                "result shard index differs from the current unit-state shard assignment"
            )
        if shard_count is not None and metadata["shard_count"] != shard_count:
            raise ResumeContradictionError(
                "result shard count differs from the current unit-state shard assignment"
            )
    elif metadata["attempt"] is not None and metadata["attempt"] != number:
        raise ResumeContradictionError(
            "result execution metadata records a different attempt than its path"
        )

    if disposition["test_split_access"] != bler_contract.TEST_SPLIT_ACCESS:
        raise ResumeChainError("work-unit result does not declare zero test-split access")

    return _fresh(
        {
            "work_unit_id": work_unit_id,
            "attempt": number,
            "status": status,
            "result": result,
            "result_sha256": digest,
            "result_bytes": len(raw),
            "logical_path": logical_result_path(context, work_unit_id, number),
            "request_sha256": request_record["request_sha256"],
            "merge_eligible": bool(disposition["merge_eligible"]),
            "required_coverage_contribution": disposition["required_coverage_contribution"],
            "trials_completed": result["measurement"]["trials_completed"],
        }
    )


def is_full_strength_merge_candidate(
    context: Any,
    result_record: Mapping[str, Any],
    request_record: Mapping[str, Any],
) -> bool:
    """Exactly the §12 production merge conditions, with no partial credit."""

    context = _resume_context(context)
    result = result_record["result"]
    request = request_record["request"]
    disposition = result["disposition"]
    measurement = result["measurement"]
    full_trials = bler_contract.full_strength_trial_count()
    return (
        result["status"] == bler_contract.STATUS_COMPLETE
        and result["identity"]["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH
        and request["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH
        and result["identity"]["trials_requested"] == full_trials
        and measurement["trials_completed"] == full_trials
        and disposition["scientific_evidence"] is True
        and disposition["merge_eligible"] is True
        and disposition["required_coverage_contribution"] == 1
        and disposition["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS
        and result["identity"]["request_sha256"] == request_record["request_sha256"]
    )


# ---------------------------------------------------------------------------
# Closed per-unit classification (G8_B3 §13, §14, §15)
# ---------------------------------------------------------------------------


def _attempts_for(census: Mapping[str, Any], key: str, work_unit_id: str) -> tuple[int, ...]:
    return tuple(census[key].get(work_unit_id, ()))


def classify_work_unit(
    context: Any,
    work_unit_id: str,
    census: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Assign exactly one closed classification to one required work unit.

    Every byte-level check of §11 and §12 runs first; the corrected §13 rule in
    :func:`classification_for_shape` then names the class.  Nothing benign is
    inferred from silence: a contradiction raises rather than downgrading to
    ``absent``.
    """

    context = _resume_context(context)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    context.ordinal(work_unit_id)

    request_attempts = _attempts_for(census, "request_attempts", work_unit_id)
    result_attempts = _attempts_for(census, "result_attempts", work_unit_id)
    state = read_unit_state_snapshot(context, work_unit_id, root=root)

    if state is None:
        # §14: an artifact without its state is a contradiction, never absent.
        if request_attempts or result_attempts:
            raise ResumeContradictionError(
                f"{work_unit_id} has attempt artifacts but no unit state"
            )
        return _classification_record(
            context,
            work_unit_id,
            CLASSIFICATION_ABSENT,
            attempt=None,
            state=None,
            request_record=None,
            result_record=None,
        )

    identity = state["identity"]
    attempt = identity["attempt"]
    status = identity["status"]

    # §15: attempts are append-only history and may never run ahead of state.
    for label, attempts in (("request", request_attempts), ("result", result_attempts)):
        ahead = [value for value in attempts if value > attempt]
        if ahead:
            raise ResumeContradictionError(
                f"{work_unit_id} has a {label} for attempt {min(ahead)} beyond state attempt {attempt}"
            )

    request_record = None
    if attempt in request_attempts:
        request_record = validate_request_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            require_full_strength=scan_mode == SCAN_MODE_PRODUCTION_MERGE,
        )
    result_record = None
    if attempt in result_attempts:
        if request_record is None:
            raise ResumeContradictionError(
                f"{work_unit_id} has a result for attempt {attempt} without its exact request"
            )
        result_record = validate_result_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            request_record=request_record,
            shard_index=identity["shard_index"],
            shard_count=identity["shard_count"],
            scan_mode=scan_mode,
        )

    _require_state_matches_artifacts(context, state, request_record, result_record)
    _require_no_stranded_older_evidence(
        context,
        work_unit_id,
        attempt,
        request_attempts,
        result_attempts,
        root=root,
        scan_mode=scan_mode,
    )

    merge_eligible = True
    if result_record is not None and result_record["status"] == bler_contract.STATUS_COMPLETE:
        merge_eligible = is_full_strength_merge_candidate(
            context, result_record, request_record
        )

    classification = classification_for_shape(
        state_status=status,
        state_request_bound=identity["request_sha256"] is not None,
        request_present=request_record is not None,
        result_status=None if result_record is None else result_record["status"],
        result_merge_eligible=merge_eligible,
        scan_mode=scan_mode,
    )
    return _classification_record(
        context,
        work_unit_id,
        classification,
        attempt=attempt,
        state=state,
        request_record=request_record,
        result_record=result_record,
    )


def _require_state_matches_artifacts(
    context: AuthenticatedResumeContext,
    state: Mapping[str, Any],
    request_record: Mapping[str, Any] | None,
    result_record: Mapping[str, Any] | None,
) -> None:
    """§14: a bound state digest or path must reproduce from the exact bytes."""

    identity = state["identity"]
    work_unit_id = identity["work_unit_id"]
    bound_request = identity["request_sha256"]
    bound_result = identity["result_sha256"]
    bound_path = identity["result_path"]

    if bound_request is not None:
        if request_record is None:
            raise ResumeContradictionError(
                f"{work_unit_id} binds a request SHA-256 but its exact request file is absent"
            )
        if bound_request != request_record["request_sha256"]:
            raise ResumeContradictionError(
                f"{work_unit_id} state request SHA-256 differs from its request file bytes"
            )
    if bound_result is not None:
        if result_record is None:
            raise ResumeContradictionError(
                f"{work_unit_id} binds a result SHA-256 but its exact result file is absent"
            )
        if bound_result != result_record["result_sha256"]:
            raise ResumeContradictionError(
                f"{work_unit_id} state result SHA-256 differs from its result file bytes"
            )
    if bound_path is not None:
        expected = logical_result_path(context, work_unit_id, identity["attempt"])
        if bound_path != expected:
            raise ResumeContradictionError(
                f"{work_unit_id} state result path is not the exact derived current-attempt path"
            )
    if identity["status"] == work_units.STATUS_RESULT_LINKED and result_record is None:
        raise ResumeContradictionError(
            f"{work_unit_id} is result_linked but has no current-attempt result"
        )
    if (
        identity["status"] == work_units.STATUS_RESULT_LINKED
        and result_record["status"] != bler_contract.STATUS_COMPLETE
    ):
        raise ResumeContradictionError(
            f"{work_unit_id} is result_linked but its result is not complete"
        )
    if (
        identity["status"] == work_units.STATUS_FAILED
        and result_record is not None
        and result_record["status"] == bler_contract.STATUS_COMPLETE
    ):
        raise ResumeContradictionError(
            f"{work_unit_id} is failed but a complete result exists for that attempt"
        )


def _require_no_stranded_older_evidence(
    context: AuthenticatedResumeContext,
    work_unit_id: str,
    attempt: int,
    request_attempts: Sequence[int],
    result_attempts: Sequence[int],
    *,
    root: Path | str | None,
    scan_mode: str,
) -> None:
    """§15: older attempts stay as history, but may not hide merge evidence.

    An older *complete, merge-eligible* result means the campaign already had
    valid coverage and then advanced past it — silently dropping required
    scientific evidence.  Older failed and non-mergeable results are ordinary
    history and contribute zero.
    """

    for older in sorted(value for value in result_attempts if value < attempt):
        if older not in request_attempts:
            raise ResumeContradictionError(
                f"{work_unit_id} has an attempt-{older} result without its exact request"
            )
        older_request = validate_request_file(
            context,
            work_unit_id,
            older,
            root=root,
            require_full_strength=scan_mode == SCAN_MODE_PRODUCTION_MERGE,
        )
        older_result = validate_result_file(
            context,
            work_unit_id,
            older,
            root=root,
            request_record=older_request,
            scan_mode=scan_mode,
        )
        if older_result["status"] != bler_contract.STATUS_COMPLETE:
            continue
        if is_full_strength_merge_candidate(context, older_result, older_request):
            raise ResumeContradictionError(
                f"{work_unit_id} has a complete merge-eligible attempt-{older} result "
                f"while state advanced to attempt {attempt}"
            )


def _classification_record(
    context: AuthenticatedResumeContext,
    work_unit_id: str,
    classification: str,
    *,
    attempt: int | None,
    state: Mapping[str, Any] | None,
    request_record: Mapping[str, Any] | None,
    result_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One deterministic per-unit record.  No path, host, PID or time enters it."""

    if classification not in CLASSIFICATIONS:
        raise ResumeHoldError(f"{classification!r} is not a closed B3 classification")
    coverage = 1 if classification == CLASSIFICATION_COMPLETED_FULL_STRENGTH else 0
    record = {
        "work_unit_id": work_unit_id,
        "canonical_ordinal": context.ordinal(work_unit_id),
        "classification": classification,
        "attempt": attempt,
        "state_status": None if state is None else state["identity"]["status"],
        "state_sha256": None if state is None else state["state_sha256"],
        "shard_index": None if state is None else state["identity"]["shard_index"],
        "shard_count": None if state is None else state["identity"]["shard_count"],
        "request_sha256": None if request_record is None else request_record["request_sha256"],
        "result_sha256": None if result_record is None else result_record["result_sha256"],
        "result_status": None if result_record is None else result_record["status"],
        "trials_completed": (
            0 if result_record is None else result_record["trials_completed"]
        ),
        "required_coverage_contribution": coverage,
        "proposed_attempt": proposed_attempt(classification, attempt),
        "repairable": classification in REPAIRABLE_CLASSIFICATIONS,
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }
    return _fresh(record)


def proposed_attempt(classification: str, attempt: int | None) -> int | None:
    """§17: the exact next attempt, or ``None`` when none is proposed."""

    if classification not in PROPOSED_ATTEMPT_POLICY:
        raise ResumeHoldError(f"{classification!r} is not a closed B3 classification")
    policy = PROPOSED_ATTEMPT_POLICY[classification]
    if policy is None:
        return None
    if policy == "attempt_1":
        return 1
    return _positive_int(attempt, "attempt", ResumeHoldError) + 1


def classify_runtime_root(
    context: Any,
    census: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> tuple[dict[str, Any], ...]:
    """Classify every required work unit in frozen authority order."""

    context = _resume_context(context)
    return tuple(
        classify_work_unit(context, work_unit_id, census, root=root, scan_mode=scan_mode)
        for work_unit_id in context.ordered_work_unit_ids
    )


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
    "CLASSIFICATIONS",
    "CLASSIFICATION_ABSENT",
    "CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED",
    "CLASSIFICATION_CLAIMED_UNBOUND",
    "CLASSIFICATION_COMPLETED_FULL_STRENGTH",
    "CLASSIFICATION_FAILED_RETRYABLE",
    "CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT",
    "CLASSIFICATION_RECOVERABLE_FAILED_RESULT",
    "CLASSIFICATION_TERMINAL_NONMERGEABLE",
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
    "FROZEN_CLAIMED_STATE_FIELDS",
    "G8BlerResumeError",
    "LOCK_MODES",
    "LOCK_MODE_EXCLUSIVE",
    "LOCK_MODE_SHARED",
    "LOCK_ORDER",
    "NON_REPAIRABLE_CLASSIFICATIONS",
    "PHASE",
    "POST_REPAIR_CLASSIFICATIONS",
    "PROPOSED_ATTEMPT_POLICY",
    "RECONCILIATION_LOCK_NAME",
    "RECOVERABLE_CLASSIFICATIONS",
    "REJECTED_UNREACHABLE_CLASSIFICATIONS",
    "REMAINING_CLASSIFICATIONS",
    "REPAIRABLE_CLASSIFICATIONS",
    "REPAIR_MATRIX",
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
    "TERMINAL_CLASSIFICATIONS",
    "UNIT_STATE_SCHEMA_VERSION",
    "WORK_UNIT_ROOT_LOGICAL_PREFIX",
    "artifact_path",
    "artifact_relative_path",
    "census_runtime_root",
    "classification_for_shape",
    "classify_runtime_root",
    "classify_work_unit",
    "format_attempt",
    "is_full_strength_merge_candidate",
    "logical_artifact_path",
    "logical_result_path",
    "parse_attempt_token",
    "proposed_attempt",
    "read_unit_state_snapshot",
    "reconciliation_lock",
    "request_path",
    "request_relative_path",
    "result_path",
    "result_relative_path",
    "state_path",
    "validate_request_file",
    "validate_result_file",
    "work_unit_digest",
]
