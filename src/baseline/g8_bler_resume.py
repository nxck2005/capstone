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
import math
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from baseline import g8_bler_contract as bler_contract
from baseline import g8_campaign
from baseline import g8_bler_work_units as work_units
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes
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
        "_authority",
        "_required_record_bytes",
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

        authority_binding = dict(state_context.authority_binding())
        required_record_bytes = {
            work_unit_id: bytes(
                state_context.execution_context.work_unit_record_bytes(work_unit_id)
            )
            for work_unit_id in state_context.ordered_work_unit_ids
        }
        self._state_context = state_context
        self._authority = MappingProxyType(authority_binding)
        self._required_record_bytes = MappingProxyType(required_record_bytes)
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
        return dict(self._authority)

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
        try:
            return json.loads(self._required_record_bytes[work_unit_id])
        except KeyError as exc:
            raise ResumeContractAuthenticationError(
                f"work unit {work_unit_id!r} is not an exact required BLER identity"
            ) from exc

    def work_unit_record_sha256(self, work_unit_id: str) -> str:
        try:
            return sha256_bytes(self._required_record_bytes[work_unit_id])
        except KeyError as exc:
            raise ResumeContractAuthenticationError(
                f"work unit {work_unit_id!r} is not an exact required BLER identity"
            ) from exc

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


def _canonical_physical_root(root: Path | str | None) -> Path:
    """Return the physical root identity without creating any path component."""

    root_path = _root_path(root)
    return Path(os.path.realpath(os.fspath(root_path)))


def _open_lock_parent(root_path: Path) -> tuple[int, os.stat_result, Path]:
    """Open the existing physical parent inode that coordinates this root."""

    parent = root_path.parent
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ResumeLockError(f"work-unit root parent may not be a symlink: {parent}") from exc
        raise ResumeLockError(
            f"cannot open the existing work-unit root parent {parent}; locking is unavailable: {exc}"
        ) from exc
    try:
        parent_stat = os.fstat(parent_fd)
    except OSError as exc:
        _close_quietly(parent_fd)
        raise ResumeLockError(f"cannot stat the work-unit root parent {parent}: {exc}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode):  # pragma: no cover - O_DIRECTORY already enforces this
        _close_quietly(parent_fd)
        raise ResumeLockError(f"work-unit root parent is not a directory: {parent}")
    return parent_fd, parent_stat, parent


def _create_root_under_parent(root_path: Path, parent_fd: int) -> None:
    """Create only an explicitly requested root, relative to its locked parent."""

    name = root_path.name
    if not name or name in {".", ".."}:
        raise ResumeLockError(f"work-unit root has no safe final component: {root_path}")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ResumeLockError(f"cannot create the work-unit root under its locked parent: {exc}") from exc
    entry = _lstat(root_path)
    if entry is None:
        raise ResumeLockError(f"work-unit root disappeared after creation: {root_path}")
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise ResumeLockError(f"work-unit root is not a real directory: {root_path}")


class ReconciliationLease:
    """Opaque process-owned lease over the root parent's directory inode."""

    __slots__ = (
        "_canonical_root",
        "_mode",
        "_owner_pid",
        "_active",
        "_parent_device",
        "_parent_inode",
        "_parent_fd",
        "_root_present",
    )

    def __init__(
        self,
        *,
        canonical_root: Path,
        mode: str,
        owner_pid: int,
        parent_stat: os.stat_result,
        parent_fd: int,
        root_present: bool,
    ) -> None:
        self._canonical_root = canonical_root
        self._mode = mode
        self._owner_pid = owner_pid
        self._active = True
        self._parent_device = int(parent_stat.st_dev)
        self._parent_inode = int(parent_stat.st_ino)
        self._parent_fd = parent_fd
        self._root_present = root_present

    @property
    def canonical_root(self) -> Path:
        return self._canonical_root

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def active(self) -> bool:
        return self._active

    @property
    def parent_device(self) -> int:
        return self._parent_device

    @property
    def parent_inode(self) -> int:
        return self._parent_inode

    @property
    def root_present(self) -> bool:
        return self._root_present

    def _assert_usable(self, root: Path | str | None, required_mode: str) -> None:
        if not self._active:
            raise ResumeLockError("global reconciliation lease is inactive")
        if self._owner_pid != os.getpid():
            raise ResumeLockError("global reconciliation lease was inherited across fork")
        if required_mode not in LOCK_MODES:
            raise ResumeLockError(f"unknown required lock mode {required_mode!r}")
        if required_mode == LOCK_MODE_EXCLUSIVE and self._mode != LOCK_MODE_EXCLUSIVE:
            raise ResumeLockError("an exclusive global lease is required")
        if self._mode not in LOCK_MODES:
            raise ResumeLockError("global reconciliation lease has an invalid mode")
        if _canonical_physical_root(root) != self._canonical_root:
            raise ResumeLockError("global reconciliation lease belongs to another physical root")
        try:
            current = os.fstat(self._parent_fd)
        except OSError as exc:
            raise ResumeLockError(f"cannot revalidate the locked parent directory: {exc}") from exc
        if (int(current.st_dev), int(current.st_ino)) != (
            self._parent_device,
            self._parent_inode,
        ):
            raise ResumeLockError("locked parent directory identity changed")

    def _release(self) -> None:
        if not self._active:
            return
        self._active = False
        _close_quietly(self._parent_fd)


@contextmanager
def reconciliation_lock(
    root: Path | str | None = None,
    *,
    mode: str = LOCK_MODE_EXCLUSIVE,
    create_missing_root: bool = False,
) -> Iterator[ReconciliationLease]:
    """Hold a flock directly on the existing parent directory inode.

    The parent is the lock target, not a file inside the runtime root.  This
    makes an absent root participate in the same lock domain as an existing
    root.  Inspection defaults to non-creating operation; only an explicit
    ``create_missing_root`` request may create the final root component after
    the parent lease is held.
    """

    if mode not in LOCK_MODES:
        raise ResumeLockError(f"unknown reconciliation lock mode {mode!r}")
    root_path = _root_path(root)
    canonical_root = _canonical_physical_root(root_path)
    parent_fd, parent_stat, _parent = _open_lock_parent(root_path)
    operation = fcntl.LOCK_EX if mode == LOCK_MODE_EXCLUSIVE else fcntl.LOCK_SH
    try:
        try:
            fcntl.flock(parent_fd, operation)
        except OSError as exc:
            if exc.errno in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise ResumeLockError(
                    f"directory locking is unsupported for the work-unit root parent: {exc}"
                ) from exc
            raise ResumeLockError(f"cannot acquire the global parent-directory lock: {exc}") from exc

        if create_missing_root and _lstat(root_path) is None:
            if mode != LOCK_MODE_SHARED:
                _create_root_under_parent(root_path, parent_fd)
            else:
                raise ResumeLockError("a shared lease may not create the runtime root")
        root_present = _lstat(root_path) is not None
        lease = ReconciliationLease(
            canonical_root=canonical_root,
            mode=mode,
            owner_pid=os.getpid(),
            parent_stat=parent_stat,
            parent_fd=parent_fd,
            root_present=root_present,
        )
        try:
            yield lease
        finally:
            lease._release()
    except BaseException:
        _close_quietly(parent_fd)
        raise


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


def _census_runtime_root_locked(
    context: Any,
    *,
    root: Path | str | None = None,
    lease: ReconciliationLease,
) -> dict[str, Any]:
    """Enumerate the runtime root no-follow and reject everything undefined.

    The returned record is ordered by the frozen required-work-unit order and
    by explicit numeric attempt.  Filesystem enumeration order never survives
    into it, so two scans of identical bytes on filesystems that enumerate
    differently produce identical output.

    An absent root is valid and simply means all work remains.
    """

    context = _resume_context(context)
    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    root_path = _root_path(root)
    ordered_ids = context.ordered_work_unit_ids

    states: dict[str, bool] = {}
    requests: dict[str, set[int]] = {}
    results: dict[str, set[int]] = {}
    ignored_staging = 0
    lock_files = 0
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
        if name == LOCK_DIRECTORY_NAME:
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise ResumeCensusError(f"{LOCK_DIRECTORY_NAME} is not a directory")
            lock_directory_present = True
            lock_files += _census_lock_directory(context, root_path / name)
            continue
        if BUCKET_RE.fullmatch(name) is None:
            raise ResumeCensusError(
                f"unknown runtime root entry {name!r}; only two-lowercase-hex buckets, "
                f"and {LOCK_DIRECTORY_NAME} are defined"
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
        lock_directory_present=lock_directory_present,
    )


def census_runtime_root(
    context: Any,
    *,
    root: Path | str | None = None,
    lease: ReconciliationLease | None = None,
) -> dict[str, Any]:
    """Take an exclusive parent-directory lease and return a read-only census.

    A caller holding the exact exclusive lease may pass it to avoid nested
    acquisition.  The census itself never creates the root, a lock file, a
    bucket, or any other filesystem object.
    """

    context = _resume_context(context)
    if lease is not None:
        return _census_runtime_root_locked(context, root=root, lease=lease)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as held:
        return _census_runtime_root_locked(context, root=root, lease=held)


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


def _fast_require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeChainError(message)


def _fast_exact_int(value: Any, name: str) -> int:
    _fast_require(
        not isinstance(value, bool) and isinstance(value, int),
        f"{name} must be an integer, not a boolean or float",
    )
    return int(value)


def _fast_nonnegative_int(value: Any, name: str) -> int:
    number = _fast_exact_int(value, name)
    _fast_require(number >= 0, f"{name} must be non-negative")
    return number


def _fast_finite(value: Any, name: str) -> float:
    _fast_require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be a real number",
    )
    _fast_require(math.isfinite(float(value)), f"{name} must be finite")
    return float(value)


def _fast_validate_request(
    context: AuthenticatedResumeContext,
    request: Mapping[str, Any],
    *,
    execution_class: str | None = None,
) -> dict[str, Any]:
    """B1C-equivalent request validation over the authenticated B3 cache.

    This intentionally does not call ``load_bler_tooling_contract``,
    ``campaign_bindings`` or ``required_work_unit``.  The state context has
    already authenticated those bytes; B3 keeps immutable scalar bindings and
    canonical required-record bytes for this strict hot path.
    """

    try:
        _fast_require(isinstance(request, Mapping), "work-unit request is not a mapping")
        _fast_require(
            set(request) == set(bler_contract.REQUEST_FIELDS),
            "work-unit request has missing or unknown fields",
        )
        _fast_require(
            request["schema_version"] == REQUEST_SCHEMA_VERSION,
            "unsupported work-unit request schema_version",
        )
        _fast_require(request["artifact_role"] == bler_contract.REQUEST_ARTIFACT_ROLE, "wrong request artifact role")
        observed_class = request["execution_class"]
        _fast_require(
            observed_class in bler_contract.EXECUTION_CLASSES,
            f"unknown execution class {observed_class!r}",
        )
        if execution_class is not None:
            _fast_require(observed_class == execution_class, f"request is {observed_class!r}, not {execution_class!r}")

        authority = context._authority
        for name in (
            "campaign_id",
            "campaign_manifest_sha256",
            "required_bler_artifact_sha256",
            "selection_policy_sha256",
            "bler_tooling_contract_id",
            "bler_tooling_contract_sha256",
        ):
            _fast_require(request[name] == authority[name], f"request binding {name} does not match the campaign")

        work_unit_id = request["work_unit_id"]
        _fast_require(isinstance(work_unit_id, str) and work_unit_id.strip() != "", "work_unit_id must be a non-blank string")
        identity = request["bler_identity"]
        _fast_require(isinstance(identity, Mapping), "bler_identity is not a mapping")
        try:
            bler_contract.BlerIdentity.from_mapping(identity)
        except Exception as exc:
            raise ResumeChainError(f"bler_identity failed B1C validation: {exc}") from exc
        _fast_finite(request["snr_db"], "snr_db")
        packet_ids = request["source_packet_config_ids"]
        _fast_require(
            isinstance(packet_ids, list)
            and packet_ids
            and all(isinstance(item, str) and item for item in packet_ids),
            "source_packet_config_ids must be a non-empty list of non-blank strings",
        )
        _fast_require(
            packet_ids == sorted(set(packet_ids)),
            "source_packet_config_ids must be unique and canonically ordered",
        )
        _fast_require(
            request["seed_derivation_identity"] == bler_contract.SEED_DERIVATION_IDENTITY,
            "request seed-derivation identity is not the frozen identity",
        )
        _fast_require(
            request["seed_domain_separator"] == bler_contract.SEED_DOMAIN_SEPARATOR,
            "request seed domain separator is not the frozen separator",
        )
        seeds = request["stream_seeds"]
        _fast_require(isinstance(seeds, Mapping), "stream_seeds is not a mapping")
        _fast_require(set(seeds) == set(bler_contract.SEED_PURPOSES), "stream_seeds must cover exactly the allowed purposes")
        for purpose in bler_contract.SEED_PURPOSES:
            expected_seed = bler_contract.seed_record(authority["campaign_id"], work_unit_id, purpose)
            _fast_require(
                canonical_json(seeds[purpose]) == canonical_json(expected_seed),
                f"stream seed record for {purpose} does not reproduce from the frozen derivation",
            )
        _fast_require(
            len({seeds[purpose]["seed_uint64"] for purpose in bler_contract.SEED_PURPOSES})
            == len(bler_contract.SEED_PURPOSES),
            "random purposes must not share a stream seed",
        )

        trials = _fast_nonnegative_int(request["trials_requested"], "trials_requested")
        _fast_require(trials > 0, "trials_requested must be positive")
        _fast_require(request["merge_eligible"] is False, "a request is never merge eligible")
        _fast_require(request["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, "request claims test-split access")
        full_count = bler_contract.full_strength_trial_count()
        if observed_class == bler_contract.EXECUTION_CLASS_FULL_STRENGTH:
            required = context.work_unit_record(work_unit_id)
            _fast_require(
                canonical_json(identity) == canonical_json(required["identity"]),
                "full-strength identity does not match the required entry exactly",
            )
            _fast_require(
                canonical_json(request["snr_db"]) == canonical_json(required["snr_db"]),
                "full-strength SNR does not match the required entry exactly",
            )
            _fast_require(
                canonical_json(packet_ids) == canonical_json(list(required["source_packet_config_ids"])),
                "full-strength source packet IDs do not match the required entry exactly",
            )
            _fast_require(
                request["trial_count_source"] == bler_contract.FULL_STRENGTH_TRIAL_COUNT_SOURCE,
                "full-strength trial count source changed",
            )
            _fast_require(trials == full_count, "full-strength request must use exactly the configured trial count")
            _fast_require(request["scientific_evidence"] is True, "full-strength request must be scientific evidence")
            _fast_require(request["label"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH, "full-strength label changed")
        else:
            _fast_require(
                request["trial_count_source"] == bler_contract.BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
                "bounded smoke trial count source changed",
            )
            _fast_require(
                trials <= bler_contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT,
                "bounded smoke exceeds its per-unit limit",
            )
            _fast_require(trials < full_count, "bounded smoke must stay below the full-strength trial count")
            _fast_require(request["scientific_evidence"] is False, "bounded smoke is scientific evidence")
            _fast_require(request["label"] == bler_contract.BOUNDED_SMOKE_LABEL, "bounded smoke label changed")
        return json.loads(canonical_json(dict(request)))
    except ResumeChainError:
        raise
    except Exception as exc:
        raise ResumeChainError(f"work-unit request failed B3 fast validation: {exc}") from exc


def _fast_validate_result(
    context: AuthenticatedResumeContext,
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """B1C-equivalent result validation without rereading B1C artifacts."""

    try:
        _fast_require(isinstance(result, Mapping), "work-unit result is not a mapping")
        _fast_require(set(result) == set(bler_contract.RESULT_FIELDS), "work-unit result has missing or unknown sections")
        _fast_require(result["schema_version"] == RESULT_SCHEMA_VERSION, "unsupported work-unit result schema_version")
        _fast_require(result["artifact_role"] == bler_contract.RESULT_ARTIFACT_ROLE, "wrong result artifact role")
        status = result["status"]
        _fast_require(status in bler_contract.RESULT_STATUSES, f"unknown result status {status!r}")

        identity = result["identity"]
        measurement = result["measurement"]
        metadata = result["execution_metadata"]
        disposition = result["disposition"]
        _fast_require(isinstance(identity, Mapping) and set(identity) == set(bler_contract.RESULT_IDENTITY_FIELDS), "result identity section has missing or unknown fields")
        _fast_require(isinstance(measurement, Mapping) and set(measurement) == set(bler_contract.RESULT_MEASUREMENT_FIELDS), "result measurement section has missing or unknown fields")
        _fast_require(isinstance(metadata, Mapping) and set(metadata) == set(bler_contract.RESULT_EXECUTION_METADATA_FIELDS), "result execution metadata has missing or unknown fields")
        try:
            bler_contract.validate_execution_metadata(metadata)
        except Exception as exc:
            raise ResumeChainError(f"result execution metadata failed B1C validation: {exc}") from exc
        _fast_require(isinstance(disposition, Mapping) and set(disposition) == set(bler_contract.RESULT_DISPOSITION_FIELDS), "result disposition section has missing or unknown fields")
        implementation = identity["implementation"]
        _fast_require(isinstance(implementation, Mapping) and set(implementation) == set(bler_contract.IMPLEMENTATION_FIELDS), "result implementation binding has missing or unknown fields")
        _fast_require(
            canonical_json(implementation) == canonical_json(bler_contract.implementation_binding()),
            "result implementation/dependency binding does not match the frozen contract",
        )

        rebuilt = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "artifact_role": bler_contract.REQUEST_ARTIFACT_ROLE,
            "execution_class": identity["execution_class"],
            "campaign_id": identity["campaign_id"],
            "bler_tooling_contract_id": identity["bler_tooling_contract_id"],
            "bler_tooling_contract_sha256": identity["bler_tooling_contract_sha256"],
            "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
            "required_bler_artifact_sha256": identity["required_bler_artifact_sha256"],
            "selection_policy_sha256": identity["selection_policy_sha256"],
            "work_unit_id": identity["work_unit_id"],
            "bler_identity": dict(identity["bler_identity"]),
            "snr_db": identity["snr_db"],
            "source_packet_config_ids": list(identity["source_packet_config_ids"]),
            "trials_requested": identity["trials_requested"],
            "trial_count_source": identity["trial_count_source"],
            "seed_derivation_identity": identity["seed_derivation_identity"],
            "seed_domain_separator": identity["seed_domain_separator"],
            "stream_seeds": dict(identity["stream_seeds"]),
            "scientific_evidence": identity["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH,
            "merge_eligible": False,
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
            "label": (
                bler_contract.EXECUTION_CLASS_FULL_STRENGTH
                if identity["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH
                else bler_contract.BOUNDED_SMOKE_LABEL
            ),
        }
        rebuilt = _fast_validate_request(context, rebuilt)
        request_digest = bler_contract.request_digest(rebuilt)
        _fast_require(identity["request_sha256"] == request_digest, "result request digest does not reproduce from its identity section")
        if request is not None:
            checked_request = _fast_validate_request(context, request)
            _fast_require(identity["request_sha256"] == bler_contract.request_digest(checked_request), "result does not bind the request it claims")
            for field in ("work_unit_id", "trials_requested", "execution_class", "trial_count_source"):
                _fast_require(canonical_json(identity[field]) == canonical_json(checked_request[field]), f"result {field} does not match its request exactly")
            _fast_require(canonical_json(identity["bler_identity"]) == canonical_json(checked_request["bler_identity"]), "result identity does not match its request exactly")
            _fast_require(canonical_json(identity["snr_db"]) == canonical_json(checked_request["snr_db"]), "result SNR does not match its request exactly")

        full_strength = identity["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH
        full_count = bler_contract.full_strength_trial_count()
        trials_requested = _fast_nonnegative_int(identity["trials_requested"], "trials_requested")
        trials_completed = _fast_nonnegative_int(measurement["trials_completed"], "trials_completed")
        information_bits = _fast_nonnegative_int(measurement["information_bits"], "information_bits")
        bit_errors = _fast_nonnegative_int(measurement["bit_errors"], "bit_errors")
        block_errors = _fast_nonnegative_int(measurement["block_errors"], "block_errors")
        _fast_require(trials_completed <= trials_requested, "trials_completed exceeds trials_requested")
        _fast_require(block_errors <= trials_completed, "block_errors exceeds trials_completed")
        try:
            information_length = _fast_exact_int(identity["bler_identity"]["k_and_n"][0], "K")
            derived = bler_contract.recompute_measurements(
                trials_completed=trials_completed,
                information_bits=information_bits,
                bit_errors=bit_errors,
                block_errors=block_errors,
                information_length=information_length,
            )
        except Exception as exc:
            raise ResumeChainError(f"result counts failed B1C validation: {exc}") from exc
        _fast_require(information_bits == trials_completed * information_length, "information_bits is not trials_completed x K")
        _fast_require(bit_errors <= information_bits, "bit_errors exceeds information_bits")
        for name, expected_value in derived.items():
            stored = measurement[name]
            if expected_value is None:
                _fast_require(stored is None, f"{name} must be null at zero completed trials")
            else:
                _fast_require(stored is not None, f"{name} is missing")
                _fast_finite(stored, name)
                _fast_require(stored == expected_value, f"stored {name} does not reproduce from the counts")
        _fast_require(measurement["confidence_interval_method"] == bler_contract.CONFIDENCE_INTERVAL_METHOD, "confidence interval method changed")
        _fast_require(measurement["confidence_interval_percent"] == bler_contract.CONFIDENCE_INTERVAL_PERCENT, "confidence interval percent changed")
        _fast_require(measurement["confidence_interval_role"] == bler_contract.CONFIDENCE_INTERVAL_ROLE, "confidence interval role changed")

        if status == bler_contract.STATUS_COMPLETE:
            _fast_require(trials_completed > 0, "completed evidence requires trials_completed > 0")
            if full_strength:
                _fast_require(trials_completed == trials_requested == full_count, "completed full-strength result needs exactly the configured trial count")
        scientific = disposition["scientific_evidence"]
        merge_eligible = disposition["merge_eligible"]
        _fast_require(type(scientific) is bool and type(merge_eligible) is bool, "disposition flags must be booleans")
        _fast_require(scientific is full_strength, "only full-strength execution is scientific evidence")
        _fast_require(disposition["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, "result claims test-split access")
        contribution = _fast_nonnegative_int(disposition["required_coverage_contribution"], "required_coverage_contribution")
        expected_merge = full_strength and status == bler_contract.STATUS_COMPLETE and trials_completed == full_count
        _fast_require(merge_eligible is expected_merge, "merge eligibility must follow exactly from status, class and counts")
        _fast_require(contribution == (1 if expected_merge else 0), "required coverage contribution must follow merge eligibility")
        if not full_strength:
            _fast_require(merge_eligible is False and contribution == 0, "bounded smoke is never merge eligible")
        if status in (bler_contract.STATUS_INCOMPLETE, bler_contract.STATUS_FAILED):
            _fast_require(merge_eligible is False, "an incomplete or failed result is never merge eligible")
        return json.loads(canonical_json(dict(result)))
    except ResumeChainError:
        raise
    except Exception as exc:
        raise ResumeChainError(f"work-unit result failed B3 fast validation: {exc}") from exc


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
    request = _fast_validate_request(
        context,
        payload,
        execution_class=(
            bler_contract.EXECUTION_CLASS_FULL_STRENGTH if require_full_strength else None
        ),
    )

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
    result = _fast_validate_result(
        context,
        payload,
        request=request_record["request"],
    )

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


def validate_attempt_history(
    context: Any,
    work_unit_id: str,
    state_attempt: Any,
    request_attempts: Sequence[int],
    result_attempts: Sequence[int],
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Validate every persisted request/result at or below a state attempt.

    The returned record is deterministic and is the single history object that
    later B3 plan, merge, and reconciliation code reuses.  Older failed
    results are retained as zero-contribution history; an older complete
    merge-eligible result, an incomplete result, a result without its exact
    request, or a request/result beyond the state attempt is a HOLD.
    """

    context = _resume_context(context)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    attempt = _positive_int(state_attempt, "state attempt", ResumeChainError)
    request_numbers = tuple(sorted(set(_positive_int(value, "request attempt", ResumeChainError) for value in request_attempts)))
    result_numbers = tuple(sorted(set(_positive_int(value, "result attempt", ResumeChainError) for value in result_attempts)))
    if len(request_numbers) != len(tuple(request_attempts)):
        raise ResumeContradictionError(f"{work_unit_id} has duplicate request attempts")
    if len(result_numbers) != len(tuple(result_attempts)):
        raise ResumeContradictionError(f"{work_unit_id} has duplicate result attempts")
    for label, values in (("request", request_numbers), ("result", result_numbers)):
        ahead = [value for value in values if value > attempt]
        if ahead:
            raise ResumeContradictionError(
                f"{work_unit_id} has a {label} for attempt {min(ahead)} beyond state attempt {attempt}"
            )

    records: list[dict[str, Any]] = []
    full_request_digest: str | None = None
    for number in sorted(set(request_numbers) | set(result_numbers)):
        request_record: dict[str, Any] | None = None
        result_record: dict[str, Any] | None = None
        if number in request_numbers:
            request_record = validate_request_file(
                context,
                work_unit_id,
                number,
                root=root,
                require_full_strength=scan_mode == SCAN_MODE_PRODUCTION_MERGE,
            )
            if request_record["execution_class"] == bler_contract.EXECUTION_CLASS_FULL_STRENGTH:
                if full_request_digest is None:
                    full_request_digest = request_record["request_sha256"]
                elif full_request_digest != request_record["request_sha256"]:
                    raise ResumeContradictionError(
                        f"{work_unit_id} full-strength retry requests are not byte-identical"
                    )
        if number in result_numbers:
            if request_record is None:
                raise ResumeContradictionError(
                    f"{work_unit_id} has a result for attempt {number} without its exact request"
                )
            result_record = validate_result_file(
                context,
                work_unit_id,
                number,
                root=root,
                request_record=request_record,
                scan_mode=scan_mode,
            )
            if result_record["status"] == bler_contract.STATUS_INCOMPLETE:
                raise ResumeContradictionError(
                    f"{work_unit_id} has a persisted incomplete result at attempt {number}"
                )
            if number < attempt and result_record["status"] == bler_contract.STATUS_COMPLETE:
                if is_full_strength_merge_candidate(context, result_record, request_record):
                    raise ResumeContradictionError(
                        f"{work_unit_id} has a complete merge-eligible attempt-{number} result "
                        f"while state advanced to attempt {attempt}"
                    )
        records.append(
            {
                "attempt": number,
                "request_sha256": None if request_record is None else request_record["request_sha256"],
                "result_sha256": None if result_record is None else result_record["result_sha256"],
                "request": request_record,
                "result": result_record,
                "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
            }
        )
    return _fresh(
        {
            "work_unit_id": work_unit_id,
            "state_attempt": attempt,
            "request_attempts": list(request_numbers),
            "result_attempts": list(result_numbers),
            "attempts": records,
            "full_strength_request_sha256": full_request_digest,
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        }
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

    history = validate_attempt_history(
        context,
        work_unit_id,
        attempt,
        request_attempts,
        result_attempts,
        root=root,
        scan_mode=scan_mode,
    )
    current_history = next(
        (entry for entry in history["attempts"] if entry["attempt"] == attempt),
        None,
    )
    request_record = None if current_history is None else current_history["request"]
    result_record = None if current_history is None else current_history["result"]
    if result_record is not None:
        # Revalidate the current result against the state shard assignment.  The
        # fast history pass intentionally has no state dependency.
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
    if identity["status"] == work_units.STATUS_RESULT_LINKED:
        if request_record is None or result_record is None:
            raise ResumeContradictionError(
                f"{work_unit_id} result_linked state is missing its current-attempt chain"
            )
        if identity["request_sha256"] != request_record["request_sha256"]:
            raise ResumeContradictionError(
                f"{work_unit_id} result_linked request SHA-256 is not the exact request digest"
            )
        if identity["result_sha256"] != result_record["result_sha256"]:
            raise ResumeContradictionError(
                f"{work_unit_id} result_linked result SHA-256 is not the exact result digest"
            )
        if identity["scientific_execution_performed"] is not True:
            raise ResumeContradictionError(
                f"{work_unit_id} result_linked state does not record scientific execution"
            )
        if identity["trials_completed"] != result_record["trials_completed"]:
            raise ResumeContradictionError(
                f"{work_unit_id} result_linked trials_completed differs from its result"
            )
    if (
        identity["status"] == work_units.STATUS_FAILED
        and identity["request_sha256"] is None
        and (request_record is not None or result_record is not None)
    ):
        raise ResumeContradictionError(
            f"{work_unit_id} failed state without a request binding has current-attempt artifacts"
        )
    if (
        identity["status"] == work_units.STATUS_FAILED
        and result_record is not None
        and identity["trials_completed"] != result_record["trials_completed"]
    ):
        raise ResumeContradictionError(
            f"{work_unit_id} failed state trials_completed differs from its result"
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
            result_record["trials_completed"]
            if result_record is not None
            else (0 if state is None else state["identity"]["trials_completed"])
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


# ---------------------------------------------------------------------------
# Deterministic resume plans and merge validation (G8_B3 §17–§18)
# ---------------------------------------------------------------------------

RESUME_PLAN_SCHEMA_VERSION = RESUME_CONTRACT_SCHEMA_VERSION
RESUME_PLAN_ARTIFACT_ROLE = "g8_bler_resume_plan"
MERGE_REPORT_SCHEMA_VERSION = RESUME_CONTRACT_SCHEMA_VERSION
MERGE_REPORT_ARTIFACT_ROLE = "g8_bler_merge_validation_report"
PLAN_DIGEST_FIELD = "plan_digest"
MERGE_REPORT_DIGEST_FIELD = "report_digest"
CAMPAIGN_RECONCILIATION_SCHEMA_VERSION = RESUME_CONTRACT_SCHEMA_VERSION
CAMPAIGN_RECONCILIATION_ARTIFACT_ROLE = "g8_campaign_reconciliation_proposal"
CAMPAIGN_STATE_LOGICAL_PATH = "results/baseline/g8/campaign_state.json"


def _scan_runtime_root_locked(
    context: AuthenticatedResumeContext,
    *,
    root: Path | str | None,
    scan_mode: str,
    lease: ReconciliationLease,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Census and classify once under one already-held exclusive lease."""

    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    census = _census_runtime_root_locked(context, root=root, lease=lease)
    records = classify_runtime_root(context, census, root=root, scan_mode=scan_mode)
    return census, records


def _resume_operation_bindings(context: AuthenticatedResumeContext) -> dict[str, Any]:
    """Return the complete immutable binding block for a B3 derived record."""

    authority = context.authority_binding()
    state_binding = context.state_contract_binding()
    resume_binding = context.require_resume_contract_binding()
    return {
        "bler_resume_contract_id": resume_binding["bler_resume_contract_id"],
        "bler_resume_contract_sha256": resume_binding["bler_resume_contract_sha256"],
        "bler_state_contract_id": state_binding["bler_state_contract_id"],
        "bler_state_contract_sha256": state_binding["bler_state_contract_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "request_schema_version": authority["request_schema_version"],
        "result_schema_version": authority["result_schema_version"],
        "unit_state_schema_version": UNIT_STATE_SCHEMA_VERSION,
    }


def _with_derived_digest(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Add one self-excluding SHA-256 over canonical identity bytes."""

    body = dict(payload)
    body.pop(field, None)
    body[field] = sha256_bytes(canonical_json(body))
    return _fresh(body)


def resume_plan_digest(plan: Mapping[str, Any]) -> str:
    """Recompute a resume-plan digest without trusting its stored value."""

    if not isinstance(plan, Mapping):
        raise ResumeChainError("resume plan must be a mapping")
    body = dict(plan)
    supplied = body.pop(PLAN_DIGEST_FIELD, None)
    _digest(supplied, "resume plan digest", ResumeChainError)
    return sha256_bytes(canonical_json(body))


def merge_report_digest(report: Mapping[str, Any]) -> str:
    """Recompute a merge-report digest without trusting its stored value."""

    if not isinstance(report, Mapping):
        raise ResumeChainError("merge report must be a mapping")
    body = dict(report)
    supplied = body.pop(MERGE_REPORT_DIGEST_FIELD, None)
    _digest(supplied, "merge report digest", ResumeChainError)
    return sha256_bytes(canonical_json(body))


def _build_resume_plan_locked(
    context: AuthenticatedResumeContext,
    *,
    root: Path | str | None,
    shard_count: Any,
    shard_index: Any,
    scan_mode: str,
    lease: ReconciliationLease,
) -> dict[str, Any]:
    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    bindings = _resume_operation_bindings(context)
    shard_plan = work_units.build_shard_plan(
        context.state_context,
        shard_count,
        shard_index,
    )
    shard_plan = work_units.validate_shard_plan(context.state_context, shard_plan)
    census, records = _scan_runtime_root_locked(
        context,
        root=root,
        scan_mode=scan_mode,
        lease=lease,
    )
    by_id = {record["work_unit_id"]: record for record in records}
    assigned_ids = list(shard_plan["assigned_work_unit_ids"])
    assigned_records = [by_id[work_unit_id] for work_unit_id in assigned_ids]
    completed = [
        record["work_unit_id"]
        for record in assigned_records
        if record["classification"] == CLASSIFICATION_COMPLETED_FULL_STRENGTH
    ]
    recoverable = [
        record["work_unit_id"]
        for record in assigned_records
        if record["classification"] in RECOVERABLE_CLASSIFICATIONS
    ]
    remaining = [
        record["work_unit_id"]
        for record in assigned_records
        if record["classification"] in REMAINING_CLASSIFICATIONS
    ]
    terminal = [
        record["work_unit_id"]
        for record in assigned_records
        if record["classification"] in TERMINAL_CLASSIFICATIONS
    ]
    proposed_attempts = [
        {
            "work_unit_id": record["work_unit_id"],
            "classification": record["classification"],
            "current_attempt": record["attempt"],
            "proposed_attempt": record["proposed_attempt"],
        }
        for record in assigned_records
        if record["classification"] in REMAINING_CLASSIFICATIONS
    ]
    plan = {
        "schema_version": RESUME_PLAN_SCHEMA_VERSION,
        "artifact_role": RESUME_PLAN_ARTIFACT_ROLE,
        **bindings,
        "required_work_unit_count": context.required_work_unit_count,
        "shard_count": shard_plan["shard_count"],
        "shard_index": shard_plan["shard_index"],
        "shard_plan_digest": shard_plan["plan_digest"],
        "assigned_work_unit_ids": assigned_ids,
        "assigned_unit_records": assigned_records,
        "completed_work_unit_ids": completed,
        "recoverable_work_unit_ids": recoverable,
        "remaining_work_unit_ids": remaining,
        "terminal_nonmergeable_work_unit_ids": terminal,
        "proposed_attempts": proposed_attempts,
        "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
        "scan_mode": scan_mode,
        "ignored_staging_count": census["ignored_orphan_staging_count"],
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }
    return _with_derived_digest(plan, PLAN_DIGEST_FIELD)


def build_resume_plan(
    context: Any,
    *,
    root: Path | str | None = None,
    shard_count: Any = 1,
    shard_index: Any = 0,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Build one byte-deterministic, exclusive-lease resume plan."""

    context = _resume_context(context)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as lease:
        return _build_resume_plan_locked(
            context,
            root=root,
            shard_count=shard_count,
            shard_index=shard_index,
            scan_mode=scan_mode,
            lease=lease,
        )


def _build_merge_report_locked(
    context: AuthenticatedResumeContext,
    *,
    root: Path | str | None,
    scan_mode: str,
    lease: ReconciliationLease,
) -> dict[str, Any]:
    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    bindings = _resume_operation_bindings(context)
    census, records = _scan_runtime_root_locked(
        context,
        root=root,
        scan_mode=scan_mode,
        lease=lease,
    )
    required_ids = list(context.ordered_work_unit_ids)
    completed = [
        record["work_unit_id"]
        for record in records
        if record["classification"] == CLASSIFICATION_COMPLETED_FULL_STRENGTH
    ]
    recoverable = [
        record["work_unit_id"]
        for record in records
        if record["classification"] in RECOVERABLE_CLASSIFICATIONS
    ]
    remaining = [
        record["work_unit_id"]
        for record in records
        if record["classification"] in REMAINING_CLASSIFICATIONS
    ]
    failed = [
        record["work_unit_id"]
        for record in records
        if record["classification"] == CLASSIFICATION_FAILED_RETRYABLE
    ]
    bounded = [
        record["work_unit_id"]
        for record in records
        if record["classification"] == CLASSIFICATION_TERMINAL_NONMERGEABLE
    ]
    missing = [work_unit_id for work_unit_id in required_ids if work_unit_id not in set(completed)]
    valid_requests = sum(1 for record in records if record["request_sha256"] is not None)
    valid_results = sum(1 for record in records if record["result_sha256"] is not None)
    valid_complete_results = sum(
        1
        for record in records
        if record["result_sha256"] is not None
        and record["result_status"] == bler_contract.STATUS_COMPLETE
    )
    duplicate_count = 0
    unknown_count = 0
    exact_coverage_count = len(completed)
    total_coverage = sum(record["required_coverage_contribution"] for record in records)
    coverage_complete = (
        required_ids == list(dict.fromkeys(required_ids))
        and len(completed) == context.required_work_unit_count
        and len(set(completed)) == context.required_work_unit_count
        and valid_requests == context.required_work_unit_count
        and valid_complete_results == context.required_work_unit_count
        and total_coverage == context.required_work_unit_count
        and not duplicate_count
        and not unknown_count
        and not missing
        and bler_contract.TEST_SPLIT_ACCESS == 0
    )
    report = {
        "schema_version": MERGE_REPORT_SCHEMA_VERSION,
        "artifact_role": MERGE_REPORT_ARTIFACT_ROLE,
        **bindings,
        "required_work_unit_count": context.required_work_unit_count,
        "required_work_unit_ids": required_ids,
        "validated_complete_work_unit_ids": completed,
        "missing_work_unit_ids": missing,
        "remaining_work_unit_ids": remaining,
        "recoverable_work_unit_ids": recoverable,
        "failed_work_unit_ids": failed,
        "bounded_nonmergeable_work_unit_ids": bounded,
        "duplicate_count": duplicate_count,
        "unknown_count": unknown_count,
        "exact_coverage_count": exact_coverage_count,
        "valid_request_count": valid_requests,
        "valid_result_count": valid_results,
        "valid_complete_result_count": valid_complete_results,
        "total_required_coverage_contribution": total_coverage,
        "coverage_complete": coverage_complete,
        "merge_ready": coverage_complete,
        "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
        "scan_mode": scan_mode,
        "ignored_staging_count": census["ignored_orphan_staging_count"],
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }
    return _with_derived_digest(report, MERGE_REPORT_DIGEST_FIELD)


def build_merge_report(
    context: Any,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Validate resume coverage under one exclusive lease without merging."""

    context = _resume_context(context)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as lease:
        return _build_merge_report_locked(
            context,
            root=root,
            scan_mode=scan_mode,
            lease=lease,
        )


# ---------------------------------------------------------------------------
# Campaign-state reconciliation (G8_B3 §19)
# ---------------------------------------------------------------------------


def _load_campaign_state_exact(
    context: AuthenticatedResumeContext,
) -> tuple[dict[str, Any], bytes, str]:
    path = context.campaign_state_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResumeCampaignError(f"cannot read campaign state {path}: {exc}") from exc
    try:
        state = g8_campaign.load_campaign_state(path)
    except g8_campaign.G8ContractError as exc:
        raise ResumeCampaignError(f"campaign state failed authentication: {exc}") from exc
    return state, raw, sha256_bytes(raw)


def _build_campaign_reconciliation_locked(
    context: AuthenticatedResumeContext,
    *,
    root: Path | str | None,
    scan_mode: str,
    lease: ReconciliationLease,
) -> dict[str, Any]:
    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")
    bindings = _resume_operation_bindings(context)
    state, state_bytes, state_sha256 = _load_campaign_state_exact(context)
    identity = state["identity"]
    if identity["phase"] != PHASE or identity["stage"] not in {
        "tooling_open",
        "tooling_smoke_complete",
    }:
        raise ResumeCampaignError(
            "campaign reconciliation is only valid during G8_B tooling stages"
        )
    census, records = _scan_runtime_root_locked(
        context,
        root=root,
        scan_mode=scan_mode,
        lease=lease,
    )
    required_ids = list(context.ordered_work_unit_ids)
    known_ids = set(required_ids)
    campaign_completed = list(identity["completed_work_unit_ids"])
    unknown_campaign_ids = [work_unit_id for work_unit_id in campaign_completed if work_unit_id not in known_ids]
    if unknown_campaign_ids:
        raise ResumeCampaignError(
            "campaign state lists unknown completed work-unit IDs: "
            + ", ".join(unknown_campaign_ids)
        )
    by_id = {record["work_unit_id"]: record for record in records}
    validated_completed = [
        record["work_unit_id"]
        for record in records
        if record["classification"] == CLASSIFICATION_COMPLETED_FULL_STRENGTH
    ]
    evidence_set = set(validated_completed)
    lead_ids = [work_unit_id for work_unit_id in campaign_completed if work_unit_id not in evidence_set]
    if lead_ids:
        raise ResumeCampaignError(
            "campaign state leads validated per-unit evidence: "
            + ", ".join(lead_ids)
        )
    # This explicit loop keeps the lead check tied to the exact terminal record,
    # rather than trusting only the classification name in the projection.
    for work_unit_id in campaign_completed:
        record = by_id[work_unit_id]
        if (
            record["classification"] != CLASSIFICATION_COMPLETED_FULL_STRENGTH
            or record["request_sha256"] is None
            or record["result_sha256"] is None
            or record["required_coverage_contribution"] != 1
            or record["test_split_access"] != bler_contract.TEST_SPLIT_ACCESS
        ):
            raise ResumeCampaignError(
                f"campaign completed ID {work_unit_id} lacks exact terminal full-strength evidence"
            )
    lagging_ids = [work_unit_id for work_unit_id in validated_completed if work_unit_id not in set(campaign_completed)]
    current_in_progress = identity["in_progress_work_unit_id"]
    proposed_in_progress = (
        None if current_in_progress in evidence_set else current_in_progress
    )
    counters = dict(identity["counters"])
    changed = campaign_completed != validated_completed or current_in_progress != proposed_in_progress
    proposal = {
        "schema_version": CAMPAIGN_RECONCILIATION_SCHEMA_VERSION,
        "artifact_role": CAMPAIGN_RECONCILIATION_ARTIFACT_ROLE,
        **bindings,
        "campaign_state_logical_path": CAMPAIGN_STATE_LOGICAL_PATH,
        "campaign_state_sha256": state_sha256,
        "campaign_state_bytes": len(state_bytes),
        "phase": identity["phase"],
        "stage": identity["stage"],
        "required_work_unit_count": context.required_work_unit_count,
        "campaign_completed_work_unit_ids": campaign_completed,
        "validated_completed_work_unit_ids": validated_completed,
        "lagging_work_unit_ids": lagging_ids,
        "proposed_completed_work_unit_ids": validated_completed,
        "in_progress_work_unit_id": current_in_progress,
        "proposed_in_progress_work_unit_id": proposed_in_progress,
        "counters": counters,
        "changed": changed,
        "scan_mode": scan_mode,
        "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
        "ignored_staging_count": census["ignored_orphan_staging_count"],
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }
    return _with_derived_digest(proposal, "proposal_digest")


def propose_campaign_reconciliation(
    context: Any,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Derive a no-write campaign-state projection under an exclusive lease."""

    context = _resume_context(context)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as lease:
        return _build_campaign_reconciliation_locked(
            context,
            root=root,
            scan_mode=scan_mode,
            lease=lease,
        )


def _validate_reconciliation_proposal(
    proposal: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(proposal, Mapping):
        raise ResumeCampaignError("campaign reconciliation proposal is not a mapping")
    supplied = dict(proposal)
    digest = supplied.get("proposal_digest")
    _digest(digest, "campaign reconciliation proposal digest", ResumeCampaignError)
    body = dict(supplied)
    body.pop("proposal_digest", None)
    if digest != sha256_bytes(canonical_json(body)):
        raise ResumeCampaignError("campaign reconciliation proposal digest does not reproduce")
    if canonical_json(supplied) != canonical_json(dict(expected)):
        raise ResumeCampaignError(
            "campaign reconciliation proposal is stale or does not describe the exact current evidence"
        )


def apply_campaign_reconciliation(
    context: Any,
    proposal: Mapping[str, Any] | None = None,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Apply only validated completed IDs through the atomic campaign writer."""

    context = _resume_context(context)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as lease:
        expected = _build_campaign_reconciliation_locked(
            context,
            root=root,
            scan_mode=scan_mode,
            lease=lease,
        )
        if proposal is not None:
            _validate_reconciliation_proposal(proposal, expected)
        state, _state_bytes, state_sha256 = _load_campaign_state_exact(context)
        if not expected["changed"]:
            return _fresh(
                {
                    "proposal": expected,
                    "applied": False,
                    "installed_campaign_state_sha256": state_sha256,
                    "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
                }
            )

        candidate = json.loads(canonical_json(state))
        candidate_identity = candidate["identity"]
        candidate_identity["completed_work_unit_ids"] = list(
            expected["proposed_completed_work_unit_ids"]
        )
        candidate_identity["in_progress_work_unit_id"] = expected[
            "proposed_in_progress_work_unit_id"
        ]
        try:
            g8_campaign.validate_state_transition(state, candidate)
        except g8_campaign.G8ContractError as exc:
            raise ResumeCampaignError(f"campaign reconciliation transition is illegal: {exc}") from exc
        proposed_body = rendered_json(candidate)
        proposed_sha256 = sha256_bytes(proposed_body)
        try:
            g8_campaign.write_campaign_state_atomically(
                context.campaign_state_path,
                candidate,
            )
        except Exception as exc:
            try:
                installed_body = context.campaign_state_path.read_bytes()
                installed = g8_campaign.load_campaign_state(context.campaign_state_path)
            except Exception as read_exc:
                raise ResumeCampaignError(
                    "campaign-state publication outcome is uncertain and the installed bytes "
                    f"could not be authenticated: {exc}"
                ) from read_exc
            if installed_body != proposed_body or sha256_bytes(installed_body) != proposed_sha256:
                raise ResumeCampaignError(
                    "campaign-state publication failed and did not install the exact proposed bytes"
                ) from exc
            _ = installed
        try:
            installed_body = context.campaign_state_path.read_bytes()
            installed = g8_campaign.load_campaign_state(context.campaign_state_path)
        except Exception as exc:
            raise ResumeCampaignError(
                f"cannot reread the installed campaign state after reconciliation: {exc}"
            ) from exc
        if installed_body != proposed_body or sha256_bytes(installed_body) != proposed_sha256:
            raise ResumeCampaignError(
                "installed campaign state does not equal the exact proposed canonical bytes"
            )
        if installed["identity"]["completed_work_unit_ids"] != expected[
            "proposed_completed_work_unit_ids"
        ]:
            raise ResumeCampaignError("installed campaign state does not contain the exact completed-ID projection")
        return _fresh(
            {
                "proposal": expected,
                "applied": True,
                "installed_campaign_state_sha256": proposed_sha256,
                "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
            }
        )


# ---------------------------------------------------------------------------
# Read-only inspection and explicit repair (G8_B3 §16)
# ---------------------------------------------------------------------------

#: Read-only inspection never repairs.  Repair is opt-in and bounded to the
#: exact two rows of the recovery matrix.
REPAIR_MODE_READ_ONLY = "read_only"
REPAIR_MODE_REPAIR_RECOVERABLE = "repair_recoverable"
REPAIR_MODES = (REPAIR_MODE_READ_ONLY, REPAIR_MODE_REPAIR_RECOVERABLE)


def _require_bounded_smoke_root(root: Path | str | None) -> Path:
    """Reject every lexical, resolved, or same-inode production-root alias."""

    if root is None:
        raise ResumeCensusError(
            "bounded-smoke inspection requires an explicit isolated root; None is the production root"
        )
    candidate = _root_path(root)
    production = _root_path(DEFAULT_WORK_UNIT_ROOT)
    candidate_norm = os.path.normpath(os.fspath(candidate))
    production_norm = os.path.normpath(os.fspath(production))
    if candidate_norm == production_norm:
        raise ResumeCensusError("bounded-smoke inspection may not use the production root")
    candidate_physical = _canonical_physical_root(candidate)
    production_physical = _canonical_physical_root(production)
    if candidate_physical == production_physical:
        raise ResumeCensusError("bounded-smoke inspection may not use an alias of the production root")
    candidate_entry = _lstat(candidate)
    production_entry = _lstat(production)
    if candidate_entry is not None and production_entry is not None:
        if (candidate_entry.st_dev, candidate_entry.st_ino) == (
            production_entry.st_dev,
            production_entry.st_ino,
        ):
            raise ResumeCensusError("bounded-smoke root is the production root inode")
    return candidate


def _repaired_state(
    context: AuthenticatedResumeContext,
    state: Mapping[str, Any],
    classification: str,
    request_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact successor state for one recovery-matrix row.

    Both rows transition *directly* from a clean claim.  Neither ever produces
    an intermediate request-bound ``claimed`` state, which B2C forbids.
    """

    identity = state["identity"]
    work_unit_id = identity["work_unit_id"]
    plan = work_units.build_shard_plan(
        context.state_context, identity["shard_count"], identity["shard_index"]
    )
    common = {
        "attempt": identity["attempt"],
        "request_sha256": request_record["request_sha256"],
        "scientific_execution_performed": True,
        "trials_completed": result_record["trials_completed"],
    }
    if classification == CLASSIFICATION_RECOVERABLE_FAILED_RESULT:
        # The failed result file stays as immutable history; a failed state
        # deliberately carries no result reference.
        return work_units.build_unit_state(
            context.state_context,
            work_unit_id,
            plan,
            status=work_units.STATUS_FAILED,
            **common,
        )
    if classification == CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT:
        return work_units.build_unit_state(
            context.state_context,
            work_unit_id,
            plan,
            status=work_units.STATUS_RESULT_LINKED,
            result_path=logical_result_path(context, work_unit_id, identity["attempt"]),
            result_sha256=result_record["result_sha256"],
            **common,
        )
    raise ResumeRepairError(f"{classification!r} is not a repairable classification")


def _repair_work_unit_locked(
    context: Any,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
    lease: ReconciliationLease,
) -> dict[str, Any]:
    """Apply the exact recovery-matrix transition for one work unit.

    The caller must already hold the exclusive global reconciliation lease.
    Everything is reread *after* the per-unit lock is taken, so a decision made
    from a pre-lock scan can never be applied to bytes that have since moved.
    """

    context = _resume_context(context)
    if scan_mode == SCAN_MODE_BOUNDED_SMOKE_INSPECTION:
        _require_bounded_smoke_root(root)
    lease._assert_usable(root, LOCK_MODE_EXCLUSIVE)
    root_path = _root_path(root)
    target = state_path(context, work_unit_id, root=root_path)

    # Complete reread immediately before proposing.  The per-unit B2C critical
    # section is entered by ``replace_unit_state`` itself, which rereads under
    # the lock and compares against ``expected`` inside that same section — the
    # compare-and-swap *is* the linearization point.  Taking the same per-unit
    # lock here as well would deadlock against it and would buy nothing: any
    # interleaving that moves the state between this reread and the swap is
    # exactly what the stale-writer check rejects.
    census = _census_runtime_root_locked(context, root=root_path, lease=lease)
    record = classify_work_unit(
        context, work_unit_id, census, root=root_path, scan_mode=scan_mode
    )
    classification = record["classification"]
    if classification not in REPAIRABLE_CLASSIFICATIONS:
        # Idempotence: a unit already repaired reports its settled class and
        # performs no second transition.
        return _repair_outcome(record, repaired=False, reason="not_repairable")

    state = read_unit_state_snapshot(context, work_unit_id, root=root_path)
    if state is None:  # pragma: no cover - classification already proved it exists
        raise ResumeRepairError(f"{work_unit_id} lost its state during repair")
    attempt = state["identity"]["attempt"]
    request_record = validate_request_file(
        context,
        work_unit_id,
        attempt,
        root=root_path,
        require_full_strength=scan_mode == SCAN_MODE_PRODUCTION_MERGE,
    )
    result_record = validate_result_file(
        context,
        work_unit_id,
        attempt,
        root=root_path,
        request_record=request_record,
        shard_index=state["identity"]["shard_index"],
        shard_count=state["identity"]["shard_count"],
        scan_mode=scan_mode,
    )
    proposed = _repaired_state(context, state, classification, request_record, result_record)
    # B2C validates the transition itself; this keeps the refusal typed and
    # local rather than surfacing as a bare contract error.
    try:
        work_units.validate_state_transition(state, proposed)
    except work_units.G8BlerWorkUnitError as exc:
        raise ResumeRepairError(f"refusing an illegal repair for {work_unit_id}: {exc}") from exc

    proposed_body = work_units.canonical_state_bytes(context.state_context, proposed)
    proposed_sha256 = sha256_bytes(proposed_body)
    try:
        work_units.replace_unit_state(
            context.state_context, target, proposed, state["state_sha256"], root=root_path
        )
    except work_units.StaleWriterError:
        raise
    except Exception as exc:
        # Publication may or may not have landed.  Never infer it from status:
        # reread the installed canonical bytes and require the exact proposed
        # digest and bytes before treating the uncertain operation as success.
        try:
            installed = read_unit_state_snapshot(context, work_unit_id, root=root_path)
            installed_body, installed_sha256, _installed_payload = _read_exact_artifact(
                target, "installed unit state"
            )
        except ResumeHoldError as read_exc:
            raise ResumeRepairError(
                f"repair of {work_unit_id} failed and installed state could not be proven: {exc}"
            ) from read_exc
        if (
            installed is None
            or installed_sha256 != proposed_sha256
            or installed_body != proposed_body
        ):
            raise ResumeRepairError(
                f"repair of {work_unit_id} failed and did not publish the exact proposed bytes: {exc}"
            ) from exc

    settled = classify_work_unit(
        context,
        work_unit_id,
        _census_runtime_root_locked(context, root=root_path, lease=lease),
        root=root_path,
        scan_mode=scan_mode,
    )
    expected_classes = POST_REPAIR_CLASSIFICATIONS[classification]
    if settled["classification"] not in expected_classes:
        raise ResumeRepairError(
            f"{work_unit_id} repaired to {settled['classification']!r}, "
            f"which is not one of {expected_classes}"
        )
    return _repair_outcome(settled, repaired=True, reason=classification)


def repair_work_unit(
    context: Any,
    work_unit_id: str,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    """Acquire the exclusive parent-directory lease before repairing one unit."""

    context = _resume_context(context)
    if scan_mode == SCAN_MODE_BOUNDED_SMOKE_INSPECTION:
        _require_bounded_smoke_root(root)
    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as lease:
        return _repair_work_unit_locked(
            context,
            work_unit_id,
            root=root,
            scan_mode=scan_mode,
            lease=lease,
        )


def _repair_outcome(record: Mapping[str, Any], *, repaired: bool, reason: str) -> dict[str, Any]:
    return _fresh(
        {
            "work_unit_id": record["work_unit_id"],
            "repaired": repaired,
            "from_classification": reason if repaired else None,
            "classification": record["classification"],
            "attempt": record["attempt"],
            "state_status": record["state_status"],
            "state_sha256": record["state_sha256"],
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        }
    )


def inspect_runtime_root(
    context: Any,
    *,
    root: Path | str | None = None,
    scan_mode: str = SCAN_MODE_PRODUCTION_MERGE,
    repair_mode: str = REPAIR_MODE_READ_ONLY,
) -> dict[str, Any]:
    """Authenticate, lock, census, classify and — only if asked — repair.

    The default is strictly read-only: it makes no filesystem change and no
    campaign-state change.  ``repair_mode`` must be set explicitly to
    ``repair_recoverable`` before any state is transitioned, and even then only
    the two recovery-matrix rows are touched.
    """

    context = _resume_context(context)
    if repair_mode not in REPAIR_MODES:
        raise ResumeRepairError(f"unknown repair mode {repair_mode!r}")
    if scan_mode not in SCAN_MODES:
        raise ResumeHoldError(f"unknown scan mode {scan_mode!r}")

    if scan_mode == SCAN_MODE_BOUNDED_SMOKE_INSPECTION:
        _require_bounded_smoke_root(root)

    with reconciliation_lock(root, mode=LOCK_MODE_EXCLUSIVE) as held:
        root_present = held.root_present
        census = _census_runtime_root_locked(context, root=root, lease=held)
        records = classify_runtime_root(context, census, root=root, scan_mode=scan_mode)
        repairs: list[dict[str, Any]] = []
        if repair_mode == REPAIR_MODE_REPAIR_RECOVERABLE:
            for record in records:
                if record["classification"] in REPAIRABLE_CLASSIFICATIONS:
                    repairs.append(
                        _repair_work_unit_locked(
                            context,
                            record["work_unit_id"],
                            root=root,
                            scan_mode=scan_mode,
                            lease=held,
                        )
                    )
            census = _census_runtime_root_locked(context, root=root, lease=held)
            records = classify_runtime_root(context, census, root=root, scan_mode=scan_mode)

    return _fresh(
        {
            "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
            "artifact_role": "g8_bler_resume_inspection",
            "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
            "root_present": root_present,
            "scan_mode": scan_mode,
            "repair_mode": repair_mode,
            "census": census,
            "classifications": list(records),
            "repairs": repairs,
            "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        }
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
    lock_directory_present: bool,
) -> dict[str, Any]:
    """Assemble the census in frozen authority order, never filesystem order."""

    record = {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "artifact_role": "g8_bler_runtime_census",
        "logical_root": WORK_UNIT_ROOT_LOGICAL_PREFIX,
        "root_present": root_present,
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
    "CAMPAIGN_RECONCILIATION_ARTIFACT_ROLE",
    "CAMPAIGN_RECONCILIATION_SCHEMA_VERSION",
    "CAMPAIGN_STATE_LOGICAL_PATH",
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
    "MERGE_REPORT_ARTIFACT_ROLE",
    "MERGE_REPORT_SCHEMA_VERSION",
    "NON_REPAIRABLE_CLASSIFICATIONS",
    "PHASE",
    "POST_REPAIR_CLASSIFICATIONS",
    "PROPOSED_ATTEMPT_POLICY",
    "ReconciliationLease",
    "RECOVERABLE_CLASSIFICATIONS",
    "REJECTED_UNREACHABLE_CLASSIFICATIONS",
    "REMAINING_CLASSIFICATIONS",
    "REPAIRABLE_CLASSIFICATIONS",
    "REPAIR_MATRIX",
    "REPAIR_MODES",
    "REPAIR_MODE_READ_ONLY",
    "REPAIR_MODE_REPAIR_RECOVERABLE",
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
    "RESUME_PLAN_ARTIFACT_ROLE",
    "RESUME_PLAN_SCHEMA_VERSION",
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
    "build_merge_report",
    "build_resume_plan",
    "apply_campaign_reconciliation",
    "census_runtime_root",
    "classification_for_shape",
    "classify_runtime_root",
    "classify_work_unit",
    "format_attempt",
    "inspect_runtime_root",
    "is_full_strength_merge_candidate",
    "logical_artifact_path",
    "logical_result_path",
    "merge_report_digest",
    "parse_attempt_token",
    "proposed_attempt",
    "propose_campaign_reconciliation",
    "read_unit_state_snapshot",
    "reconciliation_lock",
    "repair_work_unit",
    "request_path",
    "request_relative_path",
    "result_path",
    "result_relative_path",
    "resume_plan_digest",
    "state_path",
    "validate_request_file",
    "validate_result_file",
    "validate_attempt_history",
    "work_unit_digest",
]
