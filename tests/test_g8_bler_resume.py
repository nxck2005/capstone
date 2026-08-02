"""B3 exact-resume, recovery and merge-validation tests.

Every filesystem test uses an isolated temporary runtime root.  This file
contains no runner, channel, codec, dataset, classifier or test-split path, and
it never writes under the production runtime root.

Where process *lifetime* is the property under test — ``flock`` exclusion
between real processes, and release on a hard ``os._exit`` — these tests fork
real child processes.  Threads cannot substitute.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from baseline import g8_bler_contract as bler_contract
from baseline import g8_bler_resume as resume
from baseline import g8_bler_work_units as units


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def execution_context() -> units.AuthenticatedExecutionContext:
    return units.AuthenticatedExecutionContext()


@pytest.fixture(scope="module")
def state_context(
    execution_context: units.AuthenticatedExecutionContext,
) -> units.AuthenticatedUnitStateContext:
    return units.AuthenticatedUnitStateContext(execution_context)


@pytest.fixture(scope="module")
def context(
    state_context: units.AuthenticatedUnitStateContext,
) -> resume.AuthenticatedResumeContext:
    return resume.AuthenticatedResumeContext(state_context)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "work_units"


def _unit(context: resume.AuthenticatedResumeContext, index: int = 0) -> str:
    return context.ordered_work_unit_ids[index]


def _touch(path: Path, body: bytes = b"{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


@contextlib.contextmanager
def _forking() -> Any:
    """Fork children that only touch an isolated root and then ``os._exit``."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _place(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    kind: str,
    attempt: int | None = None,
    body: bytes = b"{}",
) -> Path:
    return _touch(resume.artifact_path(context, work_unit_id, kind, attempt, root=root), body)


# ---------------------------------------------------------------------------
# Authenticated context
# ---------------------------------------------------------------------------


def test_context_authenticates_the_immutable_authority(
    context: resume.AuthenticatedResumeContext,
) -> None:
    authority = context.authority_binding()
    assert authority["campaign_id"] == resume.EXPECTED_CAMPAIGN_ID
    assert authority["campaign_manifest_sha256"] == resume.EXPECTED_CAMPAIGN_MANIFEST_SHA256
    assert authority["required_bler_artifact_sha256"] == resume.EXPECTED_REQUIRED_IDENTITIES_SHA256
    assert authority["selection_policy_sha256"] == resume.EXPECTED_SELECTION_POLICY_SHA256
    assert authority["bler_tooling_contract_id"] == resume.EXPECTED_B1C_CONTRACT_ID
    assert authority["bler_tooling_contract_sha256"] == resume.EXPECTED_B1C_CONTRACT_SHA256
    assert authority["request_schema_version"] == resume.REQUEST_SCHEMA_VERSION == 2
    assert authority["result_schema_version"] == resume.RESULT_SCHEMA_VERSION == 2
    assert context.required_work_unit_count == resume.EXPECTED_REQUIRED_WORK_UNIT_COUNT
    assert len(context.ordered_work_unit_ids) == resume.EXPECTED_REQUIRED_WORK_UNIT_COUNT


def test_context_binds_the_registered_b2c_state_contract(
    context: resume.AuthenticatedResumeContext,
) -> None:
    binding = context.state_contract_binding()
    assert binding["bler_state_contract_id"] == resume.EXPECTED_B2C_CONTRACT_ID
    assert binding["bler_state_contract_sha256"] == resume.EXPECTED_B2C_CONTRACT_SHA256


def test_plain_execution_context_is_upgraded_but_a_stranger_is_rejected(
    execution_context: units.AuthenticatedExecutionContext,
) -> None:
    upgraded = resume.AuthenticatedResumeContext(execution_context)
    assert upgraded.required_work_unit_count == resume.EXPECTED_REQUIRED_WORK_UNIT_COUNT
    with pytest.raises(TypeError):
        resume.AuthenticatedResumeContext(object())


def test_public_records_are_fresh_copies(
    context: resume.AuthenticatedResumeContext,
) -> None:
    first = context.authority_binding()
    first["campaign_id"] = "poisoned"
    assert context.authority_binding()["campaign_id"] == resume.EXPECTED_CAMPAIGN_ID

    unit = _unit(context)
    record = context.work_unit_record(unit)
    record["work_unit_id"] = "poisoned"
    assert context.work_unit_record(unit)["work_unit_id"] == unit

    binding = context.state_contract_binding()
    binding["bler_state_contract_id"] = "poisoned"
    assert context.state_contract_binding()["bler_state_contract_id"] == resume.EXPECTED_B2C_CONTRACT_ID


def test_required_artifact_is_authenticated_once_per_context_not_once_per_unit(
    context: resume.AuthenticatedResumeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 8.6 MB required-identity artifact must not be reread per unit."""

    from baseline import g8_bler_contract

    calls = {"count": 0}
    original = g8_bler_contract._required_work_unit_bytes

    def counting() -> Any:
        calls["count"] += 1
        return original()

    monkeypatch.setattr(g8_bler_contract, "_required_work_unit_bytes", counting)
    for work_unit_id in context.ordered_work_unit_ids[:200]:
        context.ordinal(work_unit_id)
        context.work_unit_record_sha256(work_unit_id)
        resume.work_unit_digest(work_unit_id)
        context.work_unit_id_for_digest(resume.work_unit_digest(work_unit_id))
    assert calls["count"] == 0


def test_resume_contract_is_absent_before_registration(
    context: resume.AuthenticatedResumeContext,
) -> None:
    binding = context.resume_contract_binding()
    if binding is None:
        with pytest.raises(resume.ResumeContractAuthenticationError):
            context.require_resume_contract_binding()
    else:
        assert binding["bler_resume_contract_id"].startswith(
            f"{resume.RESUME_CONTRACT_ID_PREFIX}-"
        )


def test_context_rejects_a_campaign_state_with_two_resume_bindings(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    payload = json.loads(state_context.campaign_state_path.read_bytes())
    entry = {
        "path": resume.RESUME_CONTRACT_REPO_RELATIVE_PATH,
        "sha256": "a" * 64,
        "bytes": 10,
    }
    payload["identity"]["produced_artifacts"].extend([entry, dict(entry)])
    forged = tmp_path / "campaign_state.json"
    forged.write_bytes(json.dumps(payload).encode("utf-8"))
    with pytest.raises(resume.ResumeContractAuthenticationError):
        resume.AuthenticatedResumeContext._registered_resume_binding(forged)


def test_context_requires_registration_when_asked(
    execution_context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    """An unregistered B3 contract must fail closed, not default to absent.

    The fixture strips any resume binding from a copy of the live campaign
    state, so this holds identically before and after B3.8 registration.
    """

    payload = json.loads(units.DEFAULT_CAMPAIGN_STATE_PATH.read_bytes())
    payload["identity"]["produced_artifacts"] = [
        entry
        for entry in payload["identity"]["produced_artifacts"]
        if entry.get("path") != resume.RESUME_CONTRACT_REPO_RELATIVE_PATH
    ]
    stripped = tmp_path / "campaign_state.json"
    stripped.write_bytes(json.dumps(payload).encode("utf-8"))
    unregistered = units.AuthenticatedUnitStateContext(
        execution_context, campaign_state_path=stripped
    )
    context = resume.AuthenticatedResumeContext(unregistered)
    assert context.resume_contract_binding() is None
    with pytest.raises(resume.ResumeContractAuthenticationError):
        context.require_resume_contract_binding()
    with pytest.raises(resume.ResumeContractAuthenticationError):
        resume.AuthenticatedResumeContext(unregistered, require_resume_contract=True)


# ---------------------------------------------------------------------------
# Attempt grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["1", "2", "10", "4096"])
def test_attempt_tokens_that_are_exact_positive_integers_parse(token: str) -> None:
    assert resume.parse_attempt_token(token) == int(token)


@pytest.mark.parametrize(
    "token",
    ["01", "0", "+1", "-1", "1.0", " 1", "1 ", "", "1_0", "0x1", "١", "1\n"],
)
def test_malformed_attempt_tokens_are_rejected(token: str) -> None:
    with pytest.raises(resume.ResumeCensusError):
        resume.parse_attempt_token(token)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None])
def test_format_attempt_rejects_non_positive_exact_integers(value: Any) -> None:
    with pytest.raises(resume.ResumeCensusError):
        resume.format_attempt(value)


# ---------------------------------------------------------------------------
# Canonical runtime layout
# ---------------------------------------------------------------------------


def test_artifact_paths_follow_the_frozen_grammar(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    digest = resume.work_unit_digest(unit)
    bucket = digest[:2]

    assert resume.state_path(context, unit, root=root) == root / bucket / f"{digest}.state.json"
    assert (
        resume.request_path(context, unit, 1, root=root)
        == root / bucket / f"{digest}.attempt-1.request.json"
    )
    assert (
        resume.result_path(context, unit, 12, root=root)
        == root / bucket / f"{digest}.attempt-12.result.json"
    )
    assert resume.logical_result_path(context, unit, 3) == (
        f"results/baseline/g8/work_units/{bucket}/{digest}.attempt-3.result.json"
    )
    # The logical path never leaks the caller-supplied physical root.
    assert str(root) not in resume.logical_result_path(context, unit, 3)


def test_state_path_matches_the_b2c_derivation_exactly(
    context: resume.AuthenticatedResumeContext,
    state_context: units.AuthenticatedUnitStateContext,
    root: Path,
) -> None:
    for unit in context.ordered_work_unit_ids[:50]:
        assert resume.state_path(context, unit, root=root) == units.unit_state_path(
            state_context, unit, root=root
        )


def test_unknown_work_unit_ids_and_kinds_are_refused(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    with pytest.raises(units.AuthorityAuthenticationError):
        resume.request_path(context, "not-a-required-unit", 1, root=root)
    with pytest.raises(resume.G8BlerResumeError):
        resume.artifact_path(context, _unit(context), "manifest", 1, root=root)


def test_relative_and_symlinked_roots_are_refused(
    context: resume.AuthenticatedResumeContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=Path("relative/work_units"))

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=link)

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=dangling)


def test_a_root_that_is_a_regular_file_is_refused(
    context: resume.AuthenticatedResumeContext,
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=plain)


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def test_absent_root_is_valid_and_means_all_work_remains(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    census = resume.census_runtime_root(context, root=root)
    assert census["root_present"] is False
    assert census["state_work_unit_ids"] == []
    assert census["request_attempts"] == {}
    assert census["result_attempts"] == {}
    assert census["ignored_orphan_staging_count"] == 0
    assert census["required_work_unit_count"] == resume.EXPECTED_REQUIRED_WORK_UNIT_COUNT
    assert census["test_split_access"] == 0
    assert not root.exists()


def test_census_reports_artifacts_in_frozen_authority_order(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    units_in_reverse = list(context.ordered_work_unit_ids[:12])[::-1]
    for work_unit_id in units_in_reverse:
        _place(context, root, work_unit_id, resume.ARTIFACT_KIND_STATE)
        _place(context, root, work_unit_id, resume.ARTIFACT_KIND_REQUEST, 1)
    census = resume.census_runtime_root(context, root=root)
    assert census["state_work_unit_ids"] == list(context.ordered_work_unit_ids[:12])
    assert list(census["request_attempts"]) == list(context.ordered_work_unit_ids[:12])


def test_census_orders_attempts_numerically_not_lexically(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    for attempt in (10, 2, 1, 21, 3):
        _place(context, root, unit, resume.ARTIFACT_KIND_REQUEST, attempt)
    census = resume.census_runtime_root(context, root=root)
    assert census["request_attempts"][unit] == [1, 2, 3, 10, 21]


def test_census_is_invariant_under_randomized_enumeration_order(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    for work_unit_id in context.ordered_work_unit_ids[:20]:
        _place(context, root, work_unit_id, resume.ARTIFACT_KIND_STATE)
        _place(context, root, work_unit_id, resume.ARTIFACT_KIND_RESULT, 2)
    baseline = json.dumps(resume.census_runtime_root(context, root=root), sort_keys=True)

    original = resume._scandir

    def shuffled(path: Path) -> list[Any]:
        entries = original(path)
        random.Random(1234).shuffle(entries)
        return entries

    monkeypatch.setattr(resume, "_scandir", shuffled)
    for _ in range(5):
        assert json.dumps(resume.census_runtime_root(context, root=root), sort_keys=True) == baseline


def test_census_accepts_the_lock_directory_and_reconciliation_lock(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    digest = resume.work_unit_digest(unit)
    root.mkdir(parents=True)
    (root / resume.RECONCILIATION_LOCK_NAME).write_bytes(b"")
    locks = root / units.LOCK_DIRECTORY_NAME
    locks.mkdir()
    (locks / f"{digest}{units.LOCK_FILENAME_SUFFIX}").write_bytes(b"")
    census = resume.census_runtime_root(context, root=root)
    assert census["reconciliation_lock_present"] is True
    assert census["lock_directory_present"] is True
    assert census["lock_file_count"] == 1


def test_orphan_staging_is_ignored_but_counted(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    digest = resume.work_unit_digest(unit)
    bucket = root / digest[:2]
    bucket.mkdir(parents=True)
    (bucket / f".{digest}.state.json.4242.{'ab' * 12}.staging").write_bytes(b"partial")
    census = resume.census_runtime_root(context, root=root)
    assert census["ignored_orphan_staging_count"] == 1
    assert census["state_work_unit_ids"] == []


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        "README",
        "0",
        "AB",
        "abc",
        "0g",
        ".hidden",
        "work_units",
    ],
)
def test_unknown_top_level_entries_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    name: str,
) -> None:
    root.mkdir(parents=True)
    (root / name).write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_a_bucket_that_is_not_a_directory_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    root.mkdir(parents=True)
    (root / "ab").write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_wrong_bucket_placement_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    digest = resume.work_unit_digest(unit)
    wrong = "ff" if digest[:2] != "ff" else "ee"
    _touch(root / wrong / f"{digest}.state.json")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_uppercase_digest_filenames_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    digest = resume.work_unit_digest(_unit(context))
    _touch(root / digest[:2] / f"{digest.upper()}.state.json")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_unknown_digests_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    digest = "c" * 64
    assert not context.known_digest(digest)
    _touch(root / digest[:2] / f"{digest}.state.json")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


@pytest.mark.parametrize("token", ["01", "0", "+1", "1.0", "x"])
def test_malformed_attempt_filenames_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    token: str,
) -> None:
    digest = resume.work_unit_digest(_unit(context))
    _touch(root / digest[:2] / f"{digest}.attempt-{token}.request.json")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


@pytest.mark.parametrize(
    "name",
    [
        "{digest}.state.jsonl",
        "{digest}.state",
        "{digest}.request.json",
        "{digest}.attempt-1.json",
        "{digest}.attempt-1.request.JSON",
        "{digest}.attempt-1.result.json.bak",
        "{digest}.tmp",
        ".{digest}.state.json.swp",
    ],
)
def test_alternate_extensions_and_temporaries_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    name: str,
) -> None:
    digest = resume.work_unit_digest(_unit(context))
    _touch(root / digest[:2] / name.format(digest=digest))
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_nested_directories_inside_a_bucket_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    digest = resume.work_unit_digest(_unit(context))
    (root / digest[:2] / "nested").mkdir(parents=True)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_hard_linked_authoritative_aliases_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    other = context.ordered_work_unit_ids[1]
    first = _place(context, root, unit, resume.ARTIFACT_KIND_STATE)
    alias = resume.state_path(context, other, root=root)
    alias.parent.mkdir(parents=True, exist_ok=True)
    os.link(first, alias)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


@pytest.mark.parametrize("kind", [resume.ARTIFACT_KIND_STATE, resume.ARTIFACT_KIND_REQUEST])
def test_symlinked_and_dangling_artifacts_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    unit = _unit(context)
    attempt = None if kind == resume.ARTIFACT_KIND_STATE else 1
    target = tmp_path / "elsewhere.json"
    target.write_bytes(b"{}")
    path = resume.artifact_path(context, unit, kind, attempt, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)

    path.unlink()
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_symlinked_bucket_and_lock_directory_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    digest = resume.work_unit_digest(_unit(context))
    root.mkdir(parents=True)
    (root / digest[:2]).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)
    (root / digest[:2]).unlink()

    (root / units.LOCK_DIRECTORY_NAME).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_symlinked_reconciliation_lock_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    tmp_path: Path,
) -> None:
    root.mkdir(parents=True)
    target = tmp_path / "other.lock"
    target.write_bytes(b"")
    (root / resume.RECONCILIATION_LOCK_NAME).symlink_to(target)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)
    with pytest.raises(resume.ResumeCensusError):
        with resume.reconciliation_lock(root):
            pass


def test_non_regular_authoritative_objects_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    path = resume.state_path(context, unit, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_unknown_entries_in_the_lock_directory_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    locks = root / units.LOCK_DIRECTORY_NAME
    locks.mkdir(parents=True)
    (locks / "stray.txt").write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


def test_staging_for_a_foreign_or_unknown_digest_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    digest = resume.work_unit_digest(_unit(context))
    other = "d" * 64
    bucket = root / digest[:2]
    bucket.mkdir(parents=True)
    (bucket / f".{other}.state.json.11.{'cd' * 12}.staging").write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


# ---------------------------------------------------------------------------
# Global reconciliation lock
# ---------------------------------------------------------------------------


def test_lock_on_an_absent_root_does_not_create_it(
    root: Path,
) -> None:
    with resume.reconciliation_lock(root) as held:
        assert held == {"held": False, "root_present": False, "mode": "exclusive"}
    assert not root.exists()


def test_lock_creates_only_the_lock_file_when_the_root_exists(root: Path) -> None:
    root.mkdir(parents=True)
    with resume.reconciliation_lock(root) as held:
        assert held["held"] is True
    entries = sorted(p.name for p in root.iterdir())
    assert entries == [resume.RECONCILIATION_LOCK_NAME]
    assert stat.S_ISREG(os.lstat(root / resume.RECONCILIATION_LOCK_NAME).st_mode)


def test_lock_order_constant_is_global_then_per_unit() -> None:
    assert resume.LOCK_ORDER == ("global_reconciliation_lock", "per_unit_b2c_lock")


def test_unknown_lock_mode_is_refused(root: Path) -> None:
    root.mkdir(parents=True)
    with pytest.raises(resume.ResumeLockError):
        with resume.reconciliation_lock(root, mode="advisory"):
            pass


def _child_tries_exclusive(root: Path, ready: Path, done: Path) -> None:
    """Child body: report whether an exclusive acquisition would block."""

    import fcntl as _fcntl

    fd = os.open(str(root / resume.RECONCILIATION_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            done.write_bytes(b"blocked")
            os._exit(0)
        done.write_bytes(b"acquired")
        os._exit(0)
    finally:  # pragma: no cover - os._exit above always wins
        os.close(fd)


def test_exclusive_lock_excludes_a_real_child_process(root: Path, tmp_path: Path) -> None:
    root.mkdir(parents=True)
    done = tmp_path / "verdict"
    with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            _child_tries_exclusive(root, tmp_path / "ready", done)
        os.waitpid(pid, 0)
        assert done.read_bytes() == b"blocked"


def test_shared_locks_coexist_between_real_processes(root: Path, tmp_path: Path) -> None:
    root.mkdir(parents=True)
    done = tmp_path / "verdict"
    with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_SHARED):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            import fcntl as _fcntl

            fd = os.open(str(root / resume.RECONCILIATION_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            except OSError:
                done.write_bytes(b"blocked")
                os._exit(0)
            done.write_bytes(b"shared")
            os._exit(0)
        os.waitpid(pid, 0)
        assert done.read_bytes() == b"shared"


def test_shared_lock_is_excluded_by_an_exclusive_holder(root: Path, tmp_path: Path) -> None:
    root.mkdir(parents=True)
    done = tmp_path / "verdict"
    with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            import fcntl as _fcntl

            fd = os.open(str(root / resume.RECONCILIATION_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            except OSError:
                done.write_bytes(b"blocked")
                os._exit(0)
            done.write_bytes(b"shared")
            os._exit(0)
        os.waitpid(pid, 0)
        assert done.read_bytes() == b"blocked"


def test_hard_process_exit_releases_the_reconciliation_lock(root: Path, tmp_path: Path) -> None:
    """A killed holder must not leave the campaign permanently locked."""

    root.mkdir(parents=True)
    holding = tmp_path / "holding"
    with _forking():
        pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE):
            holding.write_bytes(b"1")
            time.sleep(30)
        os._exit(0)
    deadline = time.time() + 30
    while not holding.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert holding.exists()
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    # The kernel drops the flock at process death, so this must not hang.
    with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE) as held:
        assert held["held"] is True


def test_lock_is_released_on_exception(root: Path) -> None:
    root.mkdir(parents=True)
    with pytest.raises(ValueError):
        with resume.reconciliation_lock(root):
            raise ValueError("boom")
    with resume.reconciliation_lock(root) as held:
        assert held["held"] is True


def test_lock_never_creates_the_production_runtime_root() -> None:
    production = resume.DEFAULT_WORK_UNIT_ROOT
    existed = production.exists()
    with resume.reconciliation_lock() as held:
        if not existed:
            assert held["root_present"] is False
    assert production.exists() == existed


# ---------------------------------------------------------------------------
# B3.H1 — the corrected recovery model (G8_B3 §13, §16, §17, §29)
#
# B2C deliberately keeps a `claimed` state request-unbound: a claim is a
# pre-execution reservation, and a published request file is immutable attempt
# history rather than a state transition.  The pre-correction B3 instruction
# defined four classes and the first repair row in terms of a request-bound
# claim, which that schema makes unreachable.  These tests freeze the corrected
# eight-class enum and two-row repair matrix now; B3.6 binds them into the
# generated contract.
# ---------------------------------------------------------------------------


def _plan(context: resume.AuthenticatedResumeContext) -> dict[str, Any]:
    return units.build_shard_plan(context.state_context, 1, 0)


def _clean_claim(
    context: resume.AuthenticatedResumeContext,
    work_unit_id: str,
    *,
    attempt: int = 1,
) -> dict[str, Any]:
    return units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        attempt=attempt,
        status=units.STATUS_CLAIMED,
    )


def test_b2c_rejects_a_request_bound_claimed_state(
    context: resume.AuthenticatedResumeContext,
) -> None:
    work_unit_id = _unit(context)
    digest = "a" * 64
    with pytest.raises(units.UnitStateError):
        units.build_unit_state(
            context.state_context,
            work_unit_id,
            _plan(context),
            status=units.STATUS_CLAIMED,
            request_sha256=digest,
        )
    forged = _clean_claim(context, work_unit_id)
    forged["identity"] = dict(forged["identity"]) | {"request_sha256": digest}
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context.state_context, forged)


def test_b3_never_includes_the_rejected_unreachable_classifications() -> None:
    assert resume.REJECTED_UNREACHABLE_CLASSIFICATIONS == (
        "claimed_request_bound",
        "recoverable_request_binding",
    )
    for rejected in resume.REJECTED_UNREACHABLE_CLASSIFICATIONS:
        assert rejected not in resume.CLASSIFICATIONS
        assert rejected not in resume.REPAIRABLE_CLASSIFICATIONS
        assert rejected not in resume.RECOVERABLE_CLASSIFICATIONS
        assert rejected not in resume.REMAINING_CLASSIFICATIONS
        assert rejected not in resume.PROPOSED_ATTEMPT_POLICY
        assert rejected not in {name for name, _ in resume.REPAIR_MATRIX}
        assert rejected not in {target for _, target in resume.REPAIR_MATRIX}


def test_the_closed_enum_is_exactly_the_eight_reachable_classes() -> None:
    assert resume.CLASSIFICATIONS == (
        "absent",
        "claimed_unbound",
        "claimed_request_published",
        "recoverable_failed_result",
        "recoverable_complete_result",
        "failed_retryable",
        "completed_full_strength",
        "terminal_nonmergeable",
    )
    assert len(set(resume.CLASSIFICATIONS)) == 8
    assert set(resume.PROPOSED_ATTEMPT_POLICY) == set(resume.CLASSIFICATIONS)
    assert (
        set(resume.RECOVERABLE_CLASSIFICATIONS)
        | set(resume.REMAINING_CLASSIFICATIONS)
        | set(resume.TERMINAL_CLASSIFICATIONS)
    ) == set(resume.CLASSIFICATIONS)


def test_clean_claim_without_a_request_is_claimed_unbound() -> None:
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=False,
            request_present=False,
            result_status=None,
        )
        == "claimed_unbound"
    )


def test_clean_claim_with_one_exact_request_and_no_result_is_request_published() -> None:
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=False,
            request_present=True,
            result_status=None,
        )
        == "claimed_request_published"
    )


def test_a_published_request_is_never_repaired_and_is_remaining_work() -> None:
    assert "claimed_request_published" not in resume.REPAIRABLE_CLASSIFICATIONS
    assert "claimed_request_published" in resume.NON_REPAIRABLE_CLASSIFICATIONS
    assert "claimed_request_published" in resume.REMAINING_CLASSIFICATIONS
    assert "claimed_request_published" not in resume.RECOVERABLE_CLASSIFICATIONS
    # Read-only inspection and explicit repair both leave every other class
    # untouched; only the two result-bearing classes are ever transitioned.
    assert set(resume.NON_REPAIRABLE_CLASSIFICATIONS) == {
        "absent",
        "claimed_unbound",
        "claimed_request_published",
        "failed_retryable",
        "completed_full_strength",
        "terminal_nonmergeable",
    }


def test_a_published_request_proposes_exactly_the_next_attempt() -> None:
    assert resume.PROPOSED_ATTEMPT_POLICY["claimed_request_published"] == "old_attempt_plus_1"
    assert resume.PROPOSED_ATTEMPT_POLICY["claimed_unbound"] == "old_attempt_plus_1"
    assert resume.PROPOSED_ATTEMPT_POLICY["failed_retryable"] == "old_attempt_plus_1"
    assert resume.PROPOSED_ATTEMPT_POLICY["absent"] == "attempt_1"
    for terminal_or_recoverable in (
        "recoverable_failed_result",
        "recoverable_complete_result",
        "completed_full_strength",
        "terminal_nonmergeable",
    ):
        assert resume.PROPOSED_ATTEMPT_POLICY[terminal_or_recoverable] is None


def test_request_bytes_reproduce_identically_at_the_next_attempt(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    first = bler_contract.canonical_json(bler_contract.build_full_strength_request(work_unit_id))
    second = bler_contract.canonical_json(bler_contract.build_full_strength_request(work_unit_id))
    # Request content carries no attempt or shard identity, so a retry
    # republishes byte-identical bytes at a different attempt path.
    assert first == second
    old = resume.artifact_path(context, work_unit_id, resume.ARTIFACT_KIND_REQUEST, 1, root=root)
    new = resume.artifact_path(context, work_unit_id, resume.ARTIFACT_KIND_REQUEST, 2, root=root)
    assert old != new
    assert old.parent == new.parent


def test_clean_claim_plus_failed_result_is_recoverable_failed_result() -> None:
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=False,
            request_present=True,
            result_status=bler_contract.STATUS_FAILED,
        )
        == "recoverable_failed_result"
    )
    assert "recoverable_failed_result" in resume.RECOVERABLE_CLASSIFICATIONS


def test_clean_claim_plus_complete_result_is_recoverable_complete_result() -> None:
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=False,
            request_present=True,
            result_status=bler_contract.STATUS_COMPLETE,
        )
        == "recoverable_complete_result"
    )
    assert "recoverable_complete_result" in resume.RECOVERABLE_CLASSIFICATIONS


def test_the_repair_matrix_has_exactly_two_direct_rows() -> None:
    assert resume.REPAIR_MATRIX == (
        ("recoverable_failed_result", units.STATUS_FAILED),
        ("recoverable_complete_result", units.STATUS_RESULT_LINKED),
    )
    assert resume.POST_REPAIR_CLASSIFICATIONS == {
        "recoverable_failed_result": ("failed_retryable",),
        "recoverable_complete_result": (
            "completed_full_strength",
            "terminal_nonmergeable",
        ),
    }


def test_direct_repair_transitions_pass_the_frozen_b2c_validator(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    claimed = _clean_claim(context, work_unit_id)
    for field, expected in resume.FROZEN_CLAIMED_STATE_FIELDS.items():
        assert claimed["identity"][field] == expected

    request_sha = "b" * 64
    result_sha = "c" * 64
    result_path = str(
        resume.artifact_relative_path(context, work_unit_id, resume.ARTIFACT_KIND_RESULT, 1)
    )

    # claimed, request-unbound -> failed, request-bound
    repaired_failed = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        attempt=claimed["identity"]["attempt"],
        status=units.STATUS_FAILED,
        request_sha256=request_sha,
        scientific_execution_performed=True,
        trials_completed=1234,
    )
    units.validate_state_transition(claimed, repaired_failed)
    failed_identity = repaired_failed["identity"]
    assert failed_identity["request_sha256"] == request_sha
    assert failed_identity["result_path"] is None
    assert failed_identity["result_sha256"] is None
    assert failed_identity["scientific_execution_performed"] is True
    assert failed_identity["trials_completed"] == 1234
    assert failed_identity["test_split_access"] == 0
    assert failed_identity["attempt"] == claimed["identity"]["attempt"]
    assert failed_identity["shard_index"] == claimed["identity"]["shard_index"]

    # claimed, request-unbound -> result_linked, request-and-result-bound
    repaired_linked = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        attempt=claimed["identity"]["attempt"],
        status=units.STATUS_RESULT_LINKED,
        request_sha256=request_sha,
        result_path=result_path,
        result_sha256=result_sha,
        scientific_execution_performed=True,
        trials_completed=5000,
    )
    units.validate_state_transition(claimed, repaired_linked)
    linked_identity = repaired_linked["identity"]
    assert linked_identity["request_sha256"] == request_sha
    assert linked_identity["result_path"] == result_path
    assert linked_identity["result_sha256"] == result_sha
    assert linked_identity["scientific_execution_performed"] is True
    assert linked_identity["trials_completed"] == 5000
    assert linked_identity["test_split_access"] == 0
    assert linked_identity["attempt"] == claimed["identity"]["attempt"]
    assert linked_identity["shard_index"] == claimed["identity"]["shard_index"]


def test_a_request_bound_claim_holds_before_classification_is_reached() -> None:
    with pytest.raises(resume.ResumeContradictionError):
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=True,
            request_present=True,
            result_status=None,
        )
    with pytest.raises(resume.ResumeContradictionError):
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=True,
            request_present=True,
            result_status=bler_contract.STATUS_COMPLETE,
        )
    # A result without its exact request, and an artifact without any state,
    # are HOLDs rather than benign classifications.
    with pytest.raises(resume.ResumeContradictionError):
        resume.classification_for_shape(
            state_status=units.STATUS_CLAIMED,
            state_request_bound=False,
            request_present=False,
            result_status=bler_contract.STATUS_COMPLETE,
        )
    with pytest.raises(resume.ResumeContradictionError):
        resume.classification_for_shape(
            state_status=None,
            state_request_bound=False,
            request_present=True,
            result_status=None,
        )


def test_terminal_nonmergeable_is_bounded_smoke_only_and_holds_in_production() -> None:
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_RESULT_LINKED,
            state_request_bound=True,
            request_present=True,
            result_status=bler_contract.STATUS_COMPLETE,
            result_merge_eligible=False,
            scan_mode=resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION,
        )
        == "terminal_nonmergeable"
    )
    with pytest.raises(resume.ResumeContradictionError):
        resume.classification_for_shape(
            state_status=units.STATUS_RESULT_LINKED,
            state_request_bound=True,
            request_present=True,
            result_status=bler_contract.STATUS_COMPLETE,
            result_merge_eligible=False,
            scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        )
    assert (
        resume.classification_for_shape(
            state_status=units.STATUS_RESULT_LINKED,
            state_request_bound=True,
            request_present=True,
            result_status=bler_contract.STATUS_COMPLETE,
        )
        == "completed_full_strength"
    )


def test_the_future_generator_and_verifier_bind_the_corrected_model() -> None:
    """B3.6 binds these; until then the tools must not exist at all."""

    generator = resume.REPO_ROOT / "tools/gen_g8_bler_resume_contract.py"
    verifier = resume.REPO_ROOT / "tools/verify_g8_bler_resume_contract.py"
    for tool in (generator, verifier):
        if not tool.exists():
            continue
        source = tool.read_text(encoding="utf-8")
        for name in resume.CLASSIFICATIONS:
            assert name in source, f"{tool.name} omits classification {name}"
        for rejected in resume.REJECTED_UNREACHABLE_CLASSIFICATIONS:
            assert rejected not in source, f"{tool.name} names rejected class {rejected}"
        for name, target in resume.REPAIR_MATRIX:
            assert name in source and target in source
