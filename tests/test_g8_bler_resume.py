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
import errno
import fcntl
import json
import os
import select
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


def test_fast_validators_cover_200_request_result_pairs_without_artifact_reloads(
    context: resume.AuthenticatedResumeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict B3 hot path remains contract-equivalent after authentication."""

    requests: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for work_unit_id in context.ordered_work_unit_ids[:200]:
        request = bler_contract.build_full_strength_request(work_unit_id)
        result = bler_contract.build_work_unit_result(
            request=request,
            status=bler_contract.STATUS_COMPLETE,
            trials_completed=request["trials_requested"],
            bit_errors=5,
            block_errors=1,
            execution_metadata={
                "wall_time_s": None,
                "hostname": None,
                "device": None,
                "shard_index": 0,
                "shard_count": 1,
                "attempt": 1,
            },
        )
        requests.append(request)
        results.append(result)

    calls = {"tooling": 0, "required_bytes": 0, "required_index": 0, "required_one": 0}
    from baseline import g8_bler_contract

    def forbidden_tooling(*_args: Any, **_kwargs: Any) -> Any:
        calls["tooling"] += 1
        raise AssertionError("B3 fast validation reread the B1C tooling contract")

    def forbidden_required_bytes(*_args: Any, **_kwargs: Any) -> Any:
        calls["required_bytes"] += 1
        raise AssertionError("B3 fast validation reread the required artifact")

    def forbidden_required_index(*_args: Any, **_kwargs: Any) -> Any:
        calls["required_index"] += 1
        raise AssertionError("B3 fast validation reconstructed the required index")

    def forbidden_required_one(*_args: Any, **_kwargs: Any) -> Any:
        calls["required_one"] += 1
        raise AssertionError("B3 fast validation called the public required-unit loader")

    monkeypatch.setattr(g8_bler_contract, "load_bler_tooling_contract", forbidden_tooling)
    monkeypatch.setattr(g8_bler_contract, "_required_work_unit_bytes", forbidden_required_bytes)
    monkeypatch.setattr(g8_bler_contract, "required_work_unit_index", forbidden_required_index)
    monkeypatch.setattr(g8_bler_contract, "required_work_unit", forbidden_required_one)

    for request in requests:
        assert resume._fast_validate_request(context, request)["work_unit_id"] == request["work_unit_id"]
    for request, result in zip(requests, results, strict=True):
        assert resume._fast_validate_result(context, result, request=request)["status"] == bler_contract.STATUS_COMPLETE
    assert calls == {"tooling": 0, "required_bytes": 0, "required_index": 0, "required_one": 0}


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


def test_census_accepts_the_b2c_lock_directory_only(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    unit = _unit(context)
    digest = resume.work_unit_digest(unit)
    root.mkdir(parents=True)
    locks = root / units.LOCK_DIRECTORY_NAME
    locks.mkdir()
    (locks / f"{digest}{units.LOCK_FILENAME_SUFFIX}").write_bytes(b"")
    census = resume.census_runtime_root(context, root=root)
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


def test_unknown_root_lock_entry_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    tmp_path: Path,
) -> None:
    root.mkdir(parents=True)
    (root / "legacy.lock").write_bytes(b"")
    with pytest.raises(resume.ResumeCensusError):
        resume.census_runtime_root(context, root=root)


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
    parent_before = sorted(entry.name for entry in root.parent.iterdir())
    with resume.reconciliation_lock(root) as held:
        assert held.active is True
        assert held.root_present is False
        assert held.mode == resume.LOCK_MODE_EXCLUSIVE
        assert held.owner_pid == os.getpid()
        assert held.canonical_root == root.resolve(strict=False)
    assert not root.exists()
    assert sorted(entry.name for entry in root.parent.iterdir()) == parent_before


def test_lock_on_an_existing_root_creates_no_lock_entry(root: Path) -> None:
    root.mkdir(parents=True)
    before = sorted(entry.name for entry in root.iterdir())
    with resume.reconciliation_lock(root) as held:
        assert held.active is True
        assert held.root_present is True
        assert held.parent_device == root.parent.stat().st_dev
        assert held.parent_inode == root.parent.stat().st_ino
    entries = sorted(p.name for p in root.iterdir())
    assert entries == before == []


def test_lock_order_constant_is_global_then_per_unit() -> None:
    assert resume.LOCK_ORDER == ("global_reconciliation_lock", "per_unit_b2c_lock")


def test_unknown_lock_mode_is_refused(root: Path) -> None:
    root.mkdir(parents=True)
    with pytest.raises(resume.ResumeLockError):
        with resume.reconciliation_lock(root, mode="advisory"):
            pass


def _child_tries_parent_lock(root: Path, mode: int, done: Path) -> None:
    """Child body: report whether the parent-directory lock would block."""

    import fcntl as _fcntl

    fd = os.open(
        str(root.parent),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        try:
            _fcntl.flock(fd, mode | _fcntl.LOCK_NB)
        except OSError:
            done.write_bytes(b"blocked")
            os._exit(0)
        done.write_bytes(b"acquired")
        os._exit(0)
    finally:  # pragma: no cover - os._exit above always wins
        os.close(fd)


def _child_waits_for_parent_lock(
    root: Path,
    mode: int,
    inherited_fd: int,
    write_fd: int,
) -> None:
    """Child body for a blocking parent-directory lease assertion."""

    # The child inherited the parent's open file description.  Close only the
    # child copy before opening a fresh description; the parent's lease remains
    # held in the parent process and therefore still excludes this acquisition.
    os.close(inherited_fd)
    fd = os.open(
        str(root.parent),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        fcntl.flock(fd, mode)
        os.write(write_fd, b"acquired")
    finally:  # pragma: no cover - child exits immediately after the write
        os.close(fd)
        os.close(write_fd)
    os._exit(0)


def _wait_for_pipe_byte(read_fd: int, timeout: float = 2.0) -> bytes:
    ready, _unused_writable, _unused_errors = select.select([read_fd], [], [], timeout)
    return os.read(read_fd, 64) if ready else b""


def test_absent_root_exclusive_lock_blocks_a_shared_worker_before_root_creation(
    root: Path,
) -> None:
    read_fd, write_fd = os.pipe()
    try:
        with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE) as held:
            with _forking():
                pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                _child_waits_for_parent_lock(root, fcntl.LOCK_SH, held._parent_fd, write_fd)
            os.close(write_fd)
            write_fd = -1
            assert _wait_for_pipe_byte(read_fd, 0.2) == b""
            assert not root.exists()
        assert _wait_for_pipe_byte(read_fd) == b"acquired"
        os.waitpid(pid, 0)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_shared_worker_blocks_exclusive_inspection_until_shared_lease_releases(
    root: Path,
) -> None:
    root.mkdir(parents=True)
    read_fd, write_fd = os.pipe()
    try:
        with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_SHARED) as held:
            with _forking():
                pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                _child_waits_for_parent_lock(root, fcntl.LOCK_EX, held._parent_fd, write_fd)
            os.close(write_fd)
            write_fd = -1
            assert _wait_for_pipe_byte(read_fd, 0.2) == b""
        assert _wait_for_pipe_byte(read_fd) == b"acquired"
        os.waitpid(pid, 0)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_directory_locking_fails_closed_without_a_fallback(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(_fd: int, _operation: int) -> None:
        raise OSError(errno.ENOTSUP, "directory locks unavailable")

    parent_before = sorted(entry.name for entry in root.parent.iterdir())
    monkeypatch.setattr(resume.fcntl, "flock", unsupported)
    with pytest.raises(resume.ResumeLockError):
        with resume.reconciliation_lock(root):
            pass
    assert not root.exists()
    assert sorted(entry.name for entry in root.parent.iterdir()) == parent_before


def test_lease_rejects_wrong_root_and_inactive_leases(root: Path, tmp_path: Path) -> None:
    root.mkdir(parents=True)
    other = tmp_path / "other-work-units"
    with resume.reconciliation_lock(root) as held:
        with pytest.raises(resume.ResumeLockError):
            held._assert_usable(other, resume.LOCK_MODE_EXCLUSIVE)
    with pytest.raises(resume.ResumeLockError):
        held._assert_usable(root, resume.LOCK_MODE_EXCLUSIVE)


def test_lease_inherited_across_fork_is_rejected(root: Path) -> None:
    root.mkdir(parents=True)
    read_fd, write_fd = os.pipe()
    try:
        with resume.reconciliation_lock(root) as held:
            with _forking():
                pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                try:
                    held._assert_usable(root, resume.LOCK_MODE_EXCLUSIVE)
                except resume.ResumeLockError:
                    os.write(write_fd, b"rejected")
                else:  # pragma: no cover - assertion failure in child
                    os.write(write_fd, b"accepted")
                os._exit(0)
            os.close(write_fd)
            write_fd = -1
            assert _wait_for_pipe_byte(read_fd) == b"rejected"
            os.waitpid(pid, 0)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_public_repair_acquires_its_own_exclusive_global_lease(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    calls: list[tuple[Path | str | None, str]] = []
    original = resume.reconciliation_lock

    @contextlib.contextmanager
    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs.get("root", args[0] if args else None), kwargs.get("mode", resume.LOCK_MODE_EXCLUSIVE)))
        with original(*args, **kwargs) as lease:
            yield lease

    monkeypatch.setattr(resume, "reconciliation_lock", spy)
    resume.repair_work_unit(context, work_unit_id, root=root)
    assert calls == [(root, resume.LOCK_MODE_EXCLUSIVE)]


def test_batch_repair_reuses_one_exclusive_lease_and_does_not_nest_global_locks(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _recoverable_unit(context, root, status=bler_contract.STATUS_FAILED, index=0)
    _recoverable_unit(context, root, index=1)
    calls: list[str] = []
    original = resume.reconciliation_lock

    @contextlib.contextmanager
    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("mode", resume.LOCK_MODE_EXCLUSIVE))
        with original(*args, **kwargs) as lease:
            yield lease

    monkeypatch.setattr(resume, "reconciliation_lock", spy)
    report = resume.inspect_runtime_root(
        context,
        root=root,
        repair_mode=resume.REPAIR_MODE_REPAIR_RECOVERABLE,
    )
    assert len(report["repairs"]) == 2
    assert calls == [resume.LOCK_MODE_EXCLUSIVE]


def test_read_only_inspection_preserves_every_existing_path_and_byte(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    _recoverable_unit(context, root)
    lock_dir = root / units.LOCK_DIRECTORY_NAME
    lock_dir.mkdir()
    (lock_dir / f"{resume.work_unit_digest(_unit(context))}{units.LOCK_FILENAME_SUFFIX}").write_bytes(b"lock")

    def snapshot() -> tuple[tuple[str, str, bytes | None], ...]:
        entries: list[tuple[str, str, bytes | None]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                entries.append((relative, "directory", None))
            else:
                entries.append((relative, "file", path.read_bytes()))
        return tuple(entries)

    before = snapshot()
    resume.inspect_runtime_root(context, root=root)
    assert snapshot() == before


def test_exclusive_lock_excludes_a_real_child_process(root: Path, tmp_path: Path) -> None:
    root.mkdir(parents=True)
    done = tmp_path / "verdict"
    with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            _child_tries_parent_lock(root, fcntl.LOCK_EX, done)
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

            fd = os.open(
                str(root.parent),
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
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

            fd = os.open(
                str(root.parent),
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            except OSError:
                done.write_bytes(b"blocked")
                os._exit(0)
            done.write_bytes(b"shared")
            os._exit(0)
        os.waitpid(pid, 0)
        assert done.read_bytes() == b"blocked"


def test_hard_process_exit_releases_the_parent_directory_lock(root: Path, tmp_path: Path) -> None:
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
        assert held.active is True


def test_lock_is_released_on_exception(root: Path) -> None:
    root.mkdir(parents=True)
    with pytest.raises(ValueError):
        with resume.reconciliation_lock(root):
            raise ValueError("boom")
    with resume.reconciliation_lock(root) as held:
        assert held.active is True


def test_lock_never_creates_the_production_runtime_root() -> None:
    production = resume.DEFAULT_WORK_UNIT_ROOT
    existed = production.exists()
    with resume.reconciliation_lock() as held:
        if not existed:
            assert held.root_present is False
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


def test_bounded_smoke_inspection_requires_an_explicit_nonproduction_root(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    tmp_path: Path,
) -> None:
    for candidate in (None, resume.DEFAULT_WORK_UNIT_ROOT):
        with pytest.raises(resume.ResumeCensusError):
            resume.inspect_runtime_root(
                context,
                root=candidate,
                scan_mode=resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION,
            )

    production = resume.DEFAULT_WORK_UNIT_ROOT
    alias_parent = tmp_path / "production-parent-alias"
    alias_parent.symlink_to(production.parent, target_is_directory=True)
    alias = alias_parent / production.name
    with pytest.raises(resume.ResumeCensusError):
        resume.inspect_runtime_root(
            context,
            root=alias,
            scan_mode=resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION,
        )

    # A normal isolated root is accepted and remains absent after inspection.
    report = resume.inspect_runtime_root(
        context,
        root=root,
        scan_mode=resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION,
    )
    assert report["root_present"] is False
    assert not root.exists()


def test_bounded_smoke_repair_also_rejects_the_production_root(
    context: resume.AuthenticatedResumeContext,
) -> None:
    with pytest.raises(resume.ResumeCensusError):
        resume.repair_work_unit(
            context,
            _unit(context),
            scan_mode=resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION,
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


# ---------------------------------------------------------------------------
# B3.2 — request/result chain validation and closed classification
#
# Every fixture below publishes real canonical bytes into an isolated
# temporary root.  Nothing here runs a simulation: the "results" are assembled
# from authoritative counts by the frozen B1C builder, exactly as a merge
# validator would later read them.
# ---------------------------------------------------------------------------


def _write_canonical(path: Path, payload: dict[str, Any]) -> str:
    raw = bler_contract.canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return units.sha256_bytes(raw)


def _publish_state(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    state: dict[str, Any],
) -> str:
    path = resume.state_path(context, state["identity"]["work_unit_id"], root=root)
    return _write_canonical(path, state)


def _publish_request(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    attempt: int,
) -> tuple[dict[str, Any], str]:
    request = bler_contract.build_full_strength_request(work_unit_id)
    digest = _write_canonical(
        resume.request_path(context, work_unit_id, attempt, root=root), request
    )
    return request, digest


def _publish_result(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    attempt: int,
    request: dict[str, Any],
    *,
    status: str = bler_contract.STATUS_COMPLETE,
    trials_completed: int | None = None,
    bit_errors: int = 5,
    block_errors: int = 1,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[dict[str, Any], str]:
    if trials_completed is None:
        trials_completed = (
            request["trials_requested"] if status == bler_contract.STATUS_COMPLETE else 17
        )
    if status != bler_contract.STATUS_COMPLETE:
        bit_errors = 0
        block_errors = 0
    result = bler_contract.build_work_unit_result(
        request=request,
        status=status,
        trials_completed=trials_completed,
        bit_errors=bit_errors,
        block_errors=block_errors,
        execution_metadata={
            "wall_time_s": None,
            "hostname": None,
            "device": None,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "attempt": attempt,
        },
    )
    digest = _write_canonical(
        resume.result_path(context, work_unit_id, attempt, root=root), result
    )
    return result, digest


def _classify_one(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    *,
    scan_mode: str = resume.SCAN_MODE_PRODUCTION_MERGE,
) -> dict[str, Any]:
    census = resume.census_runtime_root(context, root=root)
    return resume.classify_work_unit(
        context, work_unit_id, census, root=root, scan_mode=scan_mode
    )


def test_an_absent_unit_classifies_as_absent_and_proposes_attempt_one(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    record = _classify_one(context, root, _unit(context))
    assert record["classification"] == "absent"
    assert record["proposed_attempt"] == 1
    assert record["state_status"] is None
    assert record["required_coverage_contribution"] == 0
    assert record["repairable"] is False


def test_an_artifact_without_its_state_is_a_contradiction_not_absent(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_request(context, root, work_unit_id, 1)
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_a_clean_claim_alone_is_claimed_unbound(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id, attempt=3))
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "claimed_unbound"
    assert record["attempt"] == 3
    assert record["proposed_attempt"] == 4
    assert record["request_sha256"] is None
    assert record["repairable"] is False


def test_a_clean_claim_with_its_request_is_claimed_request_published(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id, attempt=2))
    _request, digest = _publish_request(context, root, work_unit_id, 2)
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "claimed_request_published"
    # The request is history, not a state binding: the state stays unbound and
    # the unit is remaining work at exactly the next attempt.
    assert record["request_sha256"] == digest
    assert record["state_status"] == units.STATUS_CLAIMED
    assert record["proposed_attempt"] == 3
    assert record["repairable"] is False
    assert record["required_coverage_contribution"] == 0


def test_a_clean_claim_with_a_failed_result_is_recoverable_failed_result(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    request, _digest = _publish_request(context, root, work_unit_id, 1)
    _publish_result(
        context, root, work_unit_id, 1, request, status=bler_contract.STATUS_FAILED
    )
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "recoverable_failed_result"
    assert record["repairable"] is True
    assert record["proposed_attempt"] is None
    assert record["required_coverage_contribution"] == 0
    assert record["trials_completed"] == 17


def test_a_clean_claim_with_a_complete_result_is_recoverable_complete_result(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    request, _digest = _publish_request(context, root, work_unit_id, 1)
    _publish_result(context, root, work_unit_id, 1, request)
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "recoverable_complete_result"
    assert record["repairable"] is True
    assert record["proposed_attempt"] is None
    assert record["trials_completed"] == 5000


def test_a_terminal_result_linked_unit_is_completed_full_strength(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, request_sha = _publish_request(context, root, work_unit_id, 1)
    _result, result_sha = _publish_result(context, root, work_unit_id, 1, request)
    linked = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_RESULT_LINKED,
        request_sha256=request_sha,
        result_path=resume.logical_result_path(context, work_unit_id, 1),
        result_sha256=result_sha,
        scientific_execution_performed=True,
        trials_completed=request["trials_requested"],
    )
    _publish_state(context, root, linked)
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "completed_full_strength"
    assert record["required_coverage_contribution"] == 1
    assert record["proposed_attempt"] is None
    assert record["result_sha256"] == result_sha
    assert record["test_split_access"] == 0


def test_a_failed_state_is_failed_retryable_and_contributes_zero(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, request_sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(
        context, root, work_unit_id, 1, request, status=bler_contract.STATUS_FAILED
    )
    failed = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_FAILED,
        request_sha256=request_sha,
        scientific_execution_performed=True,
        trials_completed=17,
    )
    _publish_state(context, root, failed)
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "failed_retryable"
    assert record["proposed_attempt"] == 2
    assert record["required_coverage_contribution"] == 0
    assert record["repairable"] is False


def test_a_failed_state_without_a_result_reports_its_persisted_trial_count(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    failed = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_FAILED,
        scientific_execution_performed=True,
        trials_completed=23,
    )
    _publish_state(context, root, failed)
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "failed_retryable"
    assert record["trials_completed"] == 23
    assert record["required_coverage_contribution"] == 0


def test_result_linked_state_trials_must_equal_the_exact_result_count(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, request_sha = _publish_request(context, root, work_unit_id, 1)
    _result, result_sha = _publish_result(context, root, work_unit_id, 1, request)
    linked = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_RESULT_LINKED,
        request_sha256=request_sha,
        result_path=resume.logical_result_path(context, work_unit_id, 1),
        result_sha256=result_sha,
        scientific_execution_performed=True,
        trials_completed=4999,
    )
    _publish_state(context, root, linked)
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_incomplete_final_result_is_a_hold_not_retryable_evidence(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    request, _request_sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(
        context,
        root,
        work_unit_id,
        1,
        request,
        status=bler_contract.STATUS_INCOMPLETE,
        trials_completed=17,
    )
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_a_state_request_digest_that_does_not_reproduce_holds(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(
        context, root, work_unit_id, 1, request, status=bler_contract.STATUS_FAILED
    )
    failed = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_FAILED,
        request_sha256="d" * 64,
        scientific_execution_performed=True,
        trials_completed=17,
    )
    _publish_state(context, root, failed)
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_a_result_without_its_exact_request_holds(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    request = bler_contract.build_full_strength_request(work_unit_id)
    _publish_result(context, root, work_unit_id, 1, request)
    resume.request_path(context, work_unit_id, 1, root=root).unlink(missing_ok=True)
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_a_result_bound_to_another_request_holds(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    other_id = _unit(context, 1)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    _publish_request(context, root, work_unit_id, 1)
    foreign = bler_contract.build_full_strength_request(other_id)
    _publish_result(context, root, work_unit_id, 1, foreign)
    with pytest.raises(resume.ResumeHoldError):
        _classify_one(context, root, work_unit_id)


def test_an_artifact_beyond_the_state_attempt_holds(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id, attempt=1))
    _publish_request(context, root, work_unit_id, 2)
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_an_older_complete_merge_eligible_result_holds_when_state_advanced(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(context, root, work_unit_id, 1, request)
    _publish_state(context, root, _clean_claim(context, work_unit_id, attempt=2))
    with pytest.raises(resume.ResumeContradictionError):
        _classify_one(context, root, work_unit_id)


def test_an_older_failed_result_is_history_and_contributes_zero(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(
        context, root, work_unit_id, 1, request, status=bler_contract.STATUS_FAILED
    )
    _publish_state(context, root, _clean_claim(context, work_unit_id, attempt=2))
    record = _classify_one(context, root, work_unit_id)
    assert record["classification"] == "claimed_unbound"
    assert record["proposed_attempt"] == 3
    assert record["required_coverage_contribution"] == 0


def test_request_bytes_are_identical_across_attempts_on_disk(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _first, first_sha = _publish_request(context, root, work_unit_id, 1)
    _second, second_sha = _publish_request(context, root, work_unit_id, 2)
    assert first_sha == second_sha
    assert (
        resume.request_path(context, work_unit_id, 1, root=root).read_bytes()
        == resume.request_path(context, work_unit_id, 2, root=root).read_bytes()
    )


def test_noncanonical_request_bytes_are_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request = bler_contract.build_full_strength_request(work_unit_id)
    path = resume.request_path(context, work_unit_id, 1, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Semantically identical, byte-wise different: indented JSON.
    path.write_bytes(json.dumps(request, indent=2, sort_keys=True).encode("utf-8"))
    with pytest.raises(resume.ResumeChainError):
        resume.validate_request_file(context, work_unit_id, 1, root=root)


def test_a_full_strength_result_at_the_wrong_trial_count_is_rejected(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    """5000 is the only complete full-strength count; 4999 is not partial credit."""

    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    result, _result_sha = _publish_result(context, root, work_unit_id, 1, request)
    forged = json.loads(json.dumps(result))
    forged["measurement"]["trials_completed"] = 4999
    _write_canonical(resume.result_path(context, work_unit_id, 1, root=root), forged)
    request_record = resume.validate_request_file(context, work_unit_id, 1, root=root)
    with pytest.raises(resume.ResumeChainError):
        resume.validate_result_file(
            context, work_unit_id, 1, root=root, request_record=request_record
        )


def test_a_bounded_smoke_result_is_never_production_required_coverage(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    result, _result_sha = _publish_result(context, root, work_unit_id, 1, request)
    request_record = resume.validate_request_file(context, work_unit_id, 1, root=root)
    result_record = resume.validate_result_file(
        context, work_unit_id, 1, root=root, request_record=request_record
    )
    assert (
        resume.is_full_strength_merge_candidate(context, result_record, request_record)
        is True
    )
    # Flipping the disposition flags alone must not buy coverage: the frozen
    # contract binds them to the exact counts and execution class.
    forged_result = json.loads(json.dumps(result_record["result"]))
    forged_result["disposition"]["merge_eligible"] = False
    forged_result["disposition"]["required_coverage_contribution"] = 0
    forged_record = dict(result_record, result=forged_result)
    assert (
        resume.is_full_strength_merge_candidate(context, forged_record, request_record)
        is False
    )


def test_result_shard_metadata_must_match_the_current_state(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(context, root, work_unit_id, 1, request, shard_index=0, shard_count=4)
    request_record = resume.validate_request_file(context, work_unit_id, 1, root=root)
    with pytest.raises(resume.ResumeContradictionError):
        resume.validate_result_file(
            context,
            work_unit_id,
            1,
            root=root,
            request_record=request_record,
            shard_index=0,
            shard_count=1,
        )


def test_a_malformed_state_never_degrades_to_absent(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    path = resume.state_path(context, work_unit_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"schema_version":2}')
    with pytest.raises(resume.ResumeHoldError):
        resume.read_unit_state_snapshot(context, work_unit_id, root=root)


def test_classification_is_deterministic_and_covers_every_required_unit(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _unit(context)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    _publish_request(context, root, work_unit_id, 1)
    census = resume.census_runtime_root(context, root=root)
    first = resume.classify_runtime_root(context, census, root=root)
    second = resume.classify_runtime_root(context, census, root=root)
    assert len(first) == context.required_work_unit_count
    assert bler_contract.canonical_json(list(first)) == bler_contract.canonical_json(list(second))
    assert [record["work_unit_id"] for record in first] == list(context.ordered_work_unit_ids)
    assert first[0]["classification"] == "claimed_request_published"
    assert {record["classification"] for record in first[1:]} == {"absent"}
    for record in first:
        assert record["classification"] in resume.CLASSIFICATIONS
        assert record["test_split_access"] == 0


# ---------------------------------------------------------------------------
# B3.3 — global reconciliation locking and explicit recovery (§16)
# ---------------------------------------------------------------------------


def _recoverable_unit(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    *,
    status: str = bler_contract.STATUS_COMPLETE,
    index: int = 0,
) -> str:
    work_unit_id = _unit(context, index)
    _publish_state(context, root, _clean_claim(context, work_unit_id))
    request, _sha = _publish_request(context, root, work_unit_id, 1)
    _publish_result(context, root, work_unit_id, 1, request, status=status)
    return work_unit_id


def test_inspection_is_read_only_by_default(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    before = resume.state_path(context, work_unit_id, root=root).read_bytes()
    report = resume.inspect_runtime_root(context, root=root)
    assert report["repair_mode"] == resume.REPAIR_MODE_READ_ONLY
    assert report["repairs"] == []
    assert report["classifications"][0]["classification"] == "recoverable_complete_result"
    assert resume.state_path(context, work_unit_id, root=root).read_bytes() == before


def test_repair_transitions_a_failed_result_directly_to_failed(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _recoverable_unit(context, root, status=bler_contract.STATUS_FAILED)
    outcome = resume.repair_work_unit(context, work_unit_id, root=root)
    assert outcome["repaired"] is True
    assert outcome["from_classification"] == "recoverable_failed_result"
    assert outcome["classification"] == "failed_retryable"

    state = resume.read_unit_state_snapshot(context, work_unit_id, root=root)
    identity = state["identity"]
    assert identity["status"] == units.STATUS_FAILED
    assert identity["request_sha256"] is not None
    # A failed state deliberately carries no result reference.
    assert identity["result_path"] is None
    assert identity["result_sha256"] is None
    assert identity["scientific_execution_performed"] is True
    assert identity["trials_completed"] == 17
    assert identity["test_split_access"] == 0
    assert identity["attempt"] == 1


def test_repair_transitions_a_complete_result_directly_to_result_linked(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    outcome = resume.repair_work_unit(context, work_unit_id, root=root)
    assert outcome["repaired"] is True
    assert outcome["classification"] == "completed_full_strength"

    state = resume.read_unit_state_snapshot(context, work_unit_id, root=root)
    identity = state["identity"]
    assert identity["status"] == units.STATUS_RESULT_LINKED
    assert identity["result_path"] == resume.logical_result_path(context, work_unit_id, 1)
    assert identity["result_sha256"] is not None
    assert identity["trials_completed"] == 5000
    assert identity["test_split_access"] == 0


def test_uncertain_repair_publication_requires_exact_installed_bytes(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    original = units.replace_unit_state

    def publish_then_raise(*args: Any, **kwargs: Any) -> Any:
        installed = original(*args, **kwargs)
        raise RuntimeError("directory fsync acknowledgement was lost")

    monkeypatch.setattr(units, "replace_unit_state", publish_then_raise)
    outcome = resume.repair_work_unit(context, work_unit_id, root=root)
    assert outcome["classification"] == "completed_full_strength"
    assert outcome["repaired"] is True


def test_uncertain_repair_publication_without_exact_bytes_is_a_hold(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_unit_id = _recoverable_unit(context, root)

    def raise_before_publication(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("publication outcome is uncertain")

    monkeypatch.setattr(units, "replace_unit_state", raise_before_publication)
    with pytest.raises(resume.ResumeRepairError):
        resume.repair_work_unit(context, work_unit_id, root=root)


def test_repair_never_creates_a_request_bound_claimed_state(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    """The corrected matrix has no intermediate bound claim to pass through."""

    work_unit_id = _recoverable_unit(context, root)
    resume.repair_work_unit(context, work_unit_id, root=root)
    identity = resume.read_unit_state_snapshot(context, work_unit_id, root=root)["identity"]
    assert not (identity["status"] == units.STATUS_CLAIMED and identity["request_sha256"])


def test_repair_is_idempotent(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    resume.repair_work_unit(context, work_unit_id, root=root)
    settled = resume.state_path(context, work_unit_id, root=root).read_bytes()
    second = resume.repair_work_unit(context, work_unit_id, root=root)
    assert second["repaired"] is False
    assert second["classification"] == "completed_full_strength"
    assert resume.state_path(context, work_unit_id, root=root).read_bytes() == settled


@pytest.mark.parametrize(
    "builder",
    [
        "claimed_unbound",
        "claimed_request_published",
        "failed_retryable",
        "absent",
    ],
)
def test_repair_never_modifies_a_non_repairable_class(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    builder: str,
) -> None:
    work_unit_id = _unit(context)
    if builder == "absent":
        root.mkdir(parents=True, exist_ok=True)
    elif builder == "failed_retryable":
        request, request_sha = _publish_request(context, root, work_unit_id, 1)
        _publish_result(
            context, root, work_unit_id, 1, request, status=bler_contract.STATUS_FAILED
        )
        _publish_state(
            context,
            root,
            units.build_unit_state(
                context.state_context,
                work_unit_id,
                _plan(context),
                status=units.STATUS_FAILED,
                request_sha256=request_sha,
                scientific_execution_performed=True,
                trials_completed=17,
            ),
        )
    else:
        _publish_state(context, root, _clean_claim(context, work_unit_id))
        if builder == "claimed_request_published":
            _publish_request(context, root, work_unit_id, 1)

    state_file = resume.state_path(context, work_unit_id, root=root)
    before = state_file.read_bytes() if state_file.exists() else None
    report = resume.inspect_runtime_root(
        context, root=root, repair_mode=resume.REPAIR_MODE_REPAIR_RECOVERABLE
    )
    assert report["repairs"] == []
    assert report["classifications"][0]["classification"] == builder
    after = state_file.read_bytes() if state_file.exists() else None
    assert after == before


def test_explicit_repair_mode_repairs_only_the_two_recoverable_classes(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    failed_id = _recoverable_unit(context, root, status=bler_contract.STATUS_FAILED, index=0)
    complete_id = _recoverable_unit(context, root, index=1)
    untouched_id = _unit(context, 2)
    _publish_state(context, root, _clean_claim(context, untouched_id))
    untouched_before = resume.state_path(context, untouched_id, root=root).read_bytes()

    report = resume.inspect_runtime_root(
        context, root=root, repair_mode=resume.REPAIR_MODE_REPAIR_RECOVERABLE
    )
    repaired = {entry["work_unit_id"]: entry for entry in report["repairs"]}
    assert set(repaired) == {failed_id, complete_id}
    assert repaired[failed_id]["classification"] == "failed_retryable"
    assert repaired[complete_id]["classification"] == "completed_full_strength"
    assert (
        resume.state_path(context, untouched_id, root=root).read_bytes() == untouched_before
    )
    # The post-repair rescan is what the report returns.
    by_id = {entry["work_unit_id"]: entry for entry in report["classifications"]}
    assert by_id[failed_id]["classification"] == "failed_retryable"
    assert by_id[complete_id]["classification"] == "completed_full_strength"
    assert by_id[untouched_id]["classification"] == "claimed_unbound"


def test_a_stale_predecessor_loses_the_compare_and_swap(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    work_unit_id = _recoverable_unit(context, root)
    state = resume.read_unit_state_snapshot(context, work_unit_id, root=root)
    stale_sha = state["state_sha256"]
    resume.repair_work_unit(context, work_unit_id, root=root)

    # A writer still holding the pre-repair digest must lose, not overwrite.
    proposed = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_FAILED,
        request_sha256="e" * 64,
        scientific_execution_performed=True,
        trials_completed=3,
    )
    with pytest.raises(units.StateConflictError):
        units.replace_unit_state(
            context.state_context,
            resume.state_path(context, work_unit_id, root=root),
            proposed,
            stale_sha,
            root=root,
        )


def test_repair_refuses_an_unknown_repair_mode(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(resume.ResumeRepairError):
        resume.inspect_runtime_root(context, root=root, repair_mode="force")


def test_inspection_waits_for_the_exclusive_reconciliation_lock(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    """A reconciliation pass must not run concurrently with another holder."""

    _recoverable_unit(context, root)
    # Outside the runtime root: the census rejects any undefined entry inside it.
    started = root.parent / "child-started"
    hold_seconds = 1.0
    with _forking():
        pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        try:
            with resume.reconciliation_lock(root, mode=resume.LOCK_MODE_EXCLUSIVE):
                started.write_bytes(b"1")
                time.sleep(hold_seconds)
        finally:
            os._exit(0)
    try:
        deadline = time.monotonic() + 5.0
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        began = time.monotonic()
        report = resume.inspect_runtime_root(context, root=root)
        waited = time.monotonic() - began
    finally:
        os.waitpid(pid, 0)
    # It blocked until the child released rather than reconciling alongside it.
    assert waited >= hold_seconds * 0.5
    assert report["classifications"][0]["classification"] == "recoverable_complete_result"


def test_inspection_never_creates_the_production_runtime_root(
    context: resume.AuthenticatedResumeContext,
) -> None:
    production = resume.DEFAULT_WORK_UNIT_ROOT
    existed = production.exists()
    report = resume.inspect_runtime_root(context)
    if not existed:
        assert report["root_present"] is False
        assert report["census"]["root_present"] is False
        assert {record["classification"] for record in report["classifications"]} == {"absent"}
    assert production.exists() == existed


# ---------------------------------------------------------------------------
# B3.4 — deterministic resume plans and merge reports (§17–§18)
# ---------------------------------------------------------------------------


def _stub_registered_b3_binding(
    context: resume.AuthenticatedResumeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a candidate binding while the real B3 artifact is not registered yet."""

    binding = {
        "bler_resume_contract_id": "g8resume-" + "a" * 64,
        "bler_resume_contract_sha256": "b" * 64,
        "bler_state_contract_id": resume.EXPECTED_B2C_CONTRACT_ID,
        "bler_state_contract_sha256": resume.EXPECTED_B2C_CONTRACT_SHA256,
        "bler_tooling_contract_id": resume.EXPECTED_B1C_CONTRACT_ID,
        "bler_tooling_contract_sha256": resume.EXPECTED_B1C_CONTRACT_SHA256,
        "campaign_id": resume.EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": resume.EXPECTED_CAMPAIGN_MANIFEST_SHA256,
        "required_bler_artifact_sha256": resume.EXPECTED_REQUIRED_IDENTITIES_SHA256,
        "selection_policy_sha256": resume.EXPECTED_SELECTION_POLICY_SHA256,
        "request_schema_version": resume.REQUEST_SCHEMA_VERSION,
        "result_schema_version": resume.RESULT_SCHEMA_VERSION,
        "unit_state_schema_version": resume.UNIT_STATE_SCHEMA_VERSION,
    }
    monkeypatch.setattr(resume, "_resume_operation_bindings", lambda _context: dict(binding))


def test_plans_and_merge_reports_require_a_registered_b3_contract(
    context: resume.AuthenticatedResumeContext,
    root: Path,
) -> None:
    with pytest.raises(resume.ResumeContractAuthenticationError):
        resume.build_resume_plan(context, root=root)
    with pytest.raises(resume.ResumeContractAuthenticationError):
        resume.build_merge_report(context, root=root)


def test_empty_resume_plan_is_byte_deterministic_and_shard_ordered(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_registered_b3_binding(context, monkeypatch)
    first = resume.build_resume_plan(context, root=root, shard_count=3, shard_index=1)
    second = resume.build_resume_plan(context, root=root, shard_count=3, shard_index=1)
    assert bler_contract.canonical_json(first) == bler_contract.canonical_json(second)
    assert resume.resume_plan_digest(first) == first["plan_digest"]
    assert first["schema_version"] == 1
    assert first["artifact_role"] == "g8_bler_resume_plan"
    assert first["logical_root"] == resume.WORK_UNIT_ROOT_LOGICAL_PREFIX
    assert first["test_split_access"] == 0
    assert first["shard_plan_digest"] == units.build_shard_plan(
        context.state_context, 3, 1
    )["plan_digest"]
    expected_ids = list(context.ordered_work_unit_ids)[1::3]
    assert first["assigned_work_unit_ids"] == expected_ids
    assert [record["work_unit_id"] for record in first["assigned_unit_records"]] == expected_ids
    assert first["completed_work_unit_ids"] == []
    assert first["recoverable_work_unit_ids"] == []
    assert first["remaining_work_unit_ids"] == expected_ids
    assert all(entry["proposed_attempt"] == 1 for entry in first["proposed_attempts"])
    assert str(root) not in bler_contract.canonical_json(first).decode("ascii")
    forbidden = ("hostname", "process_id", "timestamp", "mtime", "inode")
    assert not any(name in first for name in forbidden)


def test_changing_shard_layout_changes_membership_only(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_registered_b3_binding(context, monkeypatch)
    first = resume.build_resume_plan(context, root=root, shard_count=2, shard_index=0)
    second = resume.build_resume_plan(context, root=root, shard_count=5, shard_index=0)
    first_records = {entry["work_unit_id"]: entry for entry in first["assigned_unit_records"]}
    second_records = {entry["work_unit_id"]: entry for entry in second["assigned_unit_records"]}
    overlap = set(first_records) & set(second_records)
    assert overlap
    for work_unit_id in overlap:
        assert first_records[work_unit_id] == second_records[work_unit_id]
    assert set(first["assigned_work_unit_ids"]) != set(second["assigned_work_unit_ids"])
    assert all(entry["proposed_attempt"] == 1 for entry in second["proposed_attempts"])


def test_partial_merge_report_is_explicitly_not_complete(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_registered_b3_binding(context, monkeypatch)
    work_unit_id = _unit(context)
    request, request_sha = _publish_request(context, root, work_unit_id, 1)
    _result, result_sha = _publish_result(context, root, work_unit_id, 1, request)
    linked = units.build_unit_state(
        context.state_context,
        work_unit_id,
        _plan(context),
        status=units.STATUS_RESULT_LINKED,
        request_sha256=request_sha,
        result_path=resume.logical_result_path(context, work_unit_id, 1),
        result_sha256=result_sha,
        scientific_execution_performed=True,
        trials_completed=request["trials_requested"],
    )
    _publish_state(context, root, linked)
    report = resume.build_merge_report(context, root=root)
    assert resume.merge_report_digest(report) == report["report_digest"]
    assert report["required_work_unit_ids"] == list(context.ordered_work_unit_ids)
    assert report["validated_complete_work_unit_ids"] == [work_unit_id]
    assert report["missing_work_unit_ids"] == list(context.ordered_work_unit_ids[1:])
    assert report["valid_request_count"] == 1
    assert report["valid_result_count"] == 1
    assert report["valid_complete_result_count"] == 1
    assert report["exact_coverage_count"] == 1
    assert report["total_required_coverage_contribution"] == 1
    assert report["coverage_complete"] is False
    assert report["merge_ready"] is False
    assert report["duplicate_count"] == 0
    assert report["unknown_count"] == 0
    assert report["test_split_access"] == 0


def test_plan_and_merge_scan_under_one_exclusive_lease(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_registered_b3_binding(context, monkeypatch)
    original = resume.reconciliation_lock
    calls: list[str] = []

    @contextlib.contextmanager
    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("mode", resume.LOCK_MODE_EXCLUSIVE))
        with original(*args, **kwargs) as lease:
            yield lease

    monkeypatch.setattr(resume, "reconciliation_lock", spy)
    resume.build_resume_plan(context, root=root)
    resume.build_merge_report(context, root=root)
    assert calls == [resume.LOCK_MODE_EXCLUSIVE, resume.LOCK_MODE_EXCLUSIVE]
