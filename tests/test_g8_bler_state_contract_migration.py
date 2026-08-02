"""B2C state-contract migration and recovery tests.

Every test operates on an isolated copy of the artifact/state pair, so the
committed pair is never mutated.  The migration replaces exactly one
already-registered produced-artifact binding; it never appends a duplicate
path, never touches a scientific counter, and never performs a per-unit
migration, because no unit-state file has ever existed.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import migrate_g8_bler_state_contract as migration
from baseline import g8_bler_work_units as units
from baseline.g8_campaign import rendered_json, sha256_bytes


REPO = Path(__file__).parents[1]
LIVE_CONTRACT = REPO / "results/baseline/g8/bler_state_contract.json"
LIVE_STATE = REPO / "results/baseline/g8/campaign_state.json"

# The B2 commit is immutable, pushed history; its blob is the only source of
# the exact superseded artifact bytes, and fabricating them is impossible.
SUPERSEDED_COMMIT = "6193dfda2bd2cc91e090eb3cfd57d46a3a0a9726"
SUPERSEDED_BLOB_PATH = "results/baseline/g8/bler_state_contract.json"
SUPERSEDED_STATE_BLOB_PATH = "results/baseline/g8/campaign_state.json"


def _blob(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SUPERSEDED_COMMIT}:{path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


@pytest.fixture(scope="module")
def old_contract_bytes() -> bytes:
    body = _blob(SUPERSEDED_BLOB_PATH)
    assert sha256_bytes(body) == units.SUPERSEDED_STATE_CONTRACT_SHA256
    assert len(body) == units.SUPERSEDED_STATE_CONTRACT_BYTES
    return body


@pytest.fixture(scope="module")
def old_state_bytes() -> bytes:
    return _blob(SUPERSEDED_STATE_BLOB_PATH)


@pytest.fixture(scope="module")
def new_contract_bytes() -> bytes:
    return LIVE_CONTRACT.read_bytes()


@pytest.fixture(scope="module")
def new_state_bytes() -> bytes:
    # This migration fixture is the B2C-era pair.  The live campaign advances
    # its restart command and registers B3 and B4 artifacts after the frozen
    # B2C migration, so reconstruct the exact four-artifact tooling-open state
    # that the frozen migration utility is designed to recover.  The live
    # state itself remains authoritative and is never edited by this fixture.
    state = json.loads(LIVE_STATE.read_bytes())
    identity = state["identity"]
    identity["restart_command"] = units.B3_RESTART_COMMAND
    identity["produced_artifacts"] = [
        entry
        for entry in identity["produced_artifacts"]
        if entry["path"] not in {
            "results/baseline/g8/bler_resume_contract.json",
            "results/baseline/g8/bler_runner_contract.json",
        }
    ]
    return rendered_json(state)


def _pair(tmp_path: Path, contract: bytes, state: bytes) -> tuple[Path, Path]:
    contract_path = tmp_path / "bler_state_contract.json"
    state_path = tmp_path / "campaign_state.json"
    contract_path.write_bytes(contract)
    state_path.write_bytes(state)
    return contract_path, state_path


def _identity(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_bytes())["identity"]


def _binding(state_path: Path) -> dict[str, Any]:
    entries = [
        entry
        for entry in _identity(state_path)["produced_artifacts"]
        if entry["path"] == units.STATE_CONTRACT_REPO_RELATIVE_PATH
    ]
    assert len(entries) == 1, "exactly one state-contract binding must exist"
    return entries[0]


def _migrate(contract_path: Path, state_path: Path) -> dict[str, Any]:
    return migration.migrate(artifact_path=contract_path, state_path=state_path)


# ---------------------------------------------------------------------------
# The four artifact/state recovery pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contract_fixture,state_fixture,label",
    [
        ("old_contract_bytes", "old_state_bytes", "old artifact + old state"),
        ("new_contract_bytes", "old_state_bytes", "new artifact + old state"),
        ("new_contract_bytes", "new_state_bytes", "new artifact + new state"),
        ("old_contract_bytes", "new_state_bytes", "old artifact + new state"),
    ],
)
def test_every_artifact_state_pair_recovers(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    contract_fixture: str,
    state_fixture: str,
    label: str,
    new_contract_bytes: bytes,
) -> None:
    contract_path, state_path = _pair(
        tmp_path,
        request.getfixturevalue(contract_fixture),
        request.getfixturevalue(state_fixture),
    )
    result = _migrate(contract_path, state_path)
    assert contract_path.read_bytes() == new_contract_bytes, label
    assert result["contract_id"] != units.SUPERSEDED_STATE_CONTRACT_ID
    assert result["contract_sha256"] == sha256_bytes(new_contract_bytes)
    assert _binding(state_path) == {
        "path": units.STATE_CONTRACT_REPO_RELATIVE_PATH,
        "sha256": sha256_bytes(new_contract_bytes),
        "bytes": len(new_contract_bytes),
    }


def test_migration_is_idempotent(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)
    first = _migrate(contract_path, state_path)
    first_contract = contract_path.read_bytes()
    first_state = state_path.read_bytes()
    second = _migrate(contract_path, state_path)
    assert contract_path.read_bytes() == first_contract
    assert state_path.read_bytes() == first_state
    assert second["contract_id"] == first["contract_id"]
    assert second["state_sha256"] == first["state_sha256"]


# ---------------------------------------------------------------------------
# Interruption recovery
# ---------------------------------------------------------------------------


def test_interruption_after_artifact_replacement_is_recoverable(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
    new_contract_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)

    def interrupted(*_args: Any, **_kwargs: Any) -> str:
        raise KeyboardInterrupt("simulated interruption before the state half")

    monkeypatch.setattr(migration, "write_campaign_state_atomically", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _migrate(contract_path, state_path)

    # The artifact half landed; the state half did not.
    assert contract_path.read_bytes() == new_contract_bytes
    assert _binding(state_path)["sha256"] == units.SUPERSEDED_STATE_CONTRACT_SHA256

    monkeypatch.undo()
    _migrate(contract_path, state_path)
    assert _binding(state_path)["sha256"] == sha256_bytes(new_contract_bytes)


def test_interruption_before_artifact_replacement_leaves_the_old_pair(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)
    before_state = state_path.read_bytes()

    real_replace = migration.os.replace

    def interrupted(source: Any, target: Any, *args: Any, **kwargs: Any) -> None:
        if str(target) == str(contract_path):
            raise KeyboardInterrupt("simulated interruption before publication")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(migration.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _migrate(contract_path, state_path)
    monkeypatch.undo()

    assert contract_path.read_bytes() == old_contract_bytes
    assert state_path.read_bytes() == before_state
    assert not list(tmp_path.glob("*.partial"))
    _migrate(contract_path, state_path)


def test_interruption_after_campaign_state_replacement_is_a_completed_migration(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
    new_contract_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)
    real_writer = migration.write_campaign_state_atomically

    def write_then_die(path: Path, payload: dict[str, Any]) -> str:
        real_writer(path, payload)
        raise KeyboardInterrupt("simulated interruption after the state half")

    monkeypatch.setattr(migration, "write_campaign_state_atomically", write_then_die)
    with pytest.raises(KeyboardInterrupt):
        _migrate(contract_path, state_path)
    monkeypatch.undo()

    assert contract_path.read_bytes() == new_contract_bytes
    assert _binding(state_path)["sha256"] == sha256_bytes(new_contract_bytes)
    # Rerunning is a no-op that still verifies.
    _migrate(contract_path, state_path)


# ---------------------------------------------------------------------------
# Preservation and refusal
# ---------------------------------------------------------------------------


def test_migration_preserves_every_unrelated_field(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)
    before = json.loads(old_state_bytes)["identity"]
    _migrate(contract_path, state_path)
    after = _identity(state_path)

    for field in (
        "campaign_id",
        "campaign_manifest_sha256",
        "completed_work_unit_ids",
        "counters",
        "in_progress_work_unit_id",
        "phase",
        "stage",
        "restart_command",
        "seed_derivation_identity",
    ):
        assert after[field] == before[field], field
    assert after["campaign_id"] == units.EXPECTED_CAMPAIGN_ID
    assert after["completed_work_unit_ids"] == []
    assert after["in_progress_work_unit_id"] is None
    assert after["counters"] == {
        "validation_decoding": 0,
        "inference": 0,
        "training": 0,
        "test_access": 0,
    }
    assert after["restart_command"] == units.B3_RESTART_COMMAND

    unrelated_before = {
        entry["path"]: entry
        for entry in before["produced_artifacts"]
        if entry["path"] != units.STATE_CONTRACT_REPO_RELATIVE_PATH
    }
    unrelated_after = {
        entry["path"]: entry
        for entry in after["produced_artifacts"]
        if entry["path"] != units.STATE_CONTRACT_REPO_RELATIVE_PATH
    }
    assert unrelated_after == unrelated_before
    assert unrelated_after["results/baseline/g8/bler_tooling_contract.json"]["sha256"] == (
        units.EXPECTED_B1C_CONTRACT_SHA256
    )


def test_migration_replaces_rather_than_appends_a_binding(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
) -> None:
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, old_state_bytes)
    before = json.loads(old_state_bytes)["identity"]["produced_artifacts"]
    _migrate(contract_path, state_path)
    after = _identity(state_path)["produced_artifacts"]
    assert len(after) == len(before) == 4
    assert [entry["path"] for entry in after] == sorted(entry["path"] for entry in before)


def test_unknown_artifact_binding_is_refused(
    tmp_path: Path,
    old_state_bytes: bytes,
) -> None:
    contract_path, state_path = _pair(tmp_path, b'{"not": "a contract"}\n', old_state_bytes)
    with pytest.raises(migration.G8BlerStateMigrationError, match="neither the exact B2 nor B2C"):
        _migrate(contract_path, state_path)


def test_unknown_state_binding_is_refused(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
) -> None:
    payload = json.loads(old_state_bytes)
    for entry in payload["identity"]["produced_artifacts"]:
        if entry["path"] == units.STATE_CONTRACT_REPO_RELATIVE_PATH:
            entry["sha256"] = "0" * 64
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, rendered_json(payload))
    with pytest.raises(migration.G8BlerStateMigrationError, match="unknown state-contract binding"):
        _migrate(contract_path, state_path)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda i: i.__setitem__("completed_work_unit_ids", ["u"]), "no completed scientific"),
        (lambda i: i.__setitem__("in_progress_work_unit_id", "u"), "no in-progress scientific"),
        (lambda i: i["counters"].__setitem__("inference", 1), "counters to be zero"),
        (lambda i: i["counters"].__setitem__("test_access", 1), "counters to be zero"),
        (lambda i: i.__setitem__("stage", "smoke_open"), "G8_B/tooling_open"),
        (lambda i: i.__setitem__("restart_command", "rg -n other"), "exact B3 restart command"),
    ],
)
def test_migration_refuses_any_scientific_or_phase_drift(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
    mutation: Any,
    message: str,
) -> None:
    payload = copy.deepcopy(json.loads(old_state_bytes))
    mutation(payload["identity"])
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, rendered_json(payload))
    with pytest.raises(migration.G8BlerStateMigrationError, match=message):
        _migrate(contract_path, state_path)


def test_migration_refuses_a_changed_unrelated_artifact(
    tmp_path: Path,
    old_contract_bytes: bytes,
    old_state_bytes: bytes,
) -> None:
    payload = copy.deepcopy(json.loads(old_state_bytes))
    for entry in payload["identity"]["produced_artifacts"]:
        if entry["path"] == "results/baseline/g8/bler_tooling_contract.json":
            entry["sha256"] = "0" * 64
    contract_path, state_path = _pair(tmp_path, old_contract_bytes, rendered_json(payload))
    with pytest.raises(migration.G8BlerStateMigrationError, match="unrelated produced artifact"):
        _migrate(contract_path, state_path)


def test_migration_performs_no_per_unit_migration() -> None:
    """No unit-state file has ever existed, so none may be migrated."""

    source = (REPO / "tools/migrate_g8_bler_state_contract.py").read_text(encoding="utf-8")
    for forbidden in (
        "create_unit_state_exclusive",
        "replace_unit_state",
        "read_unit_state",
        "build_unit_state",
        "work_units/",
    ):
        assert forbidden not in source, forbidden
    assert not units.DEFAULT_WORK_UNIT_ROOT.exists()


def test_live_pair_authenticates_through_the_normal_loader() -> None:
    context = units.AuthenticatedUnitStateContext()
    payload = json.loads(LIVE_CONTRACT.read_bytes())
    assert context.state_contract_id == payload["contract_id"]
    assert context.state_contract_sha256 == sha256_bytes(LIVE_CONTRACT.read_bytes())
    assert payload["checkpoint"] == "B2C"
    assert payload["supersedes"]["contract_id"] == units.SUPERSEDED_STATE_CONTRACT_ID
