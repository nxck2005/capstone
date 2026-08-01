"""C5 recoverable B1C tooling-contract migration tests.

All pairs live in a temporary directory below the G8 artifact root.  They are
contract/state plumbing only: no runner, encoder, decoder, channel or data
path is imported or executed.
"""

from __future__ import annotations

import copy
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

import gen_g8_bler_tooling_contract as generator
import migrate_g8_bler_tooling_contract as migration
from baseline.g8_campaign import (
    REPO_ROOT,
    initial_campaign_state,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
)


G8_ROOT = REPO_ROOT / "results/baseline/g8"
STARTING_SHA = "1b5d81f323fac588ca57a0be504f3ffdfa714ce4"


def _old_artifact_bytes() -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{STARTING_SHA}:results/baseline/g8/bler_tooling_contract.json"],
        cwd=REPO_ROOT,
    )


def _new_artifact_bytes() -> bytes:
    return rendered_json(generator.build())


def _write_pair(
    directory: Path,
    *,
    artifact_bytes: bytes,
    state_binding_bytes: bytes,
    state_mutation: Any | None = None,
) -> tuple[Path, Path]:
    artifact = directory / "bler_tooling_contract.json"
    state_path = directory / "campaign_state.json"
    artifact.write_bytes(artifact_bytes)
    state = initial_campaign_state(stage="preflight_complete")
    state["identity"]["phase"] = "G8_B"
    state["identity"]["stage"] = "tooling_open"
    state["identity"]["restart_command"] = migration.B2_RESTART_COMMAND
    state["identity"]["produced_artifacts"].append(
        {
            "path": str(artifact.relative_to(REPO_ROOT)),
            "sha256": sha256_bytes(state_binding_bytes),
            "bytes": len(state_binding_bytes),
        }
    )
    state["identity"]["produced_artifacts"].sort(key=lambda entry: entry["path"])
    if state_mutation is not None:
        state_mutation(state)
    state_path.write_bytes(rendered_json(state))
    return artifact, state_path


def _run_pair(
    *,
    artifact_bytes: bytes,
    state_binding_bytes: bytes,
    state_mutation: Any | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw),
            artifact_bytes=artifact_bytes,
            state_binding_bytes=state_binding_bytes,
            state_mutation=state_mutation,
        )
        result = migration.migrate(artifact_path=artifact, state_path=state_path)
        assert artifact.read_bytes() == _new_artifact_bytes()
        final = load_campaign_state(state_path)
        assert final == result["state"]
        return {
            "artifact": artifact.read_bytes(),
            "state": state_path.read_bytes(),
            "result": result,
        }


def test_old_artifact_old_state_migrates() -> None:
    old = _old_artifact_bytes()
    result = _run_pair(artifact_bytes=old, state_binding_bytes=old)
    assert result["result"]["contract_id"] == generator.build()["contract_id"]


def test_new_artifact_old_state_repairs_state_binding() -> None:
    old = _old_artifact_bytes()
    new = _new_artifact_bytes()
    result = _run_pair(artifact_bytes=new, state_binding_bytes=old)
    assert result["result"]["contract_sha256"] == sha256_bytes(new)


def test_new_artifact_new_state_is_byte_idempotent() -> None:
    new = _new_artifact_bytes()
    result = _run_pair(artifact_bytes=new, state_binding_bytes=new)
    assert result["artifact"] == new


def test_old_artifact_new_state_repairs_artifact() -> None:
    old = _old_artifact_bytes()
    new = _new_artifact_bytes()
    result = _run_pair(artifact_bytes=old, state_binding_bytes=new)
    assert result["artifact"] == new


def test_unknown_current_artifact_fails() -> None:
    old = _old_artifact_bytes()
    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(Path(raw), artifact_bytes=old + b"x", state_binding_bytes=old)
        with pytest.raises(migration.G8BlerMigrationError, match="neither the exact"):
            migration.migrate(artifact_path=artifact, state_path=state_path)


def test_unknown_state_binding_fails() -> None:
    old = _old_artifact_bytes()

    def mutate(state: dict[str, Any]) -> None:
        for entry in state["identity"]["produced_artifacts"]:
            if entry["path"].endswith("bler_tooling_contract.json"):
                entry["sha256"] = "0" * 64

    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw), artifact_bytes=old, state_binding_bytes=old, state_mutation=mutate
        )
        with pytest.raises(migration.G8BlerMigrationError, match="unknown tooling"):
            migration.migrate(artifact_path=artifact, state_path=state_path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state["identity"]["counters"].__setitem__("inference", 1), "counters"),
        (lambda state: state["identity"]["completed_work_unit_ids"].append("bler-claimed"), "completed"),
        (lambda state: state["identity"].__setitem__("in_progress_work_unit_id", "bler-live"), "in-progress"),
        (lambda state: state["identity"].__setitem__("phase", "G8_A"), "G8_B/tooling_open"),
        (lambda state: state["identity"].__setitem__("stage", "tooling_smoke_complete"), "G8_B/tooling_open"),
    ],
)
def test_invalid_state_precondition_fails(mutation: Any, message: str) -> None:
    old = _old_artifact_bytes()
    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw), artifact_bytes=old, state_binding_bytes=old, state_mutation=mutation
        )
        with pytest.raises(migration.G8BlerMigrationError, match=message):
            migration.migrate(artifact_path=artifact, state_path=state_path)


def test_changed_base_artifact_binding_fails() -> None:
    old = _old_artifact_bytes()

    def mutate(state: dict[str, Any]) -> None:
        for entry in state["identity"]["produced_artifacts"]:
            if entry["path"].endswith("campaign_manifest.json"):
                entry["sha256"] = "0" * 64

    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw), artifact_bytes=old, state_binding_bytes=old, state_mutation=mutate
        )
        with pytest.raises(migration.G8BlerMigrationError, match="non-tooling"):
            migration.migrate(artifact_path=artifact, state_path=state_path)


def test_interruption_after_artifact_replacement_can_resume() -> None:
    old = _old_artifact_bytes()
    new = _new_artifact_bytes()
    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw), artifact_bytes=old, state_binding_bytes=old
        )
        artifact.write_bytes(new)  # simulated crash after artifact replace, before state replace
        result = migration.migrate(artifact_path=artifact, state_path=state_path)
        assert result["contract_sha256"] == sha256_bytes(new)
        final = load_campaign_state(state_path)
        tooling = next(
            entry
            for entry in final["identity"]["produced_artifacts"]
            if entry["path"] == str(artifact.relative_to(REPO_ROOT))
        )
        assert tooling["sha256"] == sha256_bytes(new)


def test_repeated_migration_preserves_exact_final_bytes() -> None:
    old = _old_artifact_bytes()
    new = _new_artifact_bytes()
    with tempfile.TemporaryDirectory(dir=G8_ROOT, prefix=".b1c-migration-test-") as raw:
        artifact, state_path = _write_pair(
            Path(raw), artifact_bytes=old, state_binding_bytes=old
        )
        migration.migrate(artifact_path=artifact, state_path=state_path)
        artifact_bytes = artifact.read_bytes()
        state_bytes = state_path.read_bytes()
        migration.migrate(artifact_path=artifact, state_path=state_path)
        assert artifact.read_bytes() == artifact_bytes == new
        assert state_path.read_bytes() == state_bytes
