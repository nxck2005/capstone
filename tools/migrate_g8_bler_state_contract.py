#!/usr/bin/env python3
"""Recoverably install the corrected pre-science G8_B B2C state contract.

This utility is deliberately narrower than artifact registration.  It replaces
exactly one already-registered produced-artifact binding — the B2 state
contract — with the corrected B2C artifact.  It never appends a duplicate
path, never changes any scientific state, and replaces the contract artifact
before replacing the state binding.  If execution stops between those two
replacements, running this same command again repairs the state binding
against the already-installed B2C artifact.

Because no unit-state file has ever existed, there is nothing to migrate
per unit, and this tool deliberately implements no per-unit migration.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import gen_g8_bler_state_contract as generator  # noqa: E402
import verify_g8_bler_state_contract as independent_verifier  # noqa: E402
from baseline import g8_bler_work_units as units  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    G8ContractError,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
    validate_campaign_state,
    write_campaign_state_atomically,
)


class G8BlerStateMigrationError(RuntimeError):
    """A pre-science state-contract/state migration invariant failed."""


OLD_CONTRACT_ID = units.SUPERSEDED_STATE_CONTRACT_ID
OLD_CONTRACT_SHA256 = units.SUPERSEDED_STATE_CONTRACT_SHA256
OLD_CONTRACT_BYTES = units.SUPERSEDED_STATE_CONTRACT_BYTES
CONTRACT_RELATIVE_PATH = units.STATE_CONTRACT_REPO_RELATIVE_PATH
B3_RESTART_COMMAND = units.B3_RESTART_COMMAND

# Every produced-artifact binding other than the state contract must survive
# this migration byte-for-byte.
IMMUTABLE_ARTIFACT_SHA256 = {
    "results/baseline/g8/bler_tooling_contract.json": units.EXPECTED_B1C_CONTRACT_SHA256,
    "results/baseline/g8/campaign_manifest.json": units.EXPECTED_CAMPAIGN_MANIFEST_SHA256,
    "results/baseline/g8/required_bler_identities.json": units.EXPECTED_REQUIRED_IDENTITIES_SHA256,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8BlerStateMigrationError(message)


def _relative_path(path: Path) -> str:
    """Return the registered binding path for ``path``.

    The campaign-state binding always names the canonical repository-relative
    path.  An artifact inside the repository must therefore *be* that path; an
    artifact outside it is an isolated test or staged-verification root, whose
    binding still uses the canonical name.
    """

    try:
        relative = str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return CONTRACT_RELATIVE_PATH
    _require(
        relative == CONTRACT_RELATIVE_PATH,
        f"the migrated artifact must be the registered state-contract path, not {relative}",
    )
    return CONTRACT_RELATIVE_PATH


def _binding(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {"path": _relative_path(path), "sha256": sha256_bytes(body), "bytes": len(body)}


def _read_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8BlerStateMigrationError(f"cannot read campaign state {path}: {exc}") from exc
    _require(isinstance(payload, dict), "campaign state is not a JSON object")
    _require(raw == rendered_json(payload), "campaign state is not canonical rendered JSON")
    return payload


def _write_candidate(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise G8BlerStateMigrationError(
            f"cannot open {path} for durable publication: {exc}"
        ) from exc
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise G8BlerStateMigrationError(f"cannot fsync {path}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _state_contract_entry(state: dict[str, Any]) -> dict[str, Any]:
    identity = state.get("identity")
    _require(isinstance(identity, dict), "campaign state identity is missing")
    artifacts = identity.get("produced_artifacts")
    _require(isinstance(artifacts, list), "campaign state produced artifacts are missing")
    matches = [
        entry
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("path") == CONTRACT_RELATIVE_PATH
    ]
    _require(len(matches) == 1, "campaign state must contain exactly one state-contract binding")
    entry = matches[0]
    _require(set(entry) == {"path", "sha256", "bytes"}, "state-contract binding has the wrong schema")
    return entry


def _staged_state_for_verification(state: dict[str, Any], new_binding: dict[str, Any]) -> dict[str, Any]:
    """A campaign state that registers the *staged* bytes, for verification only."""

    staged = copy.deepcopy(state)
    for entry in staged["identity"]["produced_artifacts"]:
        if entry.get("path") == CONTRACT_RELATIVE_PATH:
            entry.clear()
            entry.update(new_binding)
    return staged


def _render_candidate() -> tuple[dict[str, Any], bytes, str]:
    payload = generator.build()
    body = rendered_json(payload)
    return payload, body, sha256_bytes(body)


def _stage_and_verify_candidate(
    artifact_path: Path,
    state: dict[str, Any],
    payload: dict[str, Any],
    body: bytes,
    digest: str,
) -> None:
    """Stage the corrected contract and verify it before publishing anything."""

    new_binding = {"path": CONTRACT_RELATIVE_PATH, "sha256": digest, "bytes": len(body)}

    candidate = _write_candidate(artifact_path, body)
    staged_state = _write_candidate(
        artifact_path.parent / "campaign_state.json.staged",
        rendered_json(_staged_state_for_verification(state, new_binding)),
    )
    try:
        # The staged contract is verified against a staged campaign state that
        # registers exactly the staged bytes.  That is what breaks the
        # artifact/state circularity without weakening either check.
        independent_verifier.verify(candidate, campaign_state_path=staged_state)
        _require(digest == sha256_file(candidate), "candidate B2C contract SHA-256 changed while staged")
        _require(
            payload["contract_id"].startswith(f"{units.STATE_CONTRACT_ID_PREFIX}-"),
            "candidate B2C contract ID has the wrong prefix",
        )
        _require(payload["contract_id"] != OLD_CONTRACT_ID, "candidate is still the superseded B2 contract")
        _require(digest != OLD_CONTRACT_SHA256, "candidate artifact is still the superseded B2 artifact")
    finally:
        candidate.unlink(missing_ok=True)
        staged_state.unlink(missing_ok=True)


def _verify_state_preconditions(
    state: dict[str, Any],
    *,
    actual_artifact_binding: dict[str, Any],
    new_artifact_binding: dict[str, Any],
) -> dict[str, Any]:
    entry = _state_contract_entry(state)
    known = {
        (OLD_CONTRACT_SHA256, OLD_CONTRACT_BYTES),
        (new_artifact_binding["sha256"], new_artifact_binding["bytes"]),
    }
    _require(
        (entry["sha256"], entry["bytes"]) in known,
        "campaign state has an unknown state-contract binding",
    )
    _require(
        (actual_artifact_binding["sha256"], actual_artifact_binding["bytes"]) in known,
        "current state-contract artifact is neither the exact B2 nor B2C artifact",
    )

    identity = state["identity"]
    _require(
        identity.get("campaign_id") == units.EXPECTED_CAMPAIGN_ID,
        "campaign ID changed",
    )
    _require(
        identity.get("campaign_manifest_sha256") == units.EXPECTED_CAMPAIGN_MANIFEST_SHA256,
        "campaign-manifest binding changed",
    )
    _require(
        identity.get("phase") == units.PHASE and identity.get("stage") == "tooling_open",
        "migration requires G8_B/tooling_open",
    )
    _require(
        identity.get("completed_work_unit_ids") == [],
        "migration requires no completed scientific work units",
    )
    _require(
        identity.get("in_progress_work_unit_id") is None,
        "migration requires no in-progress scientific work unit",
    )
    counters = identity.get("counters")
    _require(
        isinstance(counters, dict)
        and set(counters) == {"validation_decoding", "inference", "training", "test_access"},
        "campaign counters have the wrong schema",
    )
    _require(
        all(value == 0 for value in counters.values()),
        "migration requires all scientific counters to be zero",
    )
    _require(
        identity.get("restart_command") == B3_RESTART_COMMAND,
        "campaign state does not carry the exact B3 restart command",
    )
    _require(
        isinstance(identity.get("seed_derivation_identity"), str)
        and identity["seed_derivation_identity"],
        "campaign state seed derivation identity is missing",
    )

    other = {
        entry["path"]: entry
        for entry in identity["produced_artifacts"]
        if entry.get("path") != CONTRACT_RELATIVE_PATH
    }
    _require(
        set(other) == set(IMMUTABLE_ARTIFACT_SHA256),
        "the set of unrelated produced artifacts changed",
    )
    for path, expected_sha in IMMUTABLE_ARTIFACT_SHA256.items():
        _require(other[path]["sha256"] == expected_sha, f"an unrelated produced artifact changed: {path}")

    # Validate every other state invariant, temporarily substituting the
    # binding of whatever currently sits at the *registered* repository path,
    # because that is the file the shared validator re-reads.  The migrated
    # pair itself is already pinned by the exact `known` tuples above, so this
    # substitution never weakens the check; it is what makes both
    # new-artifact/old-state and old-artifact/new-state recoverable without
    # trusting either mismatched pair, and what lets an isolated test run
    # against a temporary copy.
    registered_binding = _binding(REPO / CONTRACT_RELATIVE_PATH)
    for_validation = copy.deepcopy(state)
    for item in for_validation["identity"]["produced_artifacts"]:
        if item.get("path") == CONTRACT_RELATIVE_PATH:
            item.clear()
            item.update(registered_binding)
    try:
        validate_campaign_state(for_validation)
    except G8ContractError as exc:
        raise G8BlerStateMigrationError(str(exc)) from exc
    return entry


def migrate(
    *,
    artifact_path: Path = generator.CONTRACT_PATH,
    state_path: Path = CAMPAIGN_STATE,
) -> dict[str, Any]:
    """Install the B2C contract and repair its state binding, safely rerunnable."""

    artifact_path = Path(artifact_path)
    state_path = Path(state_path)
    state = _read_state(state_path)
    before = copy.deepcopy(state["identity"])

    payload, new_body, new_sha256 = _render_candidate()
    new_binding = {
        "path": _relative_path(artifact_path),
        "sha256": new_sha256,
        "bytes": len(new_body),
    }
    actual_binding = _binding(artifact_path)

    # Refuse a drifted or unknown pair *before* doing any work, so the caller
    # sees the migration's own precondition failure rather than a downstream
    # verifier message about a state this tool would never have migrated.
    _verify_state_preconditions(
        state,
        actual_artifact_binding=actual_binding,
        new_artifact_binding=new_binding,
    )
    _stage_and_verify_candidate(artifact_path, state, payload, new_body, new_sha256)

    staged = _write_candidate(artifact_path, new_body)
    try:
        os.replace(staged, artifact_path)
    finally:
        staged.unlink(missing_ok=True)
    _fsync_directory(artifact_path.parent)

    # If the process stops here, the next invocation accepts the
    # new-artifact/old-state pair and repeats the state half only.
    migrated = copy.deepcopy(state)
    replaced = 0
    for entry in migrated["identity"]["produced_artifacts"]:
        if entry.get("path") == CONTRACT_RELATIVE_PATH:
            entry.clear()
            entry.update(new_binding)
            replaced += 1
    _require(replaced == 1, "migration did not replace exactly one state-contract binding")
    migrated["identity"]["produced_artifacts"].sort(key=lambda item: item["path"])
    state_sha256 = write_campaign_state_atomically(state_path, migrated)

    final_state = load_campaign_state(state_path)
    after = final_state["identity"]

    # Everything except the one artifact binding must be preserved exactly.
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
        _require(after[field] == before[field], f"migration changed a preserved field: {field}")
    _require(
        {
            entry["path"]: entry
            for entry in after["produced_artifacts"]
            if entry["path"] != CONTRACT_RELATIVE_PATH
        }
        == {
            entry["path"]: entry
            for entry in before["produced_artifacts"]
            if entry["path"] != CONTRACT_RELATIVE_PATH
        },
        "migration changed an unrelated produced-artifact binding",
    )
    _require(
        _state_contract_entry(final_state) == new_binding,
        "migration did not install the new state-contract binding",
    )

    # Reverify the normal loader and the independent contract verifier against
    # the published pair.
    independent_verifier.verify(artifact_path, campaign_state_path=state_path)
    state_context = units.AuthenticatedUnitStateContext(
        campaign_state_path=state_path, state_contract_path=artifact_path
    )
    _require(
        state_context.state_contract_id == payload["contract_id"]
        and state_context.state_contract_sha256 == new_sha256,
        "the authenticated unit-state context does not bind the migrated contract",
    )
    return {
        "contract_id": payload["contract_id"],
        "contract_sha256": new_sha256,
        "contract_bytes": len(new_body),
        "state_sha256": state_sha256,
        "state": final_state,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = migrate()
    except (
        G8BlerStateMigrationError,
        G8ContractError,
        independent_verifier.G8BlerStateContractError,
        units.G8BlerWorkUnitError,
    ) as exc:
        raise SystemExit(f"G8 B2C state-contract migration HOLD: {exc}") from exc
    identity = result["state"]["identity"]
    print(
        "G8 B2C state-contract migration PASS: "
        f"contract_id={result['contract_id']} contract_sha256={result['contract_sha256']} "
        f"contract_bytes={result['contract_bytes']} state_sha256={result['state_sha256']} "
        f"supersedes={OLD_CONTRACT_ID} "
        f"phase={identity['phase']}/{identity['stage']} "
        f"completed_work_units={len(identity['completed_work_unit_ids'])} "
        f"in_progress={identity['in_progress_work_unit_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
