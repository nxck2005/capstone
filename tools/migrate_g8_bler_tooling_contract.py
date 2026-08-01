#!/usr/bin/env python3
"""Recoverably install the corrected pre-science G8_B BLER contract.

This utility is deliberately narrower than artifact registration.  It accepts
only the known B1 or B1C tooling binding, never changes any scientific state,
and replaces the contract before replacing the state binding.  If execution
stops between those two replacements, running this same command again repairs
the old state binding against the already-installed B1C artifact.
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

import gen_g8_bler_tooling_contract as generator  # noqa: E402
import verify_g8_bler_tooling_contract as independent_verifier  # noqa: E402
from baseline import g8_bler_contract as contract  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    G8ContractError,
    initial_campaign_state,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
    validate_campaign_state,
    write_campaign_state_atomically,
)


class G8BlerMigrationError(RuntimeError):
    """A pre-science contract/state migration invariant failed."""


OLD_CONTRACT_ID = contract.SUPERSEDES_CONTRACT_ID
OLD_CONTRACT_SHA256 = contract.SUPERSEDES_CONTRACT_SHA256
OLD_CONTRACT_BYTES = 11924  # B1 artifact length recorded before opening B1C
B2_RESTART_COMMAND = (
    'rg -n "BLER_WORK_UNIT|derive_seed|produced_artifacts|completed_work_unit_ids|'
    'in_progress_work_unit_id|write_campaign_state_atomically" src/baseline tools tests'
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8BlerMigrationError(message)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        raise G8BlerMigrationError(
            f"migration path must be inside the repository: {path}"
        ) from None


def _binding(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {"path": _relative_path(path), "sha256": sha256_bytes(body), "bytes": len(body)}


def _read_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8BlerMigrationError(f"cannot read campaign state {path}: {exc}") from exc
    _require(raw == rendered_json(payload), "campaign state is not canonical rendered JSON")
    _require(isinstance(payload, dict), "campaign state is not a JSON object")
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
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _build_and_verify_candidate(artifact_path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = generator.build()
    body = rendered_json(payload)
    candidate = _write_candidate(artifact_path, body)
    try:
        independent_verifier.verify(candidate)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    digest = sha256_bytes(body)
    _require(digest == sha256_file(candidate), "candidate BLER contract SHA-256 changed while staged")
    _require(payload["contract_id"].startswith(f"{contract.CONTRACT_ID_PREFIX}-"),
             "candidate BLER contract ID has the wrong prefix")
    candidate.unlink(missing_ok=True)
    return payload, body, digest


def _state_tooling_entry(state: dict[str, Any], tooling_path: str) -> dict[str, Any]:
    identity = state.get("identity")
    _require(isinstance(identity, dict), "campaign state identity is missing")
    artifacts = identity.get("produced_artifacts")
    _require(isinstance(artifacts, list), "campaign state produced artifacts are missing")
    matches = [entry for entry in artifacts if isinstance(entry, dict) and entry.get("path") == tooling_path]
    _require(len(matches) == 1, "campaign state must contain exactly one tooling-contract binding")
    entry = matches[0]
    _require(set(entry) == {"path", "sha256", "bytes"}, "tooling-contract binding has the wrong schema")
    return entry


def _verify_state_preconditions(
    state: dict[str, Any],
    *,
    artifact_path: Path,
    actual_artifact_binding: dict[str, Any],
    new_artifact_binding: dict[str, Any],
) -> dict[str, Any]:
    tooling_path = actual_artifact_binding["path"]
    entry = _state_tooling_entry(state, tooling_path)
    known = {
        (OLD_CONTRACT_SHA256, OLD_CONTRACT_BYTES),
        (new_artifact_binding["sha256"], new_artifact_binding["bytes"]),
    }
    _require(
        (entry["sha256"], entry["bytes"]) in known,
        "campaign state has an unknown tooling-contract binding",
    )
    _require(
        (actual_artifact_binding["sha256"], actual_artifact_binding["bytes"]) in known,
        "current tooling artifact is neither the exact B1 nor B1C artifact",
    )

    identity = state["identity"]
    _require(identity.get("phase") == "G8_B" and identity.get("stage") == "tooling_open",
             "migration requires G8_B/tooling_open")
    _require(identity.get("completed_work_unit_ids") == [],
             "migration requires no completed scientific work units")
    _require(identity.get("in_progress_work_unit_id") is None,
             "migration requires no in-progress scientific work unit")
    counters = identity.get("counters")
    _require(isinstance(counters, dict) and set(counters) == {
        "validation_decoding", "inference", "training", "test_access"
    }, "campaign counters have the wrong schema")
    _require(all(value == 0 for value in counters.values()),
             "migration requires all scientific counters to be zero")
    _require(identity.get("restart_command") == B2_RESTART_COMMAND,
             "campaign state does not carry the exact B2 restart command")
    _require(identity.get("seed_derivation_identity") == contract.SEED_DERIVATION_IDENTITY,
             "campaign state seed derivation identity changed")

    base = initial_campaign_state(stage="preflight_complete")["identity"]["produced_artifacts"]
    other = [entry for entry in identity["produced_artifacts"] if entry.get("path") != tooling_path]
    _require(sorted(other, key=lambda item: item["path"]) == sorted(base, key=lambda item: item["path"]),
             "a non-tooling produced-artifact binding changed")

    # Validate every other state invariant against the bytes currently on
    # disk, temporarily substituting the actual artifact binding.  This is
    # what makes both new-artifact/old-state and old-artifact/new-state
    # recoverable without trusting either mismatched pair.
    for_validation = copy.deepcopy(state)
    for item in for_validation["identity"]["produced_artifacts"]:
        if item.get("path") == tooling_path:
            item.update(actual_artifact_binding)
    try:
        validate_campaign_state(for_validation)
    except G8ContractError as exc:
        raise G8BlerMigrationError(str(exc)) from exc
    return entry


def migrate(
    *,
    artifact_path: Path = generator.BLER_TOOLING_CONTRACT,
    state_path: Path = CAMPAIGN_STATE,
) -> dict[str, Any]:
    """Install B1C and repair its state binding, safely rerunnable."""

    artifact_path = Path(artifact_path)
    state_path = Path(state_path)
    payload, new_body, new_sha256 = _build_and_verify_candidate(artifact_path)
    new_binding = {
        "path": _relative_path(artifact_path),
        "sha256": new_sha256,
        "bytes": len(new_body),
    }
    actual_binding = _binding(artifact_path)
    state = _read_state(state_path)
    _verify_state_preconditions(
        state,
        artifact_path=artifact_path,
        actual_artifact_binding=actual_binding,
        new_artifact_binding=new_binding,
    )

    staged = _write_candidate(artifact_path, new_body)
    try:
        os.replace(staged, artifact_path)
        _fsync_directory(artifact_path.parent)
    finally:
        staged.unlink(missing_ok=True)

    # Verify the replaced artifact before exposing its state binding.  If the
    # process stops here, the next invocation accepts the new-artifact/old-state
    # pair and repeats the state half only.
    independent_verifier.verify(artifact_path)
    contract.load_bler_tooling_contract(artifact_path)

    migrated = copy.deepcopy(state)
    tooling_path = new_binding["path"]
    replaced = 0
    for entry in migrated["identity"]["produced_artifacts"]:
        if entry.get("path") == tooling_path:
            entry.clear()
            entry.update(new_binding)
            replaced += 1
    _require(replaced == 1, "migration did not replace exactly one tooling binding")
    migrated["identity"]["produced_artifacts"].sort(key=lambda item: item["path"])
    state_sha256 = write_campaign_state_atomically(state_path, migrated)
    final_state = load_campaign_state(state_path)
    independent_verifier.verify(artifact_path)
    contract.load_bler_tooling_contract(artifact_path)
    return {
        "contract_id": payload["contract_id"],
        "contract_sha256": new_sha256,
        "state_sha256": state_sha256,
        "state": final_state,
    }


def main() -> int:
    try:
        result = migrate()
    except (G8BlerMigrationError, G8ContractError, independent_verifier.G8BlerToolingError,
            contract.G8BlerContractError) as exc:
        raise SystemExit(f"G8 B1C tooling-contract migration HOLD: {exc}") from exc
    identity = result["state"]["identity"]
    print(
        "G8 B1C tooling-contract migration PASS: "
        f"contract_id={result['contract_id']} contract_sha256={result['contract_sha256']} "
        f"state_sha256={result['state_sha256']} phase={identity['phase']}/{identity['stage']} "
        f"completed_work_units={len(identity['completed_work_unit_ids'])} "
        f"in_progress={identity['in_progress_work_unit_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
