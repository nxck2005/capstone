#!/usr/bin/env python3
"""Migrate the one registered B4 contract binding across its verified correction.

This is intentionally narrower than ordinary artifact registration: it can
replace only the exact old runner binding and cannot alter any scientific
campaign state.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_bler_runner as runner  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    G8ContractError,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    validate_campaign_state,
    validate_state_transition,
    write_campaign_state_atomically,
)
from config.params import REPO_ROOT  # noqa: E402
import verify_g8_bler_runner_contract as runner_verifier  # noqa: E402


RUNNER_PATH = runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
REQUIRED_PHASE = "G8_B"
REQUIRED_STAGE = "tooling_open"


class RunnerContractMigrationError(G8ContractError):
    """The exact B4 runner-contract migration precondition was not met."""


def _binding(entries: list[dict[str, Any]], path: str) -> dict[str, Any]:
    matches = [entry for entry in entries if entry.get("path") == path]
    if len(matches) != 1:
        raise RunnerContractMigrationError(
            f"campaign state must contain exactly one binding for {path}"
        )
    if set(matches[0]) != {"path", "sha256", "bytes"}:
        raise RunnerContractMigrationError(f"binding for {path} is not closed")
    return dict(matches[0])


def _require_zero_science(identity: dict[str, Any]) -> None:
    if identity["phase"] != REQUIRED_PHASE or identity["stage"] != REQUIRED_STAGE:
        raise RunnerContractMigrationError("runner-contract migration requires G8_B/tooling_open")
    if identity["completed_work_unit_ids"]:
        raise RunnerContractMigrationError("runner-contract migration requires no completed units")
    if identity["in_progress_work_unit_id"] is not None:
        raise RunnerContractMigrationError("runner-contract migration requires no in-progress unit")
    if any(identity["counters"][name] != 0 for name in COUNTER_FIELDS):
        raise RunnerContractMigrationError("runner-contract migration requires zero counters")


def migrate(
    *,
    contract_path: Path | str | None = None,
    state_path: Path | str | None = None,
) -> dict[str, Any]:
    """Replace exactly the old runner binding and return the installed state."""

    artifact_path = (
        REPO_ROOT / RUNNER_PATH if contract_path is None else Path(contract_path)
    )
    live_state_path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    try:
        before_raw = live_state_path.read_bytes()
    except OSError as exc:
        raise RunnerContractMigrationError(f"cannot read campaign state: {exc}") from exc
    try:
        candidate = runner_verifier.verify(artifact_path)
    except Exception as exc:
        raise RunnerContractMigrationError(
            f"new runner contract failed independent verification: {exc}"
        ) from exc
    if candidate.get("supersedes") != {
        "contract_id": runner.SUPERSEDED_RUNNER_CONTRACT_ID,
        "contract_sha256": runner.SUPERSEDED_RUNNER_CONTRACT_SHA256,
        "contract_bytes": runner.SUPERSEDED_RUNNER_CONTRACT_BYTES,
        "reason": runner.RUNNER_CONTRACT_SUPERSESSION_REASON,
    }:
        raise RunnerContractMigrationError("new runner contract supersession is not exact")
    new_raw = artifact_path.read_bytes()
    new_binding = {"path": RUNNER_PATH, "sha256": sha256_bytes(new_raw), "bytes": len(new_raw)}
    try:
        before_payload = json.loads(before_raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerContractMigrationError("campaign state is not decodable JSON") from exc
    if before_raw != rendered_json(before_payload):
        raise RunnerContractMigrationError("campaign state is not canonical rendered JSON")
    if not isinstance(before_payload, dict) or not isinstance(before_payload.get("identity"), dict):
        raise RunnerContractMigrationError("campaign state has no closed identity object")
    original_artifacts = before_payload["identity"].get("produced_artifacts")
    if not isinstance(original_artifacts, list):
        raise RunnerContractMigrationError("campaign state produced-artifact list is malformed")
    if len(original_artifacts) != 6:
        raise RunnerContractMigrationError(
            f"runner-contract migration requires exactly six existing artifacts, found {len(original_artifacts)}"
        )
    old_binding = _binding(original_artifacts, RUNNER_PATH)
    if old_binding != {
        "path": RUNNER_PATH,
        "sha256": runner.SUPERSEDED_RUNNER_CONTRACT_SHA256,
        "bytes": runner.SUPERSEDED_RUNNER_CONTRACT_BYTES,
    }:
        raise RunnerContractMigrationError("existing runner binding is not the exact superseded B4 binding")
    if new_binding["sha256"] == old_binding["sha256"] and new_binding["bytes"] == old_binding["bytes"]:
        raise RunnerContractMigrationError("new runner contract did not supersede old bytes")

    # The generic loader quite properly rejects a state whose registered
    # artifact bytes are stale.  Validate the complete closed state schema and
    # every other artifact against a one-field candidate projection, then keep
    # the actual old binding for the guarded replacement below.
    validation_projection = copy.deepcopy(before_payload)
    projection_entries = validation_projection["identity"]["produced_artifacts"]
    for index, entry in enumerate(projection_entries):
        if entry["path"] == RUNNER_PATH:
            projection_entries[index] = new_binding
    try:
        validated_projection = validate_campaign_state(validation_projection)
    except Exception as exc:
        raise RunnerContractMigrationError(
            f"campaign state failed strict pre-migration validation: {exc}"
        ) from exc
    identity = validated_projection["identity"]
    _require_zero_science(identity)
    artifacts = original_artifacts

    previous = copy.deepcopy(before_payload)
    current = copy.deepcopy(before_payload)
    current_artifacts = current["identity"]["produced_artifacts"]
    replaced = 0
    for index, entry in enumerate(current_artifacts):
        if entry["path"] == RUNNER_PATH:
            current_artifacts[index] = new_binding
            replaced += 1
    if replaced != 1:
        raise RunnerContractMigrationError("runner binding replacement was not exactly one entry")
    current_artifacts.sort(key=lambda entry: entry["path"])
    current["identity"]["restart_command"] = identity["restart_command"]

    for field in (
        "campaign_id",
        "campaign_manifest_sha256",
        "phase",
        "stage",
        "completed_work_unit_ids",
        "in_progress_work_unit_id",
        "seed_derivation_identity",
        "counters",
    ):
        if current["identity"][field] != identity[field]:
            raise RunnerContractMigrationError(f"migration changed unrelated state field {field}")
    for before, after in zip(artifacts, current_artifacts, strict=False):
        if before["path"] != RUNNER_PATH and before != after:
            raise RunnerContractMigrationError(
                f"migration changed unrelated produced artifact {before['path']}"
            )
    # Verify that another writer did not publish a state between the read and
    # this guarded replacement.  The atomic writer itself handles durability.
    if live_state_path.read_bytes() != before_raw:
        raise RunnerContractMigrationError("campaign state became stale before runner migration")
    validate_state_transition(validated_projection, current)
    write_campaign_state_atomically(live_state_path, current)
    installed = load_campaign_state(live_state_path)
    installed_raw = live_state_path.read_bytes()
    if installed_raw != rendered_json(current) or installed != current:
        raise RunnerContractMigrationError("installed migrated campaign state bytes do not reproduce")
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=REPO_ROOT / RUNNER_PATH)
    parser.add_argument("--state", type=Path, default=CAMPAIGN_STATE)
    args = parser.parse_args(argv)
    try:
        state = migrate(contract_path=args.contract, state_path=args.state)
    except (G8ContractError, OSError) as exc:
        raise SystemExit(f"G8 B4 runner-contract migration HOLD: {exc}") from exc
    identity = state["identity"]
    binding = _binding(identity["produced_artifacts"], RUNNER_PATH)
    print(
        "G8 B4 runner-contract migration PASS: "
        f"contract_sha256={binding['sha256']} bytes={binding['bytes']} "
        f"state_sha256={sha256_bytes(rendered_json(state))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
