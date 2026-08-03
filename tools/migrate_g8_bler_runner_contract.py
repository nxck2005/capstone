#!/usr/bin/env python3
"""Safely migrate the registered G8_B runner and smoke bindings to v3.

The migration is deliberately a small state machine.  It accepts only the
registered v2 runner/smoke pair or independently verified v3 replacements,
projects stale raw state through the strict campaign-state validator, and
replaces only the two named produced-artifact bindings.  Re-running after an
interruption is therefore a no-op once each installed file and its binding
agree.
"""

from __future__ import annotations

import argparse
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

from baseline import g8_bler_runner as runner  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    G8ContractError,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    PHASE_ORDER,
    STATE_STAGES,
    write_campaign_state_atomically,
)
from config.params import REPO_ROOT  # noqa: E402
import verify_g8_bler_runner_contract as runner_verifier  # noqa: E402
import verify_g8_bounded_smoke as smoke_verifier  # noqa: E402


RUNNER_PATH = runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
SMOKE_PATH = runner.SMOKE_RECORD_REPO_RELATIVE_PATH
REQUIRED_PHASE = "G8_B"
REQUIRED_STAGE = "tooling_open"
EXPECTED_ARTIFACT_COUNT = 7
EXPECTED_OTHER_PATHS = {
    "results/baseline/g8/bler_resume_contract.json",
    "results/baseline/g8/bler_state_contract.json",
    "results/baseline/g8/bler_tooling_contract.json",
    "results/baseline/g8/campaign_manifest.json",
    "results/baseline/g8/required_bler_identities.json",
}

OLD_SMOKE_SHA256 = "cff4fb75835c4a010baed285103c3ba425b7b44b226186ce9969dcb17537763e"
OLD_SMOKE_BYTES = 36572
OLD_SMOKE_SCHEMA_VERSION = 2
OLD_SMOKE_RUNNER_ID = runner.V2_RUNNER_CONTRACT_ID
OLD_SMOKE_RUNNER_SHA256 = runner.V2_RUNNER_CONTRACT_SHA256


class RunnerContractMigrationError(G8ContractError):
    """The exact v2 -> v3 migration precondition was not met."""


def _binding(entries: list[dict[str, Any]], path: str) -> dict[str, Any]:
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == path]
    if len(matches) != 1:
        raise RunnerContractMigrationError(f"campaign state must contain exactly one binding for {path}")
    if set(matches[0]) != {"path", "sha256", "bytes"}:
        raise RunnerContractMigrationError(f"binding for {path} is not closed")
    return dict(matches[0])


def _require_zero_science(identity: dict[str, Any]) -> None:
    if identity.get("phase") != REQUIRED_PHASE or identity.get("stage") != REQUIRED_STAGE:
        raise RunnerContractMigrationError("runner-contract migration requires G8_B/tooling_open")
    if identity.get("completed_work_unit_ids") != []:
        raise RunnerContractMigrationError("runner-contract migration requires no completed units")
    if identity.get("in_progress_work_unit_id") is not None:
        raise RunnerContractMigrationError("runner-contract migration requires no in-progress unit")
    counters = identity.get("counters")
    if not isinstance(counters, dict) or any(counters.get(name) != 0 for name in COUNTER_FIELDS):
        raise RunnerContractMigrationError("runner-contract migration requires zero counters")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerContractMigrationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict) or raw != rendered_json(payload):
        raise RunnerContractMigrationError(f"{label} is not canonical rendered JSON")
    return payload, raw


def _read_runner_contract(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload, raw = _read_json(path, "runner contract")
    digest = sha256_bytes(raw)
    if (
        payload.get("schema_version") == 2
        and payload.get("contract_id") == runner.V2_RUNNER_CONTRACT_ID
        and digest == runner.V2_RUNNER_CONTRACT_SHA256
        and len(raw) == runner.V2_RUNNER_CONTRACT_BYTES
    ):
        return payload, raw, "v2"
    if payload.get("schema_version") == runner.RUNNER_CONTRACT_SCHEMA_VERSION:
        try:
            runner_verifier.verify(path)
        except Exception as exc:
            raise RunnerContractMigrationError(
                f"v3 runner contract failed independent verification: {exc}"
            ) from exc
        return payload, raw, "v3"
    raise RunnerContractMigrationError("runner contract is neither the exact registered v2 nor verified v3")


def _read_smoke_record(
    path: Path,
    *,
    expected_v3_runner_id: str,
    expected_v3_runner_sha256: str,
) -> tuple[dict[str, Any], bytes, str]:
    payload, raw = _read_json(path, "bounded smoke record")
    digest = sha256_bytes(raw)
    if (
        payload.get("schema_version") == OLD_SMOKE_SCHEMA_VERSION
        and digest == OLD_SMOKE_SHA256
        and len(raw) == OLD_SMOKE_BYTES
        and payload.get("bler_runner_contract_id") == OLD_SMOKE_RUNNER_ID
        and payload.get("bler_runner_contract_sha256") == OLD_SMOKE_RUNNER_SHA256
    ):
        return payload, raw, "v2"
    if (
        payload.get("schema_version") == runner.SMOKE_RECORD_SCHEMA_VERSION
        and payload.get("bler_runner_contract_id") == expected_v3_runner_id
        and payload.get("bler_runner_contract_sha256") == expected_v3_runner_sha256
    ):
        return payload, raw, "v3"
    raise RunnerContractMigrationError("smoke record is neither the exact registered v2 nor a v3-bound record")


def _strict_projection(
    payload: dict[str, Any],
    *,
    runner_binding: dict[str, Any],
    smoke_binding: dict[str, Any],
    runner_path: Path,
    smoke_path: Path,
) -> dict[str, Any]:
    """Validate stale raw state after projecting its changed artifact bytes."""

    projection = copy.deepcopy(payload)
    entries = projection["identity"]["produced_artifacts"]
    for entry in entries:
        if entry["path"] == RUNNER_PATH:
            entry.update(runner_binding)
        elif entry["path"] == SMOKE_PATH:
            entry.update(smoke_binding)
    if set(projection) != {"schema_version", "identity", "metadata"} or projection["schema_version"] != 1:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: schema changed")
    identity = projection.get("identity")
    metadata = projection.get("metadata")
    expected_identity_fields = {
        "campaign_id", "campaign_manifest_sha256", "phase", "stage",
        "completed_work_unit_ids", "in_progress_work_unit_id", "produced_artifacts",
        "restart_command", "seed_derivation_identity", "counters",
    }
    if not isinstance(identity, dict) or set(identity) != expected_identity_fields:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: identity schema changed")
    if not isinstance(metadata, dict) or set(metadata) != {"last_successful_checkpoint_time"}:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: metadata schema changed")
    manifest = json.loads((REPO_ROOT / "results/baseline/g8/campaign_manifest.json").read_bytes())
    if identity["campaign_id"] != manifest["campaign_id"] or identity["campaign_manifest_sha256"] != sha256_bytes((REPO_ROOT / "results/baseline/g8/campaign_manifest.json").read_bytes()):
        raise RunnerContractMigrationError("campaign state failed strict projected validation: campaign binding changed")
    if identity["phase"] not in PHASE_ORDER or identity["stage"] not in STATE_STAGES[identity["phase"]]:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: phase/stage changed")
    if identity["completed_work_unit_ids"] != sorted(set(identity["completed_work_unit_ids"])):
        raise RunnerContractMigrationError("campaign state failed strict projected validation: completed IDs changed")
    if identity["in_progress_work_unit_id"] in identity["completed_work_unit_ids"]:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: progress overlap")
    if not isinstance(identity["restart_command"], str) or not identity["restart_command"] or not isinstance(identity["seed_derivation_identity"], str) or not identity["seed_derivation_identity"]:
        raise RunnerContractMigrationError("campaign state failed strict projected validation: restart/seed identity changed")
    if set(identity["counters"]) != set(COUNTER_FIELDS) or any(type(value) is not int or value < 0 for value in identity["counters"].values()):
        raise RunnerContractMigrationError("campaign state failed strict projected validation: counters changed")
    artifacts = identity["produced_artifacts"]
    paths = [entry.get("path") for entry in artifacts] if isinstance(artifacts, list) else []
    if not isinstance(artifacts, list) or paths != sorted(set(paths)):
        raise RunnerContractMigrationError("campaign state failed strict projected validation: artifact list changed")
    for entry in artifacts:
        if set(entry) != {"path", "sha256", "bytes"} or Path(entry["path"]).is_absolute():
            raise RunnerContractMigrationError("campaign state failed strict projected validation: artifact binding schema changed")
        if entry["path"] == RUNNER_PATH:
            actual_path = runner_path
        elif entry["path"] == SMOKE_PATH:
            actual_path = smoke_path
        else:
            actual_path = REPO_ROOT / entry["path"]
        try:
            raw = actual_path.read_bytes()
        except OSError as exc:
            raise RunnerContractMigrationError(f"campaign state failed strict projected validation: cannot read {entry['path']}") from exc
        if entry["bytes"] != len(raw) or entry["sha256"] != sha256_bytes(raw):
            raise RunnerContractMigrationError(f"campaign state failed strict projected validation: binding does not reproduce for {entry['path']}")
    return projection


def _assert_only_bindings_changed(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["metadata"] != after["metadata"]:
        raise RunnerContractMigrationError("migration changed campaign metadata")
    for field in (
        "campaign_id", "campaign_manifest_sha256", "phase", "stage",
        "completed_work_unit_ids", "in_progress_work_unit_id", "restart_command",
        "seed_derivation_identity", "counters",
    ):
        if before["identity"][field] != after["identity"][field]:
            raise RunnerContractMigrationError(f"migration changed unrelated state field {field}")
    before_other = {
        entry["path"]: entry
        for entry in before["identity"]["produced_artifacts"]
        if entry["path"] not in {RUNNER_PATH, SMOKE_PATH}
    }
    after_other = {
        entry["path"]: entry
        for entry in after["identity"]["produced_artifacts"]
        if entry["path"] not in {RUNNER_PATH, SMOKE_PATH}
    }
    if before_other != after_other:
        raise RunnerContractMigrationError("migration changed an unrelated artifact binding")


def _publish_state(path: Path, payload: dict[str, Any], *, runner_path: Path, smoke_path: Path) -> None:
    """Use the repository writer live and an equivalent isolated writer in tests."""

    if runner_path == runner.DEFAULT_RUNNER_CONTRACT_PATH and smoke_path == runner.DEFAULT_SMOKE_RECORD_PATH:
        write_campaign_state_atomically(path, payload)
        return
    body = rendered_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_state_boundary(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "identity", "metadata"}:
        raise RunnerContractMigrationError("campaign state has an invalid closed schema")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise RunnerContractMigrationError("campaign state has no closed identity object")
    _require_zero_science(identity)
    artifacts = identity.get("produced_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise RunnerContractMigrationError(
            f"runner-contract migration requires exactly seven existing artifacts, found {len(artifacts) if isinstance(artifacts, list) else 'invalid'}"
        )
    runner_binding = _binding(artifacts, RUNNER_PATH)
    smoke_binding = _binding(artifacts, SMOKE_PATH)
    paths = {entry.get("path") for entry in artifacts if isinstance(entry, dict)}
    expected = EXPECTED_OTHER_PATHS | {RUNNER_PATH, SMOKE_PATH}
    if paths != expected:
        raise RunnerContractMigrationError("campaign state contains an unknown or missing artifact binding")
    return artifacts, runner_binding, smoke_binding


def _binding_for(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _verify_v3_smoke_against_projection(
    smoke_path: Path,
    state_path: Path,
    state_payload: dict[str, Any],
    *,
    runner_path: Path,
    runner_binding: dict[str, Any],
    smoke_binding: dict[str, Any],
) -> None:
    projection = copy.deepcopy(state_payload)
    for entry in projection["identity"]["produced_artifacts"]:
        if entry["path"] == RUNNER_PATH:
            entry.update(runner_binding)
        elif entry["path"] == SMOKE_PATH:
            entry.update(smoke_binding)
    with tempfile.TemporaryDirectory(prefix="g8-b5-smoke-state-") as directory:
        projected_path = Path(directory) / "campaign_state.json"
        projected_path.write_bytes(rendered_json(projection))
        try:
            smoke_verifier.verify(
                smoke_path,
                campaign_state_path=projected_path,
                runner_contract_path=runner_path,
            )
        except Exception as exc:
            raise RunnerContractMigrationError(
                f"installed v3 smoke record failed independent verification: {exc}"
            ) from exc


def migrate(
    *,
    contract_path: Path | str | None = None,
    state_path: Path | str | None = None,
    smoke_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile runner and smoke bindings, preserving every other field."""

    artifact_path = runner.DEFAULT_RUNNER_CONTRACT_PATH if contract_path is None else Path(contract_path)
    live_state_path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    record_path = runner.DEFAULT_SMOKE_RECORD_PATH if smoke_path is None else Path(smoke_path)

    before_raw = live_state_path.read_bytes()
    before_payload, parsed_raw = _read_json(live_state_path, "campaign state")
    if before_raw != parsed_raw:
        raise RunnerContractMigrationError("campaign state is not canonical rendered JSON")
    original_artifacts, old_runner_binding, old_smoke_binding = _validate_state_boundary(before_payload)

    runner_payload, runner_raw, runner_version = _read_runner_contract(artifact_path)
    smoke_payload, smoke_raw, smoke_version = _read_smoke_record(
        record_path,
        expected_v3_runner_id=runner_payload["contract_id"],
        expected_v3_runner_sha256=sha256_bytes(runner_raw),
    )
    del runner_payload, smoke_payload

    actual_runner_binding = _binding_for(RUNNER_PATH, runner_raw)
    actual_smoke_binding = _binding_for(SMOKE_PATH, smoke_raw)
    expected_old_runner_binding = {
        "path": RUNNER_PATH,
        "sha256": runner.V2_RUNNER_CONTRACT_SHA256,
        "bytes": runner.V2_RUNNER_CONTRACT_BYTES,
    }
    expected_old_smoke_binding = {
        "path": SMOKE_PATH,
        "sha256": OLD_SMOKE_SHA256,
        "bytes": OLD_SMOKE_BYTES,
    }

    if old_runner_binding != expected_old_runner_binding and old_runner_binding != actual_runner_binding:
        raise RunnerContractMigrationError("state runner binding is neither exact v2 nor the installed contract")
    if old_smoke_binding != expected_old_smoke_binding and old_smoke_binding != actual_smoke_binding:
        raise RunnerContractMigrationError("state smoke binding is neither exact v2 nor the installed record")
    if runner_version == "v2" and smoke_version == "v3":
        raise RunnerContractMigrationError("v3 smoke record cannot be installed while the runner file is v2")
    if old_runner_binding == actual_runner_binding and runner_version == "v2" and old_smoke_binding == actual_smoke_binding and smoke_version == "v2":
        projected_before = _strict_projection(
            before_payload,
            runner_binding=actual_runner_binding,
            smoke_binding=actual_smoke_binding,
            runner_path=artifact_path,
            smoke_path=record_path,
        )
        return load_campaign_state(live_state_path) if live_state_path == CAMPAIGN_STATE else projected_before

    if runner_version == "v3" and smoke_version == "v3":
        _verify_v3_smoke_against_projection(
            record_path,
            live_state_path,
            before_payload,
            runner_path=artifact_path,
            runner_binding=actual_runner_binding,
            smoke_binding=actual_smoke_binding,
        )

    projected_before = _strict_projection(
        before_payload,
        runner_binding=actual_runner_binding if runner_version == "v3" else old_runner_binding,
        smoke_binding=actual_smoke_binding if smoke_version == "v3" else old_smoke_binding,
        runner_path=artifact_path,
        smoke_path=record_path,
    )
    current = copy.deepcopy(before_payload)
    current_artifacts = current["identity"]["produced_artifacts"]
    for entry in current_artifacts:
        if entry["path"] == RUNNER_PATH and runner_version == "v3":
            entry.update(actual_runner_binding)
        elif entry["path"] == SMOKE_PATH and smoke_version == "v3":
            entry.update(actual_smoke_binding)
    current["identity"]["produced_artifacts"] = sorted(current_artifacts, key=lambda entry: entry["path"])

    _assert_only_bindings_changed(projected_before, current)
    if current == before_payload:
        return load_campaign_state(live_state_path) if live_state_path == CAMPAIGN_STATE else projected_before
    if live_state_path.read_bytes() != before_raw:
        raise RunnerContractMigrationError("campaign state became stale before binding migration")
    _publish_state(live_state_path, current, runner_path=artifact_path, smoke_path=record_path)
    installed_raw = live_state_path.read_bytes()
    if installed_raw != rendered_json(current):
        raise RunnerContractMigrationError("installed migrated campaign state bytes do not reproduce")
    if live_state_path == CAMPAIGN_STATE:
        return load_campaign_state(live_state_path)
    # The isolated state still points at repository-relative artifacts; the
    # projected validator above is the authoritative check for this test mode.
    return _strict_projection(
        json.loads(installed_raw),
        runner_binding=actual_runner_binding if runner_version == "v3" else old_runner_binding,
        smoke_binding=actual_smoke_binding if smoke_version == "v3" else old_smoke_binding,
        runner_path=artifact_path,
        smoke_path=record_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=runner.DEFAULT_RUNNER_CONTRACT_PATH)
    parser.add_argument("--state", type=Path, default=CAMPAIGN_STATE)
    parser.add_argument("--smoke", type=Path, default=runner.DEFAULT_SMOKE_RECORD_PATH)
    args = parser.parse_args(argv)
    try:
        state = migrate(contract_path=args.contract, state_path=args.state, smoke_path=args.smoke)
    except (G8ContractError, OSError) as exc:
        raise SystemExit(f"G8 B5 runner/smoke migration HOLD: {exc}") from exc
    identity = state["identity"]
    runner_binding = _binding(identity["produced_artifacts"], RUNNER_PATH)
    smoke_binding = _binding(identity["produced_artifacts"], SMOKE_PATH)
    print(
        "G8 B5 runner/smoke migration PASS: "
        f"runner_sha256={runner_binding['sha256']} runner_bytes={runner_binding['bytes']} "
        f"smoke_sha256={smoke_binding['sha256']} smoke_bytes={smoke_binding['bytes']} "
        f"state_sha256={sha256_bytes(rendered_json(state))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
