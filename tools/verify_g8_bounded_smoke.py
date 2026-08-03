#!/usr/bin/env python3
"""Independently verify the tracked, non-scientific G8_B smoke record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import g8_bler_contract as frozen  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_MANIFEST,
    CAMPAIGN_STATE,
    REQUIRED_BLER_IDENTITIES,
    SELECTION_POLICY_FIELDS,
    campaign_identifier,
    load_campaign_manifest,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
)
from config.params import get  # noqa: E402


RECORD_PATH = REPO_ROOT / "results/baseline/g8/bounded_smoke_record.json"
RUNNER_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_runner_contract.json"
STATE_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_state_contract.json"
RESUME_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_resume_contract.json"
TOOLING_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json"
MAX_UNITS = 3
MAX_TRIALS = 16
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_ROLE = "g8_bounded_smoke_record"
EXPECTED_LABEL = "NON-SCIENTIFIC BOUNDED SMOKE"
REQUIRED_ARTIFACT_PATH = "results/baseline/g8/required_bler_identities.json"


class SmokeVerificationError(RuntimeError):
    """The smoke record does not prove the bounded non-scientific contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeVerificationError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SmokeVerificationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    return payload, raw


def _artifact_binding(
    state: dict[str, Any],
    relative_path: str,
    label: str,
    *,
    actual_path: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    entries = state["identity"]["produced_artifacts"]
    matches = [entry for entry in entries if entry.get("path") == relative_path]
    _require(len(matches) == 1, f"{label} is not registered exactly once")
    binding = matches[0]
    _require(set(binding) == {"path", "sha256", "bytes"}, f"{label} binding is not closed")
    _require(type(binding["bytes"]) is int and binding["bytes"] >= 0, f"{label} byte count is invalid")
    _require(isinstance(binding["sha256"], str) and len(binding["sha256"]) == 64, f"{label} SHA-256 is invalid")
    path = REPO_ROOT / relative_path if actual_path is None else Path(actual_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SmokeVerificationError(f"cannot read registered {label}: {exc}") from exc
    _require(len(raw) == binding["bytes"], f"registered {label} byte count does not reproduce")
    _require(sha256_bytes(raw) == binding["sha256"], f"registered {label} SHA-256 does not reproduce")
    return dict(binding), raw


def _contract_binding(
    state: dict[str, Any],
    path: Path,
    label: str,
    *,
    relative_path: str,
) -> tuple[dict[str, Any], str, str, int]:
    binding, raw = _artifact_binding(state, relative_path, label, actual_path=path)
    payload, parsed_raw = _read_json(path, label)
    _require(parsed_raw == raw, f"{label} changed during verification")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    contract_id = payload.get("contract_id")
    _require(isinstance(contract_id, str) and contract_id, f"{label} has no contract ID")
    _require(binding["sha256"] == sha256_bytes(raw) and binding["bytes"] == len(raw), f"{label} binding mismatch")
    return payload, contract_id, binding["sha256"], binding["bytes"]


def _load_state_for_verification(
    path: Path,
    *,
    overrides: dict[str, Path],
) -> dict[str, Any]:
    """Load a canonical state while allowing isolated candidate artifacts."""

    try:
        raw = path.read_bytes()
        state = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SmokeVerificationError(f"cannot read campaign state: {exc}") from exc
    _require(raw == rendered_json(state), "campaign state is not canonical rendered JSON")
    _require(
        isinstance(state, dict)
        and set(state) == {"schema_version", "identity", "metadata"}
        and state["schema_version"] == 1,
        "campaign state schema changed",
    )
    identity = state["identity"]
    _require(
        isinstance(identity, dict)
        and set(identity)
        == {
            "campaign_id", "campaign_manifest_sha256", "phase", "stage",
            "completed_work_unit_ids", "in_progress_work_unit_id", "produced_artifacts",
            "restart_command", "seed_derivation_identity", "counters",
        },
        "campaign state identity schema changed",
    )
    _require(isinstance(state["metadata"], dict) and set(state["metadata"]) == {"last_successful_checkpoint_time"}, "campaign metadata schema changed")
    _require(identity["campaign_id"] == load_campaign_manifest(CAMPAIGN_MANIFEST)["campaign_id"], "campaign state campaign ID changed")
    _require(identity["campaign_manifest_sha256"] == sha256_bytes(CAMPAIGN_MANIFEST.read_bytes()), "campaign state manifest binding changed")
    _require(identity["phase"] == "G8_B" and identity["stage"] in {"tooling_open", "tooling_smoke_complete"}, "smoke verifier requires a G8_B tooling stage")
    _require(identity["completed_work_unit_ids"] == [] and identity["in_progress_work_unit_id"] is None, "smoke state contains work-unit progress")
    _require(isinstance(identity["counters"], dict) and set(identity["counters"]) == {"validation_decoding", "inference", "training", "test_access"}, "campaign counter schema changed")
    _require(all(type(value) is int and value >= 0 for value in identity["counters"].values()), "campaign counters are malformed")
    artifacts = identity["produced_artifacts"]
    _require(isinstance(artifacts, list) and [entry.get("path") for entry in artifacts] == sorted({entry.get("path") for entry in artifacts}), "campaign artifact list is malformed")
    for entry in artifacts:
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256", "bytes"} and not Path(entry["path"]).is_absolute(), "campaign artifact binding is malformed")
        target = overrides.get(entry["path"], REPO_ROOT / entry["path"])
        try:
            body = target.read_bytes()
        except OSError as exc:
            raise SmokeVerificationError(f"cannot read registered campaign artifact {entry['path']}: {exc}") from exc
        _require(type(entry["bytes"]) is int and entry["bytes"] == len(body), f"campaign artifact byte count changed: {entry['path']}")
        _require(entry["sha256"] == sha256_bytes(body), f"campaign artifact SHA changed: {entry['path']}")
    return state


def _policy_fingerprint(manifest: dict[str, Any]) -> str:
    policy = manifest.get("selection_policy")
    _require(isinstance(policy, dict), "campaign manifest selection policy is missing")
    _require(policy.get("fields") == list(SELECTION_POLICY_FIELDS), "selection-policy field ordering changed")
    w4_binding = manifest.get("w4_adjudication")
    _require(isinstance(w4_binding, dict), "campaign manifest W4 adjudication binding is missing")
    w4_path = REPO_ROOT / w4_binding.get("path", "")
    try:
        w4_raw = w4_path.read_bytes()
        w4_payload = json.loads(w4_raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SmokeVerificationError(f"cannot authenticate frozen W4 selection authority: {exc}") from exc
    _require(
        set(w4_binding) == {"path", "role", "sha256", "bytes"}
        and len(w4_raw) == w4_binding["bytes"]
        and sha256_bytes(w4_raw) == w4_binding["sha256"],
        "campaign manifest W4 adjudication binding does not reproduce",
    )
    machinery = w4_payload.get("selection_machinery")
    _require(isinstance(machinery, dict), "frozen W4 selection machinery is missing")
    covered: list[list[Any]] = []
    for field in SELECTION_POLICY_FIELDS:
        head, _, tail = field.partition(".")
        _require(head in machinery, f"selection-policy field {field} is absent from frozen authority")
        value = machinery[head]
        if tail:
            _require(isinstance(value, dict) and tail in value, f"selection-policy field {field} is malformed")
            value = value[tail]
        covered.append([field, value])
    canonical = json.dumps(covered, separators=(",", ":"), ensure_ascii=True)
    reproduced = sha256_bytes(canonical.encode("utf-8"))
    _require(machinery.get("selection_policy_sha256") == reproduced, "frozen W4 selection-policy fingerprint does not reproduce")
    return reproduced


def _authenticated_authority(
    state_path: Path,
    *,
    overrides: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    state = _load_state_for_verification(state_path, overrides=overrides)
    identity = state["identity"]
    _require(identity["phase"] == "G8_B" and identity["stage"] in {"tooling_open", "tooling_smoke_complete"}, "smoke verifier requires a G8_B tooling stage")
    _require(identity["completed_work_unit_ids"] == [], "smoke changed campaign completed IDs")
    _require(identity["in_progress_work_unit_id"] is None, "smoke changed campaign in-progress ID")
    _require(all(value == 0 for value in identity["counters"].values()), "smoke changed scientific counters")

    manifest_raw = CAMPAIGN_MANIFEST.read_bytes()
    _require(identity["campaign_manifest_sha256"] == sha256_bytes(manifest_raw), "campaign state manifest binding is not exact")
    manifest = load_campaign_manifest(CAMPAIGN_MANIFEST)
    _require(manifest["campaign_id"] == identity["campaign_id"] == campaign_identifier(manifest), "campaign manifest authentication failed")
    reproduced_policy = _policy_fingerprint(manifest)
    _require(manifest["selection_policy"]["selection_policy_sha256"] == reproduced_policy, "selection-policy fingerprint does not reproduce")

    required_binding, required_raw = _artifact_binding(state, REQUIRED_ARTIFACT_PATH, "required-BLER artifact")
    required_payload = json.loads(required_raw)
    _require(required_raw == rendered_json(required_payload), "required-BLER artifact is not canonical rendered JSON")
    _require(required_payload.get("campaign") == "G-8" and required_payload.get("schema_version") == 1, "required-BLER artifact schema changed")
    manifest_entries = [entry for entry in manifest.get("generated_preflight_artifacts", []) if entry.get("path") == REQUIRED_ARTIFACT_PATH]
    _require(len(manifest_entries) == 1, "campaign manifest does not bind the required-BLER artifact exactly once")
    manifest_entry = manifest_entries[0]
    _require(
        manifest_entry.get("sha256") == required_binding["sha256"]
        and manifest_entry.get("bytes") == required_binding["bytes"],
        "campaign manifest required-BLER binding differs from registered state artifact",
    )
    return state, manifest, required_payload, required_raw


def _expected_first_ids(required_payload: dict[str, Any]) -> list[str]:
    units = required_payload.get("required_bler_work_units")
    _require(isinstance(units, list), "required work-unit authority is not a list")
    selected: list[str] = []
    for modulation in get("baseline.modulations"):
        matches = [unit for unit in units if unit.get("identity", {}).get("modulation") == modulation]
        _require(matches, f"no required identity for configured modulation {modulation}")
        selected.append(matches[0]["work_unit_id"])
    return selected


def _expected_request(authority: dict[str, Any], work_unit_id: str, unit: dict[str, Any]) -> dict[str, Any]:
    request = {
        "schema_version": frozen.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "artifact_role": frozen.REQUEST_ARTIFACT_ROLE,
        "execution_class": frozen.EXECUTION_CLASS_BOUNDED_SMOKE,
        "campaign_id": authority["campaign_id"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "work_unit_id": work_unit_id,
        "bler_identity": dict(unit["identity"]),
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "trials_requested": MAX_TRIALS,
        "trial_count_source": frozen.BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
        "seed_derivation_identity": frozen.SEED_DERIVATION_IDENTITY,
        "seed_domain_separator": frozen.SEED_DOMAIN_SEPARATOR,
        "stream_seeds": frozen.stream_seed_records(authority["campaign_id"], work_unit_id),
        "scientific_evidence": False,
        "merge_eligible": False,
        "test_split_access": frozen.TEST_SPLIT_ACCESS,
        "label": frozen.BOUNDED_SMOKE_LABEL,
    }
    return frozen.validate_work_unit_request(request, execution_class=frozen.EXECUTION_CLASS_BOUNDED_SMOKE)


def _expected_result(
    request: dict[str, Any],
    *,
    trials_completed: int,
    bit_errors: int,
    block_errors: int,
    shard_count: int,
    shard_index: int,
    attempt: int,
) -> dict[str, Any]:
    k = int(request["bler_identity"]["k_and_n"][0])
    derived = frozen.recompute_measurements(
        trials_completed=trials_completed,
        information_bits=trials_completed * k,
        bit_errors=bit_errors,
        block_errors=block_errors,
        information_length=k,
    )
    result = {
        "schema_version": frozen.BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
        "artifact_role": frozen.RESULT_ARTIFACT_ROLE,
        "status": frozen.STATUS_COMPLETE,
        "identity": {
            "execution_class": request["execution_class"],
            "request_sha256": frozen.request_digest(request),
            "campaign_id": request["campaign_id"],
            "bler_tooling_contract_id": request["bler_tooling_contract_id"],
            "bler_tooling_contract_sha256": request["bler_tooling_contract_sha256"],
            "campaign_manifest_sha256": request["campaign_manifest_sha256"],
            "required_bler_artifact_sha256": request["required_bler_artifact_sha256"],
            "selection_policy_sha256": request["selection_policy_sha256"],
            "work_unit_id": request["work_unit_id"],
            "bler_identity": dict(request["bler_identity"]),
            "snr_db": request["snr_db"],
            "source_packet_config_ids": list(request["source_packet_config_ids"]),
            "trials_requested": request["trials_requested"],
            "trial_count_source": request["trial_count_source"],
            "seed_derivation_identity": request["seed_derivation_identity"],
            "seed_domain_separator": request["seed_domain_separator"],
            "stream_seeds": dict(request["stream_seeds"]),
            "implementation": frozen.implementation_binding(),
        },
        "measurement": {
            "trials_completed": trials_completed,
            "information_bits": trials_completed * k,
            "bit_errors": bit_errors,
            "block_errors": block_errors,
            **derived,
            "confidence_interval_method": frozen.CONFIDENCE_INTERVAL_METHOD,
            "confidence_interval_percent": frozen.CONFIDENCE_INTERVAL_PERCENT,
            "confidence_interval_role": frozen.CONFIDENCE_INTERVAL_ROLE,
        },
        "execution_metadata": {
            "wall_time_s": None,
            "hostname": None,
            "device": "cpu",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "attempt": attempt,
        },
        "disposition": {
            "scientific_evidence": False,
            "merge_eligible": False,
            "test_split_access": 0,
            "required_coverage_contribution": 0,
        },
    }
    return frozen.validate_work_unit_result(result, request=request)


def _reject_environmental_fields(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            environmental = any(token in lowered for token in ("timestamp", "hostname", "inode", "mtime"))
            _require(not environmental or child is None, f"environment-specific value is present: {path}.{key}")
            _reject_environmental_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_environmental_fields(child, f"{path}[{index}]")


def verify(
    path: Path = RECORD_PATH,
    *,
    campaign_state_path: Path = CAMPAIGN_STATE,
    runner_contract_path: Path = RUNNER_CONTRACT_PATH,
) -> dict[str, Any]:
    record, raw = _read_json(path, "bounded-smoke record")
    _require(raw == rendered_json(record), "bounded-smoke record is not canonical rendered JSON")
    expected_top = {
        "schema_version", "artifact_role", "label", "campaign_id", "campaign_manifest_sha256",
        "required_bler_artifact_sha256", "selection_policy_sha256", "bler_tooling_contract_id",
        "bler_tooling_contract_sha256", "bler_state_contract_id", "bler_state_contract_sha256",
        "bler_resume_contract_id", "bler_resume_contract_sha256", "bler_runner_contract_id",
        "bler_runner_contract_sha256", "execution_class", "selection_rule", "maximum_work_units",
        "official_work_unit_count", "maximum_trials_per_unit", "selected_work_units", "shard_count",
        "shard_index", "batch_size", "non_scientific", "merge_eligible",
        "required_coverage_contribution", "test_split_access", "production_root_used",
        "temporary_root_removed", "characterization_started", "scientific_execution_performed",
    }
    _require(set(record) == expected_top, "bounded-smoke record fields changed")
    _require(record["schema_version"] == EXPECTED_SCHEMA_VERSION, "bounded-smoke record schema changed")
    _require(record["artifact_role"] == EXPECTED_ROLE, "bounded-smoke record role changed")
    _require(record["label"] == EXPECTED_LABEL, "bounded-smoke label changed")
    _require(record["execution_class"] == frozen.EXECUTION_CLASS_BOUNDED_SMOKE, "record is not bounded smoke")
    _require(record["selection_rule"] == frozen.BOUNDED_SMOKE_SELECTION_RULE, "smoke selection rule changed")
    _require(record["maximum_work_units"] == MAX_UNITS, "smoke work-unit ceiling changed")
    _require(record["official_work_unit_count"] == len(get("baseline.modulations")) == MAX_UNITS, "official smoke count changed")
    _require(record["maximum_trials_per_unit"] == MAX_TRIALS, "smoke trial ceiling changed")
    _require(record["non_scientific"] is True, "smoke is not marked non-scientific")
    _require(record["merge_eligible"] is False, "smoke claims merge eligibility")
    _require(record["required_coverage_contribution"] == 0, "smoke contributes required coverage")
    _require(record["test_split_access"] == 0, "smoke claims test access")
    _require(record["production_root_used"] is False, "smoke used the production root")
    _require(record["temporary_root_removed"] is True, "temporary smoke root was not removed")
    _require(record["characterization_started"] is False, "smoke record starts characterization")
    _require(record["scientific_execution_performed"] is False, "smoke record claims scientific execution")
    _require(type(record["shard_count"]) is int and record["shard_count"] == 1, "official smoke shard count changed")
    _require(record["shard_index"] == 0, "official smoke shard index changed")
    _require(type(record["batch_size"]) is int and record["batch_size"] == 1, "official smoke batch size changed")

    overrides = {
        REQUIRED_ARTIFACT_PATH: REPO_ROOT / REQUIRED_ARTIFACT_PATH,
        "results/baseline/g8/bounded_smoke_record.json": path,
        str(runner_contract_path.relative_to(REPO_ROOT)) if runner_contract_path.is_relative_to(REPO_ROOT) else "results/baseline/g8/bler_runner_contract.json": runner_contract_path,
    }
    state, manifest, required_payload, required_raw = _authenticated_authority(
        campaign_state_path,
        overrides=overrides,
    )
    identity = state["identity"]
    required_binding = next(entry for entry in identity["produced_artifacts"] if entry["path"] == REQUIRED_ARTIFACT_PATH)
    _require(record["campaign_id"] == identity["campaign_id"], "smoke campaign ID mismatch")
    _require(record["campaign_manifest_sha256"] == identity["campaign_manifest_sha256"], "smoke manifest SHA mismatch")
    _require(record["required_bler_artifact_sha256"] == required_binding["sha256"], "smoke required-artifact SHA mismatch")
    _require(record["selection_policy_sha256"] == manifest["selection_policy"]["selection_policy_sha256"], "smoke selection-policy SHA mismatch")
    _require(record["selection_policy_sha256"] == _policy_fingerprint(manifest), "smoke selection-policy fingerprint is not independently reproduced")
    _require(len(required_raw) == required_binding["bytes"] and sha256_bytes(required_raw) == required_binding["sha256"], "required-BLER binding changed")

    record_binding, bound_record_raw = _artifact_binding(
        state,
        "results/baseline/g8/bounded_smoke_record.json",
        "bounded smoke record",
        actual_path=path,
    )
    _require(bound_record_raw == raw and record_binding["sha256"] == sha256_bytes(raw), "smoke record is not the registered artifact")
    tooling_payload, tooling_id, tooling_sha, tooling_bytes = _contract_binding(state, TOOLING_CONTRACT_PATH, "B1C tooling contract", relative_path="results/baseline/g8/bler_tooling_contract.json")
    state_payload, state_id, state_sha, state_bytes = _contract_binding(state, STATE_CONTRACT_PATH, "B2C state contract", relative_path="results/baseline/g8/bler_state_contract.json")
    resume_payload, resume_id, resume_sha, resume_bytes = _contract_binding(state, RESUME_CONTRACT_PATH, "B3 resume contract", relative_path="results/baseline/g8/bler_resume_contract.json")
    runner_payload, runner_id, runner_sha, runner_bytes = _contract_binding(state, runner_contract_path, "B5 runner contract", relative_path="results/baseline/g8/bler_runner_contract.json")
    _require(runner_payload.get("schema_version") == 3, "smoke is not bound to the v3 runner contract")
    _require(
        runner_payload.get("supersedes") == {
            "contract_id": "g8runner-3e4c870966837d255829dbca6afc4d1e3ce5ccf4754618460c939607d9c1c7e5",
            "contract_sha256": "21ec8ae9c3c0787fa0a43bfdc12b4362bd26534a4774ee682070d94449e11268",
            "contract_bytes": 17597,
            "reason": "complete SR-1 literal compliance for infrastructure-only staging-name entropy and provide recoverable registered-smoke rebinding; no scientific or physical-layer semantics changed",
        },
        "runner contract immediate predecessor changed",
    )
    _require(isinstance(runner_payload.get("supersession_history"), list) and len(runner_payload["supersession_history"]) == 2, "runner supersession history is incomplete")
    _require(record["bler_tooling_contract_id"] == tooling_id and record["bler_tooling_contract_sha256"] == tooling_sha, "smoke tooling binding mismatch")
    _require(record["bler_state_contract_id"] == state_id and record["bler_state_contract_sha256"] == state_sha, "smoke state binding mismatch")
    _require(record["bler_resume_contract_id"] == resume_id and record["bler_resume_contract_sha256"] == resume_sha, "smoke resume binding mismatch")
    _require(record["bler_runner_contract_id"] == runner_id and record["bler_runner_contract_sha256"] == runner_sha, "smoke runner binding mismatch")
    _require(tooling_payload.get("contract_id") == tooling_id and state_payload.get("contract_id") == state_id and resume_payload.get("contract_id") == resume_id, "contract IDs do not reproduce")
    _require(runner_bytes == len(runner_contract_path.read_bytes()), "runner contract byte count changed")

    expected_ids = _expected_first_ids(required_payload)
    selected = record["selected_work_units"]
    _require(isinstance(selected, list) and len(selected) == len(expected_ids) == MAX_UNITS, "official smoke selection count changed")
    _require([item.get("work_unit_id") for item in selected] == expected_ids, "smoke selection is not canonical authority order")
    units = {unit["work_unit_id"]: unit for unit in required_payload["required_bler_work_units"]}
    _require(len(units) == len(required_payload["required_bler_work_units"]), "required authority contains duplicate IDs")
    authority = {
        "campaign_id": identity["campaign_id"],
        "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": required_binding["sha256"],
        "selection_policy_sha256": manifest["selection_policy"]["selection_policy_sha256"],
        "bler_tooling_contract_id": tooling_id,
        "bler_tooling_contract_sha256": tooling_sha,
    }
    state_context = work_units.AuthenticatedUnitStateContext(campaign_state_path=campaign_state_path)
    resume_context = resume.AuthenticatedResumeContext(state_context=state_context, require_resume_contract=True)
    shard_plan = work_units.build_shard_plan(state_context, record["shard_count"], record["shard_index"])
    item_fields = {
        "work_unit_id", "attempt", "work_unit_record", "identity_sha256", "seed_records", "request",
        "request_sha256", "result", "result_sha256", "terminal_state", "terminal_state_sha256",
        "trials_requested", "trials_completed", "information_bits", "bit_errors", "block_errors",
        "ber", "bler", "wilson_low", "wilson_high", "classification", "required_coverage_contribution",
        "test_split_access",
    }
    for item, work_unit_id in zip(selected, expected_ids, strict=True):
        _require(isinstance(item, dict) and set(item) == item_fields, f"smoke unit fields changed: {work_unit_id}")
        unit = units[work_unit_id]
        _require(item["attempt"] == 1, f"smoke attempt changed: {work_unit_id}")
        _require(item["work_unit_record"] == unit, f"work-unit authority changed: {work_unit_id}")
        _require(item["identity_sha256"] == sha256_bytes(frozen.canonical_json(unit)), f"identity digest mismatch: {work_unit_id}")
        expected_seed_records = frozen.stream_seed_records(identity["campaign_id"], work_unit_id)
        _require(item["seed_records"] == expected_seed_records, f"seed record mismatch: {work_unit_id}")
        request = _expected_request(authority, work_unit_id, unit)
        _require(item["request"] == request, f"bounded request reconstruction mismatch: {work_unit_id}")
        _require(item["request_sha256"] == sha256_bytes(frozen.canonical_json(request)), f"request digest mismatch: {work_unit_id}")
        result_payload = item["result"]
        _require(isinstance(result_payload, dict), f"result payload is not an object: {work_unit_id}")
        completed = item["trials_completed"]
        k = int(unit["identity"]["k_and_n"][0])
        _require(item["trials_requested"] == MAX_TRIALS == request["trials_requested"], f"smoke trial request changed: {work_unit_id}")
        _require(type(completed) is int and 0 < completed <= MAX_TRIALS, f"invalid completed trials: {work_unit_id}")
        _require(item["information_bits"] == completed * k, f"information-bit count mismatch: {work_unit_id}")
        _require(type(item["bit_errors"]) is int and 0 <= item["bit_errors"] <= item["information_bits"], f"invalid bit errors: {work_unit_id}")
        _require(type(item["block_errors"]) is int and 0 <= item["block_errors"] <= completed, f"invalid block errors: {work_unit_id}")
        expected_result = _expected_result(
            request,
            trials_completed=completed,
            bit_errors=item["bit_errors"],
            block_errors=item["block_errors"],
            shard_count=record["shard_count"],
            shard_index=record["shard_index"],
            attempt=item["attempt"],
        )
        expected_result_projection = {
            key: value for key, value in expected_result.items() if key != "execution_metadata"
        }
        _require(result_payload == expected_result_projection, f"result reconstruction mismatch: {work_unit_id}")
        _require(item["result_sha256"] == sha256_bytes(frozen.canonical_json(expected_result)), f"result digest mismatch: {work_unit_id}")
        derived = frozen.recompute_measurements(
            trials_completed=completed,
            information_bits=item["information_bits"],
            bit_errors=item["bit_errors"],
            block_errors=item["block_errors"],
            information_length=k,
        )
        _require(item["ber"] == derived["ber"] and item["bler"] == derived["bler"], f"smoke rates mismatch: {work_unit_id}")
        _require(item["wilson_low"] == derived["bler_confidence_low"] and item["wilson_high"] == derived["bler_confidence_high"], f"smoke Wilson bounds mismatch: {work_unit_id}")
        _require(item["classification"] == "terminal_nonmergeable", f"smoke unit is mergeable: {work_unit_id}")
        _require(item["required_coverage_contribution"] == 0 and item["test_split_access"] == 0, f"smoke unit contributes evidence: {work_unit_id}")
        expected_state = work_units.build_unit_state(
            state_context,
            work_unit_id,
            shard_plan,
            attempt=item["attempt"],
            status=work_units.STATUS_RESULT_LINKED,
            request_sha256=item["request_sha256"],
            result_path=resume.logical_result_path(resume_context, work_unit_id, item["attempt"]),
            result_sha256=item["result_sha256"],
            scientific_execution_performed=True,
            trials_completed=completed,
            runtime_metadata={
                "hostname": None,
                "process_id": None,
                "device": "cpu",
                "wall_clock_annotation": None,
                "update_annotation": None,
            },
        )
        _require(item["terminal_state"] == {"identity": expected_state["identity"]}, f"terminal state reconstruction mismatch: {work_unit_id}")
        _require(item["terminal_state_sha256"] == sha256_bytes(work_units.canonical_state_bytes(state_context, expected_state)), f"terminal state digest mismatch: {work_unit_id}")

    _require(not work_units.DEFAULT_WORK_UNIT_ROOT.exists(), "production runtime root exists during smoke verification")
    _reject_environmental_fields(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=RECORD_PATH)
    args = parser.parse_args(argv)
    try:
        record = verify(args.path)
    except SmokeVerificationError as exc:
        raise SystemExit(f"G8 bounded-smoke verification HOLD: {exc}") from exc
    print(
        "G8 bounded-smoke verification PASS: "
        f"units={len(record['selected_work_units'])} "
        f"sha256={sha256_bytes(args.path.read_bytes())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
