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
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
)
from config.params import get  # noqa: E402


RECORD_PATH = REPO_ROOT / "results/baseline/g8/bounded_smoke_record.json"
RUNNER_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_runner_contract.json"
STATE_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_state_contract.json"
RESUME_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_resume_contract.json"
MAX_UNITS = 3
MAX_TRIALS = 16
EXPECTED_ROLE = "g8_bounded_smoke_record"
EXPECTED_LABEL = "NON-SCIENTIFIC BOUNDED SMOKE"


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


def _binding(path: Path, label: str) -> tuple[str, str, int]:
    payload, raw = _read_json(path, label)
    contract_id = payload.get("contract_id")
    _require(isinstance(contract_id, str) and contract_id, f"{label} has no contract ID")
    return contract_id, sha256_bytes(raw), len(raw)


def _expected_first_ids() -> list[str]:
    units = frozen.required_work_unit_index()
    selected: list[str] = []
    for modulation in get("baseline.modulations"):
        candidates = [
            unit_id
            for unit_id, unit in units.items()
            if unit["identity"]["modulation"] == modulation
        ]
        _require(candidates, f"no required identity for configured modulation {modulation}")
        selected.append(min(candidates, key=lambda item: units[item]["canonical_ordinal"]))
    return selected


def _reject_environmental_fields(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            _require(
                not any(token in lowered for token in ("timestamp", "hostname", "pid", "inode", "mtime")),
                f"environmental field is present: {path}.{key}",
            )
            _reject_environmental_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_environmental_fields(child, f"{path}[{index}]")


def verify(path: Path = RECORD_PATH) -> dict[str, Any]:
    record, raw = _read_json(path, "bounded-smoke record")
    _require(raw == rendered_json(record), "bounded-smoke record is not canonical rendered JSON")
    expected_top = {
        "schema_version",
        "artifact_role",
        "label",
        "campaign_id",
        "campaign_manifest_sha256",
        "required_bler_artifact_sha256",
        "selection_policy_sha256",
        "bler_tooling_contract_id",
        "bler_tooling_contract_sha256",
        "bler_state_contract_id",
        "bler_state_contract_sha256",
        "bler_resume_contract_id",
        "bler_resume_contract_sha256",
        "bler_runner_contract_id",
        "bler_runner_contract_sha256",
        "execution_class",
        "selection_rule",
        "maximum_work_units",
        "maximum_trials_per_unit",
        "selected_work_units",
        "shard_count",
        "shard_index",
        "batch_size",
        "non_scientific",
        "merge_eligible",
        "required_coverage_contribution",
        "test_split_access",
        "production_root_used",
        "temporary_root_removed",
        "characterization_started",
        "scientific_execution_performed",
    }
    _require(set(record) == expected_top, "bounded-smoke record fields changed")
    _require(record["schema_version"] == 1, "bounded-smoke record schema changed")
    _require(record["artifact_role"] == EXPECTED_ROLE, "bounded-smoke record role changed")
    _require(record["label"] == EXPECTED_LABEL, "bounded-smoke label changed")
    _require(record["execution_class"] == frozen.EXECUTION_CLASS_BOUNDED_SMOKE, "record is not bounded smoke")
    _require(record["selection_rule"] == frozen.BOUNDED_SMOKE_SELECTION_RULE, "smoke selection rule changed")
    _require(record["maximum_work_units"] == MAX_UNITS, "smoke work-unit ceiling changed")
    _require(record["maximum_trials_per_unit"] == MAX_TRIALS, "smoke trial ceiling changed")
    _require(record["non_scientific"] is True, "smoke is not marked non-scientific")
    _require(record["merge_eligible"] is False, "smoke claims merge eligibility")
    _require(record["required_coverage_contribution"] == 0, "smoke contributes required coverage")
    _require(record["test_split_access"] == 0, "smoke claims test access")
    _require(record["production_root_used"] is False, "smoke used the production root")
    _require(record["temporary_root_removed"] is True, "temporary smoke root was not removed")
    _require(record["characterization_started"] is False, "smoke record starts characterization")
    _require(record["scientific_execution_performed"] is False, "smoke record claims scientific execution")
    _require(type(record["shard_count"]) is int and record["shard_count"] > 0, "invalid smoke shard count")
    _require(
        type(record["shard_index"]) is int and 0 <= record["shard_index"] < record["shard_count"],
        "invalid smoke shard index",
    )
    _require(type(record["batch_size"]) is int and record["batch_size"] > 0, "invalid smoke batch size")

    state = load_campaign_state(CAMPAIGN_STATE)
    identity = state["identity"]
    _require(identity["phase"] == "G8_B" and identity["stage"] == "tooling_open", "smoke verifier requires G8_B/tooling_open")
    _require(identity["completed_work_unit_ids"] == [], "smoke changed campaign completed IDs")
    _require(identity["in_progress_work_unit_id"] is None, "smoke changed campaign in-progress ID")
    _require(all(value == 0 for value in identity["counters"].values()), "smoke changed scientific counters")

    runner_id, runner_sha, _runner_bytes = _binding(RUNNER_CONTRACT_PATH, "B4 runner contract")
    state_id, state_sha, _state_bytes = _binding(STATE_CONTRACT_PATH, "B2C state contract")
    resume_id, resume_sha, _resume_bytes = _binding(RESUME_CONTRACT_PATH, "B3 resume contract")
    _require(record["bler_runner_contract_id"] == runner_id, "smoke runner contract ID mismatch")
    _require(record["bler_runner_contract_sha256"] == runner_sha, "smoke runner contract SHA mismatch")
    _require(record["bler_state_contract_id"] == state_id, "smoke state contract ID mismatch")
    _require(record["bler_state_contract_sha256"] == state_sha, "smoke state contract SHA mismatch")
    _require(record["bler_resume_contract_id"] == resume_id, "smoke resume contract ID mismatch")
    _require(record["bler_resume_contract_sha256"] == resume_sha, "smoke resume contract SHA mismatch")

    tooling_id, tooling_sha, _tooling_bytes = _binding(
        REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json", "B1C tooling contract"
    )
    _require(record["bler_tooling_contract_id"] == tooling_id, "smoke tooling contract ID mismatch")
    _require(record["bler_tooling_contract_sha256"] == tooling_sha, "smoke tooling contract SHA mismatch")
    _require(record["campaign_id"] == identity["campaign_id"], "smoke campaign ID mismatch")
    _require(record["campaign_manifest_sha256"] == identity["campaign_manifest_sha256"], "smoke manifest SHA mismatch")
    _require(record["required_bler_artifact_sha256"] == identity["required_bler_artifact_sha256"], "smoke required-artifact SHA mismatch")
    _require(record["selection_policy_sha256"] == identity["selection_policy_sha256"], "smoke selection-policy SHA mismatch")

    selected = record["selected_work_units"]
    _require(isinstance(selected, list), "selected smoke units are not a list")
    expected_ids = _expected_first_ids()
    _require(len(selected) == len(expected_ids) <= MAX_UNITS, "smoke selection count changed")
    _require([item.get("work_unit_id") for item in selected] == expected_ids, "smoke selection is not canonical")
    units = frozen.required_work_unit_index()
    for item, work_unit_id in zip(selected, expected_ids):
        expected_fields = {
            "work_unit_id",
            "identity_sha256",
            "seed_records",
            "request_sha256",
            "result_sha256",
            "terminal_state_sha256",
            "trials_requested",
            "trials_completed",
            "information_bits",
            "bit_errors",
            "block_errors",
            "ber",
            "bler",
            "wilson_low",
            "wilson_high",
            "classification",
            "required_coverage_contribution",
            "test_split_access",
        }
        _require(isinstance(item, dict) and set(item) == expected_fields, f"smoke unit fields changed: {work_unit_id}")
        unit = units[work_unit_id]
        _require(item["identity_sha256"] == sha256_bytes(frozen.canonical_json(unit)), f"identity digest mismatch: {work_unit_id}")
        _require(item["seed_records"] == frozen.stream_seed_records(record["campaign_id"], work_unit_id), f"seed record mismatch: {work_unit_id}")
        for field in ("request_sha256", "result_sha256", "terminal_state_sha256"):
            _require(isinstance(item[field], str) and len(item[field]) == 64, f"{field} is not a SHA-256: {work_unit_id}")
        _require(item["trials_requested"] == MAX_TRIALS, f"smoke trial request changed: {work_unit_id}")
        k = unit["identity"]["k_and_n"][0]
        completed = item["trials_completed"]
        _require(type(completed) is int and 0 < completed <= MAX_TRIALS, f"invalid completed trials: {work_unit_id}")
        _require(item["information_bits"] == completed * k, f"information-bit count mismatch: {work_unit_id}")
        _require(type(item["bit_errors"]) is int and 0 <= item["bit_errors"] <= item["information_bits"], f"invalid bit errors: {work_unit_id}")
        _require(type(item["block_errors"]) is int and 0 <= item["block_errors"] <= completed, f"invalid block errors: {work_unit_id}")
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
