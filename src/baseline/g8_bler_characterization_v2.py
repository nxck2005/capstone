"""G8_C epoch-2 orchestration, provenance, and measured BLER-table loading.

This module is the C-owned layer around the authenticated B1C/B2C/B3 and v3
runner contracts.  It does not implement the physical-layer simulation.  The
runner remains the only production measurement path; this module validates
and assembles the evidence it publishes.

The source-manifest and table readers are deliberately fail-closed.  A table
point is accepted only when its request, result, and terminal unit state are
all authenticated against the frozen authority and the separately frozen
source/merge artifacts.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline import g8_bler_contract as bler_contract
from baseline import g8_bler_resume as resume
from baseline import g8_bler_runner as runner
from baseline import g8_bler_work_units as work_units
from baseline.classical import composition
from baseline.g8_campaign import (
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    G8ContractError,
    REPO_ROOT,
    canonical_json,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
    validate_campaign_state,
    validate_state_transition,
    write_campaign_state_atomically,
)


SOURCE_MANIFEST_RELATIVE_PATH = "results/baseline/g8/bler_characterization_source_manifest_v2.json"
MERGE_REPORT_RELATIVE_PATH = "results/baseline/g8/bler_merge_report_v2.json"
TABLE_RELATIVE_PATH = "results/baseline/g8/bler_table_v2.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / SOURCE_MANIFEST_RELATIVE_PATH
MERGE_REPORT_PATH = REPO_ROOT / MERGE_REPORT_RELATIVE_PATH
TABLE_PATH = REPO_ROOT / TABLE_RELATIVE_PATH

SOURCE_MANIFEST_SCHEMA_VERSION = 2
SOURCE_MANIFEST_ARTIFACT_ROLE = "g8_bler_characterization_source_manifest_v2"
SOURCE_MANIFEST_ID_PREFIX = "g8charsrc2"
MERGE_REPORT_SCHEMA_VERSION = 2
MERGE_REPORT_ARTIFACT_ROLE = "g8_bler_merge_report_v2"
MERGE_REPORT_ID_PREFIX = "g8merge2"
TABLE_SCHEMA_VERSION = 2
TABLE_ARTIFACT_ROLE = "g8_bler_table_v2"
TABLE_ID_PREFIX = "g8blertable2"

PHASE = "G8_C"
OPEN_STAGE = "characterization_open"
COMPLETE_STAGE = "characterization_complete"
EXECUTION_CLASS = bler_contract.EXECUTION_CLASS_FULL_STRENGTH
LOGICAL_ROOT = "results/baseline/g8/work_units"
REQUIRED_COUNT = 3213  # literal-ok: authenticated G8_A required-work-unit count
FULL_STRENGTH_TRIALS = 5000  # literal-ok: authenticated G8_A full-strength trial count
G8_D_RESTART_COMMAND = (
    'rg -n "G8_D|measurement_tooling_open|codec|reconstruction|BR-11|clean_classifier|'
    'bler_table|characterization_complete" src/baseline tools tests instructions'
)
G8_C_RESTART_COMMAND = (
    ".venv/bin/python tools/run_g8_bler_characterization_v2.py --root "
    "/home/nick/projects/capstone/results/baseline/g8/work_units --device auto "
    "--shard-count auto --batch-size auto --max-units-per-worker-batch 128"
)

PREDECESSOR_MANIFEST_RELATIVE_PATH = "results/baseline/g8/bler_characterization_source_manifest.json"
PREDECESSOR_MANIFEST_ID = "g8charsrc-6926319673ca1f55b95f8746062518c12cfa499aa827448e67850b5a1f74702a"
PREDECESSOR_MANIFEST_SHA256 = "a917f839f945232e85852d6d27f02de4b5dc272adc72b1966a95e9b5e62a014e"
PREDECESSOR_MANIFEST_BYTES = 6672  # literal-ok: immutable epoch-1 manifest byte count
EPOCH_1_START_ORDINAL = 0
EPOCH_1_END_ORDINAL = 178
EPOCH_1_ACCEPTED_COUNT = 179
EPOCH_2_START_ORDINAL = 179
EPOCH_2_END_ORDINAL = 3212
EPOCH_2_FIRST_WORK_UNIT = "bler-0e7c3102fbf553ba90fc5458"
EPOCH_2_FIRST_LEGAL_ATTEMPT = 3
V1_SOURCE_PATHS = (
    "src/baseline/g8_bler_characterization.py",
    "tools/run_g8_bler_characterization.py",
    "tools/gen_g8_bler_characterization_manifest.py",
    "tools/verify_g8_bler_characterization_manifest.py",
    "tools/merge_g8_bler_characterization.py",
    "tools/verify_g8_bler_table.py",
)

CHARACTERIZATION_SOURCE_PATHS = (
    "src/baseline/g8_bler_characterization_v2.py",
    "tools/run_g8_bler_characterization_v2.py",
    "tools/gen_g8_bler_characterization_manifest_v2.py",
    "tools/verify_g8_bler_characterization_manifest_v2.py",
    "tools/merge_g8_bler_characterization_v2.py",
    "tools/verify_g8_bler_table_v2.py",
)
CHARACTERIZATION_DEPENDENCY_PATHS = (
    "src/baseline/classical/composition.py",
    "results/baseline/g8/bler_tooling_contract.json",
    "results/baseline/g8/bler_state_contract.json",
    "results/baseline/g8/bler_resume_contract.json",
    "results/baseline/g8/bler_runner_contract.json",
)

SOURCE_MANIFEST_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "manifest_id",
    "epoch",
    "predecessor",
    "activation_boundary",
    "source_epochs",
    "scientific_execution_performed",
    "characterization_started",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "required_work_unit_count",
    "full_strength_trials",
    "request_schema_version",
    "result_schema_version",
    "unit_state_schema_version",
    "bler_tooling_contract_id",
    "bler_tooling_contract_sha256",
    "bler_state_contract_id",
    "bler_state_contract_sha256",
    "bler_resume_contract_id",
    "bler_resume_contract_sha256",
    "bler_runner_contract_id",
    "bler_runner_contract_sha256",
    "seed_derivation_identity",
    "runtime",
    "execution",
    "retry_policy",
    "count_semantics",
    "merge_completeness_predicate",
    "table_schema",
    "handoff",
    "merge_attribution",
    "table_attribution",
    "sources",
    "dependencies",
)

MERGE_REPORT_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "report_id",
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
    "source_manifest_id",
    "source_manifest_sha256",
    "source_epochs",
    "required_work_unit_count",
    "required_work_unit_ids",
    "units",
    "completed_count",
    "missing_count",
    "duplicate_count",
    "unknown_count",
    "recoverable_count",
    "failed_count",
    "terminal_nonmergeable_count",
    "coverage_contribution_sum",
    "total_trials",
    "total_information_bits",
    "total_bit_errors",
    "total_block_errors",
    "coverage_complete",
    "interpolation_used",
    "extrapolation_used",
    "test_split_access",
    "request_only_attempt_count",
    "failed_result_attempt_count",
)

TABLE_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "table_id",
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
    "source_manifest_id",
    "source_manifest_sha256",
    "merge_report_id",
    "merge_report_sha256",
    "required_work_unit_count",
    "complete_identity_count",
    "measured_point_count",
    "trials_per_point",
    "total_trials",
    "interpolation_used",
    "extrapolation_used",
    "test_split_access",
    "curves",
)

TABLE_POINT_FIELDS = (
    "work_unit_id",
    "snr_db",
    "trials",
    "information_bits",
    "bit_errors",
    "block_errors",
    "ber",
    "bler",
    "bler_confidence_low",
    "bler_confidence_high",
    "request_sha256",
    "result_sha256",
    "state_sha256",
)


class CharacterizationError(RuntimeError):
    """A G8_C artifact, state, or orchestration invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CharacterizationError(message)


def _strict_object(value: Any, fields: Sequence[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(set(value) == set(fields), f"{label} has missing or unexpected fields")
    return dict(value)


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} is not a SHA-256 digest")  # literal-ok: SHA-256 hex width
    _require(all(char in "0123456789abcdef" for char in value), f"{label} is not lowercase hex")
    return value


def _binding(path: str, role: str) -> dict[str, Any]:
    target = REPO_ROOT / path
    raw = target.read_bytes()
    return {"path": path, "role": role, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _registered_binding(state: Mapping[str, Any], relative_path: str) -> dict[str, Any]:
    artifacts = state["identity"]["produced_artifacts"]
    matches = [entry for entry in artifacts if entry.get("path") == relative_path]
    _require(len(matches) == 1, f"campaign state does not register exactly one {relative_path}")
    binding = dict(matches[0])
    _require(set(binding) == {"path", "sha256", "bytes"}, f"invalid binding for {relative_path}")
    body = (REPO_ROOT / relative_path).read_bytes()
    _require(binding["bytes"] == len(body), f"registered byte count changed for {relative_path}")
    _require(binding["sha256"] == sha256_bytes(body), f"registered digest changed for {relative_path}")
    return binding


def _read_rendered_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CharacterizationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw


def _self_excluding_id(payload: Mapping[str, Any], field: str, prefix: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return f"{prefix}-{sha256_bytes(canonical_json(body))}"


def _contract_binding(path: str, id_field: str = "contract_id") -> tuple[str, str, int]:
    payload, raw = _read_rendered_json(REPO_ROOT / path, path)
    identifier = payload.get(id_field)
    _require(isinstance(identifier, str) and identifier, f"{path} has no contract ID")
    return identifier, sha256_bytes(raw), len(raw)


def _authority_from_files() -> dict[str, Any]:
    campaign = json.loads((REPO_ROOT / "results/baseline/g8/campaign_manifest.json").read_bytes())
    required = json.loads((REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes())
    tooling, tooling_raw = _read_rendered_json(
        REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json", "B1C tooling contract"
    )
    state_contract, state_raw = _read_rendered_json(
        REPO_ROOT / "results/baseline/g8/bler_state_contract.json", "B2C state contract"
    )
    resume_contract, resume_raw = _read_rendered_json(
        REPO_ROOT / "results/baseline/g8/bler_resume_contract.json", "B3 resume contract"
    )
    runner_contract, runner_raw = _read_rendered_json(
        REPO_ROOT / "results/baseline/g8/bler_runner_contract.json", "v3 runner contract"
    )
    _require(campaign["campaign_id"] == bler_contract.campaign_bindings()["campaign_id"], "campaign ID drift")
    work_units_list = required["required_bler_work_units"]
    _require(len(work_units_list) == REQUIRED_COUNT, "required work-unit count drift")
    _require(tooling["campaign_bindings"]["required_work_unit_count"] == REQUIRED_COUNT, "B1C count drift")
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/campaign_manifest.json"),
        "required_bler_artifact_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/required_bler_identities.json"),
        "selection_policy_sha256": campaign["selection_policy"]["selection_policy_sha256"],
        "required_work_unit_count": len(work_units_list),
        "full_strength_trials": bler_contract.full_strength_trial_count(),
        "request_schema_version": tooling["request_schema"]["version"],
        "result_schema_version": tooling["result_schema"]["version"],
        "unit_state_schema_version": state_contract["unit_state_schema"]["schema_version"],
        "bler_tooling_contract_id": tooling["contract_id"],
        "bler_tooling_contract_sha256": sha256_bytes(tooling_raw),
        "bler_state_contract_id": state_contract["contract_id"],
        "bler_state_contract_sha256": sha256_bytes(state_raw),
        "bler_resume_contract_id": resume_contract["contract_id"],
        "bler_resume_contract_sha256": sha256_bytes(resume_raw),
        "bler_runner_contract_id": runner_contract["contract_id"],
        "bler_runner_contract_sha256": sha256_bytes(runner_raw),
        "seed_derivation_identity": bler_contract.SEED_DERIVATION_IDENTITY,
    }


def _activation_boundary_from_live_files() -> dict[str, Any]:
    """Authenticate the exact pre-epoch-2 boundary from the real raw root.

    This is deliberately a read-only proof.  It never repairs or reconciles the
    live campaign state and it does not treat request-only attempts as failed
    results.  The v2 manifest is generated only after this proof succeeds.
    """

    required = json.loads((REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes())
    required_ids = [entry["work_unit_id"] for entry in required["required_bler_work_units"]]
    _require(len(required_ids) == REQUIRED_COUNT, "activation boundary required-ID count drift")
    _require(required_ids[EPOCH_2_START_ORDINAL] == EPOCH_2_FIRST_WORK_UNIT, "activation boundary first unit drift")
    state = _manifest_build_state()
    identity = state["identity"]
    _require(identity["phase"] == PHASE and identity["stage"] == OPEN_STAGE, "activation boundary is not G8_C/open")
    _require(identity["completed_work_unit_ids"] == required_ids[:EPOCH_1_ACCEPTED_COUNT], "epoch-1 accepted IDs are not the exact authority prefix")
    _require(identity["in_progress_work_unit_id"] is None, "activation boundary has an in-progress unit")
    _require(not any(identity["counters"].values()), "activation boundary has a protected counter")

    context = resume.AuthenticatedResumeContext(require_resume_contract=True)
    root = validate_production_root(Path("/home/nick/projects/capstone/results/baseline/g8/work_units"))
    inspection = resume.inspect_runtime_root(
        context,
        root=root,
        scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        repair_mode=resume.REPAIR_MODE_READ_ONLY,
    )
    record = inspection["classifications"][EPOCH_2_START_ORDINAL]
    _require(record["canonical_ordinal"] == EPOCH_2_START_ORDINAL, "activation boundary ordinal drift")
    _require(record["work_unit_id"] == EPOCH_2_FIRST_WORK_UNIT, "activation boundary work-unit drift")
    _require(record["classification"] == resume.CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED, "ordinal 179 is not request-only history")
    _require(record["attempt"] == EPOCH_2_FIRST_LEGAL_ATTEMPT - 1, "ordinal 179 historical attempt drift")
    _require(record["proposed_attempt"] == EPOCH_2_FIRST_LEGAL_ATTEMPT, "ordinal 179 next-attempt drift")
    _require(inspection["census"]["request_attempts"][EPOCH_2_FIRST_WORK_UNIT] == [1, 2], "ordinal 179 request history drift")
    _require(inspection["census"]["result_attempts"].get(EPOCH_2_FIRST_WORK_UNIT, []) == [], "ordinal 179 has an unexpected result")
    _require(record["request_sha256"] == "42062b62a4f88e08193b00fee25ed998ccbcd502782a206f231d10aae4c1b1c6", "ordinal 179 request SHA drift")
    _require(record["state_sha256"] == "e4b90d82fdc5760bd6298e82650e266ad759e2271e807be57a258bc6678b9361", "ordinal 179 state SHA drift")
    _require(inspection["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, "activation boundary claims test access")
    return {
        "epoch_1_accepted_result_start_ordinal": EPOCH_1_START_ORDINAL,
        "epoch_1_accepted_result_end_ordinal": EPOCH_1_END_ORDINAL,
        "epoch_1_accepted_result_count": EPOCH_1_ACCEPTED_COUNT,
        "first_possible_epoch_2_accepted_ordinal": EPOCH_2_START_ORDINAL,
        "first_epoch_2_work_unit_id": EPOCH_2_FIRST_WORK_UNIT,
        "first_legal_epoch_2_attempt": EPOCH_2_FIRST_LEGAL_ATTEMPT,
        "ordinal_179_request_attempts": [1, 2],
        "ordinal_179_result_attempts": [],
        "ordinal_179_request_sha256": record["request_sha256"],
        "ordinal_179_state_sha256": record["state_sha256"],
        "protected_counters": {field: 0 for field in COUNTER_FIELDS},
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }


def _predecessor_binding() -> dict[str, Any]:
    path = REPO_ROOT / PREDECESSOR_MANIFEST_RELATIVE_PATH
    raw = path.read_bytes()
    _require(len(raw) == PREDECESSOR_MANIFEST_BYTES, "epoch-1 manifest byte count drift")
    _require(sha256_bytes(raw) == PREDECESSOR_MANIFEST_SHA256, "epoch-1 manifest SHA drift")
    predecessor = json.loads(raw)
    _require(predecessor.get("manifest_id") == PREDECESSOR_MANIFEST_ID, "epoch-1 manifest ID drift")
    return {
        "path": PREDECESSOR_MANIFEST_RELATIVE_PATH,
        "manifest_id": PREDECESSOR_MANIFEST_ID,
        "sha256": PREDECESSOR_MANIFEST_SHA256,
        "bytes": PREDECESSOR_MANIFEST_BYTES,
    }


def _manifest_build_state() -> dict[str, Any]:
    """Read the pre-science state, projecting only a stale v2 binding.

    A source edit after a provisional, uncommitted registration leaves the
    old state binding unable to pass the normal campaign-state loader.  The
    projection is allowed only at the exact 179-unit boundary and is fully
    validated before the new manifest is built; after epoch-2 execution the
    guarded registration path rejects the same situation.
    """

    try:
        return load_campaign_state()
    except G8ContractError:
        raw = CAMPAIGN_STATE.read_bytes()
        state = json.loads(raw)
        _require(raw == rendered_json(state), "campaign state is not canonical JSON")
        identity = state["identity"]
        _require(identity["phase"] == PHASE and identity["stage"] == OPEN_STAGE, "stale binding is not a pre-science G8_C state")
        _require(len(identity["completed_work_unit_ids"]) == EPOCH_1_ACCEPTED_COUNT, "stale binding is not at the epoch-2 boundary")
        _require(identity["in_progress_work_unit_id"] is None and not any(identity["counters"].values()), "stale binding has live scientific work")
        binding = {
            "path": SOURCE_MANIFEST_RELATIVE_PATH,
            "sha256": sha256_file(SOURCE_MANIFEST_PATH),
            "bytes": len(SOURCE_MANIFEST_PATH.read_bytes()),
        }
        matches = [entry for entry in identity["produced_artifacts"] if entry["path"] == SOURCE_MANIFEST_RELATIVE_PATH]
        _require(len(matches) == 1 and matches[0] != binding, "campaign state failed for a reason other than the provisional v2 binding")
        projected = copy.deepcopy(state)
        projected["identity"]["produced_artifacts"] = [entry for entry in identity["produced_artifacts"] if entry["path"] != SOURCE_MANIFEST_RELATIVE_PATH]
        projected["identity"]["produced_artifacts"].append(binding)
        projected["identity"]["produced_artifacts"].sort(key=lambda item: item["path"])
        validate_campaign_state(projected)
        return projected


def build_source_manifest(*, activation_boundary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the additive epoch-2 manifest at the authenticated 179-unit boundary."""

    authority = _authority_from_files()
    state = _manifest_build_state()
    identity = state["identity"]
    _require(identity["phase"] == PHASE and identity["stage"] == OPEN_STAGE, "C1 requires G8_C/characterization_open")
    _require(len(identity["completed_work_unit_ids"]) == EPOCH_1_ACCEPTED_COUNT, "epoch-2 activation requires 179 accepted epoch-1 units")
    _require(identity["in_progress_work_unit_id"] is None, "C1 cannot run with an in-progress unit")
    _require(not any(identity["counters"].values()), "C1 requires zero protected counters")
    activation = dict(_activation_boundary_from_live_files() if activation_boundary is None else activation_boundary)

    sources = [_binding(path, "g8_c_characterization_source") for path in CHARACTERIZATION_SOURCE_PATHS]
    dependencies = [_binding(path, "g8_c_frozen_dependency") for path in CHARACTERIZATION_DEPENDENCY_PATHS]
    payload: dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "artifact_role": SOURCE_MANIFEST_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": "C1",
        "manifest_id": None,
        "epoch": 2,
        "predecessor": _predecessor_binding(),
        "activation_boundary": activation,
        "source_epochs": [
            {
                "epoch": 1,
                "accepted_result_ordinals": [EPOCH_1_START_ORDINAL, EPOCH_1_END_ORDINAL],
                "accepted_result_count": EPOCH_1_ACCEPTED_COUNT,
                "manifest_path": PREDECESSOR_MANIFEST_RELATIVE_PATH,
                "manifest_id": PREDECESSOR_MANIFEST_ID,
                "manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
            },
            {
                "epoch": 2,
                "accepted_result_ordinals": [EPOCH_2_START_ORDINAL, EPOCH_2_END_ORDINAL],
                "accepted_result_count": EPOCH_2_END_ORDINAL - EPOCH_2_START_ORDINAL + 1,
                "manifest_path": SOURCE_MANIFEST_RELATIVE_PATH,
                "manifest_id": None,
                "manifest_sha256": None,
            },
        ],
        "scientific_execution_performed": False,
        "characterization_started": False,
        **authority,
        "runtime": {
            "logical_root": LOGICAL_ROOT,
            "absolute_paths_bound": False,
        },
        "execution": {
            "execution_class": EXECUTION_CLASS,
            "required_ordering": "required_bler_identities.json required_bler_work_units array order",
            "sharding_algorithm": work_units.SHARDING_ALGORITHM,
            "shard_formula": work_units.SHARD_FORMULA,
            "default_max_units_per_worker_batch": 128,  # literal-ok: bounded durable batch default
            "one_worker_per_supported_physical_gpu": True,
            "device_and_batch_size_are_provenance_only": True,
            "request_bytes_are_identity_across_attempts": True,
        },
        "retry_policy": {
            "failed_attempts_preserved": True,
            "next_attempt_is_clean": True,
            "no_mid_unit_resume": True,
            "retry_only_frozen_b3_next_attempt": True,
            "recoverable_repairs_only": [
                resume.CLASSIFICATION_RECOVERABLE_FAILED_RESULT,
                resume.CLASSIFICATION_RECOVERABLE_COMPLETE_RESULT,
            ],
        },
        "count_semantics": {
            "information_bits": "trials_completed * K",
            "bit_errors": "complete K-vector Hamming distance sum",
            "block_errors": "one for any differing K-bit vector",
            "ber": "bit_errors / information_bits",
            "bler": "block_errors / trials_completed",
            "confidence_interval": bler_contract.CONFIDENCE_INTERVAL_METHOD,
            "confidence_interval_role": "diagnostic_only",
        },
        "merge_completeness_predicate": {
            "required_ids": REQUIRED_COUNT,
            "one_accepted_terminal_result_per_id": True,
            "trials_completed_per_accepted_result": FULL_STRENGTH_TRIALS,
            "coverage_contribution_per_accepted_result": 1,
            "sum_coverage_contribution": REQUIRED_COUNT,
            "unknown_duplicate_missing_recoverable_failed_terminal_nonmergeable": 0,
            "interpolation_used": False,
            "extrapolation_used": False,
            "test_split_access": 0,
        },
        "table_schema": {
            "schema_version": TABLE_SCHEMA_VERSION,
            "artifact_role": TABLE_ARTIFACT_ROLE,
            "id_field": "table_id",
            "id_rule": "sha256(canonical JSON over complete table content excluding table_id)",
            "curves_in": "complete BlerIdentity canonical order",
            "points_in": "ascending exact measured SNR order",
            "construction": "accepted merge-report points only",
            "interpolation_during_construction": False,
            "extrapolation": False,
        },
        "handoff": {
            "next_phase": "G8_D",
            "next_stage": "measurement_tooling_open",
            "exact_restart_command": G8_D_RESTART_COMMAND,
            "g8_d_execution": False,
        },
        "merge_attribution": {
            "rule": "accepted authority ordinal 0..178 maps only to epoch 1; 179..3212 maps only to epoch 2",
            "request_only_attempts_are_not_failed_results": True,
            "one_final_complete_result_per_required_id": True,
            "no_overlap": True,
            "no_gap": True,
        },
        "table_attribution": {
            "rule": "measured points are projected only from epoch-aware accepted merge units",
            "interpolation_during_construction": False,
            "extrapolation": False,
            "measured_zero_bler_preserved": True,
        },
        "sources": sources,
        "dependencies": dependencies,
    }
    payload["manifest_id"] = _self_excluding_id(payload, "manifest_id", SOURCE_MANIFEST_ID_PREFIX)
    return payload


def _expected_activation_boundary() -> dict[str, Any]:
    return {
        "epoch_1_accepted_result_start_ordinal": EPOCH_1_START_ORDINAL,
        "epoch_1_accepted_result_end_ordinal": EPOCH_1_END_ORDINAL,
        "epoch_1_accepted_result_count": EPOCH_1_ACCEPTED_COUNT,
        "first_possible_epoch_2_accepted_ordinal": EPOCH_2_START_ORDINAL,
        "first_epoch_2_work_unit_id": EPOCH_2_FIRST_WORK_UNIT,
        "first_legal_epoch_2_attempt": EPOCH_2_FIRST_LEGAL_ATTEMPT,
        "ordinal_179_request_attempts": [1, 2],
        "ordinal_179_result_attempts": [],
        "ordinal_179_request_sha256": "42062b62a4f88e08193b00fee25ed998ccbcd502782a206f231d10aae4c1b1c6",
        "ordinal_179_state_sha256": "e4b90d82fdc5760bd6298e82650e266ad759e2271e807be57a258bc6678b9361",
        "protected_counters": {field: 0 for field in COUNTER_FIELDS},
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }


def _validate_epoch_chain(value: Mapping[str, Any]) -> None:
    predecessor = _predecessor_binding()
    _require(value["predecessor"] == predecessor, "epoch-1 predecessor binding drift")
    _require(value["epoch"] == 2, "source manifest epoch drift")
    _require(value["activation_boundary"] == _expected_activation_boundary(), "epoch-2 activation boundary drift")
    _require(
        value["source_epochs"]
        == [
            {
                "epoch": 1,
                "accepted_result_ordinals": [EPOCH_1_START_ORDINAL, EPOCH_1_END_ORDINAL],
                "accepted_result_count": EPOCH_1_ACCEPTED_COUNT,
                "manifest_path": PREDECESSOR_MANIFEST_RELATIVE_PATH,
                "manifest_id": PREDECESSOR_MANIFEST_ID,
                "manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
            },
            {
                "epoch": 2,
                "accepted_result_ordinals": [EPOCH_2_START_ORDINAL, EPOCH_2_END_ORDINAL],
                "accepted_result_count": EPOCH_2_END_ORDINAL - EPOCH_2_START_ORDINAL + 1,
                "manifest_path": SOURCE_MANIFEST_RELATIVE_PATH,
                "manifest_id": None,
                "manifest_sha256": None,
            },
        ],
        "source epoch chain is not the frozen contiguous two-epoch chain",
    )
    _require(value["merge_attribution"]["request_only_attempts_are_not_failed_results"] is True, "merge request-only attribution drift")
    _require(value["merge_attribution"]["no_overlap"] is True and value["merge_attribution"]["no_gap"] is True, "source epoch range policy drift")
    _require(value["table_attribution"]["interpolation_during_construction"] is False, "table interpolation policy drift")
    _require(value["table_attribution"]["extrapolation"] is False, "table extrapolation policy drift")


def _validate_predecessor_sources() -> None:
    predecessor, _raw = _read_rendered_json(
        REPO_ROOT / PREDECESSOR_MANIFEST_RELATIVE_PATH, "epoch-1 source manifest"
    )
    for field, paths, role in (
        ("sources", V1_SOURCE_PATHS, "g8_c_characterization_source"),
        ("dependencies", CHARACTERIZATION_DEPENDENCY_PATHS, "g8_c_frozen_dependency"),
    ):
        entries = predecessor.get(field)
        _require(isinstance(entries, list), f"epoch-1 {field} are not a list")
        _require([entry.get("path") for entry in entries] == list(paths), f"epoch-1 {field} order drift")
        for entry, path in zip(entries, paths, strict=True):
            _require(entry == _binding(path, role), f"epoch-1 {field} changed: {path}")


def validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    require_registered: bool = False,
    state_path: Path = CAMPAIGN_STATE,
    state_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate epoch-2 provenance against current bytes and post-data state."""

    value = _strict_object(payload, SOURCE_MANIFEST_FIELDS, "source manifest")
    _require(value["schema_version"] == SOURCE_MANIFEST_SCHEMA_VERSION, "source manifest schema drift")
    _require(value["artifact_role"] == SOURCE_MANIFEST_ARTIFACT_ROLE, "source manifest role drift")
    _require(value["phase"] == PHASE and value["checkpoint"] == "C1", "source manifest phase/checkpoint drift")
    _require(value["scientific_execution_performed"] is False, "source manifest claims scientific execution")
    _require(value["characterization_started"] is False, "source manifest claims characterization started")
    _require(value["manifest_id"] == _self_excluding_id(value, "manifest_id", SOURCE_MANIFEST_ID_PREFIX), "source manifest ID does not reproduce")
    _validate_epoch_chain(value)
    _validate_predecessor_sources()
    authority = _authority_from_files()
    for field, expected in authority.items():
        _require(value[field] == expected, f"source manifest authority drift: {field}")
    _require(value["runtime"]["logical_root"] == LOGICAL_ROOT, "source manifest logical root drift")
    _require(value["runtime"]["absolute_paths_bound"] is False, "source manifest binds an absolute path")
    _require(value["execution"]["execution_class"] == EXECUTION_CLASS, "source manifest execution class drift")
    _require(value["execution"]["sharding_algorithm"] == work_units.SHARDING_ALGORITHM, "source manifest sharding drift")
    _require(value["execution"]["shard_formula"] == work_units.SHARD_FORMULA, "source manifest shard formula drift")
    _require(value["retry_policy"]["no_mid_unit_resume"] is True, "source manifest permits mid-unit resume")
    _require(value["merge_completeness_predicate"]["interpolation_used"] is False, "source manifest permits interpolation")
    _require(value["merge_completeness_predicate"]["extrapolation_used"] is False, "source manifest permits extrapolation")
    _require(value["table_schema"]["interpolation_during_construction"] is False, "table construction interpolation is enabled")
    _require(value["table_schema"]["extrapolation"] is False, "table extrapolation is enabled")
    _require(value["handoff"]["exact_restart_command"] == G8_D_RESTART_COMMAND, "G8_D handoff command drift")
    for field_name, paths, role in (
        ("sources", CHARACTERIZATION_SOURCE_PATHS, "g8_c_characterization_source"),
        ("dependencies", CHARACTERIZATION_DEPENDENCY_PATHS, "g8_c_frozen_dependency"),
    ):
        entries = value[field_name]
        _require([entry.get("path") for entry in entries] == list(paths), f"source order drift: {field_name}")
        for entry, path in zip(entries, paths, strict=True):
            _require(entry["role"] == role, f"source role drift: {path}")
            current = _binding(path, role)
            _require(entry == current, f"bound C1 source changed: {path}")
    state = load_campaign_state(state_path) if state_payload is None else dict(state_payload)
    identity = state["identity"]
    _require(identity["phase"] == PHASE and identity["stage"] in {OPEN_STAGE, COMPLETE_STAGE}, "campaign is outside G8_C")
    required_ids = [entry["work_unit_id"] for entry in json.loads((REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes())["required_bler_work_units"]]
    completed = identity["completed_work_unit_ids"]
    _require(completed == required_ids[: len(completed)], "campaign completed IDs are not an authority prefix")
    _require(len(completed) >= EPOCH_1_ACCEPTED_COUNT, "epoch-2 manifest predates its activation boundary")
    _require(identity["in_progress_work_unit_id"] is None, "campaign has an in-progress unit")
    _require(not any(identity["counters"].values()), "protected campaign counter changed")
    _require(value["activation_boundary"]["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, "activation boundary claims test access")
    if require_registered:
        _registered_binding(state, SOURCE_MANIFEST_RELATIVE_PATH)
    return value


def register_source_manifest(*, state_path: Path = CAMPAIGN_STATE) -> str:
    """Append the epoch-2 source manifest without changing scientific state."""

    payload, raw = _read_rendered_json(SOURCE_MANIFEST_PATH, "G8_C source manifest")
    raw_state_bytes = state_path.read_bytes()
    raw_state = json.loads(raw_state_bytes)
    _require(raw_state_bytes == rendered_json(raw_state), "campaign state is not canonical JSON")
    raw_artifacts = raw_state["identity"]["produced_artifacts"]
    raw_existing = [entry for entry in raw_artifacts if entry["path"] == SOURCE_MANIFEST_RELATIVE_PATH]
    binding = {
        "path": SOURCE_MANIFEST_RELATIVE_PATH,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
    }
    rebind_predata = bool(raw_existing and raw_existing != [binding])
    if rebind_predata:
        _require(len(raw_state["identity"]["completed_work_unit_ids"]) == EPOCH_1_ACCEPTED_COUNT, "cannot rebind epoch-2 source manifest after scientific execution")
        _require(raw_state["identity"]["in_progress_work_unit_id"] is None, "cannot rebind epoch-2 source manifest with in-progress work")
        _require(not any(raw_state["identity"]["counters"].values()), "cannot rebind epoch-2 source manifest with protected counters")
        projected = copy.deepcopy(raw_state)
        projected["identity"]["produced_artifacts"] = [entry for entry in raw_artifacts if entry["path"] != SOURCE_MANIFEST_RELATIVE_PATH]
        projected["identity"]["produced_artifacts"].append(binding)
        projected["identity"]["produced_artifacts"].sort(key=lambda item: item["path"])
        validate_campaign_state(projected)
        previous = projected
    else:
        previous = load_campaign_state(state_path)
    validate_source_manifest(payload, state_payload=previous)
    old = previous["identity"]
    _require(old["phase"] == PHASE and old["stage"] == OPEN_STAGE, "source registration requires G8_C/open")
    _require(len(old["completed_work_unit_ids"]) == EPOCH_1_ACCEPTED_COUNT, "epoch-2 source registration requires 179 accepted epoch-1 units")
    required_ids = [entry["work_unit_id"] for entry in json.loads((REPO_ROOT / "results/baseline/g8/required_bler_identities.json").read_bytes())["required_bler_work_units"]]
    _require(old["completed_work_unit_ids"] == required_ids[:EPOCH_1_ACCEPTED_COUNT], "source registration requires the exact epoch-1 authority prefix")
    _require(old["in_progress_work_unit_id"] is None, "source registration requires no in-progress unit")
    _require(not any(old["counters"].values()), "source registration requires zero counters")
    current = copy.deepcopy(previous)
    artifacts = current["identity"]["produced_artifacts"]
    existing = [entry for entry in artifacts if entry["path"] == SOURCE_MANIFEST_RELATIVE_PATH]
    if existing:
        if existing != [binding] and not rebind_predata:
            # The first implementation pass may be corrected before any
            # epoch-2 result exists.  Once the first corrected result is
            # accepted, a changed source must fail closed rather than being
            # silently rebound under the same epoch.
            _require(
                len(old["completed_work_unit_ids"]) == EPOCH_1_ACCEPTED_COUNT
                and old["in_progress_work_unit_id"] is None
                and not any(old["counters"].values()),
                "cannot rebind epoch-2 source manifest after scientific execution",
            )
            artifacts[:] = [
                entry for entry in artifacts if entry["path"] != SOURCE_MANIFEST_RELATIVE_PATH
            ]
            artifacts.append(binding)
    else:
        artifacts.append(binding)
    artifacts.sort(key=lambda item: item["path"])
    current["identity"]["restart_command"] = G8_C_RESTART_COMMAND
    validate_state_transition(previous, current)
    digest = write_campaign_state_atomically(state_path, current)
    installed = load_campaign_state(state_path)
    _require(installed["identity"]["produced_artifacts"] == artifacts, "source registration did not install exact artifact list")
    return digest


def complete_characterization(*, state_path: Path = CAMPAIGN_STATE) -> str:
    """Advance the authenticated campaign from open to the G8_D handoff."""

    previous = load_campaign_state(state_path)
    old = previous["identity"]
    _require(old["phase"] == PHASE and old["stage"] == OPEN_STAGE, "completion requires G8_C/characterization_open")
    _require(len(old["completed_work_unit_ids"]) == REQUIRED_COUNT, "completion requires all required work units")
    _require(old["in_progress_work_unit_id"] is None, "completion requires no in-progress work unit")
    _require(not any(old["counters"].values()), "completion requires zero protected counters")
    _registered_binding(previous, SOURCE_MANIFEST_RELATIVE_PATH)
    _registered_binding(previous, MERGE_REPORT_RELATIVE_PATH)
    _registered_binding(previous, TABLE_RELATIVE_PATH)
    current = copy.deepcopy(previous)
    current["identity"]["stage"] = COMPLETE_STAGE
    current["identity"]["restart_command"] = G8_D_RESTART_COMMAND
    validate_state_transition(previous, current)
    digest = write_campaign_state_atomically(state_path, current)
    installed = load_campaign_state(state_path)
    _require(installed["identity"]["phase"] == PHASE, "completion changed campaign phase")
    _require(installed["identity"]["stage"] == COMPLETE_STAGE, "completion stage was not installed")
    _require(installed["identity"]["restart_command"] == G8_D_RESTART_COMMAND, "G8_D restart command was not installed")
    return digest


def validate_production_root(root: Path | str) -> Path:
    """Require the one exact physical production-root spelling."""

    supplied = os.fspath(root)
    expected = "/home/nick/projects/capstone/results/baseline/g8/work_units"
    _require(isinstance(supplied, str) and supplied == expected, "production root must use the exact authenticated absolute path")
    path = Path(supplied)
    _require(path.is_absolute(), "production root must be absolute")
    _require(".." not in path.parts, "production root may not contain traversal")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists():
            _require(not current.is_symlink(), f"production-root component is a symlink: {current}")
    return path


def _characterization_state(state_path: Path = CAMPAIGN_STATE) -> dict[str, Any]:
    state = load_campaign_state(state_path)
    identity = state["identity"]
    _require(identity["phase"] == PHASE, "campaign is not in G8_C")
    _require(identity["stage"] in {OPEN_STAGE, COMPLETE_STAGE}, "campaign is not in a G8_C stage")
    _require(identity["seed_derivation_identity"] == bler_contract.SEED_DERIVATION_IDENTITY, "campaign seed identity drift")
    _require(all(value == 0 for value in identity["counters"].values()), "protected campaign counters changed")
    return state


def reconcile_characterization_campaign(
    context: resume.AuthenticatedResumeContext,
    *,
    root: Path | str,
    repair_recoverable: bool = False,
    state_path: Path = CAMPAIGN_STATE,
) -> dict[str, Any]:
    """Reconcile completed IDs in G8_C using B3 inspection/repair primitives.

    B3's historical campaign-reconciliation writer is intentionally restricted
    to G8_B stages.  G8_C therefore owns the phase-specific projection while
    still using B3's authenticated census, classification, exact recovery
    matrix, and chain validators.  Only this coordinator writes campaign state;
    workers never call this function.
    """

    root = validate_production_root(root)
    state = _characterization_state(state_path)
    inspection = resume.inspect_runtime_root(
        context,
        root=root,
        scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        repair_mode=(
            resume.REPAIR_MODE_REPAIR_RECOVERABLE
            if repair_recoverable
            else resume.REPAIR_MODE_READ_ONLY
        ),
    )
    records = inspection["classifications"]
    required_ids = list(context.ordered_work_unit_ids)
    completed = [
        record["work_unit_id"]
        for record in records
        if record["classification"] == resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH
    ]
    _require(completed == sorted(completed, key=context.ordinal), "B3 completed IDs are not authority ordered")
    _require(len(completed) == len(set(completed)), "duplicate completed IDs in B3 projection")
    old_completed = state["identity"]["completed_work_unit_ids"]
    _require(set(old_completed).issubset(set(completed)), "campaign state leads raw evidence")
    current = copy.deepcopy(state)
    current["identity"]["completed_work_unit_ids"] = completed
    current["identity"]["in_progress_work_unit_id"] = None
    if current["identity"]["completed_work_unit_ids"] != old_completed:
        validate_state_transition(state, current)
        digest = write_campaign_state_atomically(state_path, current)
    else:
        digest = sha256_file(state_path)
    installed = load_campaign_state(state_path)
    _require(installed["identity"]["completed_work_unit_ids"] == completed, "campaign reconciliation installed wrong IDs")
    counts = {
        "completed_full_strength": len(completed),
        "remaining": sum(record["classification"] in resume.REMAINING_CLASSIFICATIONS for record in records),
        "recoverable": sum(record["classification"] in resume.RECOVERABLE_CLASSIFICATIONS for record in records),
        "failed_retryable": sum(record["classification"] == resume.CLASSIFICATION_FAILED_RETRYABLE for record in records),
        "terminal_nonmergeable": sum(record["classification"] == resume.CLASSIFICATION_TERMINAL_NONMERGEABLE for record in records),
        "unknown": 0,
        "duplicate": 0,
        "missing": len(required_ids) - len(completed),
    }
    return {
        "state_sha256": digest,
        "counts": counts,
        "repairs": inspection["repairs"],
        "classifications": records,
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
    }


def _load_source_manifest_for_artifact() -> dict[str, Any]:
    payload, _raw = _read_rendered_json(SOURCE_MANIFEST_PATH, "G8_C source manifest")
    return validate_source_manifest(payload, require_registered=True)


def _load_merge_report_for_artifact() -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_rendered_json(MERGE_REPORT_PATH, "G8_C merge report")
    value = _strict_object(payload, MERGE_REPORT_FIELDS, "merge report")
    _require(value["schema_version"] == MERGE_REPORT_SCHEMA_VERSION, "merge report schema drift")
    _require(value["artifact_role"] == MERGE_REPORT_ARTIFACT_ROLE, "merge report role drift")
    _require(value["phase"] == PHASE and value["checkpoint"] == "C3", "merge report phase/checkpoint drift")
    _require(value["report_id"] == _self_excluding_id(value, "report_id", MERGE_REPORT_ID_PREFIX), "merge report ID does not reproduce")
    source = _load_source_manifest_for_artifact()
    _require(value["source_manifest_id"] == source["manifest_id"], "merge report source-manifest ID mismatch")
    _require(value["source_manifest_sha256"] == sha256_file(SOURCE_MANIFEST_PATH), "merge report source-manifest SHA mismatch")
    return value, raw


def _curves_from_merge(merge: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project accepted merge rows into deterministic measured table curves."""

    units = merge.get("units")
    _require(isinstance(units, list), "merge report units are not a list")
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for unit in units:
        _require(isinstance(unit, Mapping), "merge report unit is not an object")
        identity = unit["bler_identity"]
        identity_key = canonical_json(identity).decode("ascii")
        if identity_key not in grouped:
            grouped[identity_key] = (dict(identity), [])
        grouped[identity_key][1].append(
            {
                "work_unit_id": unit["work_unit_id"],
                "snr_db": unit["snr_db"],
                "trials": unit["trials_completed"],
                "information_bits": unit["information_bits"],
                "bit_errors": unit["bit_errors"],
                "block_errors": unit["block_errors"],
                "ber": unit["ber"],
                "bler": unit["bler"],
                "bler_confidence_low": unit["bler_confidence_low"],
                "bler_confidence_high": unit["bler_confidence_high"],
                "request_sha256": unit["request_sha256"],
                "result_sha256": unit["result_sha256"],
                "state_sha256": unit["state_sha256"],
            }
        )
    curves: list[dict[str, Any]] = []
    for identity_key in sorted(grouped):
        identity, points = grouped[identity_key]
        points.sort(key=lambda point: float(point["snr_db"]))
        _require(
            all(
                left["snr_db"] < right["snr_db"]
                for left, right in zip(points, points[1:], strict=False)
            ),
            "merge report contains duplicate or unordered SNR points",
        )
        curves.append({"identity": identity, "points": points})
    return curves


def build_bler_table_payload() -> dict[str, Any]:
    """Construct the measured-only table from the registered complete merge."""

    source = _load_source_manifest_for_artifact()
    merge, merge_raw = _load_merge_report_for_artifact()
    state = load_campaign_state()
    _registered_binding(state, MERGE_REPORT_RELATIVE_PATH)
    _require(merge["coverage_complete"] is True, "cannot build a table from incomplete merge evidence")
    _require(merge["required_work_unit_count"] == REQUIRED_COUNT, "merge required count drift")
    _require(merge["completed_count"] == REQUIRED_COUNT, "merge completed count drift")
    _require(merge["coverage_contribution_sum"] == REQUIRED_COUNT, "merge coverage contribution drift")
    _require(merge["total_trials"] == REQUIRED_COUNT * FULL_STRENGTH_TRIALS, "merge total trials drift")
    _require(merge["interpolation_used"] is False and merge["extrapolation_used"] is False, "merge invented points")
    curves = _curves_from_merge(merge)
    point_count = sum(len(curve["points"]) for curve in curves)
    payload: dict[str, Any] = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "artifact_role": TABLE_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": "C5",
        "table_id": None,
        "campaign_id": source["campaign_id"],
        "campaign_manifest_sha256": source["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": source["required_bler_artifact_sha256"],
        "selection_policy_sha256": source["selection_policy_sha256"],
        "bler_tooling_contract_id": source["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": source["bler_tooling_contract_sha256"],
        "bler_state_contract_id": source["bler_state_contract_id"],
        "bler_state_contract_sha256": source["bler_state_contract_sha256"],
        "bler_resume_contract_id": source["bler_resume_contract_id"],
        "bler_resume_contract_sha256": source["bler_resume_contract_sha256"],
        "bler_runner_contract_id": source["bler_runner_contract_id"],
        "bler_runner_contract_sha256": source["bler_runner_contract_sha256"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": sha256_bytes(merge_raw),
        "required_work_unit_count": REQUIRED_COUNT,
        "complete_identity_count": len(curves),
        "measured_point_count": point_count,
        "trials_per_point": FULL_STRENGTH_TRIALS,
        "total_trials": merge["total_trials"],
        "interpolation_used": False,
        "extrapolation_used": False,
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        "curves": curves,
    }
    payload["table_id"] = _self_excluding_id(payload, "table_id", TABLE_ID_PREFIX)
    return payload


def _point_from_result(
    unit: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    state: Mapping[str, Any],
    request_sha: str,
    result_sha: str,
    state_sha: str,
) -> dict[str, Any]:
    measurement = result["measurement"]
    identity = result["identity"]
    point = {
        "work_unit_id": unit["work_unit_id"],
        "snr_db": unit["snr_db"],
        "trials": measurement["trials_completed"],
        "information_bits": measurement["information_bits"],
        "bit_errors": measurement["bit_errors"],
        "block_errors": measurement["block_errors"],
        "ber": measurement["ber"],
        "bler": measurement["bler"],
        "bler_confidence_low": measurement["bler_confidence_low"],
        "bler_confidence_high": measurement["bler_confidence_high"],
        "request_sha256": request_sha,
        "result_sha256": result_sha,
        "state_sha256": state_sha,
    }
    _require(identity["work_unit_id"] == unit["work_unit_id"], "result unit mismatch")
    _require(request["work_unit_id"] == unit["work_unit_id"], "request unit mismatch")
    _require(state["identity"]["work_unit_id"] == unit["work_unit_id"], "state unit mismatch")
    return point


def validate_table_payload(payload: Mapping[str, Any], *, require_registered: bool = False) -> dict[str, Any]:
    """Validate every measured point and return a fresh table payload."""

    value = _strict_object(payload, TABLE_FIELDS, "BLER table")
    _require(value["schema_version"] == TABLE_SCHEMA_VERSION, "BLER table schema drift")
    _require(value["artifact_role"] == TABLE_ARTIFACT_ROLE, "BLER table role drift")
    _require(value["phase"] == PHASE and value["checkpoint"] == "C5", "BLER table phase/checkpoint drift")
    _require(value["table_id"] == _self_excluding_id(value, "table_id", TABLE_ID_PREFIX), "BLER table ID does not reproduce")
    source = _load_source_manifest_for_artifact()
    merge, _merge_raw = _load_merge_report_for_artifact()
    for field in (
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
    ):
        _require(value[field] == source[field], f"BLER table authority mismatch: {field}")
    _require(value["source_manifest_id"] == source["manifest_id"], "BLER table source-manifest ID mismatch")
    _require(value["source_manifest_sha256"] == sha256_file(SOURCE_MANIFEST_PATH), "BLER table source-manifest SHA mismatch")
    _require(value["merge_report_id"] == merge["report_id"], "BLER table merge-report ID mismatch")
    _require(value["merge_report_sha256"] == sha256_file(MERGE_REPORT_PATH), "BLER table merge-report SHA mismatch")
    merge_state = load_campaign_state()
    _registered_binding(merge_state, MERGE_REPORT_RELATIVE_PATH)
    _require(value["required_work_unit_count"] == REQUIRED_COUNT, "BLER table required count drift")
    _require(value["trials_per_point"] == FULL_STRENGTH_TRIALS, "BLER table trial count drift")
    _require(value["interpolation_used"] is False and value["extrapolation_used"] is False, "BLER table construction invented points")
    _require(value["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, "BLER table claims test access")
    curves = value["curves"]
    _require(isinstance(curves, list), "BLER table curves are not a list")
    seen_identities: set[str] = set()
    point_count = 0
    for curve in curves:
        curve_value = _strict_object(curve, ("identity", "points"), "BLER curve")
        identity_key = composition.BlerIdentity.from_mapping(curve_value["identity"])
        identity_digest = sha256_bytes(canonical_json(identity_key.as_key()))
        _require(identity_digest not in seen_identities, "BLER table contains duplicate identities")
        seen_identities.add(identity_digest)
        points = curve_value["points"]
        _require(isinstance(points, list) and points, "BLER curve has no points")
        previous_snr: float | None = None
        for point in points:
            point_value = _strict_object(point, TABLE_POINT_FIELDS, "BLER point")
            snr = point_value["snr_db"]
            _require(not isinstance(snr, bool) and isinstance(snr, int | float), "BLER point SNR is not numeric")
            _require(previous_snr is None or float(snr) > previous_snr, "BLER points are not strictly ascending")
            previous_snr = float(snr)
            _require(point_value["trials"] == FULL_STRENGTH_TRIALS, "BLER point trial count changed")
            _require(point_value["block_errors"] <= point_value["trials"], "BLER block-error count invalid")
            _require(point_value["information_bits"] > 0, "BLER information-bit count invalid")
            _require(point_value["bler"] == point_value["block_errors"] / point_value["trials"], "stored BLER does not reproduce")
            _require(point_value["ber"] == point_value["bit_errors"] / point_value["information_bits"], "stored BER does not reproduce")
            _digest(point_value["request_sha256"], "point request SHA")
            _digest(point_value["result_sha256"], "point result SHA")
            _digest(point_value["state_sha256"], "point state SHA")
            point_count += 1
    _require(value["complete_identity_count"] == len(curves), "BLER table identity count does not reproduce")
    _require(value["measured_point_count"] == point_count, "BLER table point count does not reproduce")
    _require(value["total_trials"] == point_count * FULL_STRENGTH_TRIALS, "BLER table total trials do not reproduce")
    expected_curves = _curves_from_merge(merge)
    _require(curves == expected_curves, "BLER table points do not reproduce from the accepted merge report")
    _require(merge["coverage_complete"] is True, "BLER table source merge is incomplete")
    _require(merge["total_trials"] == value["total_trials"], "BLER table total trials disagree with merge")
    if require_registered:
        state = load_campaign_state()
        _registered_binding(state, TABLE_RELATIVE_PATH)
    return value


def load_bler_table(path: Path | str = TABLE_PATH) -> composition.BlerTable:
    """Load the separately bound measured G-8 table as the existing type."""

    target = Path(path)
    payload, _raw = _read_rendered_json(target, "G-8 BLER table")
    validate_table_payload(payload, require_registered=(target == TABLE_PATH))
    curves: dict[composition.BlerIdentity, Any] = {}
    for curve in payload["curves"]:
        identity = composition.BlerIdentity.from_mapping(curve["identity"])
        points = curve["points"]
        curves[identity] = composition._Curve(
            snr_db=tuple(float(point["snr_db"]) for point in points),
            bler=tuple(float(point["bler"]) for point in points),
            trials=FULL_STRENGTH_TRIALS,
        )
    _require(len(curves) == payload["complete_identity_count"], "loader identity count mismatch")
    return composition.BlerTable(curves, provenance=TABLE_RELATIVE_PATH)


def exact_table_point_lookups(table: composition.BlerTable, payload: Mapping[str, Any]) -> int:
    """Exercise every stored point as an exact, non-interpolated lookup."""

    checked = 0
    for curve in payload["curves"]:
        for point in curve["points"]:
            lookup = table.lookup(curve["identity"], point["snr_db"])
            _require(lookup.characterized and lookup.interpolated is False, "stored point was interpolated")
            _require(lookup.bler == point["bler"], "exact table lookup disagrees with stored BLER")
            checked += 1
    return checked


__all__ = [
    "CHARACTERIZATION_DEPENDENCY_PATHS",
    "CHARACTERIZATION_SOURCE_PATHS",
    "COMPLETE_STAGE",
    "EXECUTION_CLASS",
    "G8_C_RESTART_COMMAND",
    "G8_D_RESTART_COMMAND",
    "LOGICAL_ROOT",
    "MERGE_REPORT_FIELDS",
    "MERGE_REPORT_PATH",
    "MERGE_REPORT_RELATIVE_PATH",
    "SOURCE_MANIFEST_ARTIFACT_ROLE",
    "SOURCE_MANIFEST_PATH",
    "SOURCE_MANIFEST_RELATIVE_PATH",
    "TABLE_ARTIFACT_ROLE",
    "TABLE_FIELDS",
    "TABLE_PATH",
    "TABLE_RELATIVE_PATH",
    "CharacterizationError",
    "build_source_manifest",
    "build_bler_table_payload",
    "complete_characterization",
    "exact_table_point_lookups",
    "load_bler_table",
    "reconcile_characterization_campaign",
    "register_source_manifest",
    "validate_production_root",
    "validate_source_manifest",
    "validate_table_payload",
]
