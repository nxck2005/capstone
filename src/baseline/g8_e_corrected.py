"""Corrected, pre-data G8_E E1 measurement contract and E2--E4 runtime.

The first E1 epoch is intentionally left in :mod:`baseline.g8_e`.  This
module is an additive successor: it contains the complete future execution
and transformation path, while the command-line entry point remains closed
until a separate owner authorization artifact is present.

The important boundary in this module is the distinction between three
identities:

* a logical BR-4 candidate/SNR cell;
* an SNR-independent structural clean-measurement identity; and
* an exact per-image physical cache key.

No clean measurement applies the outage policy.  A clean reconstruction
failure is an observed incorrect clean outcome (0/1); a structural or codec
infeasibility has no clean count and makes the structural candidate ineligible.
The separately measured outage object remains the second BR-4 composition
term.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config.params import REPO_ROOT, get


CORRECTED_ROOT = REPO_ROOT / "results/baseline/g8_e/e1_corrected"
CORRECTED_CONTRACT_PATH = CORRECTED_ROOT / "measurement_contract.json"
CORRECTED_AUTHORITY_PATH = CORRECTED_ROOT / "measurement_authority.json"
CORRECTED_MAPPING_PATH = CORRECTED_ROOT / "logical_measurement_mapping.json"
CORRECTED_SOURCE_MANIFEST_PATH = CORRECTED_ROOT / "execution_source_manifest.json"
CORRECTION_PROVENANCE_PATH = CORRECTED_ROOT / "correction_provenance.json"
CORRECTED_RUNTIME_ROOT = CORRECTED_ROOT / "runtime"

CORRECTED_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
AUTHORITY_PREFIX = "g8emeasurementauthority-"
MAPPING_PREFIX = "g8elogicalmapping-"
SOURCE_PREFIX = "g8esourcecorrected-"
CONTRACT_PREFIX = "g8econtractcorrected-"
CAMPAIGN_PREFIX = "g8e-corrected-"
STRUCTURAL_PREFIX = "g8estruct-"
PHYSICAL_PREFIX = "g8ephysical-"
WORK_UNIT_PREFIX = "g8ework-"
RECORD_PREFIX = "g8erecordcorrected-"
AGGREGATE_PREFIX = "g8eaggregate-"
RECONSTRUCTION_PREFIX = "g8erecon-"

INITIAL_DATASET = "imagenette160"
VALIDATION_SPLIT = "val"
PRODUCTION_PROFILE_ID = "local_4060_cu130"
PRODUCTION_DEVICE = "cuda:0"
OLD_CONTRACT_ID = "g8econtract-d25df856e56b45c48fca4750b278e10c62daebced3bf6b8176232133e8c8a8b8"
OLD_CAMPAIGN_ID = "g8e-0037dfcbe2b679d8d0b09ff7116ed93a7e17099522481b7d4c1f1005d88e30bc"

OUTCOME_STRUCTURAL_INFEASIBILITY = "structural_infeasibility"
OUTCOME_CODEC_INFEASIBILITY = "codec_infeasibility"
OUTCOME_DECODE_FAILURE = "decode_failure"
OUTCOME_DELIVERED = "delivered"
OUTCOMES = (
    OUTCOME_STRUCTURAL_INFEASIBILITY,
    OUTCOME_CODEC_INFEASIBILITY,
    OUTCOME_DECODE_FAILURE,
    OUTCOME_DELIVERED,
)
EMITTED_OUTCOMES = (OUTCOME_DECODE_FAILURE, OUTCOME_DELIVERED)


class CorrectedG8EError(ValueError):
    """A corrected G8_E contract, record, cache or resume violation."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CorrectedG8EError(f"value is not canonical JSON: {exc}") from None


def rendered_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise CorrectedG8EError(f"cannot hash {path}: {exc}") from exc


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CorrectedG8EError(f"{label} is not a lowercase SHA-256")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedG8EError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != rendered_json(value):
        raise CorrectedG8EError(f"{label} is not canonical rendered JSON")
    return value, raw


def _strict(value: Any, fields: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        expected = set(fields)
        raise CorrectedG8EError(
            f"{label} schema differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return dict(value)


def _id(prefix: str, payload: Mapping[str, Any]) -> str:
    return prefix + sha256_bytes(canonical_json(payload))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorrectedG8EError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorrectedG8EError(f"{label} must be a non-negative integer")
    return value


def _copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _old_authority() -> dict[str, Any]:
    from baseline.g8_e import verify_e1_authority_file

    return verify_e1_authority_file()


def _old_contract() -> dict[str, Any]:
    from baseline.g8_e import verify_e1_contract_file

    return verify_e1_contract_file(verify_live_assets=True, verify_live_profile=False)


def _g8c_binding() -> dict[str, Any]:
    value, _raw = _read_json(REPO_ROOT / "results/baseline/g8_d/d0_open.json", "G8_D D0")
    binding = value.get("g8_c")
    if not isinstance(binding, Mapping):
        raise CorrectedG8EError("D0 has no G8_C binding")
    required = (
        "campaign_id",
        "execution_profile_id",
        "measurement_source_commit",
        "production_contract_sha256",
        "table_id",
        "table_sha256",
        "merge_report_id",
        "merge_report_sha256",
        "closeout_provenance_id",
        "closeout_provenance_sha256",
        "curves",
        "measured_points",
        "trials_per_point",
        "predecessor_table_contribution",
    )
    if not set(binding) >= set(required):
        raise CorrectedG8EError("G8_C binding is missing required fields")
    data = {name: binding[name] for name in required}
    if data["predecessor_table_contribution"] != "none":
        raise CorrectedG8EError("corrected E1 cannot bind predecessor BLER evidence")
    return data


def _codec_snapshot() -> dict[str, Any]:
    from baseline.g8_d import current_codec_snapshot

    snapshot = current_codec_snapshot()
    if not isinstance(snapshot, dict):
        raise CorrectedG8EError("codec snapshot is not an object")
    return snapshot


def _packet_for(row: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from baseline.classical.channel_transport import build_accounting
    from baseline.ldpc.transport import build_packet_plan

    dataset = str(row["dataset"])
    ratio = str(row["ratio"])
    symbols = int(get("bandwidth.k_symbols")[dataset][ratio])
    packet = build_packet_plan(symbols, str(row["modulation"]), str(row["ldpc_rate"]))
    if not packet.feasible or packet.segmentation is None:
        raise CorrectedG8EError(
            f"configured structural packet is infeasible: {dataset}/{ratio}/{row['modulation']}/{row['ldpc_rate']}"
        )
    accounting = build_accounting(packet).as_dict()
    if accounting["payload_bytes"] != packet.source_bytes:
        raise CorrectedG8EError("packet and accounting payload budgets differ")
    return packet, accounting


def _structural_payload(row: Mapping[str, Any], packet: Any, accounting: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "dataset": row["dataset"],
        "dataset_role": row["dataset_role"],
        "source_codec": row["source_codec"],
        "ratio": row["ratio"],
        "modulation": row["modulation"],
        "ldpc_rate": row["ldpc_rate"],
        "encode_axis_px": int(row["encode_axis_px"]),
        "packet_config_id": row["packet_config_id"],
        "k_symbols": int(packet.channel_bits // packet.q_m),
        "payload_budget_bytes": int(accounting["payload_bytes"]),
        "packet_accounting": _copy(accounting),
    }
    payload = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "identity_type": "g8_e_structural_measurement",
        **fields,
    }
    return {"structural_identity_id": _id(STRUCTURAL_PREFIX, payload), **payload}


def derive_measurement_authority() -> dict[str, Any]:
    """Derive all structural identities without reading an image payload."""

    old = _old_authority()
    rows = old["candidates"]
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    logical_structural: dict[str, str] = {}
    for row in rows:
        packet, accounting = _packet_for(row)
        structural = _structural_payload(row, packet, accounting)
        key = tuple(structural[field] for field in (
            "dataset", "dataset_role", "source_codec", "ratio", "modulation",
            "ldpc_rate", "encode_axis_px", "packet_config_id",
        ))
        previous = grouped.setdefault(key, structural)
        if previous != structural:
            raise CorrectedG8EError("structural identity derivation is not deterministic")
        logical_structural[str(row["candidate_id"])] = structural["structural_identity_id"]

    structural_rows = sorted(grouped.values(), key=lambda item: item["structural_identity_id"])
    if len(logical_structural) != len(rows):
        raise CorrectedG8EError("logical candidate authority contains duplicates")
    initial = [row for row in structural_rows if row["dataset"] == INITIAL_DATASET]
    all_datasets = sorted({str(row["dataset"]) for row in structural_rows})
    structural_digest = sha256_bytes(canonical_json(structural_rows))
    body: dict[str, Any] = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_measurement_authority",
        "phase": "G8_E",
        "checkpoint": "E1_corrected",
        "status": "FROZEN_PRE_DATA",
        "source_logical_authority": {
            "path": "results/baseline/g8_e/candidate_authority.json",
            "authority_id": old["authority_id"],
            "authority_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"),
            "candidate_authority_digest": old["candidate_authority_digest"],
        },
        "identity_semantics": {
            "logical_fields": list(old["candidate_fields"]),
            "structural_fields": [
                "dataset", "dataset_role", "source_codec", "ratio", "modulation",
                "ldpc_rate", "encode_axis_px", "packet_config_id", "k_symbols",
                "payload_budget_bytes", "packet_accounting",
            ],
            "snr_excluded_from_structural_identity": True,
            "snr_exclusion_reason": "clean JPEG2000 measurement and packet budget do not depend on channel SNR",
            "physical_cache_fields": [
                "source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape",
                "payload_budget_bytes", "encode_axis_px", "codec_configuration_hash",
                "codec_runtime_identity",
            ],
            "physical_cache_reuse_predicate": "all physical_cache_fields equal byte-for-byte",
        },
        "counts": {
            "logical_all_roles_snr_cells": len(rows),
            "logical_initial_snr_cells": sum(1 for row in rows if row["dataset"] == INITIAL_DATASET),
            "structural_all_roles": len(structural_rows),
            "structural_initial": len(initial),
            "snr_points": len(old["dimensions"]["snr_grid_db"]),
            "datasets": all_datasets,
        },
        "structural_digest": structural_digest,
        "structural_identities": structural_rows,
        "logical_candidate_to_structural_id": logical_structural,
        "validation_scope": {
            "measurement_dataset": INITIAL_DATASET,
            "split": VALIDATION_SPLIT,
            "fallback_dataset_not_measured": True,
            "smoke_dataset_not_measured": True,
            "test_split_sealed": True,
        },
    }
    body["authority_id"] = _id(AUTHORITY_PREFIX, {key: value for key, value in body.items() if key != "authority_id"})
    return body


def _manifest_ids(dataset: str = INITIAL_DATASET) -> tuple[str, ...]:
    from data.manifests import manifest_path, validate_manifest_bytes

    path = manifest_path(dataset, REPO_ROOT)
    rows = validate_manifest_bytes(dataset, path.read_bytes())
    ids = tuple(row.stable_sample_id for row in rows if row.split == VALIDATION_SPLIT)
    if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise CorrectedG8EError("validation manifest IDs are not a unique sorted sequence")
    return ids


def _mapping_rows(authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    old = _old_authority()
    table = _g8c_binding()
    result: list[dict[str, Any]] = []
    for row in old["candidates"]:
        candidate_id = str(row["candidate_id"])
        structural_id = authority["logical_candidate_to_structural_id"].get(candidate_id)
        if not isinstance(structural_id, str):
            raise CorrectedG8EError(f"logical candidate {candidate_id} has no structural mapping")
        linkage = {
            "logical_candidate_id": candidate_id,
            "measurement_identity_id": structural_id,
            "table_id": table["table_id"],
            "table_sha256": table["table_sha256"],
            "snr_db": row["snr_db"],
            "modulation": row["modulation"],
            "ldpc_rate": row["ldpc_rate"],
            "ratio": row["ratio"],
        }
        result.append({
            "logical_candidate_id": candidate_id,
            "candidate_id": candidate_id,
            "dataset": row["dataset"],
            "dataset_role": row["dataset_role"],
            "ratio": row["ratio"],
            "modulation": row["modulation"],
            "ldpc_rate": row["ldpc_rate"],
            "encode_axis_px": int(row["encode_axis_px"]),
            "snr_db": row["snr_db"],
            "packet_config_id": row["packet_config_id"],
            "measurement_identity_id": structural_id,
            "g8_c_linkage_id": _id("g8elink-", linkage),
            "g8_c_linkage": linkage,
        })
    result.sort(key=lambda item: item["logical_candidate_id"])
    return result


def derive_logical_measurement_mapping(authority: Mapping[str, Any] | None = None) -> dict[str, Any]:
    authority = derive_measurement_authority() if authority is None else authority
    rows = _mapping_rows(authority)
    expected_ids = {str(row["candidate_id"]) for row in _old_authority()["candidates"]}
    actual_ids = {str(row["logical_candidate_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise CorrectedG8EError("logical-to-measurement mapping is not an exact candidate set")
    body: dict[str, Any] = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_logical_to_measurement_mapping",
        "authority_id": authority["authority_id"],
        "authority_sha256": sha256_bytes(rendered_json(authority)),
        "mapping_semantics": {
            "one_logical_snr_cell_to_one_structural_measurement_identity": True,
            "one_structural_identity_has_one_clean_measurement_per_validation_sample": True,
            "all_logical_cells_preserved": True,
            "g8_c_linkage_retained_per_logical_cell": True,
        },
        "mapping_count": len(rows),
        "mapping_rows": rows,
    }
    body["mapping_digest"] = sha256_bytes(canonical_json(rows))
    body["mapping_id"] = _id(MAPPING_PREFIX, {key: value for key, value in body.items() if key != "mapping_id"})
    return body


def _physical_equivalence_plan(authority: Mapping[str, Any]) -> dict[str, Any]:
    initial = [
        row for row in authority["structural_identities"]
        if row["dataset"] == INITIAL_DATASET
    ]
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in initial:
        groups[(int(row["payload_budget_bytes"]), int(row["encode_axis_px"]))].append(
            str(row["structural_identity_id"])
        )
    ordered = []
    for (budget, axis), ids in sorted(groups.items()):
        ordered.append({
            "payload_budget_bytes": budget,
            "encode_axis_px": axis,
            "structural_identity_ids": sorted(ids),
            "equivalence_is_only_physical_cache_key": True,
        })
    return {
        "groups": ordered,
        "unique_physical_keys_per_validation_image": len(ordered),
        "cross_mcs_equivalence_group_count": sum(1 for item in ordered if len(item["structural_identity_ids"]) > 1),
        "max_structural_members_in_one_group": max(len(item["structural_identity_ids"]) for item in ordered),
        "modulation_and_rate_are_not_key_exclusions": True,
        "ratio_and_axis_alone_are_not_a_reuse_predicate": True,
    }


def derive_selection_call_plan(authority: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Derive the E5 call shape from the initial authority and frozen modes."""

    from baseline.classical.composition import SYSTEM_MODES

    authority = derive_measurement_authority() if authority is None else authority
    rows = [row for row in authority["structural_identities"] if row["dataset"] == INITIAL_DATASET]
    snr_count = int(authority["counts"]["snr_points"])
    by_ratio = defaultdict(int)
    for row in rows:
        by_ratio[str(row["ratio"])] += 1
    calls = []
    for ratio in sorted(by_ratio):
        candidates_per_snr = by_ratio[ratio]
        for mode in SYSTEM_MODES:
            calls.append({
                "dataset": INITIAL_DATASET,
                "ratio": ratio,
                "mode": mode,
                "snr_groups": snr_count,
                "candidates_per_snr": candidates_per_snr,
                "candidate_count": candidates_per_snr * snr_count,
                "samples_per_cell": len(_manifest_ids()),
            })
    if not calls:
        raise CorrectedG8EError("selection call plan is empty")
    max_candidates = max(int(call["candidate_count"]) for call in calls)
    max_samples = max(int(call["samples_per_cell"]) for call in calls)
    return {
        "grouping": "one select_operating_points call per initial dataset, ratio and system mode; each mapping contains all SNR groups",
        "calls": calls,
        "call_count": len(calls),
        "max_candidates": max_candidates,
        "max_samples": max_samples,
        "typed_authorization_fields": ["gate", "authorized_by", "reason", "max_candidates", "max_samples"],
        "typed_max_workload": None,
        "artifact_only_workload_bound": max_candidates * max_samples,
        "derived_before_observations": True,
    }


def _work_unit_id(structural_id: str, sample_id: str) -> str:
    return _id(WORK_UNIT_PREFIX, {
        "schema_version": RECORD_SCHEMA_VERSION,
        "measurement_identity_id": structural_id,
        "stable_sample_id": sample_id,
        "split": VALIDATION_SPLIT,
    })


def expected_work_units(authority: Mapping[str, Any], sample_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    initial = sorted(
        (row for row in authority["structural_identities"] if row["dataset"] == INITIAL_DATASET),
        key=lambda row: row["structural_identity_id"],
    )
    result = []
    for structural in initial:
        logical_ids = sorted(
            candidate_id
            for candidate_id, measurement_id in authority["logical_candidate_to_structural_id"].items()
            if measurement_id == structural["structural_identity_id"]
        )
        for sample_id in sample_ids:
            result.append({
                "work_unit_id": _work_unit_id(structural["structural_identity_id"], str(sample_id)),
                "ordinal": len(result),
                "measurement_identity_id": structural["structural_identity_id"],
                "logical_candidate_ids": logical_ids,
                "stable_sample_id": str(sample_id),
                "dataset": INITIAL_DATASET,
                "split": VALIDATION_SPLIT,
            })
    return tuple(result)


def _source_bindings(repo_root: Path = REPO_ROOT) -> tuple[tuple[str, str], ...]:
    return (
        ("src/baseline/g8_e_corrected.py", "corrected_e2_e3_e4_runtime"),
        ("tools/run_g8_e_corrected.py", "owner_gated_e2_runner"),
        ("tools/gen_g8_e_corrected.py", "corrected_e1_generator"),
        ("tools/verify_g8_e_corrected.py", "corrected_e1_independent_verifier"),
        ("tools/merge_g8_e_corrected.py", "e3_exact_set_merge"),
        ("tools/aggregate_g8_e_corrected.py", "e4_count_derived_accuracy"),
        ("src/baseline/g8_d.py", "frozen_codec_reconstruction_br11_transaction_seams"),
        ("src/baseline/g8_campaign.py", "frozen_structural_candidate_authority"),
        ("src/baseline/j2k.py", "frozen_jpeg2000_codec"),
        ("src/data/manifests.py", "stable_validation_id_enumeration"),
        ("src/data/registry.py", "validation_only_model_data_boundary"),
        ("src/data/adapters.py", "source_byte_identity"),
        ("src/data/identity.py", "stable_sample_id_rule"),
        ("src/data/preprocessing.py", "canonical_and_codec_resize_contract"),
        ("src/data/test_access.py", "sealed_test_boundary"),
        ("src/models/frozen_reference_classifier.py", "frozen_g1_classifier_loader"),
        ("src/models/reference_classifier.py", "frozen_g1_model_architecture"),
        ("src/baseline/classical/composition.py", "frozen_br4_composition_and_authorization_types"),
        ("src/baseline/classical/outage.py", "separate_measured_outage_policy"),
        ("src/baseline/classical/records.py", "frozen_br11_byte_split"),
        ("src/baseline/ldpc/transport.py", "frozen_packet_plan"),
        ("src/baseline/ldpc/segmentation.py", "frozen_packet_segmentation"),
        ("src/baseline/ldpc/rate_matching.py", "frozen_rate_matching"),
        ("src/baseline/ldpc/modulation.py", "frozen_modulation_bits_per_symbol"),
        ("src/baseline/classical/channel_transport.py", "frozen_br11_transport_accounting"),
        ("src/config/params.py", "generated_parameter_loader"),
        ("src/config/execution_profiles.py", "execution_profile_authentication"),
        ("src/config/run_config.py", "frozen_classifier_run_configuration"),
        ("src/env.py", "runtime_environment_and_openjpeg_guard"),
        ("results/baseline/g8_e/candidate_authority.json", "historical_logical_authority"),
        ("results/baseline/g8_e/e0_open.json", "upstream_e0_opening"),
        ("results/baseline/g8_e/measurement_contract.json", "historical_superseded_e1_contract"),
        ("results/baseline/g8_e/execution_source_manifest.json", "historical_superseded_e1_sources"),
        ("results/baseline/g8_d/measurement_contract.json", "current_g8_d_contract"),
        ("results/baseline/g8_d/d7_handoff.json", "current_g8_d_handoff"),
        ("results/baseline/w4/integration_adjudication.json", "frozen_w4_selection_machinery"),
        ("results/baseline/w4/outage_policy.json", "frozen_measured_outage"),
        ("results/baseline/g8_d/d0_open.json", "frozen_g8_c_linkage"),
        ("results/baseline/g8_pascal_successor/successor_bler_table.json", "frozen_successor_bler_table"),
    )


def build_source_manifest(source_commit: str) -> dict[str, Any]:
    entries = []
    for path, role in _source_bindings():
        full = REPO_ROOT / path
        if not full.is_file():
            raise CorrectedG8EError(f"corrected source binding is missing: {path}")
        entries.append({"path": path, "role": role, "bytes": len(full.read_bytes()), "sha256": sha256_file(full)})
    body: dict[str, Any] = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_execution_source_manifest",
        "checkpoint": "E1_corrected",
        "status": "FROZEN_PRE_DATA",
        "source_commit": source_commit,
        "source_entries": entries,
        "excludes": [
            "results/baseline/g8_e/e1_corrected/execution_source_manifest.json",
            "results/baseline/g8_e/e1_corrected/measurement_contract.json",
            "results/baseline/g8_e/e1_corrected/measurement_authority.json",
            "results/baseline/g8_e/e1_corrected/logical_measurement_mapping.json",
            "results/baseline/g8_e/e1_corrected/correction_provenance.json",
            "results/baseline/g8_e/e1_corrected/runtime/",
        ],
        "scientific_source_closure": {
            "e2_runner_is_complete_not_a_refusal_stub": True,
            "e3_merge_is_bound_before_data": True,
            "e4_count_aggregation_is_bound_before_data": True,
            "old_refusal_stub_is_not_current": True,
            "source_drift_after_freeze_is_hold": True,
        },
    }
    body["source_manifest_id"] = _id(SOURCE_PREFIX, {key: value for key, value in body.items() if key != "source_manifest_id"})
    return body


def _estimate_plan(authority: Mapping[str, Any], selection_plan: Mapping[str, Any]) -> dict[str, Any]:
    sample_count = len(_manifest_ids())
    initial = [row for row in authority["structural_identities"] if row["dataset"] == INITIAL_DATASET]
    physical = _physical_equivalence_plan(authority)
    cache_budgets_per_image = sum(int(group["payload_budget_bytes"]) for group in physical["groups"])
    physical_jobs = sample_count * int(physical["unique_physical_keys_per_validation_image"])
    record_placeholder = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_scientific_measurement_record",
        "campaign_id": CAMPAIGN_PREFIX + "0" * 64,
        "contract_id": CONTRACT_PREFIX + "0" * 64,
        "measurement_authority_id": authority["authority_id"],
        "measurement_identity_id": initial[0]["structural_identity_id"],
        "logical_candidate_ids": ["cand-" + "0" * 64] * int(authority["counts"]["snr_points"]),
        "work_unit_id": WORK_UNIT_PREFIX + "0" * 64,
        "stable_sample_id": "0" * 64,
        "label": 0,
        "source_bytes_sha256": "0" * 64,
        "canonical_pixels_sha256": "0" * 64,
        "canonical_shape": [160, 160, 3],
        "physical_cache_key": PHYSICAL_PREFIX + "0" * 64,
        "codec_cache_object_id": "g8ecodec-" + "0" * 64,
        "outcome": OUTCOME_DELIVERED,
        "emitted_codestream": {"sha256": "0" * 64, "bytes": 1},
        "reconstruction": {"object_id": RECONSTRUCTION_PREFIX + "0" * 64, "sha256": "0" * 64},
        "classifier_observation": {"label": 0, "predicted_label": 0},
        "correct_count": 1,
        "total_count": 1,
        "br11": {"bytes_sent": 1, "emitted_codestream_bytes": 1, "header_bytes": 0, "payload_bytes": 1, "payload_filler_bytes": 0},
        "g8_c_linkage_digest": "0" * 64,
        "profile_id": PRODUCTION_PROFILE_ID,
        "source_commit": "0" * 40,
        "scientific_evidence": True,
        "merge_eligible": True,
        "validation_only": True,
        "test_access": 0,
        "training": 0,
        "inference": 1,
        "outage_applied": False,
    }
    record_bytes = len(rendered_json(record_placeholder))
    smoke_path = REPO_ROOT / "results/baseline/w4/smoke_summary.json"
    smoke, _ = _read_json(smoke_path, "W4 smoke summary")
    rows = int(smoke["raw_rows_count"])
    seconds = float(smoke["wall_clock_s"])
    rate = rows / seconds
    codec_metadata_floor = len(rendered_json({
        "schema_version": RECORD_SCHEMA_VERSION,
        "physical_cache_key": PHYSICAL_PREFIX + "0" * 64,
        "status": "feasible",
        "codestream_b64": "",
    }))
    reconstruction_bytes = int(get("datasets.imagenette160.image_size")[0]) * int(get("datasets.imagenette160.image_size")[1]) * 3
    return {
        "logical_snr_cells_initial": int(authority["counts"]["logical_initial_snr_cells"]),
        "logical_snr_cells_all_roles": int(authority["counts"]["logical_all_roles_snr_cells"]),
        "structural_measurement_identities_initial": len(initial),
        "structural_measurement_identities_all_roles": int(authority["counts"]["structural_all_roles"]),
        "validation_images": sample_count,
        "scientific_per_image_work_units": len(initial) * sample_count,
        "unique_codec_cache_jobs": physical_jobs,
        "unique_reconstruction_jobs_upper_bound": physical_jobs,
        "unique_classifier_forwards_upper_bound": physical_jobs,
        "e4_measured_accuracy_objects_initial": len(initial),
        "physical_cache_keys_per_image": physical["unique_physical_keys_per_validation_image"],
        "physical_equivalence_groups": physical["groups"],
        "cross_mcs_equivalence_group_count": physical["cross_mcs_equivalence_group_count"],
        "record_bytes_estimate": record_bytes,
        "scientific_record_bytes_estimate": record_bytes * len(initial) * sample_count,
        "codec_cache_metadata_floor_bytes": codec_metadata_floor * physical_jobs,
        "codec_cache_payload_upper_bound_bytes": cache_budgets_per_image * sample_count,
        "reconstruction_pixel_upper_bound_bytes": reconstruction_bytes * physical_jobs,
        "historical_bounded_throughput_reference": {
            "source_path": str(smoke_path.relative_to(REPO_ROOT)),
            "source_sha256": sha256_file(smoke_path),
            "raw_rows": rows,
            "wall_clock_s": seconds,
            "rows_per_second": rate,
            "not_e2_measurement": True,
        },
        "projected_physical_work_seconds": physical_jobs / rate,
        "projected_physical_work_hours": physical_jobs / rate / 3600,
        "projection_status": "planning_estimate_only; no validation work observed",
        "selection_call_plan": _copy(selection_plan),
    }


def build_correction_provenance() -> dict[str, Any]:
    old_contract, old_contract_raw = _read_json(REPO_ROOT / "results/baseline/g8_e/measurement_contract.json", "old E1 contract")
    old_authority, old_authority_raw = _read_json(REPO_ROOT / "results/baseline/g8_e/candidate_authority.json", "old E1 authority")
    old_sources, old_sources_raw = _read_json(REPO_ROOT / "results/baseline/g8_e/execution_source_manifest.json", "old E1 source manifest")
    return {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "artifact_role": "g8_e_first_e1_correction_provenance",
        "status": "superseded-before-data",
        "scientific_measurement_coverage": 0,
        "reason": [
            "old E1 source epoch bound tools/run_g8_e.py while it was still a refusal stub",
            "old E1 pass-one authorization scope copied PB_3 smoke limits",
            "old E1 collapsed modulation and LDPC rate in the proposed clean-measurement key",
            "old E1 mixed outage-derived counts into acc_clean",
            "old E1 duplicated SNR-independent image records",
        ],
        "preserved_historical_artifacts": [
            "results/baseline/g8_e/e0_open.json",
            "results/baseline/g8_e/candidate_authority.json",
            "results/baseline/g8_e/measurement_contract.json",
            "results/baseline/g8_e/execution_source_manifest.json",
            "results/baseline/g8_e/corpus_spec.json",
            "tools/run_g8_e.py",
            "src/baseline/g8_e.py",
        ],
        "old_e1": {
            "contract_id": old_contract["contract_id"],
            "contract_sha256": sha256_bytes(old_contract_raw),
            "campaign_id": old_contract["campaign_id"],
            "authority_id": old_authority["authority_id"],
            "authority_sha256": sha256_bytes(old_authority_raw),
            "source_manifest_id": old_sources["manifest_id"],
            "source_manifest_sha256": sha256_bytes(old_sources_raw),
            "source_commit": old_sources["source_commit"],
            "old_runner_path": "tools/run_g8_e.py",
            "old_runner_sha256": sha256_file(REPO_ROOT / "tools/run_g8_e.py"),
            "old_runner_was_refusal_stub": True,
            "old_authorization_scope": {"max_candidates": 64, "max_samples": 25, "max_workload": 512},
        },
        "zero_data_audit": {
            "runtime_path": "results/baseline/g8_e/runtime",
            "runtime_absent": True,
            "e2_records": 0,
            "e2_campaign_state": False,
            "pass_one_pre_marker": False,
            "pass_one_completion": False,
            "pass_two": False,
            "g8_f_state": False,
            "validation_measurement_coverage": 0,
            "training": 0,
            "test_access": 0,
            "fallback": False,
            "ratio_adjudicated": False,
        },
        "classification_rule": "preserve old bytes; classify as superseded-before-data, never scientifically invalidated",
        "current_corrected_artifact_root": "results/baseline/g8_e/e1_corrected",
    }


def build_corrected_contract(
    authority: Mapping[str, Any],
    mapping: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> dict[str, Any]:
    selection_plan = derive_selection_call_plan(authority)
    seed_body = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "old_contract_id": correction["old_e1"]["contract_id"],
        "old_contract_sha256": correction["old_e1"]["contract_sha256"],
        "measurement_authority_id": authority["authority_id"],
        "measurement_authority_digest": authority["structural_digest"],
        "mapping_id": mapping["mapping_id"],
        "mapping_digest": mapping["mapping_digest"],
        "source_manifest_id": source_manifest["source_manifest_id"],
        "semantics_epoch": "clean_measurement_v2_structural_physical_split",
    }
    campaign_id = _id(CAMPAIGN_PREFIX, seed_body)
    codec_snapshot = _codec_snapshot()
    codec_hash = sha256_bytes(canonical_json(codec_snapshot))
    g8c = _g8c_binding()
    w4_path = REPO_ROOT / "results/baseline/w4/integration_adjudication.json"
    outage_path = REPO_ROOT / "results/baseline/w4/outage_policy.json"
    profile_binding = {
        "profile_id": PRODUCTION_PROFILE_ID,
        "device": PRODUCTION_DEVICE,
        "sole_writer": "local",
        "profile_switching": "forbidden; interruption requires explicit supersession/new campaign",
        "source_commit": source_manifest["source_commit"],
        "selection_status": "frozen before first scientific measurement",
    }
    old_profile = _old_contract()["execution_profile"]["authentication"]
    profile_binding.update({
        "config_hash": old_profile["config_hash"],
        "lock_file": old_profile["lock_file"],
        "lock_file_sha256": old_profile["lock_file_sha256"],
    })
    body: dict[str, Any] = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_executable_pre_data_contract",
        "phase": "G8_E",
        "checkpoint": "E1_corrected",
        "status": "FROZEN_PRE_DATA_EXECUTABLE",
        "campaign_id": campaign_id,
        "campaign_seed": seed_body,
        "contract_id": None,
        "supersedes_before_data": correction["old_e1"],
        "upstream": {
            "e0_path": "results/baseline/g8_e/e0_open.json",
            "e0_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_e/e0_open.json"),
            "g8_c_binding": g8c,
            "g8_d_contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
            "g8_d_handoff_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"),
            "w4_integration_adjudication_sha256": sha256_file(w4_path),
            "w4_outage_policy_sha256": sha256_file(outage_path),
            "outage_term_remains_separate": True,
        },
        "candidate_authority": {
            "path": "results/baseline/g8_e/candidate_authority.json",
            "authority_id": authority["source_logical_authority"]["authority_id"],
            "authority_sha256": authority["source_logical_authority"]["authority_sha256"],
            "logical_all_roles_snr_cells": authority["counts"]["logical_all_roles_snr_cells"],
            "logical_initial_snr_cells": authority["counts"]["logical_initial_snr_cells"],
            "unchanged_and_reused": True,
        },
        "measurement_authority": {
            "path": str(CORRECTED_AUTHORITY_PATH.relative_to(REPO_ROOT)),
            "authority_id": authority["authority_id"],
            "authority_sha256": sha256_bytes(rendered_json(authority)),
            "mapping_path": str(CORRECTED_MAPPING_PATH.relative_to(REPO_ROOT)),
            "mapping_id": mapping["mapping_id"],
            "mapping_sha256": sha256_bytes(rendered_json(mapping)),
            "mapping_digest": mapping["mapping_digest"],
            "logical_to_structural_is_total": True,
        },
        "execution_source_manifest": {
            "path": str(CORRECTED_SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "source_manifest_id": source_manifest["source_manifest_id"],
            "source_manifest_sha256": sha256_bytes(rendered_json(source_manifest)),
            "source_commit": source_manifest["source_commit"],
        },
        "execution_profile": profile_binding,
        "codec": {
            "configuration_hash": codec_hash,
            "snapshot": codec_snapshot,
            "physical_cache_fields": [
                "source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape",
                "payload_budget_bytes", "encode_axis_px", "codec_configuration_hash",
                "codec_runtime_identity",
            ],
            "snr_excluded": True,
            "modulation_and_ldpc_rate_are_preserved_in_structural_identity": True,
            "actual_emitted_bytes_authoritative": True,
        },
        "clean_measurement_semantics": {
            "formula": "expected_accuracy = P(TB success) * measured acc_clean + (1 - P(TB success)) * measured acc_outage",
            "acc_clean": "count-derived validation reconstruction/classifier accuracy at the structural measurement identity; no outage value is injected",
            "acc_outage": "separate measured constant-class outage object from results/baseline/w4/outage_policy.json",
            "full_noisy_phy_per_image": False,
            "structural_infeasibility": "record one work unit with null clean counts and mark the structural candidate ineligible",
            "codec_infeasibility": "record one work unit with null clean counts and mark the structural candidate ineligible",
            "clean_reconstruction_decode_failure": "record emitted BR-11 and clean count 0/1; do not apply outage policy; candidate remains measurable",
            "delivered": "record emitted BR-11, reconstruction and frozen G-1 classifier-derived count 0/1",
            "clean_denominator": "sum total_count over delivered and clean reconstruction-failure rows; infeasible rows are present but have null counts and make the candidate ineligible",
            "missing_rows": "contract failure",
            "accuracy_float_input": "forbidden; E4 sums correct_count and total_count",
        },
        "record_schema": {
            "schema_version": RECORD_SCHEMA_VERSION,
            "fields": list(MeasurementRecord.FIELDS),
            "validation_split_only": True,
            "source_and_canonical_pixels_hash_bound": True,
            "packet_budget_and_structural_identity_bound": True,
            "physical_cache_key_bound": True,
            "g8_c_linkage_bound": True,
            "br11_emitted_rows_include_decode_failures": True,
            "scientific_evidence": True,
            "merge_eligible": True,
            "caller_accuracy_field": False,
        },
        "transaction": {
            "runtime_root": str(CORRECTED_RUNTIME_ROOT.relative_to(REPO_ROOT)),
            "same_directory_atomic_publication": True,
            "exclusive_writer_lock": True,
            "exact_prefix_state": True,
            "immutable_completed_records": True,
            "immutable_physical_and_reconstruction_cache_objects": True,
            "stale_aggregate_repaired_by_deterministic_republication": True,
            "completed_output_reuse": True,
            "source_drift_rejected": True,
            "duplicate_contribution_rejected": True,
            "crash_boundaries": ["after_work_claim", "after_codec_cache", "after_record", "after_aggregate", "after_state"],
        },
        "selection_authorization": selection_plan,
        "compute_plan": _estimate_plan(authority, selection_plan),
        "safety": {
            "measurement_coverage": 0,
            "validation_decoding": 0,
            "inference": 0,
            "training": 0,
            "test_access": 0,
            "fallback_invoked": False,
            "ratio_adjudicated": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "e2_started": False,
        },
        "authorization": {
            "required": True,
            "path": "results/baseline/g8_e/e1_corrected/e2_execution_authorization.json",
            "issued": False,
            "refuse_before_validation_decode": True,
            "typed_g8_authorization_is_not_issued_here": True,
        },
        "declarations": {
            "executable_e2_source_is_frozen": True,
            "e3_source_is_frozen": True,
            "e4_source_is_frozen": True,
            "zero_full_validation_measurements": True,
            "e2_awaits_owner_execution_authorization": True,
            "no_validation_image_decoding_performed": True,
            "no_training": True,
            "no_test_access": True,
            "no_fallback": True,
            "no_ratio_adjudication": True,
        },
    }
    body["contract_id"] = _id(CONTRACT_PREFIX, {key: value for key, value in body.items() if key != "contract_id"})
    return body


def build_corrected_bundle(source_commit: str) -> dict[str, dict[str, Any]]:
    authority = derive_measurement_authority()
    mapping = derive_logical_measurement_mapping(authority)
    source = build_source_manifest(source_commit)
    correction = build_correction_provenance()
    contract = build_corrected_contract(authority, mapping, source, correction)
    return {
        "measurement_authority": authority,
        "logical_measurement_mapping": mapping,
        "execution_source_manifest": source,
        "correction_provenance": correction,
        "measurement_contract": contract,
    }


def _check_no_accuracy_float(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "accuracy":
                raise CorrectedG8EError(f"caller-supplied accuracy field is forbidden at {path}")
            _check_no_accuracy_float(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _check_no_accuracy_float(child, f"{path}[{index}]")


def validate_source_manifest(value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    required = {"schema_version", "artifact_role", "checkpoint", "status", "source_commit", "source_entries", "excludes", "scientific_source_closure", "source_manifest_id"}
    if set(value) != required:
        raise CorrectedG8EError("corrected source manifest schema differs")
    body = {key: value[key] for key in value if key != "source_manifest_id"}
    if value["source_manifest_id"] != _id(SOURCE_PREFIX, body):
        raise CorrectedG8EError("corrected source manifest ID differs")
    if value["status"] != "FROZEN_PRE_DATA" or value["checkpoint"] != "E1_corrected":
        raise CorrectedG8EError("corrected source manifest status differs")
    entries = value["source_entries"]
    if not isinstance(entries, list) or not entries:
        raise CorrectedG8EError("corrected source manifest has no entries")
    for entry in entries:
        data = _strict(entry, ("path", "role", "bytes", "sha256"), "corrected source entry")
        path = repo_root / data["path"]
        if not path.is_file() or len(path.read_bytes()) != data["bytes"] or sha256_file(path) != data["sha256"]:
            raise CorrectedG8EError(f"corrected source drift: {data['path']}")
    if value["scientific_source_closure"] != {
        "e2_runner_is_complete_not_a_refusal_stub": True,
        "e3_merge_is_bound_before_data": True,
        "e4_count_aggregation_is_bound_before_data": True,
        "old_refusal_stub_is_not_current": True,
        "source_drift_after_freeze_is_hold": True,
    }:
        raise CorrectedG8EError("corrected source closure declaration differs")
    return dict(value)


def validate_measurement_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    fresh = derive_measurement_authority()
    if dict(value) != fresh:
        raise CorrectedG8EError("measurement authority differs from live derivation")
    return dict(value)


def validate_mapping(value: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    fresh = derive_logical_measurement_mapping(authority)
    if dict(value) != fresh:
        raise CorrectedG8EError("logical measurement mapping differs from live derivation")
    return dict(value)


def validate_correction_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    fresh = build_correction_provenance()
    if dict(value) != fresh:
        raise CorrectedG8EError("correction provenance differs from preserved old E1 bytes")
    if value["status"] != "superseded-before-data" or value["scientific_measurement_coverage"] != 0:
        raise CorrectedG8EError("old E1 is not explicitly classified before data")
    return dict(value)


def validate_corrected_contract(value: Mapping[str, Any], *, verify_live_sources: bool = True) -> dict[str, Any]:
    if set(value) != {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "campaign_id", "campaign_seed", "contract_id",
        "supersedes_before_data", "upstream", "candidate_authority", "measurement_authority", "execution_source_manifest",
        "execution_profile", "codec", "clean_measurement_semantics", "record_schema", "transaction", "selection_authorization",
        "compute_plan", "safety", "authorization", "declarations",
    }:
        raise CorrectedG8EError("corrected contract top-level schema differs")
    body = {key: value[key] for key in value if key != "contract_id"}
    if value["contract_id"] != _id(CONTRACT_PREFIX, body):
        raise CorrectedG8EError("corrected contract ID differs")
    if value["status"] != "FROZEN_PRE_DATA_EXECUTABLE" or value["checkpoint"] != "E1_corrected":
        raise CorrectedG8EError("corrected contract is not the executable pre-data epoch")
    if value["campaign_id"] == OLD_CAMPAIGN_ID or value["supersedes_before_data"]["contract_id"] != OLD_CONTRACT_ID:
        raise CorrectedG8EError("old E1 campaign was not separated from current E2")
    authority, _ = _read_json(CORRECTED_AUTHORITY_PATH, "corrected measurement authority")
    mapping, _ = _read_json(CORRECTED_MAPPING_PATH, "corrected logical mapping")
    source, _ = _read_json(CORRECTED_SOURCE_MANIFEST_PATH, "corrected source manifest")
    correction, _ = _read_json(CORRECTION_PROVENANCE_PATH, "correction provenance")
    validate_measurement_authority(authority)
    validate_mapping(mapping, authority)
    validate_correction_provenance(correction)
    if value["measurement_authority"]["authority_id"] != authority["authority_id"] or value["measurement_authority"]["mapping_id"] != mapping["mapping_id"]:
        raise CorrectedG8EError("corrected contract measurement bindings differ")
    if value["execution_source_manifest"]["source_manifest_id"] != source["source_manifest_id"]:
        raise CorrectedG8EError("corrected contract source-manifest binding differs")
    if verify_live_sources:
        validate_source_manifest(source)
    if value["safety"] != {
        "measurement_coverage": 0, "validation_decoding": 0, "inference": 0, "training": 0,
        "test_access": 0, "fallback_invoked": False, "ratio_adjudicated": False,
        "pass_one_started": False, "pass_two_started": False, "e2_started": False,
    }:
        raise CorrectedG8EError("corrected contract safety counters are not zero")
    if value["authorization"]["issued"] is not False or value["authorization"]["required"] is not True:
        raise CorrectedG8EError("corrected E2 authorization boundary is not closed")
    if value["selection_authorization"]["typed_max_workload"] is not None:
        raise CorrectedG8EError("typed G8Authorization incorrectly acquired max_workload")
    return dict(value)


def verify_corrected_bundle(*, verify_live_sources: bool = True) -> dict[str, Any]:
    authority, authority_raw = _read_json(CORRECTED_AUTHORITY_PATH, "corrected authority")
    mapping, mapping_raw = _read_json(CORRECTED_MAPPING_PATH, "corrected mapping")
    source, source_raw = _read_json(CORRECTED_SOURCE_MANIFEST_PATH, "corrected source manifest")
    correction, correction_raw = _read_json(CORRECTION_PROVENANCE_PATH, "correction provenance")
    contract, contract_raw = _read_json(CORRECTED_CONTRACT_PATH, "corrected contract")
    validate_measurement_authority(authority)
    validate_mapping(mapping, authority)
    validate_correction_provenance(correction)
    validate_source_manifest(source) if verify_live_sources else None
    validate_corrected_contract(contract, verify_live_sources=verify_live_sources)
    bindings = contract["measurement_authority"]
    if bindings["authority_sha256"] != sha256_bytes(authority_raw) or bindings["mapping_sha256"] != sha256_bytes(mapping_raw):
        raise CorrectedG8EError("corrected authority/mapping bytes do not match contract")
    source_binding = contract["execution_source_manifest"]
    if source_binding["source_manifest_sha256"] != sha256_bytes(source_raw):
        raise CorrectedG8EError("corrected source-manifest bytes do not match contract")
    if not CORRECTED_RUNTIME_ROOT.exists():
        pass
    return {
        "contract": contract,
        "authority": authority,
        "mapping": mapping,
        "source_manifest": source,
        "correction_provenance": correction,
        "contract_sha256": sha256_bytes(contract_raw),
        "authority_sha256": sha256_bytes(authority_raw),
        "mapping_sha256": sha256_bytes(mapping_raw),
        "source_manifest_sha256": sha256_bytes(source_raw),
        "correction_provenance_sha256": sha256_bytes(correction_raw),
    }


@dataclass(frozen=True)
class PhysicalCacheKey:
    """Complete per-image clean-codec identity; SNR is intentionally absent."""

    source_bytes_sha256: str
    canonical_pixels_sha256: str
    canonical_shape: tuple[int, int, int]
    payload_budget_bytes: int
    encode_axis_px: int
    codec_configuration_hash: str
    codec_runtime_identity: str

    def __post_init__(self) -> None:
        for name in ("source_bytes_sha256", "canonical_pixels_sha256", "codec_configuration_hash"):
            _digest(getattr(self, name), name)
        if len(self.canonical_shape) != 3 or any(int(x) <= 0 for x in self.canonical_shape):
            raise CorrectedG8EError("physical cache canonical shape is invalid")
        _positive_int(self.payload_budget_bytes, "physical cache payload budget")
        _positive_int(self.encode_axis_px, "physical cache encode axis")
        if not self.codec_runtime_identity:
            raise CorrectedG8EError("physical cache runtime identity is empty")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CORRECTED_SCHEMA_VERSION,
            "identity_type": "g8_e_physical_cache",
            "source_bytes_sha256": self.source_bytes_sha256,
            "canonical_pixels_sha256": self.canonical_pixels_sha256,
            "canonical_shape": list(self.canonical_shape),
            "payload_budget_bytes": self.payload_budget_bytes,
            "encode_axis_px": self.encode_axis_px,
            "codec_configuration_hash": self.codec_configuration_hash,
            "codec_runtime_identity": self.codec_runtime_identity,
        }

    @property
    def key_id(self) -> str:
        return _id(PHYSICAL_PREFIX, self.payload())


def physical_cache_key(
    *,
    source_bytes: bytes,
    canonical_pixels: np.ndarray,
    payload_budget_bytes: int,
    encode_axis_px: int,
    codec_configuration_hash: str,
    codec_runtime_identity: str,
) -> PhysicalCacheKey:
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise CorrectedG8EError("source bytes are required for physical cache identity")
    if not isinstance(canonical_pixels, np.ndarray) or canonical_pixels.dtype != np.uint8 or canonical_pixels.ndim != 3 or canonical_pixels.shape[2] != 3:
        raise CorrectedG8EError("canonical pixels must be uint8 RGB HWC")
    contiguous = np.ascontiguousarray(canonical_pixels)
    return PhysicalCacheKey(
        source_bytes_sha256=sha256_bytes(source_bytes),
        canonical_pixels_sha256=sha256_bytes(contiguous.tobytes()),
        canonical_shape=tuple(int(x) for x in contiguous.shape),
        payload_budget_bytes=int(payload_budget_bytes),
        encode_axis_px=int(encode_axis_px),
        codec_configuration_hash=str(codec_configuration_hash),
        codec_runtime_identity=str(codec_runtime_identity),
    )


class CodecBackend(Protocol):
    snapshot: Mapping[str, Any]
    configuration_hash: str

    def encode_to_budget(self, image: np.ndarray, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CodecArtifact:
    key: PhysicalCacheKey
    status: str
    reason: str | None
    codestream: bytes | None
    cache_object_id: str
    cache_hit: bool


class PhysicalCodecCache:
    """Content-addressed codec search keyed by the complete physical identity."""

    def __init__(self, root: Path, backend: CodecBackend) -> None:
        self.root = Path(root).resolve()
        self.backend = backend

    def _path(self, key: PhysicalCacheKey) -> Path:
        return self.root / "codec" / f"{key.key_id}.json"

    def _load(self, path: Path, key: PhysicalCacheKey) -> CodecArtifact:
        value, _raw = _read_json(path, "physical codec cache object")
        required = {"schema_version", "artifact_role", "key", "status", "reason", "codestream_b64", "codestream_sha256", "cache_object_id"}
        if set(value) != required or value["key"] != key.payload():
            raise CorrectedG8EError("physical codec cache key differs")
        if value["status"] not in {"feasible", "codec_infeasibility"}:
            raise CorrectedG8EError("physical codec cache status differs")
        codestream = None
        if value["codestream_b64"] is not None:
            codestream = base64.b64decode(value["codestream_b64"], validate=True)
            if value["status"] != "feasible" or value["codestream_sha256"] != sha256_bytes(codestream):
                raise CorrectedG8EError("physical codec cache codestream hash differs")
        elif value["status"] == "feasible":
            raise CorrectedG8EError("feasible physical codec cache has no codestream")
        body = {name: value[name] for name in value if name != "cache_object_id"}
        if value["cache_object_id"] != _id("g8ecodec-", body):
            raise CorrectedG8EError("physical codec cache object ID differs")
        return CodecArtifact(key, str(value["status"]), value["reason"], codestream, str(value["cache_object_id"]), True)

    def get_or_create(self, key: PhysicalCacheKey, encoded_pixels: np.ndarray) -> CodecArtifact:
        path = self._path(key)
        if path.exists():
            return self._load(path, key)
        try:
            result = self.backend.encode_to_budget(
                np.ascontiguousarray(encoded_pixels),
                canonical_pixels_sha256=key.canonical_pixels_sha256,
                budget_bytes=key.payload_budget_bytes,
                encode_axis_px=key.encode_axis_px,
            )
            feasible = bool(getattr(result, "feasible"))
            codestream = getattr(result, "codestream", None)
            if feasible and not isinstance(codestream, bytes):
                raise CorrectedG8EError("codec backend marked feasible without bytes")
            if feasible and len(codestream) > key.payload_budget_bytes:
                raise CorrectedG8EError("codec backend emitted bytes above the physical payload budget")
            if not feasible:
                codestream = None
            status = "feasible" if feasible else "codec_infeasibility"
            reason = None if feasible else "budget_exceeded"
        except Exception as exc:
            status = "codec_infeasibility"
            reason = f"codec_backend_error: {exc}"
            codestream = None
        body: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_physical_codec_cache_object",
            "key": key.payload(),
            "status": status,
            "reason": reason,
            "codestream_b64": None if codestream is None else base64.b64encode(codestream).decode("ascii"),
            "codestream_sha256": None if codestream is None else sha256_bytes(codestream),
        }
        body["cache_object_id"] = _id("g8ecodec-", body)
        self.root.joinpath("codec").mkdir(parents=True, exist_ok=True)
        from baseline.g8_d import publish_immutable_object

        publish_immutable_object(path, rendered_json(body))
        return CodecArtifact(key, status, reason, codestream, str(body["cache_object_id"]), False)


class ReconstructionBackend(Protocol):
    def __call__(self, codestream: bytes) -> np.ndarray: ...


@dataclass(frozen=True)
class ReconstructionArtifact:
    object_id: str
    status: str
    reason: str | None
    pixels: np.ndarray | None
    cache_hit: bool


class PhysicalReconstructionCache:
    """Immutable success/failure reconstruction objects for exact cache keys."""

    def __init__(self, root: Path, decoder: ReconstructionBackend) -> None:
        self.root = Path(root).resolve()
        self.decoder = decoder

    def _identity(self, key: PhysicalCacheKey, codestream: bytes, output_shape: tuple[int, int, int]) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "identity_type": "g8_e_reconstruction",
            "physical_cache_key": key.payload(),
            "codestream_sha256": sha256_bytes(codestream),
            "output_shape": list(output_shape),
            "upsample_interpolation": get("preprocessing.codec_upsample_interpolation"),
        }

    def get_or_create(self, key: PhysicalCacheKey, codestream: bytes, output_shape: tuple[int, int, int]) -> ReconstructionArtifact:
        identity = self._identity(key, codestream, output_shape)
        object_id = _id(RECONSTRUCTION_PREFIX, identity)
        path = self.root / "reconstruction" / f"{object_id}.json"
        if path.exists():
            value, _raw = _read_json(path, "reconstruction cache object")
            if value.get("identity") != identity:
                raise CorrectedG8EError("reconstruction cache identity differs")
            if value["status"] == "decode_failure":
                return ReconstructionArtifact(object_id, "decode_failure", value["reason"], None, True)
            pixels = np.frombuffer(base64.b64decode(value["pixels_b64"], validate=True), dtype=np.uint8).reshape(tuple(output_shape)).copy()
            if sha256_bytes(pixels.tobytes()) != value["pixels_sha256"]:
                raise CorrectedG8EError("reconstruction cache pixels differ")
            return ReconstructionArtifact(object_id, "delivered", None, pixels, True)
        try:
            decoded = self.decoder(codestream)
            if not isinstance(decoded, np.ndarray) or decoded.dtype != np.uint8 or decoded.ndim != 3 or decoded.shape[2] != 3:
                raise CorrectedG8EError("decoder returned non-RGB uint8 pixels")
            if tuple(decoded.shape) != tuple(output_shape):
                from data.preprocessing import codec_upsample

                decoded = codec_upsample(decoded, output_hw=tuple(output_shape[:2]))
            pixels = np.ascontiguousarray(decoded)
            if tuple(pixels.shape) != tuple(output_shape):
                raise CorrectedG8EError("reconstruction output shape differs")
            status = "delivered"
            reason = None
        except Exception as exc:
            status = "decode_failure"
            reason = f"clean_reconstruction_error: {exc}"
            pixels = None
        body: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_reconstruction_cache_object",
            "identity": identity,
            "status": status,
            "reason": reason,
            "pixels_b64": None if pixels is None else base64.b64encode(pixels.tobytes()).decode("ascii"),
            "pixels_sha256": None if pixels is None else sha256_bytes(pixels.tobytes()),
        }
        self.root.joinpath("reconstruction").mkdir(parents=True, exist_ok=True)
        from baseline.g8_d import publish_immutable_object

        publish_immutable_object(path, rendered_json(body))
        return ReconstructionArtifact(object_id, status, reason, pixels, False)


class CleanClassifier(Protocol):
    def predict(self, pixels: np.ndarray) -> int: ...


class FrozenG1Classifier:
    """Lazy production adapter; importing this class never loads validation data."""

    def __init__(self, device: str = PRODUCTION_DEVICE) -> None:
        self.device = device
        self._model: Any | None = None

    def predict(self, pixels: np.ndarray) -> int:
        if self._model is None:
            from models.frozen_reference_classifier import load_frozen_reference_classifier

            self._model = load_frozen_reference_classifier(self.device, allow_download=False)
        import torch
        from data.preprocessing import reconstruction_input

        tensor = reconstruction_input(pixels)[None].to(self.device)
        with torch.inference_mode():
            return int(self._model(tensor).argmax(dim=1).item())


@dataclass(frozen=True)
class SyntheticSample:
    """Test seam; all uses must be labelled NON-SCIENTIFIC by the caller."""

    stable_sample_id: str
    label: int
    source_bytes: bytes
    canonical_pixels: np.ndarray
    dataset: str = INITIAL_DATASET
    split: str = VALIDATION_SPLIT


class MeasurementRecord:
    """Strict scientific record with count-derived clean correctness only."""

    FIELDS = (
        "schema_version", "artifact_role", "record_id", "campaign_id", "contract_id",
        "measurement_authority_id", "measurement_identity_id", "logical_candidate_ids",
        "work_unit_id", "stable_sample_id", "dataset", "split", "label",
        "source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape",
        "structural_identity", "packet_budget", "physical_cache_key", "codec_cache_object_id",
        "outcome", "failure_stage", "emitted_codestream", "reconstruction", "classifier_observation",
        "correct_count", "total_count", "br11", "g8_c_linkage_digest", "profile_id",
        "source_commit", "validation_only", "outage_applied", "scientific_evidence",
        "merge_eligible", "test_access", "training", "inference", "record_labels",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        value = _strict(payload, self.FIELDS, "measurement record")
        _check_no_accuracy_float(value)
        self.value = value
        self._validate()

    @classmethod
    def from_observation(
        cls,
        *,
        campaign_id: str,
        contract_id: str,
        authority: Mapping[str, Any],
        work_unit: Mapping[str, Any],
        structural: Mapping[str, Any],
        sample: SyntheticSample,
        physical_key: PhysicalCacheKey,
        codec: CodecArtifact,
        reconstruction: ReconstructionArtifact | None,
        predicted_label: int | None,
        profile_id: str,
        source_commit: str,
        g8_c_linkage_digest: str,
        record_labels: Sequence[str],
    ) -> "MeasurementRecord":
        if sample.split != VALIDATION_SPLIT or sample.dataset != INITIAL_DATASET:
            raise CorrectedG8EError("measurement sample is outside the Imagenette validation split")
        outcome = OUTCOME_CODEC_INFEASIBILITY if codec.status != "feasible" else OUTCOME_DELIVERED
        failure_stage = None if outcome == OUTCOME_DELIVERED else "codec_search"
        reconstruction_payload = None
        classifier_observation = None
        correct = None
        total = None
        br11 = None
        emitted = None
        if codec.status == "feasible" and codec.codestream is not None:
            emitted = {"sha256": sha256_bytes(codec.codestream), "bytes": len(codec.codestream)}
            from baseline.g8_d import EmittedFileIdentity, account_br11

            emitted_identity = EmittedFileIdentity(
                codec_search_key_id=physical_key.key_id,
                codestream_sha256=emitted["sha256"],
                emitted_bytes=emitted["bytes"],
                payload_budget_bytes=physical_key.payload_budget_bytes,
                filler_bytes=physical_key.payload_budget_bytes - emitted["bytes"],
            )
            accounted = account_br11(
                codec.codestream,
                emitted_file_identity=emitted_identity,
                bytes_sent=physical_key.payload_budget_bytes,
                verdict=OUTCOME_DELIVERED if reconstruction and reconstruction.status == "delivered" else OUTCOME_DECODE_FAILURE,
            )
            br11 = accounted.as_dict()
            if reconstruction is None or reconstruction.status == "decode_failure":
                outcome = OUTCOME_DECODE_FAILURE
                failure_stage = "clean_reconstruction"
                correct, total = 0, 1
            else:
                if predicted_label is None or isinstance(predicted_label, bool):
                    raise CorrectedG8EError("delivered clean measurement has no classifier prediction")
                outcome = OUTCOME_DELIVERED
                failure_stage = None
                classifier_observation = {"label": int(sample.label), "predicted_label": int(predicted_label)}
                correct, total = int(int(predicted_label) == int(sample.label)), 1
                reconstruction_payload = {
                    "object_id": reconstruction.object_id,
                    "status": reconstruction.status,
                    "cache_hit": reconstruction.cache_hit,
                }
            if reconstruction is None or reconstruction.status == "decode_failure":
                reconstruction_payload = None if reconstruction is None else {
                    "object_id": reconstruction.object_id,
                    "status": reconstruction.status,
                    "cache_hit": reconstruction.cache_hit,
                }
        value: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_scientific_measurement_record",
            "record_id": None,
            "campaign_id": campaign_id,
            "contract_id": contract_id,
            "measurement_authority_id": authority["authority_id"],
            "measurement_identity_id": structural["structural_identity_id"],
            "logical_candidate_ids": list(work_unit["logical_candidate_ids"]),
            "work_unit_id": work_unit["work_unit_id"],
            "stable_sample_id": sample.stable_sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "label": int(sample.label),
            "source_bytes_sha256": sha256_bytes(sample.source_bytes),
            "canonical_pixels_sha256": sha256_bytes(np.ascontiguousarray(sample.canonical_pixels).tobytes()),
            "canonical_shape": list(sample.canonical_pixels.shape),
            "structural_identity": _copy(structural),
            "packet_budget": {
                "payload_budget_bytes": structural["payload_budget_bytes"],
                "packet_accounting": _copy(structural["packet_accounting"]),
            },
            "physical_cache_key": physical_key.payload(),
            "codec_cache_object_id": codec.cache_object_id,
            "outcome": outcome,
            "failure_stage": failure_stage,
            "emitted_codestream": emitted,
            "reconstruction": reconstruction_payload,
            "classifier_observation": classifier_observation,
            "correct_count": correct,
            "total_count": total,
            "br11": br11,
            "g8_c_linkage_digest": g8_c_linkage_digest,
            "profile_id": profile_id,
            "source_commit": source_commit,
            "validation_only": True,
            "outage_applied": False,
            "scientific_evidence": True,
            "merge_eligible": True,
            "test_access": 0,
            "training": 0,
            "inference": 1 if outcome == OUTCOME_DELIVERED else 0,
            "record_labels": list(record_labels),
        }
        value["record_id"] = _id(RECORD_PREFIX, {key: child for key, child in value.items() if key != "record_id"})
        return cls(value)

    def _validate(self) -> None:
        value = self.value
        if value["schema_version"] != RECORD_SCHEMA_VERSION or value["artifact_role"] != "g8_e_scientific_measurement_record":
            raise CorrectedG8EError("measurement record header differs")
        if value["split"] != VALIDATION_SPLIT or value["dataset"] != INITIAL_DATASET:
            raise CorrectedG8EError("measurement record is not Imagenette validation-only")
        if value["validation_only"] is not True or value["test_access"] != 0 or value["training"] != 0:
            raise CorrectedG8EError("measurement record safety boundary differs")
        if value["outage_applied"] is not False or value["scientific_evidence"] is not True or value["merge_eligible"] is not True:
            raise CorrectedG8EError("measurement record scientific flags differ")
        if value["outcome"] not in OUTCOMES:
            raise CorrectedG8EError("measurement record outcome is unknown")
        if value["record_id"] != _id(RECORD_PREFIX, {key: child for key, child in value.items() if key != "record_id"}):
            raise CorrectedG8EError("measurement record ID differs")
        _digest(value["source_bytes_sha256"], "record source bytes")
        _digest(value["canonical_pixels_sha256"], "record canonical pixels")
        from data.identity import stable_sample_id_width

        if value["stable_sample_id"] != value["source_bytes_sha256"][:stable_sample_id_width()]:
            raise CorrectedG8EError("measurement stable sample ID does not match source-byte identity")
        structural = value["structural_identity"]
        packet_budget = value["packet_budget"]
        physical = value["physical_cache_key"]
        if not isinstance(structural, Mapping) or not isinstance(packet_budget, Mapping) or not isinstance(physical, Mapping):
            raise CorrectedG8EError("measurement identity/budget/cache fields are malformed")
        if packet_budget.get("payload_budget_bytes") != structural.get("payload_budget_bytes"):
            raise CorrectedG8EError("measurement packet payload budget differs from structural identity")
        if physical.get("payload_budget_bytes") != structural.get("payload_budget_bytes") or physical.get("encode_axis_px") != structural.get("encode_axis_px"):
            raise CorrectedG8EError("physical cache key does not bind the structural payload budget and axis")
        if physical.get("source_bytes_sha256") != value["source_bytes_sha256"] or physical.get("canonical_pixels_sha256") != value["canonical_pixels_sha256"]:
            raise CorrectedG8EError("physical cache key does not bind source/canonical pixels")
        if value["outcome"] in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY}:
            if value["correct_count"] is not None or value["total_count"] is not None or value["br11"] is not None or value["emitted_codestream"] is not None:
                raise CorrectedG8EError("infeasible record carries clean or emitted evidence")
        elif value["outcome"] == OUTCOME_DECODE_FAILURE:
            if value["correct_count"] != 0 or value["total_count"] != 1 or value["br11"] is None or value["outage_applied"]:
                raise CorrectedG8EError("clean reconstruction failure semantics differ")
        else:
            obs = value["classifier_observation"]
            if not isinstance(obs, Mapping) or value["total_count"] != 1 or value["correct_count"] != int(obs["predicted_label"] == obs["label"]):
                raise CorrectedG8EError("delivered classifier counts are not derived from observation")
        if value["outcome"] in EMITTED_OUTCOMES and value["br11"] is None:
            raise CorrectedG8EError("emitted outcome has no BR-11 accounting")
        if value["outcome"] not in EMITTED_OUTCOMES and value["br11"] is not None:
            raise CorrectedG8EError("infeasible outcome carries BR-11 accounting")

    def as_dict(self) -> dict[str, Any]:
        return _copy(self.value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeasurementRecord":
        return cls(value)


def clean_accuracy_from_records(records: Sequence[MeasurementRecord], structural_id: str) -> Any:
    """Build a count-derived composition object, never from a float."""

    from baseline.classical.composition import MeasuredCodecAccuracy

    selected = [record for record in records if record.value["measurement_identity_id"] == structural_id]
    if not selected:
        raise CorrectedG8EError("no records for measured structural identity")
    if any(record.value["outcome"] in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY} for record in selected):
        raise CorrectedG8EError("infeasible structural identity has no eligible clean accuracy")
    correct = sum(int(record.value["correct_count"]) for record in selected)
    total = sum(int(record.value["total_count"]) for record in selected)
    if total != len(selected):
        raise CorrectedG8EError("clean accuracy denominator does not cover every clean work unit")
    return MeasuredCodecAccuracy(correct=correct, total=total, split=VALIDATION_SPLIT, source=f"g8_e_corrected:{structural_id}")


class MeasurementExecutor:
    """Execute one clean work unit after the outer owner gate has opened."""

    def __init__(
        self,
        *,
        bundle: Mapping[str, Any],
        runtime_root: Path,
        backend: CodecBackend,
        decoder: ReconstructionBackend,
        classifier: CleanClassifier,
        non_scientific_fixture: bool = False,
    ) -> None:
        self.bundle = bundle
        self.authority = bundle["authority"]
        self.contract = bundle["contract"]
        self.runtime_root = Path(runtime_root)
        self.codec = PhysicalCodecCache(self.runtime_root / "cache", backend)
        self.reconstruction = PhysicalReconstructionCache(self.runtime_root / "cache", decoder)
        self.classifier = classifier
        self.non_scientific_fixture = non_scientific_fixture

    def __call__(self, work_unit: Mapping[str, Any], sample: SyntheticSample) -> MeasurementRecord:
        from data.identity import stable_sample_id

        if sample.stable_sample_id != stable_sample_id(sample.source_bytes):
            raise CorrectedG8EError("sample stable ID is not source-byte identity")
        structural = next(
            row for row in self.authority["structural_identities"]
            if row["structural_identity_id"] == work_unit["measurement_identity_id"]
        )
        if sample.dataset != INITIAL_DATASET or sample.split != VALIDATION_SPLIT:
            raise CorrectedG8EError("executor received non-validation sample")
        key = physical_cache_key(
            source_bytes=sample.source_bytes,
            canonical_pixels=sample.canonical_pixels,
            payload_budget_bytes=int(structural["payload_budget_bytes"]),
            encode_axis_px=int(structural["encode_axis_px"]),
            codec_configuration_hash=str(self.contract["codec"]["configuration_hash"]),
            codec_runtime_identity=str(self.contract["codec"]["snapshot"]["environment"]),
        )
        from data.preprocessing import codec_downsample

        encoded = codec_downsample(sample.canonical_pixels, key.encode_axis_px)
        codec = self.codec.get_or_create(key, encoded)
        reconstruction = None
        prediction = None
        if codec.status == "feasible" and codec.codestream is not None:
            reconstruction = self.reconstruction.get_or_create(key, codec.codestream, tuple(int(x) for x in sample.canonical_pixels.shape))
            if reconstruction.status == "delivered" and reconstruction.pixels is not None:
                prediction = self.classifier.predict(reconstruction.pixels)
        linkage_digest = sha256_bytes(canonical_json({
            "logical_candidate_ids": work_unit["logical_candidate_ids"],
            "measurement_identity_id": work_unit["measurement_identity_id"],
            "g8_c_table": self.contract["upstream"]["g8_c_binding"],
        }))
        labels = [
            "NON-SCIENTIFIC", "NON-SELECTION", "NOT E2 VALIDATION EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"
        ] if self.non_scientific_fixture else []
        return MeasurementRecord.from_observation(
            campaign_id=self.contract["campaign_id"],
            contract_id=self.contract["contract_id"],
            authority=self.authority,
            work_unit=work_unit,
            structural=structural,
            sample=sample,
            physical_key=key,
            codec=codec,
            reconstruction=reconstruction,
            predicted_label=prediction,
            profile_id=self.contract["execution_profile"]["profile_id"],
            source_commit=self.contract["execution_source_manifest"]["source_commit"],
            g8_c_linkage_digest=linkage_digest,
            record_labels=labels,
        )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CorrectedG8EError(f"immutable publication collision at {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise CorrectedG8EError(f"atomic publication collision at {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rendered_json(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class AtomicE2Campaign:
    """Exact-prefix, lock-protected transaction for corrected E2 records."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        contract: Mapping[str, Any],
        authority: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        executor: Callable[[Mapping[str, Any], SyntheticSample], MeasurementRecord],
        sample_provider: Callable[[str], SyntheticSample],
    ) -> None:
        self.root = Path(runtime_root).resolve()
        self.contract = contract
        self.authority = authority
        self.work_units = tuple(dict(unit) for unit in work_units)
        self.executor = executor
        self.sample_provider = sample_provider
        self.state_path = self.root / "campaign_state.json"
        self.lock_path = self.root / ".campaign.lock"
        self.records_dir = self.root / "records"
        self.aggregates_dir = self.root / "aggregates"

    def _new_state(self) -> dict[str, Any]:
        value = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_corrected_campaign_state",
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "measurement_authority_id": self.authority["authority_id"],
            "work_unit_ids": [unit["work_unit_id"] for unit in self.work_units],
            "completed_work_unit_ids": [],
            "record_refs": [],
            "in_progress_work_unit_id": None,
            "aggregate_ref": None,
            "counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},
        }
        value["state_sha256"] = sha256_bytes(canonical_json(value))
        return value

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._new_state()
        value, _raw = _read_json(self.state_path, "corrected campaign state")
        expected = [unit["work_unit_id"] for unit in self.work_units]
        if value.get("work_unit_ids") != expected:
            raise CorrectedG8EError("campaign state work-unit authority differs")
        state_hash = value.get("state_sha256")
        body = {key: child for key, child in value.items() if key != "state_sha256"}
        if state_hash != sha256_bytes(canonical_json(body)):
            raise CorrectedG8EError("campaign state digest differs")
        completed = value.get("completed_work_unit_ids")
        if not isinstance(completed, list) or completed != expected[: len(completed)] or len(completed) != len(set(completed)):
            raise CorrectedG8EError("campaign state is not an exact ordered prefix")
        if value.get("campaign_id") != self.contract["campaign_id"] or value.get("contract_id") != self.contract["contract_id"]:
            raise CorrectedG8EError("campaign state contract differs")
        return value

    def _record_path(self, work_unit_id: str) -> Path:
        return self.records_dir / f"{work_unit_id}.json"

    def _read_record(self, work_unit_id: str) -> tuple[MeasurementRecord, bytes] | None:
        path = self._record_path(work_unit_id)
        if not path.exists():
            return None
        value, raw = _read_json(path, "corrected measurement record")
        record = MeasurementRecord.from_mapping(value)
        if record.value["work_unit_id"] != work_unit_id:
            raise CorrectedG8EError("record path and work-unit identity differ")
        return record, raw

    def _aggregate(self, records: Sequence[MeasurementRecord]) -> dict[str, Any]:
        body = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_corrected_prefix_aggregate",
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "completed_work_unit_ids": [record.value["work_unit_id"] for record in records],
            "record_ids": [record.value["record_id"] for record in records],
            "record_sha256s": [sha256_bytes(rendered_json(record.as_dict())) for record in records],
            "correct_count": sum(int(record.value["correct_count"]) for record in records if record.value["correct_count"] is not None),
            "total_count": sum(int(record.value["total_count"]) for record in records if record.value["total_count"] is not None),
            "infeasible_count": sum(record.value["outcome"] in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY} for record in records),
        }
        body["aggregate_id"] = _id(AGGREGATE_PREFIX, body)
        return body

    def _validate_prefix(self, state: Mapping[str, Any]) -> list[MeasurementRecord]:
        records = []
        for work_unit_id in state["completed_work_unit_ids"]:
            loaded = self._read_record(work_unit_id)
            if loaded is None:
                raise CorrectedG8EError("completed work unit has no immutable record")
            records.append(loaded[0])
        expected_names = {f"{work_unit_id}.json" for work_unit_id in state["completed_work_unit_ids"]}
        if state.get("in_progress_work_unit_id") is not None:
            expected_names.add(f"{state['in_progress_work_unit_id']}.json")
        if self.records_dir.exists():
            for child in self.records_dir.iterdir():
                if child.name not in expected_names:
                    raise CorrectedG8EError("record exists outside exact prefix")
        return records

    def run_next(self, *, crash_after: str | None = None) -> bool:
        with _lock(self.lock_path):
            self.root.mkdir(parents=True, exist_ok=True)
            state = self._load_state()
            records = self._validate_prefix(state)
            completed = len(records)
            if completed == len(self.work_units):
                return False
            unit = self.work_units[completed]
            if state["in_progress_work_unit_id"] is None:
                state = dict(state)
                state["in_progress_work_unit_id"] = unit["work_unit_id"]
                state["state_sha256"] = sha256_bytes(canonical_json({key: child for key, child in state.items() if key != "state_sha256"}))
                _replace_json(self.state_path, state)
            if crash_after == "work_claim":
                raise RuntimeError("synthetic crash after work claim")
            loaded = self._read_record(unit["work_unit_id"])
            if loaded is None:
                sample = self.sample_provider(unit["stable_sample_id"])
                record = self.executor(unit, sample)
                if record.value["work_unit_id"] != unit["work_unit_id"]:
                    raise CorrectedG8EError("executor returned wrong work unit")
                _atomic_publish(self._record_path(unit["work_unit_id"]), rendered_json(record.as_dict()))
            else:
                record = loaded[0]
            if crash_after == "record":
                raise RuntimeError("synthetic crash after record publication")
            prefix = records + [record]
            aggregate = self._aggregate(prefix)
            aggregate_path = self.aggregates_dir / f"{aggregate['aggregate_id']}.json"
            _atomic_publish(aggregate_path, rendered_json(aggregate))
            if crash_after == "aggregate":
                raise RuntimeError("synthetic crash after aggregate publication")
            final = dict(state)
            final["completed_work_unit_ids"] = list(state["completed_work_unit_ids"]) + [unit["work_unit_id"]]
            final["record_refs"] = list(state["record_refs"]) + [{"work_unit_id": unit["work_unit_id"], "record_id": record.value["record_id"], "sha256": sha256_bytes(rendered_json(record.as_dict()))}]
            final["in_progress_work_unit_id"] = None
            final["aggregate_ref"] = {"aggregate_id": aggregate["aggregate_id"], "sha256": sha256_bytes(rendered_json(aggregate))}
            final["counters"] = {
                "validation_decoding": len(final["completed_work_unit_ids"]),
                "inference": sum(int(item.value["inference"]) for item in prefix),
                "training": 0,
                "test_access": 0,
            }
            final["state_sha256"] = sha256_bytes(canonical_json({key: child for key, child in final.items() if key != "state_sha256"}))
            if crash_after == "before_state":
                raise RuntimeError("synthetic crash before state publication")
            _replace_json(self.state_path, final)
            return True

    def run_all(self) -> None:
        while self.run_next():
            pass

    def state(self) -> dict[str, Any]:
        with _lock(self.lock_path):
            return self._load_state()


def merge_e3_records(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    record_values: Sequence[Mapping[str, Any] | MeasurementRecord],
    production: bool = True,
) -> dict[str, Any]:
    """Exact-set E3 merge; missing and duplicate work units are fatal."""

    expected = expected_work_units(authority, sample_ids)
    expected_ids = [unit["work_unit_id"] for unit in expected]
    seen: dict[str, MeasurementRecord] = {}
    for item in record_values:
        record = item if isinstance(item, MeasurementRecord) else MeasurementRecord.from_mapping(item)
        work_unit_id = record.value["work_unit_id"]
        if work_unit_id in seen:
            raise CorrectedG8EError("E3 duplicate measurement work unit")
        seen[work_unit_id] = record
    if set(seen) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(seen))
        extra = sorted(set(seen) - set(expected_ids))
        raise CorrectedG8EError(f"E3 exact-set mismatch: missing={missing[:3]}, extra={extra[:3]}")
    ordered = [seen[work_unit_id] for work_unit_id in expected_ids]
    if any(record.value["measurement_authority_id"] != authority["authority_id"] for record in ordered):
        raise CorrectedG8EError("E3 record authority differs")
    if production and any(record.value["record_labels"] for record in ordered):
        raise CorrectedG8EError("NON-SCIENTIFIC fixture records cannot enter production E3")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_e3_exact_merge",
        "status": "MERGED",
        "production": production,
        "work_unit_count": len(ordered),
        "work_unit_ids": expected_ids,
        "record_ids": [record.value["record_id"] for record in ordered],
        "record_sha256s": [sha256_bytes(rendered_json(record.as_dict())) for record in ordered],
        "coverage_digest": sha256_bytes(canonical_json(expected_ids)),
        "scientific_evidence": production,
        "merge_eligible": production,
    }


def aggregate_e4_counts(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    record_values: Sequence[Mapping[str, Any] | MeasurementRecord],
    production: bool = True,
) -> dict[str, Any]:
    """Construct one count-derived measured object per structural identity."""

    merged = merge_e3_records(authority=authority, sample_ids=sample_ids, record_values=record_values, production=production)
    records = [item if isinstance(item, MeasurementRecord) else MeasurementRecord.from_mapping(item) for item in record_values]
    by_structural: dict[str, list[MeasurementRecord]] = defaultdict(list)
    for record in records:
        by_structural[record.value["measurement_identity_id"]].append(record)
    initial = sorted(row["structural_identity_id"] for row in authority["structural_identities"] if row["dataset"] == INITIAL_DATASET)
    objects = []
    for structural_id in initial:
        rows = by_structural.get(structural_id, [])
        if len(rows) != len(sample_ids):
            raise CorrectedG8EError("E4 structural identity has missing measurement work units")
        infeasible = [row for row in rows if row.value["outcome"] in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY}]
        if infeasible:
            objects.append({
                "measurement_identity_id": structural_id,
                "status": "ineligible",
                "reason": "codec_or_structural_infeasibility_present",
                "correct_count": None,
                "total_count": None,
                "source_record_ids": [row.value["record_id"] for row in rows],
            })
            continue
        correct = sum(int(row.value["correct_count"]) for row in rows)
        total = sum(int(row.value["total_count"]) for row in rows)
        if total != len(sample_ids):
            raise CorrectedG8EError("E4 clean denominator does not cover every validation image")
        objects.append({
            "measurement_identity_id": structural_id,
            "status": "eligible",
            "correct_count": correct,
            "total_count": total,
            "accuracy_derivation": "sum(correct_count) / sum(total_count); no accuracy float is stored",
            "source_record_ids": [row.value["record_id"] for row in rows],
        })
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrected_e4_measured_codec_accuracy",
        "status": "COUNT_DERIVED",
        "production": production,
        "measurement_authority_id": authority["authority_id"],
        "objects": objects,
        "object_count": len(objects),
        "e3_coverage_digest": merged["coverage_digest"],
        "outage_term_included": False,
        "scientific_evidence": production,
        "merge_eligible": production,
    }


def authorization_scope_accepts(
    *,
    candidates: int,
    samples: int,
    authorization: Any | None,
) -> Any:
    from baseline.classical.composition import check_sweep_budget

    return check_sweep_budget(candidates=candidates, samples=samples, authorization=authorization)


def authenticate_owner_authorization(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    value, _raw = _read_json(path, "owner E2 authorization")
    required = {"schema_version", "artifact_role", "status", "authorized_by", "reason", "campaign_id", "contract_id", "source_manifest_id", "profile_id", "scope", "issued_sha256"}
    if set(value) != required or value["status"] != "AUTHORIZED":
        raise CorrectedG8EError("owner E2 authorization is absent or not active")
    if value["campaign_id"] != contract["campaign_id"] or value["contract_id"] != contract["contract_id"]:
        raise CorrectedG8EError("owner E2 authorization belongs to another corrected campaign")
    if value["source_manifest_id"] != contract["execution_source_manifest"]["source_manifest_id"] or value["profile_id"] != contract["execution_profile"]["profile_id"]:
        raise CorrectedG8EError("owner E2 authorization source/profile differs")
    body = {key: child for key, child in value.items() if key != "issued_sha256"}
    if value["issued_sha256"] != sha256_bytes(canonical_json(body)):
        raise CorrectedG8EError("owner E2 authorization digest differs")
    if not isinstance(value["scope"], Mapping) or value["scope"].get("validation_decode") is not True or value["scope"].get("test_access") is not False or value["scope"].get("training") is not False:
        raise CorrectedG8EError("owner E2 authorization scope opens forbidden work")
    return value


def reject_old_campaign(campaign_id: str) -> None:
    if campaign_id == OLD_CAMPAIGN_ID:
        raise CorrectedG8EError("superseded-before-data E1 campaign cannot execute E2")


__all__ = [
    "CORRECTED_ROOT", "CORRECTED_CONTRACT_PATH", "CORRECTED_AUTHORITY_PATH", "CORRECTED_MAPPING_PATH", "CORRECTED_SOURCE_MANIFEST_PATH", "CORRECTION_PROVENANCE_PATH", "CORRECTED_RUNTIME_ROOT",
    "CorrectedG8EError", "canonical_json", "rendered_json", "sha256_bytes", "sha256_file", "derive_measurement_authority", "derive_logical_measurement_mapping", "derive_selection_call_plan", "expected_work_units", "build_source_manifest", "build_correction_provenance", "build_corrected_contract", "build_corrected_bundle", "verify_corrected_bundle", "validate_corrected_contract", "PhysicalCacheKey", "physical_cache_key", "PhysicalCodecCache", "PhysicalReconstructionCache", "SyntheticSample", "MeasurementRecord", "MeasurementExecutor", "AtomicE2Campaign", "merge_e3_records", "aggregate_e4_counts", "clean_accuracy_from_records", "authorization_scope_accepts", "authenticate_owner_authorization", "reject_old_campaign", "OUTCOME_STRUCTURAL_INFEASIBILITY", "OUTCOME_CODEC_INFEASIBILITY", "OUTCOME_DECODE_FAILURE", "OUTCOME_DELIVERED", "PRODUCTION_PROFILE_ID", "PRODUCTION_DEVICE", "OLD_CAMPAIGN_ID", "OLD_CONTRACT_ID",
]
