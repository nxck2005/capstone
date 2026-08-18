"""G8_E corrected-v2 pre-data contract and production-scale execution core.

This module is an additive successor to ``g8_e_corrected``.  The first
corrective epoch remains immutable history; none of its permissive exception
handling or growing-prefix persistence is imported into the v2 scientific
path.

The v2 boundary has four deliberate properties:

* a valid codec return of ``feasible=False`` is an image-level BR-13 outage,
  not candidate invalidation;
* an unexpected backend, decoder, classifier, cache or publication error is a
  campaign HOLD and never a scientific record;
* a completed prefix is authenticated by compact counters and an
  order-sensitive rolling digest, so normal advancement is O(1); and
* start/resume, profile, contract and owner authorization are authenticated
  before the validation registry is opened.

The real owner authorization and the production runtime are intentionally not
present in this pre-data epoch.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config.params import REPO_ROOT, get


V2_ROOT = REPO_ROOT / "results/baseline/g8_e/e1_corrected_v2"
V2_CONTRACT_PATH = V2_ROOT / "measurement_contract.json"
V2_SOURCE_MANIFEST_PATH = V2_ROOT / "execution_source_manifest.json"
V2_CORRECTION_PATH = V2_ROOT / "correction_provenance.json"
V2_AUTHORITY_BINDING_PATH = V2_ROOT / "measurement_authority_binding.json"
V2_MAPPING_BINDING_PATH = V2_ROOT / "logical_measurement_mapping_binding.json"
V2_STORAGE_PLAN_PATH = V2_ROOT / "compute_storage_plan.json"
V2_SCALE_EVIDENCE_PATH = V2_ROOT / "transaction_scale_evidence.json"
V2_SYNTHETIC_PROOF_PATH = V2_ROOT / "synthetic_end_to_end_proof.json"
V2_RUNTIME_ROOT = V2_ROOT / "runtime"

V2_SCHEMA_VERSION = 2
V2_RECORD_SCHEMA_VERSION = 2
V2_STATE_SCHEMA_VERSION = 2
V2_AUTHORITY_BINDING_SCHEMA_VERSION = 1
V2_MAPPING_BINDING_SCHEMA_VERSION = 1
V2_SOURCE_MANIFEST_SCHEMA_VERSION = 1
V2_CONTRACT_PREFIX = "g8econtractcorrectedv2-"
V2_CAMPAIGN_PREFIX = "g8e-v2-"
V2_RECORD_PREFIX = "g8erecordcorrectedv2-"
V2_WORK_UNIT_PREFIX = "g8eworkv2-"
V2_PHYSICAL_PREFIX = "g8ephysicalv2-"
V2_CODEC_PREFIX = "g8ecodecv2-"
V2_RECONSTRUCTION_PREFIX = "g8ereconv2-"
V2_OBSERVATION_PREFIX = "g8eobservationv2-"
V2_DIAGNOSTIC_PREFIX = "g8ediagnosticv2-"
V2_CHAIN_PREFIX = "g8echainv2-"

INITIAL_DATASET = "imagenette160"
VALIDATION_SPLIT = "val"
PRODUCTION_PROFILE_ID = "local_4060_cu130"
PRODUCTION_DEVICE = "cuda:0"
ORIGINAL_CAMPAIGN_ID = "g8e-0037dfcbe2b679d8d0b09ff7116ed93a7e17099522481b7d4c1f1005d88e30bc"
FIRST_CORRECTED_CAMPAIGN_ID = "g8e-corrected-d55b30df0e9f580dfb8be7b19dc33f5b9092bdedad8725ca9a2ffd36814fdcd4"
FIRST_CORRECTED_CONTRACT_ID = "g8econtractcorrected-ab9c4c46be7f3bf58129274083f9a15fb0008a90a2f6b653a906b72a4efc3a39"

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

CHECKPOINT_INTERVAL = 4096  # literal-ok: frozen compact-checkpoint cadence, not a scientific parameter
HOLD_STATUS = "HOLD"
READY_STATUS = "READY"
RUNNING_STATUS = "RUNNING"
COMPLETE_STATUS = "COMPLETE"


class G8EV2Error(ValueError):
    """A v2 contract, provenance, record or cache violation."""


class FatalExecutionError(G8EV2Error):
    """An implementation/environment failure that must enter campaign HOLD."""


class CampaignHoldError(FatalExecutionError):
    """The active unit stopped without publishing scientific evidence."""


class ScientificDecodeFailure:
    """The only explicit decoder return that is a scientific decode outcome.

    Decoder exceptions are never converted into this object.  A test or a
    future backend must return this typed value deliberately to claim that the
    frozen protocol defines a decode failure for that backend.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise G8EV2Error("scientific decode failure needs a non-empty reason")
        self.reason = reason


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
        raise G8EV2Error(f"value is not canonical JSON: {exc}") from None


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
        raise G8EV2Error(f"cannot hash {path}: {exc}") from exc


def _id(prefix: str, value: Mapping[str, Any]) -> str:
    return prefix + sha256_bytes(canonical_json(value))


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64  # literal-ok: SHA-256 digest width
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise G8EV2Error(f"{label} is not a lowercase SHA-256")
    return value


def _rendered_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EV2Error(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != rendered_json(value):
        raise G8EV2Error(f"{label} is not canonical rendered JSON")
    return value, raw


def _strict(value: Any, fields: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        expected = set(fields)
        raise G8EV2Error(
            f"{label} schema differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise G8EV2Error(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise G8EV2Error(f"{label} must be a non-negative integer")
    return value


def _copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _old_authority_path() -> Path:
    return REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_authority.json"


def _old_mapping_path() -> Path:
    return REPO_ROOT / "results/baseline/g8_e/e1_corrected/logical_measurement_mapping.json"


def _validation_ids() -> tuple[str, ...]:
    """Read only the committed validation manifest identities.

    This helper is called by the runner only after authorization/profile
    authentication.  Unit tests pass synthetic IDs directly and never invoke
    it.
    """

    from data.manifests import manifest_path, validate_manifest_bytes

    path = manifest_path(INITIAL_DATASET, REPO_ROOT)
    rows = validate_manifest_bytes(INITIAL_DATASET, path.read_bytes())
    ids = tuple(row.stable_sample_id for row in rows if row.split == VALIDATION_SPLIT)
    if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise G8EV2Error("validation manifest IDs are not a sorted unique sequence")
    return ids


def _authority_binding() -> dict[str, Any]:
    authority, raw = _rendered_object(_old_authority_path(), "first corrected authority")
    expected = {
        "artifact_role": "g8_e_corrected_measurement_authority",
        "status": "FROZEN_PRE_DATA",
        "counts": {
            "logical_initial_snr_cells": 6048,  # literal-ok: immutable authority count
            "logical_all_roles_snr_cells": 12096,  # literal-ok: immutable authority count
            "structural_initial": 288,  # literal-ok: immutable authority count
            "snr_points": 21,  # literal-ok: immutable authority count
        },
    }
    if authority.get("artifact_role") != expected["artifact_role"] or authority.get("status") != expected["status"]:
        raise G8EV2Error("first corrected measurement authority is not frozen pre-data history")
    for key, expected_value in expected["counts"].items():
        if authority.get("counts", {}).get(key) != expected_value:
            raise G8EV2Error(f"first corrected authority count {key} differs")
    return {
        "path": str(_old_authority_path().relative_to(REPO_ROOT)),
        "sha256": sha256_bytes(raw),
        "authority_id": authority.get("authority_id"),
        "structural_digest": authority.get("structural_digest"),
        "counts": _copy(authority["counts"]),
        "identity_semantics": _copy(authority["identity_semantics"]),
        "reused_byte_identically": True,
    }


def _mapping_binding(authority_binding: Mapping[str, Any]) -> dict[str, Any]:
    mapping, raw = _rendered_object(_old_mapping_path(), "first corrected mapping")
    if mapping.get("status") not in (None, "FROZEN_PRE_DATA"):
        raise G8EV2Error("first corrected logical mapping is not pre-data history")
    if mapping.get("authority_id") != authority_binding["authority_id"]:
        raise G8EV2Error("first corrected mapping authority differs")
    if mapping.get("mapping_count") != 12096:  # literal-ok: immutable logical authority count
        raise G8EV2Error("first corrected mapping does not retain all logical cells")
    return {
        "path": str(_old_mapping_path().relative_to(REPO_ROOT)),
        "sha256": sha256_bytes(raw),
        "mapping_id": mapping.get("mapping_id"),
        "mapping_digest": mapping.get("mapping_digest"),
        "mapping_count": mapping.get("mapping_count"),
        "authority_id": mapping.get("authority_id"),
        "reused_byte_identically": True,
    }


def _direct_upstream_bindings() -> list[dict[str, Any]]:
    paths = (
        ("src/baseline/g8_pascal_merge.py", "g8_c_portable_merge_source"),
        ("src/baseline/g8_pascal_portable.py", "g8_c_portable_loader_source"),
        (
            "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json",
            "g8_c_portable_scientific_runtime_manifest",
        ),
        (
            "results/baseline/g8_pascal_successor/portable_verification_provenance.json",
            "g8_c_portable_verification_provenance",
        ),
        (
            "results/baseline/g8_pascal_successor/successor_bler_merge_report.json",
            "g8_c_frozen_successor_merge",
        ),
        (
            "results/baseline/g8_pascal_successor/successor_bler_table.json",
            "g8_c_frozen_successor_bler_table",
        ),
        (
            "results/baseline/g8_pascal_successor/successor_closeout_provenance.json",
            "g8_c_historical_c6_closeout_provenance",
        ),
        ("results/baseline/g8_d/measurement_contract.json", "g8_d_current_measurement_contract"),
        ("results/baseline/g8_d/d7_handoff.json", "g8_d_current_d7_handoff"),
        ("results/baseline/g8_d/portable_rebind_provenance.json", "g8_d_current_portable_rebind"),
    )
    result = []
    for path, role in paths:
        full = REPO_ROOT / path
        if not full.is_file():
            raise G8EV2Error(f"required direct upstream binding is missing: {path}")
        result.append({"path": path, "role": role, "bytes": len(full.read_bytes()), "sha256": sha256_file(full)})
    table, _ = _rendered_object(REPO_ROOT / paths[5][0], "frozen successor BlerTable")  # literal-ok: direct frozen artifact index
    portable, _ = _rendered_object(REPO_ROOT / paths[3][0], "portable G8_C provenance")  # literal-ok: direct frozen artifact index
    merge, _ = _rendered_object(REPO_ROOT / paths[4][0], "frozen successor merge")  # literal-ok: direct frozen artifact index
    if portable.get("epoch") != "g8-c-portable-scientific-runtime-v1":
        raise G8EV2Error("wrong G8_C portable verification epoch")
    if table.get("table_id") != "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f":
        raise G8EV2Error("wrong frozen successor BlerTable")
    if merge.get("accepted_count") != 3213 or table.get("measured_point_count") != 3213 or table.get("complete_identity_count") != 153:  # literal-ok: frozen G8_C successor coverage
        raise G8EV2Error("frozen G8_C successor coverage differs")
    return result


def _g1_bindings() -> list[dict[str, Any]]:
    paths = (
        ("results/reference_classifier/g1_adjudication.json", "g1_adjudication"),
        ("results/reference_classifier/best_checkpoint.json", "g1_checkpoint_binding"),
        ("results/reference_classifier/resolved_config.json", "g1_resolved_config"),
        ("results/reference_classifier/validation_summary.json", "g1_validation_summary"),
    )
    result = []
    for path, role in paths:
        full = REPO_ROOT / path
        if not full.is_file():
            raise G8EV2Error(f"G-1 binding is missing: {path}")
        result.append({"path": path, "role": role, "bytes": len(full.read_bytes()), "sha256": sha256_file(full)})
    return result


def _classifier_binding() -> dict[str, Any]:
    adjudication, _ = _rendered_object(REPO_ROOT / "results/reference_classifier/g1_adjudication.json", "G-1 classifier adjudication")
    checkpoint = adjudication.get("checkpoint_sha256")
    config_hash = adjudication.get("config_hash")
    _digest(checkpoint, "G-1 checkpoint SHA-256")
    _digest(config_hash, "G-1 config SHA-256")
    return {
        "checkpoint_sha256": checkpoint,
        "config_identity": f"g1-config-{config_hash}",
        "runtime_identity": "frozen-reference-classifier-" + sha256_file(REPO_ROOT / "src/models/frozen_reference_classifier.py"),
        "adjudication_path": "results/reference_classifier/g1_adjudication.json",
        "adjudication_sha256": sha256_file(REPO_ROOT / "results/reference_classifier/g1_adjudication.json"),
        "test_access": 0,
        "training": 0,
    }


def _old_contracts() -> dict[str, Any]:
    original, original_raw = _rendered_object(REPO_ROOT / "results/baseline/g8_e/measurement_contract.json", "original E1 contract")
    corrected, corrected_raw = _rendered_object(REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_contract.json", "first corrected E1 contract")
    if original.get("safety", {}).get("measurement_coverage") != 0 or corrected.get("safety", {}).get("measurement_coverage") != 0:
        raise G8EV2Error("a superseded E1 contract claims scientific coverage")
    return {
        "original": {"contract_id": original.get("contract_id"), "campaign_id": original.get("campaign_id"), "sha256": sha256_bytes(original_raw)},
        "first_corrected": {"contract_id": corrected.get("contract_id"), "campaign_id": corrected.get("campaign_id"), "sha256": sha256_bytes(corrected_raw)},
    }


def _selection_call_plan() -> dict[str, Any]:
    authority, _ = _rendered_object(REPO_ROOT / "results/baseline/g8_e/candidate_authority.json", "logical candidate authority")
    rows = [row for row in authority["candidates"] if row["dataset"] == INITIAL_DATASET]
    snr_values = sorted({row["snr_db"] for row in rows})
    ratios = sorted({row["ratio"] for row in rows})
    if len(rows) != 6048 or len(snr_values) != 21 or len(ratios) != 6:  # literal-ok: frozen logical E5 authority shape
        raise G8EV2Error("logical authority does not derive the expected E5 grid")
    per_snr = len(rows) // len(snr_values)
    calls = []
    for ratio in ratios:
        ratio_rows = [row for row in rows if row["ratio"] == ratio]
        candidates_per_snr = len(ratio_rows) // len(snr_values)
        for mode in ("classical_adaptive", "classical_fixed_mod", "classical_fixed_mcs"):
            calls.append({
                "dataset": INITIAL_DATASET,
                "ratio": ratio,
                "mode": mode,
                "snr_groups": len(snr_values),
                "candidates_per_snr": candidates_per_snr,
                "candidate_count": len(ratio_rows),
                "samples_per_cell": int(get("datasets.imagenette160.val_images")),
            })
    return {
        "derived_from": "logical authority plus frozen SYSTEM_MODES and validation manifest count",
        "calls": calls,
        "call_count": len(calls),
        "max_candidates": max(call["candidate_count"] for call in calls),
        "max_samples": max(call["samples_per_cell"] for call in calls),
        "max_workload": max(call["candidate_count"] * call["samples_per_cell"] for call in calls),
        "old_pb3_guard": {"max_candidates": 64, "max_samples": 25, "max_workload": 512},  # literal-ok: historical PB_3 guard
        "typed_g8_authorization_required": True,
        "authorization_issued": False,
    }


def _physical_plan(authority_binding: Mapping[str, Any]) -> dict[str, Any]:
    authority, _ = _rendered_object(_old_authority_path(), "measurement authority")
    rows = [row for row in authority["structural_identities"] if row["dataset"] == INITIAL_DATASET]
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in rows:
        groups[(int(row["payload_budget_bytes"]), int(row["encode_axis_px"]))].append(row["structural_identity_id"])
    ordered = [
        {"payload_budget_bytes": budget, "encode_axis_px": axis, "structural_identity_ids": sorted(ids)}
        for (budget, axis), ids in sorted(groups.items())
    ]
    if not ordered:
        raise G8EV2Error("physical cache plan is empty")
    return {
        "structural_initial": len(rows),
        "validation_images": int(get("datasets.imagenette160.val_images")),
        "work_units": len(rows) * int(get("datasets.imagenette160.val_images")),
        "unique_physical_jobs": len(ordered) * int(get("datasets.imagenette160.val_images")),
        "unique_classifier_observations_upper_bound": len(ordered) * int(get("datasets.imagenette160.val_images")),
        "physical_keys_per_image": len(ordered),
        "equivalence_groups": ordered,
        "reuse_predicate": [
            "source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape",
            "payload_budget_bytes", "encode_axis_px", "codec_configuration_hash",
            "codec_runtime_identity",
        ],
        "structural_identity_never_merged": True,
        "authority_id": authority_binding["authority_id"],
    }


def _json_size(value: Any) -> int:
    return len(rendered_json(value))


def build_storage_plan() -> dict[str, Any]:
    """Estimate storage from representative rendered JSON/base64 objects."""

    physical = _physical_plan(_authority_binding())
    units = physical["work_units"]
    jobs = physical["unique_physical_jobs"]
    image_bytes = 160 * 160 * 3  # literal-ok: Imagenette-160 serialized reconstruction shape
    codestream_bytes = 4096  # literal-ok: representative codec serialization size
    record_template = {
        "schema_version": V2_RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_scientific_measurement_record",
        "record_id": V2_RECORD_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "campaign_id": V2_CAMPAIGN_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "contract_id": V2_CONTRACT_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "measurement_authority_id": "g8emeasurementauthority-" + "0" * 64,  # literal-ok: representative digest-width fixture
        "authority_ordinal": 0,  # literal-ok: representative ordinal fixture
        "measurement_identity_id": "g8estruct-" + "0" * 64,  # literal-ok: representative digest-width fixture
        "logical_candidate_ids": ["cand-" + "0" * 64] * 21,  # literal-ok: representative logical fanout
        "work_unit_id": V2_WORK_UNIT_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "stable_sample_id": "0" * 16,  # literal-ok: representative stable-ID fixture
        "dataset": INITIAL_DATASET,
        "split": VALIDATION_SPLIT,
        "label": 0,  # literal-ok: representative class-label fixture
        "source_bytes_sha256": "0" * 64,  # literal-ok: representative digest-width fixture
        "canonical_pixels_sha256": "0" * 64,  # literal-ok: representative digest-width fixture
        "canonical_shape": [160, 160, 3],  # literal-ok: representative Imagenette-160 shape
        "structural_identity": {"payload_budget_bytes": 100, "encode_axis_px": 160},  # literal-ok: representative serialized row
        "packet_budget": {"payload_budget_bytes": 100},  # literal-ok: representative serialized row
        "physical_cache_key": {"payload_budget_bytes": 100, "encode_axis_px": 160},  # literal-ok: representative serialized row
        "codec_cache_object_id": V2_CODEC_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "outcome": OUTCOME_DELIVERED,
        "failure_stage": None,
        "emitted_codestream": {"sha256": "0" * 64, "bytes": codestream_bytes},  # literal-ok: representative digest-width fixture
        "reconstruction": {"object_id": V2_RECONSTRUCTION_PREFIX + "0" * 64, "sha256": "0" * 64},  # literal-ok: representative digest-width fixture
        "classifier_observation": {"object_id": V2_OBSERVATION_PREFIX + "0" * 64, "predicted_label": 0, "label": 0},  # literal-ok: representative class-label fixture
        "outage_prediction": None,
        "correct_count": 1,
        "total_count": 1,
        "br11": {"emitted_codestream_bytes": codestream_bytes, "header_bytes": 160, "payload_bytes": codestream_bytes - 160, "payload_filler_bytes": 100 - codestream_bytes},  # literal-ok: representative AM-81 accounting row
        "g8_c_linkage_digest": "0" * 64,  # literal-ok: representative digest-width fixture
        "profile_id": PRODUCTION_PROFILE_ID,
        "source_commit": "0" * 40,  # literal-ok: representative Git identity width
        "validation_only": True,
        "outage_applied": False,
        "scientific_evidence": True,
        "merge_eligible": True,
        "test_access": 0,
        "training": 0,
        "inference": 1,
        "record_labels": [],
    }
    codec_template = {
        "schema_version": V2_RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_codec_cache_object",
        "key": {"payload_budget_bytes": 100, "encode_axis_px": 160},  # literal-ok: representative serialized cache key
        "status": "feasible",
        "codestream_b64": base64.b64encode(b"0" * codestream_bytes).decode("ascii"),
        "codestream_sha256": "0" * 64,  # literal-ok: representative digest-width fixture
    }
    recon_template = {
        "schema_version": V2_RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_reconstruction_cache_object",
        "identity": {"physical_cache_key": {"payload_budget_bytes": 100}},  # literal-ok: representative cache identity
        "status": "delivered",
        "pixels_b64": base64.b64encode(b"0" * image_bytes).decode("ascii"),  # literal-ok: representative base64 reconstruction payload
        "pixels_sha256": "0" * 64,  # literal-ok: representative digest-width fixture
    }
    observation_template = {
        "schema_version": V2_RECORD_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_classifier_observation_cache_object",
        "identity": {"reconstruction_object_id": V2_RECONSTRUCTION_PREFIX + "0" * 64},  # literal-ok: representative observation identity
        "predicted_label": 0,  # literal-ok: representative class label
        "object_id": V2_OBSERVATION_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "object_sha256": "0" * 64,  # literal-ok: representative digest-width fixture
    }
    state_template = {
        "schema_version": V2_STATE_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_campaign_state",
        "campaign_id": V2_CAMPAIGN_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "contract_id": V2_CONTRACT_PREFIX + "0" * 64,  # literal-ok: representative digest-width fixture
        "authority_id": "g8emeasurementauthority-" + "0" * 64,  # literal-ok: representative digest-width fixture
        "total_required": units,  # literal-ok: runtime authority count is derived above
        "completed_prefix_count": 0,
        "last_completed_work_unit_id": None,
        "rolling_prefix_digest": "0" * 64,  # literal-ok: representative SHA-256 width
        "counters": {"validation_decoding": 0, "inference": 0, "training": 0, "test_access": 0},  # literal-ok: zero-data state fixture
        "status": READY_STATUS,
        "in_progress": None,
        "last_checkpoint": None,
    }
    e3_entry_size = _json_size({"ordinal": 0, "work_unit_id": V2_WORK_UNIT_PREFIX + "0" * 64, "record_id": V2_RECORD_PREFIX + "0" * 64, "record_sha256": "0" * 64})  # literal-ok: representative E3 entry
    e4_entry_size = _json_size({"measurement_identity_id": "g8estruct-" + "0" * 64, "correct_count": 1000, "total_count": 1000, "source_record_ids": [V2_RECORD_PREFIX + "0" * 64] * 1000})  # literal-ok: representative E4 source list
    record_bytes = _json_size(record_template)
    codec_bytes = _json_size(codec_template)
    recon_bytes = _json_size(recon_template)
    observation_bytes = _json_size(observation_template)
    state_bytes = _json_size(state_template)
    scientific_records = record_bytes * units
    codec_storage = codec_bytes * jobs
    reconstruction_storage = recon_bytes * jobs
    observation_storage = observation_bytes * jobs
    state_storage = state_bytes + (_json_size({"checkpoint": 0, "digest": "0" * 64, "counters": state_template["counters"]}) * ((units + CHECKPOINT_INTERVAL - 1) // CHECKPOINT_INTERVAL))  # literal-ok: compact checkpoint estimate
    e3_storage = e3_entry_size * units
    e4_storage = e4_entry_size * len(physical["equivalence_groups"])
    subtotal = scientific_records + codec_storage + reconstruction_storage + observation_storage + state_storage + e3_storage + e4_storage
    safety_margin = int(subtotal * 0.25)  # literal-ok: frozen 25-percent disk headroom
    return {
        "basis": "representative rendered JSON with base64 payloads; planning estimate only",
        "production_units": units,
        "unique_physical_jobs": jobs,
        "representative_sizes_bytes": {
            "scientific_record": record_bytes,
            "codec_cache_object": codec_bytes,
            "reconstruction_cache_object": recon_bytes,
            "classifier_observation_object": observation_bytes,
            "compact_state": state_bytes,
            "e3_entry": e3_entry_size,
            "e4_entry": e4_entry_size,
        },
        "estimated_bytes": {
            "scientific_records": scientific_records,
            "codec_cache": codec_storage,
            "reconstruction_cache_base64": reconstruction_storage,
            "classifier_observation_cache": observation_storage,
            "runtime_state_and_checkpoints": state_storage,
            "e3_output": e3_storage,
            "e4_output": e4_storage,
            "subtotal": subtotal,
            "safety_margin_25_percent": safety_margin,
            "required_with_safety_margin": subtotal + safety_margin,
        },
        "base64_expansion_accounted": True,
        "no_prior_evidence_deleted": True,
        "preflight_required_before_validation_decode": True,
    }


def storage_preflight(plan: Mapping[str, Any], path: Path = V2_RUNTIME_ROOT) -> dict[str, Any]:
    required = int(plan["estimated_bytes"]["required_with_safety_margin"])
    target = Path(path).resolve()
    usage = shutil.disk_usage(target.parent if target.parent.exists() else REPO_ROOT)
    available = int(usage.free)
    result = {"path": str(target), "required_bytes": required, "available_free_bytes": available, "passed": available >= required}
    if not result["passed"]:
        raise G8EV2Error(f"storage preflight failed: need {required} bytes, have {available}")
    return result


def build_correction_provenance() -> dict[str, Any]:
    old = _old_contracts()
    authority = _authority_binding()
    mapping = _mapping_binding(authority)
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrective_v2_provenance",
        "status": "CURRENT_PRE_DATA_SUPERSESSION",
        "scientific_measurement_coverage": 0,
        "reason": [
            "per-image codec/source infeasibility was incorrectly candidate-fatal",
            "unexpected backend/decoder/classifier/cache failures were laundered into scientific outcomes",
            "normal transaction advancement revalidated and reaggregated the full prefix",
            "--start and --resume were not distinct runtime state transitions",
            "classifier observations were repeated for exact reconstruction reuse",
        ],
        "preserved_superseded_epochs": {
            "original_e1": old["original"],
            "first_corrected_e1": old["first_corrected"],
            "first_corrected_artifact_root": "results/baseline/g8_e/e1_corrected",
        },
        "authority_reuse": {"authority_id": authority["authority_id"], "authority_sha256": authority["sha256"], "mapping_id": mapping["mapping_id"], "mapping_sha256": mapping["sha256"]},
        "zero_data_audit": {
            "e2_records": 0,
            "e2_completed_units": 0,
            "e3_merge_present": False,
            "e4_objects_present": False,
            "pass_one_pre_marker": False,
            "pass_one_completion": False,
            "training": 0,
            "pass_two": 0,
            "fallback_invocation": False,
            "ratio_adjudication": False,
            "test_access": 0,
            "full_validation_payload_decodes": 0,
        },
        "classification": "both previous E1 epochs remain immutable superseded-before-data history; no scientific invalidation",
    }


def build_source_manifest(source_commit: str) -> dict[str, Any]:
    paths = (
        ("src/baseline/g8_e_corrected_v2.py", "current_v2_e2_e3_e4_runtime"),
        ("tools/run_g8_e_corrected_v2.py", "current_v2_owner_gated_runner"),
        ("tools/merge_g8_e_corrected_v2.py", "current_v2_independent_e3"),
        ("tools/aggregate_g8_e_corrected_v2.py", "current_v2_count_derived_e4"),
        ("tools/verify_g8_e_corrected_v2.py", "current_v2_independent_verifier"),
        ("tools/benchmark_g8_e_v2_transaction.py", "current_v2_transaction_scale_benchmark"),
        ("tools/prove_g8_e_corrected_v2_synthetic.py", "current_v2_synthetic_proof_harness"),
        ("src/baseline/classical/composition.py", "br4_measured_composition"),
        ("src/baseline/classical/records.py", "br11_all_row_semantics"),
        ("src/baseline/classical/pipeline.py", "classical_verdict_taxonomy"),
        ("src/baseline/classical/outage.py", "frozen_outage_policy"),
        ("src/baseline/classical/channel_transport.py", "transport_accounting"),
        ("src/baseline/j2k.py", "jpeg2000_backend"),
        ("src/baseline/ldpc/transport.py", "packet_plan"),
        ("src/baseline/ldpc/segmentation.py", "packet_segmentation"),
        ("src/baseline/ldpc/rate_matching.py", "rate_matching"),
        ("src/baseline/ldpc/modulation.py", "modulation"),
        ("src/data/manifests.py", "validation_manifest_identity"),
        ("src/data/identity.py", "stable_sample_identity"),
        ("src/data/preprocessing.py", "canonical_codec_preprocessing"),
        ("src/data/registry.py", "validation_registry_boundary"),
        ("src/data/test_access.py", "sealed_test_boundary"),
        ("src/models/frozen_reference_classifier.py", "frozen_g1_loader"),
        ("src/models/reference_classifier.py", "g1_model_architecture"),
        ("src/config/execution_profiles.py", "authenticated_profile"),
        ("src/config/run_config.py", "run_configuration_identity"),
        ("src/config/params.py", "generated_parameters"),
        ("src/env.py", "runtime_environment"),
    )
    entries = []
    for path, role in paths:
        full = REPO_ROOT / path
        if not full.is_file():
            raise G8EV2Error(f"v2 source binding is missing: {path}")
        entries.append({"path": path, "role": role, "bytes": len(full.read_bytes()), "sha256": sha256_file(full)})
    direct = _direct_upstream_bindings()
    entries.extend(direct)
    entries.extend(_g1_bindings())
    body = {
        "schema_version": V2_SOURCE_MANIFEST_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrective_v2_execution_source_manifest",
        "checkpoint": "E1_corrected_v2",
        "status": "FROZEN_PRE_DATA",
        "source_commit": source_commit,
        "source_entries": entries,
        "direct_g8_c_portable_binding": True,
        "direct_g8_d_current_binding": True,
        "scientific_source_closure": {
            "runner_is_start_resume_state_machine": True,
            "transaction_normal_advancement_is_compact": True,
            "unexpected_exception_is_campaign_hold": True,
            "e3_is_independent_exact_set_scan": True,
            "e4_is_count_derived": True,
            "source_drift_is_hold": True,
        },
        "excludes": ["results/baseline/g8_e/e1_corrected_v2/runtime/", "results/baseline/g8_e/e1_corrected_v2/e2_execution_authorization.json"],
    }
    body["source_manifest_id"] = _id("g8esourcecorrectedv2-", body)
    return body


def build_contract(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    authority = _authority_binding()
    mapping = _mapping_binding(authority)
    direct = _direct_upstream_bindings()
    old = _old_contracts()
    selection = _selection_call_plan()
    physical = _physical_plan(authority)
    storage = build_storage_plan()
    classifier = _classifier_binding()
    outage_artifact, _ = _rendered_object(REPO_ROOT / "results/baseline/w4/outage_policy.json", "frozen W4 outage policy")
    profile_auth = _rendered_object(REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_contract.json", "first corrected profile contract")[0]["execution_profile"]
    g1 = _g1_bindings()
    seed = {
        "schema_version": V2_SCHEMA_VERSION,
        "semantics_epoch": "g8_e_corrected_v2_image_outage_compact_transaction",
        "old_contracts": old,
        "authority_id": authority["authority_id"],
        "authority_sha256": authority["sha256"],
        "mapping_id": mapping["mapping_id"],
        "mapping_sha256": mapping["sha256"],
        "source_manifest_id": source_manifest["source_manifest_id"],
        "source_manifest_sha256": sha256_bytes(rendered_json(source_manifest)),
    }
    campaign_id = _id(V2_CAMPAIGN_PREFIX, seed)
    body: dict[str, Any] = {
        "schema_version": V2_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrective_v2_executable_pre_data_contract",
        "phase": "G8_E",
        "checkpoint": "E1_corrected_v2",
        "status": "FROZEN_PRE_DATA_EXECUTABLE",
        "campaign_id": campaign_id,
        "campaign_seed": seed,
        "contract_id": None,
        "supersedes_before_data": {"original_e1": old["original"], "first_corrected_e1": old["first_corrected"]},
        "authority": authority,
        "mapping": mapping,
        "direct_upstream_bindings": direct,
        "g1_bindings": g1,
        "source_manifest": {"path": str(V2_SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT)), "id": source_manifest["source_manifest_id"], "sha256": sha256_bytes(rendered_json(source_manifest)), "source_commit": source_manifest["source_commit"]},
        "execution_profile": {
            "profile_id": PRODUCTION_PROFILE_ID,
            "device": PRODUCTION_DEVICE,
            "config_hash": profile_auth["config_hash"],
            "lock_file": profile_auth["lock_file"],
            "lock_file_sha256": profile_auth["lock_file_sha256"],
            "sole_writer": "local",
            "profile_frozen_before_first_measurement": True,
            "opportunistic_host_change_forbidden": True,
        },
        "codec": {
            "configuration_hash": sha256_bytes(canonical_json(get("baseline"))),
            "runtime_identity": str(get("environment.openjpeg")),
            "physical_key_fields": ["source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape", "payload_budget_bytes", "encode_axis_px", "codec_configuration_hash", "codec_runtime_identity"],
            "cross_mcs_reuse_requires_all_physical_fields_equal": True,
            "structural_identity_is_never_merged_by_cache": True,
        },
        "classifier": classifier,
        "outage_policy": {
            "path": "results/baseline/w4/outage_policy.json",
            "sha256": sha256_file(REPO_ROOT / "results/baseline/w4/outage_policy.json"),
            "selected_class": outage_artifact["selected_class"],
            "numerator": outage_artifact["numerator"],
            "denominator": outage_artifact["denominator"],
            "class_counts": outage_artifact["class_counts"],
            "selection_is_count_derived": True,
            "applies_to": [OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY, OUTCOME_DECODE_FAILURE],
        },
        "clean_measurement_semantics": {
            "delivered": "classifier observation gives one binary correct_count and total_count=1",
            "codec_infeasibility": "valid feasible=false backend return gives one BR-13 outage prediction, binary count and total_count=1; it does not erase the structural candidate",
            "decode_failure": "only explicit ScientificDecodeFailure return gives one BR-13 outage prediction; decoder exceptions are HOLD",
            "structural_infeasibility": "preserve frozen candidate-level ineligibility semantics; no codec call is made",
            "acc_clean": "sum every represented per-image correct_count, including image-level outage predictions, over the required validation denominator",
            "acc_outage": "separate measured frozen constant-class object from BR-4 outage_policy",
            "br4": "P(TB success) * acc_clean + (1 - P(TB success)) * acc_outage",
            "no_rows_dropped": True,
            "no_accuracy_float_authority": True,
        },
        "transaction": {
            "state_schema_version": V2_STATE_SCHEMA_VERSION,
            "normal_advancement": "O(1) with respect to completed prefix",
            "rolling_chain": "H0=SHA256(canonical campaign/contract/authority seed); Hi=SHA256(canonical previous_digest, authority_ordinal, work_unit_id, record_sha256)",
            "compact_state_fields": ["campaign_id", "contract_id", "authority_id", "total_required", "completed_prefix_count", "last_completed_work_unit_id", "rolling_prefix_digest", "counters", "in_progress", "last_checkpoint", "status"],
            "resume_reconciliation": "one startup scan of durable records, exact ordered prefix, independent record authentication",
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "checkpoint_is_compact": True,
            "full_prefix_aggregate_per_unit": False,
            "sole_writer_lock": True,
            "atomic_record_publication": True,
            "atomic_state_publication": True,
            "unexpected_failure_state": HOLD_STATUS,
        },
        "runner": {
            "start": {"authorization": True, "contract": True, "profile": True, "runtime_absent": True, "validation_decode_before_checks": False, "existing_runtime_refused": True},
            "resume": {"authorization": True, "contract": True, "profile": True, "runtime_present": True, "validation_decode_before_checks": False, "missing_runtime_refused": True, "foreign_campaign_refused": True},
            "first_command": "tools/run_g8_e_corrected_v2.py --start --campaign-id <current-v2-campaign> --authorization <owner-artifact>",
            "restart_command": "tools/run_g8_e_corrected_v2.py --resume --campaign-id <current-v2-campaign> --authorization <owner-artifact>",
        },
        "selection_authorization": selection,
        "compute_plan": {"physical": physical, "storage": storage, "classifier_observation_cache": True, "classifier_forwards_upper_bound": physical["unique_classifier_observations_upper_bound"]},
        "authorization": {"required": True, "issued": False, "path": "results/baseline/g8_e/e1_corrected_v2/e2_execution_authorization.json", "refuse_before_validation_decode": True, "schema_scope_frozen": {"validation_decode": True, "test_access": False, "training": False, "fallback": False, "ratio_adjudication": False, "pass_one": False, "pass_two": False}},
        "safety": {"measurement_coverage": 0, "e2_completed_units": 0, "e3_present": False, "e4_present": False, "pass_one_started": False, "pass_one_completed": False, "training": 0, "pass_two": 0, "fallback_invoked": False, "ratio_adjudicated": False, "test_access": 0, "validation_decoding": 0},
        "declarations": {"zero_full_validation_measurements": True, "e2_awaits_owner_authorization": True, "first_corrected_epoch_is_superseded_before_data": True, "original_epoch_is_superseded_before_data": True, "g8_c_science_unchanged": True, "g8_d_science_unchanged": True, "test_split_sealed": True},
    }
    body["contract_id"] = _id(V2_CONTRACT_PREFIX, {key: value for key, value in body.items() if key != "contract_id"})
    return body


def build_bundle(source_commit: str) -> dict[str, dict[str, Any]]:
    source = build_source_manifest(source_commit)
    authority = _authority_binding()
    mapping = _mapping_binding(authority)
    correction = build_correction_provenance()
    storage = build_storage_plan()
    contract = build_contract(source)
    return {"source_manifest": source, "authority_binding": authority, "mapping_binding": mapping, "correction_provenance": correction, "storage_plan": storage, "measurement_contract": contract}


def validate_source_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "artifact_role", "checkpoint", "status", "source_commit", "source_entries", "direct_g8_c_portable_binding", "direct_g8_d_current_binding", "scientific_source_closure", "excludes", "source_manifest_id"}
    if set(value) != required:
        raise G8EV2Error("v2 source manifest schema differs")
    body = {key: child for key, child in value.items() if key != "source_manifest_id"}
    if value["source_manifest_id"] != _id("g8esourcecorrectedv2-", body):
        raise G8EV2Error("v2 source manifest ID differs")
    for entry in value["source_entries"]:
        data = _strict(entry, ("path", "role", "bytes", "sha256"), "v2 source entry")
        path = REPO_ROOT / data["path"]
        if not path.is_file() or len(path.read_bytes()) != data["bytes"] or sha256_file(path) != data["sha256"]:
            raise G8EV2Error(f"v2 source drift: {data['path']}")
    if value["direct_g8_c_portable_binding"] is not True or value["direct_g8_d_current_binding"] is not True:
        raise G8EV2Error("v2 direct upstream binding declarations are false")
    return dict(value)


def validate_contract(value: Mapping[str, Any], *, verify_live_sources: bool = True) -> dict[str, Any]:
    if value.get("schema_version") != V2_SCHEMA_VERSION or value.get("checkpoint") != "E1_corrected_v2" or value.get("status") != "FROZEN_PRE_DATA_EXECUTABLE":
        raise G8EV2Error("v2 contract is not the frozen current pre-data epoch")
    body = {key: child for key, child in value.items() if key != "contract_id"}
    if value.get("contract_id") != _id(V2_CONTRACT_PREFIX, body):
        raise G8EV2Error("v2 contract ID differs")
    if value.get("campaign_id") in {ORIGINAL_CAMPAIGN_ID, FIRST_CORRECTED_CAMPAIGN_ID}:
        raise G8EV2Error("a superseded E1 campaign remains current")
    old = value.get("supersedes_before_data")
    if old is None or old["first_corrected_e1"]["contract_id"] != FIRST_CORRECTED_CONTRACT_ID:
        raise G8EV2Error("v2 contract does not preserve first corrected E1 history")
    authority = _authority_binding()
    mapping = _mapping_binding(authority)
    if value.get("authority") != authority or value.get("mapping") != mapping:
        raise G8EV2Error("v2 authority binding differs from byte-identical history")
    source, _ = _rendered_object(V2_SOURCE_MANIFEST_PATH, "v2 source manifest")
    if value.get("source_manifest", {}).get("id") != source.get("source_manifest_id"):
        raise G8EV2Error("v2 contract source-manifest ID differs")
    if verify_live_sources:
        validate_source_manifest(source)
    if value.get("authorization", {}).get("issued") is not False or value.get("safety", {}).get("measurement_coverage") != 0:
        raise G8EV2Error("v2 pre-data authorization/safety boundary is open")
    if value.get("declarations", {}).get("test_split_sealed") is not True:
        raise G8EV2Error("v2 contract releases test split")
    _direct_upstream_bindings()
    return dict(value)


def verify_bundle(*, verify_live_sources: bool = True) -> dict[str, Any]:
    contract, contract_raw = _rendered_object(V2_CONTRACT_PATH, "v2 measurement contract")
    source, source_raw = _rendered_object(V2_SOURCE_MANIFEST_PATH, "v2 source manifest")
    correction, correction_raw = _rendered_object(V2_CORRECTION_PATH, "v2 correction provenance")
    authority, authority_raw = _rendered_object(V2_AUTHORITY_BINDING_PATH, "v2 authority binding")
    mapping, mapping_raw = _rendered_object(V2_MAPPING_BINDING_PATH, "v2 mapping binding")
    storage, storage_raw = _rendered_object(V2_STORAGE_PLAN_PATH, "v2 storage plan")
    if verify_live_sources:
        validate_source_manifest(source)
    validate_contract(contract, verify_live_sources=verify_live_sources)
    expected = build_bundle(source["source_commit"])
    if authority != expected["authority_binding"] or mapping != expected["mapping_binding"] or correction != expected["correction_provenance"] or storage != expected["storage_plan"] or contract != expected["measurement_contract"]:
        raise G8EV2Error("v2 pre-data artifact bytes are stale or not derived from current frozen inputs")
    if V2_RUNTIME_ROOT.exists() or (V2_ROOT / "e2_execution_authorization.json").exists():
        raise G8EV2Error("v2 production runtime or owner authorization exists in pre-data freeze")
    return {"contract": contract, "source_manifest": source, "correction_provenance": correction, "authority_binding": authority, "mapping_binding": mapping, "storage_plan": storage, "contract_sha256": sha256_bytes(contract_raw), "source_manifest_sha256": sha256_bytes(source_raw), "correction_provenance_sha256": sha256_bytes(correction_raw), "authority_binding_sha256": sha256_bytes(authority_raw), "mapping_binding_sha256": sha256_bytes(mapping_raw), "storage_plan_sha256": sha256_bytes(storage_raw)}


@dataclass(frozen=True)
class PhysicalCacheKey:
    """The complete identity under which physical work may be reused."""

    source_bytes_sha256: str
    canonical_pixels_sha256: str
    canonical_shape: tuple[int, int, int]
    payload_budget_bytes: int
    encode_axis_px: int
    codec_configuration_hash: str
    codec_runtime_identity: str

    def __post_init__(self) -> None:
        for field in ("source_bytes_sha256", "canonical_pixels_sha256", "codec_configuration_hash"):
            _digest(getattr(self, field), field)
        if len(self.canonical_shape) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.canonical_shape
        ):
            raise G8EV2Error("physical cache shape must be a positive HWC tuple")
        _positive_int(self.payload_budget_bytes, "physical cache payload budget")
        _positive_int(self.encode_axis_px, "physical cache encode axis")
        if not isinstance(self.codec_runtime_identity, str) or not self.codec_runtime_identity:
            raise G8EV2Error("physical cache runtime identity is empty")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": V2_SCHEMA_VERSION,
            "identity_type": "g8_e_v2_physical_cache",
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
        return _id(V2_PHYSICAL_PREFIX, self.payload())


def make_physical_cache_key(
    *,
    source_bytes: bytes,
    canonical_pixels: np.ndarray,
    payload_budget_bytes: int,
    encode_axis_px: int,
    codec_configuration_hash: str,
    codec_runtime_identity: str,
) -> PhysicalCacheKey:
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise G8EV2Error("source bytes are required for a physical cache key")
    if (
        not isinstance(canonical_pixels, np.ndarray)
        or canonical_pixels.dtype != np.uint8
        or canonical_pixels.ndim != 3
        or canonical_pixels.shape[2] != 3
    ):
        raise G8EV2Error("canonical pixels must be uint8 RGB HWC")
    pixels = np.ascontiguousarray(canonical_pixels)
    return PhysicalCacheKey(
        source_bytes_sha256=sha256_bytes(source_bytes),
        canonical_pixels_sha256=sha256_bytes(pixels.tobytes()),
        canonical_shape=tuple(int(value) for value in pixels.shape),
        payload_budget_bytes=int(payload_budget_bytes),
        encode_axis_px=int(encode_axis_px),
        codec_configuration_hash=str(codec_configuration_hash),
        codec_runtime_identity=str(codec_runtime_identity),
    )


@dataclass(frozen=True)
class CodecArtifactV2:
    key: PhysicalCacheKey
    status: str
    reason: str | None
    codestream: bytes | None
    emitted_byte_count: int | None
    cache_object_id: str
    cache_hit: bool


@dataclass(frozen=True)
class ReconstructionArtifactV2:
    object_id: str
    status: str
    reason: str | None
    pixels: np.ndarray | None
    cache_hit: bool


@dataclass(frozen=True)
class ClassifierObservationV2:
    object_id: str
    predicted_label: int
    cache_hit: bool


class PhysicalCodecCacheV2:
    """Content-addressed codec cache with a typed infeasibility boundary."""

    def __init__(self, root: Path, backend: Any) -> None:
        self.root = Path(root).resolve()
        self.backend = backend

    def _path(self, key: PhysicalCacheKey) -> Path:
        return self.root / "codec" / f"{key.key_id}.json"

    def _load(self, path: Path, key: PhysicalCacheKey) -> CodecArtifactV2:
        try:
            value, _ = _rendered_object(path, "v2 codec cache object")
        except G8EV2Error as exc:
            raise FatalExecutionError(f"corrupt codec cache object: {exc}") from exc
        required = {
            "schema_version", "artifact_role", "key", "status", "reason",
            "codestream_b64", "codestream_sha256", "emitted_byte_count",
            "cache_object_id",
        }
        if set(value) != required or value["key"] != key.payload():
            raise FatalExecutionError("corrupt codec cache object or physical key")
        if value["schema_version"] != V2_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_codec_cache_object":
            raise FatalExecutionError("codec cache schema differs")
        status = value["status"]
        if status not in {"feasible", OUTCOME_CODEC_INFEASIBILITY}:
            raise FatalExecutionError("codec cache status is not a frozen outcome")
        stream: bytes | None = None
        if value["codestream_b64"] is not None:
            if status != "feasible" or not isinstance(value["codestream_b64"], str):
                raise FatalExecutionError("infeasible codec cache contains a codestream")
            try:
                stream = base64.b64decode(value["codestream_b64"], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise FatalExecutionError(f"codec cache base64 is corrupt: {exc}") from None
            if value["codestream_sha256"] != sha256_bytes(stream):
                raise FatalExecutionError("codec cache codestream digest differs")
            if value["emitted_byte_count"] != len(stream) or len(stream) > key.payload_budget_bytes:
                raise FatalExecutionError("codec cache emitted byte accounting differs")
        elif status == "feasible" or value["emitted_byte_count"] is not None or value["codestream_sha256"] is not None:
            raise FatalExecutionError("feasible codec cache has no valid emitted bytes")
        body = {field: value[field] for field in required if field != "cache_object_id"}
        if value["cache_object_id"] != _id(V2_CODEC_PREFIX, body):
            raise FatalExecutionError("codec cache object ID differs")
        return CodecArtifactV2(key, status, value["reason"], stream, value["emitted_byte_count"], value["cache_object_id"], True)

    def get_or_create(self, key: PhysicalCacheKey, encoded_pixels: np.ndarray) -> CodecArtifactV2:
        path = self._path(key)
        if path.exists():
            return self._load(path, key)
        if not isinstance(encoded_pixels, np.ndarray) or encoded_pixels.dtype != np.uint8 or encoded_pixels.ndim != 3 or encoded_pixels.shape[2] != 3:
            raise FatalExecutionError("codec input violates the uint8 RGB contract")
        try:
            result = self.backend.encode_to_budget(
                np.ascontiguousarray(encoded_pixels),
                canonical_pixels_sha256=key.canonical_pixels_sha256,
                budget_bytes=key.payload_budget_bytes,
                encode_axis_px=key.encode_axis_px,
            )
            feasible = getattr(result, "feasible")
            stream = getattr(result, "codestream")
            emitted_count = getattr(result, "emitted_byte_count")
            reason = getattr(result, "reason", None)
        except Exception as exc:
            raise FatalExecutionError(f"codec backend raised unexpectedly: {exc}") from exc
        if type(feasible) is not bool:
            raise FatalExecutionError("codec backend feasible field is not bool")
        if feasible:
            if not isinstance(stream, bytes) or not stream:
                raise FatalExecutionError("codec backend feasible result has no bytes")
            if type(emitted_count) is not int or emitted_count != len(stream) or emitted_count > key.payload_budget_bytes:
                raise FatalExecutionError("codec backend emitted bytes violate the frozen budget")
            status = "feasible"
            reason = None
        else:
            if stream is not None or emitted_count is not None:
                raise FatalExecutionError("codec backend infeasibility returned emitted bytes")
            if reason is not None and (not isinstance(reason, str) or not reason):
                raise FatalExecutionError("codec backend infeasibility reason is malformed")
            status = OUTCOME_CODEC_INFEASIBILITY
        body: dict[str, Any] = {
            "schema_version": V2_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_codec_cache_object",
            "key": key.payload(),
            "status": status,
            "reason": reason,
            "codestream_b64": None if stream is None else base64.b64encode(stream).decode("ascii"),
            "codestream_sha256": None if stream is None else sha256_bytes(stream),
            "emitted_byte_count": emitted_count,
        }
        body["cache_object_id"] = _id(V2_CODEC_PREFIX, body)
        _atomic_publish(self._path(key), rendered_json(body))
        return CodecArtifactV2(key, status, reason, stream, emitted_count, body["cache_object_id"], False)


class PhysicalReconstructionCacheV2:
    """Reconstruction cache; only an explicit typed failure is scientific."""

    def __init__(self, root: Path, decoder: Callable[[bytes], Any]) -> None:
        self.root = Path(root).resolve()
        self.decoder = decoder

    def _identity(self, key: PhysicalCacheKey, stream: bytes, output_shape: tuple[int, int, int]) -> dict[str, Any]:
        return {
            "schema_version": V2_SCHEMA_VERSION,
            "physical_cache_key": key.payload(),
            "codestream_sha256": sha256_bytes(stream),
            "output_shape": list(output_shape),
            "upsample_interpolation": get("preprocessing.codec_upsample_interpolation"),
        }

    def _path(self, object_id: str) -> Path:
        return self.root / "reconstruction" / f"{object_id}.json"

    def _load(self, path: Path, identity: Mapping[str, Any]) -> ReconstructionArtifactV2:
        try:
            value, _ = _rendered_object(path, "v2 reconstruction cache object")
        except G8EV2Error as exc:
            raise FatalExecutionError(f"corrupt reconstruction cache object: {exc}") from exc
        required = {"schema_version", "artifact_role", "identity", "status", "reason", "pixels_b64", "pixels_sha256", "object_id"}
        if set(value) != required or value["identity"] != identity or value["schema_version"] != V2_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_reconstruction_cache_object":
            raise FatalExecutionError("corrupt reconstruction cache identity")
        object_id = _id(V2_RECONSTRUCTION_PREFIX, identity)
        if value["object_id"] != object_id:
            raise FatalExecutionError("reconstruction cache object ID differs")
        if value["status"] == OUTCOME_DECODE_FAILURE:
            if value["pixels_b64"] is not None or value["pixels_sha256"] is not None:
                raise FatalExecutionError("decode-failure cache contains pixels")
            return ReconstructionArtifactV2(object_id, OUTCOME_DECODE_FAILURE, value["reason"], None, True)
        if value["status"] != "delivered" or not isinstance(value["pixels_b64"], str):
            raise FatalExecutionError("reconstruction cache status differs")
        try:
            raw = base64.b64decode(value["pixels_b64"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise FatalExecutionError(f"reconstruction cache base64 is corrupt: {exc}") from None
        shape = tuple(identity["output_shape"])
        expected = int(np.prod(shape))
        pixels = np.frombuffer(raw, dtype=np.uint8)
        if pixels.size != expected or value["pixels_sha256"] != sha256_bytes(raw):
            raise FatalExecutionError("reconstruction cache pixels differ")
        array = pixels.reshape(shape).copy()
        return ReconstructionArtifactV2(object_id, "delivered", None, array, True)

    def get_or_create(self, key: PhysicalCacheKey, stream: bytes, output_shape: tuple[int, int, int]) -> ReconstructionArtifactV2:
        identity = self._identity(key, stream, output_shape)
        object_id = _id(V2_RECONSTRUCTION_PREFIX, identity)
        path = self._path(object_id)
        if path.exists():
            return self._load(path, identity)
        try:
            decoded = self.decoder(stream)
        except Exception as exc:
            raise FatalExecutionError(f"decoder raised unexpectedly: {exc}") from exc
        if isinstance(decoded, ScientificDecodeFailure):
            status = OUTCOME_DECODE_FAILURE
            reason = decoded.reason
            pixels = None
        else:
            if not isinstance(decoded, np.ndarray) or decoded.dtype != np.uint8 or decoded.ndim != 3 or decoded.shape[2] != 3:
                raise FatalExecutionError("decoder returned malformed pixels")
            if tuple(decoded.shape) != output_shape:
                if decoded.shape[0] > output_shape[0] or decoded.shape[1] > output_shape[1]:
                    raise FatalExecutionError("decoder returned a shape that would require downsampling")
                try:
                    from data.preprocessing import codec_upsample

                    decoded = codec_upsample(decoded, output_hw=output_shape[:2])
                except Exception as exc:
                    raise FatalExecutionError(f"decoder output shape violates reconstruction contract: {exc}") from exc
            if tuple(decoded.shape) != output_shape:
                raise FatalExecutionError("decoder output shape remains invalid after explicit upsample")
            status = "delivered"
            reason = None
            pixels = np.ascontiguousarray(decoded)
        body: dict[str, Any] = {
            "schema_version": V2_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_reconstruction_cache_object",
            "identity": identity,
            "status": status,
            "reason": reason,
            "pixels_b64": None if pixels is None else base64.b64encode(pixels.tobytes()).decode("ascii"),
            "pixels_sha256": None if pixels is None else sha256_bytes(pixels.tobytes()),
        }
        body["object_id"] = object_id
        _atomic_publish(path, rendered_json(body))
        return ReconstructionArtifactV2(body["object_id"], status, reason, pixels, False)


class ClassifierObservationCacheV2:
    """Cache deterministic predictions, never an accuracy float."""

    def __init__(self, root: Path, classifier: Any, *, checkpoint_sha256: str, config_identity: str, runtime_identity: str) -> None:
        _digest(checkpoint_sha256, "classifier checkpoint SHA-256")
        if not config_identity or not runtime_identity:
            raise G8EV2Error("classifier observation identity is incomplete")
        self.root = Path(root).resolve()
        self.classifier = classifier
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_identity = config_identity
        self.runtime_identity = runtime_identity

    def _identity(self, reconstruction: ReconstructionArtifactV2, pixels: np.ndarray) -> dict[str, Any]:
        if reconstruction.status != "delivered" or reconstruction.pixels is None:
            raise FatalExecutionError("classifier observation requires a delivered reconstruction")
        return {
            "schema_version": V2_SCHEMA_VERSION,
            "reconstruction_object_id": reconstruction.object_id,
            "reconstruction_sha256": sha256_bytes(np.ascontiguousarray(pixels).tobytes()),
            "classifier_checkpoint_sha256": self.checkpoint_sha256,
            "classifier_config_identity": self.config_identity,
            "inference_runtime_identity": self.runtime_identity,
        }

    def get_or_create(self, reconstruction: ReconstructionArtifactV2) -> ClassifierObservationV2:
        if reconstruction.pixels is None:
            raise FatalExecutionError("classifier observation has no reconstruction pixels")
        identity = self._identity(reconstruction, reconstruction.pixels)
        object_id = _id(V2_OBSERVATION_PREFIX, identity)
        path = self.root / "observation" / f"{object_id}.json"
        if path.exists():
            try:
                value, _ = _rendered_object(path, "v2 classifier observation cache object")
            except G8EV2Error as exc:
                raise FatalExecutionError(f"corrupt classifier observation cache object: {exc}") from exc
            required = {"schema_version", "artifact_role", "identity", "predicted_label", "object_id", "object_sha256"}
            if set(value) != required or value["identity"] != identity or value["schema_version"] != V2_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_classifier_observation_cache_object":
                raise FatalExecutionError("classifier observation cache identity differs")
            if type(value["predicted_label"]) is not int or not 0 <= value["predicted_label"] < 10:  # literal-ok: Imagenette class vocabulary
                raise FatalExecutionError("classifier observation cache prediction is malformed")
            expected = _id(V2_OBSERVATION_PREFIX, identity)
            if value["object_id"] != expected:
                raise FatalExecutionError("classifier observation object ID differs")
            object_body = {key: child for key, child in value.items() if key != "object_sha256"}
            if value["object_sha256"] != sha256_bytes(canonical_json(object_body)):
                raise FatalExecutionError("classifier observation object digest differs")
            return ClassifierObservationV2(value["object_id"], value["predicted_label"], True)
        try:
            predicted = self.classifier.predict(np.ascontiguousarray(reconstruction.pixels))
        except Exception as exc:
            raise FatalExecutionError(f"classifier raised unexpectedly: {exc}") from exc
        if type(predicted) is not int or not 0 <= predicted < 10:  # literal-ok: Imagenette class vocabulary
            raise FatalExecutionError("classifier returned a malformed prediction")
        body: dict[str, Any] = {
            "schema_version": V2_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_classifier_observation_cache_object",
            "identity": identity,
            "predicted_label": predicted,
            "object_sha256": None,
        }
        body["object_id"] = object_id
        body["object_sha256"] = sha256_bytes(canonical_json({key: value for key, value in body.items() if key != "object_sha256"}))
        _atomic_publish(path, rendered_json(body))
        return ClassifierObservationV2(body["object_id"], predicted, False)


@dataclass(frozen=True)
class SyntheticSample:
    """A test seam.  Every such object is non-scientific by construction."""

    stable_sample_id: str
    label: int
    source_bytes: bytes
    canonical_pixels: np.ndarray
    dataset: str = INITIAL_DATASET
    split: str = VALIDATION_SPLIT

    def __post_init__(self) -> None:
        if not isinstance(self.stable_sample_id, str) or not self.stable_sample_id:
            raise G8EV2Error("sample stable ID is empty")
        if type(self.label) is not int or not 0 <= self.label < 10:  # literal-ok: Imagenette class vocabulary
            raise G8EV2Error("sample label is outside the Imagenette class vocabulary")
        if not isinstance(self.source_bytes, bytes) or not self.source_bytes:
            raise G8EV2Error("sample source bytes are empty")
        if self.dataset != INITIAL_DATASET or self.split != VALIDATION_SPLIT:
            raise G8EV2Error("sample is outside the sealed Imagenette validation scope")
        if (
            not isinstance(self.canonical_pixels, np.ndarray)
            or self.canonical_pixels.dtype != np.uint8
            or self.canonical_pixels.ndim != 3
            or self.canonical_pixels.shape[2] != 3
        ):
            raise G8EV2Error("sample canonical pixels must be uint8 RGB HWC")


def score_outage(label: int, selected_class: int) -> tuple[int, int]:
    """Return the one-row BR-13 binary count without introducing a float."""

    if type(label) is not int or type(selected_class) is not int:
        raise G8EV2Error("outage scoring labels must be integers")
    if not 0 <= label < 10 or not 0 <= selected_class < 10:  # literal-ok: Imagenette class vocabulary
        raise G8EV2Error("outage scoring label is outside the class vocabulary")
    return (int(label == selected_class), 1)


def compose_expected_accuracy(
    *,
    p_success: Any,
    acc_clean_correct: int,
    acc_clean_total: int,
    acc_outage_numerator: int,
    acc_outage_denominator: int,
) -> Any:
    """Compose BR-4 from measured count objects; callers cannot supply accuracy floats."""

    if isinstance(p_success, bool) or not isinstance(p_success, (int, float)) or not 0 <= p_success <= 1:
        raise G8EV2Error("BR-4 success probability must be in [0, 1]")
    for value, label in (
        (acc_clean_correct, "acc_clean_correct"),
        (acc_clean_total, "acc_clean_total"),
        (acc_outage_numerator, "acc_outage_numerator"),
        (acc_outage_denominator, "acc_outage_denominator"),
    ):
        _nonnegative_int(value, label)
    _positive_int(acc_clean_total, "acc_clean_total")
    _positive_int(acc_outage_denominator, "acc_outage_denominator")
    if acc_clean_correct > acc_clean_total or acc_outage_numerator > acc_outage_denominator:
        raise G8EV2Error("BR-4 count numerator exceeds denominator")
    # This is intentionally returned as an arithmetic result only.  E4 never
    # stores it as authoritative evidence; the two measured count objects do.
    return p_success * (acc_clean_correct / acc_clean_total) + (1 - p_success) * (acc_outage_numerator / acc_outage_denominator)


def _contains_accuracy_float(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"accuracy", "acc_clean", "acc_outage", "expected_accuracy", "top1_acc"}:
                raise G8EV2Error(f"authoritative accuracy float is forbidden at {path}.{key}")
            _contains_accuracy_float(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _contains_accuracy_float(child, f"{path}[{index}]")


class MeasurementRecordV2:
    """One immutable per-image row; every image contributes exactly one count."""

    FIELDS = (
        "schema_version", "artifact_role", "record_id", "campaign_id", "contract_id",
        "measurement_authority_id", "authority_ordinal", "measurement_identity_id",
        "logical_candidate_ids", "work_unit_id", "stable_sample_id", "dataset", "split", "label",
        "source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape", "structural_identity",
        "packet_budget", "physical_cache_key", "codec_cache_object_id", "outcome", "failure_stage",
        "emitted_codestream", "reconstruction", "classifier_observation", "outage_prediction",
        "correct_count", "total_count", "br11", "g8_c_linkage_digest", "profile_id", "source_commit",
        "validation_only", "outage_applied", "scientific_evidence", "merge_eligible", "test_access",
        "training", "inference", "record_labels",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        value = _strict(payload, self.FIELDS, "v2 measurement record")
        _contains_accuracy_float(value)
        self.value = value
        self._validate()

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        contract_id: str,
        authority: Mapping[str, Any],
        work_unit: Mapping[str, Any],
        structural: Mapping[str, Any],
        sample: SyntheticSample,
        physical_key: PhysicalCacheKey | None,
        codec: CodecArtifactV2 | None,
        reconstruction: ReconstructionArtifactV2 | None,
        observation: ClassifierObservationV2 | None,
        outage_policy: Mapping[str, Any],
        profile_id: str,
        source_commit: str,
        g8_c_linkage_digest: str,
        record_labels: Sequence[str] = (),
    ) -> "MeasurementRecordV2":
        structural_valid = structural.get("structurally_legal", True) is True
        if not structural_valid:
            outcome = OUTCOME_STRUCTURAL_INFEASIBILITY
            failure_stage = "structural_packet_plan"
        elif codec is None:
            raise G8EV2Error("valid structural candidate has no codec outcome")
        elif codec.status == OUTCOME_CODEC_INFEASIBILITY:
            outcome = OUTCOME_CODEC_INFEASIBILITY
            failure_stage = "codec_search"
        elif codec.status == "feasible" and reconstruction is not None and reconstruction.status == OUTCOME_DECODE_FAILURE:
            outcome = OUTCOME_DECODE_FAILURE
            failure_stage = "clean_reconstruction"
        elif codec.status == "feasible" and reconstruction is not None and reconstruction.status == "delivered" and observation is not None:
            outcome = OUTCOME_DELIVERED
            failure_stage = None
        else:
            raise G8EV2Error("measurement outcome inputs do not form a frozen result")
        outage = outcome in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY, OUTCOME_DECODE_FAILURE}
        outage_prediction = None
        if outage:
            selected = outage_policy.get("selected_class")
            numerator, denominator = score_outage(sample.label, selected)
            outage_prediction = {
                "selected_class": selected,
                "correct": numerator,
                "denominator": denominator,
                "policy_path": outage_policy.get("path"),
                "policy_sha256": outage_policy.get("sha256"),
                "selection_is_count_derived": outage_policy.get("selection_is_count_derived") is True,
            }
            correct, total = numerator, denominator
        else:
            if observation is None or reconstruction is None or reconstruction.pixels is None:
                raise G8EV2Error("delivered result lacks classifier observation")
            correct, total = int(observation.predicted_label == sample.label), 1
        emitted = None
        br11 = None
        if codec is not None and codec.status == "feasible":
            if codec.codestream is None or codec.emitted_byte_count is None:
                raise G8EV2Error("feasible record lacks emitted codestream")
            emitted = {"sha256": sha256_bytes(codec.codestream), "bytes": len(codec.codestream)}
            try:
                from baseline.g8_d import EmittedFileIdentity, account_br11

                emitted_identity = EmittedFileIdentity(
                    codec_search_key_id=physical_key.key_id if physical_key is not None else "",
                    codestream_sha256=emitted["sha256"],
                    emitted_bytes=emitted["bytes"],
                    payload_budget_bytes=physical_key.payload_budget_bytes if physical_key is not None else 0,
                    filler_bytes=(physical_key.payload_budget_bytes - emitted["bytes"]) if physical_key is not None else 0,
                )
                br11 = account_br11(
                    codec.codestream,
                    emitted_file_identity=emitted_identity,
                    bytes_sent=physical_key.payload_budget_bytes if physical_key is not None else 0,
                    verdict=OUTCOME_DELIVERED if outcome == OUTCOME_DELIVERED else OUTCOME_DECODE_FAILURE,
                ).as_dict()
            except Exception as exc:
                raise FatalExecutionError(f"BR-11 accounting failed: {exc}") from exc
        body: dict[str, Any] = {
            "schema_version": V2_RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_scientific_measurement_record",
            "record_id": None,
            "campaign_id": campaign_id,
            "contract_id": contract_id,
            "measurement_authority_id": authority["authority_id"],
            "authority_ordinal": int(work_unit["ordinal"]),
            "measurement_identity_id": work_unit["measurement_identity_id"],
            "logical_candidate_ids": list(work_unit["logical_candidate_ids"]),
            "work_unit_id": work_unit["work_unit_id"],
            "stable_sample_id": sample.stable_sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "label": sample.label,
            "source_bytes_sha256": sha256_bytes(sample.source_bytes),
            "canonical_pixels_sha256": sha256_bytes(np.ascontiguousarray(sample.canonical_pixels).tobytes()),
            "canonical_shape": list(sample.canonical_pixels.shape),
            "structural_identity": _copy(structural),
            "packet_budget": _copy(structural.get("packet_accounting")) if structural_valid else None,
            "physical_cache_key": None if physical_key is None else physical_key.payload(),
            "codec_cache_object_id": None if codec is None else codec.cache_object_id,
            "outcome": outcome,
            "failure_stage": failure_stage,
            "emitted_codestream": emitted,
            "reconstruction": None if reconstruction is None else {"object_id": reconstruction.object_id, "status": reconstruction.status, "cache_hit": reconstruction.cache_hit},
            "classifier_observation": None if observation is None else {"object_id": observation.object_id, "predicted_label": observation.predicted_label, "cache_hit": observation.cache_hit},
            "outage_prediction": outage_prediction,
            "correct_count": correct,
            "total_count": total,
            "br11": br11,
            "g8_c_linkage_digest": g8_c_linkage_digest,
            "profile_id": profile_id,
            "source_commit": source_commit,
            "validation_only": True,
            "outage_applied": outage,
            "scientific_evidence": True,
            "merge_eligible": True,
            "test_access": 0,
            "training": 0,
            "inference": 1 if outcome == OUTCOME_DELIVERED else 0,
            "record_labels": list(record_labels),
        }
        body["record_id"] = _id(V2_RECORD_PREFIX, {key: value for key, value in body.items() if key != "record_id"})
        return cls(body)

    def _validate(self) -> None:
        value = self.value
        if value["schema_version"] != V2_RECORD_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_scientific_measurement_record":
            raise G8EV2Error("v2 record schema or role differs")
        body = {key: child for key, child in value.items() if key != "record_id"}
        if value["record_id"] != _id(V2_RECORD_PREFIX, body):
            raise G8EV2Error("v2 measurement record ID differs")
        if type(value["authority_ordinal"]) is not int or value["authority_ordinal"] < 0:
            raise G8EV2Error("v2 record ordinal is invalid")
        if value["dataset"] != INITIAL_DATASET or value["split"] != VALIDATION_SPLIT or type(value["label"]) is not int or not 0 <= value["label"] < 10:  # literal-ok: Imagenette class vocabulary
            raise G8EV2Error("v2 record is outside the Imagenette validation vocabulary")
        if value["validation_only"] is not True or value["test_access"] != 0 or value["training"] != 0 or value["scientific_evidence"] is not True:
            raise G8EV2Error("v2 record crosses a safety boundary")
        if value["outcome"] not in OUTCOMES:
            raise G8EV2Error("v2 record outcome is unknown")
        if type(value["correct_count"]) is not int or type(value["total_count"]) is not int or value["total_count"] != 1 or value["correct_count"] not in {0, 1}:
            raise G8EV2Error("v2 record does not carry one binary count")
        outage = value["outcome"] in {OUTCOME_STRUCTURAL_INFEASIBILITY, OUTCOME_CODEC_INFEASIBILITY, OUTCOME_DECODE_FAILURE}
        if value["outage_applied"] is not outage:
            raise G8EV2Error("v2 outage flag differs from outcome")
        if outage:
            policy = value["outage_prediction"]
            if not isinstance(policy, Mapping) or type(policy.get("selected_class")) is not int or policy.get("denominator") != 1 or policy.get("correct") != value["correct_count"] or policy.get("policy_sha256") is None:
                raise G8EV2Error("v2 outage row lacks the bound BR-13 prediction")
        elif value["outage_prediction"] is not None or value["classifier_observation"] is None or value["reconstruction"] is None:
            raise G8EV2Error("v2 delivered row lacks a clean observation")
        if value["outcome"] == OUTCOME_STRUCTURAL_INFEASIBILITY:
            if value["physical_cache_key"] is not None or value["codec_cache_object_id"] is not None or value["emitted_codestream"] is not None or value["br11"] is not None:
                raise G8EV2Error("structural infeasibility row contains physical evidence")
        elif value["outcome"] == OUTCOME_CODEC_INFEASIBILITY:
            if value["codec_cache_object_id"] is None or value["emitted_codestream"] is not None or value["br11"] is not None:
                raise G8EV2Error("codec infeasibility row has invalid physical evidence")
        elif value["outcome"] == OUTCOME_DECODE_FAILURE:
            if value["codec_cache_object_id"] is None or value["emitted_codestream"] is None or value["br11"] is None:
                raise G8EV2Error("decode-failure row lacks emitted accounting")
        elif value["outcome"] == OUTCOME_DELIVERED:
            if value["codec_cache_object_id"] is None or value["emitted_codestream"] is None or value["br11"] is None:
                raise G8EV2Error("delivered row lacks emitted accounting")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeasurementRecordV2":
        return cls(value)

    def as_dict(self) -> dict[str, Any]:
        return _copy(self.value)


def _atomic_publish(path: Path, payload: bytes) -> None:
    """Publish immutable bytes with same-directory fsync and no replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FatalExecutionError(f"immutable publication collision at {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FatalExecutionError(f"immutable publication collision at {path}")
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    """Crash-atomic replacement for the compact mutable campaign state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FatalExecutionError(f"mutable state path is a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(rendered_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@contextmanager
def _campaign_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class MeasurementExecutorV2:
    """Run exactly one image-level transaction after all outer gates pass."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        authority: Mapping[str, Any],
        runtime_root: Path,
        backend: Any,
        decoder: Callable[[bytes], Any],
        classifier: Any,
        non_scientific_fixture: bool = False,
    ) -> None:
        self.contract = contract
        self.authority = authority
        self.runtime_root = Path(runtime_root).resolve()
        self.codec = PhysicalCodecCacheV2(self.runtime_root, backend)
        self.reconstruction = PhysicalReconstructionCacheV2(self.runtime_root, decoder)
        classifier_checkpoint = str(contract.get("classifier", {}).get("checkpoint_sha256", "0" * 64))  # literal-ok: representative synthetic identity
        if classifier_checkpoint == "0" * 64:  # literal-ok: representative synthetic identity
            # Synthetic callers may use a deterministic all-zero identity; the
            # production contract binds the real G-1 checkpoint in its source.
            classifier_checkpoint = sha256_bytes(b"synthetic-or-contract-bound-classifier")
        classifier_config = str(contract.get("classifier", {}).get("config_identity", "g1-frozen-config"))
        classifier_runtime = str(contract.get("classifier", {}).get("runtime_identity", "frozen-reference-runtime"))
        self.observations = ClassifierObservationCacheV2(
            self.runtime_root,
            classifier,
            checkpoint_sha256=classifier_checkpoint,
            config_identity=classifier_config,
            runtime_identity=classifier_runtime,
        )
        self.non_scientific_fixture = non_scientific_fixture
        self._structural = {
            row["structural_identity_id"]: row
            for row in authority.get("structural_identities", ())
            if row.get("dataset") == INITIAL_DATASET
        }

    def _outage_policy(self) -> Mapping[str, Any]:
        policy = self.contract.get("outage_policy")
        if not isinstance(policy, Mapping) or type(policy.get("selected_class")) is not int or type(policy.get("numerator")) is not int or type(policy.get("denominator")) is not int or not policy.get("selection_is_count_derived"):
            raise FatalExecutionError("current contract has no count-derived outage policy")
        return policy

    def _g8_c_digest(self) -> str:
        bindings = self.contract.get("direct_upstream_bindings", self.contract.get("upstream", {}))
        return sha256_bytes(canonical_json(bindings))

    def _encoded_pixels(self, sample: SyntheticSample, structural: Mapping[str, Any]) -> np.ndarray:
        axis = structural.get("encode_axis_px")
        if type(axis) is not int or axis <= 0:
            raise FatalExecutionError("structural encode axis is malformed")
        if self.non_scientific_fixture:
            # Synthetic fixtures intentionally avoid PIL/codec work but still
            # obey the byte/dtype boundary exercised by the real executor.
            return np.ascontiguousarray(sample.canonical_pixels)
        try:
            from data.preprocessing import codec_downsample

            return np.ascontiguousarray(codec_downsample(sample.canonical_pixels, axis))
        except Exception as exc:
            raise FatalExecutionError(f"canonical-to-codec preprocessing failed: {exc}") from exc

    def execute(self, work_unit: Mapping[str, Any], sample: SyntheticSample) -> MeasurementRecordV2:
        if sample.dataset != INITIAL_DATASET or sample.split != VALIDATION_SPLIT:
            raise FatalExecutionError("measurement sample is outside the validation-only scope")
        if str(work_unit.get("stable_sample_id")) != sample.stable_sample_id:
            raise FatalExecutionError("sample provider returned the wrong stable ID")
        structural_id = work_unit.get("measurement_identity_id")
        structural = self._structural.get(structural_id)
        if structural is None:
            raise FatalExecutionError("work unit refers to an unknown structural identity")
        structural_valid = structural.get("structurally_legal", True) is True
        key: PhysicalCacheKey | None = None
        codec: CodecArtifactV2 | None = None
        reconstruction: ReconstructionArtifactV2 | None = None
        observation: ClassifierObservationV2 | None = None
        if structural_valid:
            try:
                key = make_physical_cache_key(
                    source_bytes=sample.source_bytes,
                    canonical_pixels=sample.canonical_pixels,
                    payload_budget_bytes=int(structural["payload_budget_bytes"]),
                    encode_axis_px=int(structural["encode_axis_px"]),
                    codec_configuration_hash=str(self.contract["codec"]["configuration_hash"]),
                    codec_runtime_identity=str(self.contract["codec"]["runtime_identity"]),
                )
                codec = self.codec.get_or_create(key, self._encoded_pixels(sample, structural))
            except FatalExecutionError:
                raise
            except Exception as exc:
                raise FatalExecutionError(f"physical codec transaction failed: {exc}") from exc
            if codec.status == "feasible":
                if codec.codestream is None:
                    raise FatalExecutionError("feasible codec artifact has no codestream")
                reconstruction = self.reconstruction.get_or_create(key, codec.codestream, tuple(int(v) for v in sample.canonical_pixels.shape))
                if reconstruction.status == "delivered":
                    observation = self.observations.get_or_create(reconstruction)
        labels = (
            "NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"
        ) if self.non_scientific_fixture else ()
        return MeasurementRecordV2.build(
            campaign_id=str(self.contract["campaign_id"]),
            contract_id=str(self.contract["contract_id"]),
            authority=self.authority,
            work_unit=work_unit,
            structural=structural,
            sample=sample,
            physical_key=key,
            codec=codec,
            reconstruction=reconstruction,
            observation=observation,
            outage_policy=self._outage_policy(),
            profile_id=str(self.contract.get("execution_profile", {}).get("profile_id", PRODUCTION_PROFILE_ID)),
            source_commit=str(self.contract.get("source_manifest", self.contract.get("execution_source_manifest", {})).get("source_commit", "synthetic")),
            g8_c_linkage_digest=self._g8_c_digest(),
            record_labels=labels,
        )


def _work_unit_id(measurement_identity_id: str, stable_sample_id: str) -> str:
    return _id(V2_WORK_UNIT_PREFIX, {
        "schema_version": V2_RECORD_SCHEMA_VERSION,
        "measurement_identity_id": measurement_identity_id,
        "stable_sample_id": stable_sample_id,
        "split": VALIDATION_SPLIT,
    })


def load_measurement_authority() -> dict[str, Any]:
    value, _ = _rendered_object(_old_authority_path(), "frozen v2 measurement authority")
    if value.get("authority_id") != _authority_binding()["authority_id"]:
        raise G8EV2Error("frozen measurement authority ID differs from v2 binding")
    return value


def expected_work_units(authority: Mapping[str, Any], sample_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    ids = tuple(str(value) for value in sample_ids)
    if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise G8EV2Error("work-unit sample IDs must be sorted and unique")
    initial = sorted(
        (row for row in authority.get("structural_identities", ()) if row.get("dataset") == INITIAL_DATASET),
        key=lambda row: str(row["structural_identity_id"]),
    )
    if not initial:
        raise G8EV2Error("work-unit authority has no initial structural identities")
    result: list[dict[str, Any]] = []
    for structural in initial:
        measurement_id = str(structural["structural_identity_id"])
        logical_ids = sorted(
            str(candidate_id)
            for candidate_id, identity_id in authority.get("logical_candidate_to_structural_id", {}).items()
            if identity_id == measurement_id
        )
        for sample_id in ids:
            result.append({
                "work_unit_id": _work_unit_id(measurement_id, sample_id),
                "ordinal": len(result),
                "measurement_identity_id": measurement_id,
                "logical_candidate_ids": logical_ids,
                "stable_sample_id": sample_id,
                "dataset": INITIAL_DATASET,
                "split": VALIDATION_SPLIT,
            })
    return tuple(result)


class AtomicE2CampaignV2:
    """Compact crash-safe exact-prefix transaction for the production authority."""

    STATE_FIELDS = (
        "schema_version", "artifact_role", "campaign_id", "contract_id", "measurement_authority_id",
        "total_required", "authority_order_sha256", "completed_prefix_count", "last_completed_work_unit_id",
        "rolling_prefix_digest", "counters", "status", "in_progress", "last_checkpoint", "last_diagnostic",
        "state_sha256",
    )

    def __init__(
        self,
        *,
        runtime_root: Path,
        contract: Mapping[str, Any],
        authority: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        executor: Callable[[Mapping[str, Any], SyntheticSample], MeasurementRecordV2],
        sample_provider: Callable[[str], SyntheticSample],
        mode: str,
    ) -> None:
        if mode not in {"start", "resume"}:
            raise G8EV2Error("campaign mode must be start or resume")
        self.root = Path(runtime_root).resolve()
        self.contract = dict(contract)
        self.authority = dict(authority)
        self.work_units = tuple(dict(unit) for unit in work_units)
        self.executor = executor
        self.sample_provider = sample_provider
        self.state_path = self.root / "campaign_state.json"
        self.lock_path = self.root / ".campaign.lock"
        self.records_dir = self.root / "records"
        self.checkpoints_dir = self.root / "checkpoints"
        self.diagnostics_dir = self.root / "diagnostics"
        self.reconciliation_record_visits = 0
        self.state_publications = 0
        self.state_bytes_written = 0
        self.checkpoint_bytes_written = 0
        self.record_bytes_written = 0
        self._work_by_id = {str(unit["work_unit_id"]): unit for unit in self.work_units}
        if len(self._work_by_id) != len(self.work_units) or any(
            int(unit["ordinal"]) != ordinal for ordinal, unit in enumerate(self.work_units)
        ):
            raise G8EV2Error("campaign work units are not an exact ordered authority")
        if self.contract.get("campaign_id") in {ORIGINAL_CAMPAIGN_ID, FIRST_CORRECTED_CAMPAIGN_ID}:
            raise G8EV2Error("a superseded E1 campaign cannot be instantiated")
        if mode == "start":
            if self.root.exists() and any(self.root.iterdir()):
                raise G8EV2Error("runtime already exists; use --resume")
            self.root.mkdir(parents=True, exist_ok=True)
            self.records_dir.mkdir(parents=True, exist_ok=True)
            self._write_state(self._new_state())
        else:
            if not self.root.is_dir() or not self.state_path.is_file():
                raise G8EV2Error("runtime is absent; use --start")
            self._reconcile_once()

    @property
    def total_required(self) -> int:
        return len(self.work_units)

    def _authority_order_sha256(self) -> str:
        return sha256_bytes(canonical_json([unit["work_unit_id"] for unit in self.work_units]))

    def _chain_seed(self) -> str:
        return sha256_bytes(canonical_json({
            "chain_version": V2_STATE_SCHEMA_VERSION,
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "measurement_authority_id": self.authority["authority_id"],
            "total_required": self.total_required,
        }))

    def _chain_step(self, previous: str, unit: Mapping[str, Any], record_sha256: str) -> str:
        return sha256_bytes(canonical_json({
            "previous_digest": previous,
            "authority_ordinal": unit["ordinal"],
            "work_unit_id": unit["work_unit_id"],
            "record_sha256": record_sha256,
        }))

    @staticmethod
    def _counters_zero() -> dict[str, int]:
        return {
            "validation_decoding": 0,
            "inference": 0,
            "training": 0,
            "test_access": 0,
            "delivered": 0,
            "codec_infeasibility": 0,
            "decode_failure": 0,
            "structural_infeasibility": 0,
        }

    @classmethod
    def _counter_for(cls, record: MeasurementRecordV2) -> dict[str, int]:
        counters = cls._counters_zero()
        outcome = record.value["outcome"]
        counters[outcome] += 1
        counters["validation_decoding"] = int(outcome in {OUTCOME_DELIVERED, OUTCOME_DECODE_FAILURE})
        counters["inference"] = int(record.value["inference"])
        return counters

    @classmethod
    def _sum_counters(cls, records: Sequence[MeasurementRecordV2]) -> dict[str, int]:
        result = cls._counters_zero()
        for record in records:
            one = cls._counter_for(record)
            for key in result:
                result[key] += one[key]
        return result

    def _new_state(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": V2_STATE_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_campaign_state",
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "measurement_authority_id": self.authority["authority_id"],
            "total_required": self.total_required,
            "authority_order_sha256": self._authority_order_sha256(),
            "completed_prefix_count": 0,
            "last_completed_work_unit_id": None,
            "rolling_prefix_digest": self._chain_seed(),
            "counters": self._counters_zero(),
            "status": READY_STATUS,
            "in_progress": None,
            "last_checkpoint": None,
            "last_diagnostic": None,
        }
        return self._with_state_hash(body)

    @staticmethod
    def _with_state_hash(body: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(body)
        value["state_sha256"] = sha256_bytes(canonical_json(value))
        return value

    def _write_state(self, value: Mapping[str, Any]) -> None:
        body = {key: child for key, child in value.items() if key != "state_sha256"}
        rendered = self._with_state_hash(body)
        payload = rendered_json(rendered)
        _replace_json(self.state_path, rendered)
        self.state_publications += 1
        self.state_bytes_written += len(payload)

    def _load_state(self) -> dict[str, Any]:
        value, _ = _rendered_object(self.state_path, "v2 compact campaign state")
        if set(value) != set(self.STATE_FIELDS):
            raise G8EV2Error("v2 compact state schema differs")
        body = {key: child for key, child in value.items() if key != "state_sha256"}
        if value["state_sha256"] != sha256_bytes(canonical_json(body)):
            raise G8EV2Error("v2 compact state digest differs")
        if value["schema_version"] != V2_STATE_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_campaign_state":
            raise G8EV2Error("v2 compact state role differs")
        if value["campaign_id"] != self.contract["campaign_id"] or value["contract_id"] != self.contract["contract_id"] or value["measurement_authority_id"] != self.authority["authority_id"]:
            raise G8EV2Error("v2 compact state is foreign")
        if value["total_required"] != self.total_required or value["authority_order_sha256"] != self._authority_order_sha256():
            raise G8EV2Error("v2 compact state authority differs")
        count = value["completed_prefix_count"]
        if type(count) is not int or not 0 <= count <= self.total_required:
            raise G8EV2Error("v2 compact state prefix count is invalid")
        _digest(value["rolling_prefix_digest"], "v2 rolling prefix digest")
        if value["status"] not in {READY_STATUS, RUNNING_STATUS, HOLD_STATUS, COMPLETE_STATUS}:
            raise G8EV2Error("v2 compact state status is invalid")
        if value["in_progress"] is not None:
            claim = value["in_progress"]
            if set(claim) != {"ordinal", "work_unit_id", "transaction_id"} or claim["ordinal"] != count or self._work_by_id.get(claim["work_unit_id"], {}).get("ordinal") != count:
                raise G8EV2Error("v2 compact state claim is not the next exact unit")
        if count == 0 and value["last_completed_work_unit_id"] is not None:
            raise G8EV2Error("v2 compact state has a last unit before the prefix")
        if count > 0 and value["last_completed_work_unit_id"] != self.work_units[count - 1]["work_unit_id"]:
            raise G8EV2Error("v2 compact state last unit differs")
        return value

    def _record_path(self, work_unit_id: str) -> Path:
        return self.records_dir / f"{work_unit_id}.json"

    def _read_record(self, path: Path) -> tuple[MeasurementRecordV2, bytes]:
        value, raw = _rendered_object(path, "v2 measurement record")
        record = MeasurementRecordV2.from_mapping(value)
        if path.name != f"{record.value['work_unit_id']}.json":
            raise G8EV2Error("record pathname and work-unit identity differ")
        if record.value["campaign_id"] != self.contract["campaign_id"] or record.value["contract_id"] != self.contract["contract_id"] or record.value["measurement_authority_id"] != self.authority["authority_id"]:
            raise G8EV2Error("record belongs to a foreign v2 campaign")
        return record, raw

    def _reconcile_once(self) -> None:
        state = self._load_state()
        entries: dict[int, tuple[MeasurementRecordV2, bytes]] = {}
        if self.records_dir.exists():
            for path in sorted(self.records_dir.iterdir(), key=lambda item: item.name):
                if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                    raise G8EV2Error("unexpected object in v2 records directory")
                record, raw = self._read_record(path)
                self.reconciliation_record_visits += 1
                ordinal = record.value["authority_ordinal"]
                expected = self._work_by_id.get(record.value["work_unit_id"])
                if expected is None or expected["ordinal"] != ordinal or ordinal in entries:
                    raise G8EV2Error("v2 durable record is outside the exact authority")
                entries[ordinal] = (record, raw)
        prefix = 0
        while prefix in entries:
            prefix += 1
        if any(ordinal >= prefix for ordinal in entries):
            raise G8EV2Error("v2 durable records contain a gap or reordered suffix")
        if state["completed_prefix_count"] > prefix:
            raise G8EV2Error("v2 compact state claims a prefix record that is not durable")
        if prefix > state["completed_prefix_count"] + 1:
            raise G8EV2Error("v2 durable records advance by more than one uncommitted unit")
        records = [entries[index][0] for index in range(prefix)]
        digest = self._chain_seed()
        for index, record in enumerate(records):
            digest = self._chain_step(digest, self.work_units[index], sha256_bytes(entries[index][1]))
        if prefix > self.total_required:
            raise G8EV2Error("v2 durable record prefix exceeds authority")
        counters = self._sum_counters(records)
        if state["completed_prefix_count"] == prefix and state["rolling_prefix_digest"] != digest:
            raise G8EV2Error("v2 compact rolling chain differs from the durable prefix")
        claim = state["in_progress"]
        if claim is not None and claim["ordinal"] > prefix:
            raise G8EV2Error("v2 compact claim is beyond durable prefix")
        if prefix == self.total_required:
            status = COMPLETE_STATUS
            in_progress = None
        elif claim is not None and claim["ordinal"] == prefix:
            status = HOLD_STATUS if state["status"] == HOLD_STATUS else RUNNING_STATUS
            in_progress = claim
        else:
            status = READY_STATUS if state["status"] != HOLD_STATUS else HOLD_STATUS
            in_progress = None
        reconciled = dict(state)
        reconciled.update({
            "completed_prefix_count": prefix,
            "last_completed_work_unit_id": None if prefix == 0 else self.work_units[prefix - 1]["work_unit_id"],
            "rolling_prefix_digest": digest,
            "counters": counters,
            "status": status,
            "in_progress": in_progress,
        })
        if reconciled != state:
            self._write_state(reconciled)
        self._reconciled_state = self._load_state()

    def _diagnostic(self, unit: Mapping[str, Any], transaction_id: str, exc: BaseException) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": V2_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_operational_diagnostic",
            "diagnostic_id": None,
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "authority_ordinal": unit["ordinal"],
            "work_unit_id": unit["work_unit_id"],
            "transaction_id": transaction_id,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),  # literal-ok: bounded operational diagnostic size
            "status": HOLD_STATUS,
            "scientific_evidence": False,
            "merge_eligible": False,
            "record_labels": ["NON-SCIENTIFIC", "OPERATIONAL DIAGNOSTIC", "MERGE-INELIGIBLE FOR PRODUCTION"],
        }
        body["diagnostic_id"] = _id(V2_DIAGNOSTIC_PREFIX, {key: value for key, value in body.items() if key != "diagnostic_id"})
        return body

    def _hold(self, state: Mapping[str, Any], unit: Mapping[str, Any], transaction_id: str, exc: BaseException) -> None:
        diagnostic = self._diagnostic(unit, transaction_id, exc)
        diagnostic_path = self.diagnostics_dir / f"{diagnostic['diagnostic_id']}.json"
        _atomic_publish(diagnostic_path, rendered_json(diagnostic))
        held = dict(state)
        held["status"] = HOLD_STATUS
        held["in_progress"] = {"ordinal": unit["ordinal"], "work_unit_id": unit["work_unit_id"], "transaction_id": transaction_id}
        held["last_diagnostic"] = {"diagnostic_id": diagnostic["diagnostic_id"], "sha256": sha256_bytes(rendered_json(diagnostic))}
        self._write_state(held)

    def _checkpoint(self, state: Mapping[str, Any], record: MeasurementRecordV2, digest: str, counters: Mapping[str, int]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": V2_STATE_SCHEMA_VERSION,
            "artifact_role": "g8_e_v2_compact_checkpoint",
            "campaign_id": self.contract["campaign_id"],
            "contract_id": self.contract["contract_id"],
            "measurement_authority_id": self.authority["authority_id"],
            "completed_prefix_count": state["completed_prefix_count"] + 1,
            "rolling_prefix_digest": digest,
            "last_completed_work_unit_id": record.value["work_unit_id"],
            "counters": dict(counters),
            "scientific_evidence": False,
            "record_labels": ["NON-SCIENTIFIC", "RUNTIME CHECKPOINT", "MERGE-INELIGIBLE FOR PRODUCTION"],
        }
        body["checkpoint_id"] = _id(V2_CHAIN_PREFIX, body)
        return body

    def run_next(self, *, crash_after: str | None = None) -> bool:
        with _campaign_lock(self.lock_path):
            state = self._load_state()
            completed = state["completed_prefix_count"]
            if completed == self.total_required:
                if state["status"] != COMPLETE_STATUS:
                    state["status"] = COMPLETE_STATUS
                    self._write_state(state)
                return False
            unit = self.work_units[completed]
            claim = state["in_progress"]
            if claim is None:
                transaction_id = _id(V2_CHAIN_PREFIX, {"campaign_id": self.contract["campaign_id"], "ordinal": completed, "work_unit_id": unit["work_unit_id"], "attempt": state.get("last_diagnostic")})
                claimed = dict(state)
                claimed["status"] = RUNNING_STATUS
                claimed["in_progress"] = {"ordinal": completed, "work_unit_id": unit["work_unit_id"], "transaction_id": transaction_id}
                self._write_state(claimed)
                state = self._load_state()
            else:
                transaction_id = str(claim["transaction_id"])
                if claim["ordinal"] != completed or claim["work_unit_id"] != unit["work_unit_id"]:
                    raise G8EV2Error("v2 transaction claim is not the next exact work unit")
            if crash_after in {"claim", "work_claim"}:
                raise RuntimeError("synthetic crash after v2 work-unit claim")
            path = self._record_path(unit["work_unit_id"])
            try:
                if path.exists():
                    record, raw = self._read_record(path)
                else:
                    sample = self.sample_provider(unit["stable_sample_id"])
                    record = self.executor(unit, sample)
                    if record.value["authority_ordinal"] != unit["ordinal"] or record.value["work_unit_id"] != unit["work_unit_id"]:
                        raise FatalExecutionError("executor returned a record for the wrong authority unit")
                    raw = rendered_json(record.as_dict())
                    _atomic_publish(path, raw)
                    self.record_bytes_written += len(raw)
            except Exception as exc:
                try:
                    self._hold(state, unit, transaction_id, exc)
                except Exception as hold_exc:
                    raise CampaignHoldError(f"v2 runtime failed and HOLD publication failed: {hold_exc}") from exc
                raise CampaignHoldError(f"v2 work unit entered HOLD: {exc}") from exc
            if crash_after == "record":
                raise RuntimeError("synthetic crash after v2 record publication")
            record_sha = sha256_bytes(raw)
            digest = self._chain_step(state["rolling_prefix_digest"], unit, record_sha)
            final_counters = dict(state["counters"])
            for key, amount in self._counter_for(record).items():
                final_counters[key] += amount
            final = dict(state)
            final.update({
                "completed_prefix_count": completed + 1,
                "last_completed_work_unit_id": unit["work_unit_id"],
                "rolling_prefix_digest": digest,
                "counters": final_counters,
                "status": COMPLETE_STATUS if completed + 1 == self.total_required else READY_STATUS,
                "in_progress": None,
                "last_diagnostic": None,
            })
            if (completed + 1) % CHECKPOINT_INTERVAL == 0 or completed + 1 == self.total_required:
                checkpoint = self._checkpoint(state, record, digest, final_counters)
                checkpoint_path = self.checkpoints_dir / f"{checkpoint['checkpoint_id']}.json"
                checkpoint_raw = rendered_json(checkpoint)
                _atomic_publish(checkpoint_path, checkpoint_raw)
                self.checkpoint_bytes_written += len(checkpoint_raw)
                final["last_checkpoint"] = {"checkpoint_id": checkpoint["checkpoint_id"], "sha256": sha256_bytes(checkpoint_raw)}
                if crash_after == "checkpoint":
                    raise RuntimeError("synthetic crash after v2 compact checkpoint")
            if crash_after == "state":
                raise RuntimeError("synthetic crash before v2 state publication")
            self._write_state(final)
            return True

    def run_all(self) -> None:
        while self.run_next():
            pass

    def state(self) -> dict[str, Any]:
        with _campaign_lock(self.lock_path):
            return self._load_state()


def _record_list(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    record_values: Sequence[Mapping[str, Any] | MeasurementRecordV2] | None,
    runtime_root: Path | None,
    contract: Mapping[str, Any] | None,
    production: bool,
) -> tuple[tuple[dict[str, Any], ...], list[MeasurementRecordV2], list[bytes]]:
    expected = expected_work_units(authority, sample_ids)
    expected_ids = [unit["work_unit_id"] for unit in expected]
    loaded: list[MeasurementRecordV2] = []
    raws: list[bytes] = []
    if record_values is not None and runtime_root is not None:
        raise G8EV2Error("E3 accepts either explicit records or a runtime root, not both")
    if runtime_root is not None:
        records_dir = Path(runtime_root).resolve() / "records"
        if not records_dir.is_dir():
            raise G8EV2Error("E3 runtime has no records directory")
        paths = sorted(records_dir.iterdir(), key=lambda item: item.name)
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise G8EV2Error("E3 runtime contains a non-record object")
            record, raw = _rendered_object(path, "E3 runtime record")
            loaded.append(MeasurementRecordV2.from_mapping(record))
            raws.append(raw)
    elif record_values is not None:
        for item in record_values:
            if isinstance(item, MeasurementRecordV2):
                record = item
                raw = rendered_json(record.as_dict())
            else:
                record = MeasurementRecordV2.from_mapping(item)
                raw = rendered_json(record.as_dict())
            loaded.append(record)
            raws.append(raw)
    else:
        raise G8EV2Error("E3 has no durable record source")
    if len(loaded) != len(expected_ids):
        raise G8EV2Error(f"E3 exact-set count differs: {len(loaded)} != {len(expected_ids)}")
    seen: set[str] = set()
    by_id: dict[str, tuple[MeasurementRecordV2, bytes]] = {}
    for index, (record, raw) in enumerate(zip(loaded, raws)):
        value = record.value
        work_id = value["work_unit_id"]
        if work_id in seen:
            raise G8EV2Error("E3 duplicate work unit")
        seen.add(work_id)
        if work_id not in expected_ids:
            raise G8EV2Error("E3 extra or foreign work unit")
        if record_values is not None and value["authority_ordinal"] != index:
            raise G8EV2Error("E3 explicit record sequence is reordered")
        by_id[work_id] = (record, raw)
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise G8EV2Error(f"E3 exact-set mismatch: missing={missing[:3]}, extra={extra[:3]}")
    ordered: list[MeasurementRecordV2] = []
    ordered_raw: list[bytes] = []
    expected_profile = contract.get("execution_profile", {}).get("profile_id") if contract is not None else None
    expected_source = contract.get("source_manifest", contract.get("execution_source_manifest", {})).get("source_commit") if contract is not None else None
    expected_linkage = sha256_bytes(canonical_json(contract.get("direct_upstream_bindings", contract.get("upstream", {})))) if contract is not None else None
    for ordinal, unit in enumerate(expected):
        record, raw = by_id[unit["work_unit_id"]]
        value = record.value
        if value["authority_ordinal"] != ordinal or value["measurement_identity_id"] != unit["measurement_identity_id"] or value["logical_candidate_ids"] != unit["logical_candidate_ids"] or value["stable_sample_id"] != unit["stable_sample_id"]:
            raise G8EV2Error("E3 record is substituted for the expected authority unit")
        structural = next((row for row in authority["structural_identities"] if row["structural_identity_id"] == unit["measurement_identity_id"]), None)
        if structural is None or value["structural_identity"] != structural:
            raise G8EV2Error("E3 record structural identity differs from authority")
        if contract is not None:
            if value["contract_id"] != contract["contract_id"] or value["campaign_id"] != contract["campaign_id"] or value["measurement_authority_id"] != authority["authority_id"]:
                raise G8EV2Error("E3 record source/contract linkage differs")
            if expected_profile is not None and value["profile_id"] != expected_profile:
                raise G8EV2Error("E3 record execution profile differs")
            if expected_source is not None and value["source_commit"] != expected_source:
                raise G8EV2Error("E3 record source commit differs")
            if expected_linkage is not None and value["g8_c_linkage_digest"] != expected_linkage:
                raise G8EV2Error("E3 record G8_C linkage differs")
        if value["test_access"] != 0 or value["training"] != 0 or value["validation_only"] is not True:
            raise G8EV2Error("E3 record crosses a safety boundary")
        if production and (value["record_labels"] or value["scientific_evidence"] is not True or value["merge_eligible"] is not True):
            raise G8EV2Error("E3 synthetic or non-mergeable record entered production merge")
        ordered.append(record)
        ordered_raw.append(raw)
    return tuple(expected), ordered, ordered_raw


def merge_e3_records_v2(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    record_values: Sequence[Mapping[str, Any] | MeasurementRecordV2] | None = None,
    runtime_root: Path | None = None,
    contract: Mapping[str, Any] | None = None,
    production: bool = True,
) -> dict[str, Any]:
    """Independently authenticate the complete expected record set once."""

    expected, ordered, raws = _record_list(
        authority=authority,
        sample_ids=sample_ids,
        record_values=record_values,
        runtime_root=runtime_root,
        contract=contract,
        production=production,
    )
    if runtime_root is not None and production:
        root = Path(runtime_root).resolve()
        for record in ordered:
            value = record.value
            if value["codec_cache_object_id"] is not None:
                physical = value["physical_cache_key"]
                if not isinstance(physical, Mapping):
                    raise G8EV2Error("E3 feasible/outage record has no physical cache key")
                key = PhysicalCacheKey(
                    source_bytes_sha256=physical["source_bytes_sha256"],
                    canonical_pixels_sha256=physical["canonical_pixels_sha256"],
                    canonical_shape=tuple(physical["canonical_shape"]),
                    payload_budget_bytes=physical["payload_budget_bytes"],
                    encode_axis_px=physical["encode_axis_px"],
                    codec_configuration_hash=physical["codec_configuration_hash"],
                    codec_runtime_identity=physical["codec_runtime_identity"],
                )
                codec_path = root / "codec" / f"{key.key_id}.json"
                if not codec_path.is_file():
                    raise G8EV2Error("E3 record references a missing codec cache object")
                cached = PhysicalCodecCacheV2(root, None)._load(codec_path, key)
                if cached.cache_object_id != value["codec_cache_object_id"]:
                    raise G8EV2Error("E3 record codec cache object ID differs")
            reconstruction = value["reconstruction"]
            if reconstruction is not None and not (root / "reconstruction" / f"{reconstruction['object_id']}.json").is_file():
                raise G8EV2Error("E3 record references a missing reconstruction object")
            observation = value["classifier_observation"]
            if observation is not None and not (root / "observation" / f"{observation['object_id']}.json").is_file():
                raise G8EV2Error("E3 record references a missing classifier observation")
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_e3_exact_set_merge",
        "status": "MERGED",
        "production": production,
        "scientific_evidence": production,
        "merge_eligible": production,
        "measurement_authority_id": authority["authority_id"],
        "work_unit_count": len(ordered),
        "expected_work_unit_ids": [unit["work_unit_id"] for unit in expected],
        "record_ids": [record.value["record_id"] for record in ordered],
        "record_sha256s": [sha256_bytes(raw) for raw in raws],
        "ordered_prefix_digest": sha256_bytes(canonical_json([sha256_bytes(raw) for raw in raws])),
        "record_labels": [] if production else ["NON-SCIENTIFIC", "NON-SELECTION", "MERGE-INELIGIBLE FOR PRODUCTION"],
    }


def aggregate_e4_counts_v2(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    record_values: Sequence[Mapping[str, Any] | MeasurementRecordV2] | None = None,
    runtime_root: Path | None = None,
    contract: Mapping[str, Any] | None = None,
    production: bool = True,
) -> dict[str, Any]:
    """Derive E4 objects from row counts, including image-level outages."""

    merged = merge_e3_records_v2(
        authority=authority,
        sample_ids=sample_ids,
        record_values=record_values,
        runtime_root=runtime_root,
        contract=contract,
        production=production,
    )
    expected, records, raws = _record_list(
        authority=authority,
        sample_ids=sample_ids,
        record_values=record_values,
        runtime_root=runtime_root,
        contract=contract,
        production=production,
    )
    by_identity: dict[str, list[tuple[MeasurementRecordV2, bytes]]] = defaultdict(list)
    for record, raw in zip(records, raws):
        by_identity[record.value["measurement_identity_id"]].append((record, raw))
    policy = contract.get("outage_policy", {}) if contract is not None else {}
    outage_object = {
        "selected_class": policy.get("selected_class", 0),
        "numerator": policy.get("numerator", 0),
        "denominator": policy.get("denominator", 1),
        "policy_path": policy.get("path", "synthetic/outage_policy.json"),
        "policy_sha256": policy.get("sha256", sha256_bytes(b"synthetic-outage-policy")),
        "selection_is_count_derived": policy.get("selection_is_count_derived", True),
    }
    objects: list[dict[str, Any]] = []
    structural_rows = {
        row["structural_identity_id"]: row
        for row in authority.get("structural_identities", ())
        if row.get("dataset") == INITIAL_DATASET
    }
    for structural_id in sorted(structural_rows):
        structural = structural_rows[structural_id]
        rows = by_identity.get(structural_id, [])
        if len(rows) != len(sample_ids):
            raise G8EV2Error("E4 structural identity does not have the exact validation denominator")
        source_ids = [record.value["record_id"] for record, _ in rows]
        source_digests = [sha256_bytes(raw) for _, raw in rows]
        if structural.get("structurally_legal", True) is not True:
            objects.append({
                "measurement_identity_id": structural_id,
                "status": "ineligible",
                "reason": "structurally_impossible_packet_configuration",
                "delivered_count": 0,
                "codec_infeasibility_count": 0,
                "decode_failure_count": 0,
                "correct_count": None,
                "total_count": None,
                "infeasible_rate": None,
                "source_record_ids": source_ids,
                "source_record_sha256s": source_digests,
            })
            continue
        delivered = sum(record.value["outcome"] == OUTCOME_DELIVERED for record, _ in rows)
        codec_infeasible = sum(record.value["outcome"] == OUTCOME_CODEC_INFEASIBILITY for record, _ in rows)
        decode_failure = sum(record.value["outcome"] == OUTCOME_DECODE_FAILURE for record, _ in rows)
        structural_infeasible = sum(record.value["outcome"] == OUTCOME_STRUCTURAL_INFEASIBILITY for record, _ in rows)
        if structural_infeasible:
            raise G8EV2Error("structural infeasibility appeared in a candidate marked structurally legal")
        correct = sum(record.value["correct_count"] for record, _ in rows)
        total = sum(record.value["total_count"] for record, _ in rows)
        if total != len(sample_ids):
            raise G8EV2Error("E4 acc_clean denominator dropped an image-level row")
        if any(record.value["outage_prediction"] is None for record, _ in rows if record.value["outcome"] in {OUTCOME_CODEC_INFEASIBILITY, OUTCOME_DECODE_FAILURE}):
            raise G8EV2Error("E4 outage row is missing its BR-13 prediction")
        objects.append({
            "measurement_identity_id": structural_id,
            "status": "eligible",
            "delivered_count": delivered,
            "codec_infeasibility_count": codec_infeasible,
            "decode_failure_count": decode_failure,
            "correct_count": correct,
            "total_count": total,
            "infeasible_rate": codec_infeasible / total,
            "clean_accuracy_counts": {"correct_count": correct, "total_count": total},
            "outage_accuracy_binding": _copy(outage_object),
            "source_record_ids": source_ids,
            "source_record_sha256s": source_digests,
        })
    result = {
        "schema_version": V2_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_e4_count_derived_objects",
        "status": "COUNT_DERIVED",
        "production": production,
        "scientific_evidence": production,
        "merge_eligible": production,
        "measurement_authority_id": authority["authority_id"],
        "object_count": len(objects),
        "objects": objects,
        "outage_accuracy": outage_object,
        "br4_formula": "P(TB success) * acc_clean + (1 - P(TB success)) * acc_outage",
        "e3_record_count": merged["work_unit_count"],
        "e3_ordered_prefix_digest": merged["ordered_prefix_digest"],
        "record_labels": [] if production else ["NON-SCIENTIFIC", "NON-SELECTION", "MERGE-INELIGIBLE FOR PRODUCTION"],
    }
    return result


def authenticate_owner_authorization_v2(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the later owner artifact without creating one in E1."""

    value, _ = _rendered_object(Path(path), "v2 owner E2 authorization")
    required = {
        "schema_version", "artifact_role", "status", "authorized_by", "reason",
        "campaign_id", "contract_id", "source_manifest_id", "profile_id", "scope", "issued_sha256",
    }
    if set(value) != required or value["schema_version"] != V2_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_owner_e2_authorization" or value["status"] != "AUTHORIZED":
        raise G8EV2Error("v2 owner authorization is absent or not active")
    source = contract.get("source_manifest", {})
    profile = contract.get("execution_profile", {})
    if value["campaign_id"] != contract["campaign_id"] or value["contract_id"] != contract["contract_id"] or value["source_manifest_id"] != source.get("id") or value["profile_id"] != profile.get("profile_id"):
        raise G8EV2Error("v2 owner authorization belongs to another campaign/source/profile")
    body = {key: child for key, child in value.items() if key != "issued_sha256"}
    if value["issued_sha256"] != sha256_bytes(canonical_json(body)):
        raise G8EV2Error("v2 owner authorization digest differs")
    scope = value["scope"]
    frozen_scope = contract.get("authorization", {}).get("schema_scope_frozen", {})
    if not isinstance(scope, Mapping) or scope.get("validation_decode") is not True or scope.get("test_access") is not False or scope.get("training") is not False or scope.get("fallback") is not False or scope.get("ratio_adjudication") is not False or scope.get("pass_one") is not False or scope.get("pass_two") is not False:
        raise G8EV2Error("v2 owner authorization scope opens forbidden work")
    if frozen_scope and dict(scope) != dict(frozen_scope):
        raise G8EV2Error("v2 owner authorization scope differs from the frozen schema")
    return value


def reject_superseded_campaign(campaign_id: str) -> None:
    if campaign_id in {ORIGINAL_CAMPAIGN_ID, FIRST_CORRECTED_CAMPAIGN_ID}:
        raise G8EV2Error("superseded-before-data E1 campaign cannot execute E2")


def check_runtime_mode(mode: str, runtime_root: Path) -> None:
    """Enforce the real --start/--resume distinction without opening data."""

    root = Path(runtime_root)
    if mode == "start":
        if root.exists() and any(root.iterdir()):
            raise G8EV2Error("runtime already exists; use --resume")
        return
    if mode == "resume":
        if not root.is_dir() or not (root / "campaign_state.json").is_file():
            raise G8EV2Error("runtime does not exist; use --start")
        return
    raise G8EV2Error("runtime mode must be start or resume")
