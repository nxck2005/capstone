"""G8_E corrected-v3 lifecycle and exact-set evidence closure.

This additive pre-data epoch preserves corrected-v2 scientific semantics while
repairing lifecycle verification, authority-digest scaling, E3 exact-set
complexity, E3/E4 provenance chaining, validation-data identity binding and
cache-reference authentication.  It never creates owner authorization and it
contains no command that implicitly opens validation data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline import g8_e_corrected_v2 as v2
from config.params import REPO_ROOT, get


V3_ROOT = REPO_ROOT / "results/baseline/g8_e/e1_corrected_v3"
V3_CONTRACT_PATH = V3_ROOT / "measurement_contract.json"
V3_SOURCE_MANIFEST_PATH = V3_ROOT / "execution_source_manifest.json"
V3_DATA_IDENTITY_PATH = V3_ROOT / "scientific_data_identity_manifest.json"
V3_CORRECTION_PATH = V3_ROOT / "correction_provenance.json"
V3_STORAGE_PLAN_PATH = V3_ROOT / "compute_storage_plan.json"
V3_COMPLEXITY_PATH = V3_ROOT / "complexity_scale_evidence.json"
V3_SYNTHETIC_PROOF_PATH = V3_ROOT / "synthetic_lifecycle_proof.json"
V3_RUNTIME_ROOT = V3_ROOT / "runtime"
V3_AUTHORIZATION_PATH = V3_ROOT / "e2_execution_authorization.json"
V3_E2_COMPLETION_PATH = V3_RUNTIME_ROOT / "e2_completion.json"
V3_E3_PATH = V3_RUNTIME_ROOT / "e3_exact_set_closure.json"
V3_E4_PATH = V3_RUNTIME_ROOT / "e4_count_derived.json"

V3_SCHEMA_VERSION = 3
V3_STATE_SCHEMA_VERSION = v2.V2_STATE_SCHEMA_VERSION
V3_CONTRACT_PREFIX = "g8econtractcorrectedv3-"
V3_CAMPAIGN_PREFIX = "g8e-v3-"
V3_SOURCE_PREFIX = "g8esourcecorrectedv3-"
V3_DATA_PREFIX = "g8edataidentityv3-"
V3_E2_COMPLETION_PREFIX = "g8ee2completev3-"
V3_E3_PREFIX = "g8ee3v3-"
V3_E4_PREFIX = "g8ee4v3-"

INITIAL_DATASET = v2.INITIAL_DATASET
VALIDATION_SPLIT = v2.VALIDATION_SPLIT
PRODUCTION_PROFILE_ID = v2.PRODUCTION_PROFILE_ID
PRODUCTION_DEVICE = v2.PRODUCTION_DEVICE

G8EV3Error = v2.G8EV2Error
FatalExecutionError = v2.FatalExecutionError
CampaignHoldError = v2.CampaignHoldError
ScientificDecodeFailure = v2.ScientificDecodeFailure
SyntheticSample = v2.SyntheticSample
MeasurementRecordV3 = v2.MeasurementRecordV2
PhysicalCacheKey = v2.PhysicalCacheKey

canonical_json = v2.canonical_json
rendered_json = v2.rendered_json
sha256_bytes = v2.sha256_bytes
sha256_file = v2.sha256_file
_id = v2._id
_digest = v2._digest
_copy = v2._copy
_rendered_object = v2._rendered_object
_strict = v2._strict
_atomic_publish = v2._atomic_publish


class MeasurementExecutorV3(v2.MeasurementExecutorV2):
    """v2 science plus an executable source-byte/stable-ID cross-check."""

    def execute(self, work_unit: Mapping[str, Any], sample: SyntheticSample) -> MeasurementRecordV3:
        from data.identity import stable_sample_id

        if not self.non_scientific_fixture and stable_sample_id(sample.source_bytes) != sample.stable_sample_id:
            raise FatalExecutionError("source payload bytes differ from the frozen stable sample identity")
        return super().execute(work_unit, sample)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def validation_identity_from_manifest_bytes(
    payload: bytes,
    *,
    archive_sha256: str | None = None,
    archive_bytes: int | None = None,
    class_mapping: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Derive the frozen validation identity without opening image payloads."""

    from data.manifests import validate_manifest_bytes

    rows = validate_manifest_bytes(INITIAL_DATASET, payload)
    validation = tuple(row for row in rows if row.split == VALIDATION_SPLIT)
    configured = get(f"datasets.{INITIAL_DATASET}")
    expected_count = int(configured["val_images"])
    if len(validation) != expected_count:
        raise G8EV3Error("Imagenette validation count differs from the frozen count")
    ordered_ids = [row.stable_sample_id for row in validation]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise G8EV3Error("Imagenette validation identity contains duplicates")
    mapping = dict(class_mapping or {
        "n01440764": 0,  # literal-ok: frozen Imagenette class-index vocabulary
        "n02102040": 1,  # literal-ok: frozen Imagenette class-index vocabulary
        "n02979186": 2,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03000684": 3,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03028079": 4,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03394916": 5,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03417042": 6,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03425413": 7,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03445777": 8,  # literal-ok: frozen Imagenette class-index vocabulary
        "n03888257": 9,  # literal-ok: frozen Imagenette class-index vocabulary
    })
    expected_labels = list(range(int(configured["classes"])))
    if sorted(mapping.values()) != expected_labels:
        raise G8EV3Error("Imagenette authoritative class mapping differs")
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_scientific_data_identity_manifest",
        "dataset": INITIAL_DATASET,
        "manifest_path": "data/manifests/imagenette160.csv",
        "manifest_bytes": len(payload),
        "manifest_sha256": sha256_bytes(payload),
        "configured_manifest_sha256": str(configured["manifest_sha256"]),
        "validation_count": len(validation),
        "ordered_validation_stable_ids_sha256": sha256_bytes(canonical_json(ordered_ids)),
        "validation_stable_id_set_sha256": sha256_bytes(canonical_json(sorted(ordered_ids))),
        "ordered_validation_id_label_sha256": sha256_bytes(
            canonical_json([[row.stable_sample_id, row.label] for row in validation])
        ),
        "archive_filename": str(configured["archive_filename"]),
        "archive_bytes": int(configured["archive_bytes"] if archive_bytes is None else archive_bytes),
        "archive_sha256": str(configured["archive_sha256"] if archive_sha256 is None else archive_sha256),
        "source_url": str(configured["source_url"]),
        "loader": str(configured["loader"]),
        "loader_size_arg": str(configured["loader_size_arg"]),
        "class_index_source": str(configured["class_index_source"]),
        "class_mapping": mapping,
        "class_mapping_sha256": sha256_bytes(canonical_json(mapping)),
        "class_count": int(configured["classes"]),
        "expected_split_counts": {
            "train": int(configured["train_images"]),
            "val": int(configured["val_images"]),
            "test": int(configured["test_images"]),
        },
        "stable_id_rule": str(get("datasets.stable_sample_id_rule")),
        "payload_decode_performed": False,
        "test_access": 0,
    }
    body["data_identity_id"] = _id(V3_DATA_PREFIX, body)
    return body


def build_scientific_data_identity(*, verify_archive_bytes: bool = True) -> dict[str, Any]:
    """Bind the already-frozen manifest, archive and authoritative classes."""

    from data.adapters import _adapter
    from data.manifests import manifest_path
    from data.provenance import dataset_root, verify_archive, verify_extracted_dataset

    path = manifest_path(INITIAL_DATASET, REPO_ROOT)
    payload = path.read_bytes()
    if verify_archive_bytes:
        provenance = verify_extracted_dataset(INITIAL_DATASET, REPO_ROOT)
    else:
        configured = get(f"datasets.{INITIAL_DATASET}")
        provenance = type("ConfiguredProvenance", (), {
            "sha256": str(configured["archive_sha256"]),
            "byte_length": int(configured["archive_bytes"]),
        })()
    adapter = _adapter(INITIAL_DATASET, dataset_root(INITIAL_DATASET, REPO_ROOT))
    mapping = dict(adapter.class_mapping())
    result = validation_identity_from_manifest_bytes(
        payload,
        archive_sha256=provenance.sha256,
        archive_bytes=provenance.byte_length,
        class_mapping=mapping,
    )
    if result["manifest_sha256"] != result["configured_manifest_sha256"]:
        raise G8EV3Error("live Imagenette manifest does not match its configured pin")
    return result


def verify_live_validation_identity(expected: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail before payload decode on any manifest/archive/class drift."""

    live = build_scientific_data_identity(verify_archive_bytes=True)
    verify_scientific_data_identity(expected, live)
    from data.manifests import manifest_path, validate_manifest_bytes

    rows = validate_manifest_bytes(
        INITIAL_DATASET,
        manifest_path(INITIAL_DATASET, REPO_ROOT).read_bytes(),
    )
    ids = tuple(row.stable_sample_id for row in rows if row.split == VALIDATION_SPLIT)
    if sha256_bytes(canonical_json(list(ids))) != expected["ordered_validation_stable_ids_sha256"]:
        raise G8EV3Error("live validation ID order differs")
    if sha256_bytes(canonical_json(sorted(ids))) != expected["validation_stable_id_set_sha256"]:
        raise G8EV3Error("live validation ID membership differs")
    return ids


def verify_scientific_data_identity(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if dict(expected) != dict(observed):
        raise G8EV3Error("live Imagenette validation identity differs before payload decode")


def frozen_validation_metadata(expected: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, int]]:
    """Read and authenticate only manifest metadata; never open image payloads."""

    from data.manifests import manifest_path, validate_manifest_bytes

    payload = manifest_path(INITIAL_DATASET, REPO_ROOT).read_bytes()
    if len(payload) != expected["manifest_bytes"] or sha256_bytes(payload) != expected["manifest_sha256"]:
        raise G8EV3Error("Imagenette manifest bytes differ before payload decode")
    rows = validate_manifest_bytes(INITIAL_DATASET, payload)
    ids = tuple(row.stable_sample_id for row in rows if row.split == VALIDATION_SPLIT)
    if len(ids) != expected["validation_count"]:
        raise G8EV3Error("Imagenette validation count differs before payload decode")
    if sha256_bytes(canonical_json(list(ids))) != expected["ordered_validation_stable_ids_sha256"]:
        raise G8EV3Error("Imagenette validation order differs before payload decode")
    if sha256_bytes(canonical_json(sorted(ids))) != expected["validation_stable_id_set_sha256"]:
        raise G8EV3Error("Imagenette validation membership differs before payload decode")
    labels = {row.stable_sample_id: int(row.label) for row in rows if row.split == VALIDATION_SPLIT}
    if sha256_bytes(canonical_json([[sample_id, labels[sample_id]] for sample_id in ids])) != expected["ordered_validation_id_label_sha256"]:
        raise G8EV3Error("Imagenette validation labels differ before payload decode")
    return ids, labels


def frozen_validation_ids(expected: Mapping[str, Any]) -> tuple[str, ...]:
    return frozen_validation_metadata(expected)[0]


def _source_paths() -> tuple[tuple[str, str], ...]:
    return (
        ("src/baseline/g8_e_corrected_v3.py", "v3_e2_e3_e4_lifecycle"),
        ("src/baseline/g8_e_corrected_v2.py", "preserved_v2_scientific_semantics"),
        ("tools/run_g8_e_corrected_v3.py", "owner_gated_v3_runner"),
        ("tools/merge_g8_e_corrected_v3.py", "v3_e3_cli"),
        ("tools/aggregate_g8_e_corrected_v3.py", "v3_e4_cli"),
        ("tools/verify_g8_e_corrected_v3.py", "v3_lifecycle_verifier"),
        ("tools/freeze_g8_e_corrected_v3.py", "v3_pre_data_freezer"),
        ("tools/benchmark_g8_e_v3_scale.py", "v3_complexity_benchmark"),
        ("tools/prove_g8_e_corrected_v3_lifecycle.py", "v3_non_scientific_lifecycle_proof"),
        ("src/baseline/classical/composition.py", "br4_composition"),
        ("src/baseline/classical/records.py", "br11_record_semantics"),
        ("src/baseline/classical/pipeline.py", "classical_verdict_semantics"),
        ("src/baseline/classical/outage.py", "br13_outage_policy"),
        ("src/baseline/classical/channel_transport.py", "transport_accounting"),
        ("src/baseline/j2k.py", "jpeg2000_backend_and_backend_cache"),
        ("src/baseline/g8_d.py", "br11_and_emitted_evidence_authentication"),
        ("src/baseline/ldpc/transport.py", "packet_plan"),
        ("src/baseline/ldpc/segmentation.py", "packet_segmentation"),
        ("src/baseline/ldpc/rate_matching.py", "rate_matching"),
        ("src/baseline/ldpc/modulation.py", "modulation"),
        ("src/data/manifests.py", "manifest_identity"),
        ("src/data/identity.py", "stable_sample_identity"),
        ("src/data/provenance.py", "archive_identity"),
        ("src/data/adapters.py", "authoritative_class_identity"),
        ("src/data/preprocessing.py", "canonical_preprocessing"),
        ("src/data/registry.py", "validation_payload_boundary"),
        ("src/data/test_access.py", "sealed_test_boundary"),
        ("data/manifests/imagenette160.csv", "frozen_validation_membership"),
        ("src/models/frozen_reference_classifier.py", "frozen_g1_loader"),
        ("src/models/reference_classifier.py", "g1_architecture"),
        ("src/config/execution_profiles.py", "execution_profile_authentication"),
        ("src/config/run_config.py", "run_identity"),
        ("src/config/params.py", "parameter_loader"),
        ("src/env.py", "runtime_environment"),
        ("requirements.lock", "python_cuda_dependency_lock"),
        ("docs/deployment-dossier.md", "runtime_cost_receiver_allowance"),
    )


def build_source_manifest(source_commit: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path_text, role in _source_paths():
        path = REPO_ROOT / path_text
        if not path.is_file():
            raise G8EV3Error(f"v3 source binding is missing: {path_text}")
        raw = path.read_bytes()
        entries.append({"path": path_text, "role": role, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    for entry in v2._direct_upstream_bindings() + v2._g1_bindings():
        if entry["path"] not in {item["path"] for item in entries}:
            entries.append(entry)
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_execution_source_manifest",
        "status": "FROZEN_PRE_DATA",
        "source_commit": source_commit,
        "source_entries": entries,
        "source_classes": {
            "code": [entry["path"] for entry in entries if entry["path"].startswith(("src/", "tools/"))],
            "scientific_data_identity": [
                "results/baseline/g8_e/e1_corrected/measurement_authority.json",
                "results/baseline/g8_e/e1_corrected/logical_measurement_mapping.json",
                "data/manifests/imagenette160.csv",
            ],
            "environment": ["requirements.lock", "src/config/execution_profiles.py", "src/env.py"],
        },
        "runtime_outputs_excluded": [
            "results/baseline/g8_e/e1_corrected_v3/runtime/",
            "results/baseline/g8_e/e1_corrected_v3/e2_execution_authorization.json",
        ],
    }
    body["source_manifest_id"] = _id(V3_SOURCE_PREFIX, body)
    return body


def validate_source_manifest(value: Mapping[str, Any], *, verify_live_sources: bool = True) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "status", "source_commit", "source_entries",
        "source_classes", "runtime_outputs_excluded", "source_manifest_id",
    }
    if set(value) != required or value["schema_version"] != V3_SCHEMA_VERSION:
        raise G8EV3Error("v3 source manifest schema differs")
    body = {key: child for key, child in value.items() if key != "source_manifest_id"}
    if value["source_manifest_id"] != _id(V3_SOURCE_PREFIX, body):
        raise G8EV3Error("v3 source manifest ID differs")
    source_commit = str(value["source_commit"])
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{source_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != source_commit:
        raise G8EV3Error("v3 source commit is not an exact available Git commit")
    if verify_live_sources:
        for item in value["source_entries"]:
            entry = _strict(item, ("path", "role", "bytes", "sha256"), "v3 source entry")
            path = REPO_ROOT / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise G8EV3Error(f"v3 frozen source drift: {entry['path']}")
            historical = subprocess.run(
                ["git", "show", f"{source_commit}:{entry['path']}"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
            )
            if historical.returncode != 0 or len(historical.stdout) != entry["bytes"] or sha256_bytes(historical.stdout) != entry["sha256"]:
                raise G8EV3Error(f"v3 source entry is not bound to source_commit: {entry['path']}")
    return dict(value)


def build_storage_plan() -> dict[str, Any]:
    base = v2.build_storage_plan()
    units = int(base["production_units"])
    jobs = int(base["unique_physical_jobs"])
    backend_representative = 6838  # literal-ok: measured mean of 68,000 existing .j2kcache objects
    backend = jobs * backend_representative
    e3_compact = 8192  # literal-ok: conservative compact closure estimate
    e4 = int(base["estimated_bytes"]["e4_output"])
    state = int(base["estimated_bytes"]["runtime_state_and_checkpoints"])
    diagnostics = 8 * 1024 * 1024  # literal-ok: bounded operational reserve, not scientific data
    atomic_temp = max(
        int(base["representative_sizes_bytes"]["reconstruction_cache_object"]),
        backend_representative,
    )
    components = {
        "scientific_records": int(base["estimated_bytes"]["scientific_records"]),
        "v3_codec_cache": int(base["estimated_bytes"]["codec_cache"]),
        "backend_j2k_cache": backend,
        "reconstruction_cache": int(base["estimated_bytes"]["reconstruction_cache_base64"]),
        "classifier_observation_cache": int(base["estimated_bytes"]["classifier_observation_cache"]),
        "mutable_state_and_checkpoints": state,
        "diagnostics_reserve": diagnostics,
        "e2_completion": 8192,  # literal-ok: compact completion estimate
        "e3_compact_closure": e3_compact,
        "e4_output": e4,
        "atomic_publication_peak": atomic_temp,
    }
    subtotal = sum(components.values())
    margin = (subtotal + 3) // 4  # literal-ok: frozen 25% safety margin
    usage = shutil.disk_usage(V3_ROOT.parent)
    stat = os.statvfs(V3_ROOT.parent)
    counts = {
        "scientific_records": units,
        "v3_codec_cache": jobs,
        "backend_j2k_cache": jobs,
        "reconstruction_cache": jobs,
        "classifier_observation_cache_upper_bound": jobs,
        "checkpoints": (units + v2.CHECKPOINT_INTERVAL - 1) // v2.CHECKPOINT_INTERVAL,
        "fixed_outputs_and_state": 16,  # literal-ok: compact fixed artifact allowance
    }
    total_files = sum(counts.values())
    if not V3_COMPLEXITY_PATH.is_file():
        raise G8EV3Error("v3 scale evidence must exist before the storage/runtime plan is frozen")
    complexity, complexity_raw = _rendered_object(V3_COMPLEXITY_PATH, "v3 scale evidence")
    if complexity.get("status") != "PASS" or complexity.get("sizes") != [2500, 5000, 10000, 20000]:  # literal-ok: mandated v3 asymptotic benchmark series
        raise G8EV3Error("v3 scale evidence does not cover the frozen benchmark series")
    extrapolated = complexity["extrapolation_288000"]
    e2_total = float(extrapolated["e2_seconds_linear"])
    checkpoint_fraction = int(extrapolated["checkpoint_count"]) / int(extrapolated["normal_state_publications"])
    checkpoint_seconds = e2_total * checkpoint_fraction
    transaction_seconds = e2_total - checkpoint_seconds
    historical_contract, _ = _rendered_object(
        REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_contract.json",
        "first corrected E1 runtime projection",
    )
    jpeg2000_seconds = float(historical_contract["compute_plan"]["projected_physical_work_seconds"])
    unique_jobs = int(base["unique_physical_jobs"])
    conservative_stage_seconds_per_job = 0.160  # literal-ok: deployment dossier receiver decode+classifier allowance
    reconstruction_seconds = unique_jobs * conservative_stage_seconds_per_job
    classifier_seconds = unique_jobs * conservative_stage_seconds_per_job
    resume_seconds = float(extrapolated["e3_seconds_linear"])
    e3_seconds = float(extrapolated["e3_seconds_linear"])
    e4_seconds = float(extrapolated["e4_seconds_linear"])
    filesystem_seconds = 0.15 * (  # literal-ok: explicit conservative 15% cache/filesystem allowance
        jpeg2000_seconds + reconstruction_seconds + classifier_seconds + e2_total
    )
    runtime_total = sum((
        jpeg2000_seconds,
        reconstruction_seconds,
        classifier_seconds,
        transaction_seconds,
        checkpoint_seconds,
        resume_seconds,
        e3_seconds,
        e4_seconds,
        filesystem_seconds,
    ))
    return {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_compute_storage_plan",
        "basis": {
            "v2_plan_sha256": sha256_file(v2.V2_STORAGE_PLAN_PATH),
            "backend_j2k_cache_path": "runtime/backend/<sha256>.j2kcache",
            "backend_j2k_representative_bytes": backend_representative,
            "backend_sample_count": 68000,  # literal-ok: existing transparency cache measurement
            "backend_role": "operational and reconstructible; required for efficient E2 resume; retained through E3",
            "backend_post_e3": "eligible for explicit owner cleanup only after immutable E3; never auto-deleted",
        },
        "estimated_bytes": {
            **components,
            "subtotal": subtotal,
            "safety_margin_25_percent": margin,
            "required_with_safety_margin": subtotal + margin,
            "actual_free_bytes_at_freeze": int(usage.free),
        },
        "projected_files": {**counts, "total": total_files, "largest_single_directory": units, "largest_single_directory_name": "runtime/records"},
        "filesystem": {
            "available_inodes_at_freeze": int(stat.f_favail),
            "total_inodes": int(stat.f_files),
            "inode_headroom_after_projection": int(stat.f_favail) - total_files,
            "simple_one_directory_layout_measured_safe": int(stat.f_favail) > total_files * 2 and units < 1_000_000,  # literal-ok: conservative operational guard
            "sharding": "none; hash-named ext4 htree directory retained because measured capacity is ample",
        },
        "preflight_before_payload_decode": True,
        "production_runtime_estimate": {
            "status": "PASS_PLANNING_ESTIMATE_ONLY",
            "jpeg2000_physical_work_seconds": jpeg2000_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "classifier_observations_seconds": classifier_seconds,
            "e2_transaction_seconds": transaction_seconds,
            "checkpoints_seconds": checkpoint_seconds,
            "resume_reconciliation_allowance_seconds": resume_seconds,
            "final_e3_seconds": e3_seconds,
            "e4_seconds": e4_seconds,
            "filesystem_cache_allowance_seconds": filesystem_seconds,
            "total_seconds": runtime_total,
            "total_hours": runtime_total / 3600,  # literal-ok: seconds per hour
            "practical_on_frozen_local_profile": True,
            "basis": {
                "jpeg2000": "conservatively assigns the entire historical corrected-E1 physical-work proxy to JPEG2000 rather than calling it total campaign runtime",
                "reconstruction_and_classifier": "each independently receives the full 160 ms receiver allowance from docs/deployment-dossier.md; deliberate double allocation is conservative",
                "transaction_e3_e4": "linear extrapolation from the largest N=20,000 v3 synthetic benchmark",
                "resume": "one complete record-authentication traversal allowance",
                "filesystem": "explicit 15% allowance in addition to measured atomic transaction time",
                "historical_projection_path": "results/baseline/g8_e/e1_corrected/measurement_contract.json",
                "historical_projection_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_contract.json"),
                "deployment_dossier_sha256": sha256_file(REPO_ROOT / "docs/deployment-dossier.md"),
                "complexity_evidence_path": _relative(V3_COMPLEXITY_PATH),
                "complexity_evidence_sha256": sha256_bytes(complexity_raw),
            },
        },
    }


def storage_preflight(plan: Mapping[str, Any], path: Path = V3_RUNTIME_ROOT) -> dict[str, Any]:
    target = Path(path).resolve()
    parent = target.parent if target.parent.exists() else REPO_ROOT
    usage = shutil.disk_usage(parent)
    stat = os.statvfs(parent)
    required = int(plan["estimated_bytes"]["required_with_safety_margin"])
    required_files = int(plan["projected_files"]["total"])
    result = {
        "path": str(target),
        "required_bytes": required,
        "available_free_bytes": int(usage.free),
        "required_files": required_files,
        "available_inodes": int(stat.f_favail),
        "passed": int(usage.free) >= required and int(stat.f_favail) >= required_files,
    }
    if not result["passed"]:
        raise G8EV3Error("v3 storage/inode preflight failed")
    return result


def build_correction_provenance() -> dict[str, Any]:
    v2_contract, v2_raw = _rendered_object(v2.V2_CONTRACT_PATH, "v2 historical contract")
    if v2_contract.get("safety", {}).get("measurement_coverage") != 0 or v2.V2_RUNTIME_ROOT.exists() or (v2.V2_ROOT / "e2_execution_authorization.json").exists():
        raise G8EV3Error("v2 scientific evidence exists; pre-data supersession is forbidden")
    return {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrective_v3_provenance",
        "status": "CURRENT_PRE_DATA_SUPERSESSION",
        "scientific_measurement_coverage": 0,
        "superseded_v2": {
            "status": "superseded-before-data",
            "contract_id": v2_contract["contract_id"],
            "campaign_id": v2_contract["campaign_id"],
            "contract_sha256": sha256_bytes(v2_raw),
            "source_epoch": v2_contract["source_manifest"]["source_commit"],
        },
        "reasons": [
            "verification conflated immutable contract authenticity with pre-data zero state",
            "compact state loads recomputed the full authority-order digest",
            "E3 membership and structural lookup were quadratic",
            "E3 was not an immutable content-addressed artifact consumed by E4",
            "the actual Imagenette validation membership/order/class/archive identity was not directly frozen",
            "E3 authenticated reconstruction and observation references by existence only",
            "the backend .j2kcache storage and inode burden was omitted",
            "E3 CLI printed the complete artifact",
            "resume did not independently authenticate the durable exact prefix before validation payload opening",
            "E3 did not rederive emitted-codestream and BR-11 accounting from authenticated codec bytes",
            "E3 did not cross-check each scientific row's class label against the frozen manifest",
            "the source-byte digest was not executable-linked back to the stable sample ID",
        ],
        "zero_data_audit": {
            "production_e2_accepted_records": 0,
            "production_e2_completed_units": 0,
            "real_validation_payload_decoding": 0,
            "e3_production_artifact": False,
            "e4_production_artifact": False,
            "pass_one": False,
            "training": 0,
            "pass_two": 0,
            "fallback": False,
            "ratio_adjudication": False,
            "test_access": 0,
        },
        "science_preserved": {
            "logical_initial_snr_cells": 6048,  # literal-ok: immutable authority fact
            "structural_initial": 288,  # literal-ok: immutable authority fact
            "work_units": 288000,  # literal-ok: immutable authority fact
            "authority_and_mapping_reused_byte_identically": True,
            "g8_c_unchanged": True,
            "g8_d_unchanged": True,
            "spec_amendment": False,
        },
    }


def build_contract(
    source: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> dict[str, Any]:
    authority = v2._authority_binding()
    mapping = v2._mapping_binding(authority)
    full_authority = load_measurement_authority()
    if full_authority["authority_id"] != authority["authority_id"]:
        raise G8EV3Error("v3 full authority differs from its frozen binding")
    production_work_units = expected_work_units(full_authority, frozen_validation_ids(data_identity))
    production_authority_order_sha256 = sha256_bytes(
        canonical_json([unit["work_unit_id"] for unit in production_work_units])
    )
    v2_contract, v2_raw = _rendered_object(v2.V2_CONTRACT_PATH, "v2 historical contract")
    selection_plan = v2._selection_call_plan()
    if selection_plan != v2_contract["selection_authorization"]:
        raise G8EV3Error("v3 mechanically derived pass-one call plan differs from v2")
    seed = {
        "schema_version": V3_SCHEMA_VERSION,
        "semantics_epoch": "g8_e_corrected_v3_lifecycle_linear_exact_set",
        "v2_contract_id": v2_contract["contract_id"],
        "v2_contract_sha256": sha256_bytes(v2_raw),
        "authority_id": authority["authority_id"],
        "authority_sha256": authority["sha256"],
        "mapping_id": mapping["mapping_id"],
        "mapping_sha256": mapping["sha256"],
        "source_manifest_id": source["source_manifest_id"],
        "source_manifest_sha256": sha256_bytes(rendered_json(source)),
        "data_identity_id": data_identity["data_identity_id"],
        "data_identity_sha256": sha256_bytes(rendered_json(data_identity)),
    }
    campaign_id = _id(V3_CAMPAIGN_PREFIX, seed)
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_corrective_v3_executable_pre_data_contract",
        "phase": "G8_E",
        "checkpoint": "E1_corrected_v3",
        "status": "FROZEN_PRE_DATA_EXECUTABLE",
        "campaign_id": campaign_id,
        "campaign_seed": seed,
        "contract_id": None,
        "supersedes_before_data": {
            "original_e1": v2_contract["supersedes_before_data"]["original_e1"],
            "first_corrected_e1": v2_contract["supersedes_before_data"]["first_corrected_e1"],
            "corrected_v2": {
                "contract_id": v2_contract["contract_id"],
                "campaign_id": v2_contract["campaign_id"],
                "sha256": sha256_bytes(v2_raw),
                "coverage": 0,
            },
        },
        "authority": authority,
        "mapping": mapping,
        "source_manifest": {
            "path": _relative(V3_SOURCE_MANIFEST_PATH),
            "id": source["source_manifest_id"],
            "sha256": sha256_bytes(rendered_json(source)),
            "source_commit": source["source_commit"],
        },
        "scientific_data_identity": {
            "path": _relative(V3_DATA_IDENTITY_PATH),
            "id": data_identity["data_identity_id"],
            "sha256": sha256_bytes(rendered_json(data_identity)),
            "manifest_sha256": data_identity["manifest_sha256"],
            "validation_count": data_identity["validation_count"],
            "ordered_validation_stable_ids_sha256": data_identity["ordered_validation_stable_ids_sha256"],
            "validation_stable_id_set_sha256": data_identity["validation_stable_id_set_sha256"],
        },
        "direct_upstream_bindings": v2._direct_upstream_bindings(),
        "g1_bindings": v2._g1_bindings(),
        "execution_profile": _copy(v2_contract["execution_profile"]),
        "codec": _copy(v2_contract["codec"]),
        "classifier": _copy(v2_contract["classifier"]),
        "outage_policy": _copy(v2_contract["outage_policy"]),
        "clean_measurement_semantics": _copy(v2_contract["clean_measurement_semantics"]),
        "transaction": {
            **_copy(v2_contract["transaction"]),
            "production_total_required": len(production_work_units),
            "production_authority_order_sha256": production_authority_order_sha256,
            "authority_order_digest_computations_per_process_max": 1,
            "full_authority_id_visits_during_normal_progression": 0,
            "durable_record_before_state_interruption": "crash_reconcile_not_scientific_hold",
            "failure_before_durable_record": "HOLD_with_non_scientific_diagnostic",
        },
        "lifecycle_verifiers": {
            "frozen_contract": "phase_invariant",
            "predata_zero_state": "E1_readiness_only",
            "active_e2": "authorization_plus_current_runtime",
            "e2_complete": "exact_completion_plus_immutable_completion_artifact",
            "e3_complete": "immutable_content_addressed_exact_set_closure",
            "e4_complete": "count_derived_output_bound_to_exact_E3_id_and_sha256",
        },
        "e3": {
            "complexity": "O(N)",
            "indexed_structures": ["expected_by_id", "expected_by_ordinal", "structural_by_id"],
            "cache_authentication": ["physical_key", "codec", "reconstruction", "classifier_observation", "outage_policy"],
            "artifact_path": _relative(V3_E3_PATH),
        },
        "e4": {
            "complexity": "O(N)",
            "requires_exact_e3_id_and_sha256": True,
            "complete_e3_ingest_repeated": False,
            "artifact_path": _relative(V3_E4_PATH),
        },
        "selection_authorization": selection_plan,
        "compute_plan": {
            "physical": _copy(v2_contract["compute_plan"]["physical"]),
            "storage_path": _relative(V3_STORAGE_PLAN_PATH),
            "storage_sha256": None,
        },
        "authorization": {
            "required": True,
            "issued": False,
            "path": _relative(V3_AUTHORIZATION_PATH),
            "artifact_role": "g8_e_v3_owner_e2_authorization",
            "required_bindings": ["campaign_id", "contract_id", "source_manifest_id", "source_manifest_sha256", "data_identity_id", "data_identity_sha256", "profile_id", "scope"],
            "scope": _copy(v2_contract["authorization"]["schema_scope_frozen"]),
            "refuse_before_validation_payload_decode": True,
        },
        "safety": {
            "measurement_coverage": 0,
            "e2_completed_units": 0,
            "e3_present": False,
            "e4_present": False,
            "pass_one_started": False,
            "pass_one_completed": False,
            "training": 0,
            "pass_two": 0,
            "fallback_invoked": False,
            "ratio_adjudicated": False,
            "test_access": 0,
            "validation_decoding": 0,
        },
    }
    body["compute_plan"]["storage_sha256"] = sha256_bytes(rendered_json(storage))
    body["contract_id"] = _id(V3_CONTRACT_PREFIX, {key: child for key, child in body.items() if key != "contract_id"})
    return body


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != V3_SCHEMA_VERSION or value.get("checkpoint") != "E1_corrected_v3" or value.get("status") != "FROZEN_PRE_DATA_EXECUTABLE":
        raise G8EV3Error("v3 contract is not the frozen corrected-v3 epoch")
    body = {key: child for key, child in value.items() if key != "contract_id"}
    if value.get("contract_id") != _id(V3_CONTRACT_PREFIX, body):
        raise G8EV3Error("v3 contract ID differs")
    if value.get("campaign_id") in {
        v2.ORIGINAL_CAMPAIGN_ID,
        v2.FIRST_CORRECTED_CAMPAIGN_ID,
        _rendered_object(v2.V2_CONTRACT_PATH, "v2 contract")[0]["campaign_id"],
    }:
        raise G8EV3Error("a superseded E1 campaign remains current")
    if value.get("authorization", {}).get("issued") is not False or value.get("safety", {}).get("measurement_coverage") != 0:
        raise G8EV3Error("the immutable v3 contract itself must remain pre-data")
    return dict(value)


def verify_v3_frozen_contract(*, verify_live_sources: bool = True, verify_live_data: bool = True) -> dict[str, Any]:
    """Phase-invariant verification valid before, during and after E2-E4."""

    contract, contract_raw = _rendered_object(V3_CONTRACT_PATH, "v3 measurement contract")
    source, source_raw = _rendered_object(V3_SOURCE_MANIFEST_PATH, "v3 source manifest")
    data, data_raw = _rendered_object(V3_DATA_IDENTITY_PATH, "v3 scientific data identity")
    correction, correction_raw = _rendered_object(V3_CORRECTION_PATH, "v3 correction provenance")
    storage, storage_raw = _rendered_object(V3_STORAGE_PLAN_PATH, "v3 storage plan")
    validate_source_manifest(source, verify_live_sources=verify_live_sources)
    validate_contract(contract)
    if contract["source_manifest"] != {
        "path": _relative(V3_SOURCE_MANIFEST_PATH),
        "id": source["source_manifest_id"],
        "sha256": sha256_bytes(source_raw),
        "source_commit": source["source_commit"],
    }:
        raise G8EV3Error("v3 contract/source manifest binding differs")
    if contract["scientific_data_identity"]["id"] != data.get("data_identity_id") or contract["scientific_data_identity"]["sha256"] != sha256_bytes(data_raw):
        raise G8EV3Error("v3 contract/data identity binding differs")
    data_body = {key: child for key, child in data.items() if key != "data_identity_id"}
    if data.get("data_identity_id") != _id(V3_DATA_PREFIX, data_body):
        raise G8EV3Error("v3 scientific data identity ID differs")
    if verify_live_data and data != build_scientific_data_identity(verify_archive_bytes=True):
        raise G8EV3Error("v3 live scientific data identity drifted")
    required = int(storage.get("estimated_bytes", {}).get("required_with_safety_margin", -1))
    subtotal = int(storage.get("estimated_bytes", {}).get("subtotal", -1))
    margin = int(storage.get("estimated_bytes", {}).get("safety_margin_25_percent", -1))
    if required != subtotal + margin or margin != (subtotal + 3) // 4:  # literal-ok: exact integer ceiling for frozen 25% margin
        raise G8EV3Error("v3 storage plan arithmetic differs")
    runtime_estimate = storage.get("production_runtime_estimate", {})
    runtime_components = (
        "jpeg2000_physical_work_seconds", "reconstruction_seconds", "classifier_observations_seconds",
        "e2_transaction_seconds", "checkpoints_seconds", "resume_reconciliation_allowance_seconds",
        "final_e3_seconds", "e4_seconds", "filesystem_cache_allowance_seconds",
    )
    if (
        runtime_estimate.get("status") != "PASS_PLANNING_ESTIMATE_ONLY"
        or runtime_estimate.get("practical_on_frozen_local_profile") is not True
        or abs(sum(float(runtime_estimate.get(key, -1)) for key in runtime_components) - float(runtime_estimate.get("total_seconds", -2))) > 1e-6  # literal-ok: arithmetic serialization tolerance only
        or runtime_estimate.get("basis", {}).get("complexity_evidence_sha256") != sha256_file(V3_COMPLEXITY_PATH)
    ):
        raise G8EV3Error("v3 production runtime estimate differs")
    if contract.get("compute_plan", {}).get("storage_sha256") != sha256_bytes(storage_raw):
        raise G8EV3Error("v3 contract/storage plan binding differs")
    if contract != build_contract(source, data, storage):
        raise G8EV3Error("v3 contract does not independently reproduce from frozen inputs")
    if correction != build_correction_provenance():
        raise G8EV3Error("v3 correction provenance does not independently reproduce")
    storage_preflight(storage)
    if correction.get("scientific_measurement_coverage") != 0:
        raise G8EV3Error("v3 correction provenance claims scientific coverage")
    v2._direct_upstream_bindings()
    return {
        "contract": contract,
        "source_manifest": source,
        "scientific_data_identity": data,
        "correction_provenance": correction,
        "storage_plan": storage,
        "contract_sha256": sha256_bytes(contract_raw),
        "source_manifest_sha256": sha256_bytes(source_raw),
        "data_identity_sha256": sha256_bytes(data_raw),
        "correction_provenance_sha256": sha256_bytes(correction_raw),
        "storage_plan_sha256": sha256_bytes(storage_raw),
    }


def verify_v3_predata_zero_state(**kwargs: Any) -> dict[str, Any]:
    """E1-only proof that no legitimate lifecycle transition has begun."""

    bundle = verify_v3_frozen_contract(**kwargs)
    forbidden = [V3_AUTHORIZATION_PATH, V3_RUNTIME_ROOT, V3_E2_COMPLETION_PATH, V3_E3_PATH, V3_E4_PATH]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise G8EV3Error(f"v3 pre-data zero state is closed by legitimate/foreign lifecycle artifacts: {present}")
    return {**bundle, "phase": "PRE_DATA_ZERO", "production_e2_records": 0, "production_e2_completed_units": 0}


def authenticate_owner_authorization_v3(path: Path, contract: Mapping[str, Any], data_identity: Mapping[str, Any]) -> dict[str, Any]:
    value, _ = _rendered_object(Path(path), "v3 owner E2 authorization")
    required = {
        "schema_version", "artifact_role", "status", "authorized_by", "reason",
        "campaign_id", "contract_id", "source_manifest_id", "source_manifest_sha256",
        "data_identity_id", "data_identity_sha256", "profile_id", "scope", "issued_sha256",
    }
    if set(value) != required or value["schema_version"] != V3_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v3_owner_e2_authorization" or value["status"] != "AUTHORIZED":
        raise G8EV3Error("v3 owner authorization schema/status differs")
    expected = {
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "data_identity_id": data_identity["data_identity_id"],
        "data_identity_sha256": contract["scientific_data_identity"]["sha256"],
        "profile_id": contract["execution_profile"]["profile_id"],
        "scope": contract["authorization"]["scope"],
    }
    for key, child in expected.items():
        if value.get(key) != child:
            raise G8EV3Error(f"v3 owner authorization {key} binding differs")
    body = {key: child for key, child in value.items() if key != "issued_sha256"}
    if value["issued_sha256"] != sha256_bytes(canonical_json(body)):
        raise G8EV3Error("v3 owner authorization digest differs")
    if not str(value["authorized_by"]).strip() or not str(value["reason"]).strip():
        raise G8EV3Error("v3 owner authorization lacks accountable text")
    return value


class AtomicE2CampaignV3(v2.AtomicE2CampaignV2):
    """v2 compact transaction with one cached authority traversal per process."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.full_authority_digest_computations = 0
        self.full_authority_id_visits_initialization = 0
        self.full_authority_id_visits_during_progression = 0
        self.state_loads = 0
        self.record_validations = 0
        self._authority_order_sha256_cached: str | None = None
        self._progression_started = False
        super().__init__(*args, **kwargs)
        self._progression_started = True

    def _authority_order_sha256(self) -> str:
        if self._authority_order_sha256_cached is None:
            ids = [unit["work_unit_id"] for unit in self.work_units]
            self.full_authority_digest_computations += 1
            if self._progression_started:
                self.full_authority_id_visits_during_progression += len(ids)
            else:
                self.full_authority_id_visits_initialization += len(ids)
            self._authority_order_sha256_cached = sha256_bytes(canonical_json(ids))
        return self._authority_order_sha256_cached

    def _load_state(self) -> dict[str, Any]:
        self.state_loads += 1
        return super()._load_state()

    def _read_record(self, path: Path) -> tuple[MeasurementRecordV3, bytes]:
        self.record_validations += 1
        return super()._read_record(path)

    def instrumentation(self) -> dict[str, int]:
        return {
            "authority_order_digest_computations": self.full_authority_digest_computations,
            "full_authority_id_visits_initialization": self.full_authority_id_visits_initialization,
            "full_authority_id_visits_during_normal_progression": self.full_authority_id_visits_during_progression,
            "state_loads": self.state_loads,
            "state_writes": self.state_publications,
            "record_validations": self.record_validations,
            "bytes_written": self.state_bytes_written + self.checkpoint_bytes_written + self.record_bytes_written,
            "checkpoint_count": len(list(self.checkpoints_dir.glob("*.json"))) if self.checkpoints_dir.exists() else 0,
        }


def expected_work_units(authority: Mapping[str, Any], sample_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    return v2.expected_work_units(authority, sample_ids)


def load_measurement_authority() -> dict[str, Any]:
    return v2.load_measurement_authority()


def _state_for_runtime(runtime_root: Path) -> dict[str, Any]:
    state, _ = _rendered_object(Path(runtime_root) / "campaign_state.json", "v3 active campaign state")
    return state


def verify_runtime_prefix_readonly(
    *,
    runtime_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Independently authenticate compact state and its exact durable prefix."""

    root = Path(runtime_root)
    state = _state_for_runtime(root)
    if set(state) != set(v2.AtomicE2CampaignV2.STATE_FIELDS):
        raise G8EV3Error("v3 compact state schema differs")
    body = {key: child for key, child in state.items() if key != "state_sha256"}
    if state["state_sha256"] != sha256_bytes(canonical_json(body)):
        raise G8EV3Error("v3 compact state digest differs")
    expected = expected_work_units(authority, sample_ids)
    expected_order = sha256_bytes(canonical_json([unit["work_unit_id"] for unit in expected]))
    if (
        state["schema_version"] != V3_STATE_SCHEMA_VERSION
        or state["artifact_role"] != "g8_e_v2_campaign_state"
        or state["campaign_id"] != contract["campaign_id"]
        or state["contract_id"] != contract["contract_id"]
        or state["measurement_authority_id"] != authority["authority_id"]
        or state["total_required"] != len(expected)
        or state["authority_order_sha256"] != expected_order
    ):
        raise G8EV3Error("v3 compact state binding differs")
    if expected_order != contract["transaction"]["production_authority_order_sha256"]:
        raise G8EV3Error("v3 runtime authority differs from the frozen production order")
    count = state["completed_prefix_count"]
    if type(count) is not int or not 0 <= count <= len(expected):
        raise G8EV3Error("v3 compact state prefix count differs")
    records_dir = root / "records"
    paths = tuple(records_dir.iterdir()) if records_dir.is_dir() else ()
    if any(path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in paths):
        raise G8EV3Error("v3 durable prefix contains a foreign filesystem object")
    if len(paths) not in {count, count + 1} or len(paths) > len(expected):
        raise G8EV3Error("v3 compact state and durable record count are not exactly reconciliable")
    expected_names = {f"{expected[index]['work_unit_id']}.json" for index in range(len(paths))}
    if {path.name for path in paths} != expected_names:
        raise G8EV3Error("v3 durable records are not one exact authority prefix")
    rolling = sha256_bytes(canonical_json({
        "chain_version": V3_STATE_SCHEMA_VERSION,
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "measurement_authority_id": authority["authority_id"],
        "total_required": len(expected),
    }))
    counters = v2.AtomicE2CampaignV2._counters_zero()
    for ordinal in range(count):
        unit = expected[ordinal]
        path = records_dir / f"{unit['work_unit_id']}.json"
        value, raw = _rendered_object(path, "v3 durable prefix record")
        record = MeasurementRecordV3.from_mapping(value)
        if (
            record.value["authority_ordinal"] != ordinal
            or record.value["work_unit_id"] != unit["work_unit_id"]
            or record.value["campaign_id"] != contract["campaign_id"]
            or record.value["contract_id"] != contract["contract_id"]
        ):
            raise G8EV3Error("v3 durable record prefix differs")
        rolling = sha256_bytes(canonical_json({
            "previous_digest": rolling,
            "authority_ordinal": ordinal,
            "work_unit_id": unit["work_unit_id"],
            "record_sha256": sha256_bytes(raw),
        }))
        one = v2.AtomicE2CampaignV2._counter_for(record)
        for key in counters:
            counters[key] += one[key]
    if len(paths) == count + 1:
        unit = expected[count]
        value, _ = _rendered_object(records_dir / f"{unit['work_unit_id']}.json", "v3 reconciliable durable record")
        record = MeasurementRecordV3.from_mapping(value)
        if (
            record.value["authority_ordinal"] != count
            or record.value["work_unit_id"] != unit["work_unit_id"]
            or record.value["campaign_id"] != contract["campaign_id"]
            or record.value["contract_id"] != contract["contract_id"]
        ):
            raise G8EV3Error("v3 reconciliable durable record differs from the next authority unit")
    expected_last = None if count == 0 else expected[count - 1]["work_unit_id"]
    if state["rolling_prefix_digest"] != rolling or state["counters"] != counters or state["last_completed_work_unit_id"] != expected_last:
        raise G8EV3Error("v3 compact state does not reproduce from its durable prefix")
    if state["status"] not in {v2.READY_STATUS, v2.RUNNING_STATUS, v2.HOLD_STATUS, v2.COMPLETE_STATUS}:
        raise G8EV3Error("v3 compact state status differs")
    claim = state["in_progress"]
    if claim is not None and (
        set(claim) != {"ordinal", "work_unit_id", "transaction_id"}
        or claim["ordinal"] != count
        or count == len(expected)
        or claim["work_unit_id"] != expected[count]["work_unit_id"]
    ):
        raise G8EV3Error("v3 compact state next-unit claim differs")
    if state["status"] == v2.COMPLETE_STATUS and count != len(expected):
        raise G8EV3Error("v3 compact state claims premature completion")
    return state


def verify_v3_active_e2(*, runtime_root: Path = V3_RUNTIME_ROOT, authorization_path: Path = V3_AUTHORIZATION_PATH, **kwargs: Any) -> dict[str, Any]:
    bundle = verify_v3_frozen_contract(**kwargs)
    authorization = authenticate_owner_authorization_v3(authorization_path, bundle["contract"], bundle["scientific_data_identity"])
    authority = load_measurement_authority()
    sample_ids = frozen_validation_ids(bundle["scientific_data_identity"])
    state = verify_runtime_prefix_readonly(
        runtime_root=runtime_root,
        contract=bundle["contract"],
        authority=authority,
        sample_ids=sample_ids,
    )
    return {**bundle, "authorization": authorization, "state": state, "phase": "ACTIVE_E2"}


def build_e2_completion(
    *,
    runtime_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    production: bool = True,
) -> dict[str, Any]:
    state = _state_for_runtime(runtime_root)
    if state.get("status") != v2.COMPLETE_STATUS or state.get("completed_prefix_count") != state.get("total_required"):
        raise G8EV3Error("E2 completion requires an exact complete runtime")
    if state.get("campaign_id") != contract["campaign_id"] or state.get("contract_id") != contract["contract_id"] or state.get("measurement_authority_id") != authority["authority_id"]:
        raise G8EV3Error("E2 completion runtime is foreign")
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_e2_completion",
        "status": "E2_COMPLETE",
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "measurement_authority_id": authority["authority_id"],
        "authority_order_sha256": state["authority_order_sha256"],
        "required_work_unit_count": state["total_required"],
        "completed_work_unit_count": state["completed_prefix_count"],
        "rolling_prefix_digest": state["rolling_prefix_digest"],
        "counters": _copy(state["counters"]),
        "test_access": 0,
        "training": 0,
        "pass_one": False,
        "production": production,
        "record_labels": [] if production else [
            "NON-SCIENTIFIC",
            "NON-SELECTION",
            "NOT PRODUCTION E2 EVIDENCE",
            "MERGE-INELIGIBLE FOR PRODUCTION",
        ],
    }
    body["completion_id"] = _id(V3_E2_COMPLETION_PREFIX, body)
    body["artifact_content_sha256"] = sha256_bytes(canonical_json(body))
    return body


def publish_e2_completion(
    *,
    runtime_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    production: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    value = build_e2_completion(
        runtime_root=runtime_root,
        contract=contract,
        authority=authority,
        production=production,
    )
    path = Path(runtime_root) / "e2_completion.json"
    raw = rendered_json(value)
    _atomic_publish(path, raw)
    return value, path, sha256_bytes(raw)


def verify_e2_completion_artifact(
    *,
    runtime_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    production: bool,
) -> tuple[dict[str, Any], str]:
    expected = build_e2_completion(
        runtime_root=runtime_root,
        contract=contract,
        authority=authority,
        production=production,
    )
    path = Path(runtime_root) / "e2_completion.json"
    observed, raw = _rendered_object(path, "v3 E2 completion")
    if observed != expected:
        raise G8EV3Error("v3 E2 completion artifact differs")
    return observed, sha256_bytes(raw)


def verify_v3_e2_complete(*, runtime_root: Path = V3_RUNTIME_ROOT, completion_path: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    active = verify_v3_active_e2(runtime_root=runtime_root, **kwargs)
    if completion_path is not None and completion_path != Path(runtime_root) / "e2_completion.json":
        raise G8EV3Error("v3 E2 completion path differs from the immutable runtime location")
    observed, digest = verify_e2_completion_artifact(
        runtime_root=runtime_root,
        contract=active["contract"],
        authority=load_measurement_authority(),
        production=True,
    )
    return {**active, "completion": observed, "completion_sha256": digest, "phase": "E2_COMPLETE"}


def _expected_indexes(authority: Mapping[str, Any], sample_ids: Sequence[str]) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    expected = expected_work_units(authority, sample_ids)
    expected_by_id = {str(unit["work_unit_id"]): unit for unit in expected}
    structural_by_id = {
        str(row["structural_identity_id"]): row
        for row in authority.get("structural_identities", ())
        if row.get("dataset") == INITIAL_DATASET
    }
    if len(expected_by_id) != len(expected) or not structural_by_id:
        raise G8EV3Error("v3 E3 expected authority index is malformed")
    return expected, expected_by_id, structural_by_id


def _load_observation(path: Path, *, reconstruction_id: str, reconstruction_sha256: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    value, _ = _rendered_object(path, "v3 classifier observation")
    required = {"schema_version", "artifact_role", "identity", "predicted_label", "object_id", "object_sha256"}
    if set(value) != required or value["schema_version"] != v2.V2_SCHEMA_VERSION or value["artifact_role"] != "g8_e_v2_classifier_observation_cache_object":
        raise G8EV3Error("classifier observation schema differs")
    identity = value["identity"]
    expected_identity = {
        "schema_version": v2.V2_SCHEMA_VERSION,
        "reconstruction_object_id": reconstruction_id,
        "reconstruction_sha256": reconstruction_sha256,
        "classifier_checkpoint_sha256": contract["classifier"]["checkpoint_sha256"],
        "classifier_config_identity": contract["classifier"]["config_identity"],
        "inference_runtime_identity": contract["classifier"]["runtime_identity"],
    }
    if identity != expected_identity:
        raise G8EV3Error("classifier observation identity differs")
    if value["object_id"] != _id(v2.V2_OBSERVATION_PREFIX, identity):
        raise G8EV3Error("classifier observation object ID differs")
    object_body = {key: child for key, child in value.items() if key != "object_sha256"}
    if value["object_sha256"] != sha256_bytes(canonical_json(object_body)):
        raise G8EV3Error("classifier observation object digest differs")
    if type(value["predicted_label"]) is not int or not 0 <= value["predicted_label"] < int(get("datasets.imagenette160.classes")):
        raise G8EV3Error("classifier observation prediction differs")
    return value


def _authenticate_record_caches(*, runtime_root: Path, record: MeasurementRecordV3, structural: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, int]:
    value = record.value
    summary = {"physical_key": 0, "codec": 0, "reconstruction": 0, "observation": 0, "outage": 0}
    physical = value["physical_cache_key"]
    codec: v2.CodecArtifactV2 | None = None
    if physical is not None:
        key = PhysicalCacheKey(
            source_bytes_sha256=physical["source_bytes_sha256"],
            canonical_pixels_sha256=physical["canonical_pixels_sha256"],
            canonical_shape=tuple(physical["canonical_shape"]),
            payload_budget_bytes=physical["payload_budget_bytes"],
            encode_axis_px=physical["encode_axis_px"],
            codec_configuration_hash=physical["codec_configuration_hash"],
            codec_runtime_identity=physical["codec_runtime_identity"],
        )
        if physical["source_bytes_sha256"] != value["source_bytes_sha256"] or physical["canonical_pixels_sha256"] != value["canonical_pixels_sha256"] or physical["canonical_shape"] != value["canonical_shape"]:
            raise G8EV3Error("E3 physical key source/canonical identity differs")
        if physical["payload_budget_bytes"] != structural["payload_budget_bytes"] or physical["encode_axis_px"] != structural["encode_axis_px"]:
            raise G8EV3Error("E3 physical key budget/axis differs")
        if physical["codec_configuration_hash"] != contract["codec"]["configuration_hash"] or physical["codec_runtime_identity"] != contract["codec"]["runtime_identity"]:
            raise G8EV3Error("E3 physical key codec identity differs")
        summary["physical_key"] = 1
        codec_path = Path(runtime_root) / "codec" / f"{key.key_id}.json"
        if not codec_path.is_file():
            raise G8EV3Error("E3 codec cache reference is absent")
        try:
            codec = v2.PhysicalCodecCacheV2(runtime_root, None)._load(codec_path, key)
        except v2.FatalExecutionError as exc:
            raise G8EV3Error(f"E3 codec cache authentication failed: {exc}") from exc
        if codec.cache_object_id != value["codec_cache_object_id"]:
            raise G8EV3Error("E3 codec cache object ID differs")
        if codec.status == "feasible":
            if codec.codestream is None:
                raise G8EV3Error("E3 feasible codec object lacks emitted bytes")
            emitted = {
                "sha256": sha256_bytes(codec.codestream),
                "bytes": len(codec.codestream),
            }
            if value["emitted_codestream"] != emitted:
                raise G8EV3Error("E3 record/emitted codestream identity differs")
            try:
                from baseline.g8_d import EmittedFileIdentity, account_br11

                emitted_identity = EmittedFileIdentity(
                    codec_search_key_id=key.key_id,
                    codestream_sha256=emitted["sha256"],
                    emitted_bytes=emitted["bytes"],
                    payload_budget_bytes=key.payload_budget_bytes,
                    filler_bytes=key.payload_budget_bytes - emitted["bytes"],
                )
                expected_br11 = account_br11(
                    codec.codestream,
                    emitted_file_identity=emitted_identity,
                    bytes_sent=key.payload_budget_bytes,
                    verdict=value["outcome"],
                ).as_dict()
            except Exception as exc:
                raise G8EV3Error(f"E3 BR-11 authentication failed: {exc}") from exc
            if value["br11"] != expected_br11:
                raise G8EV3Error("E3 record BR-11 accounting differs from emitted bytes")
        elif value["emitted_codestream"] is not None or value["br11"] is not None:
            raise G8EV3Error("E3 infeasible codec object has emitted accounting")
        summary["codec"] = 1
    elif value["outcome"] != v2.OUTCOME_STRUCTURAL_INFEASIBILITY:
        raise G8EV3Error("E3 non-structural row has no physical key")
    reconstruction_ref = value["reconstruction"]
    reconstruction: v2.ReconstructionArtifactV2 | None = None
    if reconstruction_ref is not None:
        if codec is None or codec.codestream is None or physical is None:
            raise G8EV3Error("E3 reconstruction lacks authenticated codestream")
        key = PhysicalCacheKey(
            source_bytes_sha256=physical["source_bytes_sha256"], canonical_pixels_sha256=physical["canonical_pixels_sha256"],
            canonical_shape=tuple(physical["canonical_shape"]), payload_budget_bytes=physical["payload_budget_bytes"],
            encode_axis_px=physical["encode_axis_px"], codec_configuration_hash=physical["codec_configuration_hash"],
            codec_runtime_identity=physical["codec_runtime_identity"],
        )
        cache = v2.PhysicalReconstructionCacheV2(runtime_root, lambda _: None)
        identity = cache._identity(key, codec.codestream, tuple(value["canonical_shape"]))
        path = Path(runtime_root) / "reconstruction" / f"{reconstruction_ref['object_id']}.json"
        if not path.is_file():
            raise G8EV3Error("E3 reconstruction reference is absent")
        try:
            reconstruction = cache._load(path, identity)
        except v2.FatalExecutionError as exc:
            raise G8EV3Error(f"E3 reconstruction authentication failed: {exc}") from exc
        if reconstruction.object_id != reconstruction_ref["object_id"] or reconstruction.status != reconstruction_ref["status"]:
            raise G8EV3Error("E3 reconstruction reference differs")
        if value["outcome"] == v2.OUTCOME_DELIVERED and reconstruction.pixels is None:
            raise G8EV3Error("E3 delivered record lacks reconstruction pixels")
        if value["outcome"] == v2.OUTCOME_DECODE_FAILURE and reconstruction.status != v2.OUTCOME_DECODE_FAILURE:
            raise G8EV3Error("E3 decode-failure record disagrees with reconstruction")
        summary["reconstruction"] = 1
    observation_ref = value["classifier_observation"]
    if observation_ref is not None:
        if reconstruction is None or reconstruction.pixels is None:
            raise G8EV3Error("E3 observation lacks authenticated reconstruction")
        path = Path(runtime_root) / "observation" / f"{observation_ref['object_id']}.json"
        if not path.is_file():
            raise G8EV3Error("E3 classifier observation reference is absent")
        observation = _load_observation(
            path,
            reconstruction_id=reconstruction.object_id,
            reconstruction_sha256=sha256_bytes(reconstruction.pixels.tobytes()),
            contract=contract,
        )
        if observation["object_id"] != observation_ref["object_id"] or observation["predicted_label"] != observation_ref["predicted_label"]:
            raise G8EV3Error("E3 record/observation prediction differs")
        if value["correct_count"] != int(observation["predicted_label"] == value["label"]):
            raise G8EV3Error("E3 delivered correct count differs from observation")
        summary["observation"] = 1
    elif value["outcome"] == v2.OUTCOME_DELIVERED:
        raise G8EV3Error("E3 delivered record lacks classifier observation")
    if value["outage_applied"]:
        policy = value["outage_prediction"]
        frozen = contract["outage_policy"]
        if policy.get("selected_class") != frozen["selected_class"] or policy.get("policy_sha256") != frozen["sha256"] or policy.get("denominator") != 1:
            raise G8EV3Error("E3 outage policy binding differs")
        if value["correct_count"] != int(value["label"] == frozen["selected_class"]):
            raise G8EV3Error("E3 outage binary count differs")
        summary["outage"] = 1
    return summary


def build_e3_artifact(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    runtime_root: Path,
    contract: Mapping[str, Any],
    sample_labels: Mapping[str, int] | None = None,
    production: bool = True,
    authenticate_caches: bool = True,
    instrumentation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """One O(N) exact-set and cache-authentication pass."""

    expected, expected_by_id, structural_by_id = _expected_indexes(authority, sample_ids)
    labels = dict(sample_labels or {})
    if set(labels) != set(sample_ids):
        raise G8EV3Error("E3 requires the exact frozen sample-to-class mapping")
    counters = instrumentation if instrumentation is not None else defaultdict(int)
    records_dir = Path(runtime_root) / "records"
    if not records_dir.is_dir():
        raise G8EV3Error("E3 runtime records directory is absent")
    by_id: dict[str, tuple[MeasurementRecordV3, bytes]] = {}
    duplicate = extra = foreign = 0
    for path in records_dir.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise G8EV3Error("E3 records directory contains a foreign object")
        value, raw = _rendered_object(path, "v3 E3 record")
        record = MeasurementRecordV3.from_mapping(value)
        counters["records_parsed"] = counters.get("records_parsed", 0) + 1
        work_id = str(record.value["work_unit_id"])
        if work_id in by_id:
            duplicate += 1
            raise G8EV3Error("E3 duplicate work-unit record")
        counters["expected_id_lookup_operations"] = counters.get("expected_id_lookup_operations", 0) + 1
        unit = expected_by_id.get(work_id)
        if unit is None:
            extra += 1
            raise G8EV3Error("E3 extra/foreign work-unit substitution")
        if path.name != f"{work_id}.json":
            foreign += 1
            raise G8EV3Error("E3 record pathname is foreign")
        by_id[work_id] = (record, raw)
    missing = len(expected) - len(by_id)
    if len(by_id) != len(expected):
        raise G8EV3Error(f"E3 exact-set count differs: missing={max(missing, 0)}, extra={max(-missing, 0)}")
    ordered_record_ids: list[str] = []
    ordered_record_sha256s: list[str] = []
    cache_summary = defaultdict(int)
    for ordinal, unit in enumerate(expected):
        counters["expected_id_lookup_operations"] = counters.get("expected_id_lookup_operations", 0) + 1
        record, raw = by_id[unit["work_unit_id"]]
        value = record.value
        if value["authority_ordinal"] != ordinal or value["measurement_identity_id"] != unit["measurement_identity_id"] or value["logical_candidate_ids"] != unit["logical_candidate_ids"] or value["stable_sample_id"] != unit["stable_sample_id"]:
            raise G8EV3Error("E3 ordinal/order/candidate identity differs")
        if value["label"] != labels[unit["stable_sample_id"]]:
            raise G8EV3Error("E3 record class label differs from the frozen validation manifest")
        if production:
            from data.identity import stable_sample_id_width

            width = stable_sample_id_width()
            if value["source_bytes_sha256"][:width] != value["stable_sample_id"]:
                raise G8EV3Error("E3 source-byte digest differs from the stable validation identity")
        counters["structural_lookup_operations"] = counters.get("structural_lookup_operations", 0) + 1
        structural = structural_by_id.get(unit["measurement_identity_id"])
        if structural is None or value["structural_identity"] != structural:
            raise G8EV3Error("E3 structural identity differs")
        if value["campaign_id"] != contract["campaign_id"] or value["contract_id"] != contract["contract_id"] or value["measurement_authority_id"] != authority["authority_id"]:
            raise G8EV3Error("E3 campaign/contract/authority binding differs")
        if value["source_commit"] != contract["source_manifest"]["source_commit"] or value["profile_id"] != contract["execution_profile"]["profile_id"]:
            raise G8EV3Error("E3 source/profile binding differs")
        if value["g8_c_linkage_digest"] != sha256_bytes(canonical_json(contract["direct_upstream_bindings"])):
            raise G8EV3Error("E3 upstream linkage differs")
        if production and (value["record_labels"] or value["scientific_evidence"] is not True or value["merge_eligible"] is not True):
            raise G8EV3Error("E3 production merge contains fixture evidence")
        if authenticate_caches:
            summary = _authenticate_record_caches(runtime_root=Path(runtime_root), record=record, structural=structural, contract=contract)
            for key, amount in summary.items():
                cache_summary[key] += amount
                counters["cache_auth_operations"] = counters.get("cache_auth_operations", 0) + amount
        ordered_record_ids.append(value["record_id"])
        ordered_record_sha256s.append(sha256_bytes(raw))
    set_digest = sha256_bytes(canonical_json(sorted(by_id)))
    ordered_work_digest = sha256_bytes(canonical_json([unit["work_unit_id"] for unit in expected]))
    record_set_digest = sha256_bytes(canonical_json(sorted(ordered_record_sha256s)))
    rolling_digest = sha256_bytes(canonical_json(ordered_record_sha256s))
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_e3_exact_set_closure",
        "status": "E3_COMPLETE",
        "production": production,
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "profile_id": contract["execution_profile"]["profile_id"],
        "measurement_authority_id": authority["authority_id"],
        "authority_sha256": contract["authority"]["sha256"],
        "validation_data_identity_id": contract["scientific_data_identity"]["id"],
        "validation_manifest_sha256": contract["scientific_data_identity"]["manifest_sha256"],
        "required_work_unit_count": len(expected),
        "observed_work_unit_count": len(by_id),
        "ordered_work_unit_sha256": ordered_work_digest,
        "work_unit_set_sha256": set_digest,
        "ordered_record_sha256": rolling_digest,
        "record_set_sha256": record_set_digest,
        "ordered_record_id_sha256": sha256_bytes(canonical_json(ordered_record_ids)),
        "missing_count": 0,
        "duplicate_count": duplicate,
        "extra_count": extra,
        "foreign_count": foreign,
        "verification_summary": {
            "codec_cache_objects": cache_summary["codec"],
            "reconstruction_objects": cache_summary["reconstruction"],
            "classifier_observations": cache_summary["observation"],
            "physical_keys": cache_summary["physical_key"],
            "outage_rows": cache_summary["outage"],
            "cache_authentication_complete": authenticate_caches,
        },
        "runtime_records_path": _relative(Path(runtime_root) / "records"),
        "complexity": "O(N): one directory scan, indexed membership, one authority-order scan",
        "record_labels": [] if production else ["NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"],
    }
    body["e3_id"] = _id(V3_E3_PREFIX, body)
    body["artifact_content_sha256"] = sha256_bytes(canonical_json(body))
    return body


def publish_e3_artifact(**kwargs: Any) -> tuple[dict[str, Any], Path, str]:
    value = build_e3_artifact(**kwargs)
    output = Path(kwargs["runtime_root"]) / "e3_exact_set_closure.json"
    raw = rendered_json(value)
    _atomic_publish(output, raw)
    return value, output, sha256_bytes(raw)


def verify_e3_artifact(path: Path, *, contract: Mapping[str, Any], expected_sha256: str | None = None) -> dict[str, Any]:
    value, raw = _rendered_object(Path(path), "v3 E3 artifact")
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise G8EV3Error("E3 artifact SHA-256 differs")
    if value.get("schema_version") != V3_SCHEMA_VERSION or value.get("artifact_role") != "g8_e_v3_e3_exact_set_closure" or value.get("status") != "E3_COMPLETE":
        raise G8EV3Error("E3 artifact schema/status differs")
    body_without_digest = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    if value.get("artifact_content_sha256") != sha256_bytes(canonical_json(body_without_digest)):
        raise G8EV3Error("E3 artifact content digest differs")
    body_without_id = {key: child for key, child in body_without_digest.items() if key != "e3_id"}
    if value.get("e3_id") != _id(V3_E3_PREFIX, body_without_id):
        raise G8EV3Error("E3 artifact ID differs")
    if (
        value.get("campaign_id") != contract["campaign_id"]
        or value.get("contract_id") != contract["contract_id"]
        or value.get("source_manifest_id") != contract["source_manifest"]["id"]
        or value.get("source_manifest_sha256") != contract["source_manifest"]["sha256"]
        or value.get("profile_id") != contract["execution_profile"]["profile_id"]
        or value.get("measurement_authority_id") != contract.get("authority", {}).get("authority_id", value.get("measurement_authority_id"))
        or value.get("authority_sha256") != contract["authority"]["sha256"]
        or value.get("validation_data_identity_id") != contract["scientific_data_identity"]["id"]
        or value.get("validation_manifest_sha256") != contract["scientific_data_identity"]["manifest_sha256"]
    ):
        raise G8EV3Error("E3 artifact campaign/source/data binding differs")
    if any(value.get(key) != 0 for key in ("missing_count", "duplicate_count", "extra_count", "foreign_count")) or value.get("required_work_unit_count") != value.get("observed_work_unit_count"):
        raise G8EV3Error("E3 artifact is not an exact complete set")
    if value.get("production") is True:
        transaction = contract.get("transaction", {})
        if (
            value.get("required_work_unit_count") != transaction.get("production_total_required")
            or value.get("ordered_work_unit_sha256") != transaction.get("production_authority_order_sha256")
            or value.get("record_labels") != []
            or value.get("verification_summary", {}).get("cache_authentication_complete") is not True
        ):
            raise G8EV3Error("production E3 closure differs from the frozen authority/cache contract")
    elif set(value.get("record_labels", ())) != {
        "NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION",
    }:
        raise G8EV3Error("fixture E3 closure lacks all merge-ineligible labels")
    return value


def build_e4_artifact(
    *,
    authority: Mapping[str, Any],
    sample_ids: Sequence[str],
    runtime_root: Path,
    contract: Mapping[str, Any],
    e3_path: Path,
    e3_sha256: str,
    production: bool = True,
    instrumentation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Authenticate E3 once, then aggregate approved records in one O(N) pass."""

    counters = instrumentation if instrumentation is not None else defaultdict(int)
    e3 = verify_e3_artifact(e3_path, contract=contract, expected_sha256=e3_sha256)
    if e3.get("production") is not production:
        raise G8EV3Error("E4 production/fixture phase differs from the frozen E3 artifact")
    counters["e3_artifact_verifications"] = counters.get("e3_artifact_verifications", 0) + 1
    expected, _, structural_by_id = _expected_indexes(authority, sample_ids)
    if e3["required_work_unit_count"] != len(expected) or e3["ordered_work_unit_sha256"] != sha256_bytes(canonical_json([unit["work_unit_id"] for unit in expected])):
        raise G8EV3Error("E4 authority differs from the exact E3 closure")
    by_identity: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "delivered_count": 0,
        "codec_infeasibility_count": 0,
        "decode_failure_count": 0,
        "structural_infeasibility_count": 0,
        "correct_count": 0,
        "total_count": 0,
        "source_record_ids": [],
        "source_record_sha256s": [],
    })
    ordered_record_ids: list[str] = []
    ordered_record_sha256s: list[str] = []
    for unit in expected:
        path = Path(runtime_root) / "records" / f"{unit['work_unit_id']}.json"
        value, raw = _rendered_object(path, "E4 E3-approved record")
        record = MeasurementRecordV3.from_mapping(value)
        counters["record_traversals"] = counters.get("record_traversals", 0) + 1
        if record.value["work_unit_id"] != unit["work_unit_id"] or record.value["measurement_identity_id"] != unit["measurement_identity_id"] or record.value["stable_sample_id"] != unit["stable_sample_id"]:
            raise G8EV3Error("E4 record no longer matches the E3-approved authority order")
        target = by_identity[unit["measurement_identity_id"]]
        target[f"{record.value['outcome']}_count"] += 1
        target["correct_count"] += record.value["correct_count"]
        target["total_count"] += record.value["total_count"]
        target["source_record_ids"].append(record.value["record_id"])
        digest = sha256_bytes(raw)
        target["source_record_sha256s"].append(digest)
        ordered_record_ids.append(record.value["record_id"])
        ordered_record_sha256s.append(digest)
        counters["object_aggregation_operations"] = counters.get("object_aggregation_operations", 0) + 1
    if sha256_bytes(canonical_json(ordered_record_sha256s)) != e3["ordered_record_sha256"] or sha256_bytes(canonical_json(sorted(ordered_record_sha256s))) != e3["record_set_sha256"] or sha256_bytes(canonical_json(ordered_record_ids)) != e3["ordered_record_id_sha256"]:
        raise G8EV3Error("E4 records mutated after the frozen E3 closure")
    objects: list[dict[str, Any]] = []
    for structural_id in sorted(structural_by_id):
        structural = structural_by_id[structural_id]
        counts = by_identity.get(structural_id)
        if counts is None or counts["total_count"] != len(sample_ids):
            raise G8EV3Error("E4 structural denominator differs from validation count")
        if structural.get("structurally_legal", True) is not True:
            objects.append({
                "measurement_identity_id": structural_id,
                "status": "ineligible",
                "reason": "structurally_impossible_packet_configuration",
                "correct_count": None,
                "total_count": None,
                "source_record_ids": counts["source_record_ids"],
                "source_record_sha256s": counts["source_record_sha256s"],
            })
        else:
            objects.append({
                "measurement_identity_id": structural_id,
                "status": "eligible",
                "delivered_count": counts["delivered_count"],
                "codec_infeasibility_count": counts["codec_infeasibility_count"],
                "decode_failure_count": counts["decode_failure_count"],
                "correct_count": counts["correct_count"],
                "total_count": counts["total_count"],
                "clean_accuracy_counts": {"correct_count": counts["correct_count"], "total_count": counts["total_count"]},
                "outage_accuracy_binding": _copy(contract["outage_policy"]),
                "source_record_ids": counts["source_record_ids"],
                "source_record_sha256s": counts["source_record_sha256s"],
            })
    body: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_e4_count_derived_objects",
        "status": "E4_COMPLETE",
        "production": production,
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "measurement_authority_id": authority["authority_id"],
        "e3_id": e3["e3_id"],
        "e3_sha256": e3_sha256,
        "e3_ordered_record_sha256": e3["ordered_record_sha256"],
        "record_traversal_count": len(expected),
        "object_count": len(objects),
        "objects": objects,
        "validation_denominator": len(sample_ids),
        "outage_accuracy": _copy(contract["outage_policy"]),
        "br4_formula": "P(TB success) * acc_clean + (1 - P(TB success)) * acc_outage",
        "complete_e3_ingest_repeated": False,
        "complexity": "O(N): O(1) E3 authentication plus one authority-ordered record traversal",
        "record_labels": [] if production else ["NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"],
    }
    body["e4_id"] = _id(V3_E4_PREFIX, body)
    body["artifact_content_sha256"] = sha256_bytes(canonical_json(body))
    return body


def publish_e4_artifact(**kwargs: Any) -> tuple[dict[str, Any], Path, str]:
    value = build_e4_artifact(**kwargs)
    output = Path(kwargs["runtime_root"]) / "e4_count_derived.json"
    raw = rendered_json(value)
    _atomic_publish(output, raw)
    return value, output, sha256_bytes(raw)


def verify_e4_artifact(path: Path, *, contract: Mapping[str, Any], e3_path: Path, e3_sha256: str, expected_sha256: str | None = None) -> dict[str, Any]:
    e3 = verify_e3_artifact(e3_path, contract=contract, expected_sha256=e3_sha256)
    value, raw = _rendered_object(path, "v3 E4 artifact")
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise G8EV3Error("E4 artifact SHA-256 differs")
    if value.get("schema_version") != V3_SCHEMA_VERSION or value.get("artifact_role") != "g8_e_v3_e4_count_derived_objects" or value.get("status") != "E4_COMPLETE":
        raise G8EV3Error("E4 artifact schema/status differs")
    body_without_digest = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    if value.get("artifact_content_sha256") != sha256_bytes(canonical_json(body_without_digest)):
        raise G8EV3Error("E4 content digest differs")
    body_without_id = {key: child for key, child in body_without_digest.items() if key != "e4_id"}
    if value.get("e4_id") != _id(V3_E4_PREFIX, body_without_id):
        raise G8EV3Error("E4 ID differs")
    if value.get("campaign_id") != contract["campaign_id"] or value.get("contract_id") != contract["contract_id"] or value.get("e3_id") != e3["e3_id"] or value.get("e3_sha256") != e3_sha256:
        raise G8EV3Error("E4 campaign/E3 binding differs")
    if (
        value.get("production") is not e3.get("production")
        or value.get("e3_ordered_record_sha256") != e3.get("ordered_record_sha256")
        or value.get("record_traversal_count") != e3.get("observed_work_unit_count")
        or value.get("object_count") != len(value.get("objects", ()))
        or value.get("complete_e3_ingest_repeated") is not False
    ):
        raise G8EV3Error("E4 exact E3/count binding differs")
    if value.get("production") is True and value.get("validation_denominator") != contract["scientific_data_identity"]["validation_count"]:
        raise G8EV3Error("production E4 validation denominator differs")
    for obj in value.get("objects", ()):
        if obj.get("status") == "eligible" and obj.get("total_count") != value.get("validation_denominator"):
            raise G8EV3Error("E4 eligible object denominator differs")
        if obj.get("status") == "ineligible" and (obj.get("correct_count") is not None or obj.get("total_count") is not None):
            raise G8EV3Error("E4 ineligible object contains scientific counts")
    return value


def verify_v3_e3_complete(*, e3_path: Path = V3_E3_PATH, e3_sha256: str | None = None, **kwargs: Any) -> dict[str, Any]:
    runtime_root = Path(e3_path).parent
    complete = verify_v3_e2_complete(runtime_root=runtime_root, **kwargs)
    value = verify_e3_artifact(e3_path, contract=complete["contract"], expected_sha256=e3_sha256)
    return {**complete, "e3": value, "e3_sha256": sha256_file(e3_path), "phase": "E3_COMPLETE"}


def verify_v3_e4_complete(*, e4_path: Path = V3_E4_PATH, e3_path: Path = V3_E3_PATH, e3_sha256: str | None = None, **kwargs: Any) -> dict[str, Any]:
    e3_complete = verify_v3_e3_complete(e3_path=e3_path, e3_sha256=e3_sha256, **kwargs)
    bound_e3_sha = e3_sha256 or sha256_file(e3_path)
    value = verify_e4_artifact(e4_path, contract=e3_complete["contract"], e3_path=e3_path, e3_sha256=bound_e3_sha)
    return {**e3_complete, "e4": value, "e4_sha256": sha256_file(e4_path), "phase": "E4_COMPLETE"}


def reject_superseded_campaign(campaign_id: str) -> None:
    old = {
        v2.ORIGINAL_CAMPAIGN_ID,
        v2.FIRST_CORRECTED_CAMPAIGN_ID,
        _rendered_object(v2.V2_CONTRACT_PATH, "v2 historical contract")[0]["campaign_id"],
    }
    if campaign_id in old:
        raise G8EV3Error("superseded-before-data E1 campaign cannot execute as current")
