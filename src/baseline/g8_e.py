"""Pre-data G8_E contract primitives.

This module contains only phase-opening and contract-validation code.  It does
not load a dataset, decode an image, invoke a classifier, construct a
``G8Authorization`` or start a scientific campaign.  The scientific runner
belongs to the later E2 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config.params import REPO_ROOT, get


class G8EContractError(ValueError):
    """A fail-closed G8_E opening-contract error."""


E0_PATH = REPO_ROOT / "results/baseline/g8_e/e0_open.json"
E0_SCHEMA_VERSION = 1
E0_ARTIFACT_ROLE = "g8_e_pre_data_opening"
E0_ARTIFACT_PREFIX = "g8e0-"

E1_SCHEMA_VERSION = 1
E1_ARTIFACT_ROLE = "g8_e_pre_data_validation_contract"
E1_AUTHORITY_ROLE = "g8_e_complete_logical_candidate_authority"
E1_SOURCE_MANIFEST_ROLE = "g8_e_execution_source_manifest"
E1_CORPUS_SPEC_ROLE = "g8_e_future_training_corpus_specification"
E1_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"
E1_SOURCE_MANIFEST_PATH = REPO_ROOT / "results/baseline/g8_e/execution_source_manifest.json"
E1_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8_e/measurement_contract.json"
E1_CORPUS_SPEC_PATH = REPO_ROOT / "results/baseline/g8_e/corpus_spec.json"
E1_AUTHORITY_PREFIX = "g8eauthority-"
E1_SOURCE_MANIFEST_PREFIX = "g8esource-"
E1_CONTRACT_PREFIX = "g8econtract-"
E1_CORPUS_SPEC_PREFIX = "g8ecorpusspec-"
E1_RECORD_PREFIX = "g8erecord-"
E1_PROFILE_ID = "local_4060_cu130"
E1_DEVICE = "cuda:0"
E1_INITIAL_DATASET = "imagenette160"
E1_FALLBACK_DATASET = "stl10"
E1_SMOKE_DATASET = "cifar10"
E1_VALIDATION_SPLIT = "val"
E1_ALLOWED_OUTCOMES = frozenset(
    {"structural_infeasibility", "codec_infeasibility", "decode_failure", "delivered"}
)
E1_CANDIDATE_FIELDS = (
    "candidate_id",
    "composition_candidate_identity",
    "dataset",
    "dataset_role",
    "source_codec",
    "ratio",
    "encode_axis_px",
    "modulation",
    "ldpc_rate",
    "snr_db",
    "packet_config_id",
)
E1_SOURCE_BINDING_PATHS: tuple[tuple[str, str], ...] = (
    ("src/baseline/g8_e.py", "g8_e_runtime_contract_source"),
    ("tools/run_g8_e.py", "g8_e_runner_orchestrator"),
    ("tools/gen_g8_e_e1.py", "g8_e_contract_generator"),
    ("tools/verify_g8_e_e1.py", "g8_e_contract_verifier"),
    ("src/baseline/g8_campaign.py", "g8_candidate_authority_source"),
    ("src/baseline/g8_d.py", "g8_d_identity_cache_source"),
    ("src/baseline/j2k.py", "jpeg2000_implementation"),
    ("src/baseline/classical/pipeline.py", "classical_pipeline"),
    ("src/baseline/classical/channel_transport.py", "classical_channel_transport"),
    ("src/baseline/classical/records.py", "br11_record_accounting"),
    ("src/baseline/classical/composition.py", "analytic_composition_and_selection"),
    ("src/baseline/classical/outage.py", "measured_outage_policy"),
    ("src/baseline/ldpc/transport.py", "ldpc_transport_adapter"),
    ("src/baseline/ldpc/segmentation.py", "ldpc_segmentation"),
    ("src/baseline/ldpc/rate_matching.py", "ldpc_rate_matching"),
    ("src/baseline/ldpc/modulation.py", "ldpc_modulation"),
    ("src/data/manifests.py", "dataset_manifest_reader"),
    ("src/data/adapters.py", "dataset_adapter"),
    ("src/data/identity.py", "stable_sample_identity"),
    ("src/data/provenance.py", "dataset_archive_provenance"),
    ("src/data/registry.py", "dataset_registry"),
    ("src/data/preprocessing.py", "canonical_preprocessing"),
    ("src/data/classifier.py", "classifier_data_path"),
    ("src/data/test_access.py", "sealed_test_access_boundary"),
    ("src/models/frozen_reference_classifier.py", "frozen_classifier_model"),
    ("src/config/params.py", "parameter_loader"),
    ("src/config/execution_profiles.py", "execution_profile_authentication"),
    ("src/config/run_config.py", "run_configuration_identity"),
    ("src/env.py", "runtime_environment"),
    ("src/baseline/g8_pascal_merge.py", "g8_c_frozen_successor_loader_wrapper"),
    ("src/baseline/g8_pascal_portable.py", "g8_c_portable_successor_verifier"),
    ("results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json", "g8_c_portable_scientific_runtime_manifest"),
    ("results/baseline/g8_pascal_successor/portable_verification_provenance.json", "g8_c_portable_verification_provenance"),
    ("results/baseline/g8_pascal_successor/successor_bler_merge_report.json", "g8_c_frozen_successor_merge"),
    ("results/baseline/g8_pascal_successor/successor_bler_table.json", "g8_c_frozen_successor_table"),
    ("results/baseline/g8_pascal_successor/successor_closeout_provenance.json", "g8_c_historical_c6_closeout"),
    ("results/baseline/g8_d/measurement_contract.json", "g8_d_current_measurement_contract"),
    ("results/baseline/g8_d/d7_handoff.json", "g8_d_current_d7_handoff"),
    ("results/baseline/g8_d/portable_rebind_provenance.json", "g8_d_portable_rebind_provenance"),
    ("tools/verify_g8_pascal_portable.py", "g8_c_portable_verifier_tool"),
    ("tools/verify_g8_pascal_successor.py", "g8_c_successor_verifier_tool"),
    ("tools/verify_g8_pascal_closeout.py", "g8_c_closeout_verifier_tool"),
    ("results/baseline/g8/required_bler_identities.json", "g8_bler_authority"),
    ("results/baseline/g2/g2_adjudication.json", "g2_bler_reference_adjudication"),
    ("results/baseline/w4/integration_adjudication.json", "w4_selection_adjudication"),
    ("results/baseline/w4/outage_policy.json", "w4_outage_record"),
    ("results/baseline/w4/overhead_table.json", "w4_br11_overhead_table"),
    ("results/baseline/w4/execution_source_manifest.json", "w4_execution_source_manifest"),
    ("results/reference_classifier/g1_adjudication.json", "g1_classifier_adjudication"),
    ("configs/reference-classifier-clean.yaml", "g1_classifier_config"),
    ("results/reference_classifier/resolved_config.json", "g1_resolved_config"),
    ("results/reference_classifier/best_checkpoint.json", "g1_checkpoint_metadata"),
    ("results/reference_classifier/validation_summary.json", "g1_validation_summary"),
    ("spec/SPEC.md", "normative_specification"),
    ("spec/params.generated.yaml", "generated_parameters"),
    ("requirements.lock", "runtime_dependency_lock"),
)

UPSTREAM_BINDING_PATHS: tuple[tuple[str, str], ...] = (
    (
        "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json",
        "g8_c_portable_scientific_runtime_manifest",
    ),
    (
        "results/baseline/g8_pascal_successor/portable_verification_provenance.json",
        "g8_c_portable_verification_provenance",
    ),
    ("src/baseline/g8_pascal_portable.py", "g8_c_portable_verifier_source"),
    ("src/baseline/g8_pascal_merge.py", "g8_c_legacy_loader_wrapper_source"),
    (
        "results/baseline/g8_pascal_successor/successor_bler_merge_report.json",
        "g8_c_frozen_successor_merge",
    ),
    (
        "results/baseline/g8_pascal_successor/successor_bler_table.json",
        "g8_c_frozen_successor_table",
    ),
    (
        "results/baseline/g8_pascal_successor/successor_closeout_provenance.json",
        "g8_c_historical_c6_closeout",
    ),
    ("results/baseline/g8_d/measurement_contract.json", "g8_d_current_contract"),
    ("results/baseline/g8_d/d7_handoff.json", "g8_d_current_d7_handoff"),
    (
        "results/baseline/g8_d/portable_rebind_provenance.json",
        "g8_d_portable_rebind_provenance",
    ),
    ("src/baseline/g8_d.py", "g8_d_identity_cache_source"),
)


def canonical_json(value: Any) -> bytes:
    """Return the repository's compact, deterministic JSON identity bytes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered_json(value: Any) -> bytes:
    """Return canonical human-readable JSON bytes used by committed artifacts."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8EContractError(message)


def _digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} is not a SHA-256 digest")  # literal-ok: SHA-256 hex width
    try:
        int(value, 16)  # literal-ok: hexadecimal digest radix
    except ValueError:
        raise G8EContractError(f"{label} is not a hexadecimal SHA-256 digest") from None
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _binding(path_text: str, role: str) -> dict[str, Any]:
    path = REPO_ROOT / path_text
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise G8EContractError(f"cannot bind upstream path {path_text}: {exc}") from exc
    return {"path": path_text, "role": role, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,  # literal-ok: bounded provenance command
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise G8EContractError(f"cannot resolve opening Git commit: {exc}") from exc
    _require(len(value) == 40, "opening Git commit is not a full object ID")  # literal-ok: full Git object ID width
    return value


def _portable_epoch(provenance: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    portable = provenance.get("portable_evidence")
    classification = provenance.get("classification")
    _require(isinstance(portable, Mapping), "portable provenance has no portable evidence")
    _require(isinstance(classification, Mapping), "portable provenance has no defect classification")
    _require(provenance.get("epoch") == "g8-c-portable-scientific-runtime-v1", "portable verification epoch differs")
    _require(provenance.get("provenance_id") == "g8pportableprov-" + sha256_bytes(
        canonical_json({key: value for key, value in provenance.items() if key != "provenance_id"})
    ), "portable provenance ID does not reproduce")
    _require(manifest.get("manifest_id") == portable.get("manifest_id"), "portable manifest/provenance ID differs")
    _require(manifest.get("scientific_runtime_sha256") == portable.get("scientific_runtime_sha256"), "portable runtime digest differs")
    return {
        "epoch": provenance["epoch"],
        "classification": dict(classification),
        "repair_commit": provenance["repair_commit"],
        "repair_source_digest": provenance["repair_source_digest"],
        "portable_manifest_id": portable["manifest_id"],
        "portable_manifest_sha256": sha256_file(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
        ),
        "scientific_runtime_sha256": portable["scientific_runtime_sha256"],
        "exact_authority_count": portable["exact_authority_count"],
        "trials_per_identity": portable["trials_per_identity"],
        "legacy_runtime_tree_sha256": provenance["historical_g8_c"]["legacy_runtime_tree_sha256"],
        "legacy_tree_digest_is_historical_only": True,
    }


def build_e0_opening(*, opening_commit: str | None = None) -> dict[str, Any]:
    """Build the deterministic E0 opening witness from current upstream bytes."""

    from baseline.g8_pascal_portable import verify_portable_successor

    # This is a strict upstream read/verification.  It does not touch any
    # validation image or invoke a model-facing data path.
    portable_result = verify_portable_successor()
    _require(portable_result["status"] == "PASS", "portable G8_C verification did not pass")
    manifest = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
    )
    provenance = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"
    )
    merge = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"
    )
    table = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    )
    closeout = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"
    )
    d_contract = _read_object(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json")
    d_handoff = _read_object(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json")
    d_rebind = _read_object(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json")

    _require(merge.get("report_id") == "g8pmerge-" + sha256_bytes(canonical_json({
        key: value for key, value in merge.items() if key != "report_id"
    })), "frozen successor merge ID does not reproduce")
    _require(table.get("table_id") == "g8pblertable-" + sha256_bytes(canonical_json({
        key: value for key, value in table.items() if key != "table_id"
    })), "frozen successor table ID does not reproduce")
    _require(closeout.get("closure_id") == "g8pcloseout-" + sha256_bytes(canonical_json({
        key: value for key, value in closeout.items() if key != "closure_id"
    })), "historical C6 ID does not reproduce")
    _require(d_handoff.get("contract_id") == d_contract.get("contract_id"), "G8_D handoff is not bound to current contract")
    _require(d_contract.get("next_gate") == "G8_E/E0", "G8_D does not release E0")
    _require(d_handoff.get("g8_e_released") is True, "G8_D handoff does not release E0")
    _require(d_handoff.get("full_campaign_not_started") is True, "G8_D full campaign is not unopened")
    _require(d_rebind.get("current_contract", {}).get("contract_id") == d_contract.get("contract_id"), "G8_D rebind does not name current contract")
    _require(d_rebind.get("current_handoff", {}).get("artifact_id") == d_handoff.get("artifact_id"), "G8_D rebind does not name current handoff")

    upstream = sorted(
        (_binding(path, role) for path, role in UPSTREAM_BINDING_PATHS),
        key=lambda item: item["path"],
    )
    body: dict[str, Any] = {
        "schema_version": E0_SCHEMA_VERSION,
        "artifact_role": E0_ARTIFACT_ROLE,
        "phase": "G8_E",
        "checkpoint": "E0",
        "status": "OPEN",
        "opening_commit": opening_commit or _git_head(),
        "upstream_verification": {
            "g8_c_successor_verifier": "PASS",
            "g8_c_portable_verifier": "PASS",
            "g8_c_closeout_compatibility": "PASS",
            "exact_required_identities": 3213,
            "exact_accepted_identities": portable_result["accepted_count"],
            "exact_frozen_points": portable_result["measured_point_count"],
            "trials_per_identity": portable_result["trials_per_point"],
            "retry_history_ordinals": [0, 1],
            "predecessor_table_contribution": "none",
            "protected_counters": {
                "inference": 0,
                "training": 0,
                "validation_decoding": 0,
                "test_access": 0,
            },
        },
        "g8_c": {
            "campaign_id": merge["campaign_id"],
            "merge_report_id": merge["report_id"],
            "merge_report_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"),
            "table_id": table["table_id"],
            "table_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"),
            "historical_c6_id": closeout["closure_id"],
            "historical_c6_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"),
            "portable_manifest_id": manifest["manifest_id"],
            "portable_manifest_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"),
            "portable_provenance_id": provenance["provenance_id"],
            "portable_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"),
            "portable_scientific_runtime_sha256": manifest["scientific_runtime_sha256"],
            "portable_verification_epoch": _portable_epoch(provenance, manifest),
        },
        "g8_d": {
            "contract_id": d_contract["contract_id"],
            "contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
            "handoff_id": d_handoff["artifact_id"],
            "handoff_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"),
            "portable_rebind_provenance_id": d_rebind["provenance_id"],
            "portable_rebind_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json"),
            "d4_scientific_evidence": False,
            "d4_merge_eligible": False,
            "d6_scientific_evidence": False,
            "d6_merge_eligible": False,
        },
        "upstream_bindings": upstream,
        "safety": {
            "g8_e_measurement_coverage": 0,
            "e2_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "training_started": False,
            "fallback_invoked": False,
            "ratio_adjudicated": False,
            "test_access": 0,
            "inference": 0,
            "training": 0,
            "validation_decoding": 0,
        },
        "declarations": {
            "g8_c_scientific_state_frozen": True,
            "portable_verification_pass": True,
            "g8_d_green": True,
            "g8_e_measurement_coverage": 0,
            "e2_not_started": True,
            "pass_one_not_started": True,
            "training_forbidden": True,
            "test_forbidden": True,
            "validation_image_decoding_required_to_open": False,
        },
    }
    body["artifact_id"] = E0_ARTIFACT_PREFIX + sha256_bytes(canonical_json(body))
    return body


def _verify_am87_g8d_source_compatibility(binding: Mapping[str, Any]) -> None:
    am87_path = REPO_ROOT / "results/baseline/g8_f/am87_g8e_source_compatibility.json"
    am88_path = REPO_ROOT / "results/baseline/g8_f/am88_g8e_source_compatibility.json"
    try:
        am87_raw = am87_path.read_bytes()
        am87 = json.loads(am87_raw)
        compatibility = json.loads(am88_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot load AM-87/AM-88 E0 source compatibility: {exc}") from None
    am87_body = {key: child for key, child in am87.items() if key != "compatibility_id"}
    body = {key: child for key, child in compatibility.items() if key != "compatibility_id"}
    _require(
        am87.get("compatibility_id") == "g8esourcecompat-" + sha256_bytes(canonical_json(am87_body))
        and compatibility.get("compatibility_id") == "g8esourcecompat-" + sha256_bytes(canonical_json(body)),
        "AM-87/AM-88 E0 source-compatibility ID differs",
    )
    _require(
        compatibility.get("amendment") == "AM-88"
        and compatibility.get("timing") == "post_am87_pre_f0_execution_zero"
        and compatibility.get("prior_compatibility") == {
            "path": str(am87_path.relative_to(REPO_ROOT)),
            "compatibility_id": am87["compatibility_id"],
            "sha256": sha256_bytes(am87_raw),
        }
        and compatibility.get("protected_boundary", {}).get("g8_d_changed") is False,
        "AM-88 E0 source-compatibility boundary differs",
    )
    prior_entries = am87.get("entries")
    entries = compatibility.get("entries")
    _require(isinstance(prior_entries, list) and isinstance(entries, list), "AM-87/AM-88 E0 source entries differ")
    prior = [entry for entry in prior_entries if isinstance(entry, Mapping) and entry.get("path") == binding["path"]]
    matches = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("path") == binding["path"]]
    _require(len(prior) == len(matches) == 1, "AM-87/AM-88 E0 G8_D source entry differs")
    entry = matches[0]
    current_path = REPO_ROOT / str(binding["path"])
    _require(
        prior[0].get("archived_bytes") == binding["bytes"]
        and prior[0].get("archived_sha256") == binding["sha256"]
        and entry.get("archived_bytes") == prior[0].get("current_bytes")
        and entry.get("archived_sha256") == prior[0].get("current_sha256")
        and entry.get("current_bytes") == current_path.stat().st_size
        and entry.get("current_sha256") == sha256_file(current_path)
        and entry.get("scientific_execution_reachable") is False,
        "AM-87/AM-88 E0 G8_D source byte chain differs",
    )


def validate_e0_opening(value: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    """Validate E0's exact schema and all current upstream bindings."""

    _require(isinstance(value, Mapping), "E0 artifact is not an object")
    required = {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "opening_commit",
        "upstream_verification", "g8_c", "g8_d", "upstream_bindings", "safety", "declarations", "artifact_id",
    }
    _require(set(value) == required, "E0 artifact schema differs")
    _require(value["schema_version"] == E0_SCHEMA_VERSION, "E0 schema version differs")
    _require((value["artifact_role"], value["phase"], value["checkpoint"], value["status"]) == (E0_ARTIFACT_ROLE, "G8_E", "E0", "OPEN"), "E0 header differs")
    _require(isinstance(value["opening_commit"], str) and len(value["opening_commit"]) == 40, "E0 opening commit is not a full SHA")  # literal-ok: full Git object ID width
    if expected_commit is not None:
        _require(value["opening_commit"] == expected_commit, "E0 opening commit differs")
    body = dict(value)
    artifact_id = body.pop("artifact_id")
    _require(artifact_id == E0_ARTIFACT_PREFIX + sha256_bytes(canonical_json(body)), "E0 artifact ID differs")

    verification = value["upstream_verification"]
    _require(isinstance(verification, Mapping), "E0 upstream verification is not an object")
    _require(all(verification[field] == "PASS" for field in ("g8_c_successor_verifier", "g8_c_portable_verifier", "g8_c_closeout_compatibility")), "E0 upstream verifier status is not PASS")
    _require((verification["exact_required_identities"], verification["exact_accepted_identities"], verification["exact_frozen_points"], verification["trials_per_identity"]) == (3213, 3213, 3213, 5000), "E0 G8_C coverage differs")  # literal-ok: frozen G8_C authority/trial contract
    _require(verification["retry_history_ordinals"] == [0, 1] and verification["predecessor_table_contribution"] == "none", "E0 retry/predecessor boundary differs")
    _require(verification["protected_counters"] == {"inference": 0, "training": 0, "validation_decoding": 0, "test_access": 0}, "E0 upstream protected counters are nonzero")

    g8c = value["g8_c"]
    _require(g8c["merge_report_id"] == "g8pmerge-2e861c39d8981af0e2d57dc8ded5828b9ed56a1459491e04929b5e9c3418de89", "E0 merge ID differs")
    _require(g8c["table_id"] == "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f", "E0 table ID differs")
    _require(g8c["historical_c6_id"] == "g8pcloseout-8cc5be86e6bbb350ca35c1806686e751f4528ad32aa7083e9a754c4849feba70", "E0 C6 ID differs")
    for field in ("merge_report_sha256", "table_sha256", "historical_c6_sha256", "portable_manifest_sha256", "portable_provenance_sha256", "portable_scientific_runtime_sha256"):
        _digest(g8c[field], f"E0 G8_C {field}")
    epoch = g8c["portable_verification_epoch"]
    _require(epoch["epoch"] == "g8-c-portable-scientific-runtime-v1" and epoch["legacy_tree_digest_is_historical_only"] is True, "E0 portable epoch differs")
    _digest(epoch["scientific_runtime_sha256"], "E0 portable runtime digest")

    g8d = value["g8_d"]
    _require(g8d["contract_id"] == "g8dcontract-c1ebf0b23e0e5725d387f447e633b37f123688d2595695f92e86a1c663db7889", "E0 G8_D contract ID differs")
    _require(g8d["handoff_id"] == "g8dhandoff-31c48fcabe765a0e70bcd7bcfec5f4bd705b88ee3cb0d140ba9aecf67e1dfd4c", "E0 D7 handoff ID differs")
    _require(g8d["d4_scientific_evidence"] is False and g8d["d4_merge_eligible"] is False and g8d["d6_scientific_evidence"] is False and g8d["d6_merge_eligible"] is False, "E0 binds a scientific/mergeable D record")

    bindings = value["upstream_bindings"]
    expected_bindings = sorted((path, role) for path, role in UPSTREAM_BINDING_PATHS)
    _require(isinstance(bindings, list) and [item["path"] for item in bindings] == [path for path, _role in expected_bindings], "E0 upstream path set/order differs")
    for item, (_path, role) in zip(bindings, expected_bindings):
        _require(set(item) == {"path", "role", "bytes", "sha256"}, "E0 upstream binding schema differs")
        _require(item["role"] == role, f"E0 upstream role differs for {item['path']}")
        path = REPO_ROOT / item["path"]
        _require(path.is_file(), f"E0 upstream binding is missing: {item['path']}")
        exact = item["bytes"] == path.stat().st_size and item["sha256"] == sha256_file(path)
        if not exact:
            _require(item["path"] == "src/baseline/g8_d.py", f"E0 upstream bytes changed: {item['path']}")
            _verify_am87_g8d_source_compatibility(item)

    safety = value["safety"]
    _require(safety == {
        "g8_e_measurement_coverage": 0,
        "e2_started": False,
        "pass_one_started": False,
        "pass_two_started": False,
        "training_started": False,
        "fallback_invoked": False,
        "ratio_adjudicated": False,
        "test_access": 0,
        "inference": 0,
        "training": 0,
        "validation_decoding": 0,
    }, "E0 safety state is nonzero")
    declarations = value["declarations"]
    _require(all(declarations[field] is True for field in ("g8_c_scientific_state_frozen", "portable_verification_pass", "g8_d_green", "e2_not_started", "pass_one_not_started", "training_forbidden", "test_forbidden")), "E0 declarations are not closed")
    _require(declarations["g8_e_measurement_coverage"] == 0 and declarations["validation_image_decoding_required_to_open"] is False, "E0 opening boundary differs")
    return dict(value)


def verify_e0_file(path: Path = E0_PATH, *, expected_commit: str | None = None) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read E0 artifact: {exc}") from exc
    _require(raw == rendered_json(value), "E0 artifact is not canonical rendered JSON")
    return validate_e0_opening(value, expected_commit=expected_commit)


def _strict_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    _require(
        set(value) == set(expected),
        f"{label} schema differs: missing={sorted(set(expected) - set(value))}, "
        f"extra={sorted(set(value) - set(expected))}",
    )


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,  # literal-ok: bounded provenance command
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise G8EContractError(f"cannot inspect Git cleanliness: {exc}") from exc
    return bool(result.stdout.strip())


def _git_commit_count(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 40, f"{label} is not a full Git SHA")  # literal-ok: full Git object ID width
    try:
        int(value, 16)  # literal-ok: hexadecimal Git object ID radix
    except ValueError:
        raise G8EContractError(f"{label} is not hexadecimal") from None


def _dataset_manifest_metadata(dataset: str, *, require_extracted: bool) -> dict[str, Any]:
    """Authenticate archive/extraction and parse manifest metadata only.

    This function deliberately never calls an adapter and never opens an image.
    It is the E1 provenance boundary: archive bytes, extraction marker and
    canonical CSV bytes must all authenticate before the contract can freeze.
    """

    from data.manifests import manifest_path, validate_manifest_bytes
    from data.provenance import verify_archive, verify_extracted_dataset

    try:
        archive = verify_extracted_dataset(dataset) if require_extracted else verify_archive(dataset)
        manifest_file = manifest_path(dataset)
        manifest_bytes = manifest_file.read_bytes()
        rows = validate_manifest_bytes(dataset, manifest_bytes)
    except Exception as exc:
        raise G8EContractError(f"{dataset}: validation asset authentication failed: {exc}") from exc

    config = get(f"datasets.{dataset}")
    validation_rows = tuple(row for row in rows if row.split == E1_VALIDATION_SPLIT)
    train_rows = tuple(row for row in rows if row.split == "train")
    test_rows = tuple(row for row in rows if row.split == "test")
    validation_ids = [row.stable_sample_id for row in validation_rows]
    train_ids = [row.stable_sample_id for row in train_rows]
    test_ids = [row.stable_sample_id for row in test_rows]
    _require(validation_ids == sorted(validation_ids), f"{dataset}: validation IDs are not stable ordered")
    _require(train_ids == sorted(train_ids), f"{dataset}: train IDs are not stable ordered")
    _require(test_ids == sorted(test_ids), f"{dataset}: test IDs are not stable ordered")
    return {
        "dataset": dataset,
        "role": config["role"],
        "dataset_version": str(config["archive_sha256"]),
        "archive": {
            "path": str(archive.path.relative_to(REPO_ROOT)),
            "filename": archive.filename,
            "bytes": archive.byte_length,
            "sha256": archive.sha256,
            "url": archive.url,
        },
        "manifest": {
            "path": str(manifest_file.relative_to(REPO_ROOT)),
            "bytes": len(manifest_bytes),
            "sha256": sha256_bytes(manifest_bytes),
            "split": E1_VALIDATION_SPLIT,
            "validation_count": len(validation_rows),
            "validation_ids": validation_ids,
            "validation_id_set_sha256": sha256_bytes(canonical_json(validation_ids)),
            "train_count": len(train_rows),
            "train_ids": train_ids,
            "train_id_set_sha256": sha256_bytes(canonical_json(train_ids)),
            "test_count": len(test_rows),
            "test_ids": test_ids,
            "test_id_set_sha256": sha256_bytes(canonical_json(test_ids)),
        },
        "class_count": int(config["classes"]),
        "image_size": list(config["image_size"]),
        "loader": config["loader"],
        "source_payload": config["source_payload"],
        "class_index_source": config["class_index_source"],
        "extracted_asset_authenticated": require_extracted,
    }


def _asset_file_binding(path_text: str, role: str) -> dict[str, Any]:
    return _binding(path_text, role)


def _current_g8_c_binding() -> dict[str, Any]:
    merge_path = REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"
    table_path = REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    closeout_path = REPO_ROOT / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"
    merge = _read_object(merge_path)
    table = _read_object(table_path)
    closeout = _read_object(closeout_path)
    return {
        "campaign_id": merge["campaign_id"],
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": sha256_file(merge_path),
        "table_id": table["table_id"],
        "table_sha256": sha256_file(table_path),
        "historical_c6_id": closeout["closure_id"],
        "historical_c6_sha256": sha256_file(closeout_path),
        "portable_manifest_id": _read_object(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
        )["manifest_id"],
        "portable_manifest_sha256": sha256_file(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
        ),
        "portable_provenance_id": _read_object(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"
        )["provenance_id"],
        "portable_provenance_sha256": sha256_file(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"
        ),
        "portable_verification_epoch": "g8-c-portable-scientific-runtime-v1",
        "portable_scientific_runtime_sha256": _read_object(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
        )["scientific_runtime_sha256"],
        "trials_per_point": 5000,  # literal-ok: frozen G8_C trial contract
        "measured_points": 3213,  # literal-ok: frozen G8_C point authority
        "curves": 153,  # literal-ok: frozen G8_C curve authority
        "predecessor_table_contribution": "none",
    }


def _current_classifier_binding(d_contract: Mapping[str, Any]) -> dict[str, Any]:
    binding = d_contract.get("classifier_binding")
    _require(isinstance(binding, Mapping), "G8_D contract has no classifier binding")
    return dict(binding)


def _current_d4_contract_flags(d_contract: Mapping[str, Any]) -> dict[str, bool]:
    identity = d_contract.get("handoff_schema")
    _require(isinstance(identity, Mapping), "G8_D contract has no handoff schema")
    _require(d_contract.get("checkpoint") == "D7", "G8_D contract is not current D7")
    return {
        "d4_scientific_evidence": False,
        "d4_merge_eligible": False,
        "d6_scientific_evidence": False,
        "d6_merge_eligible": False,
    }


def build_e1_candidate_authority() -> dict[str, Any]:
    """Derive the complete ordered BR-4 logical authority from live code.

    The implementation intentionally delegates enumeration to the frozen G-8
    structural preflight and then validates its identity independently.  No
    image, codec, classifier or channel path is entered.
    """

    from baseline.g8_campaign import build_structural_preflight

    preflight = build_structural_preflight()
    candidates = preflight["structural_candidates"]
    _require(isinstance(candidates, list) and candidates, "structural preflight has no candidates")
    ordered = sorted(candidates, key=lambda row: row["candidate_id"])
    _require(candidates == ordered, "structural preflight candidate order changed")
    identities = [canonical_json({key: row[key] for key in E1_CANDIDATE_FIELDS}) for row in candidates]
    _require(len(identities) == len(set(identities)), "candidate authority contains a duplicate identity")
    candidate_ids = [row["candidate_id"] for row in candidates]
    _require(len(candidate_ids) == len(set(candidate_ids)), "candidate authority contains a duplicate ID")
    _require(all(set(row) == set(E1_CANDIDATE_FIELDS) for row in candidates), "candidate authority row schema differs")

    structural_keys = {
        canonical_json(
            {
                key: row[key]
                for key in (
                    "dataset",
                    "dataset_role",
                    "source_codec",
                    "ratio",
                    "encode_axis_px",
                    "modulation",
                    "ldpc_rate",
                )
            }
        )
        for row in candidates
    }
    candidate_digest = sha256_bytes(canonical_json(candidates))
    axes = preflight["axes"]
    body: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "artifact_role": E1_AUTHORITY_ROLE,
        "phase": "G8_E",
        "authority_order": "candidate_id_ascending",
        "candidate_fields": list(E1_CANDIDATE_FIELDS),
        "dimensions": {
            "datasets": axes["datasets"],
            "ratios": axes["ratios"],
            "source_codecs": axes["source_codecs"],
            "encode_axis_px": axes["encode_axis_px"],
            "modulations": axes["modulations"],
            "ldpc_rates": axes["ldpc_rates"],
            "snr_convention": axes["snr_convention"],
            "snr_grid_db": axes["snr_grid_db"],
            "decoder": axes["decoder"],
        },
        "candidate_count": len(candidates),
        "structural_candidate_count": len(structural_keys),
        "candidate_x_snr_count": len(candidates),
        "duplicate_count": 0,  # literal-ok: derived duplicate audit result
        "alias_count": 0,  # literal-ok: derived complete-identity audit result
        "missing_combination_count": 0,  # literal-ok: independently derived Cartesian authority
        "candidates": candidates,
        "candidate_authority_digest": candidate_digest,
    }
    body["authority_id"] = E1_AUTHORITY_PREFIX + candidate_digest
    return body


def validate_e1_candidate_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive and strictly verify the complete candidate authority."""

    _require(isinstance(value, Mapping), "candidate authority is not an object")
    required = (
        "schema_version",
        "artifact_role",
        "phase",
        "authority_order",
        "candidate_fields",
        "dimensions",
        "candidate_count",
        "structural_candidate_count",
        "candidate_x_snr_count",
        "duplicate_count",
        "alias_count",
        "missing_combination_count",
        "candidates",
        "candidate_authority_digest",
        "authority_id",
    )
    _strict_keys(value, required, "candidate authority")
    _require(value["schema_version"] == E1_SCHEMA_VERSION, "candidate authority schema version differs")
    _require(value["artifact_role"] == E1_AUTHORITY_ROLE and value["phase"] == "G8_E", "candidate authority header differs")
    _require(value["authority_order"] == "candidate_id_ascending", "candidate authority ordering differs")
    fresh = build_e1_candidate_authority()
    for field in (
        "candidate_fields",
        "dimensions",
        "candidate_count",
        "structural_candidate_count",
        "candidate_x_snr_count",
        "duplicate_count",
        "alias_count",
        "missing_combination_count",
        "candidates",
        "candidate_authority_digest",
        "authority_id",
    ):
        _require(value[field] == fresh[field], f"candidate authority {field} differs from live preflight")
    return dict(value)


def verify_e1_authority_file(path: Path = E1_AUTHORITY_PATH) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read candidate authority: {exc}") from exc
    _require(raw == rendered_json(value), "candidate authority is not canonical rendered JSON")
    return validate_e1_candidate_authority(value)


def build_e1_source_manifest(
    campaign_id: str,
    *,
    source_commit: str | None = None,
    dataset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind every E2-affecting source and external asset before data.

    The manifest excludes itself, the E1 contract and the authority because
    those are mutually referential artifacts.  It does bind the generator and
    verifier tools: changing the source that creates or checks a scientific
    record requires a new pre-data freeze.
    """

    _require(isinstance(campaign_id, str) and campaign_id.startswith("g8e-"), "invalid E1 campaign ID")
    commit = source_commit or _git_head()
    _git_commit_count(commit, "E1 source manifest commit")
    _require(not _git_dirty(), "E1 source manifest requires a clean worktree")
    bindings = [_binding(path, role) for path, role in E1_SOURCE_BINDING_PATHS]
    metadata = dataset_metadata or {
        dataset: _dataset_manifest_metadata(dataset, require_extracted=dataset in {E1_INITIAL_DATASET, E1_FALLBACK_DATASET})
        for dataset in (E1_INITIAL_DATASET, E1_FALLBACK_DATASET, E1_SMOKE_DATASET)
    }
    for dataset in (E1_INITIAL_DATASET, E1_FALLBACK_DATASET, E1_SMOKE_DATASET):
        info = metadata[dataset]
        bindings.append(_asset_file_binding(info["archive"]["path"], f"{dataset}_archive_asset"))
        bindings.append(_asset_file_binding(info["manifest"]["path"], f"{dataset}_validation_manifest"))
    bindings.append(_asset_file_binding("checkpoints/reference_classifier/epoch-99.pt", "g1_classifier_checkpoint"))
    bindings.sort(key=lambda item: (item["path"], item["role"]))
    paths = [item["path"] for item in bindings]
    _require(len(paths) == len(set(paths)), "E1 source manifest contains a duplicate path")
    body: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "artifact_role": E1_SOURCE_MANIFEST_ROLE,
        "phase": "G8_E",
        "campaign_id": campaign_id,
        "source_commit": commit,
        "git_dirty": False,
        "bindings": bindings,
        "external_assets_are_byte_authenticated": True,
        "manifest_excluded_from_its_own_bindings": True,
        "contract_excluded_from_bindings_to_avoid_cycle": True,
    }
    body["manifest_id"] = E1_SOURCE_MANIFEST_PREFIX + sha256_bytes(canonical_json(body))
    return body


def _git_show_bytes(commit: str, path_text: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path_text}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=15,  # literal-ok: bounded provenance command
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def validate_e1_source_manifest(
    value: Mapping[str, Any],
    *,
    expected_campaign_id: str | None = None,
) -> dict[str, Any]:
    required = (
        "schema_version",
        "artifact_role",
        "phase",
        "campaign_id",
        "source_commit",
        "git_dirty",
        "bindings",
        "external_assets_are_byte_authenticated",
        "manifest_excluded_from_its_own_bindings",
        "contract_excluded_from_bindings_to_avoid_cycle",
        "manifest_id",
    )
    _strict_keys(value, required, "E1 source manifest")
    _require(value["schema_version"] == E1_SCHEMA_VERSION, "E1 source manifest schema differs")
    _require(value["artifact_role"] == E1_SOURCE_MANIFEST_ROLE and value["phase"] == "G8_E", "E1 source manifest header differs")
    if expected_campaign_id is not None:
        _require(value["campaign_id"] == expected_campaign_id, "E1 source manifest campaign differs")
    _git_commit_count(value["source_commit"], "E1 source manifest commit")
    _require(value["git_dirty"] is False, "E1 source manifest was created dirty")
    _require(value["external_assets_are_byte_authenticated"] is True, "E1 external assets are not byte-authenticated")
    _require(value["manifest_excluded_from_its_own_bindings"] is True and value["contract_excluded_from_bindings_to_avoid_cycle"] is True, "E1 source-manifest cycle policy differs")
    body = dict(value)
    manifest_id = body.pop("manifest_id")
    _require(manifest_id == E1_SOURCE_MANIFEST_PREFIX + sha256_bytes(canonical_json(body)), "E1 source manifest ID differs")
    bindings = value["bindings"]
    _require(isinstance(bindings, list) and bindings, "E1 source manifest has no bindings")
    previous = ""
    for item in bindings:
        _strict_keys(item, ("path", "role", "bytes", "sha256"), "E1 source binding")
        _require(isinstance(item["path"], str) and item["path"] > previous, "E1 source bindings are not unique ordered paths")
        previous = item["path"]
        _digest(item["sha256"], f"E1 source binding {item['path']}")
        path = REPO_ROOT / item["path"]
        _require(path.is_file(), f"E1 source binding is missing: {item['path']}")
        _require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"], f"E1 source/config drift: {item['path']}")
        # Source files are also checked against the committed source snapshot,
        # so a later descendant cannot silently replace a bound implementation.
        if item["role"] not in {"g1_classifier_checkpoint"} and not item["role"].endswith("_archive_asset"):
            historical = _git_show_bytes(value["source_commit"], item["path"])
            if historical is not None:
                _require(sha256_bytes(historical) == item["sha256"], f"E1 source snapshot differs: {item['path']}")
    return dict(value)


def verify_e1_source_manifest_file(
    path: Path = E1_SOURCE_MANIFEST_PATH,
    *,
    expected_campaign_id: str | None = None,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read E1 source manifest: {exc}") from exc
    _require(raw == rendered_json(value), "E1 source manifest is not canonical rendered JSON")
    return validate_e1_source_manifest(value, expected_campaign_id=expected_campaign_id)


def build_e1_corpus_spec(dataset_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    initial = dataset_metadata[E1_INITIAL_DATASET]
    train_manifest = initial["manifest"]
    body: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "artifact_role": E1_CORPUS_SPEC_ROLE,
        "phase": "G8_E",
        "checkpoint": "E1",
        "training_only": True,
        "materialized": False,
        "materialized_object_count": 0,  # literal-ok: schema-only artifact has no objects
        "dataset": E1_INITIAL_DATASET,
        "dataset_role": "headline",
        "generator": {
            "owner_phase": "G8_F",
            "source_must_be_bound_at": "G8_F/F0",
            "source_path_is_not_executed_in_E1": True,
            "deterministic_generation_required": True,
        },
        "train_manifest": {
            "path": train_manifest["path"],
            "sha256": train_manifest["sha256"],
            "archive_sha256": initial["archive"]["sha256"],
            "stable_id_order": "ascending_manifest_order",
            "expected_count": train_manifest["train_count"],
            "expected_stable_id_set_sha256": train_manifest["train_id_set_sha256"],
        },
        "selected_pass_one_lineage": {
            "required": True,
            "state_path": "results/baseline/g8_e/pass_one_state.json",
            "state_sha256": None,
            "selection_record_field": "authority_candidate_id",
            "selection_state_must_be_immutable": True,
        },
        "forbidden_membership": {
            "validation_manifest_path": train_manifest["path"],
            "validation_id_set_sha256": initial["manifest"]["validation_id_set_sha256"],
            "test_id_set_sha256": initial["manifest"]["test_id_set_sha256"],
            "validation_ids_forbidden": True,
            "test_ids_forbidden": True,
            "validation_or_test_ids_may_not_be_materialized": True,
        },
        "generation_rules": {
            "input_split": "train",
            "output_split": "train",
            "source_images_are_canonical_validation_free_records": True,
            "selected_configs_are_referenced_not_recomputed": True,
            "no_validation_or_test_fallback": True,
        },
    }
    body["corpus_spec_id"] = E1_CORPUS_SPEC_PREFIX + sha256_bytes(canonical_json(body))
    return body


def validate_e1_corpus_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version",
        "artifact_role",
        "phase",
        "checkpoint",
        "training_only",
        "materialized",
        "materialized_object_count",
        "dataset",
        "dataset_role",
        "generator",
        "train_manifest",
        "selected_pass_one_lineage",
        "forbidden_membership",
        "generation_rules",
        "corpus_spec_id",
    )
    _strict_keys(value, required, "E1 corpus specification")
    _require(value["schema_version"] == E1_SCHEMA_VERSION and value["artifact_role"] == E1_CORPUS_SPEC_ROLE, "E1 corpus header differs")
    _require(value["phase"] == "G8_E" and value["checkpoint"] == "E1", "E1 corpus phase differs")
    _require(value["training_only"] is True and value["materialized"] is False and value["materialized_object_count"] == 0, "E1 corpus is materialized or not training-only")
    _require(value["dataset"] == E1_INITIAL_DATASET and value["dataset_role"] == "headline", "E1 corpus dataset role differs")
    _strict_keys(value["generator"], ("owner_phase", "source_must_be_bound_at", "source_path_is_not_executed_in_E1", "deterministic_generation_required"), "E1 corpus generator")
    _require(value["generator"] == {
        "owner_phase": "G8_F",
        "source_must_be_bound_at": "G8_F/F0",
        "source_path_is_not_executed_in_E1": True,
        "deterministic_generation_required": True,
    }, "E1 corpus generator policy differs")
    _strict_keys(value["train_manifest"], ("path", "sha256", "archive_sha256", "stable_id_order", "expected_count", "expected_stable_id_set_sha256"), "E1 corpus train manifest")
    _require(value["train_manifest"]["stable_id_order"] == "ascending_manifest_order", "E1 corpus train order differs")
    _digest(value["train_manifest"]["sha256"], "E1 corpus train manifest SHA")
    _digest(value["train_manifest"]["archive_sha256"], "E1 corpus train archive SHA")
    _digest(value["train_manifest"]["expected_stable_id_set_sha256"], "E1 corpus train ID digest")
    _strict_keys(value["selected_pass_one_lineage"], ("required", "state_path", "state_sha256", "selection_record_field", "selection_state_must_be_immutable"), "E1 corpus lineage")
    _require(value["selected_pass_one_lineage"]["required"] is True and value["selected_pass_one_lineage"]["state_sha256"] is None, "E1 corpus pass-one lineage is already materialized")
    _strict_keys(value["forbidden_membership"], ("validation_manifest_path", "validation_id_set_sha256", "test_id_set_sha256", "validation_ids_forbidden", "test_ids_forbidden", "validation_or_test_ids_may_not_be_materialized"), "E1 corpus forbidden membership")
    _require(value["forbidden_membership"]["validation_ids_forbidden"] is True and value["forbidden_membership"]["test_ids_forbidden"] is True and value["forbidden_membership"]["validation_or_test_ids_may_not_be_materialized"] is True, "E1 corpus split isolation differs")
    _digest(value["forbidden_membership"]["validation_id_set_sha256"], "E1 corpus validation ID digest")
    _digest(value["forbidden_membership"]["test_id_set_sha256"], "E1 corpus test ID digest")
    _strict_keys(value["generation_rules"], ("input_split", "output_split", "source_images_are_canonical_validation_free_records", "selected_configs_are_referenced_not_recomputed", "no_validation_or_test_fallback"), "E1 corpus generation rules")
    _require(value["generation_rules"] == {
        "input_split": "train",
        "output_split": "train",
        "source_images_are_canonical_validation_free_records": True,
        "selected_configs_are_referenced_not_recomputed": True,
        "no_validation_or_test_fallback": True,
    }, "E1 corpus generation policy differs")
    body = dict(value)
    corpus_id = body.pop("corpus_spec_id")
    _require(corpus_id == E1_CORPUS_SPEC_PREFIX + sha256_bytes(canonical_json(body)), "E1 corpus specification ID differs")
    return dict(value)


def verify_e1_corpus_spec_file(path: Path = E1_CORPUS_SPEC_PATH) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read E1 corpus specification: {exc}") from exc
    _require(raw == rendered_json(value), "E1 corpus specification is not canonical rendered JSON")
    return validate_e1_corpus_spec(value)


E1_RECORD_FIELDS = (
    "schema_version",
    "artifact_role",
    "record_id",
    "campaign_id",
    "contract_id",
    "authority_ordinal",
    "authority_candidate_id",
    "work_unit_id",
    "dataset",
    "dataset_role",
    "validation_split",
    "validation_split_identity",
    "image_identity",
    "source_image",
    "candidate",
    "budget_identity",
    "codec_configuration",
    "emitted_codestream",
    "reconstruction",
    "reconstruction_cache_object_id",
    "br11",
    "g8_c_table",
    "bler_linkage",
    "classifier",
    "outcome",
    "correct_count",
    "total_count",
    "accuracy_derivation",
    "provenance",
    "validation_only",
    "test_access",
    "training",
    "scientific_evidence",
    "merge_eligible",
)


@dataclass(frozen=True)
class G8EScientificMeasurementRecord:
    """Strict per-image E2 record; distinct from the non-scientific D4 record."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_e1_scientific_record(self.value)

    @property
    def record_id(self) -> str:
        return str(self.value["record_id"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "G8EScientificMeasurementRecord":
        return cls(json.loads(canonical_json(value)))


def _reject_accuracy_keys(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(key != "accuracy", f"{path} contains a caller-supplied accuracy field")
            _reject_accuracy_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_accuracy_keys(item, f"{path}[{index}]")


def _record_id(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "record_id"}
    return E1_RECORD_PREFIX + sha256_bytes(canonical_json(body))


def _require_optional_digest(value: Any, label: str) -> None:
    if value is not None:
        _digest(value, label)


def _validate_e1_outcome(value: Mapping[str, Any]) -> None:
    _strict_keys(value, ("status", "selection_eligible", "failure_semantics"), "E1 record outcome")
    _require(value["status"] in E1_ALLOWED_OUTCOMES, "E1 record outcome status is unknown")
    _require(isinstance(value["selection_eligible"], bool), "E1 record eligibility is not boolean")
    _require(isinstance(value["failure_semantics"], str) and value["failure_semantics"], "E1 record failure semantics are empty")
    if value["status"] in {"structural_infeasibility", "codec_infeasibility"}:
        _require(value["selection_eligible"] is False, "infeasible E1 record is selectable")
    else:
        _require(value["selection_eligible"] is True, "emitted E1 record is not selectable")


def validate_e1_scientific_record(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an E2 record without accepting a float as scientific truth."""

    _require(isinstance(value, Mapping), "E1 scientific record is not an object")
    _strict_keys(value, E1_RECORD_FIELDS, "E1 scientific record")
    _reject_accuracy_keys(value)
    _require(value["schema_version"] == E1_SCHEMA_VERSION, "E1 scientific record schema differs")
    _require(value["artifact_role"] == "g8_e_scientific_measurement_record", "E1 scientific record role differs")
    _require(isinstance(value["record_id"], str) and value["record_id"] == _record_id(value), "E1 scientific record ID differs")
    _require(isinstance(value["campaign_id"], str) and value["campaign_id"].startswith("g8e-"), "E1 scientific record campaign differs")
    _require(isinstance(value["contract_id"], str) and value["contract_id"].startswith(E1_CONTRACT_PREFIX), "E1 scientific record contract is not an E1 contract")
    _require(isinstance(value["authority_ordinal"], int) and not isinstance(value["authority_ordinal"], bool) and value["authority_ordinal"] >= 0, "E1 scientific record authority ordinal is invalid")
    _require(isinstance(value["authority_candidate_id"], str) and value["authority_candidate_id"], "E1 scientific record has no authority candidate")
    _require(isinstance(value["work_unit_id"], str) and value["work_unit_id"].startswith("g8eunit-"), "E1 scientific record work-unit ID is invalid")
    _require(value["dataset"] == E1_INITIAL_DATASET, "E1 scientific record is outside the initial dataset boundary")
    _require(value["dataset_role"] == "headline", "E1 scientific record dataset role differs")
    _require(value["validation_split"] == E1_VALIDATION_SPLIT, "E1 scientific record is not validation-only")
    _strict_keys(value["validation_split_identity"], ("dataset", "split", "dataset_version", "manifest_sha256", "stable_id_set_sha256"), "E1 record validation split identity")
    split_identity = value["validation_split_identity"]
    _require(split_identity["dataset"] == E1_INITIAL_DATASET and split_identity["split"] == E1_VALIDATION_SPLIT, "E1 record validation split identity differs")
    for field in ("dataset_version", "manifest_sha256", "stable_id_set_sha256"):
        _digest(split_identity[field], f"E1 record split {field}")
    _strict_keys(value["image_identity"], ("stable_sample_id", "label", "dataset", "split", "image_identity_id"), "E1 record image identity")
    image = value["image_identity"]
    _require(image["dataset"] == E1_INITIAL_DATASET and image["split"] == E1_VALIDATION_SPLIT, "E1 record image is not initial validation data")
    _require(isinstance(image["stable_sample_id"], str) and image["stable_sample_id"], "E1 record stable sample ID is empty")
    _require(isinstance(image["label"], int) and not isinstance(image["label"], bool) and 0 <= image["label"] < 10, "E1 record label is invalid")  # literal-ok: Imagenette-160 class-index contract
    _require(isinstance(image["image_identity_id"], str) and image["image_identity_id"], "E1 record image identity ID is empty")
    _strict_keys(value["source_image"], ("source_bytes_sha256", "canonical_pixels_sha256", "canonical_shape", "source_payload_rule"), "E1 record source image")
    for field in ("source_bytes_sha256", "canonical_pixels_sha256"):
        _digest(value["source_image"][field], f"E1 record source image {field}")
    shape = value["source_image"]["canonical_shape"]
    _require(isinstance(shape, list) and len(shape) == 3 and shape[2] == 3 and all(isinstance(item, int) and item > 0 for item in shape), "E1 record canonical shape is invalid")
    _require(value["source_image"]["source_payload_rule"] in {"exact_encoded_jpeg_file_bytes", "exact_per_image_record_from_train_X_or_test_X_before_axis_transpose_or_pil"}, "E1 record source payload rule differs")
    _require(isinstance(value["candidate"], Mapping), "E1 record candidate is not an object")
    _require(set(value["candidate"]) == set(E1_CANDIDATE_FIELDS), "E1 record candidate dimensions are incomplete")
    _require(value["candidate"]["candidate_id"] == value["authority_candidate_id"], "E1 record candidate and authority ID differ")
    _require(value["candidate"]["dataset"] == E1_INITIAL_DATASET and value["candidate"]["dataset_role"] == "headline", "E1 record candidate dataset role differs")
    _require(value["candidate"]["encode_axis_px"] in {160, 128, 96, 64}, "E1 record candidate axis is not in the frozen Imagenette authority")  # literal-ok: frozen Imagenette-160 axis ladder
    _strict_keys(value["budget_identity"], ("ratio", "k_symbols", "payload_budget_bytes", "packet_config_id"), "E1 record budget identity")
    _require(value["budget_identity"]["ratio"] == value["candidate"]["ratio"] and value["budget_identity"]["packet_config_id"] == value["candidate"]["packet_config_id"], "E1 record budget and candidate differ")
    _require(isinstance(value["budget_identity"]["k_symbols"], int) and value["budget_identity"]["k_symbols"] > 0, "E1 record budget symbols are invalid")
    _require(isinstance(value["budget_identity"]["payload_budget_bytes"], int) and value["budget_identity"]["payload_budget_bytes"] > 0, "E1 record budget bytes are invalid")
    _strict_keys(value["codec_configuration"], ("identity_type", "configuration_hash", "snapshot_sha256", "runtime_version", "encode_axis_px"), "E1 record codec configuration")
    _require(value["codec_configuration"]["identity_type"] == "jpeg2000_configuration" and value["codec_configuration"]["encode_axis_px"] == value["candidate"]["encode_axis_px"], "E1 record codec configuration differs")
    _digest(value["codec_configuration"]["configuration_hash"], "E1 record codec configuration hash")
    _digest(value["codec_configuration"]["snapshot_sha256"], "E1 record codec snapshot hash")
    _validate_e1_outcome(value["outcome"])
    outcome = value["outcome"]["status"]
    if outcome in {"structural_infeasibility", "codec_infeasibility"}:
        _require(value["emitted_codestream"] is None and value["reconstruction"] is None and value["reconstruction_cache_object_id"] is None and value["br11"] is None, "infeasible E1 record contains an emitted or reconstruction object")
        _require(value["correct_count"] is None and value["total_count"] is None, "infeasible E1 record enters the clean-accuracy denominator")
    else:
        _require(isinstance(value["emitted_codestream"], Mapping) and isinstance(value["br11"], Mapping), "emitted E1 record lacks codestream or BR-11 identity")
        _strict_keys(value["emitted_codestream"], ("emitted_file_identity_id", "codestream_sha256", "emitted_bytes", "payload_budget_bytes", "filler_bytes", "actual_bytes_authoritative"), "E1 emitted codestream")
        _digest(value["emitted_codestream"]["codestream_sha256"], "E1 codestream SHA")
        _require(value["emitted_codestream"]["actual_bytes_authoritative"] is True, "E1 emitted bytes are not actual-byte authoritative")
        _require(isinstance(value["emitted_codestream"]["emitted_bytes"], int) and value["emitted_codestream"]["emitted_bytes"] > 0, "E1 emitted byte count is invalid")
        _require(isinstance(value["reconstruction"], Mapping) and isinstance(value["reconstruction_cache_object_id"], str), "E1 emitted record lacks reconstruction identity")
        _strict_keys(value["reconstruction"], ("identity_id", "decoded_pixels_sha256", "output_shape", "codec_configuration_hash", "image_identity_id"), "E1 reconstruction identity")
        _digest(value["reconstruction"]["decoded_pixels_sha256"], "E1 reconstruction pixels SHA")
        _digest(value["reconstruction"]["codec_configuration_hash"], "E1 reconstruction codec hash")
        _require(value["reconstruction"]["codec_configuration_hash"] == value["codec_configuration"]["configuration_hash"], "E1 reconstruction codec hash differs")
        _require(isinstance(value["br11"], Mapping), "E1 BR-11 accounting is not an object")
        _require(value["br11"].get("accounting_rule") == "AM-81" and value["br11"].get("verdict") == outcome, "E1 BR-11 semantics differ from outcome")
        _require(isinstance(value["correct_count"], int) and not isinstance(value["correct_count"], bool) and value["correct_count"] in {0, 1}, "E1 per-image correct count is invalid")  # literal-ok: one per-image record
        _require(value["total_count"] == 1, "E1 per-image denominator is not one")  # literal-ok: one per-image record
    _require(value["accuracy_derivation"] == "sum correct_count / sum total_count; no caller-supplied accuracy field", "E1 accuracy derivation differs")
    _strict_keys(value["g8_c_table"], ("table_id", "table_sha256", "merge_report_id", "merge_report_sha256", "historical_c6_id", "historical_c6_sha256", "portable_manifest_id", "portable_manifest_sha256", "portable_provenance_id", "portable_provenance_sha256", "portable_scientific_runtime_sha256", "portable_verification_epoch"), "E1 record G8_C binding")
    for field in ("table_sha256", "merge_report_sha256", "historical_c6_sha256", "portable_manifest_sha256", "portable_provenance_sha256", "portable_scientific_runtime_sha256"):
        _digest(value["g8_c_table"][field], f"E1 record G8_C {field}")
    _require(value["g8_c_table"]["portable_verification_epoch"] == "g8-c-portable-scientific-runtime-v1", "E1 record portable epoch differs")
    _strict_keys(value["bler_linkage"], ("lookup_identity", "table_id", "table_sha256", "lookup_mode", "interpolation", "extrapolation", "uncharacterized_is_ineligible"), "E1 BLER linkage")
    _require(value["bler_linkage"]["table_id"] == value["g8_c_table"]["table_id"] and value["bler_linkage"]["table_sha256"] == value["g8_c_table"]["table_sha256"], "E1 BLER table linkage differs")
    _require(value["bler_linkage"]["lookup_mode"] == "exact_frozen_identity_and_exact_snr_or_explicit_uncharacterized" and value["bler_linkage"]["interpolation"] is False and value["bler_linkage"]["extrapolation"] is False and value["bler_linkage"]["uncharacterized_is_ineligible"] is True, "E1 BLER lookup semantics differ")
    _require(isinstance(value["classifier"], Mapping), "E1 classifier identity is missing")
    _require(value["classifier"].get("identity_type") in {"g1_clean_classifier", "frozen_constant_outage_policy"}, "E1 classifier identity is wrong")
    _require(value["validation_only"] is True and value["test_access"] == 0 and value["training"] is False, "E1 record isolation flags differ")
    _require(value["scientific_evidence"] is True and value["merge_eligible"] is True, "E1 scientific record is non-scientific or non-mergeable")
    _strict_keys(value["provenance"], ("source_manifest_id", "source_manifest_sha256", "source_commit", "execution_profile_id", "execution_profile_selection_sha256", "contract_id"), "E1 record provenance")
    _digest(value["provenance"]["source_manifest_sha256"], "E1 record source manifest SHA")
    _digest(value["provenance"]["execution_profile_selection_sha256"], "E1 record profile selection SHA")
    _git_commit_count(value["provenance"]["source_commit"], "E1 record source commit")
    if contract is not None:
        _require(value["contract_id"] == contract["contract_id"] and value["provenance"]["contract_id"] == contract["contract_id"], "E1 record contract binding differs")
    if authority is not None:
        _require(value["authority_ordinal"] < authority["candidate_count"], "E1 record authority ordinal is outside authority")
        candidate = authority["candidates"][value["authority_ordinal"]]
        _require(candidate["candidate_id"] == value["authority_candidate_id"] and dict(value["candidate"]) == candidate, "E1 record candidate is not the exact authority row")
    return dict(value)


def _published_binding(path_text: str, role: str) -> dict[str, Any]:
    return _binding(path_text, role)


def _dataset_contract_view(info: Mapping[str, Any]) -> dict[str, Any]:
    manifest = info["manifest"]
    return {
        "dataset": info["dataset"],
        "role": info["role"],
        "dataset_version": info["dataset_version"],
        "archive": dict(info["archive"]),
        "manifest": {
            "path": manifest["path"],
            "bytes": manifest["bytes"],
            "sha256": manifest["sha256"],
            "split": manifest["split"],
            "validation_count": manifest["validation_count"],
            "validation_id_set_sha256": manifest["validation_id_set_sha256"],
            "train_count": manifest["train_count"],
            "train_id_set_sha256": manifest["train_id_set_sha256"],
            "test_count": manifest["test_count"],
            "test_id_set_sha256": manifest["test_id_set_sha256"],
        },
        "class_count": info["class_count"],
        "image_size": list(info["image_size"]),
        "loader": info["loader"],
        "source_payload": info["source_payload"],
        "class_index_source": info["class_index_source"],
        "extracted_asset_authenticated": info["extracted_asset_authenticated"],
    }


def _outage_identity(outage: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "dataset": outage["dataset"],
        "split": outage["split"],
        "manifest_sha256": outage["manifest_sha256"],
        "selected_class": outage["selected_class"],
        "numerator": outage["numerator"],
        "denominator": outage["denominator"],
        "class_counts": outage["class_counts"],
        "selection_rule": outage["selection_rule"],
        "tie_break": outage["tie_break"],
    }
    return {
        "object_id": "g8eoutage-" + sha256_bytes(canonical_json(fields)),
        "path": "results/baseline/w4/outage_policy.json",
        "sha256": sha256_file(REPO_ROOT / "results/baseline/w4/outage_policy.json"),
        **fields,
        "accuracy_is_count_derived": True,
        "uniform_assumption_rejected": True,
    }


def _selection_policy(integration: Mapping[str, Any]) -> dict[str, Any]:
    machinery = integration.get("selection_machinery")
    _require(isinstance(machinery, Mapping), "W4 integration has no selection machinery")
    required = (
        "selection_policy_sha256",
        "tie_break_order",
        "tie_equality",
        "fixed_modulation",
        "selection_passes",
        "selection_termination_pass",
        "system_modes",
        "uncharacterized_candidates_are",
    )
    _require(all(field in machinery for field in required), "W4 selection policy is incomplete")
    fixed = machinery["fixed_modulation"]
    _require(isinstance(fixed, Mapping), "W4 fixed-modulation policy is not an object")
    return {
        "integration_adjudication_sha256": sha256_file(REPO_ROOT / "results/baseline/w4/integration_adjudication.json"),
        "selection_policy_sha256": machinery["selection_policy_sha256"],
        "tie_break_order": list(machinery["tie_break_order"]),
        "tie_equality": machinery["tie_equality"],
        "fixed_modulation": dict(fixed),
        "selection_passes": list(machinery["selection_passes"]),
        "selection_termination_pass": machinery["selection_termination_pass"],
        "system_modes": list(machinery["system_modes"]),
        "uncharacterized_candidates_are": machinery["uncharacterized_candidates_are"],
        "frozen_before_g8": machinery.get("tie_break_frozen_before_g8") is True,
    }


def _representative_record(authority_row: Mapping[str, Any], contract_seed: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-sized placeholder used only for storage estimation."""

    zero = "0" * 64  # literal-ok: placeholder SHA width for storage estimation
    return {
        "schema_version": E1_SCHEMA_VERSION,
        "artifact_role": "g8_e_scientific_measurement_record",
        "record_id": E1_RECORD_PREFIX + zero,
        "campaign_id": contract_seed["campaign_id"],
        "contract_id": E1_CONTRACT_PREFIX + zero,
        "authority_ordinal": 0,  # literal-ok: representative schema row only
        "authority_candidate_id": authority_row["candidate_id"],
        "work_unit_id": "g8eunit-" + zero,
        "dataset": authority_row["dataset"],
        "dataset_role": authority_row["dataset_role"],
        "validation_split": E1_VALIDATION_SPLIT,
        "validation_split_identity": {
            "dataset": E1_INITIAL_DATASET,
            "split": E1_VALIDATION_SPLIT,
            "dataset_version": zero,
            "manifest_sha256": zero,
            "stable_id_set_sha256": zero,
        },
        "image_identity": {
            "stable_sample_id": zero[:32],  # literal-ok: stable sample ID width
            "label": 0,  # literal-ok: representative schema row only
            "dataset": authority_row["dataset"],
            "split": E1_VALIDATION_SPLIT,
            "image_identity_id": "g8dimage-" + zero,
        },
        "source_image": {
            "source_bytes_sha256": zero,
            "canonical_pixels_sha256": zero,
            "canonical_shape": [160, 160, 3],  # literal-ok: representative Imagenette schema shape
            "source_payload_rule": "exact_encoded_jpeg_file_bytes",
        },
        "candidate": dict(authority_row),
        "budget_identity": {
            "ratio": authority_row["ratio"],
            "k_symbols": 1,  # literal-ok: representative schema row only
            "payload_budget_bytes": 1,  # literal-ok: representative schema row only
            "packet_config_id": authority_row["packet_config_id"],
        },
        "codec_configuration": {
            "identity_type": "jpeg2000_configuration",
            "configuration_hash": zero,
            "snapshot_sha256": zero,
            "runtime_version": "2.5.4",
            "encode_axis_px": authority_row["encode_axis_px"],
        },
        "emitted_codestream": None,
        "reconstruction": None,
        "reconstruction_cache_object_id": None,
        "br11": None,
        "g8_c_table": {"table_id": "g8pblertable-placeholder"},
        "bler_linkage": {"lookup_identity": {}, "table_id": "g8pblertable-placeholder", "table_sha256": zero, "lookup_mode": "exact", "interpolation": False, "extrapolation": False, "uncharacterized_is_ineligible": True},
        "classifier": {"identity_type": "g1_clean_classifier"},
        "outcome": {"status": "codec_infeasibility", "selection_eligible": False, "failure_semantics": "placeholder_only"},
        "correct_count": None,
        "total_count": None,
        "accuracy_derivation": "sum correct_count / sum total_count; no caller-supplied accuracy field",
        "provenance": {"source_manifest_id": "g8esource-placeholder", "source_manifest_sha256": zero, "source_commit": zero[:40], "execution_profile_id": E1_PROFILE_ID, "execution_profile_selection_sha256": zero, "contract_id": E1_CONTRACT_PREFIX + zero},  # literal-ok: placeholder Git width
        "validation_only": True,
        "test_access": 0,  # literal-ok: representative schema row only
        "training": False,
        "scientific_evidence": True,
        "merge_eligible": True,
    }


def build_e1_bundle() -> dict[str, Any]:
    """Build E1 artifacts and the frozen pre-data contract, without E2 work."""

    verify_e0_file()
    authority = build_e1_candidate_authority()
    metadata = {
        dataset: _dataset_manifest_metadata(dataset, require_extracted=dataset in {E1_INITIAL_DATASET, E1_FALLBACK_DATASET})
        for dataset in (E1_INITIAL_DATASET, E1_FALLBACK_DATASET, E1_SMOKE_DATASET)
    }
    d_contract = _read_object(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json")
    d_handoff = _read_object(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json")
    d_rebind = _read_object(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json")
    g8_c = _current_g8_c_binding()
    classifier = _current_classifier_binding(d_contract)
    _require(classifier["dataset"] == E1_INITIAL_DATASET and classifier["split"] == E1_VALIDATION_SPLIT, "G1 classifier is not the clean Imagenette validation classifier")
    _require(classifier["checkpoint_sha256"] == sha256_file(REPO_ROOT / "checkpoints/reference_classifier/epoch-99.pt"), "G1 checkpoint asset does not authenticate")
    _require(classifier["manifest_sha256"] == metadata[E1_INITIAL_DATASET]["manifest"]["sha256"], "G1 classifier manifest differs from current validation manifest")
    codec_snapshot = json.loads(canonical_json(__import__("baseline.g8_d", fromlist=["current_codec_snapshot"]).current_codec_snapshot()))
    codec_snapshot_sha = sha256_bytes(canonical_json(codec_snapshot))
    _require(codec_snapshot_sha == d_contract["codec_binding"]["configuration_hash"], "G8_D codec snapshot drifted before E1")
    integration = _read_object(REPO_ROOT / "results/baseline/w4/integration_adjudication.json")
    selection = _selection_policy(integration)
    outage = _read_object(REPO_ROOT / "results/baseline/w4/outage_policy.json")
    outage_identity = _outage_identity(outage)
    source_commit = _git_head()
    e0_binding = _published_binding("results/baseline/g8_e/e0_open.json", "g8_e_e0_opening")
    preflight_binding = _published_binding("results/baseline/g8/required_bler_identities.json", "g8_complete_candidate_preflight")
    campaign_seed = {
        "phase": "G8_E",
        "checkpoint": "E1",
        "source_commit": source_commit,
        "e0_sha256": e0_binding["sha256"],
        "authority_digest": authority["candidate_authority_digest"],
        "g8_c_table_id": g8_c["table_id"],
        "g8_c_table_sha256": g8_c["table_sha256"],
        "g8_d_contract_id": d_contract["contract_id"],
        "g8_d_contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
        "initial_manifest_sha256": metadata[E1_INITIAL_DATASET]["manifest"]["sha256"],
        "classifier_checkpoint_sha256": classifier["checkpoint_sha256"],
        "classifier_config_sha256": classifier["classifier_config_sha256"],
        "codec_configuration_hash": codec_snapshot_sha,
        "selection_policy_sha256": selection["selection_policy_sha256"],
        "execution_profile_id": E1_PROFILE_ID,
    }
    campaign_id = "g8e-" + sha256_bytes(canonical_json(campaign_seed))
    try:
        from config.execution_profiles import authenticate_execution_profile, selection_record

        profile_auth = authenticate_execution_profile(
            E1_PROFILE_ID,
            device=E1_DEVICE,
            config_hash=classifier["classifier_config_sha256"],
            require_openjpeg=True,
        )
        profile_selection = selection_record(
            scope_id=campaign_id,
            scope_kind="G8_E_validation_campaign",
            profile_id=E1_PROFILE_ID,
            git_commit=source_commit,
            config_hash=classifier["classifier_config_sha256"],
        )
    except Exception as exc:
        raise G8EContractError(f"E1 execution profile authentication failed: {exc}") from exc
    _require(profile_auth["git_commit"] == source_commit and profile_auth["git_dirty"] is False, "E1 profile authentication source state differs")
    source_manifest = build_e1_source_manifest(campaign_id, source_commit=source_commit, dataset_metadata=metadata)
    corpus_spec = build_e1_corpus_spec(metadata)
    authority_raw_sha = sha256_bytes(rendered_json(authority))
    source_raw_sha = sha256_bytes(rendered_json(source_manifest))
    corpus_raw_sha = sha256_bytes(rendered_json(corpus_spec))
    initial = metadata[E1_INITIAL_DATASET]
    initial_candidates = [row for row in authority["candidates"] if row["dataset"] == E1_INITIAL_DATASET]
    codec_keys = {
        (row["dataset"], row["ratio"], row["encode_axis_px"])
        for row in initial_candidates
    }
    logical_records = len(initial["manifest"]["validation_ids"]) * len(initial_candidates)
    codec_jobs = len(initial["manifest"]["validation_ids"]) * len(codec_keys)
    reuse = len(initial_candidates) // len(codec_keys)
    representative = _representative_record(initial_candidates[0], {"campaign_id": campaign_id})
    representative_bytes = len(rendered_json(representative))
    smoke = _read_object(REPO_ROOT / "results/baseline/w4/smoke_summary.json")
    bounded_rows = int(smoke["raw_rows_count"])
    bounded_seconds = float(smoke["wall_clock_s"])
    bounded_rate = bounded_rows / bounded_seconds
    projected_seconds = codec_jobs / bounded_rate
    contract: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "artifact_role": E1_ARTIFACT_ROLE,
        "phase": "G8_E",
        "checkpoint": "E1",
        "status": "FROZEN_PRE_DATA",
        "campaign_id": campaign_id,
        "campaign_seed": campaign_seed,
        "e0_binding": e0_binding,
        "candidate_authority": {
            "path": str(E1_AUTHORITY_PATH.relative_to(REPO_ROOT)),
            "authority_id": authority["authority_id"],
            "authority_sha256": authority_raw_sha,
            "candidate_authority_digest": authority["candidate_authority_digest"],
            "candidate_count": authority["candidate_count"],
            "initial_dataset_candidate_count": len(initial_candidates),
            "complete_all_roles": True,
            "source_preflight": preflight_binding,
            "dimensions": authority["dimensions"],
            "identity_fields": list(E1_CANDIDATE_FIELDS),
            "duplicate_alias_missing_proof": {
                "duplicate_count": 0,  # literal-ok: authority audit result
                "alias_count": 0,  # literal-ok: authority audit result
                "missing_combination_count": 0,  # literal-ok: authority audit result
                "identity_includes_all_dimensions": True,
            },
        },
        "dataset_boundary": {
            "initial_scientific_dataset": _dataset_contract_view(initial),
            "initial_scope": "headline_only",
            "fallback_headline": _dataset_contract_view(metadata[E1_FALLBACK_DATASET]),
            "fallback_invocation_condition": "invoked at G-8 if compute or a degenerate baseline forces it",
            "fallback_condition_source": "SPEC.md DEC-1 dataset ladder",
            "fallback_invocation_prohibited_in_g8_e": True,
            "smoke_only_dataset": _dataset_contract_view(metadata[E1_SMOKE_DATASET]),
            "smoke_only_disposition": "CIFAR-10 is transport/verdict/accounting/cache plumbing only; no task score or headline authority",
            "test_split": {
                "sealed": True,
                "model_facing_access": False,
                "provenance_only_scan": False,
                "test_access_counter": 0,  # literal-ok: pre-data safety counter
            },
        },
        "codec_and_preprocessing": {
            "codec_identity_type": "jpeg2000_configuration",
            "configuration_hash": codec_snapshot_sha,
            "snapshot_sha256": codec_snapshot_sha,
            "snapshot": codec_snapshot,
            "actual_emitted_bytes_authoritative": True,
            "cache_key_fields": list(codec_snapshot["baseline"]["j2k_cache_key"]),
            "snr_is_excluded_from_codec_search_key": True,
            "preprocessing_source": "src/data/preprocessing.py",
            "canonical_image_rule": get("preprocessing.canonical_image"),
            "codec_input": get("preprocessing.codec_input"),
            "downsample_interpolation": get("preprocessing.codec_downsample_interpolation"),
            "upsample_interpolation": get("preprocessing.codec_upsample_interpolation"),
            "preserves_aspect": get("preprocessing.codec_resize_preserves_aspect"),
        },
        "g8_c_binding": g8_c,
        "g8_d_binding": {
            "contract_id": d_contract["contract_id"],
            "contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
            "handoff_id": d_handoff["artifact_id"],
            "handoff_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"),
            "portable_rebind_provenance_id": d_rebind["provenance_id"],
            "portable_rebind_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json"),
            "d4_d6_flags": _current_d4_contract_flags(d_contract),
            "d4_source_is_not_reused_as_e_record": True,
        },
        "g1_classifier_binding": classifier,
        "w4_selection_binding": selection,
        "measured_outage_binding": outage_identity,
        "scientific_record": {
            "schema_version": E1_SCHEMA_VERSION,
            "artifact_role": "g8_e_scientific_measurement_record",
            "fields": list(E1_RECORD_FIELDS),
            "accuracy_field_permitted": False,
            "count_derivation": "sum correct_count / sum total_count; no caller-supplied accuracy field",
            "per_image_total_count": 1,  # literal-ok: one immutable per-image record
            "validation_only": True,
            "scientific_evidence": True,
            "merge_eligible": True,
        },
        "feasibility_and_denominators": {
            "outcomes": sorted(E1_ALLOWED_OUTCOMES),
            "structural_infeasibility": "record exactly one row before codec/channel; no emitted bytes, BR-11, reconstruction or accuracy count; retained in coverage and ineligible",
            "codec_infeasibility": "record exactly one row after structural plan and before emitted codestream; no reconstruction or accuracy count; retained in coverage and ineligible",
            "decode_failure": "emitted codestream and BR-11 survive; reconstruction is absent; count is derived from frozen outage prediction and denominator is one",
            "delivered": "emitted codestream, BR-11, reconstruction and frozen G-1 classifier correctness are recorded; denominator is one",
            "br11_denominator": "emitted-row count includes delivered and decode_failure; AM-81 header/payload sums include every emitted codestream",
            "clean_accuracy_denominator": "sum total_count over delivered and decode_failure records only; infeasible rows remain present but have null counts and are never zeroed, ignored or imputed",
            "selection_eligibility": "candidate is eligible only when exact complete coverage exists, all required upstream identities resolve and its measured object is count-derived; any infeasible or uncharacterized cell remains explicit and blocks eligibility",
            "missing_row_policy": "missing is a contract failure, never a zero, success, ignored row or imputation",
        },
        "compute_plan": {
            "authority_all_roles_logical_candidates": authority["candidate_count"],
            "initial_headline_logical_candidates": len(initial_candidates),
            "candidate_x_snr_cells_all_roles": authority["candidate_count"],
            "candidate_x_snr_cells_initial": len(initial_candidates),
            "validation_image_count": len(initial["manifest"]["validation_ids"]),
            "logical_image_candidate_records": logical_records,
            "unique_codec_search_computations": codec_jobs,
            "unique_reconstruction_computations_scheduled": codec_jobs,
            "unique_classifier_forwards_scheduled": codec_jobs,
            "cache_reuse": {
                "codec_search_key_excludes_snr_modulation_and_rate": True,
                "candidate_rows_per_image_per_codec_key": reuse,
                "expected_logical_rows_per_unique_key": reuse,
                "reconstruction_reused_across_snr_modulation_rate": True,
                "classifier_forward_reused_across_snr_modulation_rate": True,
            },
            "aggregate_measured_codec_accuracy_objects": len(codec_keys),
            "representative_logical_record_bytes": representative_bytes,
            "logical_record_storage_bytes_estimate": logical_records * representative_bytes,
            "codec_cache_entries": codec_jobs,
            "reconstruction_cache_entries_upper_bound": codec_jobs,
            "historical_bounded_throughput_reference": {
                "source_path": "results/baseline/w4/smoke_summary.json",
                "source_sha256": sha256_file(REPO_ROOT / "results/baseline/w4/smoke_summary.json"),
                "raw_rows": bounded_rows,
                "wall_clock_s": bounded_seconds,
                "rows_per_second": bounded_rate,
                "status": "non_scientific_bounded_validation_plumbing_reference_only",
                "classifier_inference_in_reference": True,
                "not_an_e2_measurement": True,
            },
            "projected_e2_wall_clock": {
                "basis": "unique_codec_reconstruction_classifier jobs divided by bounded W4 rows/sec proxy",
                "seconds": projected_seconds,
                "hours": projected_seconds / 3600,  # literal-ok: seconds-to-hours reporting conversion
                "status": "planning_estimate_not_observed_scientific_output",
            },
        },
        "execution_profile": {
            "selection": profile_selection,
            "authentication": profile_auth,
            "profile_switching": "forbidden; interruption requires explicit supersession/new campaign",
            "sole_writer": "local",
            "profile_id": E1_PROFILE_ID,
            "device": E1_DEVICE,
        },
        "source_manifest_binding": {
            "path": str(E1_SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "manifest_id": source_manifest["manifest_id"],
            "sha256": source_raw_sha,
            "source_commit": source_commit,
            "direct_portable_bindings_required": True,
        },
        "resume_and_custody": {
            "runtime_root": "results/baseline/g8_e/runtime",
            "runtime_root_must_be_absent_at_e1": True,
            "sole_writer": True,
            "parallelism": "disabled; one deterministic writer is sufficient for the frozen plan",
            "work_unit_order": "authority ordinal ascending, then stable validation ID ascending",
            "work_unit_id": "g8eunit-sha256(campaign_id, authority_ordinal, stable_sample_id)",
            "completed_records": "immutable content-addressed files; exact duplicate bytes may be reused, differing bytes reject",
            "publication": "same-directory staged bytes, descriptor-relative no-follow immutable publication, directory fsync; no direct final-path writes",
            "state_durability": "state and aggregate are staged and fsynced before publication; aggregate digest is bound by state",
            "resume": "exact ordered prefix only; no skipped unit, duplicate contribution, silent overwrite or host switch",
            "source_config_drift": "current source manifest, contract, candidate authority, G8_C, G8_D, classifier, manifests, codec snapshot and profile selection must match before resume",
            "stale_aggregate": "reject and reconstruct from immutable completed records; never trust or silently overwrite a stale aggregate",
            "completed_output_reuse": "reuse only when work-unit ID, record ID, source/config digest and exact bytes match",
            "crash_recovery_matrix": [
                "before_cache_publication: no completion contribution",
                "after_cache_publication_before_record: cache may be reused but unit remains incomplete",
                "after_record_publication_before_aggregate: record is authoritative; aggregate is rebuilt",
                "after_aggregate_before_state: aggregate is provisional until state binds it",
                "after_state_publication: exact prefix advances one unit",
            ],
            "first_command": f".venv/bin/python tools/run_g8_e.py --campaign-id {campaign_id} --contract results/baseline/g8_e/measurement_contract.json --runtime-root results/baseline/g8_e/runtime --profile {E1_PROFILE_ID} --device {E1_DEVICE} --start",
            "restart_command": f".venv/bin/python tools/run_g8_e.py --campaign-id {campaign_id} --contract results/baseline/g8_e/measurement_contract.json --runtime-root results/baseline/g8_e/runtime --profile {E1_PROFILE_ID} --device {E1_DEVICE} --resume",
            "am86_exception_used": False,
        },
        "pass_one_preconditions": {
            "authorization_issued": False,
            "authorization_type": "G8Authorization",
            "authorization_scope": {"gate": "G-8", "max_candidates": 64, "max_samples": 25, "max_workload": 512},  # literal-ok: frozen W4 guard limits
            "scorer": "frozen G-1 clean classifier",
            "required_system_modes": selection["system_modes"],
            "required_snr_points": authority["dimensions"]["snr_grid_db"],
            "selection_policy_sha256": selection["selection_policy_sha256"],
            "tie_break_order": selection["tie_break_order"],
            "candidate_authority_digest": authority["candidate_authority_digest"],
            "complete_E4_coverage_predicate": {
                "every_initial_candidate_and_validation_id_exactly_once": True,
                "all_four_outcome_states_explicit": True,
                "all_emitted_rows_have_AM81_BR11": True,
                "all_delivered_rows_have_count_derived_classifier_correctness": True,
                "all_decode_failures_have_frozen_outage_count": True,
                "all_infeasible_rows_have_null_accuracy_counts": True,
                "all_G8_C_lookups_exact_or_explicit_uncharacterized": True,
                "no_duplicate_or_missing_work_unit": True,
            },
            "maximum_candidate_count": 64,  # literal-ok: frozen W4 unauthorized guard
            "maximum_sample_count": 25,  # literal-ok: frozen W4 unauthorized guard
            "maximum_workload": 512,  # literal-ok: frozen W4 unauthorized guard
            "pre_execution_marker_required": True,
            "single_immutable_completion_required": True,
            "no_third_pass": True,
            "pass_one_started": False,
            "pass_two_started": False,
        },
        "corpus_spec_binding": {
            "path": str(E1_CORPUS_SPEC_PATH.relative_to(REPO_ROOT)),
            "corpus_spec_id": corpus_spec["corpus_spec_id"],
            "sha256": corpus_raw_sha,
            "materialized": False,
        },
        "unopened_state": {
            "scientific_campaign_exists": False,
            "scientific_runtime_exists": False,
            "e2_record_exists": False,
            "pass_one_marker_exists": False,
            "pass_one_completion_exists": False,
            "pass_two_state_exists": False,
            "artifact_training_corpus_exists": False,
            "g8_f_state_exists": False,
            "forbidden_paths_checked": [
                "results/baseline/g8_e/runtime",
                "results/baseline/g8_e/campaign_state.json",
                "results/baseline/g8_e/e2_in_progress.json",
                "results/baseline/g8_e/e2_complete.json",
                "results/baseline/g8_e/pass_one_pre_execution.json",
                "results/baseline/g8_e/pass_one_state.json",
                "results/baseline/g8_e/pass_one_complete.json",
                "results/baseline/g8_e/pass_two_state.json",
                "results/baseline/g8_f",
            ],
        },
        "safety": {
            "measurement_coverage": 0,  # literal-ok: E1 pre-data coverage
            "validation_decoding": 0,  # literal-ok: E1 metadata-only boundary
            "inference": 0,  # literal-ok: E1 metadata-only boundary
            "training": 0,  # literal-ok: training prohibited
            "test_access": 0,  # literal-ok: test remains sealed
            "fallback_invoked": False,
            "ratio_adjudicated": False,
            "e2_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
        },
        "declarations": {
            "G8_E_E0_E1_ready_pre_data": True,
            "zero_full_validation_measurements": True,
            "E2_awaits_owner_execution_authorization": True,
            "training_forbidden": True,
            "test_forbidden": True,
            "fallback_forbidden": True,
            "ratio_adjudication_forbidden": True,
            "no_G8Authorization_issued": True,
            "no_validation_image_decoding_performed": True,
        },
    }
    contract["contract_id"] = E1_CONTRACT_PREFIX + sha256_bytes(canonical_json({key: value for key, value in contract.items() if key != "contract_id"}))
    return {
        "authority": authority,
        "source_manifest": source_manifest,
        "corpus_spec": corpus_spec,
        "contract": contract,
        "metadata": metadata,
        "profile_authentication": profile_auth,
    }


E1_CONTRACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "status",
    "campaign_id",
    "campaign_seed",
    "e0_binding",
    "candidate_authority",
    "dataset_boundary",
    "codec_and_preprocessing",
    "g8_c_binding",
    "g8_d_binding",
    "g1_classifier_binding",
    "w4_selection_binding",
    "measured_outage_binding",
    "scientific_record",
    "feasibility_and_denominators",
    "compute_plan",
    "execution_profile",
    "source_manifest_binding",
    "resume_and_custody",
    "pass_one_preconditions",
    "corpus_spec_binding",
    "unopened_state",
    "safety",
    "declarations",
    "contract_id",
)


def _verify_unopened_state(value: Mapping[str, Any]) -> None:
    _strict_keys(value, (
        "scientific_campaign_exists",
        "scientific_runtime_exists",
        "e2_record_exists",
        "pass_one_marker_exists",
        "pass_one_completion_exists",
        "pass_two_state_exists",
        "artifact_training_corpus_exists",
        "g8_f_state_exists",
        "forbidden_paths_checked",
    ), "E1 unopened state")
    booleans = (
        "scientific_campaign_exists",
        "scientific_runtime_exists",
        "e2_record_exists",
        "pass_one_marker_exists",
        "pass_one_completion_exists",
        "pass_two_state_exists",
        "artifact_training_corpus_exists",
        "g8_f_state_exists",
    )
    _require(all(value[field] is False for field in booleans), "E1 scientific state was not unopened")
    _require(isinstance(value["forbidden_paths_checked"], list), "E1 unopened path audit is not a list")
    for path_text in value["forbidden_paths_checked"]:
        _require(not (REPO_ROOT / path_text).exists(), f"unexpected G8_E/G8_F state exists: {path_text}")


def _verify_dataset_contract_view(value: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    _require(dict(value) == _dataset_contract_view(live), f"dataset boundary drifted for {live['dataset']}")


def validate_e1_contract(
    value: Mapping[str, Any],
    *,
    verify_live_assets: bool = True,
    verify_live_profile: bool = False,
) -> dict[str, Any]:
    """Independently verify E1 and reject any post-freeze scientific drift."""

    _require(isinstance(value, Mapping), "E1 contract is not an object")
    _strict_keys(value, E1_CONTRACT_FIELDS, "E1 contract")
    _require(value["schema_version"] == E1_SCHEMA_VERSION and value["artifact_role"] == E1_ARTIFACT_ROLE, "E1 contract header differs")
    _require(value["phase"] == "G8_E" and value["checkpoint"] == "E1" and value["status"] == "FROZEN_PRE_DATA", "E1 contract phase/status differs")
    _require(isinstance(value["campaign_id"], str) and value["campaign_id"].startswith("g8e-"), "E1 campaign ID is invalid")
    body = dict(value)
    contract_id = body.pop("contract_id")
    _require(contract_id == E1_CONTRACT_PREFIX + sha256_bytes(canonical_json(body)), "E1 contract ID differs")
    e0 = value["e0_binding"]
    _strict_keys(e0, ("path", "role", "bytes", "sha256"), "E1 E0 binding")
    _require(e0["path"] == "results/baseline/g8_e/e0_open.json" and e0["role"] == "g8_e_e0_opening", "E1 E0 path binding differs")
    _require(e0["bytes"] == (REPO_ROOT / e0["path"]).stat().st_size and e0["sha256"] == sha256_file(REPO_ROOT / e0["path"]), "E1 E0 bytes changed")
    verify_e0_file()

    authority_binding = value["candidate_authority"]
    _strict_keys(authority_binding, ("path", "authority_id", "authority_sha256", "candidate_authority_digest", "candidate_count", "initial_dataset_candidate_count", "complete_all_roles", "source_preflight", "dimensions", "identity_fields", "duplicate_alias_missing_proof"), "E1 authority binding")
    authority = verify_e1_authority_file(REPO_ROOT / authority_binding["path"])
    _require(authority_binding["authority_id"] == authority["authority_id"] and authority_binding["candidate_authority_digest"] == authority["candidate_authority_digest"], "E1 authority ID/digest binding differs")
    _require(authority_binding["authority_sha256"] == sha256_file(REPO_ROOT / authority_binding["path"]), "E1 authority bytes changed")
    _require(authority_binding["candidate_count"] == authority["candidate_count"] and authority_binding["dimensions"] == authority["dimensions"], "E1 authority dimensions/count differ")
    _require(authority_binding["identity_fields"] == list(E1_CANDIDATE_FIELDS) and authority_binding["complete_all_roles"] is True, "E1 authority identity scope differs")
    _require(authority_binding["duplicate_alias_missing_proof"] == {
        "duplicate_count": 0,  # literal-ok: authority audit result
        "alias_count": 0,  # literal-ok: authority audit result
        "missing_combination_count": 0,  # literal-ok: authority audit result
        "identity_includes_all_dimensions": True,
    }, "E1 authority completeness proof differs")
    preflight = authority_binding["source_preflight"]
    _strict_keys(preflight, ("path", "role", "bytes", "sha256"), "E1 preflight binding")
    _require(preflight["path"] == "results/baseline/g8/required_bler_identities.json" and preflight["sha256"] == sha256_file(REPO_ROOT / preflight["path"]), "E1 preflight binding changed")
    initial_count = sum(1 for row in authority["candidates"] if row["dataset"] == E1_INITIAL_DATASET)
    _require(authority_binding["initial_dataset_candidate_count"] == initial_count, "E1 initial candidate count differs")

    source_binding = value["source_manifest_binding"]
    _strict_keys(source_binding, ("path", "manifest_id", "sha256", "source_commit", "direct_portable_bindings_required"), "E1 source-manifest binding")
    source = verify_e1_source_manifest_file(REPO_ROOT / source_binding["path"], expected_campaign_id=value["campaign_id"])
    _require(source_binding["manifest_id"] == source["manifest_id"] and source_binding["sha256"] == sha256_file(REPO_ROOT / source_binding["path"]), "E1 source-manifest bytes changed")
    _require(source_binding["source_commit"] == source["source_commit"] and source_binding["direct_portable_bindings_required"] is True, "E1 source-manifest commit/boundary differs")
    source_paths = {item["path"] for item in source["bindings"]}
    required_direct = {
        "src/baseline/g8_pascal_merge.py",
        "src/baseline/g8_pascal_portable.py",
        "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json",
        "results/baseline/g8_pascal_successor/portable_verification_provenance.json",
        "results/baseline/g8_d/measurement_contract.json",
        "results/baseline/g8_d/d7_handoff.json",
        "results/baseline/g8_d/portable_rebind_provenance.json",
    }
    _require(required_direct <= source_paths, "E1 source manifest omits an explicit portable/G8_D binding")

    corpus_binding = value["corpus_spec_binding"]
    _strict_keys(corpus_binding, ("path", "corpus_spec_id", "sha256", "materialized"), "E1 corpus binding")
    corpus = verify_e1_corpus_spec_file(REPO_ROOT / corpus_binding["path"])
    _require(corpus_binding["corpus_spec_id"] == corpus["corpus_spec_id"] and corpus_binding["sha256"] == sha256_file(REPO_ROOT / corpus_binding["path"]), "E1 corpus bytes changed")
    _require(corpus_binding["materialized"] is False, "E1 corpus was materialized")

    current_g8c = _current_g8_c_binding()
    _require(value["g8_c_binding"] == current_g8c, "E1 G8_C binding differs from frozen successor")
    d_contract = _read_object(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json")
    d_handoff = _read_object(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json")
    d_rebind = _read_object(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json")
    expected_d = {
        "contract_id": d_contract["contract_id"],
        "contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
        "handoff_id": d_handoff["artifact_id"],
        "handoff_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"),
        "portable_rebind_provenance_id": d_rebind["provenance_id"],
        "portable_rebind_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json"),
        "d4_d6_flags": _current_d4_contract_flags(d_contract),
        "d4_source_is_not_reused_as_e_record": True,
    }
    _require(value["g8_d_binding"] == expected_d, "E1 G8_D binding differs")
    _require(value["g1_classifier_binding"] == _current_classifier_binding(d_contract), "E1 G1 classifier binding differs")
    if verify_live_profile:
        try:
            from config.execution_profiles import authenticate_execution_profile

            live = authenticate_execution_profile(
                value["execution_profile"]["profile_id"],
                device=value["execution_profile"]["device"],
                config_hash=value["g1_classifier_binding"]["classifier_config_sha256"],
                require_openjpeg=True,
            )
        except Exception as exc:
            raise G8EContractError(f"live E1 profile authentication failed: {exc}") from exc
        stored = value["execution_profile"]["authentication"]
        for field in ("execution_profile_id", "lock_file", "lock_file_sha256", "python_version", "torch_version", "torch_cuda_build", "torchvision_version", "numpy_version", "sionna_version", "openjpeg_version", "deterministic_backend", "amp", "gpu_name", "gpu_uuid", "gpu_compute_capability", "gpu_index", "nvidia_smi_index", "config_hash"):
            _require(live[field] == stored[field], f"E1 live profile drift in {field}")
        _require(live["git_dirty"] is False, "E1 live profile worktree is dirty")
    try:
        from config.execution_profiles import verify_selection_record

        verify_selection_record(value["execution_profile"]["selection"], expected_scope_id=value["campaign_id"])
    except Exception as exc:
        raise G8EContractError(f"E1 profile selection record is invalid: {exc}") from exc
    _require(value["execution_profile"]["selection"]["execution_profile_id"] == E1_PROFILE_ID and value["execution_profile"]["device"] == E1_DEVICE, "E1 execution profile differs")
    _require(value["w4_selection_binding"]["selection_policy_sha256"] == "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e", "E1 selection policy digest differs")
    _require(value["w4_selection_binding"]["integration_adjudication_sha256"] == sha256_file(REPO_ROOT / "results/baseline/w4/integration_adjudication.json"), "E1 W4 adjudication changed")
    _require(value["w4_selection_binding"]["system_modes"] == ["classical_adaptive", "classical_fixed_mod", "classical_fixed_mcs"], "E1 system modes differ")
    outage = _read_object(REPO_ROOT / "results/baseline/w4/outage_policy.json")
    _require(value["measured_outage_binding"] == _outage_identity(outage), "E1 outage object changed")

    codec_snapshot = json.loads(canonical_json(__import__("baseline.g8_d", fromlist=["current_codec_snapshot"]).current_codec_snapshot()))
    codec_sha = sha256_bytes(canonical_json(codec_snapshot))
    _require(value["codec_and_preprocessing"]["configuration_hash"] == codec_sha and value["codec_and_preprocessing"]["snapshot"] == codec_snapshot, "E1 codec/preprocessing snapshot drifted")
    _require(value["codec_and_preprocessing"]["snr_is_excluded_from_codec_search_key"] is True, "E1 cache reuse omits no SNR boundary")
    _require(value["scientific_record"]["accuracy_field_permitted"] is False and value["scientific_record"]["count_derivation"] == "sum correct_count / sum total_count; no caller-supplied accuracy field", "E1 record accuracy semantics differ")
    _require(value["feasibility_and_denominators"]["outcomes"] == sorted(E1_ALLOWED_OUTCOMES), "E1 feasibility outcome set differs")

    boundary = value["dataset_boundary"]
    for boundary_key in ("initial_scientific_dataset", "fallback_headline", "smoke_only_dataset"):
        manifest_view = boundary[boundary_key]["manifest"]
        manifest_path = REPO_ROOT / manifest_view["path"]
        _require(manifest_path.is_file(), f"E1 validation manifest is missing: {manifest_view['path']}")
        _require(manifest_view["sha256"] == sha256_file(manifest_path), f"E1 validation manifest drifted: {manifest_view['path']}")
    if verify_live_assets:
        metadata = {
            dataset: _dataset_manifest_metadata(dataset, require_extracted=dataset in {E1_INITIAL_DATASET, E1_FALLBACK_DATASET})
            for dataset in (E1_INITIAL_DATASET, E1_FALLBACK_DATASET, E1_SMOKE_DATASET)
        }
        _verify_dataset_contract_view(boundary["initial_scientific_dataset"], metadata[E1_INITIAL_DATASET])
        _verify_dataset_contract_view(boundary["fallback_headline"], metadata[E1_FALLBACK_DATASET])
        _verify_dataset_contract_view(boundary["smoke_only_dataset"], metadata[E1_SMOKE_DATASET])
        _require(value["g1_classifier_binding"]["checkpoint_sha256"] == sha256_file(REPO_ROOT / "checkpoints/reference_classifier/epoch-99.pt"), "E1 classifier checkpoint drifted")
    _require(boundary["initial_scientific_dataset"]["dataset"] == E1_INITIAL_DATASET and boundary["initial_scientific_dataset"]["role"] == "headline", "E1 initial dataset role differs")
    _require(boundary["fallback_headline"]["dataset"] == E1_FALLBACK_DATASET and boundary["fallback_headline"]["role"] == "fallback_headline", "E1 fallback dataset role differs")
    _require(boundary["fallback_invocation_condition"] == "invoked at G-8 if compute or a degenerate baseline forces it" and boundary["fallback_invocation_prohibited_in_g8_e"] is True, "E1 fallback condition or prohibition differs")
    _require(boundary["smoke_only_dataset"]["dataset"] == E1_SMOKE_DATASET and boundary["smoke_only_dataset"]["role"] == "smoke_only", "E1 smoke dataset role differs")
    _require(boundary["test_split"]["sealed"] is True and boundary["test_split"]["model_facing_access"] is False and boundary["test_split"]["test_access_counter"] == 0, "E1 test boundary differs")

    compute = value["compute_plan"]
    initial_candidates = [row for row in authority["candidates"] if row["dataset"] == E1_INITIAL_DATASET]
    initial_count = boundary["initial_scientific_dataset"]["manifest"]["validation_count"]
    codec_keys = {(row["dataset"], row["ratio"], row["encode_axis_px"]) for row in initial_candidates}
    _require(compute["authority_all_roles_logical_candidates"] == authority["candidate_count"] and compute["initial_headline_logical_candidates"] == len(initial_candidates), "E1 compute authority counts differ")
    _require(compute["validation_image_count"] == initial_count and compute["logical_image_candidate_records"] == initial_count * len(initial_candidates), "E1 logical record count differs")
    _require(compute["unique_codec_search_computations"] == initial_count * len(codec_keys), "E1 unique codec job count differs")
    _require(compute["unique_reconstruction_computations_scheduled"] == compute["unique_codec_search_computations"] and compute["unique_classifier_forwards_scheduled"] == compute["unique_codec_search_computations"], "E1 expensive-work reuse plan differs")
    _require(compute["aggregate_measured_codec_accuracy_objects"] == len(codec_keys), "E1 measured-object count differs")
    _verify_unopened_state(value["unopened_state"])

    custody = value["resume_and_custody"]
    _require(custody["runtime_root_must_be_absent_at_e1"] is True and not (REPO_ROOT / custody["runtime_root"]).exists(), "E1 runtime root is not unopened")
    _require(custody["sole_writer"] is True and custody["parallelism"].startswith("disabled"), "E1 custody/parallelism differs")
    _require(custody["am86_exception_used"] is False, "E1 improperly uses the Pascal custody exception")
    passes = value["pass_one_preconditions"]
    _require(passes["authorization_issued"] is False and passes["pass_one_started"] is False and passes["pass_two_started"] is False, "E1 pass-one state is not unopened")
    _require(passes["selection_policy_sha256"] == value["w4_selection_binding"]["selection_policy_sha256"] and passes["candidate_authority_digest"] == authority["candidate_authority_digest"], "E1 pass-one source bindings differ")
    _require(passes["pre_execution_marker_required"] is True and passes["single_immutable_completion_required"] is True and passes["no_third_pass"] is True, "E1 exact-once pass policy differs")
    safety = value["safety"]
    _require(safety == {
        "measurement_coverage": 0,  # literal-ok: pre-data safety state
        "validation_decoding": 0,  # literal-ok: pre-data safety state
        "inference": 0,  # literal-ok: pre-data safety state
        "training": 0,  # literal-ok: pre-data safety state
        "test_access": 0,  # literal-ok: pre-data safety state
        "fallback_invoked": False,
        "ratio_adjudicated": False,
        "e2_started": False,
        "pass_one_started": False,
        "pass_two_started": False,
    }, "E1 safety counters are nonzero")
    _require(value["declarations"] == {
        "G8_E_E0_E1_ready_pre_data": True,
        "zero_full_validation_measurements": True,
        "E2_awaits_owner_execution_authorization": True,
        "training_forbidden": True,
        "test_forbidden": True,
        "fallback_forbidden": True,
        "ratio_adjudication_forbidden": True,
        "no_G8Authorization_issued": True,
        "no_validation_image_decoding_performed": True,
    }, "E1 declarations differ")
    return dict(value)


def verify_e1_contract_file(
    path: Path = E1_CONTRACT_PATH,
    *,
    verify_live_assets: bool = True,
    verify_live_profile: bool = False,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read E1 contract: {exc}") from exc
    _require(raw == rendered_json(value), "E1 contract is not canonical rendered JSON")
    return validate_e1_contract(value, verify_live_assets=verify_live_assets, verify_live_profile=verify_live_profile)
