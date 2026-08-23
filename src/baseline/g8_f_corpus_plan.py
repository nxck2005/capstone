"""Metadata-only G8_F corpus-plan derivation for the AM-87 protocol repair.

This module does not load source payloads, decode images, invoke a classifier,
materialize a JPEG 2000 artifact, construct an execution authorization, train,
or run pass two.  It projects already-frozen G8_E candidate metadata onto the
pre-channel codec-artifact identity and freezes the complete training attempt
plan for separate owner audit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT, get

PLAN_SCHEMA_VERSION = 1
QUALITY_SCHEMA_VERSION = 1
PLAN_ROLE = "g8_f_am87_metadata_only_corpus_plan"
PLAN_PREFIX = "g8fcorpusplan-"
QUALITY_PREFIX = "g8fquality-"
AMENDMENT_ID = "AM-87"
DISCOVERY_DATE = "2026-08-23"
PLAN_PATH = REPO_ROOT / "results/baseline/g8_f/corpus_plan.json"

SPEC_PATH = REPO_ROOT / "spec/SPEC.md"
PARAMS_PATH = REPO_ROOT / "spec/params.generated.yaml"
CORPUS_SPEC_PATH = REPO_ROOT / "results/baseline/g8_e/corpus_spec.json"
CANDIDATE_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"
MEASUREMENT_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_authority.json"
PASS_ONE_PATH = REPO_ROOT / "results/baseline/g8_e/pass_one_state.json"
V3S_ROOT = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor"
V3S_CONTRACT_PATH = V3S_ROOT / "measurement_contract.json"
E4_PATH = V3S_ROOT / "runtime/e4_count_derived.json"
E6_PATH = V3S_ROOT / "e6_pass_one_freeze.json"
E7_PATH = V3S_ROOT / "e7_handoff.json"
G8D_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"
G1_PATH = REPO_ROOT / "results/reference_classifier/g1_adjudication.json"
STORAGE_BASIS_PATH = V3S_ROOT / "compute_storage_plan.json"
POST_CAMPAIGN_SOURCE_COMPATIBILITY_PATH = REPO_ROOT / "results/baseline/g8_f/am87_post_campaign_source_compatibility.json"
G8E_SOURCE_COMPATIBILITY_PATH = REPO_ROOT / "results/baseline/g8_f/am87_g8e_source_compatibility.json"
PROJECTION_SOURCE_PATH = Path(__file__).resolve()
GENERATOR_TOOL_PATH = REPO_ROOT / "tools/gen_g8_f_corpus_plan.py"
VERIFIER_TOOL_PATH = REPO_ROOT / "tools/verify_g8_f_corpus_plan.py"

QUALITY_IDENTITY_FIELDS = (
    "dataset",
    "source_codec",
    "payload_budget_bytes",
    "encode_axis_px",
    "codec_configuration_id",
)
PHY_ONLY_OR_LINEAGE_FIELDS = (
    "candidate_id",
    "composition_candidate_identity",
    "ratio",
    "snr_db",
    "modulation",
    "ldpc_rate",
    "packet_config_id",
)
EXPECTED_MODES = (
    "classical_adaptive",
    "classical_fixed_mcs",
    "classical_fixed_mod",
)
PROTECTED_ZERO_COUNTERS = (
    "training",
    "pass_two",
    "pass_three",
    "fallback_invoked",
    "ratio_adjudicated",
    "test_access",
    "learned_system_training",
    "g8_f_execution",
)

# These two commits delimit an observed production window already in Git.  The
# interval is an upper-bound scaling basis, not a new throughput measurement:
# science could start only after authorization publication and completed E2/E3/
# E4 evidence was first published by the closeout commit.
AUTHORIZATION_COMMIT = "493d65608206a840ac80cab0b62cdf204186ebbc"
CLOSEOUT_PUBLICATION_COMMIT = "5d76142abf806cfc850c721b1dfcbaa6ed38f8d5"


class G8FCorpusPlanError(ValueError):
    """A corpus-plan input, identity, schema, or frozen-boundary violation."""


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
        raise G8FCorpusPlanError(f"value is not canonical JSON: {exc}") from None


def rendered_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FCorpusPlanError(f"cannot read {path}: {exc}") from None
    if not isinstance(value, dict):
        raise G8FCorpusPlanError(f"{path} is not a JSON object")
    return value


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FCorpusPlanError(message)


def _identity(prefix: str, payload: Mapping[str, Any]) -> str:
    return prefix + sha256_bytes(canonical_json(payload))


def _codec_configuration_id(codec_binding: Mapping[str, Any]) -> str:
    _require(
        set(codec_binding)
        == {"schema_version", "identity_type", "snapshot", "configuration_hash", "runtime_version"},
        "G8_D codec configuration schema differs",
    )
    _require(codec_binding["identity_type"] == "jpeg2000_configuration", "G8_D codec type differs")
    snapshot = codec_binding["snapshot"]
    _require(isinstance(snapshot, Mapping), "G8_D codec snapshot is not an object")
    _require(
        sha256_bytes(canonical_json(snapshot)) == codec_binding["configuration_hash"],
        "G8_D codec configuration hash does not reproduce",
    )
    return _identity("g8dcodec-", codec_binding)


def project_artifact_quality(
    candidate: Mapping[str, Any],
    structural: Mapping[str, Any],
    *,
    codec_configuration_id: str,
) -> dict[str, Any]:
    """Project one logical BR-4 candidate onto its pre-channel quality.

    The projection deliberately ignores SNR and every PHY field after checking
    that the candidate and structural authority agree.  The payload budget is
    retained: it is the sole way ratio/modulation/rate can affect JPEG 2000.
    """

    for field in ("dataset", "source_codec", "ratio", "encode_axis_px", "modulation", "ldpc_rate", "packet_config_id"):
        _require(candidate.get(field) == structural.get(field), f"candidate/structural {field} differs")
    budget = structural.get("payload_budget_bytes")
    _require(type(budget) is int and budget > 0, "artifact quality has no positive payload budget")
    accounting = structural.get("packet_accounting")
    _require(isinstance(accounting, Mapping), "artifact quality has no packet accounting")
    _require(accounting.get("payload_bytes") == budget, "packet payload differs from structural budget")
    for field in ("reconciles", "channel_reconciles", "channel_uses_exact"):
        _require(accounting.get(field) is True, f"packet accounting {field} is not true")
    _require(candidate.get("source_codec") == "jpeg2000", "foreign source codec in G8_F scope")
    _require(isinstance(codec_configuration_id, str) and codec_configuration_id.startswith("g8dcodec-"), "codec configuration ID differs")
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "identity_type": "g8_f_codec_artifact_quality",
        "dataset": candidate["dataset"],
        "source_codec": candidate["source_codec"],
        "payload_budget_bytes": budget,
        "encode_axis_px": structural["encode_axis_px"],
        "codec_configuration_id": codec_configuration_id,
    }


def quality_id(quality: Mapping[str, Any]) -> str:
    expected = {"schema_version", "identity_type", *QUALITY_IDENTITY_FIELDS}
    _require(set(quality) == expected, "artifact-quality identity schema differs")
    _require(quality["schema_version"] == QUALITY_SCHEMA_VERSION, "artifact-quality schema version differs")
    _require(quality["identity_type"] == "g8_f_codec_artifact_quality", "artifact-quality type differs")
    return _identity(QUALITY_PREFIX, quality)


def _commit_time(commit: str) -> datetime:
    try:
        value = subprocess.run(
            ["git", "show", "-s", "--format=%cI", commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,  # literal-ok: bounded local provenance query
        ).stdout.strip()
        return datetime.fromisoformat(value)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise G8FCorpusPlanError(f"cannot resolve timing-basis commit {commit}: {exc}") from None


def _rule_parameters() -> dict[str, Any]:
    classifier = get("reference_classifier")
    if not isinstance(classifier, Mapping):
        raise G8FCorpusPlanError("reference-classifier parameters are absent")
    names = (
        "artifact_finetune_corpus",
        "artifact_finetune_corpus_split",
        "artifact_finetune_corpus_breadth",
        "artifact_finetune_scope_anchor",
        "artifact_finetune_scope_modes",
        "artifact_finetune_scope_snr_rule",
        "artifact_finetune_candidate_source",
        "artifact_finetune_quality_identity_fields",
        "artifact_finetune_quality_projected_away",
        "artifact_finetune_candidate_admission",
        "artifact_finetune_quality_set_rule",
        "artifact_finetune_deduplication",
        "artifact_finetune_multiplicity",
        "artifact_finetune_traversal_order",
        "artifact_finetune_image_codec_infeasibility",
        "artifact_finetune_non_codec_failure",
        "artifact_finetune_validation_feasibility_role",
    )
    _require(all(name in classifier for name in names), "AM-87 corpus parameters are incomplete")
    result = {name: classifier[name] for name in names}
    result["channel_train_snr_db_fixed"] = get("channel.train_snr_db_fixed")
    _require(tuple(result["artifact_finetune_quality_identity_fields"]) == QUALITY_IDENTITY_FIELDS, "quality identity fields differ from AM-87")
    _require(tuple(result["artifact_finetune_quality_projected_away"]) == PHY_ONLY_OR_LINEAGE_FIELDS, "quality projection exclusions differ from AM-87")
    return result


def _training_rows(manifest_path: Path) -> tuple[list[str], list[int], dict[str, str]]:
    from data.manifests import validate_manifest_bytes

    rows = validate_manifest_bytes("imagenette160", manifest_path.read_bytes())
    by_id = {row.stable_sample_id: row.split for row in rows}
    train = [row for row in rows if row.split == "train"]
    ids = [row.stable_sample_id for row in train]
    labels = [row.label for row in train]
    _require(ids == sorted(ids), "training stable IDs are not in ascending manifest order")
    _require(len(ids) == len(set(ids)), "training stable IDs are duplicated")
    return ids, labels, by_id


def _feasibility_class(delivered: int, codec_infeasible: int, decode_failure: int, total: int) -> str:
    _require(delivered + codec_infeasible + decode_failure == total, "E4 feasibility counts do not cover the denominator")
    _require(decode_failure == 0, "E4 decode failure cannot define a training artifact quality")
    if delivered == total:
        return "all_validation_images_emitted_verified_artifacts"
    if codec_infeasible == total:
        return "all_validation_images_typed_codec_infeasible"
    return "mixed_validation_image_codec_feasibility"


def build_corpus_plan() -> dict[str, Any]:
    """Reproduce the exact metadata-only corpus plan from frozen authorities."""

    rule_parameters = _rule_parameters()
    train_snr = rule_parameters["channel_train_snr_db_fixed"]
    _require(not isinstance(train_snr, bool) and isinstance(train_snr, int | float), "training SNR is not numeric")

    corpus_spec = _read_object(CORPUS_SPEC_PATH)
    candidate_authority = _read_object(CANDIDATE_AUTHORITY_PATH)
    measurement_authority = _read_object(MEASUREMENT_AUTHORITY_PATH)
    pass_one = _read_object(PASS_ONE_PATH)
    v3s_contract = _read_object(V3S_CONTRACT_PATH)
    e4 = _read_object(E4_PATH)
    e6 = _read_object(E6_PATH)
    e7 = _read_object(E7_PATH)
    g8d_contract = _read_object(G8D_CONTRACT_PATH)
    g1 = _read_object(G1_PATH)
    storage_basis = _read_object(STORAGE_BASIS_PATH)

    counters = e7.get("counters")
    _require(isinstance(counters, Mapping), "E7 protected counters are absent")
    _require(counters.get("pass_one_executed_count") == 1, "pass one was not executed exactly once")
    _require(all(counters.get(name) == 0 for name in PROTECTED_ZERO_COUNTERS), "a protected E7 counter is nonzero")
    _require(e7.get("g8_f", {}).get("authorized") is False, "G8_F is already authorized")
    _require(e7.get("g8_f", {}).get("execution_count") == 0, "G8_F execution is nonzero")
    _require(e6.get("corpus_specification_binding", {}).get("materialized") is False, "E6 says the corpus is materialized")
    _require(e6.get("corpus_specification_binding", {}).get("materialized_object_count") == 0, "E6 materialized count is nonzero")
    _require(corpus_spec.get("materialized") is False and corpus_spec.get("materialized_object_count") == 0, "E1 corpus spec is not unopened")

    candidate_rows = candidate_authority.get("candidates")
    structural_rows = measurement_authority.get("structural_identities")
    mapping = measurement_authority.get("logical_candidate_to_structural_id")
    _require(isinstance(candidate_rows, Sequence), "candidate authority rows are absent")
    _require(isinstance(structural_rows, Sequence), "measurement structural rows are absent")
    _require(isinstance(mapping, Mapping), "logical-to-structural mapping is absent")
    candidates = {row["candidate_id"]: row for row in candidate_rows}
    structurals = {row["structural_identity_id"]: row for row in structural_rows}
    _require(len(candidates) == candidate_authority.get("candidate_count"), "candidate authority count differs")
    _require(len(structurals) == measurement_authority.get("counts", {}).get("structural_all_roles"), "structural authority count differs")

    modes = tuple(sorted(call["mode"] for call in pass_one.get("calls", ())))
    _require(set(modes) == set(EXPECTED_MODES) and len(modes) == len(EXPECTED_MODES) * len(get("bandwidth.ratios")), "pass-one mode/ratio calls differ")
    anchor_ids: list[str] = []
    scope_pairs: set[tuple[str, str]] = set()
    for call in pass_one["calls"]:
        for cell in call["per_snr"]:
            if float(cell["snr_db"]) > float(train_snr):
                continue
            candidate_id = cell["authority_candidate_id"]
            _require(candidate_id in candidates, "pass one references a foreign candidate")
            candidate = candidates[candidate_id]
            _require(float(candidate["snr_db"]) == float(cell["snr_db"]), "pass-one candidate SNR differs")
            anchor_ids.append(candidate_id)
            scope_pairs.add((candidate["dataset"], candidate["ratio"]))
    _require(anchor_ids, "pass-one scope anchors are empty")
    _require(len(anchor_ids) == len(set((index, value) for index, value in enumerate(anchor_ids))), "pass-one anchor traversal is malformed")

    codec_binding = g8d_contract.get("codec_binding")
    _require(isinstance(codec_binding, Mapping), "G8_D codec binding is absent")
    codec_configuration_id = _codec_configuration_id(codec_binding)
    production_codec = v3s_contract.get("codec")
    _require(isinstance(production_codec, Mapping), "G8_E production codec binding is absent")
    _require(production_codec.get("runtime_identity") == codec_binding["runtime_version"], "G8_D/G8_E codec runtime differs")

    region_candidates = [
        row
        for row in candidate_rows
        if (row["dataset"], row["ratio"]) in scope_pairs
        and float(row["snr_db"]) <= float(train_snr)
    ]
    _require(region_candidates, "authority has no candidates in the anchored scope")
    projected: dict[str, dict[str, Any]] = {}
    source_structurals: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_candidate_counts: dict[str, int] = defaultdict(int)
    for candidate in region_candidates:
        structural_id = mapping.get(candidate["candidate_id"])
        _require(structural_id in structurals, "candidate has no frozen structural mapping")
        structural = structurals[structural_id]
        quality = project_artifact_quality(
            candidate,
            structural,
            codec_configuration_id=codec_configuration_id,
        )
        qid = quality_id(quality)
        if qid in projected:
            _require(projected[qid] == quality, "artifact-quality ID collision")
        projected[qid] = quality
        source_candidate_counts[qid] += 1
        source_structurals[qid][structural_id] = {
            "structural_identity_id": structural_id,
            "ratio": structural["ratio"],
            "modulation": structural["modulation"],
            "ldpc_rate": structural["ldpc_rate"],
            "packet_config_id": structural["packet_config_id"],
        }

    e4_objects = {obj["measurement_identity_id"]: obj for obj in e4.get("objects", ())}
    initial_structural_ids = {
        structural_id
        for structural_id, structural in structurals.items()
        if any(dataset == structural["dataset"] for dataset, _ratio in scope_pairs)
    }
    _require(initial_structural_ids == set(e4_objects), "E4 does not exactly cover the scoped structural authority")

    qualities: list[dict[str, Any]] = []
    feasibility_counts: dict[str, int] = defaultdict(int)
    validation_expected_objects = 0
    for qid in sorted(projected):
        aliases = sorted(source_structurals[qid].values(), key=lambda item: item["structural_identity_id"])
        observations = [e4_objects[item["structural_identity_id"]] for item in aliases]
        observed_tuples = {
            (
                obj["delivered_count"],
                obj["codec_infeasibility_count"],
                obj["decode_failure_count"],
                obj["total_count"],
                obj["status"],
            )
            for obj in observations
        }
        _require(len(observed_tuples) == 1, "PHY aliases disagree on validation codec feasibility")
        delivered, codec_infeasible, decode_failure, total, status = next(iter(observed_tuples))
        _require(status == "eligible", "E4 artifact quality is not selection-eligible")
        feasibility = _feasibility_class(delivered, codec_infeasible, decode_failure, total)
        feasibility_counts[feasibility] += 1
        qualities.append(
            {
                "quality_id": qid,
                "identity": projected[qid],
                "source_structural_identities": aliases,
                "source_structural_identity_count": len(aliases),
                "source_region_logical_candidate_count": source_candidate_counts[qid],
                "validation_feasibility_audit": {
                    "classification": feasibility,
                    "delivered_count": delivered,
                    "codec_infeasibility_count": codec_infeasible,
                    "decode_failure_count": decode_failure,
                    "denominator": total,
                    "membership_effect": "none_estimate_and_audit_only",
                },
            }
        )

    manifest_path = REPO_ROOT / corpus_spec["train_manifest"]["path"]
    training_ids, training_labels, split_by_id = _training_rows(manifest_path)
    _require(len(training_ids) == corpus_spec["train_manifest"]["expected_count"], "training count differs from E1 corpus spec")
    _require(sha256_bytes(canonical_json(training_ids)) == corpus_spec["train_manifest"]["expected_stable_id_set_sha256"], "training stable-ID digest differs from E1 corpus spec")
    _require(all(split_by_id[value] == "train" for value in training_ids), "non-training stable ID entered the plan")

    quality_ids = [row["quality_id"] for row in qualities]
    _require(quality_ids == sorted(quality_ids) and len(quality_ids) == len(set(quality_ids)), "quality IDs are not ascending unique")
    attempt_count = len(quality_ids) * len(training_ids)
    validation_expected_objects = sum(
        len(training_ids)
        * row["validation_feasibility_audit"]["delivered_count"]
        // row["validation_feasibility_audit"]["denominator"]
        for row in qualities
    )

    estimated = storage_basis["estimated_bytes"]
    projected_files = storage_basis["projected_files"]
    physical_jobs = projected_files["backend_j2k_cache"]
    _require(physical_jobs == projected_files["v3_codec_cache"] == projected_files["reconstruction_cache"], "G8_E storage basis has unequal physical-object counts")
    bytes_per_materialized = (
        storage_basis["basis"]["backend_j2k_representative_bytes"]
        + estimated["v3_codec_cache"] // physical_jobs
        + estimated["reconstruction_cache"] // physical_jobs
    )
    start = _commit_time(AUTHORIZATION_COMMIT)
    end = _commit_time(CLOSEOUT_PUBLICATION_COMMIT)
    window_seconds = int((end - start).total_seconds())
    _require(window_seconds > 0, "confessor timing basis is non-positive")

    rule_sha = sha256_bytes(canonical_json(rule_parameters))
    quality_order_sha = sha256_bytes(canonical_json(quality_ids))
    training_order_sha = sha256_bytes(canonical_json(training_ids))
    traversal_basis = {
        "outer": "quality_id_ascending",
        "inner": "training_stable_id_ascending",
        "quality_order_sha256": quality_order_sha,
        "training_order_sha256": training_order_sha,
    }
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "artifact_role": PLAN_ROLE,
        "phase": "G8_F",
        "checkpoint": "F0_PROTOCOL_PLAN_ONLY",
        "status": "FROZEN_AWAITING_OWNER_AUDIT_AND_SEPARATE_F0_AUTHORIZATION",
        "amendment": {
            "id": AMENDMENT_ID,
            "discovery_date": DISCOVERY_DATE,
            "timing": "post_pass_one_pre_g8f_execution",
            "specification": _binding(SPEC_PATH),
            "generated_parameters": _binding(PARAMS_PATH),
            "rule_parameters": rule_parameters,
            "rule_parameters_sha256": rule_sha,
        },
        "frozen_bindings": {
            "plan_implementation": {
                "projection_rule_version": "g8_f_artifact_quality_projection_v1",
                "projection_source": _binding(PROJECTION_SOURCE_PATH),
                "generator_tool": _binding(GENERATOR_TOOL_PATH),
                "verifier_tool": _binding(VERIFIER_TOOL_PATH),
            },
            "post_campaign_source_compatibility": {
                **_binding(POST_CAMPAIGN_SOURCE_COMPATIBILITY_PATH),
                "compatibility_id": _read_object(POST_CAMPAIGN_SOURCE_COMPATIBILITY_PATH)["compatibility_id"],
                "purpose": "preserve_completed_g8c_g8e_verification_under_exact_am87_off_measurement_path_parameter_change",
            },
            "g8e_source_compatibility": {
                **_binding(G8E_SOURCE_COMPATIBILITY_PATH),
                "compatibility_id": _read_object(G8E_SOURCE_COMPATIBILITY_PATH)["compatibility_id"],
                "purpose": "preserve_frozen_g8d_identity_and_g8e_source_verification_after_am87",
            },
            "e7_handoff": {**_binding(E7_PATH), "handoff_id": e7["handoff_id"]},
            "e6_freeze": {**_binding(E6_PATH), "freeze_id": e6["e6_freeze_id"]},
            "e1_corpus_spec": {**_binding(CORPUS_SPEC_PATH), "corpus_spec_id": corpus_spec["corpus_spec_id"], "preserved_unchanged": True},
            "pass_one": {
                **_binding(PASS_ONE_PATH),
                "state_id": pass_one["state_id"],
                "state_content_sha256": pass_one["state_sha256"],
                "used_only_for_original_scope_anchor": True,
            },
            "candidate_authority": {**_binding(CANDIDATE_AUTHORITY_PATH), "authority_id": candidate_authority["authority_id"], "authority_digest": candidate_authority["candidate_authority_digest"]},
            "measurement_authority": {**_binding(MEASUREMENT_AUTHORITY_PATH), "authority_id": measurement_authority["authority_id"], "structural_digest": measurement_authority["structural_digest"]},
            "e4_feasibility_audit": {**_binding(E4_PATH), "e4_id": e4["e4_id"], "membership_effect": "none"},
            "g1_clean_classifier": {**_binding(G1_PATH), "checkpoint_sha256": g1["checkpoint_sha256"], "config_hash": g1["config_hash"], "classifier_variant": g1["classifier_variant"]},
            "train_manifest": {**_binding(manifest_path), "dataset_archive_sha256": corpus_spec["train_manifest"]["archive_sha256"]},
            "g8d_codec_configuration": {**_binding(G8D_CONTRACT_PATH), "codec_configuration_id": codec_configuration_id, "configuration_hash": codec_binding["configuration_hash"], "runtime_version": codec_binding["runtime_version"]},
            "g8e_physical_cache_namespace": {**_binding(V3S_CONTRACT_PATH), "configuration_hash": production_codec["configuration_hash"], "runtime_identity": production_codec["runtime_identity"]},
        },
        "scope": {
            "anchor_rule": rule_parameters["artifact_finetune_scope_anchor"],
            "anchor_reference_count": len(anchor_ids),
            "anchor_reference_order_sha256": sha256_bytes(canonical_json(anchor_ids)),
            "dataset_ratio_scope": [
                {"dataset": dataset, "ratio": ratio}
                for dataset, ratio in sorted(scope_pairs)
            ],
            "modes": list(EXPECTED_MODES),
            "snr_rule": rule_parameters["artifact_finetune_scope_snr_rule"],
            "snr_upper_inclusive_db": train_snr,
            "region_logical_candidate_count": len(region_candidates),
            "region_structural_identity_count": len({item["structural_identity_id"] for values in source_structurals.values() for item in values.values()}),
            "pass_one_scores_ranks_margins_used_for_membership": False,
        },
        "artifact_quality_projection": {
            "rule_version": "g8_f_artifact_quality_projection_v1",
            "identity_fields": list(QUALITY_IDENTITY_FIELDS),
            "projected_away_fields": list(PHY_ONLY_OR_LINEAGE_FIELDS),
            "deduplication": rule_parameters["artifact_finetune_deduplication"],
            "ordering": "quality_id_ascending",
            "quality_count": len(qualities),
            "quality_order_sha256": quality_order_sha,
            "qualities": qualities,
        },
        "training_membership": {
            "dataset": "imagenette160",
            "split": "train",
            "stable_id_count": len(training_ids),
            "stable_id_order": "ascending_manifest_order",
            "stable_id_order_sha256": training_order_sha,
            "stable_id_label_order_sha256": sha256_bytes(canonical_json(list(zip(training_ids, training_labels, strict=True)))),
            "stable_ids": training_ids,
            "validation_ids_forbidden": True,
            "test_ids_forbidden": True,
            "validation_or_test_fallback": False,
        },
        "multiplicity_and_feasibility": {
            "multiplicity_rule": rule_parameters["artifact_finetune_multiplicity"],
            "traversal": traversal_basis,
            "traversal_basis_sha256": sha256_bytes(canonical_json(traversal_basis)),
            "exact_attempt_count": attempt_count,
            "materialized_object_count_rule": "sum_one_for_each_attempt_returning_a_verified_decoded_reconstruction",
            "materialized_object_count_is_variable_until_f1": True,
            "materialized_object_count_minimum": 0,
            "materialized_object_count_maximum": attempt_count,
            "typed_codec_infeasibility_rule": rule_parameters["artifact_finetune_image_codec_infeasibility"],
            "typed_codec_infeasibility_effect": "omit_exact_pair_record_quality_and_class_coverage_no_substitution",
            "structural_feasibility_rule": rule_parameters["artifact_finetune_candidate_admission"],
            "positive_codec_payload_required": True,
            "configuration_level_codec_feasibility_inferred_from_validation": False,
            "validation_feasibility_role": rule_parameters["artifact_finetune_validation_feasibility_role"],
            "validation_feasibility_quality_counts": dict(sorted(feasibility_counts.items())),
            "validation_incidence_projected_object_estimate": validation_expected_objects,
            "other_failure_rule": rule_parameters["artifact_finetune_non_codec_failure"],
            "outage_image_substitution": False,
            "neighbour_quality_substitution": False,
        },
        "compute_consequence": {
            "storage_basis": _binding(STORAGE_BASIS_PATH),
            "estimated_bytes_per_materialized_pair": bytes_per_materialized,
            "maximum_estimated_bytes": attempt_count * bytes_per_materialized,
            "validation_incidence_estimated_bytes": validation_expected_objects * bytes_per_materialized,
            "maximum_with_25_percent_safety_bytes": attempt_count * bytes_per_materialized + (attempt_count * bytes_per_materialized) // 4,  # literal-ok: arithmetic 25-percent safety denominator
            "confessor_wall_time_basis": {
                "kind": "conservative_git_publication_window_scaling_not_new_measurement",
                "authorization_commit": AUTHORIZATION_COMMIT,
                "authorization_commit_time": start.isoformat(),
                "closeout_publication_commit": CLOSEOUT_PUBLICATION_COMMIT,
                "closeout_publication_commit_time": end.isoformat(),
                "observed_window_seconds": window_seconds,
                "source_physical_jobs": physical_jobs,
            },
            "maximum_scaled_confessor_seconds": window_seconds * attempt_count / physical_jobs,
            "validation_incidence_scaled_confessor_seconds": window_seconds * validation_expected_objects / physical_jobs,
        },
        "protected_boundary": {
            "plan_only": True,
            "owner_audit_required": True,
            "f0_execution_authorized": False,
            "corpus_materialized": False,
            "materialized_object_count": 0,
            "classifier_inference_performed": False,
            "optimizer_steps": 0,
            "pass_two": 0,
            "test_access": 0,
            "confessor_started": False,
            "next_action": "OWNER_AUDIT_OF_CORRECTED_G8_F_CORPUS_PLAN_AND_SEPARATE_F0_AUTHORIZATION",
            "e7_counters": dict(counters),
        },
    }
    body["plan_id"] = _identity(PLAN_PREFIX, body)
    return body


def verify_plan_value(value: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate internal identities, protected scope, then independently rebuild."""

    _require(isinstance(value, Mapping), "corpus plan is not an object")
    _require(value.get("schema_version") == PLAN_SCHEMA_VERSION and value.get("artifact_role") == PLAN_ROLE, "corpus-plan header differs")
    body = dict(value)
    plan_id = body.pop("plan_id", None)
    _require(plan_id == _identity(PLAN_PREFIX, body), "corpus-plan ID does not reproduce")

    projection = value.get("artifact_quality_projection")
    membership = value.get("training_membership")
    boundary = value.get("protected_boundary")
    multiplicity = value.get("multiplicity_and_feasibility")
    _require(all(isinstance(item, Mapping) for item in (projection, membership, boundary, multiplicity)), "corpus-plan sections are absent")
    quality_rows = projection["qualities"]
    quality_ids = [row["quality_id"] for row in quality_rows]
    _require(quality_ids == sorted(quality_ids) and len(quality_ids) == len(set(quality_ids)), "quality order is not ascending unique")
    _require(all(row["quality_id"] == quality_id(row["identity"]) for row in quality_rows), "projected artifact-quality identity differs")
    _require(projection["quality_count"] == len(quality_ids), "quality count differs")
    _require(projection["quality_order_sha256"] == sha256_bytes(canonical_json(quality_ids)), "quality-order digest differs")

    training_ids = membership["stable_ids"]
    _require(training_ids == sorted(training_ids) and len(training_ids) == len(set(training_ids)), "training stable IDs are not ascending unique")
    _require(membership["stable_id_count"] == len(training_ids), "training stable-ID count differs")
    _require(membership["stable_id_order_sha256"] == sha256_bytes(canonical_json(training_ids)), "training stable-ID digest differs")
    manifest_path = REPO_ROOT / value["frozen_bindings"]["train_manifest"]["path"]
    current_train, _labels, by_id = _training_rows(manifest_path)
    _require(training_ids == current_train, "training stable IDs are not the complete frozen train split")
    _require(all(by_id.get(stable_id) == "train" for stable_id in training_ids), "validation or test stable ID entered the corpus plan")

    _require(multiplicity["exact_attempt_count"] == len(quality_ids) * len(training_ids), "corpus multiplicity differs")
    _require(multiplicity["outage_image_substitution"] is False and multiplicity["neighbour_quality_substitution"] is False, "corpus substitution was enabled")
    _require(boundary["plan_only"] is True and boundary["f0_execution_authorized"] is False, "plan opened G8_F execution")
    _require(boundary["corpus_materialized"] is False and boundary["materialized_object_count"] == 0, "plan claims materialized artifacts")
    _require(boundary["optimizer_steps"] == boundary["pass_two"] == boundary["test_access"] == 0, "protected corpus-plan counter is nonzero")
    _require(boundary["confessor_started"] is False, "corpus plan claims confessor was started")

    rebuilt = dict(build_corpus_plan() if expected is None else expected)
    _require(dict(value) == rebuilt, "corpus plan differs from independent frozen-input reproduction")
    return dict(value)


def verify_corpus_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FCorpusPlanError(f"cannot read corpus plan: {exc}") from None
    _require(raw == rendered_json(value), "corpus-plan file is not canonical rendered JSON")
    return verify_plan_value(value)
