"""AM-88 metadata-only balanced sampler for the G8_F training corpus.

AM-87 remains the immutable definition of legitimate JPEG 2000 artifact-quality
support.  This module samples training-image × supported-quality assignments; it
never reads image payloads, runs a codec or classifier, authorizes F0, trains,
runs pass two, or accesses test through the guarded boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT, get
from baseline.g8_campaign import G8ContractError, _load_am89_compatibility
from baseline.g8_f_corpus_plan import project_artifact_quality, quality_id

SCHEMA_VERSION = 1
ARTIFACT_ROLE = "g8_f_am88_metadata_only_balanced_sampler_plan"
PLAN_PREFIX = "g8fsamplerplan-"
AMENDMENT = "AM-88"
DISCOVERY_DATE = "2026-08-24"
PLAN_PATH = REPO_ROOT / "results/baseline/g8_f/am88_sampler_plan.json"
AM87_PLAN_PATH = REPO_ROOT / "results/baseline/g8_f/corpus_plan.json"
AM87_PLAN_ID = "g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148"
AM87_PLAN_FILE_SHA256 = "733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c"
MANIFEST_PATH = REPO_ROOT / "data/manifests/imagenette160.csv"
SPEC_PATH = REPO_ROOT / "spec/SPEC.md"
PARAMS_PATH = REPO_ROOT / "spec/params.generated.yaml"
CANDIDATE_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"
MEASUREMENT_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/e1_corrected/measurement_authority.json"
PASS_ONE_PATH = REPO_ROOT / "results/baseline/g8_e/pass_one_state.json"
G8D_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"
E7_PATH = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
STORAGE_BASIS_PATH = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor/compute_storage_plan.json"
AM88_POST_COMPATIBILITY_PATH = REPO_ROOT / "results/baseline/g8_f/am88_post_campaign_source_compatibility.json"
AM88_G8E_COMPATIBILITY_PATH = REPO_ROOT / "results/baseline/g8_f/am88_g8e_source_compatibility.json"
SOURCE_PATH = Path(__file__).resolve()
GENERATOR_PATH = REPO_ROOT / "tools/gen_g8_f_sampler_plan.py"
VERIFIER_PATH = REPO_ROOT / "tools/verify_g8_f_sampler_plan.py"

EXPECTED_QUALITY_COUNT = 120  # literal-ok: immutable AM-87 support count
EXPECTED_TRAINING_COUNT = 8469  # literal-ok: frozen Imagenette training count
EXPECTED_VARIANTS = 6  # literal-ok: AM-88 fixed variants per training image
EXPECTED_ATTEMPTS = 50_814
EXPECTED_SEED = "am88-g8f-balanced-sampler-20260824-v1"
EXPECTED_VERSION = "g8_f_balanced_sampler_v1"
EXPECTED_ALGORITHM = "sha256_keyed_stable_id_order_global_quality_permutation_class_chunks_cyclic_v1"
EXPECTED_BALANCE_RULE = "concatenate_class_label_ordered_attempt_chunks_over_one_seed_permuted_quality_cycle"
PROTECTED_ZERO_COUNTERS = (
    "g8_f_execution", "training", "pass_two", "pass_three", "fallback_invoked",
    "ratio_adjudicated", "test_access", "learned_system_training",
)


class G8FSamplerPlanError(ValueError):
    """A frozen input, sampler invariant, or protected boundary differs."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise G8FSamplerPlanError(f"value is not canonical JSON: {exc}") from None


def rendered_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FSamplerPlanError(message)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FSamplerPlanError(f"cannot read {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    relative = str(path.relative_to(REPO_ROOT))
    if relative in {
        "spec/SPEC.md", "spec/params.generated.yaml", "src/baseline/g8_f_sampler_plan.py",
        "tools/verify_g8_f_sampler_plan.py",
    }:
        try:
            compatibility = _load_am89_compatibility()
        except G8ContractError as exc:
            raise G8FSamplerPlanError(f"AM-89 sampler compatibility differs: {exc}") from None
        matches = [
            entry for entry in compatibility.get("entries", [])
            if isinstance(entry, Mapping) and entry.get("path") == relative
        ]
        _require(len(matches) == 1, f"AM-89 sampler compatibility omits {relative}")
        entry = matches[0]
        _require(
            entry.get("current_bytes") == len(raw)
            and entry.get("current_sha256") == sha256_bytes(raw),
            f"AM-89 sampler current bytes differ: {relative}",
        )
        return {
            "path": relative,
            "bytes": entry["archived_bytes"],
            "sha256": entry["archived_sha256"],
        }
    return {"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _identity(prefix: str, body: Mapping[str, Any]) -> str:
    return prefix + sha256_bytes(canonical_json(body))


def _key(seed: str, domain: str, value: Any) -> str:
    return sha256_bytes(canonical_json({"seed": seed, "domain": domain, "value": value}))


def _seed_order(values: Iterable[str], *, seed: str, domain: str) -> list[str]:
    return sorted(values, key=lambda value: (_key(seed, domain, value), value))


def _load_am87_support() -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    raw = AM87_PLAN_PATH.read_bytes()
    _require(sha256_bytes(raw) == AM87_PLAN_FILE_SHA256, "AM-87 support-plan file bytes differ")
    plan = json.loads(raw)
    _require(isinstance(plan, dict) and raw == rendered_json(plan), "AM-87 support plan is not canonical rendered JSON")
    body = dict(plan)
    plan_id = body.pop("plan_id", None)
    _require(plan_id == AM87_PLAN_ID == _identity("g8fcorpusplan-", body), "AM-87 support-plan identity differs")
    rows = plan.get("artifact_quality_projection", {}).get("qualities")
    _require(isinstance(rows, list), "AM-87 quality support is absent")
    quality_ids = [row.get("quality_id") for row in rows]
    _require(len(quality_ids) == EXPECTED_QUALITY_COUNT and quality_ids == sorted(set(quality_ids)), "AM-87 quality order/count differs")
    _require(all(row.get("quality_id") == quality_id(row.get("identity", {})) for row in rows), "AM-87 quality identity does not reproduce")
    by_id = {row["quality_id"]: row for row in rows}
    return plan, quality_ids, by_id


def _derive_am87_support_ids() -> list[str]:
    """Independently re-project frozen pre-F1 authorities onto AM-87 support."""

    candidates_value = _read(CANDIDATE_AUTHORITY_PATH)
    measurement = _read(MEASUREMENT_AUTHORITY_PATH)
    pass_one = _read(PASS_ONE_PATH)
    g8d = _read(G8D_CONTRACT_PATH)
    candidates = candidates_value.get("candidates")
    structurals = measurement.get("structural_identities")
    mapping = measurement.get("logical_candidate_to_structural_id")
    _require(isinstance(candidates, list) and isinstance(structurals, list) and isinstance(mapping, dict), "AM-87 frozen authorities are malformed")
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    structural_by_id = {row["structural_identity_id"]: row for row in structurals}
    train_snr = get("channel.train_snr_db_fixed")
    scope: set[tuple[str, str]] = set()
    for call in pass_one.get("calls", []):
        for cell in call.get("per_snr", []):
            if float(cell["snr_db"]) <= float(train_snr):
                candidate = candidate_by_id.get(cell["authority_candidate_id"])
                _require(isinstance(candidate, dict), "pass one references foreign authority")
                scope.add((candidate["dataset"], candidate["ratio"]))
    _require(scope, "AM-87 scope is empty")
    codec = g8d.get("codec_binding")
    _require(isinstance(codec, dict), "G8_D codec binding is absent")
    codec_id = "g8dcodec-" + sha256_bytes(canonical_json(codec))
    projected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if (candidate["dataset"], candidate["ratio"]) not in scope or float(candidate["snr_db"]) > float(train_snr):
            continue
        structural = structural_by_id.get(mapping.get(candidate["candidate_id"]))
        _require(isinstance(structural, dict), "AM-87 candidate has no structural mapping")
        identity = project_artifact_quality(candidate, structural, codec_configuration_id=codec_id)
        qid = quality_id(identity)
        _require(qid not in projected or projected[qid] == identity, "AM-87 projected quality collision")
        projected[qid] = identity
    return sorted(projected)


def _training_membership() -> tuple[list[str], dict[str, int], dict[int, list[str]], dict[str, str]]:
    from data.manifests import validate_manifest_bytes

    rows = validate_manifest_bytes("imagenette160", MANIFEST_PATH.read_bytes())
    split_by_id = {row.stable_sample_id: row.split for row in rows}
    train_rows = [row for row in rows if row.split == "train"]
    ids = [row.stable_sample_id for row in train_rows]
    labels = {row.stable_sample_id: row.label for row in train_rows}
    _require(ids == sorted(ids) and len(ids) == len(set(ids)) == EXPECTED_TRAINING_COUNT, "frozen training membership differs")
    by_class: dict[int, list[str]] = defaultdict(list)
    for stable_id in ids:
        by_class[labels[stable_id]].append(stable_id)
    return ids, labels, dict(sorted(by_class.items())), split_by_id


def derive_assignments(
    quality_ids: Sequence[str],
    ids_by_class: Mapping[int, Sequence[str]],
    *,
    seed: str,
    variants_per_image: int,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Apply the frozen cyclic construction; no runtime outcome is an input."""

    _require(len(quality_ids) == len(set(quality_ids)) and variants_per_image <= len(quality_ids), "quality support cannot supply distinct assignments")
    quality_cycle = _seed_order(quality_ids, seed=seed, domain="am88_quality_permutation")
    pairs: list[tuple[str, str]] = []
    cursor = 0
    for label in sorted(ids_by_class):
        image_order = _seed_order(ids_by_class[label], seed=seed, domain=f"am88_training_ids_class_{label}")
        for stable_id in image_order:
            assigned = [quality_cycle[(cursor + slot) % len(quality_cycle)] for slot in range(variants_per_image)]
            _require(len(assigned) == len(set(assigned)), "one image received duplicate qualities")
            pairs.extend((stable_id, qid) for qid in assigned)
            cursor += variants_per_image
    return pairs, quality_cycle


def _pair_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return sha256_bytes(canonical_json([[stable_id, qid] for stable_id, qid in pairs]))


def _summary(
    pairs: Sequence[tuple[str, str]],
    quality_ids: Sequence[str],
    labels: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, Any]]:
    global_counts = Counter(qid for _stable_id, qid in pairs)
    class_counts: dict[int, Counter[str]] = {label: Counter() for label in sorted(set(labels.values()))}
    image_counts: Counter[str] = Counter()
    for stable_id, qid in pairs:
        image_counts[stable_id] += 1
        class_counts[labels[stable_id]][qid] += 1
    global_exact = {qid: global_counts[qid] for qid in quality_ids}
    class_exact = {str(label): {qid: counts[qid] for qid in quality_ids} for label, counts in class_counts.items()}
    global_values = list(global_exact.values())
    class_ranges = {
        str(label): {
            "attempts": sum(counts.values()),
            "minimum_per_quality": min(counts.values()),
            "maximum_per_quality": max(counts.values()),
            "range": max(counts.values()) - min(counts.values()),
        }
        for label, counts in class_exact.items()
    }
    balance = {
        "global": {
            "attempts": len(pairs),
            "minimum_per_quality": min(global_values),
            "maximum_per_quality": max(global_values),
            "range": max(global_values) - min(global_values),
            "arithmetic_minimum_range": 0 if len(pairs) % len(quality_ids) == 0 else 1,
        },
        "per_class": class_ranges,
        "per_image_minimum": min(image_counts.values()),
        "per_image_maximum": max(image_counts.values()),
    }
    return global_exact, class_exact, balance


def _sampler_parameters() -> dict[str, Any]:
    classifier = get("reference_classifier")
    names = (
        "artifact_finetune_support_plan", "artifact_finetune_sampling_plan",
        "artifact_finetune_sampler_version", "artifact_finetune_sampler_seed",
        "artifact_finetune_assignment_algorithm", "artifact_finetune_variants_per_training_image",
        "artifact_finetune_assignment_information", "artifact_finetune_assignment_forbidden_information",
        "artifact_finetune_balance_rule", "artifact_finetune_multiplicity",
        "artifact_finetune_traversal_order", "artifact_finetune_image_codec_infeasibility",
        "artifact_finetune_non_codec_failure",
    )
    _require(isinstance(classifier, Mapping) and all(name in classifier for name in names), "AM-88 sampler parameters are incomplete")
    value = {name: classifier[name] for name in names}
    _require(value["artifact_finetune_sampler_version"] == EXPECTED_VERSION, "sampler version differs")
    _require(value["artifact_finetune_sampler_seed"] == EXPECTED_SEED, "sampler seed differs")
    _require(value["artifact_finetune_assignment_algorithm"] == EXPECTED_ALGORITHM, "assignment algorithm differs")
    _require(value["artifact_finetune_variants_per_training_image"] == EXPECTED_VARIANTS, "variants per image differs")
    _require(value["artifact_finetune_balance_rule"] == EXPECTED_BALANCE_RULE, "balance rule differs")
    forbidden = value["artifact_finetune_assignment_forbidden_information"]
    _require(isinstance(forbidden, list) and len(forbidden) == len(set(forbidden)), "forbidden assignment information differs")
    return value


def build_sampler_plan() -> dict[str, Any]:
    """Build compact AM-88 evidence from frozen metadata only."""

    parameters = _sampler_parameters()
    am87, quality_ids, quality_rows = _load_am87_support()
    _require(_derive_am87_support_ids() == quality_ids, "AM-87 support no longer reproduces from frozen authorities")
    training_ids, labels, ids_by_class, split_by_id = _training_membership()
    _require(training_ids == am87["training_membership"]["stable_ids"], "AM-87/current frozen training IDs differ")
    _require(all(split_by_id[stable_id] == "train" for stable_id in training_ids), "non-training ID entered AM-88")
    pairs, quality_cycle = derive_assignments(quality_ids, ids_by_class, seed=EXPECTED_SEED, variants_per_image=EXPECTED_VARIANTS)
    _require(len(pairs) == EXPECTED_ATTEMPTS == EXPECTED_TRAINING_COUNT * EXPECTED_VARIANTS, "nominal attempt count differs")
    _require(len(set(pairs)) == len(pairs), "duplicate image-quality pair")
    per_image = Counter(stable_id for stable_id, _qid in pairs)
    _require(set(per_image) == set(training_ids) and set(per_image.values()) == {EXPECTED_VARIANTS}, "per-image assignment coverage differs")
    _require({qid for _stable_id, qid in pairs} == set(quality_ids), "not every AM-87 quality is represented")
    global_counts, class_counts, balance = _summary(pairs, quality_ids, labels)
    _require(balance["global"]["range"] == balance["global"]["arithmetic_minimum_range"] == 1, "global quality balance differs")
    _require(all(value["range"] <= 1 for value in balance["per_class"].values()), "within-class quality balance differs")

    e7 = _read(E7_PATH)
    counters = e7.get("counters")
    _require(isinstance(counters, Mapping) and counters.get("pass_one_executed_count") == 1, "pass one boundary differs")
    _require(all(counters.get(name) == 0 for name in PROTECTED_ZERO_COUNTERS), "protected E7 counter is nonzero")
    _require(e7.get("g8_f", {}).get("authorized") is False and e7.get("g8_f", {}).get("execution_count") == 0, "G8_F is already authorized or executed")

    sorted_pairs = sorted(pairs)
    ordered_digest = _pair_digest(pairs)
    set_digest = _pair_digest(sorted_pairs)
    label_pairs = [[stable_id, labels[stable_id]] for stable_id in training_ids]
    delivered_quality_ids = {
        qid for qid, row in quality_rows.items()
        if row["validation_feasibility_audit"]["classification"] == "all_validation_images_emitted_verified_artifacts"
    }
    incidence_objects = sum(global_counts[qid] for qid in delivered_quality_ids)
    storage = _read(STORAGE_BASIS_PATH)
    physical_jobs = storage["projected_files"]["backend_j2k_cache"]
    bytes_per_pair = (
        storage["basis"]["backend_j2k_representative_bytes"]
        + storage["estimated_bytes"]["v3_codec_cache"] // physical_jobs
        + storage["estimated_bytes"]["reconstruction_cache"] // physical_jobs
    )
    old_window = am87["compute_consequence"]["confessor_wall_time_basis"]
    window_seconds = old_window["observed_window_seconds"]
    source_jobs = old_window["source_physical_jobs"]

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "phase": "G8_F",
        "checkpoint": "F0_AM88_PROTOCOL_PLAN_ONLY",
        "status": "FROZEN_AWAITING_OWNER_AUDIT_AND_SEPARATE_F0_AUTHORIZATION",
        "amendment": {
            "id": AMENDMENT,
            "discovery_date": DISCOVERY_DATE,
            "timing": "post_pass_one_post_am87_pre_f0_execution_zero",
            "specification": _binding(SPEC_PATH),
            "generated_parameters": _binding(PARAMS_PATH),
            "sampler_parameters": parameters,
            "sampler_parameters_sha256": sha256_bytes(canonical_json(parameters)),
        },
        "frozen_bindings": {
            "am87_support_plan": {**_binding(AM87_PLAN_PATH), "plan_id": AM87_PLAN_ID, "role": "immutable_complete_legitimate_quality_support_not_execution_multiplicity"},
            "training_manifest": _binding(MANIFEST_PATH),
            "pass_one": {**_binding(PASS_ONE_PATH), "state_id": _read(PASS_ONE_PATH)["state_id"], "assignment_performance_fields_used": False},
            "e7_handoff": {**_binding(E7_PATH), "handoff_id": e7["handoff_id"]},
            "am88_post_campaign_compatibility": {**_binding(AM88_POST_COMPATIBILITY_PATH), "compatibility_id": _read(AM88_POST_COMPATIBILITY_PATH)["compatibility_id"]},
            "am88_g8e_compatibility": {**_binding(AM88_G8E_COMPATIBILITY_PATH), "compatibility_id": _read(AM88_G8E_COMPATIBILITY_PATH)["compatibility_id"]},
            "generator_source": _binding(SOURCE_PATH),
            "generator_tool": _binding(GENERATOR_PATH),
            "independent_verifier": _binding(VERIFIER_PATH),
        },
        "support": {
            "source": "AM-87",
            "quality_count": len(quality_ids),
            "quality_ids_in_am87_order": quality_ids,
            "quality_order_sha256": sha256_bytes(canonical_json(quality_ids)),
            "quality_set_sha256": sha256_bytes(canonical_json(sorted(quality_ids))),
            "am87_authority_projection_reproduced_exactly": True,
            "support_changed_by_am88": False,
        },
        "training_membership": {
            "dataset": "imagenette160",
            "split": "train",
            "stable_id_count": len(training_ids),
            "stable_id_set_sha256": sha256_bytes(canonical_json(sorted(training_ids))),
            "stable_id_label_mapping_sha256": sha256_bytes(canonical_json(label_pairs)),
            "class_image_counts": {str(label): len(ids) for label, ids in ids_by_class.items()},
            "all_am87_training_ids_participate": True,
            "validation_id_count": 0,
            "test_id_count": 0,
        },
        "sampler": {
            "version": EXPECTED_VERSION,
            "seed": EXPECTED_SEED,
            "algorithm": EXPECTED_ALGORITHM,
            "balance_rule": EXPECTED_BALANCE_RULE,
            "variants_per_training_image": EXPECTED_VARIANTS,
            "quality_cycle_sha256": sha256_bytes(canonical_json(quality_cycle)),
            "pair_order": "class_label_ascending_then_seed_keyed_training_stable_id_then_slot",
            "assignment_inputs": parameters["artifact_finetune_assignment_information"],
            "forbidden_assignment_inputs": parameters["artifact_finetune_assignment_forbidden_information"],
            "sampling_before_f1_materialization": True,
            "outcome_independent": True,
        },
        "assignment_evidence": {
            "nominal_attempt_count": len(pairs),
            "ordered_pair_sha256": ordered_digest,
            "pair_set_sha256": set_digest,
            "unique_pair_count": len(set(pairs)),
            "duplicate_pair_count": 0,
            "participating_training_id_count": len(per_image),
            "per_image_assignment_minimum": min(per_image.values()),
            "per_image_assignment_maximum": max(per_image.values()),
            "quality_attempt_counts": global_counts,
            "class_quality_attempt_counts": class_counts,
            "balance": balance,
        },
        "f1_outcome_semantics": {
            "typed_image_codec_infeasibility": "record_omitted_assigned_pair_no_replacement_no_resampling",
            "neighbour_quality_substitution": False,
            "outage_image_substitution": False,
            "unexpected_codec_or_decoder_failure": "HOLD",
            "runtime_exception": "HOLD",
            "foreign_or_corrupt_identity": "HOLD",
            "unverified_artifact": "HOLD",
        },
        "compute_consequence": {
            "kind": "planning_extrapolation_not_measured_f1_benchmark",
            "old_am87_nominal_attempts": am87["multiplicity_and_feasibility"]["exact_attempt_count"],
            "nominal_attempts": len(pairs),
            "exact_reduction_factor": am87["multiplicity_and_feasibility"]["exact_attempt_count"] // len(pairs),
            "storage_basis": _binding(STORAGE_BASIS_PATH),
            "estimated_bytes_per_materialized_pair": bytes_per_pair,
            "validation_incidence_projected_object_estimate": incidence_objects,
            "validation_incidence_estimated_bytes": incidence_objects * bytes_per_pair,
            "maximum_estimated_bytes": len(pairs) * bytes_per_pair,
            "maximum_with_25_percent_safety_bytes": len(pairs) * bytes_per_pair + (len(pairs) * bytes_per_pair) // 4,  # literal-ok: existing 25-percent safety denominator
            "time_basis": old_window,
            "validation_incidence_scaled_confessor_seconds": window_seconds * incidence_objects / source_jobs,
            "maximum_scaled_confessor_seconds": window_seconds * len(pairs) / source_jobs,
            "measured_f1_timing_seconds": None,
        },
        "protected_boundary": {
            "metadata_only": True,
            "owner_audit_required": True,
            "f0_execution_authorized": False,
            "corpus_materialized": False,
            "materialized_object_count": 0,
            "image_payloads_decoded": 0,
            "jpeg2000_invocations": 0,
            "classifier_inference": 0,
            "optimizer_steps": 0,
            "pass_two": 0,
            "test_access": 0,
            "confessor_started": False,
            "prior_scientific_recomputation": "NONE",
            "next_action": "OWNER_AUDIT_AM88_PLAN_BEFORE_SEPARATE_F0_AUTHORIZATION",
            "e7_counters": dict(counters),
        },
    }
    body["plan_id"] = _identity(PLAN_PREFIX, body)
    return body
