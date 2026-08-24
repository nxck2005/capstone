#!/usr/bin/env python3
"""Independent metadata-only verifier for the frozen AM-88 G8_F sampler.

This verifier deliberately restates the assignment algorithm instead of calling
the generator.  It reads only tracked JSON/YAML/CSV/source metadata and cannot
authorize F0, decode an image, invoke JPEG 2000 or a classifier, train, run pass
two, or access the guarded test loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "results/baseline/g8_f/am88_sampler_plan.json"
AM87 = REPO / "results/baseline/g8_f/corpus_plan.json"
PARAMS = REPO / "spec/params.generated.yaml"
MANIFEST = REPO / "data/manifests/imagenette160.csv"
E7 = REPO / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
EXPECTED_ROLE = "g8_f_am88_metadata_only_balanced_sampler_plan"
EXPECTED_AM87_ID = "g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148"
EXPECTED_AM87_SHA = "733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c"
EXPECTED_SEED = "am88-g8f-balanced-sampler-20260824-v1"
EXPECTED_VERSION = "g8_f_balanced_sampler_v1"
EXPECTED_ALGORITHM = "sha256_keyed_stable_id_order_global_quality_permutation_class_chunks_cyclic_v1"
EXPECTED_BALANCE_RULE = "concatenate_class_label_ordered_attempt_chunks_over_one_seed_permuted_quality_cycle"
EXPECTED_QUALITY_COUNT = 120
EXPECTED_TRAINING_COUNT = 8469
EXPECTED_VARIANTS = 6
EXPECTED_ATTEMPTS = 50_814
ALLOWED_ASSIGNMENT_INPUTS = ["training_stable_id", "class_label", "am87_quality_id_order", "sampler_seed", "sampler_version"]
FORBIDDEN_ASSIGNMENT_INPUTS = [
    "pass_one_expected_accuracy", "pass_one_score", "pass_one_rank", "pass_one_margin",
    "selected_phy_tuple", "validation_e4_feasibility", "validation_artifact_performance",
    "f1_codec_outcome", "pass_two_result", "learned_result", "test_result", "runtime_order",
]


class G8FSamplerVerificationError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise G8FSamplerVerificationError(f"non-canonical value: {exc}") from None


def rendered(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FSamplerVerificationError(message)


def read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FSamplerVerificationError(f"cannot read {label}: {exc}") from None
    require(isinstance(value, dict), f"{label} is not an object")
    return value, raw


def verify_binding(binding: Mapping[str, Any], label: str) -> None:
    require(set(binding) >= {"path", "bytes", "sha256"}, f"{label} binding schema differs")
    path = REPO / str(binding["path"])
    raw = path.read_bytes()
    require(len(raw) == binding["bytes"] and digest(raw) == binding["sha256"], f"{label} binding bytes differ")


def keyed_order(values: Sequence[str], seed: str, domain: str) -> list[str]:
    def key(value: str) -> tuple[str, str]:
        payload = {"seed": seed, "domain": domain, "value": value}
        return digest(canonical(payload)), value
    return sorted(values, key=key)


def pair_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return digest(canonical([[stable_id, quality_id] for stable_id, quality_id in pairs]))


def manifest_training() -> tuple[list[str], dict[str, int], dict[int, list[str]], set[str], set[str]]:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise G8FSamplerVerificationError(f"cannot read training manifest: {exc}") from None
    require(rows and set(rows[0]) == {"stable_sample_id", "label", "split"}, "manifest schema differs")
    all_ids = [row["stable_sample_id"] for row in rows]
    require(all_ids == sorted(all_ids) and len(all_ids) == len(set(all_ids)), "manifest stable IDs are not globally ascending unique")
    train = [row for row in rows if row["split"] == "train"]
    ids = [row["stable_sample_id"] for row in train]
    labels = {row["stable_sample_id"]: int(row["label"]) for row in train}
    by_class: dict[int, list[str]] = defaultdict(list)
    for stable_id in ids:
        by_class[labels[stable_id]].append(stable_id)
    validation = {row["stable_sample_id"] for row in rows if row["split"] == "val"}
    test = {row["stable_sample_id"] for row in rows if row["split"] == "test"}
    return ids, labels, dict(sorted(by_class.items())), validation, test


def independent_assignments(
    quality_ids: Sequence[str], ids_by_class: Mapping[int, Sequence[str]], seed: str, variants: int
) -> tuple[list[tuple[str, str]], list[str]]:
    require(0 < variants <= len(quality_ids), "variants cannot be distinct within support")
    cycle = keyed_order(list(quality_ids), seed, "am88_quality_permutation")
    pairs: list[tuple[str, str]] = []
    cursor = 0
    for label in sorted(ids_by_class):
        image_order = keyed_order(list(ids_by_class[label]), seed, f"am88_training_ids_class_{label}")
        for stable_id in image_order:
            selected = [cycle[(cursor + slot) % len(cycle)] for slot in range(variants)]
            require(len(selected) == len(set(selected)), "duplicate quality within one image")
            pairs.extend((stable_id, quality_id) for quality_id in selected)
            cursor += variants
    return pairs, cycle


def independent_counts(
    pairs: Sequence[tuple[str, str]], quality_ids: Sequence[str], labels: Mapping[str, int]
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, Any], Counter[str]]:
    global_counter = Counter(quality_id for _stable_id, quality_id in pairs)
    image_counter = Counter(stable_id for stable_id, _quality_id in pairs)
    class_counter: dict[int, Counter[str]] = {label: Counter() for label in sorted(set(labels.values()))}
    for stable_id, quality_id in pairs:
        class_counter[labels[stable_id]][quality_id] += 1
    global_exact = {quality_id: global_counter[quality_id] for quality_id in quality_ids}
    class_exact = {
        str(label): {quality_id: counts[quality_id] for quality_id in quality_ids}
        for label, counts in class_counter.items()
    }
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
        "per_image_minimum": min(image_counter.values()),
        "per_image_maximum": max(image_counter.values()),
    }
    return global_exact, class_exact, balance, image_counter


def verify_sampler_plan(path: Path = PLAN) -> dict[str, Any]:
    value, raw = read_object(path, "AM-88 sampler plan")
    require(raw == rendered(value), "AM-88 plan is not canonical rendered JSON")
    require(value.get("schema_version") == 1 and value.get("artifact_role") == EXPECTED_ROLE, "AM-88 plan header differs")
    body = dict(value)
    plan_id = body.pop("plan_id", None)
    require(plan_id == "g8fsamplerplan-" + digest(canonical(body)), "AM-88 plan ID does not reproduce")

    am87, am87_raw = read_object(AM87, "AM-87 support plan")
    require(digest(am87_raw) == EXPECTED_AM87_SHA, "AM-87 support-plan file SHA differs")
    am87_body = dict(am87)
    am87_id = am87_body.pop("plan_id", None)
    require(am87_id == EXPECTED_AM87_ID == "g8fcorpusplan-" + digest(canonical(am87_body)), "AM-87 support-plan ID differs")
    quality_rows = am87.get("artifact_quality_projection", {}).get("qualities")
    require(isinstance(quality_rows, list), "AM-87 support rows are absent")
    quality_ids = [row.get("quality_id") for row in quality_rows]
    require(len(quality_ids) == EXPECTED_QUALITY_COUNT and quality_ids == sorted(set(quality_ids)), "AM-87 quality support count/order differs")
    for row in quality_rows:
        identity = row.get("identity")
        require(isinstance(identity, dict), "AM-87 quality identity is absent")
        require(row.get("quality_id") == "g8fquality-" + digest(canonical(identity)), "AM-87 quality identity hash differs")

    params = yaml.safe_load(PARAMS.read_bytes())
    classifier = params.get("reference_classifier", {}) if isinstance(params, dict) else {}
    require(classifier.get("artifact_finetune_sampler_seed") == EXPECTED_SEED, "configured AM-88 seed differs")
    require(classifier.get("artifact_finetune_sampler_version") == EXPECTED_VERSION, "configured sampler version differs")
    require(classifier.get("artifact_finetune_assignment_algorithm") == EXPECTED_ALGORITHM, "configured assignment algorithm differs")
    require(classifier.get("artifact_finetune_balance_rule") == EXPECTED_BALANCE_RULE, "configured balance rule differs")
    require(classifier.get("artifact_finetune_variants_per_training_image") == EXPECTED_VARIANTS, "configured variants per image differs")
    require(classifier.get("artifact_finetune_assignment_information") == ALLOWED_ASSIGNMENT_INPUTS, "assignment information set differs")
    require(classifier.get("artifact_finetune_assignment_forbidden_information") == FORBIDDEN_ASSIGNMENT_INPUTS, "forbidden assignment information set differs")

    membership = value.get("training_membership")
    support = value.get("support")
    sampler = value.get("sampler")
    evidence = value.get("assignment_evidence")
    boundary = value.get("protected_boundary")
    semantics = value.get("f1_outcome_semantics")
    require(all(isinstance(section, dict) for section in (membership, support, sampler, evidence, boundary, semantics)), "AM-88 plan section is absent")
    require(support["quality_ids_in_am87_order"] == quality_ids, "AM-88 narrowed or reordered AM-87 support")
    require(support["quality_count"] == EXPECTED_QUALITY_COUNT and support["support_changed_by_am88"] is False, "AM-88 support boundary differs")
    require(support["quality_order_sha256"] == digest(canonical(quality_ids)), "quality order digest differs")
    require(support["quality_set_sha256"] == digest(canonical(sorted(quality_ids))), "quality set digest differs")

    training_ids, labels, ids_by_class, validation_ids, test_ids = manifest_training()
    require(len(training_ids) == EXPECTED_TRAINING_COUNT, "frozen training count differs")
    require(am87["training_membership"]["stable_ids"] == training_ids, "AM-87 training membership no longer matches manifest")
    label_pairs = [[stable_id, labels[stable_id]] for stable_id in training_ids]
    require(membership["stable_id_count"] == len(training_ids), "AM-88 training count differs")
    require(membership["stable_id_set_sha256"] == digest(canonical(sorted(training_ids))), "training stable-ID set digest differs")
    require(membership["stable_id_label_mapping_sha256"] == digest(canonical(label_pairs)), "training class-label mapping digest differs")
    require(membership["class_image_counts"] == {str(label): len(ids) for label, ids in ids_by_class.items()}, "class image counts differ")

    require(sampler["seed"] == EXPECTED_SEED and sampler["version"] == EXPECTED_VERSION, "frozen seed/version differs")
    require(sampler["algorithm"] == EXPECTED_ALGORITHM and sampler["balance_rule"] == EXPECTED_BALANCE_RULE, "frozen sampler algorithm differs")
    require(sampler["variants_per_training_image"] == EXPECTED_VARIANTS, "frozen variants per image differs")
    require(sampler["assignment_inputs"] == ALLOWED_ASSIGNMENT_INPUTS and sampler["forbidden_assignment_inputs"] == FORBIDDEN_ASSIGNMENT_INPUTS, "assignment input boundary differs")
    require(sampler["sampling_before_f1_materialization"] is True and sampler["outcome_independent"] is True, "sampling timing/outcome boundary differs")

    pairs, cycle = independent_assignments(quality_ids, ids_by_class, EXPECTED_SEED, EXPECTED_VARIANTS)
    require(len(pairs) == EXPECTED_ATTEMPTS == EXPECTED_TRAINING_COUNT * EXPECTED_VARIANTS, "nominal attempt count differs")
    require(len(set(pairs)) == len(pairs), "duplicate image-quality pair exists")
    require(not ({stable_id for stable_id, _quality_id in pairs} & validation_ids), "validation ID entered assignment")
    require(not ({stable_id for stable_id, _quality_id in pairs} & test_ids), "test ID entered assignment")
    require({quality_id for _stable_id, quality_id in pairs} == set(quality_ids), "not every AM-87 quality is represented")
    global_counts, class_counts, balance, image_counts = independent_counts(pairs, quality_ids, labels)
    require(set(image_counts) == set(training_ids) and set(image_counts.values()) == {EXPECTED_VARIANTS}, "not every training image has six assignments")
    require(balance["global"]["range"] == balance["global"]["arithmetic_minimum_range"] == 1, "global balance is not arithmetically minimum")
    require(all(stats["range"] <= 1 for stats in balance["per_class"].values()), "within-class balance exceeds arithmetic bound")
    require(sampler["quality_cycle_sha256"] == digest(canonical(cycle)), "quality-cycle digest differs")
    require(evidence["ordered_pair_sha256"] == pair_digest(pairs), "ordered-pair digest differs")
    require(evidence["pair_set_sha256"] == pair_digest(sorted(pairs)), "pair-set digest differs")
    require(evidence["nominal_attempt_count"] == EXPECTED_ATTEMPTS, "recorded nominal attempts differ")
    require(evidence["unique_pair_count"] == EXPECTED_ATTEMPTS and evidence["duplicate_pair_count"] == 0, "pair uniqueness evidence differs")
    require(evidence["participating_training_id_count"] == EXPECTED_TRAINING_COUNT, "participating training-ID count differs")
    require(evidence["quality_attempt_counts"] == global_counts, "global quality counts differ")
    require(evidence["class_quality_attempt_counts"] == class_counts, "class-quality counts differ")
    require(evidence["balance"] == balance, "balance summary differs")

    require(semantics["typed_image_codec_infeasibility"] == "record_omitted_assigned_pair_no_replacement_no_resampling", "typed codec infeasibility may resample")
    require(semantics["neighbour_quality_substitution"] is False and semantics["outage_image_substitution"] is False, "substitution is enabled")
    require(all(semantics[name] == "HOLD" for name in ("unexpected_codec_or_decoder_failure", "runtime_exception", "foreign_or_corrupt_identity", "unverified_artifact")), "unexpected failure can become an omission")

    e7, _ = read_object(E7, "E7 handoff")
    require(e7.get("g8_f", {}).get("authorized") is False and e7.get("g8_f", {}).get("execution_count") == 0, "E7 says G8_F is authorized/executed")
    require(boundary["metadata_only"] is True and boundary["f0_execution_authorized"] is False, "AM-88 opened F0")
    require(boundary["corpus_materialized"] is False and boundary["materialized_object_count"] == 0, "AM-88 materialized corpus objects")
    for name in ("image_payloads_decoded", "jpeg2000_invocations", "classifier_inference", "optimizer_steps", "pass_two", "test_access"):
        require(boundary[name] == 0, f"protected {name} counter is nonzero")
    require(boundary["confessor_started"] is False and boundary["owner_audit_required"] is True, "owner/worker boundary differs")
    require(boundary["e7_counters"] == e7["counters"], "protected E7 counters differ")

    for name, binding in value["frozen_bindings"].items():
        verify_binding(binding, name)
    require(value["frozen_bindings"]["am87_support_plan"]["plan_id"] == EXPECTED_AM87_ID, "AM-87 binding ID differs")
    return value


def main() -> int:
    try:
        plan = verify_sampler_plan()
    except (G8FSamplerVerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    evidence = plan["assignment_evidence"]
    print(json.dumps({
        "status": "PASS",
        "verdict": "AM-88 GREEN - BALANCED G8_F TRAINING SAMPLER FROZEN; F0 STILL REQUIRES SEPARATE OWNER AUTHORIZATION",
        "plan_id": plan["plan_id"],
        "plan_file_sha256": digest(PLAN.read_bytes()),
        "support_quality_count": plan["support"]["quality_count"],
        "training_stable_id_count": plan["training_membership"]["stable_id_count"],
        "variants_per_image": plan["sampler"]["variants_per_training_image"],
        "nominal_attempt_count": evidence["nominal_attempt_count"],
        "global_balance": evidence["balance"]["global"],
        "per_class_balance": evidence["balance"]["per_class"],
        "ordered_pair_sha256": evidence["ordered_pair_sha256"],
        "pair_set_sha256": evidence["pair_set_sha256"],
        "materialized_object_count": plan["protected_boundary"]["materialized_object_count"],
        "optimizer_steps": plan["protected_boundary"]["optimizer_steps"],
        "pass_two": plan["protected_boundary"]["pass_two"],
        "test_access": plan["protected_boundary"]["test_access"],
        "f0_authorized": plan["protected_boundary"]["f0_execution_authorized"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
