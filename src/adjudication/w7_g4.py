"""Fixture-driven, fail-closed G-4 λ adjudication.

The implementation is complete during W7-A, but no scientific candidate is
loaded or adjudicated here.  ``fixture_candidate`` exists solely for offline
unit tests and carries a fixture lineage that cannot pass a production source
manifest check.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from config.params import get
from training.deterministic_core import canonical_sha256
from training.w7_protocol import (
    W7_CALIBRATION_SNR_DB,
    W7_CONTRACT_VERSION,
    W7_DATASET,
    W7_EXECUTION_IMAGE_FAMILY,
    W7_LAMBDA_GRID,
    W7_PROFILE_ID,
    W7_PSNR_SNR_DB,
    W7_RATIO,
    W7_SELECTED_GPU_UUID,
    W7_TRAINING_SNR_DB,
    W7_VALIDATION_NOISE_POLICY,
    load_w7_config,
    protocol_config_hash,
    protocol_descriptor,
)


G4_SCHEMA_VERSION = 1
CANDIDATE_ROLE = "W7_G4_LAMBDA_CANDIDATE_COMPLETION"
CANDIDATE_ELIGIBILITY = {
    "selection_eligibility": "ELIGIBLE_FOR_OWN_G4_ONLY",
    "w7_g4_eligibility": "ELIGIBLE_FOR_G4_CANDIDATE",
    "w8_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
    "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
}
INFINITY_TOKEN = "inf"


class G4Hold(RuntimeError):
    """A malformed candidate set cannot be adjudicated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G4Hold(message)


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise G4Hold(f"{label} is not numeric") from None
    if not math.isfinite(result):
        raise G4Hold(f"{label} is non-finite")
    return result


def _psnr_value(value: object) -> float:
    if value == INFINITY_TOKEN:
        return math.inf
    result = _finite(value, "candidate PSNR")
    return result


def _candidate_lambda(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("lambda")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise G4Hold("candidate lambda is not numeric")
    value = float(value)
    if not math.isfinite(value) or value not in W7_LAMBDA_GRID:
        raise G4Hold(f"candidate lambda is not in the frozen grid: {value!r}")
    return value


def validate_candidate(candidate: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_role",
        "candidate_id",
        "status",
        "authentication_status",
        "eligibility",
        "lambda",
        "lineage",
        "selected_validation",
        "psnr_evaluation",
        "selected_validation_result_digest",
        "selected_evidence",
        "test_access",
    }
    _require(isinstance(candidate, Mapping), "G-4 candidate is not a mapping")
    value = dict(candidate)
    _require(set(value) == required, "G-4 candidate schema differs")
    _require(value["schema_version"] == G4_SCHEMA_VERSION, "G-4 candidate schema version differs")
    _require(value["artifact_role"] == CANDIDATE_ROLE, "G-4 candidate role is not a pilot completion")
    _require(isinstance(value["candidate_id"], str) and value["candidate_id"], "G-4 candidate ID is empty")
    _require(value["status"] == "COMPLETE", "G-4 candidate is incomplete")
    _require(value["authentication_status"] == "PASSED", "G-4 candidate is not authenticated")
    _require(value["eligibility"] == CANDIDATE_ELIGIBILITY, "G-4 candidate eligibility differs")
    candidate_lambda = _candidate_lambda(value)
    _require(value["test_access"] == 0, "G-4 candidate contains test access")
    selected_digest = value["selected_validation_result_digest"]
    _require(isinstance(selected_digest, str) and len(selected_digest) == 64 and all(character in "0123456789abcdef" for character in selected_digest), "G-4 selected result digest is invalid")  # literal-ok: SHA-256 width
    _require(
        value["candidate_id"] == "w7candidate-" + canonical_sha256({"lambda": candidate_lambda, "selected": selected_digest}),
        "G-4 candidate ID does not authenticate its selected result",
    )
    evidence = value["selected_evidence"]
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == {"path", "result_digest", "file_sha256"}
        and isinstance(evidence["path"], str)
        and evidence["path"]
        and evidence["result_digest"] == selected_digest
        and isinstance(evidence["file_sha256"], str)
        and len(evidence["file_sha256"]) == 64  # literal-ok: SHA-256 width
        and all(character in "0123456789abcdef" for character in evidence["file_sha256"]),
        "G-4 selected evidence binding differs",
    )
    evidence_path = evidence["path"]
    _require(
        not evidence_path.startswith("/")
        and ".." not in evidence_path.replace("\\", "/").split("/")
        and "\x00" not in evidence_path,
        "G-4 selected evidence path is unsafe",
    )

    lineage_required = {
        "protocol_version",
        "source_commit",
        "source_manifest_id",
        "source_manifest_sha256",
        "protocol_config_hash",
        "execution_image",
        "execution_profile_id",
        "gpu_uuid",
        "dataset",
        "split_manifest_hash",
        "architecture",
        "ratio",
        "k",
        "train_seed",
        "channel_seed",
        "train_snr_db",
        "epochs",
        "optimizer",
        "scheduler",
        "checkpoint_selection",
        "validation_noise_policy",
    }
    lineage = value["lineage"]
    _require(isinstance(lineage, Mapping) and set(lineage) == lineage_required, "G-4 candidate lineage schema differs")
    expected = {
        "protocol_version": W7_CONTRACT_VERSION,
        "protocol_config_hash": protocol_config_hash(load_w7_config(lambda_value=candidate_lambda)),
        "execution_profile_id": W7_PROFILE_ID,
        "execution_image": W7_EXECUTION_IMAGE_FAMILY,
        "dataset": W7_DATASET,
        "ratio": W7_RATIO,
        "train_seed": 0,  # literal-ok: owner-frozen fixture seed
        "channel_seed": 0,  # literal-ok: owner-frozen fixture seed
        "train_snr_db": W7_TRAINING_SNR_DB,
        "epochs": 100,  # literal-ok: owner-frozen fixture schedule
        "optimizer": get("learned_system.optimizer"),
        "scheduler": get("learned_system.lr_schedule"),
        "architecture": get("learned_system.encoder_arch"),
        "k": get(f"bandwidth.k_symbols.{W7_DATASET}.{W7_RATIO}"),
        "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
    }
    for key, expected_value in expected.items():
        _require(lineage[key] == expected_value, f"G-4 candidate lineage {key} differs")
    _require(isinstance(lineage["source_commit"], str) and len(lineage["source_commit"]) == 40 and all(character in "0123456789abcdef" for character in lineage["source_commit"]), "G-4 source commit is invalid")  # literal-ok: Git SHA-1 width
    _require(isinstance(lineage["source_manifest_id"], str) and lineage["source_manifest_id"], "G-4 source manifest ID is empty")
    _require(isinstance(lineage["source_manifest_sha256"], str) and len(lineage["source_manifest_sha256"]) == 64 and all(character in "0123456789abcdef" for character in lineage["source_manifest_sha256"]), "G-4 source manifest SHA is invalid")  # literal-ok: SHA-256 width
    _require(isinstance(lineage["split_manifest_hash"], str) and len(lineage["split_manifest_hash"]) == 64 and all(character in "0123456789abcdef" for character in lineage["split_manifest_hash"]), "G-4 split manifest SHA is invalid")  # literal-ok: SHA-256 width
    _require(lineage["execution_image"] == W7_EXECUTION_IMAGE_FAMILY, "G-4 execution image family differs")
    _require(lineage["gpu_uuid"] == W7_SELECTED_GPU_UUID, "G-4 GPU/profile homogeneity differs from the frozen Pascal GPU")
    _require(isinstance(lineage["checkpoint_selection"], Mapping), "G-4 checkpoint rule is not a mapping")
    _require(
        dict(lineage["checkpoint_selection"])
        == {
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_db": W7_CALIBRATION_SNR_DB,
        },
        "G-4 checkpoint-selection rule differs",
    )

    validation_required = {"checkpoint_id", "epoch", "n_correct", "n_total", "top1_accuracy"}
    validation = value["selected_validation"]
    _require(isinstance(validation, Mapping) and set(validation) == validation_required, "G-4 validation result schema differs")
    _require(isinstance(validation["epoch"], int) and not isinstance(validation["epoch"], bool) and 0 <= validation["epoch"] < int(get(f"learned_system.epochs.{W7_DATASET}")), "G-4 selected epoch is invalid")
    _require(isinstance(validation["checkpoint_id"], str) and len(validation["checkpoint_id"]) == 64 and all(character in "0123456789abcdef" for character in validation["checkpoint_id"]), "G-4 selected checkpoint ID is invalid")  # literal-ok: SHA-256 width
    _require(validation["n_total"] == get(f"datasets.{W7_DATASET}.val_images"), "G-4 validation denominator differs")
    _require(isinstance(validation["n_correct"], int) and 0 <= validation["n_correct"] <= validation["n_total"], "G-4 validation count is invalid")
    _require(validation["top1_accuracy"] == validation["n_correct"] / validation["n_total"], "G-4 top-1 is not count-derived")

    psnr_required = {"snr_db", "denominator", "psnr_db", "data_range", "per_image_digest"}
    psnr = value["psnr_evaluation"]
    _require(isinstance(psnr, Mapping) and set(psnr) == psnr_required, "G-4 PSNR result schema differs")
    _require(psnr["snr_db"] == W7_PSNR_SNR_DB, "G-4 PSNR SNR differs")
    _require(psnr["denominator"] == get(f"datasets.{W7_DATASET}.val_images"), "G-4 PSNR denominator differs")
    _require(float(psnr["data_range"]) == float(get("preprocessing.psnr_data_range")), "G-4 PSNR data range differs")
    _psnr_value(psnr["psnr_db"])
    _require(isinstance(psnr["per_image_digest"], str) and len(psnr["per_image_digest"]) == 64 and all(character in "0123456789abcdef" for character in psnr["per_image_digest"]), "G-4 PSNR evidence digest is invalid")  # literal-ok: SHA-256 width
    return value


def _homogeneity_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    lineage = dict(candidate["lineage"])
    # Every lineage field is a campaign-wide invariant.  Candidate-specific
    # checkpoint/result IDs live outside this projection and are intentionally
    # excluded; source epoch and source-manifest identity must remain included.
    return {
        "artifact_role": candidate["artifact_role"],
        "authentication_status": candidate["authentication_status"],
        "eligibility": candidate["eligibility"],
        "lineage": lineage,
        "test_access": candidate["test_access"],
    }


def adjudicate_g4(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Adjudicate exactly one complete, homogeneous candidate per λ.

    This function is deliberately not called by any W7-A launcher.  It has no
    side effect and never writes ``lambda_core``.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise G4Hold("G-4 candidates must be a sequence")
    _require(len(candidates) == len(W7_LAMBDA_GRID), "G-4 requires exactly one candidate per configured lambda")
    _require(all(isinstance(candidate, Mapping) for candidate in candidates), "G-4 candidate is not a mapping")
    lambdas = [_candidate_lambda(candidate) for candidate in candidates]
    _require(len(set(lambdas)) == len(lambdas), "G-4 candidate lambda is duplicated")
    _require(set(lambdas) == set(W7_LAMBDA_GRID), "G-4 candidate lambda set is incomplete or foreign")
    validated = [validate_candidate(candidate) for candidate in candidates]
    baseline_projection = _homogeneity_projection(validated[0])
    for candidate in validated[1:]:
        _require(_homogeneity_projection(candidate) == baseline_projection, "G-4 candidate lineage/profile homogeneity differs")
    by_lambda = {float(_candidate_lambda(candidate)): candidate for candidate in validated}
    baseline = by_lambda[0.0]
    baseline_top1 = float(baseline["selected_validation"]["top1_accuracy"])
    tolerance_fraction = float(get("learned_system.lambda_acc_tolerance_pp")) / 100.0  # literal-ok: percentage-point conversion
    accuracy_floor = baseline_top1 - tolerance_fraction
    accuracy_ok = {
        value: float(candidate["selected_validation"]["top1_accuracy"]) >= accuracy_floor
        for value, candidate in by_lambda.items()
    }
    primary_floor = float(get("learned_system.lambda_psnr_floor_db"))
    relaxed_floor = float(get("learned_system.lambda_psnr_floor_relaxed_db"))

    def qualifying(floor: float) -> list[float]:
        return sorted(
            value
            for value, candidate in by_lambda.items()
            if accuracy_ok[value] and _psnr_value(candidate["psnr_evaluation"]["psnr_db"]) >= floor
        )

    primary = qualifying(primary_floor)
    if primary:
        status = "G4_ADJUDICATED_PRIMARY"
        selected = primary[0]
        tier = "PRIMARY"
    else:
        relaxed = qualifying(relaxed_floor)
        if relaxed:
            status = "G4_ADJUDICATED_RELAXED"
            selected = relaxed[0]
            tier = "RELAXED"
        else:
            result = {
                "schema_version": G4_SCHEMA_VERSION,
                "artifact_role": "G4_ADJUDICATED",
                "status": "G4_HOLD_DEC2_REVERSAL_REPLAN_REQUIRED",
                "selected_lambda": None,
                "baseline_lambda_zero_top1": baseline_top1,
                "accuracy_tolerance_pp": float(get("learned_system.lambda_acc_tolerance_pp")),
                "accuracy_floor": accuracy_floor,
                "primary_psnr_floor_db": primary_floor,
                "relaxed_psnr_floor_db": relaxed_floor,
                "primary_qualifying_lambdas": [],
                "relaxed_qualifying_lambdas": [],
                "candidate_lambdas": list(W7_LAMBDA_GRID),
                "scientific_side_effects": {"lambda_core_updated": False},
            }
            result["adjudication_id"] = "g4adjudication-" + canonical_sha256(result)
            return result
    result = {
        "schema_version": G4_SCHEMA_VERSION,
        "artifact_role": "G4_ADJUDICATED",
        "status": status,
        "selected_lambda": selected,
        "selection_tier": tier,
        "baseline_lambda_zero_top1": baseline_top1,
        "accuracy_tolerance_pp": float(get("learned_system.lambda_acc_tolerance_pp")),
        "accuracy_floor": accuracy_floor,
        "primary_psnr_floor_db": primary_floor,
        "relaxed_psnr_floor_db": relaxed_floor,
        "primary_qualifying_lambdas": primary,
        "relaxed_qualifying_lambdas": qualifying(relaxed_floor),
        "candidate_lambdas": list(W7_LAMBDA_GRID),
        "scientific_side_effects": {"lambda_core_updated": False},
    }
    result["adjudication_id"] = "g4adjudication-" + canonical_sha256(result)
    return result


def fixture_candidate(
    lambda_value: float,
    *,
    top1: float = 0.8,
    psnr_db: float | str = 20.0,  # literal-ok: fixture-only metric default
    gpu_uuid: str = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
    source_commit: str = "f" * 40,  # literal-ok: fixture Git SHA-1 width
) -> dict[str, Any]:
    """Build an explicitly non-production candidate for adjudicator tests."""

    total = int(get(f"datasets.{W7_DATASET}.val_images"))
    correct = int(round(float(top1) * total))
    validation_top1 = correct / total
    lineage = {
        "protocol_version": W7_CONTRACT_VERSION,
        "source_commit": source_commit,
        "source_manifest_id": "fixture-source",
        "source_manifest_sha256": "a" * 64,  # literal-ok: fixture SHA-256 width
        "protocol_config_hash": protocol_config_hash(load_w7_config(lambda_value=lambda_value)),
        "execution_image": W7_EXECUTION_IMAGE_FAMILY,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": gpu_uuid,
        "dataset": W7_DATASET,
        "split_manifest_hash": "b" * 64,  # literal-ok: fixture SHA-256 width
        "architecture": get("learned_system.encoder_arch"),
        "ratio": W7_RATIO,
        "k": get(f"bandwidth.k_symbols.{W7_DATASET}.{W7_RATIO}"),
        "train_seed": 0,
        "channel_seed": 0,
        "train_snr_db": W7_TRAINING_SNR_DB,
        "epochs": 100,  # literal-ok: owner-frozen fixture schedule
        "optimizer": "adam",
        "scheduler": "cosine",
        "checkpoint_selection": {
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_db": W7_CALIBRATION_SNR_DB,
        },
        "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
    }
    candidate = {
        "schema_version": G4_SCHEMA_VERSION,
        "artifact_role": CANDIDATE_ROLE,
        "candidate_id": "w7candidate-" + canonical_sha256({"lambda": float(lambda_value), "selected": "e" * 64}),  # literal-ok: fixture SHA-256 width
        "status": "COMPLETE",
        "authentication_status": "PASSED",
        "eligibility": dict(CANDIDATE_ELIGIBILITY),
        "lambda": float(lambda_value),
        "lineage": lineage,
        "selected_validation": {
            "checkpoint_id": "c" * 64,  # literal-ok: fixture SHA-256 width
            "epoch": 0,
            "n_correct": correct,
            "n_total": total,
            "top1_accuracy": validation_top1,
        },
        "psnr_evaluation": {
            "snr_db": W7_PSNR_SNR_DB,
            "denominator": total,
            "psnr_db": psnr_db,
            "data_range": float(get("preprocessing.psnr_data_range")),
            "per_image_digest": "d" * 64,  # literal-ok: fixture SHA-256 width
        },
        "selected_validation_result_digest": "e" * 64,  # literal-ok: fixture SHA-256 width
        "selected_evidence": {
            "path": "fixtures/w7/selected.json",
            "result_digest": "e" * 64,  # literal-ok: fixture SHA-256 width
            "file_sha256": "f" * 64,  # literal-ok: fixture SHA-256 width
        },
        "test_access": 0,
    }
    return candidate
