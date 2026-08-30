#!/usr/bin/env python3
"""Fail-closed terminal authentication for the W7-C G-4 decision.

The verifier consumes only the compact B2R JSON evidence.  It authenticates the
frozen adjudicator identity and independently reconstructs the same deterministic
rule; it never opens a checkpoint and never performs model-facing inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from gen_w7_g4_authorization import (  # noqa: E402
    ADJUDICATOR_BLOB,
    ADJUDICATOR_PATH,
    AUTHORIZATION_PATH,
    AUTHORIZATION_PREFIX,
    B2R_EVIDENCE,
    FROZEN_PROTOCOL,
    SCIENTIFIC_SOURCE_COMMIT,
    STARTING_MAIN_SHA,
    canonical_sha256,
    verify_authorization,
)
from verify_w7_b2r import verify_artifacts  # noqa: E402

G4_PATH = REPO / "results/learned/w7/w7_g4_result.json"
TERMINAL_PATH = REPO / "results/learned/w7/w7_completion.json"
G4_OUTER_PREFIX = "w7g4adjudication-"
SCHEMA_VERSION = 1


class VerificationError(RuntimeError):
    """A consequential W7-C invariant failed."""


def fail(message: str) -> None:
    raise VerificationError(message)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def _expect_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        fail(f"{label} schema differs")


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    raise AssertionError("unreachable")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric")
    if not math.isfinite(result):
        fail(f"{label} is non-finite")
    return result


def _psnr(value: Any) -> float:
    if value == "inf":
        return math.inf
    return _finite(value, "candidate PSNR")


def _expected_candidate_metrics(index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index["candidates"]:
        raw = item["candidate_completion"]["value"]
        selected = item["selected"]
        rows.append({
            "candidate_id": raw["candidate_id"],
            "lambda": raw["lambda"],
            "selected_epoch": selected["selected_epoch"],
            "selected_checkpoint_id": selected["selected_checkpoint_id"],
            "validation": {
                "snr_db": FROZEN_PROTOCOL["calibration_snr_db"],
                "n_correct": raw["selected_validation"]["n_correct"],
                "n_total": raw["selected_validation"]["n_total"],
                "top1_accuracy": raw["selected_validation"]["top1_accuracy"],
            },
            "psnr_evaluation": {
                "snr_db": FROZEN_PROTOCOL["psnr_evaluation_snr_db"],
                "mean_psnr_db": raw["psnr_evaluation"]["psnr_db"],
            },
            "factual_papr": {
                "snr_db": FROZEN_PROTOCOL["psnr_evaluation_snr_db"],
                "mean_papr": selected["derived_mean_papr"],
            },
        })
    return rows


def reconstruct_frozen_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the frozen source rule without calling the adjudicator."""

    if len(candidates) != len(FROZEN_PROTOCOL["lambda_grid"]):
        fail("G-4 reconstruction requires exactly five candidates")
    if [candidate["lambda"] for candidate in candidates] != FROZEN_PROTOCOL["lambda_grid"]:
        fail("G-4 reconstruction lambda order differs")
    baseline = next(candidate for candidate in candidates if candidate["lambda"] == 0.0)
    baseline_top1 = float(baseline["validation"]["top1_accuracy"])
    tolerance = float(FROZEN_PROTOCOL["accuracy_tolerance_pp"])
    accuracy_floor = baseline_top1 - tolerance / 100.0
    accuracy_ok = {
        candidate["lambda"]: float(candidate["validation"]["top1_accuracy"]) >= accuracy_floor
        for candidate in candidates
    }
    primary_floor = float(FROZEN_PROTOCOL["primary_psnr_floor_db"])
    relaxed_floor = float(FROZEN_PROTOCOL["relaxed_psnr_floor_db"])

    def qualifying(floor: float) -> list[float]:
        return sorted(
            candidate["lambda"]
            for candidate in candidates
            if accuracy_ok[candidate["lambda"]]
            and _psnr(candidate["psnr_evaluation"]["mean_psnr_db"]) >= floor
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
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "artifact_role": "G4_ADJUDICATED",
                "status": "G4_HOLD_DEC2_REVERSAL_REPLAN_REQUIRED",
                "selected_lambda": None,
                "baseline_lambda_zero_top1": baseline_top1,
                "accuracy_tolerance_pp": tolerance,
                "accuracy_floor": accuracy_floor,
                "primary_psnr_floor_db": primary_floor,
                "relaxed_psnr_floor_db": relaxed_floor,
                "primary_qualifying_lambdas": [],
                "relaxed_qualifying_lambdas": [],
                "candidate_lambdas": list(FROZEN_PROTOCOL["lambda_grid"]),
                "scientific_side_effects": {"lambda_core_updated": False},
            }
            result["adjudication_id"] = "g4adjudication-" + canonical_sha256(result)
            return result
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "G4_ADJUDICATED",
        "status": status,
        "selected_lambda": selected,
        "selection_tier": tier,
        "baseline_lambda_zero_top1": baseline_top1,
        "accuracy_tolerance_pp": tolerance,
        "accuracy_floor": accuracy_floor,
        "primary_psnr_floor_db": primary_floor,
        "relaxed_psnr_floor_db": relaxed_floor,
        "primary_qualifying_lambdas": primary,
        "relaxed_qualifying_lambdas": qualifying(relaxed_floor),
        "candidate_lambdas": list(FROZEN_PROTOCOL["lambda_grid"]),
        "scientific_side_effects": {"lambda_core_updated": False},
    }
    result["adjudication_id"] = "g4adjudication-" + canonical_sha256(result)
    return result


def _load_b2r_index() -> dict[str, Any]:
    return _read_json(REPO / "results/learned/w7/w7_b2_reconciliation_index.json")


def _verify_b2r_and_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        authorization = verify_authorization(AUTHORIZATION_PATH)
        b2r = verify_artifacts(REPO / "results/learned/w7", reauthenticate_upstream=False)
    except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        fail(f"W7-B2R or authorization authentication failed: {exc}")
    if b2r != {
        "status": "PASS",
        "campaign_id": "w7-b2-g4-pascal-20260829",
        "candidate_count": 5,
        "completed_epoch_cycles": 500,
        "checkpoint_count": 500,
        "index_id": "w7b2rindex-171ac64c8c56a00bff1e943aacfbe9fb92d1ecfc23686d9f6f55054c1ffbf3f0",
        "custody_id": "w7b2rcustody-1bbe4b083907b5debdb8104a3818ee9a5161d2bbccf963bc1ce1e2a0c53eb9ec",
        "common_noise_audit_id": "w7b2rnoise-f7f162de8664c8b03e15983e436f66918827cce8bcd1444ed2c53ff3d72662e5",
        "reconciliation_id": "w7b2rreconciliation-981bce14b3d851dd68a8304823fec86d5b6bcf9a948b1f1914ca7e4cd4cf168e",
        "completion_id": "w7b2rcompletion-172842c61df0231efd451d3d66b7857b5a67e79af887ff2d2bd8bcd9c801bee3",
        "g4_adjudication_run": 0,
        "lambda_decision": "NOT_PERFORMED",
        "lambda_core_updated": False,
        "w8_final_training_runs": 0,
        "test_model_facing_access": 0,
    }:
        fail("B2R protected counters or identities differ")
    if _git("merge-base", "--is-ancestor", STARTING_MAIN_SHA, "HEAD") is None:  # pragma: no cover
        fail("authenticated B2R main is not an ancestor of HEAD")
    return authorization, _load_b2r_index()


def verify_adjudication(path: Path = G4_PATH) -> dict[str, Any]:
    authorization, index = _verify_b2r_and_authority()
    value = _read_json(path)
    expected_keys = {
        "schema_version", "artifact_role", "status", "scientific_execution_authorization",
        "procedural_authorization", "b2r_evidence", "scientific_source",
        "homogeneity_authority", "frozen_protocol", "candidate_ids", "candidate_lambdas",
        "candidates", "A0", "accuracy_tolerance_pp", "accuracy_floor", "accuracy_qualification",
        "primary_psnr_floor_db", "relaxed_psnr_floor_db", "primary_qualifying_lambdas",
        "relaxed_qualifying_lambdas", "selection_tier", "selected_lambda", "adjudicator_output",
        "exact_frozen_adjudicator_identity", "adjudication_boundary", "w7_pilot_weights",
        "execution_proof", "adjudication_id",
    }
    _expect_keys(value, expected_keys, "G-4 adjudication")
    if value["adjudication_id"] != G4_OUTER_PREFIX + canonical_sha256({key: item for key, item in value.items() if key != "adjudication_id"}):
        fail("G-4 outer ID does not authenticate its body")
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_role"] != "W7_G4_ADJUDICATION":
        fail("G-4 role/schema differs")
    if value["scientific_execution_authorization"] != "W7_C_PROCEDURAL_AUTHORIZATION_ONLY":
        fail("G-4 authorization scope differs")
    auth_ref = value["procedural_authorization"]
    _expect_keys(auth_ref, {"authorization_id", "path", "file_sha256", "git_blob_sha1"}, "G-4 authorization reference")
    expected_auth_ref = {
        "authorization_id": authorization["authorization_id"],
        "path": str(AUTHORIZATION_PATH.relative_to(REPO)),
        "file_sha256": _sha256_file(AUTHORIZATION_PATH),
        "git_blob_sha1": _git("hash-object", str(AUTHORIZATION_PATH.relative_to(REPO))),
    }
    if auth_ref != expected_auth_ref:
        fail("G-4 authorization reference differs")
    if value["b2r_evidence"] != [dict(item) for item in B2R_EVIDENCE]:
        fail("G-4 B2R binding differs")
    if value["scientific_source"] != {
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "adjudicator_path": ADJUDICATOR_PATH,
        "adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
        "scientific_source_adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
    }:
        fail("G-4 scientific source binding differs")
    if value["homogeneity_authority"] != authorization["homogeneity_authority"] or value["frozen_protocol"] != FROZEN_PROTOCOL:
        fail("G-4 homogeneity/protocol binding differs")
    if _git("hash-object", ADJUDICATOR_PATH) != ADJUDICATOR_BLOB or _git("rev-parse", f"{SCIENTIFIC_SOURCE_COMMIT}:{ADJUDICATOR_PATH}") != ADJUDICATOR_BLOB:
        fail("frozen adjudicator blob equality failed")

    candidates = _expected_candidate_metrics(index)
    if value["candidate_ids"] != [row["candidate_id"] for row in candidates] or value["candidate_lambdas"] != FROZEN_PROTOCOL["lambda_grid"]:
        fail("G-4 candidate ID/grid binding differs")
    if value["candidates"] != candidates:
        fail("G-4 selected candidate metrics differ from authenticated B2R")
    expected = reconstruct_frozen_decision(candidates)
    accuracy = [
        {
            "lambda": row["lambda"],
            "top1_accuracy": row["validation"]["top1_accuracy"],
            "qualified": row["validation"]["top1_accuracy"] >= expected["accuracy_floor"],
        }
        for row in candidates
    ]
    if value["A0"] != expected["baseline_lambda_zero_top1"] or value["accuracy_tolerance_pp"] != expected["accuracy_tolerance_pp"] or value["accuracy_floor"] != expected["accuracy_floor"] or value["accuracy_qualification"] != accuracy:
        fail("G-4 accuracy baseline/tolerance/floor qualification differs")
    if value["primary_psnr_floor_db"] != expected["primary_psnr_floor_db"] or value["relaxed_psnr_floor_db"] != expected["relaxed_psnr_floor_db"]:
        fail("G-4 PSNR floors differ")
    for key in ("status", "primary_qualifying_lambdas", "relaxed_qualifying_lambdas", "selection_tier", "selected_lambda"):
        if value[key] != expected.get(key):
            fail(f"G-4 persisted {key} differs from frozen adjudicator")
    if value["adjudicator_output"] != expected:
        fail("persisted G-4 output differs from independent frozen-rule reconstruction")

    expected_boundary = {
        "g4_adjudication_run": 1, "g4_adjudications": 1, "lambda_decision": "PERFORMED",
        "lambda_core_updated": False, "w8_final_training_runs": 0, "w8_state": "UNOPENED",
        "test_model_facing_access": 0, "learned_test_inference": 0, "test_access": 0,
        "test_state": "SEALED",
    }
    if value["adjudication_boundary"] != expected_boundary:
        fail("G-4 protected boundary differs")
    if value["w7_pilot_weights"] != {
        "w8_initialization_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
        "optimizer_state_transfer": False,
        "scheduler_state_transfer": False,
        "scaler_state_transfer": False,
        "checkpoint_initialization_transfer": False,
    }:
        fail("W7 pilot W8-eligibility boundary differs")
    if value["execution_proof"] != {
        "adjudicator_invocations": 1,
        "checkpoint_opened": False,
        "model_inference_performed": False,
        "validation_inference_performed": False,
        "psnr_inference_performed": False,
        "papr_inference_performed": False,
        "training_performed": False,
    }:
        fail("G-4 validation-only execution proof differs")
    if not expected["primary_qualifying_lambdas"] or value["selected_lambda"] != min(expected["primary_qualifying_lambdas"]):
        fail("selected lambda is not the numeric minimum of the primary qualifying tier")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--path", type=Path, default=G4_PATH)
    args = parser.parse_args(argv)
    try:
        value = verify_adjudication(args.path)
    except (VerificationError, RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "PASS",
        "adjudication_id": value["adjudication_id"],
        "selected_lambda": value["selected_lambda"],
        "selection_tier": value["selection_tier"],
        "g4_adjudication_run": value["adjudication_boundary"]["g4_adjudication_run"],
        "w8_final_training_runs": value["adjudication_boundary"]["w8_final_training_runs"],
        "test_model_facing_access": value["adjudication_boundary"]["test_model_facing_access"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
