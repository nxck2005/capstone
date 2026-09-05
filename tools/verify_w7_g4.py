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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from evaluation.g10_spec_compatibility import (  # noqa: E402
    PREDECESSOR_COMMIT as AM94_PREDECESSOR_COMMIT,
    load as load_am94_spec_compatibility,
)
from gen_w7_g4_authorization import (  # noqa: E402
    ADJUDICATOR_BLOB,
    ADJUDICATOR_PATH,
    AUTHORIZATION_PATH,
    B2R_EVIDENCE,
    FROZEN_PROTOCOL,
    SCIENTIFIC_SOURCE_COMMIT,
    STARTING_MAIN_SHA,
    canonical_sha256,
    verify_authorization,
)
from training.deterministic_core import canonical_sha256 as frozen_canonical_sha256  # noqa: E402
from verify_w7_b2r import verify_artifacts  # noqa: E402

G4_PATH = REPO / "results/learned/w7/w7_g4_result.json"
TERMINAL_PATH = REPO / "results/learned/w7/w7_completion.json"
G4_OUTER_PREFIX = "w7g4adjudication-"
SCHEMA_VERSION = 1
TERMINAL_PREFIX = "w7completion-"
TERMINAL_SELECTED_LAMBDA = 3.0
TERMINAL_SELECTED_STATUS = "selected_at_G-4"
TERMINAL_W7C_AUTH_CI = {
    "run_id": 33333468110,
    "job_id": 99316097630,
    "head_sha": "a5d60d8704e2b3c1e2680d1bea606f2b7a8266fc",
    "status": "completed",
    "conclusion": "success",
    "workflow": "ci",
}
TERMINAL_B2R_CI = {
    "run_id": 33330122577,
    "job_id": 99307087064,
    "head_sha": STARTING_MAIN_SHA,
    "status": "completed",
    "conclusion": "success",
    "workflow": "ci",
}
TERMINAL_B2R_CI_CLOSURE = {
    "path": "audit/w7-b2r-ci-provenance-closure-2026-08-30.md",
    "file_sha256": "081177997c0c3eb2fa27fdcdc1cbac4b4b6c333d12f4c7dbea0dfbd6e19d1816",
    "git_blob_sha1": "117a4b0ba402176afe7e5a5c23b7f2d4d138783c",
    "accepted_terminal_ci": TERMINAL_B2R_CI,
}
TERMINAL_UPSTREAM = (
    ("W5_REPAIRED_COMPLETION", "results/learned/w5/w5_gradscaler_accounting_repair_completion.json", "repair_id", "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89", "fdfc1515139afc4796156b88c80f11e939d85d7e947a50c39e67b88984193dd7"),
    ("W6_COMPLETION", "results/baseline/w6/w6_completion.json", "completion_id", "w6completion-f992e38e553dce4075406ef8f08df0d42feb2a141a3b00b0ae29a0490e834515", "8fcad25149eb1610c5a5a44d2eae084267b2330b601c0ed501466ad7e1bde2e3"),
    ("W7_A_COMPLETION", "results/learned/w7/w7_a_completion.json", "completion_id", "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17", "61f6de39d43cee82d8ae05e7e9ada33fc659f1e57574a437509343e343c8d2ff"),
    ("W7_A_TEST_HARDENING_COMPLETION", "results/learned/w7/w7_a_test_hardening_completion.json", "completion_id", "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3", "d54e5aa7507f7d9e976fdac029ead08c346acab53c17ca1254773df16dad2bf2"),
    ("W7_B1_EXECUTION_AUTHORIZATION", "results/learned/w7/w7_execution_authorization.json", "authorization_id", "w7auth-1d44b66884f48f980576dde94c43eb745227b4ecc48fb964acf90285a854862d", "5784ec7ece15051586f915e4e834ca732778f09c2ce537dbd8af4f6e597a8349"),
    ("W7_B1_SOURCE_MANIFEST", "results/learned/w7/w7_b1_source_manifest.json", "manifest_id", "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f", "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392"),
    ("W7_B1_COMPLETION", "results/learned/w7/w7_b1_completion.json", "completion_id", "w7b1completion-701c12a084aa7a5b47b1d05f74a4b31ae5cbce4df622a82f0aafcdc9cd5228a3", "c27eedf99cf158af436fda507be99ab1ed4e1753a7e5b0389098818045d30472"),
)
TERMINAL_GENERATED_SPEC_PATHS = (
    "spec/params.generated.yaml",
    "spec/DATASHEET.md",
    "spec/concerns/amendments.md",
    "spec/concerns/baseline.md",
    "spec/concerns/demo.md",
    "spec/concerns/experiments.md",
    "spec/concerns/hardware.md",
    "spec/concerns/programme.md",
    "spec/concerns/roadmap.md",
    "spec/concerns/system.md",
)
# AM-93 changes only the current normative views.  The W7 terminal embeds the
# pre-AM-93 bytes, so this later additive record lets the old verifier retain
# those exact bytes without treating current spec drift as a general exception.
SPEC_COMPATIBILITY_PATH = REPO / "results/learned/w7/w7_spec_additive_compatibility.json"
SPEC_COMPATIBILITY_PREFIX = "w7speccompat-"


class VerificationError(RuntimeError):
    """A consequential W7-C invariant failed."""


def fail(message: str) -> None:
    raise VerificationError(message)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _is_ancestor(base: str, tip: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, tip],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        fail(f"cannot authenticate B2R main ancestry: {exc}")
    return result.returncode == 0


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
            result["adjudication_id"] = "g4adjudication-" + frozen_canonical_sha256(result)
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
    result["adjudication_id"] = "g4adjudication-" + frozen_canonical_sha256(result)
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
    if not _is_ancestor(STARTING_MAIN_SHA, _git("rev-parse", "HEAD")):
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


def _terminal_ref(
    path_text: str,
    *,
    identity_field: str | None = None,
    identity: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = REPO / path_text
    if not path.is_file() or path.is_symlink():
        fail(f"terminal-bound artifact is missing or unsafe: {path_text}")
    if identity_field is not None:
        value = _read_json(path)
        if value.get(identity_field) != identity:
            fail(f"terminal-bound identity differs: {path_text}")
    file_sha = _sha256_file(path)
    if expected_sha256 is not None and file_sha != expected_sha256:
        fail(f"terminal-bound file SHA differs: {path_text}")
    return {
        "path": path_text,
        "identity_field": identity_field,
        "identity": identity,
        "file_sha256": file_sha,
        "git_blob_sha1": _git("hash-object", path_text),
    }


def _git_bytes_at(commit: str, path: str) -> tuple[bytes, str]:
    try:
        raw = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
        blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot read historical W7 spec view {commit}:{path}: {exc}")
    return raw, blob


def _historical_spec_ref(ref: Mapping[str, Any], commit: str) -> dict[str, Any]:
    _expect_keys(
        ref,
        {"path", "identity_field", "identity", "file_sha256", "git_blob_sha1"},
        "historical W7 spec reference",
    )
    if ref["identity_field"] is not None or ref["identity"] is not None:
        fail("historical W7 spec reference carries an unexpected identity")
    raw, blob = _git_bytes_at(commit, str(ref["path"]))
    expected = {
        "path": str(ref["path"]),
        "identity_field": None,
        "identity": None,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": blob,
    }
    if dict(ref) != expected:
        fail(f"historical W7 spec view bytes differ: {ref.get('path')}")
    return expected


def _terminal_spec_view_refs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the immutable pre-AM-93 refs embedded by the W7 terminal.

    Before AM-93 the current files were the terminal refs.  After it, the
    additive compatibility artifact pins both sides and authenticates the old
    refs directly from their historical Git commit; no broad "ignore current
    params" behaviour is permitted.
    """

    if not SPEC_COMPATIBILITY_PATH.exists():
        current_source = _terminal_ref("spec/SPEC.md")
        current_generated = [_terminal_ref(item) for item in TERMINAL_GENERATED_SPEC_PATHS]
        return current_source, current_generated
    compatibility = _read_json(SPEC_COMPATIBILITY_PATH)
    expected_keys = {
        "schema_version", "artifact_role", "status", "terminal_completion_id",
        "historical_commit", "historical_spec_views", "current_spec_views",
        "allowed_change", "scientific_effect", "compatibility_id",
    }
    _expect_keys(compatibility, expected_keys, "W7 spec compatibility")
    body = dict(compatibility)
    identifier = body.pop("compatibility_id")
    if identifier != SPEC_COMPATIBILITY_PREFIX + canonical_sha256(body):
        fail("W7 spec compatibility ID does not authenticate its body")
    if compatibility["schema_version"] != 1 or compatibility["artifact_role"] != "W7_HISTORICAL_SPEC_ADDITIVE_COMPATIBILITY" or compatibility["status"] != "ADDITIVE_FAIL_CLOSED":
        fail("W7 spec compatibility role/status differs")
    if compatibility["terminal_completion_id"] != "w7completion-fcd91d565ec3c98e1aff6c69a71b86af398971e7f8e898efa0499dc6e5c3dc1f":
        fail("W7 spec compatibility terminal binding differs")
    commit = compatibility["historical_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit) or commit != "11be6d6f519094fe37ada347bdc678c99d066521":  # literal-ok: Git SHA-1 width
        fail("W7 spec compatibility historical commit differs")
    historical = compatibility["historical_spec_views"]
    _expect_keys(historical, {"source", "generated_views"}, "historical W7 spec views")
    if len(historical["generated_views"]) != len(TERMINAL_GENERATED_SPEC_PATHS):
        fail("historical W7 generated view count differs")
    expected_historical_source = _historical_spec_ref(historical["source"], commit)
    expected_historical_generated = [
        _historical_spec_ref(ref, commit) for ref in historical["generated_views"]
    ]
    if [ref["path"] for ref in expected_historical_generated] != list(TERMINAL_GENERATED_SPEC_PATHS):
        fail("historical W7 generated view order differs")
    current = compatibility["current_spec_views"]
    _expect_keys(current, {"source", "generated_views"}, "current W7 spec views")
    expected_am93_source = _historical_spec_ref(current["source"], AM94_PREDECESSOR_COMMIT)
    expected_am93_generated = [
        _historical_spec_ref(ref, AM94_PREDECESSOR_COMMIT)
        for ref in current["generated_views"]
    ]
    if [ref["path"] for ref in expected_am93_generated] != list(TERMINAL_GENERATED_SPEC_PATHS):
        fail("AM-93 generated view order differs")
    try:
        load_am94_spec_compatibility(REPO)
    except Exception as exc:
        fail(f"AM-94 successor spec compatibility differs: {exc}")
    if compatibility["allowed_change"] != {
        "amendment": "AM-93",
        "parameter": "params.learned_system.checkpoint_selection_snr_db",
        "resolution": "params.channel.train_snr_db_fixed",
        "paths": ["spec/SPEC.md", *TERMINAL_GENERATED_SPEC_PATHS],
        "schedule": "G-4 -> W8 -> W9/G-10/G-11",
    }:
        fail("W7 spec compatibility allowance differs")
    if compatibility["scientific_effect"] != {
        "w7_result_changed": False,
        "g4_result_changed": False,
        "w8_science_performed": False,
        "test_access": 0,
        "scope": "checkpoint-selection-SNR clarification and schedule-wording correction only",
    }:
        fail("W7 spec compatibility scientific boundary differs")
    if get("learned_system.checkpoint_selection_snr_db") != "train_snr_db_fixed":
        fail("current W8 checkpoint-selection SNR is not the AM-93 binding")
    return expected_historical_source, expected_historical_generated


def _expected_terminal_upstream() -> list[dict[str, Any]]:
    upstream = (
        ("W5_REPAIRED_COMPLETION", "results/learned/w5/w5_gradscaler_accounting_repair_completion.json", "repair_id", "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89", "fdfc1515139afc4796156b88c80f11e939d85d7e947a50c39e67b88984193dd7"),
        ("W6_COMPLETION", "results/baseline/w6/w6_completion.json", "completion_id", "w6completion-f992e38e553dce4075406ef8f08df0d42feb2a141a3b00b0ae29a0490e834515", "8fcad25149eb1610c5a5a44d2eae084267b2330b601c0ed501466ad7e1bde2e3"),
        ("W7_A_COMPLETION", "results/learned/w7/w7_a_completion.json", "completion_id", "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17", "61f6de39d43cee82d8ae05e7e9ada33fc659f1e57574a437509343e343c8d2ff"),
        ("W7_A_TEST_HARDENING_COMPLETION", "results/learned/w7/w7_a_test_hardening_completion.json", "completion_id", "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3", "d54e5aa7507f7d9e976fdac029ead08c346acab53c17ca1254773df16dad2bf2"),
        ("W7_B1_EXECUTION_AUTHORIZATION", "results/learned/w7/w7_execution_authorization.json", "authorization_id", "w7auth-1d44b66884f48f980576dde94c43eb745227b4ecc48fb964acf90285a854862d", "5784ec7ece15051586f915e4e834ca732778f09c2ce537dbd8af4f6e597a8349"),
        ("W7_B1_SOURCE_MANIFEST", "results/learned/w7/w7_b1_source_manifest.json", "manifest_id", "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f", "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392"),
        ("W7_B1_COMPLETION", "results/learned/w7/w7_b1_completion.json", "completion_id", "w7b1completion-701c12a084aa7a5b47b1d05f74a4b31ae5cbce4df622a82f0aafcdc9cd5228a3", "c27eedf99cf158af436fda507be99ab1ed4e1753a7e5b0389098818045d30472"),
    )
    refs = []
    for role, path, identity_field, identity, file_sha in upstream:
        ref = _terminal_ref(path, identity_field=identity_field, identity=identity, expected_sha256=file_sha)
        ref["role"] = role
        refs.append(ref)
    return refs


def verify_terminal_completion(path: Path = TERMINAL_PATH, *, g4: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticate the W7 terminal state without scientific execution."""

    authorization, _ = _verify_b2r_and_authority()
    if g4 is None:
        g4 = verify_adjudication(G4_PATH)
    value = _read_json(path)
    expected_keys = {
        "schema_version", "artifact_role", "status", "upstream", "b2r_terminal",
        "b2r_ci_provenance_closure", "w7c_authorization", "g4_adjudication",
        "normative_lambda", "protected_counters", "w7_pilot_weights",
        "scientific_boundary", "future_boundary", "completion_id",
    }
    _expect_keys(value, expected_keys, "W7 terminal completion")
    if value["completion_id"] != TERMINAL_PREFIX + canonical_sha256({key: item for key, item in value.items() if key != "completion_id"}):
        fail("W7 terminal completion ID does not authenticate its body")
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_role"] != "W7_TERMINAL_COMPLETION" or value["status"] != "W7_GREEN_CLOSED":
        fail("W7 terminal completion role/status differs")
    if value["upstream"] != _expected_terminal_upstream():
        fail("W7 terminal upstream completion binding differs")

    evidence_by_role = {item["role"]: dict(item) for item in B2R_EVIDENCE}
    expected_b2r = {
        "candidate_index_id": evidence_by_role["B2R_CANDIDATE_INDEX"]["content_id"],
        "completion_id": evidence_by_role["B2R_COMPLETION"]["content_id"],
        "common_noise_audit_id": evidence_by_role["B2R_COMMON_NOISE_AUDIT"]["content_id"],
        "custody_id": evidence_by_role["B2R_CHECKPOINT_CUSTODY"]["content_id"],
        "reconciliation_id": evidence_by_role["B2R_RECONCILIATION"]["content_id"],
        "evidence": [dict(item) for item in B2R_EVIDENCE],
        "candidate_count": 5,
        "completed_epoch_cycles": 500,
        "checkpoint_count": 500,
        "g4_adjudication_run_before_closeout": 0,
        "lambda_decision_before_closeout": "NOT_PERFORMED",
        "lambda_core_updated_before_closeout": False,
    }
    if value["b2r_terminal"] != expected_b2r:
        fail("W7 terminal B2R binding differs")
    expected_ci_closure = {
        "path": "audit/w7-b2r-ci-provenance-closure-2026-08-30.md",
        "file_sha256": "081177997c0c3eb2fa27fdcdc1cbac4b4b6c333d12f4c7dbea0dfbd6e19d1816",
        "git_blob_sha1": "117a4b0ba402176afe7e5a5c23b7f2d4d138783c",
        "accepted_terminal_ci": TERMINAL_B2R_CI,
    }
    if value["b2r_ci_provenance_closure"] != expected_ci_closure:
        fail("B2R CI/provenance closure binding differs")
    if _terminal_ref(expected_ci_closure["path"], expected_sha256=expected_ci_closure["file_sha256"])["git_blob_sha1"] != expected_ci_closure["git_blob_sha1"]:
        fail("B2R CI/provenance closure Git blob differs")

    expected_auth_ref = _terminal_ref(
        "results/learned/w7/w7_g4_procedural_authorization.json",
        identity_field="authorization_id",
        identity=authorization["authorization_id"],
        expected_sha256="b1ad0d032eac17bd07d785323ae4ff3468017b53d971f2226abff0e880faa584",
    )
    expected_auth = {
        **expected_auth_ref,
        "authorization_scope": "EXACTLY_ONE_DETERMINISTIC_G4_ADJUDICATION",
        "ci": TERMINAL_W7C_AUTH_CI,
        "commit": TERMINAL_W7C_AUTH_CI["head_sha"],
    }
    if value["w7c_authorization"] != expected_auth:
        fail("W7-C authorization/CI binding differs")

    expected_g4_ref = _terminal_ref(
        "results/learned/w7/w7_g4_result.json",
        identity_field="adjudication_id",
        identity=g4["adjudication_id"],
    )
    expected_candidate_table = [
        {
            "candidate_id": item["candidate_id"],
            "lambda": item["lambda"],
            "selected_epoch": item["selected_epoch"],
            "selected_checkpoint_id": item["selected_checkpoint_id"],
        }
        for item in g4["candidates"]
    ]
    expected_g4 = {
        **expected_g4_ref,
        "status": g4["status"],
        "selection_tier": g4["selection_tier"],
        "selected_lambda": g4["selected_lambda"],
        "candidate_table": expected_candidate_table,
        "candidate_ids": list(g4["candidate_ids"]),
        "candidate_lambdas": list(g4["candidate_lambdas"]),
        "inner_adjudication_id": g4["adjudicator_output"]["adjudication_id"],
        "g4_adjudication_run": g4["adjudication_boundary"]["g4_adjudication_run"],
        "accuracy_baseline_A0": g4["A0"],
        "accuracy_floor": g4["accuracy_floor"],
        "accuracy_tolerance_pp": g4["accuracy_tolerance_pp"],
        "primary_qualifying_lambdas": list(g4["primary_qualifying_lambdas"]),
        "relaxed_qualifying_lambdas": list(g4["relaxed_qualifying_lambdas"]),
    }
    if value["g4_adjudication"] != expected_g4:
        fail("W7 terminal G-4 binding differs")

    expected_source_ref, expected_generated_refs = _terminal_spec_view_refs()
    if value["normative_lambda"] != {
        "source_of_truth": "spec/SPEC.md",
        "lambda_core": TERMINAL_SELECTED_LAMBDA,
        "lambda_status": TERMINAL_SELECTED_STATUS,
        "provisional_g4_status_cleared": True,
        "spec_views": {"source": expected_source_ref, "generated_views": expected_generated_refs},
    }:
        fail("W7 terminal normative lambda binding differs")
    if get("learned_system.lambda_core") != TERMINAL_SELECTED_LAMBDA or get("learned_system.lambda_status") != TERMINAL_SELECTED_STATUS:
        fail("current normative lambda state is not the authenticated G-4 selection")

    if value["protected_counters"] != {
        "w7_lambda_pilot_runs": 5,
        "completed_epoch_cycles": 500,
        "checkpoint_count": 500,
        "g4_adjudications": 1,
        "g4_adjudication_run": 1,
        "lambda_decision": "PERFORMED",
        "lambda_core_updated": True,
        "w8_final_training_runs": 0,
        "w8_state": "UNOPENED",
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "test_state": "SEALED",
    }:
        fail("W7 terminal protected counters differ")
    if value["w7_pilot_weights"] != {
        "status": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
        "w8_initialization_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
        "optimizer_state_transfer": False,
        "scheduler_state_transfer": False,
        "scaler_state_transfer": False,
        "checkpoint_initialization_transfer": False,
    }:
        fail("W7 pilot W8-ineligibility binding differs")
    if value["scientific_boundary"] != {
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "adjudicator_path": ADJUDICATOR_PATH,
        "adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
        "scientific_source_adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
        "adjudicator_invocations": 1,
        "training_performed": False,
        "model_inference_performed": False,
        "validation_inference_performed": False,
        "psnr_inference_performed": False,
        "papr_inference_performed": False,
        "checkpoint_opened": False,
    }:
        fail("W7 terminal scientific boundary differs")
    if value["future_boundary"] != {
        "w8_state": "UNOPENED",
        "w8_requires_separate_authorization": True,
        "w8_final_training_runs": 0,
        "w8_initialization_from_w7_pilot": False,
        "test_state": "SEALED",
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "next_action": "SEPARATE_W8_FINAL_MULTI_SEED_TRAINING_AUTHORIZATION",
    }:
        fail("W7 terminal future boundary differs")
    if _git("hash-object", ADJUDICATOR_PATH) != ADJUDICATOR_BLOB or _git("rev-parse", f"{SCIENTIFIC_SOURCE_COMMIT}:{ADJUDICATOR_PATH}") != ADJUDICATOR_BLOB:
        fail("terminal frozen adjudicator blob equality failed")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--path", type=Path, default=G4_PATH)
    args = parser.parse_args(argv)
    try:
        value = verify_adjudication(args.path)
        terminal = None
        if args.path == G4_PATH and TERMINAL_PATH.exists():
            terminal = verify_terminal_completion(TERMINAL_PATH, g4=value)
    except (VerificationError, RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    output = {
        "status": "PASS",
        "adjudication_id": value["adjudication_id"],
        "selected_lambda": value["selected_lambda"],
        "selection_tier": value["selection_tier"],
        "g4_adjudication_run": value["adjudication_boundary"]["g4_adjudication_run"],
        "w8_final_training_runs": value["adjudication_boundary"]["w8_final_training_runs"],
        "test_model_facing_access": value["adjudication_boundary"]["test_model_facing_access"],
    }
    if terminal is not None:
        output["terminal_completion_id"] = terminal["completion_id"]
        output["lambda_core"] = terminal["normative_lambda"]["lambda_core"]
        output["lambda_status"] = terminal["normative_lambda"]["lambda_status"]
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
