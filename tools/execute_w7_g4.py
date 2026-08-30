#!/usr/bin/env python3
"""Apply the frozen G-4 adjudicator exactly once to authenticated B2R evidence.

This is a validation-only boundary.  It reads compact JSON evidence, never
opens a checkpoint, never runs inference and refuses to replace an existing
adjudication or terminal record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from adjudication.w7_g4 import adjudicate_g4  # noqa: E402
from gen_w7_g4_authorization import (  # noqa: E402
    ADJUDICATOR_BLOB,
    ADJUDICATOR_PATH,
    AUTHORIZATION_PATH,
    B2R_EVIDENCE,
    SCIENTIFIC_SOURCE_COMMIT,
    STARTING_MAIN_SHA,
    canonical_sha256,
    rendered_json,
    verify_authorization,
)
from verify_w7_b2r import verify_artifacts  # noqa: E402

G4_PATH = REPO / "results/learned/w7/w7_g4_result.json"
TERMINAL_PATH = REPO / "results/learned/w7/w7_completion.json"
OUTER_PREFIX = "w7g4adjudication-"
SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace existing immutable artifact: {path}")
    temporary = path.parent / f".{path.name}.staging-{os.getpid()}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _candidate_metrics(index: dict[str, Any]) -> list[dict[str, Any]]:
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
                "snr_db": 7,
                "n_correct": raw["selected_validation"]["n_correct"],
                "n_total": raw["selected_validation"]["n_total"],
                "top1_accuracy": raw["selected_validation"]["top1_accuracy"],
            },
            "psnr_evaluation": {
                "snr_db": 15,
                "mean_psnr_db": raw["psnr_evaluation"]["psnr_db"],
            },
            "factual_papr": {
                "snr_db": 15,
                "mean_papr": selected["derived_mean_papr"],
            },
        })
    return rows


def build_adjudication(result: dict[str, Any], index: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_metrics(index)
    accuracy = [
        {
            "lambda": row["lambda"],
            "top1_accuracy": row["validation"]["top1_accuracy"],
            "qualified": row["validation"]["top1_accuracy"] >= result["accuracy_floor"],
        }
        for row in candidates
    ]
    auth_path = AUTHORIZATION_PATH.relative_to(REPO).as_posix()
    auth_git_blob = _git("hash-object", auth_path)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_G4_ADJUDICATION",
        "status": result["status"],
        "scientific_execution_authorization": "W7_C_PROCEDURAL_AUTHORIZATION_ONLY",
        "procedural_authorization": {
            "authorization_id": authorization["authorization_id"],
            "path": auth_path,
            "file_sha256": _sha256_file(AUTHORIZATION_PATH),
            "git_blob_sha1": auth_git_blob,
        },
        "b2r_evidence": [dict(item) for item in B2R_EVIDENCE],
        "scientific_source": {
            "source_commit": SCIENTIFIC_SOURCE_COMMIT,
            "adjudicator_path": ADJUDICATOR_PATH,
            "adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
            "scientific_source_adjudicator_git_blob_sha1": ADJUDICATOR_BLOB,
        },
        "homogeneity_authority": authorization["homogeneity_authority"],
        "frozen_protocol": authorization["frozen_protocol"],
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_lambdas": [row["lambda"] for row in candidates],
        "candidates": candidates,
        "A0": result["baseline_lambda_zero_top1"],
        "accuracy_tolerance_pp": result["accuracy_tolerance_pp"],
        "accuracy_floor": result["accuracy_floor"],
        "accuracy_qualification": accuracy,
        "primary_psnr_floor_db": result["primary_psnr_floor_db"],
        "relaxed_psnr_floor_db": result["relaxed_psnr_floor_db"],
        "primary_qualifying_lambdas": result["primary_qualifying_lambdas"],
        "relaxed_qualifying_lambdas": result["relaxed_qualifying_lambdas"],
        "selection_tier": result["selection_tier"],
        "selected_lambda": result["selected_lambda"],
        "adjudicator_output": result,
        "exact_frozen_adjudicator_identity": {
            "path": ADJUDICATOR_PATH,
            "git_blob_sha1": ADJUDICATOR_BLOB,
            "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
            "scientific_source_git_blob_sha1": ADJUDICATOR_BLOB,
        },
        "adjudication_boundary": {
            "g4_adjudication_run": 1,
            "g4_adjudications": 1,
            "lambda_decision": "PERFORMED",
            "lambda_core_updated": False,
            "w8_final_training_runs": 0,
            "w8_state": "UNOPENED",
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "test_access": 0,
            "test_state": "SEALED",
        },
        "w7_pilot_weights": {
            "w8_initialization_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
            "optimizer_state_transfer": False,
            "scheduler_state_transfer": False,
            "scaler_state_transfer": False,
            "checkpoint_initialization_transfer": False,
        },
        "execution_proof": {
            "adjudicator_invocations": 1,
            "checkpoint_opened": False,
            "model_inference_performed": False,
            "validation_inference_performed": False,
            "psnr_inference_performed": False,
            "papr_inference_performed": False,
            "training_performed": False,
        },
    }
    body["adjudication_id"] = OUTER_PREFIX + canonical_sha256(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute",))
    args = parser.parse_args(argv)
    if G4_PATH.exists() or TERMINAL_PATH.exists():
        raise SystemExit("refusing a second G-4 adjudication or terminal completion")
    authorization = verify_authorization(AUTHORIZATION_PATH)
    if _git("merge-base", "--is-ancestor", STARTING_MAIN_SHA, "HEAD") is None:  # pragma: no cover - git exits on failure
        raise SystemExit("authenticated B2R main is not an ancestor of the execution checkout")
    if _git("hash-object", ADJUDICATOR_PATH) != ADJUDICATOR_BLOB:
        raise SystemExit("current frozen adjudicator blob differs")
    if _git("rev-parse", f"{SCIENTIFIC_SOURCE_COMMIT}:{ADJUDICATOR_PATH}") != ADJUDICATOR_BLOB:
        raise SystemExit("scientific-source frozen adjudicator blob differs")

    b2r = verify_artifacts(REPO / "results/learned/w7", reauthenticate_upstream=False)
    if b2r["candidate_count"] != 5 or b2r["completed_epoch_cycles"] != 500 or b2r["checkpoint_count"] != 500:
        raise SystemExit("authenticated B2R counters differ")
    if b2r["g4_adjudication_run"] != 0 or b2r["lambda_decision"] != "NOT_PERFORMED" or b2r["lambda_core_updated"] is not False:
        raise SystemExit("B2R is not at the accepted pre-G4 boundary")
    index = json.loads((REPO / "results/learned/w7/w7_b2_reconciliation_index.json").read_bytes())
    candidates = [item["candidate_completion"]["value"] for item in index["candidates"]]
    result = adjudicate_g4(candidates)
    artifact = build_adjudication(result, index, authorization)
    _publish_once(G4_PATH, rendered_json(artifact))
    print(json.dumps({
        "status": "ADJUDICATED",
        "adjudication_id": artifact["adjudication_id"],
        "selected_lambda": artifact["selected_lambda"],
        "selection_tier": artifact["selection_tier"],
        "path": str(G4_PATH.relative_to(REPO)),
        "file_sha256": _sha256_file(G4_PATH),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
