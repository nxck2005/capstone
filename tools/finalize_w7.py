#!/usr/bin/env python3
"""Publish the immutable W7 terminal completion after the selected G-4 result.

This tool is a read-only closeout boundary.  It authenticates the already
published W7-C result and upstream compact evidence, checks the generated
normative parameter state, and writes one content-addressed completion record.
It never opens a checkpoint, performs inference, or calls the G-4 adjudicator.
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

from config.params import get  # noqa: E402
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
from verify_w7_g4 import G4_PATH, verify_adjudication  # noqa: E402

TERMINAL_PATH = REPO / "results/learned/w7/w7_completion.json"
TERMINAL_PREFIX = "w7completion-"
SCHEMA_VERSION = 1
SELECTED_LAMBDA = 3.0
SELECTED_STATUS = "selected_at_G-4"

UPSTREAM = (
    ("W5_REPAIRED_COMPLETION", "results/learned/w5/w5_gradscaler_accounting_repair_completion.json", "repair_id", "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89", "fdfc1515139afc4796156b88c80f11e939d85d7e947a50c39e67b88984193dd7"),
    ("W6_COMPLETION", "results/baseline/w6/w6_completion.json", "completion_id", "w6completion-f992e38e553dce4075406ef8f08df0d42feb2a141a3b00b0ae29a0490e834515", "8fcad25149eb1610c5a5a44d2eae084267b2330b601c0ed501466ad7e1bde2e3"),
    ("W7_A_COMPLETION", "results/learned/w7/w7_a_completion.json", "completion_id", "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17", "61f6de39d43cee82d8ae05e7e9ada33fc659f1e57574a437509343e343c8d2ff"),
    ("W7_A_TEST_HARDENING_COMPLETION", "results/learned/w7/w7_a_test_hardening_completion.json", "completion_id", "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3", "d54e5aa7507f7d9e976fdac029ead08c346acab53c17ca1254773df16dad2bf2"),
    ("W7_B1_EXECUTION_AUTHORIZATION", "results/learned/w7/w7_execution_authorization.json", "authorization_id", "w7auth-1d44b66884f48f980576dde94c43eb745227b4ecc48fb964acf90285a854862d", "5784ec7ece15051586f915e4e834ca732778f09c2ce537dbd8af4f6e597a8349"),
    ("W7_B1_SOURCE_MANIFEST", "results/learned/w7/w7_b1_source_manifest.json", "manifest_id", "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f", "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392"),
    ("W7_B1_COMPLETION", "results/learned/w7/w7_b1_completion.json", "completion_id", "w7b1completion-701c12a084aa7a5b47b1d05f74a4b31ae5cbce4df622a82f0aafcdc9cd5228a3", "c27eedf99cf158af436fda507be99ab1ed4e1753a7e5b0389098818045d30472"),
)

GENERATED_SPEC_PATHS = (
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

B2R_CI = {
    "run_id": 33330122577,
    "job_id": 99307087064,
    "head_sha": STARTING_MAIN_SHA,
    "status": "completed",
    "conclusion": "success",
    "workflow": "ci",
}
W7C_AUTH_CI = {
    "run_id": 33333468110,
    "job_id": 99316097630,
    "head_sha": "a5d60d8704e2b3c1e2680d1bea606f2b7a8266fc",
    "status": "completed",
    "conclusion": "success",
    "workflow": "ci",
}
B2R_CI_CLOSURE = {
    "path": "audit/w7-b2r-ci-provenance-closure-2026-08-30.md",
    "file_sha256": "081177997c0c3eb2fa27fdcdc1cbac4b4b6c333d12f4c7dbea0dfbd6e19d1816",
    "git_blob_sha1": "117a4b0ba402176afe7e5a5c23b7f2d4d138783c",
    "accepted_terminal_ci": B2R_CI,
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path_text: str, *, identity_field: str | None = None, identity: str | None = None, expected_sha256: str | None = None) -> dict[str, Any]:
    path = REPO / path_text
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required closeout artifact is missing or unsafe: {path_text}")
    value = json.loads(path.read_bytes()) if path.suffix == ".json" else None
    if identity_field is not None:
        if not isinstance(value, dict) or value.get(identity_field) != identity:
            raise RuntimeError(f"upstream identity differs: {path_text}")
    file_sha = _sha256_file(path)
    if expected_sha256 is not None and file_sha != expected_sha256:
        raise RuntimeError(f"upstream file SHA differs: {path_text}")
    return {
        "path": path_text,
        "identity_field": identity_field,
        "identity": identity,
        "file_sha256": file_sha,
        "git_blob_sha1": _git("hash-object", path_text),
    }


def _publish_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace immutable terminal artifact: {path}")
    staging = path.parent / f".{path.name}.staging-{os.getpid()}"
    try:
        with staging.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path, follow_symlinks=False)
        staging.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)


def _upstream_refs() -> list[dict[str, Any]]:
    refs = []
    for role, path, identity_field, identity, file_sha in UPSTREAM:
        ref = _ref(path, identity_field=identity_field, identity=identity, expected_sha256=file_sha)
        ref["role"] = role
        refs.append(ref)
    return refs


def _b2r_summary() -> dict[str, Any]:
    by_role = {item["role"]: dict(item) for item in B2R_EVIDENCE}
    return {
        "candidate_index_id": by_role["B2R_CANDIDATE_INDEX"]["content_id"],
        "completion_id": by_role["B2R_COMPLETION"]["content_id"],
        "common_noise_audit_id": by_role["B2R_COMMON_NOISE_AUDIT"]["content_id"],
        "custody_id": by_role["B2R_CHECKPOINT_CUSTODY"]["content_id"],
        "reconciliation_id": by_role["B2R_RECONCILIATION"]["content_id"],
        "evidence": [dict(item) for item in B2R_EVIDENCE],
        "candidate_count": 5,
        "completed_epoch_cycles": 500,
        "checkpoint_count": 500,
        "g4_adjudication_run_before_closeout": 0,
        "lambda_decision_before_closeout": "NOT_PERFORMED",
        "lambda_core_updated_before_closeout": False,
    }


def _spec_refs() -> dict[str, Any]:
    source = _ref("spec/SPEC.md")
    generated = [_ref(path) for path in GENERATED_SPEC_PATHS]
    return {"source": source, "generated_views": generated}


def build_terminal(g4: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    if g4["selected_lambda"] != SELECTED_LAMBDA or g4["selection_tier"] != "PRIMARY":
        raise RuntimeError("terminal closeout requires the authenticated primary 3.0 selection")
    auth_ref = _ref(
        "results/learned/w7/w7_g4_procedural_authorization.json",
        identity_field="authorization_id",
        identity=authorization["authorization_id"],
        expected_sha256="b1ad0d032eac17bd07d785323ae4ff3468017b53d971f2226abff0e880faa584",
    )
    g4_ref = _ref(
        "results/learned/w7/w7_g4_result.json",
        identity_field="adjudication_id",
        identity=g4["adjudication_id"],
        expected_sha256="06f67ce3bcf6c3d2d8facf3bc014c79676b08b7575390f62f6070d1d8b757e3f",
    )
    candidate_table = [
        {
            "candidate_id": item["candidate_id"],
            "lambda": item["lambda"],
            "selected_epoch": item["selected_epoch"],
            "selected_checkpoint_id": item["selected_checkpoint_id"],
        }
        for item in g4["candidates"]
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_TERMINAL_COMPLETION",
        "status": "W7_GREEN_CLOSED",
        "upstream": _upstream_refs(),
        "b2r_terminal": _b2r_summary(),
        "b2r_ci_provenance_closure": dict(B2R_CI_CLOSURE),
        "w7c_authorization": {
            **auth_ref,
            "authorization_scope": "EXACTLY_ONE_DETERMINISTIC_G4_ADJUDICATION",
            "ci": dict(W7C_AUTH_CI),
            "commit": W7C_AUTH_CI["head_sha"],
        },
        "g4_adjudication": {
            **g4_ref,
            "status": g4["status"],
            "selection_tier": g4["selection_tier"],
            "selected_lambda": g4["selected_lambda"],
            "candidate_table": candidate_table,
            "candidate_ids": list(g4["candidate_ids"]),
            "candidate_lambdas": list(g4["candidate_lambdas"]),
            "inner_adjudication_id": g4["adjudicator_output"]["adjudication_id"],
            "g4_adjudication_run": g4["adjudication_boundary"]["g4_adjudication_run"],
            "accuracy_baseline_A0": g4["A0"],
            "accuracy_floor": g4["accuracy_floor"],
            "accuracy_tolerance_pp": g4["accuracy_tolerance_pp"],
            "primary_qualifying_lambdas": list(g4["primary_qualifying_lambdas"]),
            "relaxed_qualifying_lambdas": list(g4["relaxed_qualifying_lambdas"]),
        },
        "normative_lambda": {
            "source_of_truth": "spec/SPEC.md",
            "lambda_core": SELECTED_LAMBDA,
            "lambda_status": SELECTED_STATUS,
            "provisional_g4_status_cleared": True,
            "spec_views": _spec_refs(),
        },
        "protected_counters": {
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
        },
        "w7_pilot_weights": {
            "status": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
            "w8_initialization_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
            "optimizer_state_transfer": False,
            "scheduler_state_transfer": False,
            "scaler_state_transfer": False,
            "checkpoint_initialization_transfer": False,
        },
        "scientific_boundary": {
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
        },
        "future_boundary": {
            "w8_state": "UNOPENED",
            "w8_requires_separate_authorization": True,
            "w8_final_training_runs": 0,
            "w8_initialization_from_w7_pilot": False,
            "test_state": "SEALED",
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "next_action": "SEPARATE_W8_FINAL_MULTI_SEED_TRAINING_AUTHORIZATION",
        },
    }
    body["completion_id"] = TERMINAL_PREFIX + canonical_sha256(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish",))
    args = parser.parse_args(argv)
    if TERMINAL_PATH.exists() or TERMINAL_PATH.is_symlink():
        raise SystemExit("refusing a second W7 terminal completion")
    authorization = verify_authorization(AUTHORIZATION_PATH)
    b2r = verify_artifacts(REPO / "results/learned/w7", reauthenticate_upstream=False)
    if b2r["candidate_count"] != 5 or b2r["completed_epoch_cycles"] != 500 or b2r["checkpoint_count"] != 500:
        raise SystemExit("B2R terminal counters differ")
    if b2r["g4_adjudication_run"] != 0 or b2r["lambda_decision"] != "NOT_PERFORMED" or b2r["lambda_core_updated"] is not False:
        raise SystemExit("B2R is not the accepted pre-G4 evidence boundary")
    if get("learned_system.lambda_core") != SELECTED_LAMBDA or get("learned_system.lambda_status") != SELECTED_STATUS:
        raise SystemExit("normative lambda state is not the authenticated G-4 selection")
    g4 = verify_adjudication(G4_PATH)
    artifact = build_terminal(g4, authorization)
    _publish_once(TERMINAL_PATH, rendered_json(artifact))
    print(json.dumps({
        "status": "PUBLISHED",
        "completion_id": artifact["completion_id"],
        "path": str(TERMINAL_PATH.relative_to(REPO)),
        "file_sha256": _sha256_file(TERMINAL_PATH),
        "selected_lambda": artifact["g4_adjudication"]["selected_lambda"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
