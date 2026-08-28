#!/usr/bin/env python3
"""Terminal W7-A verifier: pre-execution only, never a W7-B launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from config.params import get  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.w7_protocol import W7_LAMBDA_GRID, protocol_descriptor  # noqa: E402
from gen_w7_a_contract import verify as verify_contract  # noqa: E402
from gen_w7_source_manifest import verify as verify_source_manifest  # noqa: E402
from verify_w7_profile import verify as verify_profile  # noqa: E402


COMPLETION_PATH = REPO / "results/learned/w7/w7_a_completion.json"
CONTRACT_PATH = REPO / "results/learned/w7/w7_a_contract.json"
SCHEMA_PATH = REPO / "spec/schemas/w7_g4_artifacts.schema.json"
SOURCE_MANIFEST_PATH = REPO / "results/learned/w7/w7_source_manifest.json"
PROFILE_FREEZE_PATH = REPO / "results/learned/w7/w7_pascal_profile_freeze.json"
PROFILE_REPORT_PATH = REPO / "results/learned/w7/w7_pascal_profile.json"

COUNTER_KEYS = (
    "w7_scientific_optimizer_steps",
    "w7_lambda_pilot_runs",
    "w7_candidate_results",
    "g4_adjudications",
    "w8_final_training_runs",
    "learned_test_inference",
    "test_model_facing_access",
    "g8_scientific_changes",
    "f1_reruns",
    "f2_optimizer_steps_during_w7",
    "f3_reruns",
    "pass_one_reruns",
    "pass_two_reruns",
    "pass_three",
    "bler_regeneration",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_upstream() -> None:
    subprocess.run([sys.executable, str(REPO / "tools/verify_w5_training_system.py")], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(REPO / "tools/verify_w6_complete.py")], cwd=REPO, check=True)


def verify_profile_freeze(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "status", "execution_profile_id", "selection_rule",
        "attempts", "selected_attempt_index", "selected_report", "gpu_uuid", "gpu_name",
        "gpu_compute_capability", "physical_batch_size", "accumulation_factor", "effective_batch_size",
        "validation_batch_size", "profile_config_hash", "source_lineage", "projected_runtime",
        "scientific_status", "promotion", "protected_counters", "profile_freeze_id",
    }
    _require(set(value) == required, "W7 profile freeze schema differs")
    body = dict(value)
    freeze_id = body.pop("profile_freeze_id")
    _require(freeze_id == "w7profilefreeze-" + canonical_sha256(body), "W7 profile freeze ID differs")
    _require(value["status"] == "FROZEN" and value["execution_profile_id"] == "confessor_pascal_cu126", "W7 profile freeze status/profile differs")
    attempts = value["attempts"]
    _require(isinstance(attempts, list) and attempts, "W7 profile attempts are missing")
    passed = [index for index, attempt in enumerate(attempts) if attempt["status"] == "PASSED"]
    _require(passed and passed[0] == value["selected_attempt_index"], "W7 profile selection is not first-pass ladder selection")
    _require(value["selected_report"] == {
        "path": attempts[passed[0]]["path"],
        "report_id": attempts[passed[0]]["report_id"],
        "file_sha256": attempts[passed[0]]["sha256"],
    }, "W7 selected profile report binding differs")
    selected_path = Path(value["selected_report"]["path"])
    _require(selected_path.is_file() and _sha(selected_path) == value["selected_report"]["file_sha256"], "W7 selected profile report is missing/drifted")
    report = verify_profile(json.loads(selected_path.read_bytes()))
    _require(report["gpu_binding"]["gpu_uuid"] == value["gpu_uuid"], "W7 frozen GPU UUID differs")
    _require(report["gpu_binding"]["gpu_name"] == value["gpu_name"], "W7 frozen GPU name differs")
    _require(report["batch"]["physical_batch_size"] == value["physical_batch_size"], "W7 frozen physical batch differs")
    _require(report["batch"]["accumulation_factor"] == value["accumulation_factor"], "W7 frozen accumulation differs")
    _require(value["effective_batch_size"] == value["physical_batch_size"] * value["accumulation_factor"] == 32, "W7 frozen effective batch differs")  # literal-ok: owner-frozen W7 batch
    _require(value["validation_batch_size"] == 32, "W7 frozen validation batch differs")  # literal-ok: owner-frozen W7 batch
    runtime = value["projected_runtime"]
    _require(runtime["one_lambda_seconds"] < runtime["cap_seconds"], "W7 one-lambda projection exceeds cap")
    _require(runtime["five_lambda_seconds"] == runtime["one_lambda_seconds"] * 5.0, "W7 five-lambda projection differs")  # literal-ok: five frozen candidates
    _require(all(counter == 0 for counter in value["protected_counters"].values()), "W7 profile freeze protected counter is nonzero")
    _require(value["scientific_status"] == "NON_SCIENTIFIC_ZERO_G4_COVERAGE" and value["promotion"] == {
        "eligible_for_g4": False,
        "eligible_for_w8": False,
        "eligible_for_test": False,
        "profile_checkpoint_may_not_initialize_w7": True,
    }, "W7 profile promotion boundary differs")
    return value


def verify_completion(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "completion_id", "status", "scientific_execution_authorization",
        "g4_status", "lambda_status", "w8_status", "test_status", "protocol", "contract", "schema",
        "source_manifest", "w5", "w6", "profile", "protected_counters", "unauthorized_scientific_outputs",
        "tests", "next_action",
    }
    _require(set(value) == required, "W7-A completion schema differs")
    body = dict(value)
    completion_id = body.pop("completion_id")
    _require(completion_id == "w7acompletion-" + canonical_sha256(body), "W7-A completion ID differs")
    _require(value["status"] == "GREEN_PRE_EXECUTION", "W7-A status differs")
    _require(value["scientific_execution_authorization"] == "ABSENT", "W7-A scientific authorization is present")
    _require(value["g4_status"] == "UNRESOLVED" and value["lambda_status"] == "PROVISIONAL_UNTIL_G4", "W7-A G4/lambda status differs")
    _require(value["w8_status"] == "UNOPENED" and value["test_status"] == "SEALED", "W7-A W8/test boundary differs")
    _require(value["protocol"] == protocol_descriptor(), "W7-A protocol differs")
    _require(all(value == 0 for value in value["protected_counters"].values()), "W7-A protected counter is nonzero")
    _require(set(value["protected_counters"]) == set(COUNTER_KEYS), "W7-A protected counter set differs")
    _require(value["unauthorized_scientific_outputs"] == {
        "w7_scientific_optimizer_steps": 0,
        "w7_lambda_pilot_runs": 0,
        "w7_candidate_results": 0,
        "g4_adjudications": 0,
        "w8_final_training_runs": 0,
        "learned_test_inference": 0,
        "test_model_facing_access": 0,
    }, "W7-A unauthorized-output counters differ")
    return value


def verify(*, run_upstream: bool = True) -> dict[str, Any]:
    if run_upstream:
        _run_upstream()
    contract = json.loads(CONTRACT_PATH.read_bytes())
    verify_contract(contract)
    json.loads(SCHEMA_PATH.read_bytes())
    source = json.loads(SOURCE_MANIFEST_PATH.read_bytes())
    verify_source_manifest(source, current=True)
    profile_report = verify_profile(json.loads(PROFILE_REPORT_PATH.read_bytes()))
    freeze = verify_profile_freeze(json.loads(PROFILE_FREEZE_PATH.read_bytes()))
    completion = verify_completion(json.loads(COMPLETION_PATH.read_bytes()))
    _require(completion["contract"]["id"] == contract["contract_id"], "W7-A completion contract binding differs")
    _require(completion["schema"]["sha256"] == _sha(SCHEMA_PATH), "W7-A schema binding differs")
    _require(completion["source_manifest"]["id"] == source["manifest_id"], "W7-A source manifest binding differs")
    _require(completion["profile"]["freeze_id"] == freeze["profile_freeze_id"], "W7-A profile freeze binding differs")
    _require(completion["profile"]["report_id"] == profile_report["report_id"], "W7-A profile report binding differs")
    w7_root = COMPLETION_PATH.parent
    forbidden = []
    for path in w7_root.rglob("*"):
        if path.is_file() and any(token in path.name for token in ("candidate", "campaign_completion", "g4_adjudication")):
            forbidden.append(str(path.relative_to(REPO)))
    _require(not forbidden, f"W7-A has unauthorized scientific output files: {forbidden}")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upstream", action="store_true")
    args = parser.parse_args()
    value = verify(run_upstream=not args.no_upstream)
    print(f"W7-A GREEN — pre-execution verifier PASS: {value['completion_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
