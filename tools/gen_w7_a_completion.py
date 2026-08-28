#!/usr/bin/env python3
"""Build the additive W7-A pre-execution completion record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from training.deterministic_core import canonical_bytes  # noqa: E402
from training.w7_protocol import protocol_descriptor  # noqa: E402

W7_ROOT = REPO / "results/learned/w7"
OUTPUT = W7_ROOT / "w7_a_completion.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"W7-A completion already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build(*, ci: dict[str, Any] | None = None) -> dict[str, Any]:
    contract_path = W7_ROOT / "w7_a_contract.json"
    schema_path = REPO / "spec/schemas/w7_g4_artifacts.schema.json"
    source_path = W7_ROOT / "w7_source_manifest.json"
    report_path = W7_ROOT / "w7_pascal_profile.json"
    freeze_path = W7_ROOT / "w7_pascal_profile_freeze.json"
    confirmation_path = W7_ROOT / "w7_final_source_profile_confirmation.json"
    w5_path = REPO / "results/learned/w5/w5_gradscaler_accounting_repair_completion.json"
    w6_path = REPO / "results/baseline/w6/w6_completion.json"
    contract = json.loads(contract_path.read_bytes())
    schema = json.loads(schema_path.read_bytes())
    source = json.loads(source_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    freeze = json.loads(freeze_path.read_bytes())
    confirmation = json.loads(confirmation_path.read_bytes())
    w5 = json.loads(w5_path.read_bytes())
    w6 = json.loads(w6_path.read_bytes())
    tests = ci or {"status": "PENDING_EXACT_SHA_CI", "run_id": None, "job_id": None, "head_sha": None}
    return {
        "schema_version": 1,
        "artifact_role": "W7_A_PRE_EXECUTION_COMPLETION",
        "status": "GREEN_PRE_EXECUTION",
        "scientific_execution_authorization": "ABSENT",
        "g4_status": "UNRESOLVED",
        "lambda_status": "PROVISIONAL_UNTIL_G4",
        "w8_status": "UNOPENED",
        "test_status": "SEALED",
        "protocol": protocol_descriptor(),
        "contract": {
            "path": str(contract_path.relative_to(REPO)),
            "id": contract["contract_id"],
            "sha256": _sha(contract_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(REPO)),
            "id": schema["$id"],
            "sha256": _sha(schema_path),
        },
        "source_manifest": {
            "path": str(source_path.relative_to(REPO)),
            "id": source["manifest_id"],
            "sha256": _sha(source_path),
            "execution_source_commit": source["source_commit"],
            "entry_count": len(source["entries"]),
        },
        "w5": {
            "status": "GREEN",
            "completion_id": w5["repair_id"],
            "path": str(w5_path.relative_to(REPO)),
            "sha256": _sha(w5_path),
            "historical_semantics_unchanged": True,
            "w7_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
        },
        "w6": {
            "status": "GREEN",
            "completion_id": w6["completion_id"],
            "path": str(w6_path.relative_to(REPO)),
            "sha256": _sha(w6_path),
            "test_status": "SEALED",
        },
        "profile": {
            "report_path": str(report_path.relative_to(REPO)),
            "report_id": report["report_id"],
            "report_sha256": _sha(report_path),
            "freeze_path": str(freeze_path.relative_to(REPO)),
            "freeze_id": freeze["profile_freeze_id"],
            "freeze_sha256": _sha(freeze_path),
            "execution_profile_id": freeze["execution_profile_id"],
            "gpu_uuid": freeze["gpu_uuid"],
            "gpu_name": freeze["gpu_name"],
            "physical_batch_size": freeze["physical_batch_size"],
            "accumulation_factor": freeze["accumulation_factor"],
            "effective_batch_size": freeze["effective_batch_size"],
            "validation_batch_size": freeze["validation_batch_size"],
            "one_lambda_projected_seconds": freeze["projected_runtime"]["one_lambda_seconds"],
            "five_lambda_projected_seconds": freeze["projected_runtime"]["five_lambda_seconds"],
            "per_run_cap_hours": float(get("compute.max_wall_clock_hours_per_run")),
            "scientific_coverage": 0,
            "final_source_confirmation": {
                "path": str(confirmation_path.relative_to(REPO)),
                "confirmation_id": confirmation["confirmation_id"],
                "sha256": _sha(confirmation_path),
                "checkout_commit": confirmation["source_lineage"]["checkout_commit"],
                "execution_source_commit": confirmation["source_lineage"]["execution_source_commit"],
            },
        },
        "protected_counters": {
            "w7_scientific_optimizer_steps": 0,
            "w7_lambda_pilot_runs": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
            "g8_scientific_changes": 0,
            "f1_reruns": 0,
            "f2_optimizer_steps_during_w7": 0,
            "f3_reruns": 0,
            "pass_one_reruns": 0,
            "pass_two_reruns": 0,
            "pass_three": 0,
            "bler_regeneration": 0,
        },
        "unauthorized_scientific_outputs": {
            "w7_scientific_optimizer_steps": 0,
            "w7_lambda_pilot_runs": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
        },
        "tests": tests,
        "next_action": "SEPARATE_W7_B_SCIENTIFIC_CAMPAIGN_AUTHORIZATION_ON_CONFESSOR",
    }


def write(path: Path = OUTPUT, *, ci: dict[str, Any] | None = None) -> dict[str, Any]:
    value = build(ci=ci)
    value["completion_id"] = "w7acompletion-" + hashlib.sha256(canonical_bytes(value)).hexdigest()
    _publish_immutable(path, canonical_bytes(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ci-run-id", type=int)
    parser.add_argument("--ci-job-id", type=int)
    parser.add_argument("--ci-head-sha")
    args = parser.parse_args()
    ci = None
    if args.ci_run_id is not None or args.ci_job_id is not None or args.ci_head_sha is not None:
        ci = {"status": "success", "run_id": args.ci_run_id, "job_id": args.ci_job_id, "head_sha": args.ci_head_sha}
    value = write(args.output, ci=ci)
    print(f"wrote {args.output.relative_to(REPO)}: {value['completion_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
