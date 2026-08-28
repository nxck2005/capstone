#!/usr/bin/env python3
"""Generate/verify the result-independent W7-A contract artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_protocol import protocol_descriptor  # noqa: E402

OUTPUT = REPO / "results/learned/w7/w7_a_contract.json"
TEXT = REPO / "instructions/W7-A.txt"
SCHEMA = REPO / "spec/schemas/w7_g4_artifacts.schema.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_role": "W7_A_PRE_RESULT_CONTRACT",
        "contract_version": "w7-g4-pre-execution-v1",
        "scientific_execution_authorization": "ABSENT",
        "created_before_scientific_results": True,
        "contract_text": {"path": "instructions/W7-A.txt", "sha256": _sha(TEXT)},
        "schema": {"path": "spec/schemas/w7_g4_artifacts.schema.json", "sha256": _sha(SCHEMA)},
        "protocol": protocol_descriptor(),
        "profile": {
            "execution_profile_id": "confessor_pascal_cu126",
            "requirements_lock": "requirements-pascal.lock",
            "requirements_lock_sha256": str(get("environment.execution_profiles.confessor_pascal_cu126.lock_file_sha256")),
            "gpu_ladder": [
                {"uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b", "name": "NVIDIA GeForce GTX 1080 Ti", "compute_capability": "6.1"},
                {"uuid": "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a", "name": "NVIDIA TITAN Xp", "compute_capability": "6.1"},
            ],
            "uuid_is_authoritative": True,
            "fallback_only_after_hard_failure": True,
        },
        "eligibility_roles": {
            "profile": "NON_SCIENTIFIC_PROFILE",
            "pilot_checkpoint": "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT",
            "campaign": "CAMPAIGN_COMPLETE_NOT_ADJUDICATED",
            "adjudication": "G4_ADJUDICATED",
            "w8_ineligible": "W8_INELIGIBLE",
        },
        "w5_w6_reauthentication": {
            "w5_completion": "results/learned/w5/w5_gradscaler_accounting_repair_completion.json",
            "w5_status": "GREEN",
            "w6_completion": "results/baseline/w6/w6_completion.json",
            "w6_status": "GREEN",
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
        "next_gate": "SEPARATE_W7_B_SCIENTIFIC_EXECUTION_AUTHORIZATION_REQUIRED",
    }


def write(path: Path = OUTPUT) -> dict[str, Any]:
    value = build()
    value["contract_id"] = "w7contract-" + canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return value


def verify(value: dict[str, Any]) -> None:
    required = {"schema_version", "artifact_role", "contract_version", "scientific_execution_authorization", "created_before_scientific_results", "contract_text", "schema", "protocol", "profile", "eligibility_roles", "w5_w6_reauthentication", "protected_counters", "next_gate", "contract_id"}
    if set(value) != required:
        raise ValueError("W7-A contract schema differs")
    body = dict(value)
    contract_id = body.pop("contract_id")
    expected = build()
    if contract_id != "w7contract-" + canonical_sha256(expected) or body != expected:
        raise ValueError("W7-A contract bytes/content differ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.check:
        verify(json.loads(args.output.read_bytes()))
        print(f"W7-A contract PASS: {json.loads(args.output.read_bytes())['contract_id']}")
    else:
        value = write(args.output)
        print(f"wrote {args.output.relative_to(REPO)}: {value['contract_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
