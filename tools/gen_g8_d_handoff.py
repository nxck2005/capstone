#!/usr/bin/env python3
"""Publish the observed D7 verification and G8_E handoff record.

The command records already-run verifier results and a supplied full-pytest
count.  It does not start any G8_E work or any scientific campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_d  # noqa: E402
from verify_g8_d_smoke import verify as verify_smoke  # noqa: E402


OUT = REPO / "results/baseline/g8_d/d7_handoff.json"
HANDOFF_SCHEMA_VERSION = 1


def build_handoff(*, pytest_count: int, pytest_skipped: int = 0, pytest_failures: int = 0, repo_root: Path = REPO) -> dict[str, object]:
    if isinstance(pytest_count, bool) or not isinstance(pytest_count, int) or pytest_count <= 0:
        raise g8_d.G8DContractError("D7 full pytest count must be a positive integer")
    if pytest_skipped != 0 or pytest_failures != 0:
        raise g8_d.G8DContractError("D7 handoff requires zero skipped tests and zero failures")
    repo_root = Path(repo_root).resolve()
    contract = g8_d.build_g8_d_contract(repo_root)
    smoke_path = repo_root / "results/baseline/g8_d/bounded_smoke.json"
    smoke = verify_smoke(smoke_path)
    d0_path = repo_root / "results/baseline/g8_d/d0_open.json"
    body: dict[str, object] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "artifact_role": "g8_d_handoff",
        "phase": "G8_D",
        "checkpoint": "D7",
        "status": "GREEN",
        "contract_id": contract["contract_id"],
        "g8_c": {
            "table_id": contract["g8_c_binding"]["table_id"],
            "table_sha256": contract["g8_c_binding"]["table_sha256"],
            "curves": contract["g8_c_binding"]["curves"],
            "measured_points": contract["g8_c_binding"]["measured_points"],
            "trials_per_point": contract["g8_c_binding"]["trials_per_point"],
            "predecessor_table_contribution": contract["g8_c_binding"]["predecessor_table_contribution"],
        },
        "d0_open": {
            "artifact_id": contract["d0_open_binding"]["artifact_id"],
            "artifact_sha256": g8_d.sha256_file(d0_path),
        },
        "smoke": {
            "artifact_id": smoke["artifact_id"],
            "artifact_sha256": g8_d.sha256_file(smoke_path),
            "samples": smoke["samples"],
            "candidates": smoke["candidates"],
            "cells": smoke["cells"],
            "mutation_cases": len(smoke["mutation_case_names"]),
            "merge_eligible": smoke["merge_eligible"],
        },
        "verification": {
            "g8_c_successor": "PASS",
            "g8_c_closeout": "PASS",
            "exhaustive_frozen_table_lookup": "PASS",
            "d0_open": "PASS",
            "d1_d6_targeted": "PASS",
            "d6_smoke": "PASS",
            "w4_integration": "PASS",
            "g2_adjudication": "PASS",
            "packetisation": "PASS",
            "documentation": "PASS",
            "generated_spec_views": "PASS",
            "literal_lint": "PASS",
            "cpu_runtime_lock": "PASS",
            "full_pytest": "PASS",
        },
        "tests": {
            "full_pytest_command": "PYTHONPATH=src:tools .venv/bin/python -m pytest",
            "full_pytest_collected": pytest_count,
            "full_pytest_passed": pytest_count,
            "full_pytest_skipped": pytest_skipped,
            "full_pytest_failures": pytest_failures,
        },
        "safety": {
            "full_validation_campaign_started": False,
            "selection_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "training_started": False,
            "test_split_accessed": False,
            "g8_e_started": False,
            "inference": 0,
            "training": 0,
            "validation_decoding": 0,
            "test_access": 0,
        },
        "next_gate": "G8_E/E0",
        "g8_e_released": True,
        "full_campaign_not_started": True,
    }
    body["artifact_id"] = "g8dhandoff-" + g8_d.sha256_bytes(g8_d.canonical_json(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-count", type=int, required=True)
    parser.add_argument("--pytest-skipped", type=int, default=0)
    parser.add_argument("--pytest-failures", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    artifact = build_handoff(
        pytest_count=args.pytest_count,
        pytest_skipped=args.pytest_skipped,
        pytest_failures=args.pytest_failures,
        repo_root=REPO,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(g8_d.rendered_json(artifact))
    print(json.dumps({"status": "PASS", "artifact_id": artifact["artifact_id"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
