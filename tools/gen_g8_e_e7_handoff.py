#!/usr/bin/env python3
"""Generate or check the deterministic G8_E E7 terminal handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402

OUT = REPO / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
INCIDENT_AUDIT = REPO / "audit/g8-e-clean-checkout-runtime-incident-2026-08-23.md"
VERIFIER = REPO / "tools/verify_g8_e_complete.py"
SCHEMA_VERSION = 1


def build_handoff(report: dict, incident_raw: bytes) -> dict:
    if report.get("status") != "PASS" or not str(report.get("verdict", "")).startswith("G8_E GREEN"):
        raise ValueError("E7 handoff requires the terminal pre-handoff PASS report")
    counters = report["counters"]
    if counters.get("pass_one_executed_count") != 1 or any(
        counters.get(name) != 0
        for name in (
            "training", "pass_two", "pass_three", "fallback_invoked",
            "ratio_adjudicated", "test_access", "learned_system_training",
            "g8_f_execution",
        )
    ):
        raise ValueError("E7 handoff counters differ from the closed G8_E boundary")
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "g8_e_e7_handoff",
        "phase": "G8_E",
        "checkpoint": "E7",
        "status": "GREEN",
        "verdict": report["verdict"],
        "campaign_id": report["campaign_id"],
        "contract_id": report["contract_id"],
        "verification_report_sha256": v3.sha256_bytes(v3.canonical_json(report)),
        "e2_e4": {
            "e2_completion_sha256": report["e2_completion_sha256"],
            "e3_id": report["e3_id"],
            "e3_sha256": report["e3_sha256"],
            "e4_id": report["e4_id"],
            "e4_sha256": report["e4_sha256"],
        },
        "selection": {
            "authorization_issued_sha256": report["e5_authorization_issued_sha256"],
            "marker_issued_sha256": report["e5_marker_issued_sha256"],
            "policy_sha256": report["selection_policy_sha256"],
            "pass_one_state_id": report["pass_one_state_id"],
            "pass_one_state_content_sha256": report["pass_one_state_content_sha256"],
            "pass_one_state_file_sha256": report["pass_one_state_file_sha256"],
            "selections": report["pass_one_selections"],
            "cells_without_selection": report["pass_one_cells_without_selection"],
        },
        "e6": {
            "freeze_id": report["e6_freeze_id"],
            "freeze_file_sha256": report["e6_freeze_file_sha256"],
            "corpus_spec_id": report["corpus_spec_id"],
            "training_only": True,
            "materialized": False,
        },
        "upstream": {
            "bler_table_sha256": report["bler_table_sha256"],
            "w4_integration_adjudication_sha256": report["w4_integration_adjudication_sha256"],
        },
        "post_closeout_incident_audit": {
            "path": str(INCIDENT_AUDIT.relative_to(REPO)),
            "sha256": v3.sha256_bytes(incident_raw),
            "classification": "TEST_HARNESS_PRODUCTION_ESCAPE_SEPARATE_RUNTIME",
            "scratch_runtime_merge_eligible": False,
            "scratch_runtime_successor_coverage": 0,
        },
        "counters": dict(counters),
        "g8_f": {
            "ready": True,
            "authorized": False,
            "execution_count": 0,
            "next_gate": "G8_F/F0_OWNER_AUTHORIZATION",
        },
    }
    body["handoff_id"] = v3._id("g8ee7handoff-", body)
    return body


def pre_handoff_report() -> dict:
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), "--without-e7-handoff"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        incident_raw = INCIDENT_AUDIT.read_bytes()
        artifact = build_handoff(pre_handoff_report(), incident_raw)
        payload = v3.rendered_json(artifact)
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != payload:
                raise ValueError("E7 handoff is absent or differs from the verified inputs")
            status = "PASS"
        else:
            if args.output.exists():
                raise ValueError("E7 handoff already exists; use --check")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            staging = args.output.parent / f".{args.output.name}.staging"
            staging.write_bytes(payload)
            staging.replace(args.output)
            status = "FROZEN"
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": status,
        "handoff_id": artifact["handoff_id"],
        "handoff_file_sha256": v3.sha256_bytes(payload),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
