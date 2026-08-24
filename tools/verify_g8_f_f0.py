#!/usr/bin/env python3
"""Verify the frozen G8_F/F0 opening without starting F1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline.g8_f_f0 import AUTHORIZATION_PATH, G8FF0Error, verify_f0_authorization  # noqa: E402
from baseline.g8_f_materializer import load_frozen_assignments  # noqa: E402
from verify_g8_f_sampler_plan import verify_sampler_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-runtime", action="store_true")
    parser.add_argument("--path", type=Path, default=AUTHORIZATION_PATH)
    args = parser.parse_args(argv)
    try:
        value = verify_f0_authorization(args.path, live_runtime=args.live_runtime)
        sampler = verify_sampler_plan()
        assignments = load_frozen_assignments()
        subprocess.run(
            [sys.executable, str(REPO / "tools/verify_g8_e_complete.py")],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        if sampler["assignment_evidence"]["nominal_attempt_count"] != len(assignments):
            raise G8FF0Error("independent AM-88 assignment count differs at F0")
    except (G8FF0Error, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "PASS",
        "verdict": "F0 GREEN - G8_F EXECUTION CONTRACT/AUTHORIZATION FROZEN; F1 NOT STARTED; SEPARATE OWNER/OPERATOR LAUNCH REQUIRED",
        "authorization_id": value["authorization_id"],
        "authorization_file_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
        "intended_f1_source_commit": value["source"]["intended_f1_source_commit"],
        "quality_count": value["protocol"]["quality_count"],
        "training_stable_id_count": value["protocol"]["training_stable_id_count"],
        "variants_per_training_image": value["protocol"]["sampler"]["variants_per_training_image"],
        "nominal_attempt_count": len(assignments),
        "materialized_artifact_objects": value["protected_starting_state"]["materialized_artifact_objects"],
        "optimizer_steps": value["protected_starting_state"]["artifact_classifier_optimizer_steps"],
        "pass_two": value["protected_starting_state"]["pass_two"],
        "test_access": value["protected_starting_state"]["test_access"],
        "f1_started": value["protected_starting_state"]["f1_started"],
        "live_runtime_authenticated": args.live_runtime,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
