#!/usr/bin/env python3
"""Independently reproduce and verify the frozen AM-87 G8_F corpus plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_corpus_plan import (  # noqa: E402
    PLAN_PATH,
    G8FCorpusPlanError,
    sha256_bytes,
    verify_corpus_plan,
)


def main() -> int:
    try:
        plan = verify_corpus_plan()
    except G8FCorpusPlanError as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "verdict": "G8_F PROTOCOL PLAN FROZEN - OWNER AUDIT REQUIRED; EXECUTION NOT AUTHORIZED",
                "plan_id": plan["plan_id"],
                "plan_file_sha256": sha256_bytes(PLAN_PATH.read_bytes()),
                "quality_count": plan["artifact_quality_projection"]["quality_count"],
                "training_stable_id_count": plan["training_membership"]["stable_id_count"],
                "exact_attempt_count": plan["multiplicity_and_feasibility"]["exact_attempt_count"],
                "materialized_object_count": plan["protected_boundary"]["materialized_object_count"],
                "optimizer_steps": plan["protected_boundary"]["optimizer_steps"],
                "pass_two": plan["protected_boundary"]["pass_two"],
                "test_access": plan["protected_boundary"]["test_access"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
