#!/usr/bin/env python3
"""Verify immutable AM-87 support evidence after AM-88 superseded multiplicity."""

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
    rendered_json,
    verify_plan_value,
)


def main() -> int:
    try:
        raw = PLAN_PATH.read_bytes()
        value = json.loads(raw)
        if raw != rendered_json(value):
            raise G8FCorpusPlanError("historical AM-87 plan rendering differs")
        plan = verify_plan_value(value, expected=value)
    except (G8FCorpusPlanError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "verdict": "AM-87 SUPPORT PLAN PRESERVED - AM-88 SAMPLER GOVERNS FUTURE EXECUTION; F0 NOT AUTHORIZED",
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
