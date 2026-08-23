#!/usr/bin/env python3
"""Structurally check the immutable historical AM-87 G8_F support plan.

AM-88 supersedes only AM-87's Cartesian execution multiplicity. Current sampler
reproduction is handled by ``gen_g8_f_sampler_plan.py``; this command preserves
AM-87's exact plan bytes and support identity without rewriting history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_corpus_plan import (  # noqa: E402
    PLAN_PATH,
    rendered_json,
    sha256_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the tracked plan to a fresh metadata-only derivation")
    args = parser.parse_args(argv)

    if not args.check:
        raise SystemExit("FAIL: AM-87 plan is immutable historical evidence; generation is disabled after AM-88")
    try:
        actual = PLAN_PATH.read_bytes()
        value = __import__("json").loads(actual)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"FAIL: cannot read {PLAN_PATH}: {exc}") from None
    if actual != rendered_json(value):
        raise SystemExit("FAIL: historical AM-87 plan rendering differs")
    from baseline.g8_f_corpus_plan import verify_plan_value
    plan = verify_plan_value(value, expected=value)
    print(
        {
            "status": "PASS",
            "path": str(PLAN_PATH.relative_to(REPO)),
            "plan_id": plan["plan_id"],
            "file_sha256": sha256_bytes(PLAN_PATH.read_bytes()),
            "quality_count": plan["artifact_quality_projection"]["quality_count"],
            "training_stable_id_count": plan["training_membership"]["stable_id_count"],
            "exact_attempt_count": plan["multiplicity_and_feasibility"]["exact_attempt_count"],
            "execution_authorized": plan["protected_boundary"]["f0_execution_authorized"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
