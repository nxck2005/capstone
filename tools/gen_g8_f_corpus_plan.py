#!/usr/bin/env python3
"""Generate or check the metadata-only AM-87 G8_F corpus plan.

This command reads frozen JSON/CSV metadata only.  It cannot authorize F0,
materialize an image, invoke a classifier, train, run pass two, or access a test
payload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_corpus_plan import (  # noqa: E402
    PLAN_PATH,
    build_corpus_plan,
    rendered_json,
    sha256_bytes,
    verify_corpus_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the tracked plan to a fresh metadata-only derivation")
    args = parser.parse_args(argv)

    expected = rendered_json(build_corpus_plan())
    if args.check:
        try:
            actual = PLAN_PATH.read_bytes()
        except OSError as exc:
            raise SystemExit(f"FAIL: cannot read {PLAN_PATH}: {exc}") from None
        if actual != expected:
            raise SystemExit("FAIL: tracked G8_F corpus plan is stale")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_bytes(expected)

    plan = verify_corpus_plan()
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
