#!/usr/bin/env python3
"""Generate/check the metadata-only AM-88 balanced G8_F sampler plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_sampler_plan import (  # noqa: E402
    PLAN_PATH,
    build_sampler_plan,
    rendered_json,
    sha256_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare tracked bytes with a fresh metadata-only derivation")
    args = parser.parse_args(argv)
    expected = rendered_json(build_sampler_plan())
    if args.check:
        try:
            actual = PLAN_PATH.read_bytes()
        except OSError as exc:
            raise SystemExit(f"FAIL: cannot read {PLAN_PATH}: {exc}") from None
        if actual != expected:
            raise SystemExit("FAIL: tracked AM-88 sampler plan is stale")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_bytes(expected)
    value = build_sampler_plan()
    print({
        "status": "PASS",
        "path": str(PLAN_PATH.relative_to(REPO)),
        "plan_id": value["plan_id"],
        "file_sha256": sha256_bytes(PLAN_PATH.read_bytes()),
        "support_quality_count": value["support"]["quality_count"],
        "training_stable_id_count": value["training_membership"]["stable_id_count"],
        "variants_per_image": value["sampler"]["variants_per_training_image"],
        "nominal_attempt_count": value["assignment_evidence"]["nominal_attempt_count"],
        "f0_authorized": value["protected_boundary"]["f0_execution_authorized"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
