#!/usr/bin/env python3
"""Manage W10's validation-only one-cell/full-grid/all-systems rehearsal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write, read_json  # noqa: E402
from evaluation.w10_rehearsal import closeout, validate_unit, work_units  # noqa: E402
from verify_w10_rehearsal import verify_authority  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "closeout"))
    args = parser.parse_args(argv)
    authority = verify_authority()
    runtime = REPO / authority["runtime_root"]
    if args.action == "plan":
        value = {
            "schema_version": 1, "artifact_role": "W10_VALIDATION_REHEARSAL_PLAN",
            "authority_id": authority["authority_id"], "work_units": list(work_units()),
            "validation_only": True, "test": "SEALED", "test_access": 0,
        }
        immutable_write(runtime / "plan.json", value)
        print(f"W10 validation rehearsal plan ready: {len(work_units())} exact units")
        return 0
    paths = sorted((runtime / "units").glob("*.json"))
    if len(paths) != len(work_units()):
        raise SystemExit(f"W10 cannot close: {len(paths)}/{len(work_units())} units present")
    results = []
    for path, expected in zip(paths, work_units(), strict=True):
        value = read_json(path, f"W10 unit {expected['ordinal']}")
        validate_unit(value, expected)
        results.append(value)
    value = closeout(results, source_commit=authority["source_commit"], authority_id=authority["authority_id"])
    immutable_write(REPO / "results/learned/w10/w10_rehearsal_closeout.json", value)
    print(f"W10 validation rehearsal complete: {value['closeout_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
