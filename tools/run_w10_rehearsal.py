#!/usr/bin/env python3
"""Manage W10's validation-only rehearsal: plan, execute, closeout (AM-98)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write, read_json  # noqa: E402
from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_dispatch import dispatch  # noqa: E402
from evaluation.w10_rehearsal import closeout, validate_unit, work_units  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_w10_rehearsal import verify_authority  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute", "closeout"))
    args = parser.parse_args(argv)
    authority = verify_authority()
    runtime = REPO / authority["runtime_root"]
    if args.action == "plan":
        value = {
            "schema_version": 2,
            "artifact_role": "W10_VALIDATION_REHEARSAL_PLAN",
            "authority_id": authority["authority_id"],
            "scope_sha256": authority["scope_sha256"],
            "work_units": list(work_units()),
            "validation_only": True,
            "test": "SEALED",
            "test_access": 0,
        }
        immutable_write(runtime / "plan.json", value)
        print(f"W10 validation rehearsal plan ready: {len(work_units())} exact units")
        return 0
    if args.action == "execute":
        authenticate_live_w9_pascal(
            REPO,
            authority,
            config_hash=canonical_sha256({"scope": authority["scope"], "units": work_units()}),
        )
        view = ValidationView()
        backend = dispatch(REPO, device="cuda:0", authority=authority)
        from evaluation.w10_rehearsal import execute  # noqa: PLC0415

        results = execute(runtime, authority=authority, backend=backend, expected_stable_ids=view.stable_ids)
        print(f"W10 validation rehearsal executed/resumed: {len(results)} units")
        return 0
    paths = sorted((runtime / "units").glob("*.json"))
    if len(paths) != len(work_units()):
        raise SystemExit(f"W10 cannot close: {len(paths)}/{len(work_units())} units present")
    results = []
    for path, expected in zip(paths, work_units(), strict=True):
        value = read_json(path, f"W10 unit {expected['ordinal']}")
        validate_unit(value, expected)
        results.append(value)
    value = closeout(runtime, results, authority=authority)
    immutable_write(REPO / "results/learned/w10/w10_rehearsal_closeout.json", value)
    print(f"W10 validation rehearsal complete: {value['closeout_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
