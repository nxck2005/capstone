#!/usr/bin/env python3
"""Plan, execute, or close the W10 v9 suffix after the separate execution grant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write, read_json, require  # noqa: E402
from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_continuation import (  # noqa: E402
    CONTINUATION_PLAN_PATH,
    build_continuation_closeout,
    build_continuation_plan,
    published_continuation_units,
    verify_continuation_authority,
    verify_continuation_plan,
    verify_launch_authorization,
    verify_suffix_evidence_set,
    verify_suffix_streams,
    verify_worker_prefix,
)
from evaluation.w10_dispatch import dispatch  # noqa: E402
from evaluation.w10_rehearsal import execute, validate_unit, work_units  # noqa: E402
from evaluation.w10_evidence import unit_relative_path  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

PUBLISHED_ROOT = REPO / "results/learned/w10"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute", "closeout"))
    args = parser.parse_args(argv)
    authority = verify_continuation_authority(REPO)
    runtime = REPO / authority["runtime_root"]
    view = ValidationView()
    custody = verify_worker_prefix(REPO, expected_stable_ids=view.stable_ids, allow_suffix=True)
    verify_suffix_evidence_set(runtime, custody, authority)
    if args.action == "plan":
        immutable_write(runtime / CONTINUATION_PLAN_PATH, build_continuation_plan(authority))
        print("W10 v9 continuation plan ready: historical 0–125; new 126–251")
        return 0
    plan = verify_continuation_plan(runtime, authority)
    verify_launch_authorization(REPO, authority, plan)
    if args.action == "execute":
        authenticate_live_w9_pascal(
            REPO,
            authority,
            config_hash=canonical_sha256({"scope": authority["scope"], "units": work_units()}),
        )
        backend = dispatch(REPO, device="cuda:0", authority=authority)
        results = execute(runtime, authority=authority, backend=backend, expected_stable_ids=view.stable_ids)
        print(f"W10 v9 suffix executed/resumed: {len(results)}/252 units")
        return 0
    results = []
    verify_suffix_streams(runtime, expected_stable_ids=view.stable_ids, authority=authority)
    for unit in work_units():
        path = runtime / unit_relative_path(unit, "json")
        require(path.is_file() and not path.is_symlink(), "W10 continuation cannot close incomplete scope")
        value = read_json(path, "W10 continuation unit")
        validate_unit(value, unit)
        results.append(value)
    closeout, units, images = build_continuation_closeout(runtime, results, authority)
    published = published_continuation_units(results, authority)
    for relative, value in (
        ("continuation_unit_manifest_v9.json", units),
        ("continuation_per_image_manifest_v9.json", images),
        ("continuation_closeout_v9.json", closeout),
    ):
        immutable_write(runtime / relative, value)
    for relative, value in (
        ("w10_continuation_unit_manifest_v9.json", units),
        ("w10_continuation_per_image_manifest_v9.json", images),
        ("w10_continuation_units_v9.json", published),
        ("w10_continuation_closeout_v9.json", closeout),
    ):
        immutable_write(PUBLISHED_ROOT / relative, value)
    print(f"W10 v9 continuation closeout: {closeout['closeout_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
