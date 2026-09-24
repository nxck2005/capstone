#!/usr/bin/env python3
"""Freeze the later owner-issued W10 v9 suffix launch grant after plan authentication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write  # noqa: E402
from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_continuation import (  # noqa: E402
    CONTINUATION_LAUNCH_PATH,
    build_launch_authorization,
    verify_continuation_authority,
    verify_continuation_plan,
    verify_suffix_evidence_set,
    verify_worker_prefix,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="authenticate without writing a launch grant")
    args = parser.parse_args(argv)
    authority = verify_continuation_authority(REPO)
    runtime = REPO / authority["runtime_root"]
    custody = verify_worker_prefix(REPO, expected_stable_ids=ValidationView().stable_ids)
    verify_suffix_evidence_set(runtime, custody, authority)
    plan = verify_continuation_plan(runtime, authority)
    body = build_launch_authorization(authority, plan)
    if args.preflight:
        print(f"W10 v9 launch preflight PASS: {body['launch_id']}; no grant written")
        return 0
    immutable_write(REPO / CONTINUATION_LAUNCH_PATH, body)
    print(f"W10 v9 suffix launch authorization: {body['launch_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
