#!/usr/bin/env python3
"""Freeze a distinct W10 v9 suffix authority after the independently audited v9 source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write  # noqa: E402
from evaluation.w10_continuation import CONTINUATION_AUTHORITY_PATH, build_continuation_authority  # noqa: E402
from evaluation.w10_rehearsal import work_units  # noqa: E402
from runtime.source_epochs import load_w10_manifest  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="validate without writing an authority")
    args = parser.parse_args(argv)
    source = load_w10_manifest(REPO, live=True, epoch="v9")
    body = build_continuation_authority(REPO, source)
    if args.preflight:
        print(f"W10 v9 continuation preflight: {len(body['new_authorized_ordinals'])} suffix units; historical custody {body['historical_custody_id']}")
        return 0
    live = authenticate_live_w9_pascal(
        REPO,
        body,
        config_hash=canonical_sha256({"scope": body["scope"], "units": work_units()}),
    )
    if body["cuda_mapping"] != live["environment"]["cuda_mapping"]:
        raise SystemExit("W10 continuation exact CUDA mapping differs from the original authority")
    body["authority_id"] = "w10continuationauth-" + canonical_sha256(body)
    immutable_write(REPO / CONTINUATION_AUTHORITY_PATH, body)
    print(f"W10 v9 continuation authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
