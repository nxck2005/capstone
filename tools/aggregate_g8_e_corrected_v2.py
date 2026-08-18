#!/usr/bin/env python3
"""Run count-derived v2 E4 aggregation after independent E3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=v2.V2_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print("REFUSED: E4 execution is closed without --execute and owner-authorized post-E2 state", file=sys.stderr)
        return 2
    try:
        bundle = v2.verify_bundle()
        contract = bundle["contract"]
        authority = v2.load_measurement_authority()
        sample_ids = v2._validation_ids()
        result = v2.aggregate_e4_counts_v2(authority=authority, sample_ids=sample_ids, runtime_root=args.runtime_root, contract=contract, production=True)
        output = args.output or args.runtime_root / "e4_count_derived.json"
        output.write_bytes(v2.rendered_json(result))
        print({"status": result["status"], "object_count": result["object_count"]})
        return 0
    except (OSError, v2.G8EV2Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
