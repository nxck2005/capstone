#!/usr/bin/env python3
"""Verify the corrected-v3 worker-successor frozen contract or a named phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3s as v3s  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("frozen", "predata", "active", "e2", "e3", "e4"), default="predata")
    parser.add_argument("--skip-live-sources", action="store_true")
    parser.add_argument("--skip-live-data", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "verify_live_sources": not args.skip_live_sources,
        "verify_live_data": not args.skip_live_data,
    }
    try:
        if args.phase == "frozen":
            result = v3s.verify_frozen_contract(**kwargs)
        elif args.phase == "predata":
            result = v3s.verify_predata_zero_state(**kwargs)
        elif args.phase == "active":
            result = v3s.verify_active_e2(**kwargs)
        elif args.phase == "e2":
            result = v3s.verify_e2_complete(**kwargs)
        elif args.phase == "e3":
            result = v3s.verify_e3_complete(**kwargs)
        else:
            result = v3s.verify_e4_complete(**kwargs)
        contract = result["contract"]
        print({
            "status": "PASS",
            "phase": result.get("phase", "FROZEN_CONTRACT"),
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "profile_id": contract["execution_profile"]["profile_id"],
            "device": contract["execution_profile"]["device"],
            "work_units": contract["transaction"]["production_total_required"],
            "coverage": contract["safety"]["measurement_coverage"],
        })
        return 0
    except (OSError, v3s.G8EV3SError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
