#!/usr/bin/env python3
"""Verify corrected-v3 frozen contract or a named lifecycle phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402


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
            result = v3.verify_v3_frozen_contract(**kwargs)
        elif args.phase == "predata":
            result = v3.verify_v3_predata_zero_state(**kwargs)
        elif args.phase == "active":
            result = v3.verify_v3_active_e2(**kwargs)
        elif args.phase == "e2":
            result = v3.verify_v3_e2_complete(**kwargs)
        elif args.phase == "e3":
            result = v3.verify_v3_e3_complete(**kwargs)
        else:
            result = v3.verify_v3_e4_complete(**kwargs)
        contract = result["contract"]
        print({
            "status": "PASS",
            "phase": result.get("phase", "FROZEN_CONTRACT"),
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "logical_initial": contract["authority"]["counts"]["logical_initial_snr_cells"],
            "structural_initial": contract["authority"]["counts"]["structural_initial"],
            "work_units": contract["compute_plan"]["physical"]["work_units"],
            "coverage": contract["safety"]["measurement_coverage"],
        })
        return 0
    except (OSError, v3.G8EV3Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
