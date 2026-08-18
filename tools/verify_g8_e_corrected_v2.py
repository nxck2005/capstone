#!/usr/bin/env python3
"""Verify the frozen G8_E corrected-v2 pre-data bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-live-sources", action="store_true")
    args = parser.parse_args()
    try:
        bundle = v2.verify_bundle(verify_live_sources=not args.skip_live_sources)
        contract = bundle["contract"]
        if contract["safety"]["measurement_coverage"] != 0 or contract["authorization"]["issued"] is not False:
            raise v2.G8EV2Error("v2 bundle is not zero-data and unauthorized")
        if contract["compute_plan"]["physical"]["work_units"] != 288000:
            raise v2.G8EV2Error("v2 work-unit authority is not 288,000")
        if contract["selection_authorization"]["call_count"] != 18 or contract["selection_authorization"]["max_candidates"] != 1008 or contract["selection_authorization"]["max_samples"] != 1000:
            raise v2.G8EV2Error("v2 pass-one bounds are not derived from the frozen call plan")
        if v2.V2_RUNTIME_ROOT.exists() or (v2.V2_ROOT / "e2_execution_authorization.json").exists():
            raise v2.G8EV2Error("v2 pre-data root contains runtime or owner authorization")
        print({
            "status": "PASS",
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "logical_initial": contract["authority"]["counts"]["logical_initial_snr_cells"],
            "structural_initial": contract["authority"]["counts"]["structural_initial"],
            "work_units": contract["compute_plan"]["physical"]["work_units"],
            "coverage": contract["safety"]["measurement_coverage"],
        })
        return 0
    except (OSError, v2.G8EV2Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
