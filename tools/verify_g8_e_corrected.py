#!/usr/bin/env python3
"""Independently verify the corrected G8_E E1 pre-data bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected as corrected  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-live-sources", action="store_true")
    args = parser.parse_args()
    try:
        result = corrected.verify_corrected_bundle(verify_live_sources=not args.skip_live_sources)
        if corrected.CORRECTED_RUNTIME_ROOT.exists():
            raise corrected.CorrectedG8EError("corrected E2 runtime root is already present before data")
        auth = corrected.CORRECTED_ROOT / "e2_execution_authorization.json"
        if auth.exists():
            raise corrected.CorrectedG8EError("owner E2 authorization exists in the pre-data freeze")
        contract = result["contract"]
        print({
            "status": "PASS",
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "logical_initial_snr_cells": contract["compute_plan"]["logical_snr_cells_initial"],
            "logical_all_roles_snr_cells": contract["compute_plan"]["logical_snr_cells_all_roles"],
            "structural_initial": contract["compute_plan"]["structural_measurement_identities_initial"],
            "physical_cache_keys_per_image": contract["compute_plan"]["physical_cache_keys_per_image"],
            "measurement_coverage": contract["safety"]["measurement_coverage"],
            "e2_authorized": contract["authorization"]["issued"],
        })
    except (OSError, corrected.CorrectedG8EError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
