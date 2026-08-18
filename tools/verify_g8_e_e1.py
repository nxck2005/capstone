#!/usr/bin/env python3
"""Verify the G8_E E1 pre-data contract and its live upstream bindings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_e import (  # noqa: E402
    E1_CONTRACT_PATH,
    G8EContractError,
    verify_e1_contract_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=E1_CONTRACT_PATH)
    parser.add_argument(
        "--skip-live-profile",
        action="store_true",
        help="skip the CUDA/profile probe; source and artifact bindings remain strict",
    )
    args = parser.parse_args()
    try:
        value = verify_e1_contract_file(
            args.path,
            verify_live_assets=True,
            verify_live_profile=not args.skip_live_profile,
        )
    except G8EContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        {
            "status": "PASS",
            "contract_id": value["contract_id"],
            "campaign_id": value["campaign_id"],
            "candidate_count": value["candidate_authority"]["candidate_count"],
            "initial_candidate_count": value["candidate_authority"]["initial_dataset_candidate_count"],
            "measurement_coverage": value["safety"]["measurement_coverage"],
            "test_access": value["safety"]["test_access"],
            "authorization_issued": value["pass_one_preconditions"]["authorization_issued"],
            "corpus_materialized": value["corpus_spec_binding"]["materialized"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
