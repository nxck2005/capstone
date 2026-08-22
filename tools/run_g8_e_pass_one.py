#!/usr/bin/env python3
"""Owner-gated G8_E E5/E6 pass-one entry point.

Subcommands:
  verify-inputs   authenticate every frozen pass-one input, execute nothing
  execute         run selection pass one exactly once (requires the frozen
                  authorization and the pre-execution marker; refuses if the
                  immutable completion record already exists)
  verify-state    independently authenticate an existing completion record
                  against a full recomputation from the frozen inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_pass_one as pass_one  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify-inputs", "execute", "verify-state"),
    )
    parser.add_argument("--authorization", default=str(pass_one.E5_AUTHORIZATION_PATH))
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-inputs":
            context = pass_one.authenticate_inputs(Path(args.authorization))
            print(json.dumps({
                "status": "PASS",
                "phase": "E5_INPUTS_AUTHENTICATED",
                "campaign_id": context["contract"]["campaign_id"],
                "e4_id": context["e4"]["e4_id"],
                "objects": len(context["e4"]["objects"]),
                "calls": context["plan"]["call_count"],
                "selection_policy_sha256": context["authorization"][
                    "selection_policy_sha256"
                ],
                "state_path": context["authorization"]["state_path"],
            }, sort_keys=True))
            return 0
        if args.command == "execute":
            body = pass_one.run_pass_one(Path(args.authorization))
            print(json.dumps({
                "status": "PASS",
                "phase": "PASS_ONE_COMPLETE",
                "state_id": body["state_id"],
                "state_sha256": body["state_sha256"],
                "selections": body["totals"]["snr_cells_with_selection"],
                "cells_without_selection": body["totals"]["snr_cells_without_selection"],
                "calls": body["call_count"],
            }, sort_keys=True))
            return 0
        result = pass_one.verify_pass_one_state(authorization_path=Path(args.authorization))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
