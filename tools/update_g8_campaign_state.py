#!/usr/bin/env python3
"""Initialize or advance only the non-scientific G8_A preflight state."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    initial_campaign_state,
    load_campaign_state,
    validate_state_transition,
    write_campaign_state_atomically,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--initialize", action="store_true")
    action.add_argument("--complete-preflight", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        if CAMPAIGN_STATE.exists():
            raise SystemExit("campaign_state.json already exists; refusing to overwrite")
        state = initial_campaign_state(stage="contract_open")
    else:
        previous = load_campaign_state()
        if previous["identity"]["phase"] != "G8_A" or previous["identity"]["stage"] != "contract_open":
            raise SystemExit("only the G8_A contract_open state can complete preflight")
        state = copy.deepcopy(previous)
        state["identity"]["stage"] = "preflight_complete"
        validate_state_transition(previous, state)
    digest = write_campaign_state_atomically(CAMPAIGN_STATE, state)
    print(
        f"wrote {CAMPAIGN_STATE.relative_to(REPO)} "
        f"phase={state['identity']['phase']} stage={state['identity']['stage']} "
        f"sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
