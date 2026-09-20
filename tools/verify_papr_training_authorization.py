#!/usr/bin/env python3
"""Verify the frozen one-run PAPR-constrained training authority (AM-98).

Authority-only mode authenticates the immutable authority bytes and their
frozen source epoch without a runtime.  ``--terminal`` additionally requires the
worker-local PAPR runtime and proves the complete lifecycle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.papr_constrained import PAPR_AUTHORITY_PATH, PAPR_SELECTED_CHECKPOINT_PATH  # noqa: E402
from training.papr_lifecycle import verify_papr_authority, verify_papr_terminal  # noqa: E402

AUTHORITY = REPO / PAPR_AUTHORITY_PATH
SELECTED = REPO / PAPR_SELECTED_CHECKPOINT_PATH


def verify_authority(path: Path = AUTHORITY) -> dict:
    return verify_papr_authority(REPO, path)


def verify_selected() -> dict:
    return verify_papr_terminal(REPO, authority_path=AUTHORITY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    authority = verify_authority()
    if args.terminal:
        value = verify_selected()
        print(f"PAPR constrained training verifier PASS: terminal {value['completion_id']}")
        return 0
    print(f"PAPR constrained training verifier PASS: authority {authority['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
