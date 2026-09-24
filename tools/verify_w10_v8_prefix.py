#!/usr/bin/env python3
"""Re-authenticate the failed v8 prefix in worker custody without creating science."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_continuation import verify_worker_prefix  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source", action="store_true", help="authenticate v8 against its own commit before v9 exists")
    args = parser.parse_args(argv)
    custody = verify_worker_prefix(
        REPO,
        expected_stable_ids=ValidationView().stable_ids,
        live_source=not args.historical_source,
    )
    print(f"W10 v8 failed-prefix custody PASS: {custody['custody_id']}; units={custody['unit_count']}; streams={custody['stream_count']}; failed={custody['failed_ordinal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
