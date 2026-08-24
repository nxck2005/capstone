#!/usr/bin/env python3
"""Generate/check the owner-authorized G8_F/F0-v2 execution handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_f0 import (  # noqa: E402
    AUTHORIZATION_PATH,
    build_f0_authorization,
    rendered_json,
    verify_f0_authorization,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorization-date", default="2026-08-24")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        value = verify_f0_authorization(AUTHORIZATION_PATH)
    else:
        value = build_f0_authorization(
            source_commit=args.source_commit,
            authorization_date=args.authorization_date,
        )
        AUTHORIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUTHORIZATION_PATH.write_bytes(rendered_json(value))
        value = verify_f0_authorization(AUTHORIZATION_PATH)
    print(json.dumps({
        "status": "PASS",
        "path": str(AUTHORIZATION_PATH.relative_to(REPO)),
        "authorization_id": value["authorization_id"],
        "file_sha256": hashlib.sha256(AUTHORIZATION_PATH.read_bytes()).hexdigest(),
        "source_commit": value["source"]["intended_f1_source_commit"],
        "f1_started": value["protected_starting_state"]["f1_started"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
