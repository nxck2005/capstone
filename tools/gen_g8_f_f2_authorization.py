#!/usr/bin/env python3
"""Generate or verify the canonical owner-authorized F2 execution artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.g8_f_f2_authorization import (
    AUTHORIZATION_PATH,
    build_authorization,
    rendered_json,
    verify_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--issued-at")
    parser.add_argument("--output", type=Path, default=AUTHORIZATION_PATH)
    args = parser.parse_args()
    if args.check:
        value = verify_authorization(args.output)
        print(f"PASS {value['authorization_id']}")
        return 0
    if not args.source_commit or not args.preflight or not args.issued_at:
        parser.error("generation requires --source-commit, --preflight, and --issued-at")
    preflight = json.loads(args.preflight.read_bytes())
    authorization = build_authorization(
        source_commit=args.source_commit,
        issued_at=args.issued_at,
        preflight=preflight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered_json(authorization))
    print(authorization["authorization_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
