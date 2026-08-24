#!/usr/bin/env python3
"""Generate/check the separate owner authorization for the exact Pascal G8_F/F1 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline.g8_f_f0 import AUTHORIZATION_PATH, rendered_json, verify_f0_authorization  # noqa: E402
from baseline.g8_f_materializer import canonical_json  # noqa: E402
from run_g8_f_f1 import verify_separate_f1_launch  # noqa: E402

LAUNCH_PATH = REPO / "results/baseline/g8_f/f1_launch_authorization.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_launch_authorization(f0_path: Path, *, issued_at: str) -> dict[str, object]:
    if not issued_at:
        raise ValueError("issued_at must be explicit")
    f0 = verify_f0_authorization(f0_path, require_zero_prefix=True)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": "g8_f_f1_owner_launch_authorization",
        "status": "OWNER_AUTHORIZED_F1_LAUNCH",
        "scope": "G8_F_F1_ONLY",
        "issued_at": issued_at,
        "f0_authorization_id": f0["authorization_id"],
        "f0_file_sha256": _sha(f0_path.read_bytes()),
        "intended_f1_source_commit": f0["source"]["intended_f1_source_commit"],
        "owner_statement": "DELIBERATE_SEPARATE_OWNER_ACTION_AFTER_F0_FREEZE",
    }
    body["launch_id"] = "g8ff1launch-" + _sha(canonical_json(body))
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--path", type=Path, default=LAUNCH_PATH)
    parser.add_argument("--issued-at")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        if args.issued_at is not None:
            parser.error("--check does not accept --issued-at")
        f0 = verify_f0_authorization(args.f0_authorization, require_zero_prefix=False)
        value = verify_separate_f1_launch(args.path, args.f0_authorization, f0)
    else:
        if args.issued_at is None:
            parser.error("generation requires explicit --issued-at")
        value = build_launch_authorization(args.f0_authorization, issued_at=args.issued_at)
        args.path.parent.mkdir(parents=True, exist_ok=True)
        args.path.write_bytes(rendered_json(value))
        f0 = verify_f0_authorization(args.f0_authorization, require_zero_prefix=False)
        value = verify_separate_f1_launch(args.path, args.f0_authorization, f0)
    print(json.dumps({
        "status": "PASS",
        "path": str(args.path.relative_to(REPO)),
        "launch_id": value["launch_id"],
        "file_sha256": _sha(args.path.read_bytes()),
        "f0_authorization_id": value["f0_authorization_id"],
        "source_commit": value["intended_f1_source_commit"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
