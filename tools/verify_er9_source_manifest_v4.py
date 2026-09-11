#!/usr/bin/env python3
"""Verify the W9 v4 full-tree source closure and live working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from runtime.source_guard import SourceGuardHold, assert_clean_source_closure, assert_v4_manifest_contract  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "results/learned/er9/er_execution_source_manifest_v4.json")
    args = parser.parse_args(argv)
    path = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    value = json.loads(path.read_bytes())
    body = dict(value)
    manifest_id = body.pop("manifest_id", None)
    digest = hashlib.sha256((json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")).hexdigest()
    if manifest_id != "er9sourcev4-" + digest:
        raise SystemExit("v4 source manifest ID differs")
    try:
        assert_v4_manifest_contract(value)
    except SourceGuardHold as exc:
        raise SystemExit(f"v4 source manifest contract differs: {exc}") from None
    report = assert_clean_source_closure(REPO, value)
    print(f"W9 v4 source manifest PASS: {manifest_id}; protected drift=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
