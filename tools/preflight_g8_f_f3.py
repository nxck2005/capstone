#!/usr/bin/env python3
"""Authenticate/freeze the existing G8_E cache, then issue the F3 contract."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline.g8_f_f3 import (
    CACHE_MANIFEST_PATH,
    CONTRACT_PATH,
    RUNTIME_ROOT,
    atomic_bytes,
    build_cache_manifest,
    build_contract,
    rendered_json,
    verify_contract,
)


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    contract = commands.add_parser("contract")
    contract.add_argument("--source-commit", default=None)
    commands.add_parser("verify-contract")
    args = parser.parse_args()
    if args.command == "cache":
        value, raw = build_cache_manifest(args.runtime_root)
        atomic_bytes(CACHE_MANIFEST_PATH, raw)
        print("F3 cache preflight PASS:", value["cache_manifest_id"], f"rows={value['row_count']}", f"reconstructions={value['unique_reconstruction_object_count']}", "inference=0")
        return 0
    if args.command == "contract":
        value = build_contract(source_commit=args.source_commit or head())
        atomic_bytes(CONTRACT_PATH, rendered_json(value), refuse_existing=True)
        verify_contract(CONTRACT_PATH)
        print("F3 contract freeze PASS:", value["contract_id"], value["source_commit"], "inference=0")
        return 0
    value = verify_contract(CONTRACT_PATH)
    print("F3 contract verification PASS:", value["contract_id"], "inference=0 pass_two=0 test=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
