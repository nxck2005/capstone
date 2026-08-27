#!/usr/bin/env python3
"""Fail-closed verification of the W6-A pre-test classical evidence boundary."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline.w6_evidence import W6Hold, verify_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-upstream", action="store_true", help="skip expensive upstream replay; inner and exact bindings still verify")
    args = parser.parse_args()
    try:
        index, matrix = verify_all(invoke_upstream=not args.no_upstream)
    except (W6Hold, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True), file=sys.stderr); return 2
    print(json.dumps({"status": "PASS", "stage": index["stage"], "index_id": index["index_id"], "matrix_id": matrix["matrix_id"], "counts": matrix["counts"], "pass_two": 1, "pass_three": 0, "scientific_learned_training": 0, "test": 0, "terminal_w6_completion_published": False}, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
