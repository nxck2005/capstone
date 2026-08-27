#!/usr/bin/env python3
"""Build or check deterministic W6-A index and requirement matrix artifacts."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline.w6_evidence import INDEX_PATH, MATRIX_PATH, build_index, build_matrix, canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    index = build_index(); matrix = build_matrix(index)
    rendered = ((INDEX_PATH, canonical(index)), (MATRIX_PATH, canonical(matrix)))
    if args.check:
        for path, raw in rendered:
            if not path.is_file() or path.read_bytes() != raw:
                print(f"W6-A HOLD: stale or missing {path.relative_to(ROOT)}", file=sys.stderr); return 2
        print(f"W6-A deterministic artifacts PASS: {index['index_id']} {matrix['matrix_id']}")
        return 0
    for path, raw in rendered:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    print(f"wrote {INDEX_PATH.relative_to(ROOT)}: {index['index_id']}")
    print(f"wrote {MATRIX_PATH.relative_to(ROOT)}: {matrix['matrix_id']}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
