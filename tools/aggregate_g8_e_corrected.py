#!/usr/bin/env python3
"""Execute the already-frozen corrected E4 count-derived aggregation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected as corrected  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="required explicit later-phase execution flag")
    args = parser.parse_args()
    try:
        if not args.execute:
            raise corrected.CorrectedG8EError("E4 is source-frozen but execution-gated")
        bundle = corrected.verify_corrected_bundle()
        sample_ids = tuple(__import__("baseline.g8_e_corrected", fromlist=["_manifest_ids"])._manifest_ids())
        records = [json.loads(path.read_text()) for path in sorted((args.runtime_root / "records").glob("*.json"))]
        result = corrected.aggregate_e4_counts(authority=bundle["authority"], sample_ids=sample_ids, record_values=records)
        args.output.write_bytes(corrected.rendered_json(result))
        print({"status": "PASS", "objects": result["object_count"]})
    except (OSError, corrected.CorrectedG8EError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
