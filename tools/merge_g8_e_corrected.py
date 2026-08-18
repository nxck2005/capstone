#!/usr/bin/env python3
"""Execute the already-frozen corrected E3 exact-set merge."""

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
            raise corrected.CorrectedG8EError("E3 is source-frozen but execution-gated")
        bundle = corrected.verify_corrected_bundle()
        sample_ids = tuple(__import__("baseline.g8_e_corrected", fromlist=["_manifest_ids"])._manifest_ids())
        records = []
        records_dir = args.runtime_root / "records"
        for path in sorted(records_dir.glob("*.json")):
            records.append(json.loads(path.read_text()))
        merged = corrected.merge_e3_records(authority=bundle["authority"], sample_ids=sample_ids, record_values=records)
        args.output.write_bytes(corrected.rendered_json(merged))
        print({"status": "PASS", "work_units": merged["work_unit_count"], "coverage_digest": merged["coverage_digest"]})
    except (OSError, corrected.CorrectedG8EError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
