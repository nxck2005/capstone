#!/usr/bin/env python3
"""Publish corrected-v3 E4 from one exact SHA-bound E3 closure."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402


def main(argv: Sequence[str] | None = None, *, fixture: Mapping[str, Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=v3.V3_RUNTIME_ROOT)
    parser.add_argument("--e3", type=Path)
    parser.add_argument("--e3-sha256", required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        print("REFUSED: E4 requires --execute and an exact E3 SHA-256", file=sys.stderr)
        return 2
    try:
        e3_path = args.e3 or args.runtime_root / "e3_exact_set_closure.json"
        if fixture is None:
            if args.runtime_root.resolve() != v3.V3_RUNTIME_ROOT.resolve() or e3_path.resolve() != v3.V3_E3_PATH.resolve():
                raise v3.G8EV3Error("production E4 must use the frozen v3 runtime/E3 paths")
            complete = v3.verify_v3_e3_complete(e3_path=e3_path, e3_sha256=args.e3_sha256)
            contract = complete["contract"]
            authority = v3.load_measurement_authority()
            sample_ids = v3.verify_live_validation_identity(complete["scientific_data_identity"])
            production = True
        else:
            contract = fixture["contract"]
            authority = fixture["authority"]
            sample_ids = fixture["sample_ids"]
            production = False
        value, path, digest = v3.publish_e4_artifact(
            authority=authority,
            sample_ids=sample_ids,
            runtime_root=args.runtime_root,
            contract=contract,
            e3_path=e3_path,
            e3_sha256=args.e3_sha256,
            production=production,
        )
        print({
            "status": value["status"],
            "e4_id": value["e4_id"],
            "sha256": digest,
            "e3_id": value["e3_id"],
            "object_count": value["object_count"],
            "record_traversal_count": value["record_traversal_count"],
            "output": str(path),
        })
        return 0
    except (OSError, v3.G8EV3Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
