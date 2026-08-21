#!/usr/bin/env python3
"""Publish the compact corrected-v3 E3 exact-set closure."""

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
    args = parser.parse_args(argv)
    if not args.execute:
        print("REFUSED: E3 requires --execute after exact E2 completion", file=sys.stderr)
        return 2
    try:
        if fixture is None:
            if args.runtime_root.resolve() != v3.V3_RUNTIME_ROOT.resolve():
                raise v3.G8EV3Error("production E3 must use the frozen v3 runtime root")
            complete = v3.verify_v3_e2_complete(runtime_root=args.runtime_root)
            contract = complete["contract"]
            authority = v3.load_measurement_authority()
            sample_ids, sample_labels = v3.frozen_validation_metadata(complete["scientific_data_identity"])
            production = True
            authenticate_caches = True
        else:
            contract = fixture["contract"]
            authority = fixture["authority"]
            sample_ids = fixture["sample_ids"]
            sample_labels = fixture["sample_labels"]
            production = False
            authenticate_caches = True
            v3.verify_e2_completion_artifact(
                runtime_root=args.runtime_root,
                contract=contract,
                authority=authority,
                production=False,
            )
        value, path, digest = v3.publish_e3_artifact(
            authority=authority,
            sample_ids=sample_ids,
            sample_labels=sample_labels,
            runtime_root=args.runtime_root,
            contract=contract,
            production=production,
            authenticate_caches=authenticate_caches,
        )
        print({
            "status": value["status"],
            "e3_id": value["e3_id"],
            "sha256": digest,
            "required": value["required_work_unit_count"],
            "observed": value["observed_work_unit_count"],
            "missing": value["missing_count"],
            "duplicate": value["duplicate_count"],
            "extra": value["extra_count"],
            "digest": value["ordered_record_sha256"],
            "output": str(path),
        })
        return 0
    except (OSError, v3.G8EV3Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
