#!/usr/bin/env python3
"""Prove the batched classical path reproduces committed W10 per-image rows exactly.

Re-runs chosen W10 validation classical units (JPEG 2000 and JPEG) with
``batched_classical=True`` under the frozen W10 authority bindings and compares
every per-image row, for every scorer variant, against the W10 runtime files.
Validation only; it never reaches a test loader.

    python tools/check_batched_classical_equivalence.py --ordinals 88 96 100 117 140 162 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

AUTHORITY = REPO / "results/learned/w10/w10_rehearsal_authorization.json"
RUNTIME = REPO / "checkpoints/w10_rehearsal/per_image"


def _rows(path: Path) -> list[dict]:
    value = json.loads(path.read_bytes())
    for key in ("rows", "per_image"):
        if key in value:
            return value[key]
    raise SystemExit(f"no rows in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from evaluation.w10_backends import W10Execution
    from evaluation.w10_classical import classical_unit
    from evaluation.w10_scope import entry_for, work_units

    authority = json.loads(AUTHORITY.read_bytes())
    bindings = {str(item["scope"]["role"]): item for item in authority["bindings"]}
    units = {int(unit["ordinal"]): unit for unit in work_units()}
    context = W10Execution(root=REPO, device=args.device, batched_classical=True)
    failures = 0
    for ordinal in args.ordinals:
        unit = units[ordinal]
        entry = entry_for(str(unit["system"]), str(unit["bw_ratio"]))
        bound = bindings[entry.role]
        started = time.monotonic()
        if entry.backend == "classical":
            aggregate = classical_unit(context, unit, root=REPO, checkpoint_id=str(bound["checkpoint"].get("kind", "untrained")), binding=bound["selection"], codec_kind="jpeg2000")
        elif entry.backend == "jpeg_secondary":
            selection = bound["selection"]
            point = next(item for item in selection["selections"] if float(item["snr_db"]) == float(unit["snr_db"]))
            aggregate = classical_unit(context, unit, root=REPO, checkpoint_id="not_applicable_untrained_codec", binding={"kind": "w10_jpeg_validation_selection", "selection_id": selection["selection_id"], "selections": selection["selections"]}, codec_kind="jpeg", quality=int(point["quality"]))
        else:
            raise SystemExit(f"ordinal {ordinal} is not a classical unit")
        elapsed = time.monotonic() - started
        snr = int(unit["snr_db"])
        stem = f"{ordinal:03d}-{unit['system']}-{unit['bw_ratio']}-snr{snr:+03d}"
        streams = [aggregate["per_image"]] + [extra["per_image"] for extra in aggregate.get("secondary_streams", ())]
        for index, rows in enumerate(streams):
            path = RUNTIME / (f"{stem}.json" if index == 0 else f"{stem}.{index}.json")
            expected = _rows(path)
            same = rows == expected
            failures += not same
            if not same:
                diffs = sum(1 for a, b in zip(rows, expected, strict=False) if a != b)
                print(f"MISMATCH {path.name}: {diffs} rows differ of {len(expected)}")
            else:
                print(f"identical {path.name}: {len(rows)} rows ({elapsed:.0f}s batched)")
    print("EQUIVALENCE PASS" if failures == 0 else f"EQUIVALENCE FAIL: {failures} streams")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
