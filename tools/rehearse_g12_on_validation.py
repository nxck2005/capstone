#!/usr/bin/env python3
"""Rehearse the exact G-12 routing on the validation split before the freeze.

Runs chosen SNRs of every G-12 arm and cell through ``evaluation.g12_dispatch``
with a validation view, so model loading, per-cell checkpoints, entropy refits,
operating points and row construction fail here rather than after test opens.
It reads no test sample.  Output goes to the worker-local runtime only.

    python tools/rehearse_g12_on_validation.py --snrs -4 7 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "checkpoints/g12_rehearsal/summary.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snrs", type=int, nargs="+", default=[-4, 7])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roles", nargs="*", default=None)
    args = parser.parse_args()

    import torch

    from evaluation.g12_bindings import resolve
    from evaluation.g12_dispatch import dispatch
    from evaluation.g12_scope import work_units
    from evaluation.w10_backends import ValidationView

    bindings = resolve(REPO)
    view = ValidationView()
    backend = dispatch(REPO, device=args.device, bindings=bindings, view=view, split="val", j2k_cache_dir="checkpoints/w10_rehearsal/j2k_cache")
    results = []
    for unit in work_units():
        if int(unit["snr_db"]) not in args.snrs or (args.roles and unit["role"] not in args.roles):
            continue
        rehearsal = {**unit, "split": "val"}
        started = time.monotonic()
        with torch.no_grad():
            aggregate = backend(rehearsal)
        rows = aggregate["per_image"]
        assert len(rows) == 1000 and all(row["split"] == "val" for row in rows)  # literal-ok: validation denominator
        record = {"role": unit["role"], "cell": unit["train_seed"], "snr_db": unit["snr_db"], "n_correct": aggregate["n_correct"], "coverage_rate": aggregate["coverage_rate"], "seconds": round(time.monotonic() - started, 1)}
        results.append(record)
        print(json.dumps(record), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(f"REHEARSAL PASS: {len(results)} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
