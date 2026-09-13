#!/usr/bin/env python3
"""Freeze one validation-only G11/H4 execution after its inputs close."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import (  # noqa: E402
    G11_AUTHORITY_PATH,
    PRODUCTION_CELLS,
    immutable_write,
    load_source,
    read_json,
    sha256_file,
    source_record,
)
from training.deterministic_core import canonical_sha256  # noqa: E402

G10_MANIFEST = REPO / "results/learned/w9/g10_runtime_manifest.json"
ER9_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"
ER2_COMPLETION = REPO / "results/learned/er2_randomized/er2_randomized_completion_v4.json"


def main() -> int:
    source = load_source(REPO)
    subprocess.run([sys.executable, str(REPO / "tools/verify_g10_w9.py")], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(REPO / "tools/verify_er9_production_v4.py"), "--terminal"], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(REPO / "tools/verify_er2_randomized.py")], cwd=REPO, check=True)
    er9 = read_json(ER9_CLOSEOUT, "ER-9 production closeout")
    er2 = read_json(ER2_COMPLETION, "randomized ER-2 completion")
    body = {
        "schema_version": 1,
        "authority_kind": "G11_H4_VALIDATION_ONLY_EXECUTION_AUTHORITY",
        "status": "FROZEN_G11_ONLY_PRE_EXECUTION",
        "source_manifest": source_record(REPO, source),
        "source_commit": source["source_commit"],
        "source_binding": source,
        "input_bindings": {
            "ordinary_learned_w8_g10": {
                "path": str(G10_MANIFEST.relative_to(REPO)),
                "sha256": sha256_file(G10_MANIFEST),
            },
            "final_production_er9": {
                "path": str(ER9_CLOSEOUT.relative_to(REPO)),
                "sha256": sha256_file(ER9_CLOSEOUT),
                "closeout_id": er9["closeout_id"],
            },
            "randomized_er2_prerequisite_not_comparator": {
                "path": str(ER2_COMPLETION.relative_to(REPO)),
                "sha256": sha256_file(ER2_COMPLETION),
                "completion_id": er2["completion_id"],
            },
        },
        "cells": [list(cell) for cell in PRODUCTION_CELLS],
        "learned_arm": "ordinary_learned_w8_g10",
        "comparator": "final_production_er9",
        "randomized_er2_as_h4_arm": False,
        "bootstrap_resamples": 10000,  # literal-ok: AM-97 frozen resamples
        "quantiles": ["q50", "q95", "q99"],
        "reference_pp": 2,
        "pointwise_only": True,
        "full_h4_power_claim": False,
        "g11_execution_count": 1,
        "training_authorized": False,
        "w10_authorized": False,
        "test_authorized": False,
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["authority_id"] = "g11h4auth-" + canonical_sha256(body)
    immutable_write(REPO / G11_AUTHORITY_PATH, body)
    print(f"G11/H4 authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
