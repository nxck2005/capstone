#!/usr/bin/env python3
"""Freeze authority for exactly three fresh final ER-9 production cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import (  # noqa: E402
    FINAL_PAIR, PRODUCTION_AUTHORITY_PATH, PRODUCTION_RUNTIME_ROOT, STAGE2_ID,
    STAGE2_PATH, STAGE2_SHA256, cell_config_records, immutable_write, load_source,
    source_record, TITAN_XP_NAME, TITAN_XP_UUID,
)
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args(argv)
    source = load_source(REPO)
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    if (args.gpu_name, args.gpu_uuid) != (TITAN_XP_NAME, TITAN_XP_UUID):
        raise SystemExit("final downstream production requires the exact Confessor TITAN Xp")
    cells = cell_config_records(source)
    body = {
        "schema_version": 1,
        "authority_kind": "W9_ER9_PRODUCTION_EXECUTION_AUTHORITY_V4",
        "status": "FROZEN_PRODUCTION_ONLY_PRE_SCIENCE",
        "authorization_scope": "W9_ER9_PRODUCTION_ONLY",
        "source_manifest": source_record(REPO, source),
        "source_commit": source["source_commit"],
        "source_binding": source,
        "stage2_closeout": {"path": STAGE2_PATH, "selection_id": STAGE2_ID, "sha256": STAGE2_SHA256},
        "config_path": "configs/er9-digital-pascal-v4.yaml",
        "config_source_blob_sha256": source["relevant_config_sha256"]["configs/er9-digital-pascal-v4.yaml"],
        "selected_pair": FINAL_PAIR,
        "seed_cells": cells,
        "training_count": 3,  # literal-ok: exact frozen production cells
        "epochs": 100,  # literal-ok: AM-96 frozen Imagenette epoch count
        "checkpoint_selection": {"metric": "validation_n_correct", "mode": "max", "tie_break": "earliest_epoch"},
        "no_best_seed_selection": True,
        "stage1_promotion": False,
        "fresh_initialization_required": True,
        "resume_semantics": "exact_authenticated_completed_epoch_model_optimizer_scaler",
        "runtime_root": PRODUCTION_RUNTIME_ROOT,
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": args.gpu_name,
        "gpu_uuid": args.gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": "cuda:0",
        "cuda_visible_devices": args.gpu_uuid,
        "sole_writer": True,
        "live_authentication_required_immediately_before_model_and_data": True,
        "source_working_tree_guard_required": True,
        "er2_authorized": False,
        "g11_authorized": False,
        "w10_authorized": False,
        "test_authorized": False,
        "test": "SEALED",
        "test_access": 0,
    }
    # Authority is intentionally frozen only on the selected Pascal worker.
    live = authenticate_live_w9_pascal(REPO, body, config_hash=str(cells[0]["config_hash"]))
    body["cuda_mapping"] = live["environment"]["cuda_mapping"]
    body["authority_id"] = "w9er9productionv4auth-" + canonical_sha256(body)
    target = REPO / PRODUCTION_AUTHORITY_PATH
    immutable_write(target, body)
    print(f"ER-9 production authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
