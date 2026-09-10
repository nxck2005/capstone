#!/usr/bin/env python3
"""Freeze the Pascal-only, Stage-1-only W9 v4 execution authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


MANIFEST = REPO / "results/learned/er9/er_execution_source_manifest_v4.json"
TARGET = REPO / "results/learned/er9/er9_stage1_execution_authorization_v4.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: dict) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
    if path.exists():
        if path.read_bytes() != raw:
            raise SystemExit(f"immutable Stage-1 authority differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not MANIFEST.is_file():
        raise SystemExit("generate the v4 source manifest first")
    manifest = json.loads(MANIFEST.read_bytes())
    if manifest.get("schema_version") != 2 or not str(manifest.get("manifest_id", "")).startswith("er9sourcev4-"):
        raise SystemExit("v4 source manifest schema differs")
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    if args.gpu_name not in profile["allowed_gpu_names"] or args.gpu_uuid not in profile["allowed_gpu_uuids"]:
        raise SystemExit("GPU is not registered in confessor_pascal_cu126")
    if args.device != "cuda:0":
        raise SystemExit("W9 v4 Stage-1 requires cuda:0")
    config = load_experiment("configs/er9-digital-pascal-v4.yaml", train_seed=0, channel_seed=0)
    candidates = [
        {"transmit_dim": int(value), "quantiser_bits": 2}
        for value in (64, 128, 256, 512, 1024, 2048)
    ]
    body = {
        "schema_version": 1,
        "authority_kind": "W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4",
        "status": "FROZEN_STAGE1_ONLY_PRE_SCIENCE",
        "source_manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "manifest_id": manifest["manifest_id"],
            "sha256": _sha(MANIFEST),
        },
        "source_commit": manifest["source_commit"],
        "source_binding": manifest,
        "config_path": "configs/er9-digital-pascal-v4.yaml",
        "config_hash": config_hash(config),
        "runtime_root": "checkpoints/er9_pascal_v4",
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": args.gpu_name,
        "gpu_uuid": args.gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": args.device,
        "cuda_visible_devices": args.gpu_uuid,
        "sole_writer": True,
        "live_authentication_required_immediately_before_model_and_data": True,
        "source_working_tree_guard_required": True,
        "dataset": "imagenette160",
        "split": "train_then_validation_only",
        "ratio": "r_1_6",
        "search_seed_cell": {"train_seed": 0, "channel_seed": 0},
        "evaluation_snr_db": 7,
        "stage1_candidates": candidates,
        "stage1_candidate_count": len(candidates),
        "stage1_training_count": len(candidates),
        "stage1_rule": {
            "candidate_order": "ascending_numeric_transmit_dim",
            "cross_product": False,
            "quantiser_bits": 2,
            "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
            "tie_break": "smallest_transmit_dim",
        },
        "packet_budget_admissibility_proof": {
            "k_symbols": int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
            "metadata_bits": 1,
            "packet_floor": "exact_bpsk_rate_1_3_packetisation_payload_at_matched_k",
            "configured_transmit_dim_grid": list(get("digital_semantic_control.transmit_dim_grid")),
            "admissible_stage1_candidates": candidates,
            "rejected_stage1_dimensions": [4096, 8192],
        },
        "fresh_initialization_required": True,
        "v1_v2_v3_checkpoints_ineligible": True,
        "smoke_checkpoints_ineligible": True,
        "stage2_authorized": False,
        "production_authorized": False,
        "randomized_er2_authorized": False,
        "pre_execution_counters": {
            "new_er9_v4_training": 0,
            "randomized_er2_scientific_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "test": "SEALED",
        "test_access": 0,
    }
    body["authority_id"] = "w9er9stage1v4auth-" + canonical_sha256(body)
    _write_immutable(TARGET, body)
    print(f"W9 v4 Stage-1 authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
