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
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates  # noqa: E402
from runtime.source_guard import SourceGuardHold, assert_clean_source_closure  # noqa: E402
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
    try:
        assert_clean_source_closure(REPO, manifest)
    except SourceGuardHold as exc:
        raise SystemExit(f"v4 source closure is not authenticated: {exc}") from None
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    if args.gpu_name not in profile["allowed_gpu_names"] or args.gpu_uuid not in profile["allowed_gpu_uuids"]:
        raise SystemExit("GPU is not registered in confessor_pascal_cu126")
    if args.device != "cuda:0":
        raise SystemExit("W9 v4 Stage-1 requires cuda:0")
    config = load_experiment("configs/er9-digital-pascal-v4.yaml", train_seed=0, channel_seed=0)
    floor = packetisation_floor(int(config.resolved["k"]))
    configured = all_configured_pairs()
    admissible = feasible_pairs(floor.payload_bits)
    candidates = stage1_candidates(floor.payload_bits)
    if (
        int(config.resolved["k"]) != 12800
        or floor.payload_bits != 4248
        or len(configured) != 32
        or len(admissible) != 19
        or tuple(item.transmit_dim for item in candidates) != (64, 128, 256, 512, 1024, 2048)
        or any(item.quantiser_bits != 2 for item in candidates)
    ):
        raise SystemExit("v4 Stage-1 solver result differs from the frozen AM-96 expectation; HOLD")
    candidate_dicts = [item.as_dict() for item in candidates]
    configured_dicts = [item.as_dict() for item in configured]
    admissible_dicts = [item.as_dict() for item in admissible]
    rejected_dicts = [
        {**item.as_dict(), "reason": "raw_bound_exceeds_A_floor"}
        for item in configured
        if item not in admissible
    ]
    config_relative = "configs/er9-digital-pascal-v4.yaml"
    config_hash_from_manifest = manifest.get("relevant_config_sha256", {}).get(config_relative)
    if config_hash_from_manifest is None:
        raise SystemExit("v4 source manifest does not bind the Pascal Stage-1 config")
    body = {
        "schema_version": 1,
        "authority_kind": "W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4",
        "status": "FROZEN_STAGE1_ONLY_PRE_SCIENCE",
        "authorization_scope": "W9_ER9_STAGE1_ONLY",
        "source_manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "manifest_id": manifest["manifest_id"],
            "sha256": _sha(MANIFEST),
        },
        "source_commit": manifest["source_commit"],
        "source_binding": manifest,
        "config_path": "configs/er9-digital-pascal-v4.yaml",
        "config_hash": config_hash(config),
        "config_source_blob_sha256": config_hash_from_manifest,
        "runtime_root": "checkpoints/er9_pascal_v4",
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": args.gpu_name,
        "gpu_uuid": args.gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": args.device,
        "cuda_visible_devices": args.gpu_uuid,
        "cuda_mapping": {
            "cuda_visible_devices": args.gpu_uuid,
            "logical_device": "cuda:0",
            "cuda0_gpu_uuid": args.gpu_uuid,
            "cuda0_gpu_name": args.gpu_name,
            "cuda0_compute_capability": str(profile["compute_capability"]),
            "device_count": 1,
        },
        "sole_writer": True,
        "live_authentication_required_immediately_before_model_and_data": True,
        "source_working_tree_guard_required": True,
        "dataset": "imagenette160",
        "split": "train_then_validation_only",
        "ratio": "r_1_6",
        "k_symbols": int(config.resolved["k"]),
        "metadata_bits": 1,
        "packet_floor": floor.as_dict(),
        "configured_pair_count": len(configured_dicts),
        "admissible_pair_count": len(admissible_dicts),
        "admissible_pairs": admissible_dicts,
        "rejected_pairs": rejected_dicts,
        "search_seed_cell": {"train_seed": 0, "channel_seed": 0},
        "evaluation_snr_db": 7,
        "stage1_candidates": candidate_dicts,
        "stage1_candidate_count": len(candidate_dicts),
        "stage1_training_count": len(candidate_dicts),
        "stage1_rule": {
            "candidate_order": "ascending_numeric_transmit_dim",
            "cross_product": False,
            "quantiser_bits": 2,
            "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
            "tie_break": "smallest_transmit_dim",
        },
        "checkpoint_selection_rule": {
            "metric": "validation_n_correct",
            "mode": "max",
            "tie_break": "earliest_epoch",
        },
        "packet_budget_admissibility_proof": {
            "k_symbols": int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
            "metadata_bits": 1,
            "packet_floor": floor.as_dict(),
            "configured_pairs": configured_dicts,
            "admissible_pairs": admissible_dicts,
            "rejected_pairs": rejected_dicts,
            "stage1_candidates": candidate_dicts,
            "configured_pair_count": len(configured_dicts),
            "admissible_pair_count": len(admissible_dicts),
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
