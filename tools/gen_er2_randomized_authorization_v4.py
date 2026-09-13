#!/usr/bin/env python3
"""Freeze the future Pascal-only randomized ER-2 v4 execution authority.

This command publishes authority metadata only.  It does not construct a
model, dataset, loader, checkpoint or scientific result.  The later ER-2
runner consumes this authority and the same W9 v4 transactional trainer used
by the ER-9 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.downstream_v4 import (  # noqa: E402
    ER2_AUTHORITY_PATH, ER2_RUNTIME_ROOT, SOURCE_PATH, TITAN_XP_NAME,
    TITAN_XP_UUID, load_source, source_record,
)
from runtime.w9_authority import W9AuthorityHold, authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


MANIFEST = REPO / SOURCE_PATH
TARGET = REPO / ER2_AUTHORITY_PATH
CONFIG_PATH = "configs/learned-er2-randomized-pascal-v4.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise SystemExit(f"immutable ER-2 v4 authority differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("xb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    finally:
        descriptor.close()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    manifest = load_source(REPO)
    manifest_id = manifest["manifest_id"]
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    if (args.gpu_name, args.gpu_uuid) != (TITAN_XP_NAME, TITAN_XP_UUID):
        raise SystemExit("final downstream randomized ER-2 requires the exact Confessor TITAN Xp")
    if args.device != "cuda:0":
        raise SystemExit("W9 v4 randomized ER-2 requires cuda:0")
    config = load_experiment(CONFIG_PATH, train_seed=0, channel_seed=0)
    body: dict[str, Any] = {
        "schema_version": 1,
        "authority_kind": "W9_ER2_RANDOMIZED_EXECUTION_AUTHORITY_V4",
        "status": "FROZEN_ER2_ONLY_PRE_SCIENCE",
        "authorization_scope": "W9_ER2_RANDOMIZED_ONLY",
        "source_manifest": source_record(REPO, manifest),
        "source_commit": manifest["source_commit"],
        "source_binding": manifest,
        "config_path": CONFIG_PATH,
        "config_hash": config_hash(config),
        "config_source_blob_sha256": manifest["relevant_config_sha256"][CONFIG_PATH],
        "runtime_root": ER2_RUNTIME_ROOT,
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
        "seed_cell": {"train_seed": 0, "channel_seed": 0},
        "lambda": 3.0,  # literal-ok: AM-95 frozen lambda_core value
        "randomized_snr_rule": "AM-95_one_keyed_selection_per_sample_per_epoch_reused_for_forward",
        "randomized_snr_domain": [int(value) for value in get("channel.train_snr_db_set")],
        "scientific_training_run_count": 1,
        "fresh_initialization_required": True,
        "v1_v2_v3_checkpoints_ineligible": True,
        "smoke_checkpoints_ineligible": True,
        "stage1_authorized": False,
        "stage2_authorized": False,
        "production_authorized": False,
        "randomized_er2_authorized": True,
        "g11_authorized": False,
        "w10_authorized": False,
        "test_authorized": False,
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
    try:
        live = authenticate_live_w9_pascal(REPO, body, config_hash=body["config_hash"])
    except (W9AuthorityHold, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"live Pascal authority authentication failed: {exc}") from None
    mapping = live.get("environment", {}).get("cuda_mapping")
    if (
        not isinstance(mapping, dict)
        or mapping.get("logical_device") != "cuda:0"
        or mapping.get("cuda0_gpu_uuid") != args.gpu_uuid
        or mapping.get("cuda0_gpu_name") != args.gpu_name
        or mapping.get("cuda0_compute_capability") != str(profile["compute_capability"])
    ):
        raise SystemExit("live Pascal CUDA mapping differs from the selected ER-2 GPU")
    body["cuda_mapping"] = mapping
    body["authority_id"] = "w9er2randomizedv4auth-" + canonical_sha256(body)
    _write_immutable(TARGET, body)
    print(f"W9 v4 randomized ER-2 authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
