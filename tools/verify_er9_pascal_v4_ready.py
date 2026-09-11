#!/usr/bin/env python3
"""Perform visible W9 Pascal v4 prelaunch checks without model/data creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates  # noqa: E402
from runtime.source_guard import SourceGuardHold, assert_clean_source_closure, assert_v4_manifest_contract  # noqa: E402
from runtime.w9_authority import W9AuthorityHold, authenticate_live_w9_pascal, load_authority, resolve_runtime_root  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


AUTHORITY = REPO / "results/learned/er9/er9_stage1_execution_authorization_v4.json"
MANIFEST = REPO / "results/learned/er9/er_execution_source_manifest_v4.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W9AuthorityHold(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    authority = load_authority(AUTHORITY, kind="W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4")
    body = dict(authority)
    authority_id = body.pop("authority_id", None)
    _require(authority_id == "w9er9stage1v4auth-" + canonical_sha256(body), "Stage-1 authority ID differs")
    _require(MANIFEST.is_file() and _sha(MANIFEST) == authority["source_manifest"]["sha256"], "Stage-1 source manifest binding differs")
    manifest = json.loads(MANIFEST.read_bytes())
    manifest_body = dict(manifest)
    manifest_id = manifest_body.pop("manifest_id", None)
    _require(manifest_id == "er9sourcev4-" + canonical_sha256(manifest_body), "Stage-1 source manifest ID differs")
    try:
        assert_v4_manifest_contract(manifest)
    except SourceGuardHold as exc:
        raise W9AuthorityHold(f"Stage-1 source manifest contract differs: {exc}") from None
    _require(authority.get("schema_version") == 1, "Stage-1 authority schema differs")
    _require(authority.get("status") == "FROZEN_STAGE1_ONLY_PRE_SCIENCE", "Stage-1 authority status differs")
    source_record = authority.get("source_manifest")
    _require(
        source_record == {
            "path": str(MANIFEST.relative_to(REPO)),
            "manifest_id": manifest_id,
            "sha256": _sha(MANIFEST),
        },
        "Stage-1 source manifest record differs",
    )
    _require(authority.get("source_commit") == manifest.get("source_commit"), "Stage-1 source commit differs")
    _require(manifest.get("manifest_id") == authority["source_manifest"]["manifest_id"], "Stage-1 source manifest ID differs")
    try:
        assert_clean_source_closure(REPO, manifest)
    except SourceGuardHold as exc:
        raise W9AuthorityHold(f"Stage-1 source closure differs: {exc}") from None
    _require(authority.get("source_binding") == manifest, "Stage-1 full source binding differs")
    _require(authority.get("authorization_scope") == "W9_ER9_STAGE1_ONLY", "Stage-1 authority scope differs")
    _require(authority.get("config_path") == "configs/er9-digital-pascal-v4.yaml", "Stage-1 config path differs")
    config = load_experiment(str(REPO / authority["config_path"]), train_seed=0, channel_seed=0)
    _require(config_hash(config) == authority["config_hash"], "Stage-1 config hash differs")
    _require(authority.get("config_source_blob_sha256") == manifest["relevant_config_sha256"][authority["config_path"]], "Stage-1 config source blob differs")
    _require(authority["execution_profile_id"] == "confessor_pascal_cu126" and authority["host"] == "confessor", "Stage-1 is not Pascal/Confessor bound")
    _require(authority["runtime_root"] == "checkpoints/er9_pascal_v4", "Stage-1 runtime root differs")
    _require(authority.get("device") == "cuda:0" and authority.get("cuda_visible_devices") == authority.get("gpu_uuid"), "Stage-1 CUDA mapping authority differs")
    mapping = authority.get("cuda_mapping")
    _require(
        mapping == {
            "cuda_visible_devices": authority.get("gpu_uuid"),
            "logical_device": "cuda:0",
            "cuda0_gpu_uuid": authority.get("gpu_uuid"),
            "cuda0_gpu_name": authority.get("gpu_name"),
            "cuda0_compute_capability": authority.get("compute_capability"),
            "device_count": 1,
        },
        "Stage-1 CUDA mapping record differs",
    )
    _require(
        authority.get("sole_writer") is True
        and authority.get("live_authentication_required_immediately_before_model_and_data") is True
        and authority.get("source_working_tree_guard_required") is True,
        "Stage-1 execution safety policy differs",
    )
    _require(authority.get("v1_v2_v3_checkpoints_ineligible") is True and authority.get("smoke_checkpoints_ineligible") is True, "Stage-1 checkpoint custody policy differs")
    _require(authority["stage2_authorized"] is False and authority["production_authorized"] is False and authority["randomized_er2_authorized"] is False, "authority scope is wider than Stage-1")
    _require(authority["fresh_initialization_required"] is True, "Stage-1 does not require fresh initialization")
    _require(authority["test"] == "SEALED" and authority["test_access"] == 0, "Stage-1 test boundary differs")
    floor = packetisation_floor(int(config.resolved["k"]))
    configured = all_configured_pairs()
    admissible = feasible_pairs(floor.payload_bits)
    candidates = stage1_candidates(floor.payload_bits)
    _require(authority.get("k_symbols") == int(config.resolved["k"]) and authority.get("metadata_bits") == 1, "Stage-1 packet inputs differ")
    _require(authority.get("packet_floor") == floor.as_dict(), "Stage-1 packet floor differs")
    _require(authority.get("configured_pair_count") == len(configured) and authority.get("admissible_pair_count") == len(admissible), "Stage-1 solver counts differ")
    _require(authority.get("admissible_pairs") == [item.as_dict() for item in admissible], "Stage-1 admissible pair set differs")
    _require(authority.get("rejected_pairs") == [{**item.as_dict(), "reason": "raw_bound_exceeds_A_floor"} for item in configured if item not in admissible], "Stage-1 rejected pair set differs")
    _require(authority.get("stage1_candidates") == [item.as_dict() for item in candidates] and authority.get("stage1_candidate_count") == len(candidates) and authority.get("stage1_training_count") == len(candidates), "Stage-1 solver candidates differ")
    _require(authority.get("checkpoint_selection_rule") == {"metric": "validation_n_correct", "mode": "max", "tie_break": "earliest_epoch"}, "Stage-1 checkpoint selection rule differs")
    runtime = resolve_runtime_root(REPO, authority)
    _require(not runtime.exists() and not runtime.is_symlink(), "Stage-1 runtime already exists; fresh initialization is not safe")
    authenticate_live_w9_pascal(REPO, authority, config_hash=authority["config_hash"])
    counters = authority["pre_execution_counters"]
    _require(
        counters == {
            "new_er9_v4_training": 0,
            "randomized_er2_scientific_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "prelaunch scientific counters are not zero",
    )
    print("PASCAL EXEC READY:")
    print("host=confessor")
    print("profile=confessor_pascal_cu126")
    print(f"gpu={authority['gpu_name']}")
    print(f"uuid={authority['gpu_uuid']}")
    print(f"source={authority['source_commit']}")
    print(f"source_manifest={manifest['manifest_id']}")
    print(f"authority={authority_id}")
    print(f"runtime_root={authority['runtime_root']}")
    print(f"stage1_candidates={authority['stage1_candidate_count']}")
    print("test=SEALED")
    print("scientific_optimizer_steps=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
