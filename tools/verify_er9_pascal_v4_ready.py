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
    parser.add_argument("--skip-live", action="store_true", help="only for unit tests; never a launch authorization")
    args = parser.parse_args(argv)
    authority = load_authority(AUTHORITY, kind="W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4")
    body = dict(authority)
    authority_id = body.pop("authority_id", None)
    _require(authority_id == "w9er9stage1v4auth-" + canonical_sha256(body), "Stage-1 authority ID differs")
    _require(MANIFEST.is_file() and _sha(MANIFEST) == authority["source_manifest"]["sha256"], "Stage-1 source manifest binding differs")
    manifest = json.loads(MANIFEST.read_bytes())
    _require(manifest.get("manifest_id") == authority["source_manifest"]["manifest_id"], "Stage-1 source manifest ID differs")
    config = load_experiment(str(REPO / authority["config_path"]), train_seed=0, channel_seed=0)
    _require(config_hash(config) == authority["config_hash"], "Stage-1 config hash differs")
    _require(authority["execution_profile_id"] == "confessor_pascal_cu126" and authority["host"] == "confessor", "Stage-1 is not Pascal/Confessor bound")
    _require(authority["runtime_root"] == "checkpoints/er9_pascal_v4", "Stage-1 runtime root differs")
    _require(len(authority["stage1_candidates"]) == authority["stage1_candidate_count"] == authority["stage1_training_count"] == 6, "Stage-1 candidate count differs")
    _require(authority["stage2_authorized"] is False and authority["production_authorized"] is False and authority["randomized_er2_authorized"] is False, "authority scope is wider than Stage-1")
    _require(authority["fresh_initialization_required"] is True, "Stage-1 does not require fresh initialization")
    _require(authority["test"] == "SEALED" and authority["test_access"] == 0, "Stage-1 test boundary differs")
    runtime = resolve_runtime_root(REPO, authority)
    _require(not runtime.exists() and not runtime.is_symlink(), "Stage-1 runtime already exists; fresh initialization is not safe")
    if not args.skip_live:
        authenticate_live_w9_pascal(REPO, authority, config_hash=authority["config_hash"])
    counters = authority["pre_execution_counters"]
    _require(all(int(value) == 0 for value in counters.values()), "prelaunch scientific counters are not zero")
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
