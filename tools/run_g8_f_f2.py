#!/usr/bin/env python3
"""Production-only G8_F/F2 BR-12 artifact-classifier entry point."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.execution_profiles import authenticate_execution_profile
from training.g8_f_f2 import (
    EXPECTED_MATERIALIZED,
    EXPECTED_OPTIMIZER_STEPS,
    F2ArtifactDataset,
    F2Hold,
    F2Trainer,
    atomic_bytes,
    canonical_json,
    f2_recipe_sha256,
    sha256_bytes,
)
from training.g8_f_f2_authorization import (
    AUTHORIZATION_PATH,
    EXPECTED_DEVICE,
    EXPECTED_HOST,
    PROFILE_ID,
    verify_authorization,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--resume", action="store_true")
    value.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    value.add_argument("--f1-runtime", type=Path, default=ROOT / "results/baseline/g8_f/runtime")
    value.add_argument("--runtime-root", type=Path, default=ROOT / "results/baseline/g8_f/f2_runtime")
    value.add_argument("--device", default=EXPECTED_DEVICE)
    return value


def main() -> int:
    args = parser().parse_args()
    if socket.gethostname() != EXPECTED_HOST:
        raise F2Hold(f"F2 production host differs: {socket.gethostname()!r}")
    if args.device != EXPECTED_DEVICE:
        raise F2Hold("F2 production device differs")
    authorization = verify_authorization(args.authorization)
    authorization_sha256 = sha256_bytes(args.authorization.read_bytes())
    profile = authenticate_execution_profile(
        PROFILE_ID,
        device=args.device,
        config_hash=f2_recipe_sha256(),
        require_openjpeg=False,
    )
    if profile["git_dirty"]:
        raise F2Hold("F2 execution checkout is dirty")
    if profile["git_commit"] == authorization["source_commit"]:
        pass
    else:
        # Launch HEAD contains only the frozen authorization/evidence/handoff
        # commit above the byte-identical scientific source closure.
        if any((ROOT / entry["path"]).read_bytes() != subprocess.run(
            ["git", "show", f"{authorization['source_commit']}:{entry['path']}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout for entry in authorization["source_closure"]):
            raise F2Hold("launch HEAD changes the frozen F2 scientific source")
    runtime = args.runtime_root.resolve()
    if args.start:
        if runtime.exists():
            raise F2Hold("F2 --start refuses an existing runtime")
    else:
        if not runtime.is_dir() or runtime.is_symlink():
            raise F2Hold("F2 --resume requires the existing regular runtime")
    dataset = F2ArtifactDataset.production(epoch=0, runtime_root=args.f1_runtime.resolve(), authenticate_objects=True)
    if len(dataset) != EXPECTED_MATERIALIZED:
        raise F2Hold("F2 logical dataset length differs")
    trainer = F2Trainer(
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        runtime_root=runtime,
        dataset=dataset,
        device=args.device,
    )
    runtime.mkdir(parents=True, exist_ok=args.resume)
    launch = {
        "schema_version": 1,
        "artifact_role": "g8_f_f2_live_launch",
        "status": "AUTHENTICATED_BEFORE_OPTIMIZER",
        "mode": "resume" if args.resume else "start",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_sha256,
        "source_commit": authorization["source_commit"],
        "launch_head": profile["git_commit"],
        "profile": profile,
        "dataset_summary": dataset.summary.__dict__,
        "g1_parent": authorization["g1_parent"],
        "g1_parent_loaded_and_authenticated": True,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "protected_counters": {"artifact_classifier_optimizer_steps": 0, "f2_checkpoint_selection_validation_inference": 0, "f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0},
    }
    atomic_bytes(runtime / "launch.json", canonical_json(launch))
    if args.resume:
        trainer.resume()
    trainer.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except F2Hold as exc:
        print(f"F2 LAUNCH HOLD — {exc}", file=sys.stderr)
        raise SystemExit(2)
