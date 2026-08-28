#!/usr/bin/env python3
"""Authenticate the final W7-A source/profile binding without training.

This is a final-source, non-scientific confirmation.  It probes only the
selected Pascal GPU and the frozen profile/config lineage; it performs no data
loading, optimizer step, validation inference, candidate evaluation or test
access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import authenticate_w7_gpu, verify_frozen_gpu_binding  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_g4 import profile_config  # noqa: E402
from training.w7_protocol import (  # noqa: E402
    W7_EXECUTION_IMAGE_FAMILY,
    W7_PROFILE_ID,
    W7_VALIDATION_BATCH_SIZE,
)
from gen_w7_source_manifest import verify as verify_source_manifest  # noqa: E402
from verify_w7_a import verify_profile_freeze  # noqa: E402

OUTPUT = REPO / "results/learned/w7/w7_final_source_profile_confirmation.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"final W7 source confirmation already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise RuntimeError(f"final W7 source confirmation already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write(
    *,
    source_manifest_path: Path,
    profile_freeze_path: Path,
    gpu_uuid: str,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"final W7 source confirmation already exists: {output}")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("final W7 source confirmation requires a clean checkout")
    source_manifest = json.loads(source_manifest_path.read_bytes())
    verify_source_manifest(source_manifest, current=True)
    freeze = verify_profile_freeze(json.loads(profile_freeze_path.read_bytes()))
    if freeze["execution_profile_id"] != W7_PROFILE_ID or freeze["gpu_uuid"] != gpu_uuid:
        raise RuntimeError("final W7 source confirmation GPU/profile differs")
    config = profile_config(
        physical_batch_size=int(freeze["physical_batch_size"]),
        accumulation_factor=int(freeze["accumulation_factor"]),
    )
    config_digest = run_config_hash(config)
    if config_digest != freeze["profile_config_hash"]:
        raise RuntimeError("final W7 source confirmation config differs from profile freeze")
    if freeze["validation_batch_size"] != W7_VALIDATION_BATCH_SIZE:
        raise RuntimeError("final W7 source confirmation validation batch differs")
    binding = authenticate_w7_gpu(
        config_hash=config_digest,
        expected_gpu_uuid=gpu_uuid,
        device="cuda:0",
        require_openjpeg=False,
    )
    verify_frozen_gpu_binding(binding, config_hash=config_digest)
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "W7_FINAL_SOURCE_PROFILE_CONFIRMATION",
        "status": "PASSED",
        "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
        "execution_profile_id": W7_PROFILE_ID,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "source_lineage": {
            "checkout_commit": _git("rev-parse", "HEAD"),
            "execution_source_commit": source_manifest["source_commit"],
            "source_manifest_id": source_manifest["manifest_id"],
            "source_manifest_sha256": _sha(source_manifest_path),
        },
        "profile_freeze": {
            "path": str(profile_freeze_path.relative_to(REPO)),
            "profile_freeze_id": freeze["profile_freeze_id"],
            "file_sha256": _sha(profile_freeze_path),
        },
        "config_hash": config_digest,
        "gpu_binding": binding,
        "batch": {
            "physical_batch_size": freeze["physical_batch_size"],
            "accumulation_factor": freeze["accumulation_factor"],
            "effective_batch_size": freeze["effective_batch_size"],
            "validation_batch_size": freeze["validation_batch_size"],
        },
        "optimizer_steps": 0,
        "validation": {"performed": False},
        "g4_coverage": 0,
        "test_access": 0,
        "protected_counters": {
            "w7_scientific_optimizer_steps": 0,
            "w7_lambda_pilot_runs": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
        },
    }
    value["confirmation_id"] = "w7profileconfirm-" + canonical_sha256(value)
    _publish_immutable(output, canonical_bytes(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--profile-freeze", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = write(
        source_manifest_path=args.source_manifest,
        profile_freeze_path=args.profile_freeze,
        gpu_uuid=args.gpu_uuid,
        output=args.output,
    )
    print(f"W7 final-source profile confirmation PASS: {value['confirmation_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
