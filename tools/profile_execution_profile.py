#!/usr/bin/env python3
"""Synthetic, non-training performance profile for one execution device."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
for root in (REPO / "src", REPO / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from config.execution_profiles import authenticate_execution_profile, canonical_json_bytes  # noqa: E402
from config.params import get  # noqa: E402
from qualify_execution_profile import _djscc, _synthetic_djscc_config  # noqa: E402


def profile(profile_id: str, device_name: str, *, batch_sizes: list[int], iterations: int) -> dict[str, Any]:
    config = _synthetic_djscc_config()
    config_hash = hashlib.sha256(canonical_json_bytes({"profile": "synthetic_performance", "config": config.to_dict()})).hexdigest()
    environment = authenticate_execution_profile(
        profile_id,
        device=device_name,
        config_hash=config_hash,
        require_openjpeg=False,
        allow_pending_qualification=True,
    )
    measurements = []
    for batch_size in batch_sizes:
        try:
            result = _djscc(torch.device(device_name), batch_size=batch_size, iterations=iterations)
            measurements.append({
                "batch_size": batch_size,
                "status": "PASS",
                "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                "sustained_elapsed_s": result["sustained_elapsed_s"],
                "throughput_images_per_s": result["throughput_images_per_s"],
                "projected_epoch_time_s": float(get("datasets.cifar10.train_images")) / float(result["throughput_images_per_s"]),
                "all_finite": bool(result["finite_loss"] and result["finite_gradients"]),
            })
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            measurements.append({"batch_size": batch_size, "status": "OOM", "error_type": type(exc).__name__})
            torch.cuda.empty_cache()
            break
    passed = [item for item in measurements if item["status"] == "PASS"]
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_synthetic_performance",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile_id,
        "device": device_name,
        "environment": environment,
        "synthetic_data_only": True,
        "training_campaign": False,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "inference": 0,
        "measurements": measurements,
        "safe_batch_size": max((item["batch_size"] for item in passed), default=0),
        "peak_gpu_memory_bytes_at_safe_batch": max((item["peak_gpu_memory_bytes"] for item in passed), default=0),
        "projected_epoch_time_s_at_safe_batch": next((item["projected_epoch_time_s"] for item in reversed(passed) if item["batch_size"] == max(x["batch_size"] for x in passed)), None) if passed else None,
        "notes": "synthetic DJSCC forward/backward only; not a training run and not a scientific timing claim",
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="2,4,8,16,32")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    if not batch_sizes or any(value <= 0 for value in batch_sizes) or args.iterations <= 0:
        parser.error("batch sizes and iterations must be positive")
    report = profile(args.profile, args.device, batch_sizes=batch_sizes, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"profile": args.profile, "device": args.device, "safe_batch_size": report["safe_batch_size"], "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
