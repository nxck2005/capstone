#!/usr/bin/env python3
"""Freeze the one-run PAPR-constrained training authority (AM-98). Not run by CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from evaluation.downstream_v4 import immutable_write  # noqa: E402
from runtime.source_epochs import load_w10_manifest, source_record  # noqa: E402
from config.w8_execution import authenticate_w8_gpu  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_GPU_NAME,
    PAPR_GPU_UUID,
    PAPR_RUNTIME_ROOT,
    load_papr_config,
    papr_protocol,
)

TARGET = REPO / PAPR_AUTHORITY_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", default=PAPR_GPU_NAME)
    parser.add_argument("--gpu-uuid", default=PAPR_GPU_UUID)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if (args.gpu_name, args.gpu_uuid) != (PAPR_GPU_NAME, PAPR_GPU_UUID):
        raise SystemExit("PAPR-constrained training requires the exact W8 GTX 1080 Ti")
    source = load_w10_manifest(REPO, live=True)
    config = load_papr_config()
    config_hash_value = run_config_hash(config)
    protocol = papr_protocol(REPO, source_record_value=source_record(REPO, source), config_hash_value=config_hash_value)
    if args.preflight:
        print(f"PAPR training preflight: cap={protocol['papr_cap_db']} runs={protocol['training_runs']} config_hash={config_hash_value}")
        return 0
    if TARGET.is_file() or TARGET.is_symlink():
        raise SystemExit("PAPR training authority already exists and is immutable")
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    body = {
        "schema_version": 1,
        "authority_kind": "W10_PAPR_CONSTRAINED_TRAINING_AUTHORITY",
        "status": "FROZEN_PRE_EXECUTION",
        "authorization_scope": "PAPR_CONSTRAINED_TRAINING_ONLY",
        "source_manifest": source_record(REPO, source),
        "source_commit": source["source_commit"],
        "source_binding": source,
        "protocol": protocol,
        "config_path": "configs/learned-papr-constrained-r1-6.yaml",
        "config_hash": config_hash_value,
        "runtime_root": f"{PAPR_RUNTIME_ROOT}/train0_channel0",
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": args.gpu_name,
        "gpu_uuid": args.gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": "cuda:0",
        "cuda_visible_devices": args.gpu_uuid,
        "training_count": 1,
        "w10_authorized": False,
        "test_authorized": False,
        "test": "SEALED",
        "test_access": 0,
    }
    binding = authenticate_w8_gpu(config_hash=config_hash_value, expected_gpu_uuid=PAPR_GPU_UUID)
    body["cuda_mapping"] = binding["profile_environment"]["cuda_mapping"]
    body["authority_id"] = "paprtrainingauth-" + canonical_sha256(body)
    immutable_write(TARGET, body)
    print(f"PAPR constrained training authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
