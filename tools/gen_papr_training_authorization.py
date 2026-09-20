#!/usr/bin/env python3
"""Freeze the one-run PAPR-constrained training authority (AM-98). Not run by CI.

``--preflight`` validates the projected config, the active successor source
epoch and every frozen precondition without writing anything.  A real freeze
only writes the immutable authority bytes; committing them is the evidence
publication that defines the worker's execution commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.execution_profiles import authenticate_cuda_visible_mapping  # noqa: E402
from config.w8_execution import authenticate_w8_gpu  # noqa: E402
from evaluation.downstream_v4 import immutable_write  # noqa: E402
from runtime.source_epochs import load_w10_manifest, source_record  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_AUTHORITY_PREFIX,
    PAPR_CHANNEL_SEED,
    PAPR_GPU_NAME,
    PAPR_GPU_UUID,
    PAPR_RUN_DIRECTORY,
    PAPR_RUNTIME_ROOT,
    PAPR_SCHEMA_VERSION_AUTHORITY,
    PAPR_TRAIN_SEED,
    load_papr_config,
    papr_authority_protocol,
    papr_protocol_config_hash,
)

TARGET = REPO / PAPR_AUTHORITY_PATH


def authenticate_papr_cuda_mapping() -> dict:
    """Authenticate the physical GPU to logical ``cuda:0`` in this process."""

    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    return authenticate_cuda_visible_mapping(
        expected_gpu_uuid=PAPR_GPU_UUID,
        device="cuda:0",
        expected_gpu_name=PAPR_GPU_NAME,
        expected_compute_capability=str(profile["compute_capability"]),
    )


def build_authority(*, config, source: dict, gpu_name: str, gpu_uuid: str, cuda_mapping: dict) -> dict:
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    record = source_record(REPO, source)
    protocol = papr_authority_protocol(config, source_record_value=record)
    body: dict = {
        "schema_version": PAPR_SCHEMA_VERSION_AUTHORITY,
        "authority_kind": "W10_PAPR_CONSTRAINED_TRAINING_AUTHORITY",
        "status": "FROZEN_PRE_EXECUTION",
        "authorization_scope": "PAPR_CONSTRAINED_TRAINING_ONLY",
        "source_manifest": record,
        "source_binding": source,
        "source_commit": source["source_commit"],
        "protocol": protocol,
        "protocol_config_hash": papr_protocol_config_hash(config),
        "config_path": "configs/learned-papr-constrained-r1-6.yaml",
        "config_hash": run_config_hash(config),
        "runtime_root": f"{PAPR_RUNTIME_ROOT}/{PAPR_RUN_DIRECTORY}",
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": "cuda:0",
        "cuda_visible_devices": gpu_uuid,
        "training_count": 1,
        "w10_authorized": False,
        "test_authorized": False,
        "test": "SEALED",
        "test_access": 0,
    }
    body["cuda_mapping"] = cuda_mapping
    body["authority_id"] = PAPR_AUTHORITY_PREFIX + canonical_sha256(body)
    return body


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
    protocol = papr_authority_protocol(config, source_record_value=source_record(REPO, source))
    if args.preflight:
        print(
            f"PAPR training preflight: cap={protocol['papr_cap_db']} runs={protocol['training_runs']} "
            f"config_hash={protocol['config_hash']} manifest={source['manifest_id']}"
        )
        return 0
    if TARGET.is_file() or TARGET.is_symlink():
        raise SystemExit("PAPR training authority already exists and is immutable")
    authenticate_w8_gpu(
        config_hash=run_config_hash(config), expected_gpu_uuid=PAPR_GPU_UUID
    )
    # W8 authentication deliberately retains its historical/general contract;
    # the PAPR authority additionally requires the process-local logical-device
    # mapping that the worker will use.
    cuda_mapping = authenticate_papr_cuda_mapping()
    body = build_authority(
        config=config,
        source=source,
        gpu_name=args.gpu_name,
        gpu_uuid=args.gpu_uuid,
        cuda_mapping=cuda_mapping,
    )
    immutable_write(TARGET, body)
    print(f"PAPR constrained training authority: {body['authority_id']}")
    print("Commit this exact file; the worker must run at that commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
