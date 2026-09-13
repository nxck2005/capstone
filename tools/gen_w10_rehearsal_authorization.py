#!/usr/bin/env python3
"""Freeze the W10 validation-only rehearsal authority."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import EXPECTED_SNR_GRID, TITAN_XP_NAME, TITAN_XP_UUID, immutable_write, load_source, source_record  # noqa: E402
from evaluation.w10_rehearsal import W10_SYSTEMS, work_units  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

TARGET = REPO / "results/learned/w10/w10_rehearsal_authorization.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args(argv)
    source = load_source(REPO)
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    if (args.gpu_name, args.gpu_uuid) != (TITAN_XP_NAME, TITAN_XP_UUID):
        raise SystemExit("final downstream W10 requires the exact Confessor TITAN Xp")
    body = {
        "schema_version": 1, "authority_kind": "W10_VALIDATION_REHEARSAL_AUTHORITY",
        "status": "FROZEN_W10_ONLY_PRE_EXECUTION", "source_manifest": source_record(REPO, source),
        "source_commit": source["source_commit"], "source_binding": source,
        "split": "val", "cell": {"train_seed": 0, "channel_seed": 0},
        "snr_grid_db": list(EXPECTED_SNR_GRID),
        "systems": list(W10_SYSTEMS), "unit_count": len(work_units()),
        "runtime_root": "results/learned/w10/runtime", "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor", "gpu_name": args.gpu_name, "gpu_uuid": args.gpu_uuid,
        "compute_capability": str(profile["compute_capability"]), "device": "cuda:0", "cuda_visible_devices": args.gpu_uuid,
        "production_training_authorized": False, "er2_training_authorized": False, "g11_authorized": False,
        "test_authorized": False, "validation_only": True, "test": "SEALED", "test_access": 0,
    }
    live = authenticate_live_w9_pascal(REPO, body, config_hash=canonical_sha256({"scope": "W10", "units": work_units()}))
    body["cuda_mapping"] = live["environment"]["cuda_mapping"]
    body["authority_id"] = "w10rehearsalauth-" + canonical_sha256(body)
    immutable_write(TARGET, body)
    print(f"W10 rehearsal authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
