#!/usr/bin/env python3
"""Verify W10's exact validation-only rehearsal scope and optional closeout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import EXPECTED_SNR_GRID, TITAN_XP_NAME, TITAN_XP_UUID, load_source, read_json, require, source_record  # noqa: E402
from evaluation.w10_rehearsal import W10_SYSTEMS, validate_unit, work_units  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

AUTHORITY = REPO / "results/learned/w10/w10_rehearsal_authorization.json"


def verify_authority(path: Path = AUTHORITY) -> dict:
    source = load_source(REPO)
    value = read_json(path, "W10 authority")
    body = dict(value); identifier = body.pop("authority_id", None)
    require(identifier == "w10rehearsalauth-" + canonical_sha256(body), "W10 authority ID differs")
    require(value.get("source_manifest") == source_record(REPO, source) and value.get("source_binding") == source, "W10 source binding differs")
    require(value.get("authority_kind") == "W10_VALIDATION_REHEARSAL_AUTHORITY" and value.get("status") == "FROZEN_W10_ONLY_PRE_EXECUTION", "W10 authority role/status differs")
    require(value.get("split") == "val" and value.get("cell") == {"train_seed": 0, "channel_seed": 0}, "W10 split/cell differs")
    require(value.get("snr_grid_db") == list(EXPECTED_SNR_GRID) and value.get("systems") == list(W10_SYSTEMS) and value.get("unit_count") == len(work_units()), "W10 full-grid/all-systems scope differs")
    require(value.get("validation_only") is True and value.get("test_authorized") is False and value.get("test_access") == 0 and value.get("test") == "SEALED", "W10 test boundary differs")
    require(value.get("host") == "confessor" and value.get("device") == "cuda:0" and value.get("gpu_name") == TITAN_XP_NAME and value.get("gpu_uuid") == TITAN_XP_UUID, "W10 exact Pascal worker differs")
    require(value.get("cuda_visible_devices") == TITAN_XP_UUID and value.get("cuda_mapping") == {
        "cuda_visible_devices": TITAN_XP_UUID,
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": TITAN_XP_UUID,
        "cuda0_gpu_name": TITAN_XP_NAME,
        "cuda0_compute_capability": value.get("compute_capability"),
        "device_count": 1,
    }, "W10 CUDA mapping differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    authority = verify_authority()
    if args.terminal:
        root = REPO / authority["runtime_root"] / "units"
        paths = sorted(root.glob("*.json"))
        require(len(paths) == len(work_units()), "W10 terminal unit count differs")
        for path, expected in zip(paths, work_units(), strict=True):
            validate_unit(json.loads(path.read_bytes()), expected)
        closeout = read_json(REPO / "results/learned/w10/w10_rehearsal_closeout.json", "W10 closeout")
        body = dict(closeout); identifier = body.pop("closeout_id", None)
        require(identifier == "w10closeout-" + canonical_sha256(body), "W10 closeout ID differs")
        require(closeout.get("authority_id") == authority["authority_id"] and closeout.get("source_commit") == authority["source_commit"], "W10 closeout authority/source differs")
        require(closeout.get("split") == "val" and closeout.get("cell") == [0, 0] and closeout.get("snr_grid_db") == list(EXPECTED_SNR_GRID), "W10 closeout scope differs")
        require(closeout.get("systems") == list(W10_SYSTEMS) and closeout.get("unit_count") == len(work_units()), "W10 closeout systems/count differs")
        require(closeout.get("validation_only") is True and closeout.get("test") == "SEALED" and closeout.get("test_access") == 0, "W10 closeout crossed test boundary")
    print("W10 rehearsal verifier PASS" + (": terminal" if args.terminal else ": authority only"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
