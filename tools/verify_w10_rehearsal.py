#!/usr/bin/env python3
"""Verify W10's exact validation-only scope, bindings and optional closeout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import read_json, require  # noqa: E402
from evaluation.w10_bindings import pending_states, resolve_scope_bindings  # noqa: E402
from evaluation.w10_rehearsal import build_closeout, validate_unit, W10_SYSTEMS, work_units  # noqa: E402
from evaluation.w10_scope import (  # noqa: E402
    SCOPE,
    W10_DATASET,
    W10_SPLIT,
    W10_VALIDATION_DENOMINATOR,
    scope_identity,
    scope_sha256,
    snr_grid,
    unit_count,
)
from runtime.source_epochs import load_w10_manifest, source_record  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

AUTHORITY = REPO / "results/learned/w10/w10_rehearsal_authorization.json"
CLOSEOUT = REPO / "results/learned/w10/w10_rehearsal_closeout.json"


def verify_authority(path: Path = AUTHORITY) -> dict:
    value = read_json(path, "W10 authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "w10rehearsalauth-" + canonical_sha256(body), "W10 authority ID differs")
    require(value.get("schema_version") == 2 and value.get("authority_kind") == "W10_VALIDATION_REHEARSAL_AUTHORITY", "W10 authority role differs")
    require(value.get("status") == "FROZEN_W10_ONLY_PRE_EXECUTION", "W10 authority status differs")
    source = load_w10_manifest(REPO, live=True)
    require(value.get("source_manifest") == source_record(REPO, source) and value.get("source_binding") == source and value.get("source_commit") == source["source_commit"], "W10 source binding differs")
    require(value.get("scope") == scope_identity(), "W10 authority scope differs")
    require(value.get("scope_sha256") == scope_sha256(), "W10 authority scope digest differs")
    require(value.get("scope_table_module") == "src/evaluation/w10_scope.py", "W10 scope table module differs")
    require(value.get("snr_grid_db") == list(snr_grid()), "W10 authority SNR grid differs")
    require(value.get("systems") == list(dict.fromkeys(entry.system for entry in SCOPE)), "W10 authority systems differ")
    require(value.get("ratios") == sorted({entry.bw_ratio for entry in SCOPE}), "W10 authority ratios differ")
    require(value.get("learned_ratios") == list(get("evaluation.w10_learned_ratios")), "W10 authority learned ratios differ")
    require(value.get("cell") == {"train_seed": 0, "channel_seed": 0} and value.get("split") == W10_SPLIT and value.get("dataset") == W10_DATASET, "W10 authority cell/split/dataset differs")
    require(value.get("denominator") == W10_VALIDATION_DENOMINATOR, "W10 authority denominator differs")
    require(value.get("unit_count") == unit_count() and value.get("unit_count_derived_from_scope") is True, "W10 authority unit count differs")
    bindings = resolve_scope_bindings(REPO)
    require(value.get("bindings") == bindings, "W10 authority bindings differ from the live frozen artifacts")
    require(not pending_states(bindings), "W10 authority carries a pending binding")
    require(value.get("validation_only") is True and value.get("test_authorized") is False and value.get("test_access") == 0 and value.get("test") == "SEALED", "W10 authority test boundary differs")
    require(value.get("papr_training_authorized") is False and value.get("papr_training_run_count") == 0, "W10 authority improperly authorizes PAPR training")
    require(value.get("host") == "confessor" and value.get("device") == "cuda:0", "W10 authority host/device differs")
    require(value.get("gpu_name") == "NVIDIA TITAN Xp" and value.get("gpu_uuid") == "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a", "W10 authority worker is not the exact Confessor TITAN Xp")  # literal-ok: frozen profile identity
    require(value.get("cuda_visible_devices") == value.get("gpu_uuid"), "W10 authority CUDA binding differs")
    mapping = value.get("cuda_mapping")
    require(mapping == {
        "cuda_visible_devices": value.get("gpu_uuid"),
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": value.get("gpu_uuid"),
        "cuda0_gpu_name": value.get("gpu_name"),
        "cuda0_compute_capability": value.get("compute_capability"),
        "device_count": 1,
    }, "W10 authority CUDA mapping differs")
    return value


def verify_terminal() -> None:
    authority = verify_authority()
    runtime = REPO / str(authority["runtime_root"])
    units = []
    for expected in work_units():
        path = runtime / "units" / f"{expected['ordinal']:03d}-{expected['system']}-{expected['bw_ratio']}-snr{int(expected['snr_db']):+03d}.json"
        value = read_json(path, f"W10 unit {expected['ordinal']}")
        validate_unit(value, expected)
        units.append(value)
    body, units_manifest, images_manifest = build_closeout(runtime, units, authority=authority)
    stored_units = read_json(runtime / "unit_manifest.json", "W10 unit manifest")
    stored_images = read_json(runtime / "per_image_manifest.json", "W10 per-image manifest")
    require(stored_units == units_manifest and stored_images == images_manifest, "W10 closeout manifests differ from the rebuilt evidence")
    stored = read_json(CLOSEOUT, "W10 closeout")
    require(stored == body, "W10 closeout does not content-bind the exact completed evidence")
    require(stored.get("unit_count") == unit_count(), "W10 closeout unit count differs")
    require(stored.get("ordered_unit_ids_digest") == units_manifest["ordered_unit_ids_digest"], "W10 closeout unit digest differs")
    require(stored.get("ordered_per_image_digest") == images_manifest["ordered_per_image_digest"], "W10 closeout per-image digest differs")
    require(stored.get("test") == "SEALED" and stored.get("test_access") == 0, "W10 closeout crossed test boundary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    verify_authority()
    if args.terminal:
        verify_terminal()
    print("W10 rehearsal verifier PASS" + (": terminal" if args.terminal else ": authority only"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
