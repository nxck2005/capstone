#!/usr/bin/env python3
"""Produce the validation-only G11/H4 artifact from authenticated trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.architecture_audit import compare_architectures  # noqa: E402
from evaluation.downstream_v4 import EXPECTED_SNR_GRID, PRODUCTION_CELLS, immutable_write, load_source, read_json, sha256_file, validate_final_er9_cell  # noqa: E402
from evaluation.h4_precision import compute_h4_precision_diagnostic  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_g11 import verify_authority  # noqa: E402

ROOT = REPO / "results/learned/g11"
G10_MANIFEST = REPO / "results/learned/w9/g10_runtime_manifest.json"
ER9_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"


def _g10_trajectories() -> tuple[dict, list[dict]]:
    subprocess.run([sys.executable, str(REPO / "tools/verify_g10_w9.py")], cwd=REPO, check=True)
    manifest = read_json(G10_MANIFEST, "G10 runtime manifest")
    result: dict[str, dict] = {f"{a}/{b}": {} for a, b in PRODUCTION_CELLS}
    bindings = []
    for record in manifest["cells"]:
        path = Path(record["file_path"])
        if sha256_file(path) != record["file_sha256"]:
            raise RuntimeError("G11 G10 cell hash differs")
        cell = read_json(path, "G10 learned validation cell")
        key = f"{cell['train_seed']}/{cell['channel_seed']}"
        snr = str(cell["snr_db"])
        for row in cell["rows"]:
            result[key].setdefault(str(row["stable_sample_id"]), {})[snr] = {"correct": bool(row["correct"])}
        bindings.append({"path": str(path), "sha256": record["file_sha256"], "artifact_id": record["artifact_id"]})
    return result, bindings


def _er9_trajectories() -> tuple[dict, list[dict]]:
    closeout = read_json(ER9_CLOSEOUT, "ER-9 production closeout")
    if closeout.get("training_count") != 3 or closeout.get("stage1_promotion_count") != 0:
        raise RuntimeError("G11 ER-9 production scope differs")
    result: dict[str, dict] = {}
    bindings = []
    for cell, record in zip(PRODUCTION_CELLS, closeout["seed_cells"], strict=True):
        path = REPO / record["validation_path"]
        if sha256_file(path) != record["validation_sha256"]:
            raise RuntimeError("G11 ER-9 validation hash differs")
        value = read_json(path, "ER-9 final validation")
        validate_final_er9_cell(value, cell=cell)
        key = f"{cell[0]}/{cell[1]}"
        result[key] = {}
        for point in value["points"]:
            for row in point["per_image"]:
                result[key].setdefault(str(row["stable_sample_id"]), {})[str(point["snr_db"])] = {"correct": bool(row["correct"])}
        bindings.append({"path": str(path.relative_to(REPO)), "sha256": record["validation_sha256"], "validation_id": value["validation_id"]})
    return result, bindings


def _architecture_audit() -> dict:
    learned = get("learned_system")
    shared = {
        "encoder_arch": "djscc_residual_v1", "encoder_trunk": "shared_pre_channel_residual_trunk",
        "preprocessing": get("preprocessing"), "task_head_arch": "classification_global_average_pool_linear_v1",
        "train_split": "train", "augmentation": learned["augmentation"], "optimizer": learned["optimizer"],
        "epochs": learned["epochs"]["imagenette160"],
    }
    ordinary = {**shared, "interface": {"channel_interface": "analog_complex_power_normalised_awgn"}}
    er9 = {**shared, "interface": {"channel_interface": "scalar_quantise_entropy_frame_ldpc_qpsk_family"}}
    return compare_architectures(ordinary, er9)


def run() -> dict:
    authority = verify_authority()
    source = load_source(REPO)
    learned, learned_bindings = _g10_trajectories()
    er9, er9_bindings = _er9_trajectories()
    diagnostic = compute_h4_precision_diagnostic(learned, er9, snr_grid_db=EXPECTED_SNR_GRID)
    diagnostic.update({
        "source_commit": source["source_commit"], "source_manifest_id": source["manifest_id"],
        "authority_id": authority["authority_id"],
        "input_bindings": {"ordinary_learned_w8_g10": learned_bindings, "final_production_er9": er9_bindings},
    })
    diagnostic["diagnostic_id"] = "g11h4v4-" + canonical_sha256(diagnostic)
    h4_path = ROOT / "h4_pointwise_precision_v4.json"
    architecture = _architecture_audit()
    architecture_path = ROOT / "er9_architecture_difference_v4.json"
    immutable_write(h4_path, diagnostic)
    immutable_write(architecture_path, architecture)
    body = {
        "schema_version": 2, "artifact_role": "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT",
        "status": "G11_GREEN_NO_TEST_ACCESS", "decision": "GREEN",
        "source_commit": source["source_commit"], "source_manifest_id": source["manifest_id"],
        "authority_id": authority["authority_id"],
        "input_sources": {"learned": "ordinary_learned_w8_g10", "comparator": "final_production_er9", "randomized_er2": False},
        "cells": [list(cell) for cell in PRODUCTION_CELLS],
        "h4": {"artifact": str(h4_path.relative_to(REPO)), "artifact_sha256": sha256_file(h4_path)},
        "er9": {"production_closeout": str(ER9_CLOSEOUT.relative_to(REPO)), "production_closeout_sha256": sha256_file(ER9_CLOSEOUT), "architecture_difference": str(architecture_path.relative_to(REPO)), "architecture_difference_sha256": sha256_file(architecture_path), "validation_cells": er9_bindings},
        "am97_semantics": {"signed_correctness_difference": True, "within_image_three_cell_mean": True, "stable_image_complete_trajectory_bootstrap": True, "bootstrap_resamples": 10000, "quantiles": ["q50", "q95", "q99"], "reference_pp": 2, "pointwise_only": True, "full_h4_power_claim": False},  # literal-ok: AM-97 constants
        "protected_counters": {"production_er9_training": 3, "randomized_er2_training": 1, "g11": 1, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0},
        "test": "SEALED", "test_access": 0,
    }
    body["terminal_id"] = "g11terminal-" + canonical_sha256(body)
    immutable_write(ROOT / "g11_terminal_closeout.json", body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    value = run()
    print(f"G11 complete: {value['terminal_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
