#!/usr/bin/env python3
"""Aggregate and adjudicate W9-A/G-10 only after all 63 cells are complete."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.g10_crossover import AccuracyCount, MeasuredPoint, decide_g10  # noqa: E402
from evaluation.g10_protocol import (  # noqa: E402
    ADJUDICATION_PATH,
    AUTHORIZATION_PATH,
    CELL_INDEX_PATH,
    CLASSICAL_EXTRACT_PATH,
    COMPLETION_PATH,
    EXPECTED_CELL_COUNT,
    EXPECTED_GRID,
    G10ProtocolHold,
    HEADLINE_CURVE_PATH,
    RECONCILIATION_PATH,
    RUNTIME_MANIFEST_PATH,
    canonical_sha256,
    cell_key,
    load_json,
    rendered_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_authorization,
    verify_identified,
)
from evaluation.g10_runner import _verify_cell  # noqa: E402


def _publish_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise G10ProtocolHold(f"refusing to replace immutable G-10 closeout artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered_json(value))


def _fraction(value: Any) -> dict[str, Any]:
    return {"numerator": value.numerator, "denominator": value.denominator, "fraction": f"{value.numerator}/{value.denominator}"}


def _event(value: Any) -> dict[str, Any]:
    return {
        "direction": value.direction,
        "location_kind": value.location_kind,
        "first_snr_db": value.first_snr_db,
        "last_snr_db": value.last_snr_db,
    }


def _identified(body: dict[str, Any], field: str, prefix: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = prefix + canonical_sha256(value)
    value["artifact_content_sha256"] = canonical_sha256(value)
    return value


def closeout(*, runtime_root: Path, root: Path = REPO) -> dict[str, Any]:
    authorization = verify_authorization(root / AUTHORIZATION_PATH, root=root, allow_outcomes=False)
    runtime_manifest_path = runtime_root / "runtime_manifest.json"
    runtime_manifest, runtime_manifest_raw = load_json(runtime_manifest_path, "G-10 runtime manifest")
    require(runtime_manifest.get("status") == "COMPLETE_MATRIX_READY_FOR_AGGREGATION", "G-10 runtime matrix is not complete")
    require(runtime_manifest.get("authority_id") == authorization["authorization_id"], "G-10 runtime authority differs")
    require(runtime_manifest.get("matrix_shape") == {"checkpoints": 3, "snr_points": 21, "cells": EXPECTED_CELL_COUNT}, "G-10 runtime matrix shape differs")
    require(len(runtime_manifest.get("cells", [])) == EXPECTED_CELL_COUNT, "G-10 runtime cell count differs")
    classical, _ = load_json(root / CLASSICAL_EXTRACT_PATH, "G-10 classical extract")
    classical_by_snr = {int(row["snr_db"]): row for row in classical["points"]}
    checkpoint_rows = authorization["checkpoints"]
    cells: list[dict[str, Any]] = []
    measurements: dict[int, list[dict[str, Any]]] = {snr: [] for snr in EXPECTED_GRID}
    expected_keys = [cell_key(seed, seed, snr) for seed in range(3) for snr in EXPECTED_GRID]
    runtime_by_key = {row["cell_key"]: row for row in runtime_manifest["cells"]}
    require(tuple(runtime_by_key) == tuple(expected_keys), "G-10 runtime cell order/coverage differs")
    for index, key in enumerate(expected_keys):
        runtime_row = runtime_by_key[key]
        checkpoint = checkpoint_rows[index // len(EXPECTED_GRID)]
        snr = int(EXPECTED_GRID[index % len(EXPECTED_GRID)])
        cell_path = Path(runtime_row["file_path"])
        value, cell_raw = load_json(cell_path, f"G-10 runtime cell {key}")
        require(sha256_file(cell_path) == runtime_row["file_sha256"], f"G-10 runtime cell bytes differ: {key}")
        _verify_cell(value, expected_index=index, expected_checkpoint=checkpoint, expected_snr=snr)
        require(value["cell_key"] == key and value["artifact_id"] == runtime_row["artifact_id"], f"G-10 runtime cell binding differs: {key}")
        row = {
            "cell_index": index,
            "cell_key": key,
            "train_seed": value["train_seed"],
            "channel_seed": value["channel_seed"],
            "selected_epoch": value["selected_epoch"],
            "checkpoint_id": value["checkpoint_id"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "ratio": value["ratio"],
            "snr_db": value["snr_db"],
            "dataset": value["dataset"],
            "validation_split": value["validation_split"],
            "validation_manifest_sha256": value["validation_manifest_sha256"],
            "validation_denominator": value["validation_denominator"],
            "n_correct": value["n_correct"],
            "top1_accuracy": value["top1_accuracy"],
            "noise": value["noise"],
            "prediction_digest": value["prediction_digest"],
            "row_digest": value["row_digest"],
            "scientific_source_commit": value["scientific_source_commit"],
            "execution_checkout_commit": value["execution_checkout_commit"],
            "execution_profile": value["execution_profile"],
            "config_hash": value["config_hash"],
            "protocol_sha256": value["protocol_sha256"],
            "authority_id": value["authority_id"],
            "artifact_id": value["artifact_id"],
            "artifact_content_sha256": value["artifact_content_sha256"],
            "runtime_file_sha256": runtime_row["file_sha256"],
            "runtime_file_path": runtime_row["file_path"],
        }
        cells.append(row)
        measurements[snr].append(value)
    require(len(cells) == EXPECTED_CELL_COUNT, "G-10 compact index does not contain exactly 63 cells")
    # The three cells are grouped by SNR, not by a selected/best seed.
    points = []
    measured_points = []
    for snr in EXPECTED_GRID:
        rows = measurements[int(snr)]
        require(len(rows) == 3, f"G-10 SNR cell multiplicity differs at {snr} dB")
        classical_row = classical_by_snr[int(snr)]
        measured = MeasuredPoint(
            snr_db=int(snr),
            learned_cells=tuple(AccuracyCount(int(row["n_correct"]), int(row["validation_denominator"])) for row in rows),  # type: ignore[arg-type]
            classical_adaptive=AccuracyCount(int(classical_row["comparator_correct_count"]), int(classical_row["comparator_denominator"])),
        )
        measured_points.append(measured)
    decision = decide_g10(measured_points)
    for point in decision.points:
        classical_row = classical_by_snr[int(point.snr_db)]
        rows = measurements[int(point.snr_db)]
        points.append({
            "snr_db": point.snr_db,
            "learned_cells": [
                {"correct_count": int(row["n_correct"]), "denominator": int(row["validation_denominator"]), "top1_accuracy": row["top1_accuracy"], "cell_key": row["cell_key"], "artifact_id": row["artifact_id"]}
                for row in rows
            ],
            "learned_mean": _fraction(point.learned_mean),
            "learned_mean_accuracy": float(point.learned_mean),
            "learned_population_sd": point.learned_population_sd,
            "classical_adaptive": {
                "correct_count": classical_row["comparator_correct_count"],
                "denominator": classical_row["comparator_denominator"],
                "fraction": classical_row["comparator_fraction"],
                "accuracy": float(point.classical_adaptive_accuracy),
                "source_measurement_clean_correct_count": classical_row["clean_correct_count"],
                "source_measurement_clean_denominator": classical_row["clean_denominator"],
            },
            "gap": _fraction(point.gap),
            "sign": point.sign,
        })
    curve_body = {
        "schema_version": 1,
        "artifact_role": "G10_LEARNED_AGGREGATE_HEADLINE_CURVE",
        "status": "COMPLETE_21_POINT_EXACT_COUNT_CURVE",
        "authority_id": authorization["authorization_id"],
        "scientific_source_commit": authorization["scientific_source"]["commit"],
        "ratio": "r_1_6",
        "dataset": "imagenette160",
        "split": "val",
        "snr_parameter": "params.channel.test_snr_grid_db",
        "snr_grid_db": list(EXPECTED_GRID),
        "points": points,
        "aggregation": "arithmetic_mean_of_three_exact_correct_count_fractions",
        "population_sd_ddof": 0,
        "sd_descriptive_only": True,
        "cell_index_path": str(CELL_INDEX_PATH),
    }
    curve = _identified(curve_body, "curve_id", "g10curve-")
    index_body = {
        "schema_version": 1,
        "artifact_role": "G10_EXACT_LEARNED_3X21_CELL_INDEX",
        "status": "COMPLETE_EXACTLY_63_CELLS",
        "authority_id": authorization["authorization_id"],
        "scientific_source_commit": authorization["scientific_source"]["commit"],
        "matrix_shape": {"checkpoints": 3, "snr_points": 21, "cells": EXPECTED_CELL_COUNT},
        "cell_order": "train_seed_ascending_then_snr_grid_order",
        "snr_grid_db": list(EXPECTED_GRID),
        "ratio": "r_1_6",
        "dataset": "imagenette160",
        "split": "val",
        "cells": cells,
        "no_best_seed_selection": True,
        "no_r_1_24": True,
        "no_other_lambda": True,
        "external_runtime_manifest": {"path": str(runtime_manifest_path), "runtime_manifest_id": runtime_manifest["runtime_manifest_id"], "file_sha256": sha256_bytes(runtime_manifest_raw)},
    }
    index = _identified(index_body, "index_id", "g10index-")
    adjudication_body = {
        "schema_version": 1,
        "artifact_role": "G10_AM94_OBSERVABLE_CROSSOVER_ADJUDICATION",
        "status": "COMPLETE_VALIDATION_ONLY",
        "authority_id": authorization["authorization_id"],
        "scientific_source_commit": authorization["scientific_source"]["commit"],
        "am94_freeze_id": authorization["am94_semantics"]["freeze_id"],
        "am94_freeze_sha256": authorization["am94_semantics"]["freeze_sha256"],
        "predicate_source_path": authorization["am94_semantics"]["predicate_source_path"],
        "predicate_source_sha256": authorization["am94_semantics"]["predicate_source_sha256"],
        "dataset": "imagenette160",
        "split": "val",
        "ratio": "r_1_6",
        "snr_parameter": "params.channel.test_snr_grid_db",
        "snr_grid_db": list(EXPECTED_GRID),
        "matrix": {"checkpoints": 3, "snr_points": 21, "complete_learned_evaluations": EXPECTED_CELL_COUNT, "denominator_per_cell": 1000},
        "headline_comparator": "G8/F3 adaptive/oracle classical r_1_6 validation curve",
        "classical_extract_id": classical["extract_id"],
        "classical_extract_sha256": sha256_file(root / CLASSICAL_EXTRACT_PATH),
        "fixed_profile_classical_in_headline": False,
        "interpolation_used": False,
        "sign_sequence": [point.sign for point in decision.points],
        "points": points,
        "events": [_event(event) for event in decision.events],
        "headline_expected_event": None if decision.headline_expected_event is None else _event(decision.headline_expected_event),
        "multiple_ordering_reversals": decision.multiple_crossings,
        "event_count_class": decision.event_count_class,
        "classification": decision.classification,
        "zero_run_contacts": [
            {"snr_db": point.snr_db, "sign": point.sign}
            for point in decision.points
            if point.sign == 0
        ],
        "runtime_manifest_id": runtime_manifest["runtime_manifest_id"],
        "runtime_manifest_sha256": sha256_bytes(runtime_manifest_raw),
        "cell_index_id": index["index_id"],
        "curve_id": curve["curve_id"],
        "protected_counters": {"g10_model_facing_evaluations": EXPECTED_CELL_COUNT, "g10_outcomes_observed": EXPECTED_CELL_COUNT, "er9_training": 0, "er2_randomized_training": 0, "g11": 0, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0, "training": 0, "test": "SEALED"},
        "predicate_modifiers": {"tolerance": False, "epsilon": False, "displayed_rounding": False, "confidence_interval": False, "bootstrap": False, "seed_vote": False, "significance_test": False},
    }
    adjudication = _identified(adjudication_body, "adjudication_id", "g10adjudication-")
    completion_body = {
        "schema_version": 1,
        "artifact_role": "W9A_G10_TERMINAL_COMPLETION",
        "status": "W9A_G10_GREEN_TERMINAL_VALIDATION_ONLY",
        "authority": {"path": str(AUTHORIZATION_PATH), "authorization_id": authorization["authorization_id"], "file_sha256": sha256_file(root / AUTHORIZATION_PATH)},
        "source": {"commit": authorization["scientific_source"]["commit"], "manifest_path": authorization["scientific_source"]["manifest"]["path"], "manifest_id": authorization["scientific_source"]["manifest"]["manifest_id"]},
        "runtime_manifest": {"path": str(RUNTIME_MANIFEST_PATH), "runtime_manifest_id": runtime_manifest["runtime_manifest_id"]},
        "cell_index": {"path": str(CELL_INDEX_PATH), "index_id": index["index_id"]},
        "headline_curve": {"path": str(HEADLINE_CURVE_PATH), "curve_id": curve["curve_id"]},
        "adjudication": {"path": str(ADJUDICATION_PATH), "adjudication_id": adjudication["adjudication_id"]},
        "classification": decision.classification,
        "complete_learned_evaluations": EXPECTED_CELL_COUNT,
        "grid_db": list(EXPECTED_GRID),
        "headline_comparator": "G8/F3 adaptive/oracle classical r_1_6",
        "downstream": {"er9": 0, "randomized_er2_training": 0, "g11": 0, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0, "test": "SEALED"},
        "stop_boundary": "STOP_AFTER_W9A_G10_TERMINAL_RECONCILIATION",
    }
    completion = _identified(completion_body, "completion_id", "w9acompletion-")
    _publish_once(root / RUNTIME_MANIFEST_PATH, runtime_manifest)
    _publish_once(root / CELL_INDEX_PATH, index)
    _publish_once(root / HEADLINE_CURVE_PATH, curve)
    _publish_once(root / ADJUDICATION_PATH, adjudication)
    _publish_once(root / COMPLETION_PATH, completion)
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        completion = closeout(runtime_root=args.runtime_root)
    except (G10ProtocolHold, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"G-10 CLOSEOUT HOLD — {exc}", file=sys.stderr)
        return 1
    print(f"G-10 closeout PASS: {completion['completion_id']} classification={completion['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
