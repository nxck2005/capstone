#!/usr/bin/env python3
"""Fresh-process verifier for the terminal W9-A/G-10 evidence."""

from __future__ import annotations

import subprocess
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
    OUTCOME_FILES,
    RECONCILIATION_PATH,
    RUNTIME_MANIFEST_PATH,
    canonical_sha256,
    cell_key,
    expected_cell_keys,
    load_json,
    sha256_file,
    verify_authorization,
    verify_identified,
    verify_source_manifest,
)
from config.params import get  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G10ProtocolHold(message)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO,
        check=False,
    ).returncode == 0


def _fraction(value: dict[str, Any]) -> tuple[int, int]:
    return int(value["numerator"]), int(value["denominator"])


def verify(root: Path = REPO) -> dict[str, Any]:
    authorization = verify_authorization(root / AUTHORIZATION_PATH, root=root, allow_outcomes=True)
    source_manifest, _ = load_json(root / "results/learned/w9/g10_source_manifest.json", "G-10 source manifest")
    verify_source_manifest(source_manifest, root)
    source_commit = authorization["scientific_source"]["commit"]
    _require(_is_ancestor(source_commit, _git_head()), "G-10 source epoch is not an ancestor of terminal evidence")
    current_files = frozenset(path.relative_to(root).as_posix() for path in (root / "results/learned/w9").glob("**/*") if path.is_file())
    allowed = {
        "results/learned/w9/am94_pre_science_freeze.json",
        str(AUTHORIZATION_PATH),
        "results/learned/w9/g10_source_manifest.json",
        str(CLASSICAL_EXTRACT_PATH),
        *OUTCOME_FILES,
    }
    _require(current_files == allowed, f"terminal W9 artifact set differs: unexpected={sorted(current_files - allowed)} missing={sorted(allowed - current_files)}")
    classical, _ = load_json(root / CLASSICAL_EXTRACT_PATH, "G-10 classical extract")
    classical_by_snr = {int(row["snr_db"]): row for row in classical["points"]}
    runtime, runtime_raw = load_json(root / RUNTIME_MANIFEST_PATH, "committed G-10 runtime manifest")
    verify_identified(runtime, field="runtime_manifest_id", prefix="g10runtime-", label="G-10 runtime manifest")
    _require(runtime["status"] == "COMPLETE_MATRIX_READY_FOR_AGGREGATION" and runtime["matrix_shape"] == {"checkpoints": 3, "snr_points": 21, "cells": EXPECTED_CELL_COUNT}, "G-10 runtime manifest differs")
    _require(runtime["authority_id"] == authorization["authorization_id"] and runtime["source_commit"] == source_commit, "G-10 runtime authority/source differs")
    index, _ = load_json(root / CELL_INDEX_PATH, "G-10 cell index")
    verify_identified(index, field="index_id", prefix="g10index-", label="G-10 cell index")
    _require(index["matrix_shape"] == {"checkpoints": 3, "snr_points": 21, "cells": EXPECTED_CELL_COUNT}, "G-10 cell index shape differs")
    _require(index["snr_grid_db"] == list(EXPECTED_GRID) and tuple(index["snr_grid_db"]) == tuple(get("channel.test_snr_grid_db")), "G-10 grid is not exactly params.channel.test_snr_grid_db")
    _require(index["ratio"] == "r_1_6" and index["no_best_seed_selection"] is True and index["no_r_1_24"] is True and index["no_other_lambda"] is True, "G-10 index selection boundary differs")
    cells = index["cells"]
    _require(len(cells) == EXPECTED_CELL_COUNT, "G-10 index is not exactly 63 cells")
    keys = [cell["cell_key"] for cell in cells]
    _require(tuple(keys) == expected_cell_keys() and len(set(keys)) == EXPECTED_CELL_COUNT, "G-10 cell keys are not the exact 3x21 matrix")
    for index_number, cell in enumerate(cells):
        expected_seed = index_number // len(EXPECTED_GRID)
        expected_snr = int(EXPECTED_GRID[index_number % len(EXPECTED_GRID)])
        _require(cell["cell_key"] == cell_key(expected_seed, expected_seed, expected_snr), f"G-10 cell key differs at {index_number}")
        _require(cell["train_seed"] == expected_seed and cell["channel_seed"] == expected_seed and cell["snr_db"] == expected_snr, f"G-10 cell coordinates differ at {index_number}")
        _require(cell["ratio"] == "r_1_6" and cell["validation_denominator"] == 1000 and 0 <= cell["n_correct"] <= 1000, f"G-10 cell denominator/scope differs at {index_number}")
        _require(cell["top1_accuracy"] == cell["n_correct"] / 1000, f"G-10 cell top-1 is not count-derived at {index_number}")
        _require(cell["scientific_source_commit"] == source_commit and cell["authority_id"] == authorization["authorization_id"], f"G-10 cell source/authority differs at {index_number}")
        _require(cell["execution_profile"]["execution_profile_id"] == "confessor_pascal_cu126", f"G-10 cell profile differs at {index_number}")
    curve, _ = load_json(root / HEADLINE_CURVE_PATH, "G-10 headline curve")
    verify_identified(curve, field="curve_id", prefix="g10curve-", label="G-10 headline curve")
    _require(curve["snr_grid_db"] == list(EXPECTED_GRID) and len(curve["points"]) == 21 and curve["ratio"] == "r_1_6", "G-10 headline curve scope differs")
    measured_points = []
    for snr in EXPECTED_GRID:
        seed_cells = [cell for cell in cells if cell["snr_db"] == snr]
        _require(len(seed_cells) == 3, f"G-10 curve seed count differs at {snr} dB")
        classical_row = classical_by_snr[int(snr)]
        measured_points.append(MeasuredPoint(
            snr_db=snr,
            learned_cells=tuple(AccuracyCount(int(cell["n_correct"]), int(cell["validation_denominator"])) for cell in seed_cells),  # type: ignore[arg-type]
            classical_adaptive=AccuracyCount(int(classical_row["comparator_correct_count"]), int(classical_row["comparator_denominator"])),
        ))
    decision = decide_g10(measured_points)
    for point, stored in zip(decision.points, curve["points"], strict=True):
        _require(stored["snr_db"] == point.snr_db and stored["sign"] == point.sign, f"G-10 curve sign differs at {point.snr_db} dB")
        _require(_fraction(stored["learned_mean"]) == (point.learned_mean.numerator, point.learned_mean.denominator), f"G-10 learned mean differs at {point.snr_db} dB")
        _require(_fraction(stored["gap"]) == (point.gap.numerator, point.gap.denominator), f"G-10 exact gap differs at {point.snr_db} dB")
        _require(stored["learned_population_sd"] == point.learned_population_sd, f"G-10 population SD differs at {point.snr_db} dB")
        _require(stored["classical_adaptive"]["correct_count"] == classical_by_snr[int(point.snr_db)]["comparator_correct_count"] and stored["classical_adaptive"]["denominator"] == classical_by_snr[int(point.snr_db)]["comparator_denominator"], f"G-10 classical comparator differs at {point.snr_db} dB")
    adjudication, _ = load_json(root / ADJUDICATION_PATH, "G-10 adjudication")
    verify_identified(adjudication, field="adjudication_id", prefix="g10adjudication-", label="G-10 adjudication")
    _require(adjudication["classification"] == decision.classification and adjudication["sign_sequence"] == [point.sign for point in decision.points], "G-10 AM-94 adjudication differs")
    _require(adjudication["events"] == [{"direction": event.direction, "location_kind": event.location_kind, "first_snr_db": event.first_snr_db, "last_snr_db": event.last_snr_db} for event in decision.events], "G-10 event list differs")
    _require(adjudication["headline_comparator"] == "G8/F3 adaptive/oracle classical r_1_6 validation curve" and adjudication["fixed_profile_classical_in_headline"] is False, "G-10 headline comparator differs")
    _require(adjudication["protected_counters"] == {"g10_model_facing_evaluations": 63, "g10_outcomes_observed": 63, "er9_training": 0, "er2_randomized_training": 0, "g11": 0, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0, "training": 0, "test": "SEALED"}, "G-10 protected counters differ")
    completion, _ = load_json(root / COMPLETION_PATH, "W9-A completion")
    verify_identified(completion, field="completion_id", prefix="w9acompletion-", label="W9-A completion")
    _require(completion["status"] == "W9A_G10_GREEN_TERMINAL_VALIDATION_ONLY" and completion["complete_learned_evaluations"] == 63 and completion["classification"] == decision.classification, "W9-A completion differs")
    _require(completion["downstream"] == {"er9": 0, "randomized_er2_training": 0, "g11": 0, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0, "test": "SEALED"}, "W9-A downstream boundary differs")
    reconciliation_path = root / RECONCILIATION_PATH
    if reconciliation_path.exists():
        reconciliation, _ = load_json(reconciliation_path, "W9-A reconciliation")
        verify_identified(reconciliation, field="reconciliation_id", prefix="w9areconcile-", label="W9-A reconciliation")
        _require(reconciliation["status"] == "W9A_G10_RECONCILED_GREEN_STOP" and reconciliation["completion_id"] == completion["completion_id"], "W9-A reconciliation differs")
    return {"authorization": authorization, "runtime": runtime, "index": index, "curve": curve, "adjudication": adjudication, "completion": completion, "decision": decision, "runtime_manifest_sha256": sha256_bytes(runtime_raw)}


def main() -> int:
    try:
        value = verify()
    except (G10ProtocolHold, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"G-10 TERMINAL VERIFY HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "G-10 terminal verification PASS: "
        f"{value['completion']['completion_id']} "
        f"classification={value['adjudication']['classification']} "
        f"cells={len(value['index']['cells'])} test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
