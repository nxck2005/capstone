#!/usr/bin/env python3
"""Offline fail-closed verification of the validation-only transparency probe."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from baseline.j2k import J2KCodec  # noqa: E402
from data.registry import load_dataset  # noqa: E402
from probes.transparency_bitrate import (  # noqa: E402
    AGGREGATE_FIELDS,
    PER_IMAGE_FIELDS,
    aggregate,
    design_fingerprint,
    load_design,
    parameter_snapshot,
    read_csv,
    selection_aware_bootstrap,
    sha256_path,
    threshold_forecast,
)

EVIDENCE = REPO / "results/probes/transparency_bitrate"
CONFIG = REPO / "configs/transparency-bitrate-probe.yaml"
_SUMMARY_FIELDS = {
    "schema_version",
    "probe_status",
    "prominent_declaration",
    "measurement_commit",
    "git_dirty_state",
    "dataset",
    "split",
    "dataset_identity",
    "archive_identity",
    "manifest_identity",
    "classifier_checkpoint_identity",
    "classifier_config_identity",
    "classifier_variant",
    "clean_validation",
    "codec_configuration",
    "codec_configuration_hash",
    "openjpeg_version",
    "glymur_version",
    "budget_grid",
    "encode_axis_order",
    "bootstrap",
    "bootstrap_resamples",
    "threshold_definitions",
    "point_estimate_best_axes",
    "probe_efficiency_threshold",
    "probe_crossover_threshold",
    "test_isolation_declaration",
    "cache_manifest_hash",
    "per_image_file_hash",
    "aggregate_file_hash",
    "resolved_config_hash",
    "commands_used",
    "completed_validation_cells",
    "codec_totals",
    "probe_wall_time_s",
    "cache_size_bytes",
    "provisional_bandwidth_parameters",
    "g8_status",
    "training_performed",
}
_RESOLVED_FIELDS = {
    "schema_version",
    "source",
    "design",
    "parameters",
    "design_hash",
    "measurement_commit",
    "measurement_git_dirty",
}
_CACHE_MANIFEST_FIELDS = {
    "schema_version",
    "cache_root",
    "entry_count",
    "entries",
}
_CACHE_ENTRY_FIELDS = {
    "cache_key",
    "cache_path",
    "cache_file_sha256",
    "codestream_sha256",
    "emitted_bytes",
    "feasible",
}
_DISCLAIMER = (
    "Validation-only engineering probe. "
    "Does not select or replace G-8 operating points."
)


class VerificationError(ValueError):
    """A closed evidence-verification failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    missing = expected - set(value)
    unexpected = set(value) - expected
    _require(
        not missing and not unexpected,
        f"{label} fields differ: missing={sorted(missing)}, "
        f"unexpected={sorted(unexpected)}",
    )
    return value


def _json(path: Path, fields: set[str] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid JSON {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} must contain an object")
    if fields is not None:
        _exact_fields(value, fields, path.name)
    return value


def _bool(value: str, field: str) -> bool:
    _require(value in {"true", "false"}, f"invalid boolean {field}={value!r}")
    return value == "true"


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path, PER_IMAGE_FIELDS)
    integer_fields = {
        "label",
        "budget_bytes",
        "encode_axis",
        "emitted_bytes",
        "search_iterations",
        "predicted_class",
        "clean_predicted_class",
    }
    float_fields = {"requested_bpp", "realized_bpp", "psnr", "ssim"}
    bool_fields = {"feasible", "decode_success", "correct", "clean_correct"}
    rows: list[dict[str, Any]] = []
    for raw_row in raw:
        row: dict[str, Any] = {}
        for field, value in raw_row.items():
            if value == "":
                row[field] = None
            elif field in integer_fields:
                try:
                    row[field] = int(value)
                except ValueError:
                    raise VerificationError(f"invalid integer {field}") from None
            elif field in float_fields:
                try:
                    row[field] = float(value)
                except ValueError:
                    raise VerificationError(f"invalid float {field}") from None
            elif field in bool_fields:
                row[field] = _bool(value, field)
            else:
                row[field] = value
        rows.append(row)
    return rows


def _manifest_validation_identity() -> dict[str, int]:
    dataset = load_dataset("imagenette160", "val", REPO)
    return {
        dataset.source_sample(index).stable_sample_id: dataset.source_sample(index).label
        for index in range(len(dataset))
    }


def _verify_rows(rows: list[dict[str, Any]], design: dict[str, Any]) -> None:
    expected_ids = _manifest_validation_identity()
    budgets = {
        int(item["budget_bytes"]): item for item in design["budget_grid"]
    }
    axes = [int(axis) for axis in design["encode_axes_px"]]
    expected_cells = len(expected_ids) * len(budgets) * len(axes)
    _require(len(rows) == expected_cells, "missing validation cells")
    keys: set[tuple[str, int, int]] = set()
    seen_ids: set[str] = set()
    clean_by_id: dict[str, tuple[int, bool]] = {}
    pixel_count = math.prod(
        int(value) for value in get("datasets.imagenette160.image_size")[:2]
    )
    maximum_iterations = int(get("baseline.j2k_search_max_iters"))
    for row in rows:
        stable_id = row["stable_sample_id"]
        _require(stable_id in expected_ids, "stable ID is outside validation manifest")
        seen_ids.add(stable_id)
        _require(row["label"] == expected_ids[stable_id], "row label disagrees")
        budget = row["budget_bytes"]
        axis = row["encode_axis"]
        _require(budget in budgets and axis in axes, "unexpected budget or axis cell")
        key = (stable_id, budget, axis)
        _require(key not in keys, "duplicate budget/axis/sample cell")
        keys.add(key)
        budget_item = budgets[budget]
        _require(
            row["budget_source"] == "|".join(budget_item["sources"]),
            "budget source disagrees",
        )
        _require(
            row["requested_bpp"] == float(budget_item["requested_bpp"]),
            "requested bpp disagrees",
        )
        clean_value = (row["clean_predicted_class"], row["clean_correct"])
        if stable_id in clean_by_id:
            _require(
                clean_by_id[stable_id] == clean_value,
                "clean result changes across stable-ID trajectory",
            )
        clean_by_id[stable_id] = clean_value
        _require(
            isinstance(row["clean_predicted_class"], int)
            and 0 <= row["clean_predicted_class"] < int(get("datasets.imagenette160.classes")),
            "invalid clean prediction",
        )
        _require(
            row["clean_correct"]
            == (row["clean_predicted_class"] == row["label"]),
            "clean correctness disagrees",
        )
        _require(
            isinstance(row["search_iterations"], int)
            and 1 <= row["search_iterations"] <= maximum_iterations,
            "invalid search-iteration count",
        )
        _require(
            isinstance(row["cache_key"], str)
            and re.fullmatch(r"[0-9a-f]{64}", row["cache_key"]) is not None,
            "invalid cache key",
        )
        if row["feasible"]:
            _require(row["decode_success"], "invalid feasible/decode status combination")
            _require(
                isinstance(row["emitted_bytes"], int)
                and 0 < row["emitted_bytes"] <= budget,
                "emitted bytes exceed budget",
            )
            _require(
                isinstance(row["codestream_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", row["codestream_sha256"])
                is not None,
                "invalid codestream identity",
            )
            expected_bpp = (
                row["emitted_bytes"] * 8 / pixel_count  # literal-ok: bits per byte
            )
            _require(row["realized_bpp"] == expected_bpp, "incorrect realized bpp")
            _require(
                isinstance(row["predicted_class"], int)
                and 0 <= row["predicted_class"] < int(get("datasets.imagenette160.classes")),
                "invalid codec prediction",
            )
            _require(
                row["correct"] == (row["predicted_class"] == row["label"]),
                "codec correctness disagrees",
            )
            _require(
                isinstance(row["psnr"], float)
                and math.isfinite(row["psnr"])
                and isinstance(row["ssim"], float)
                and math.isfinite(row["ssim"])
                and -1 <= row["ssim"] <= 1,
                "invalid PSNR/SSIM value",
            )
        else:
            _require(
                not row["decode_success"]
                and row["codestream_sha256"] is None
                and row["emitted_bytes"] is None
                and row["realized_bpp"] is None
                and row["predicted_class"] is None
                and row["correct"] is False
                and row["psnr"] is None
                and row["ssim"] is None,
                "invalid infeasible/decode status combination",
            )
    _require(seen_ids == set(expected_ids), "missing or duplicate stable IDs")
    _require(len(keys) == expected_cells, "missing budget/axis/sample cells")
    _require(
        sum(value[1] for value in clean_by_id.values()) == 898,
        "clean result is not 898/1000",
    )


def _parse_aggregate(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path, AGGREGATE_FIELDS)
    integer_fields = {
        "budget_bytes",
        "encode_axis",
        "n_correct",
        "n_total",
        "maximum_emitted_bytes",
        "infeasible_count",
        "decode_failure_count",
    }
    string_fields = {"budget_source"}
    rows: list[dict[str, Any]] = []
    for raw_row in raw:
        row: dict[str, Any] = {}
        for field, value in raw_row.items():
            if value == "":
                row[field] = None
            elif field in integer_fields:
                row[field] = int(value)
            elif field in string_fields:
                row[field] = value
            elif field == "selected_point_estimate":
                row[field] = _bool(value, field)
            else:
                row[field] = float(value)
        rows.append(row)
    return rows


def _verify_aggregates(
    rows: list[dict[str, Any]],
    aggregate_path: Path,
    design: dict[str, Any],
) -> dict[int, int]:
    expected, selected = aggregate(rows, design)
    actual = _parse_aggregate(aggregate_path)
    _require(len(actual) == len(expected), "aggregate cell count differs")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        _require(set(left) == set(right), f"aggregate row {index} schema differs")
        for field in left:
            _require(
                left[field] == right[field],
                f"incorrect aggregate {field} at row {index}",
            )
    return selected


def _verify_cache_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> None:
    _exact_fields(manifest, _CACHE_MANIFEST_FIELDS, "cache_manifest")
    _require(manifest["schema_version"] == 1, "cache manifest schema differs")
    _require(manifest["cache_root"] == design["cache_root"], "cache root differs")
    _require(
        manifest["entry_count"] == len(rows) == len(manifest["entries"]),
        "cache manifest entry count differs",
    )
    by_key = {str(row["cache_key"]): row for row in rows}
    seen: set[str] = set()
    for entry in manifest["entries"]:
        _exact_fields(entry, _CACHE_ENTRY_FIELDS, "cache entry")
        key = entry["cache_key"]
        _require(key in by_key and key not in seen, "cache manifest key differs")
        seen.add(key)
        row = by_key[key]
        _require(
            entry["cache_path"] == f"{design['cache_root']}/{key}.j2kcache",
            "cache manifest path differs",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", entry["cache_file_sha256"]) is not None,
            "cache file hash is invalid",
        )
        _require(
            entry["codestream_sha256"] == row["codestream_sha256"]
            and entry["emitted_bytes"] == row["emitted_bytes"]
            and entry["feasible"] is row["feasible"],
            "cache manifest cell identity differs",
        )
    _require(seen == set(by_key), "cache manifest misses cell entries")


def _verify_no_tracked_cache_or_codestream(design: dict[str, Any]) -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(result.returncode == 0, "cannot enumerate tracked files")
    tracked = result.stdout.splitlines()
    forbidden_suffixes = (".j2k", ".j2c", ".jp2", ".j2kcache")
    _require(
        all(
            not path.startswith(f"{design['cache_root']}/")
            and not path.lower().endswith(forbidden_suffixes)
            for path in tracked
        ),
        "a cache or codestream is committed",
    )


def verify(
    *,
    evidence_dir: Path = EVIDENCE,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    design = load_design(config_path, repo_root=REPO)
    summary_path = evidence_dir / "summary.json"
    resolved_path = evidence_dir / "resolved_config.json"
    per_image_path = evidence_dir / "per_image.csv"
    aggregate_path = evidence_dir / "aggregate.csv"
    cache_manifest_path = evidence_dir / "cache_manifest.json"
    summary = _json(summary_path, _SUMMARY_FIELDS)
    resolved = _json(resolved_path, _RESOLVED_FIELDS)
    cache_manifest = _json(cache_manifest_path, _CACHE_MANIFEST_FIELDS)

    _require(summary["schema_version"] == 1, "summary schema version differs")
    _require(summary["probe_status"] == "COMPLETE", "probe is not complete")
    _require(summary["prominent_declaration"] == _DISCLAIMER, "G-8 disclaimer differs")
    _require(
        summary["measurement_commit"] == resolved["measurement_commit"],
        "wrong measurement commit",
    )
    _require(
        summary["git_dirty_state"] is False
        and resolved["measurement_git_dirty"] is False,
        "measurement state was dirty",
    )
    _require(
        summary["dataset"] == "imagenette160"
        and summary["split"] == "validation",
        "wrong dataset or non-validation split",
    )
    expected_archive = get("datasets.imagenette160.archive_sha256")
    expected_manifest = get("datasets.imagenette160.manifest_sha256")
    _require(
        summary["dataset_identity"]
        == summary["archive_identity"]
        == expected_archive,
        "wrong dataset/archive identity",
    )
    _require(summary["manifest_identity"] == expected_manifest, "wrong manifest identity")
    _require(
        summary["classifier_checkpoint_identity"]
        == design["classifier"]["checkpoint_sha256"],
        "wrong classifier checkpoint identity",
    )
    _require(
        summary["classifier_config_identity"]
        == design["classifier"]["config_hash"],
        "wrong classifier config identity",
    )
    _require(
        summary["classifier_variant"] == "clean",
        "wrong classifier variant",
    )
    _require(
        summary["clean_validation"]
        == {"n_correct": 898, "n_total": 1000, "top1_accuracy": 0.898},
        "clean result is not 898/1000",
    )
    isolation = summary["test_isolation_declaration"]
    _require(
        isolation
        == {
            "test_split_sealed": True,
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
        },
        "summary contains a test-access claim",
    )
    _require(summary["training_performed"] is False, "summary claims model training")
    _require(
        summary["g8_status"] == "unresolved",
        "summary claims G-8 selection",
    )
    _require(
        summary["budget_grid"] == design["budget_grid"]
        and summary["encode_axis_order"] == design["encode_axes_px"],
        "summary grid or axis order differs",
    )
    codec = J2KCodec(REPO / design["cache_root"])
    _require(
        summary["codec_configuration"] == codec.snapshot
        and summary["codec_configuration_hash"] == codec.configuration_hash,
        "codec configuration or hash differs",
    )
    _require(
        summary["openjpeg_version"] == get("environment.openjpeg")
        and summary["glymur_version"] == get("environment.glymur"),
        "JPEG 2000 runtime versions differ",
    )
    _require(
        summary["bootstrap_resamples"] == design["bootstrap"]["resamples"],
        "wrong bootstrap resample count",
    )
    _require(
        summary["threshold_definitions"] == design["thresholds"],
        "threshold definitions differ",
    )
    expected_provisional = {
        "crossover_ratio": "r_1_3",
        "crossover_ratio_status": "provisional_until_G-8",
        "efficiency_ratio": "r_1_6",
        "efficiency_ratio_status": "provisional_until_G-8",
        "low_ratio_operating_point": "r_1_12",
        "low_ratio_operating_point_status": "provisional_until_G-8",
    }
    _require(
        summary["provisional_bandwidth_parameters"] == expected_provisional,
        "a provisional bandwidth parameter changed",
    )
    for key, expected in expected_provisional.items():
        _require(get(f"bandwidth.{key}") == expected, "current provisional bandwidth parameters changed")
    _require(
        resolved["schema_version"] == 1
        and resolved["source"] == "configs/transparency-bitrate-probe.yaml"
        and resolved["design"] == design
        and resolved["parameters"] == parameter_snapshot()
        and resolved["design_hash"] == design_fingerprint(design),
        "resolved probe configuration differs",
    )
    _require(
        summary["per_image_file_hash"] == sha256_path(per_image_path)
        and summary["aggregate_file_hash"] == sha256_path(aggregate_path)
        and summary["cache_manifest_hash"] == sha256_path(cache_manifest_path)
        and summary["resolved_config_hash"] == sha256_path(resolved_path),
        "evidence file hash disagreement",
    )

    rows = _parse_rows(per_image_path)
    _verify_rows(rows, design)
    selected = _verify_aggregates(rows, aggregate_path, design)
    _require(
        summary["point_estimate_best_axes"]
        == [
            {"budget_bytes": budget, "encode_axis": selected[budget]}
            for budget in (
                int(item["budget_bytes"]) for item in design["budget_grid"]
            )
        ],
        "incorrect best-axis selection or tie break",
    )
    bootstrap = selection_aware_bootstrap(rows, design)
    _require(summary["bootstrap"] == bootstrap, "bootstrap result cannot be reproduced")
    efficiency = threshold_forecast(
        bootstrap,
        key="meets_5pp",
        label="probe_efficiency_threshold",
    )
    crossover = threshold_forecast(
        bootstrap,
        key="meets_2pp",
        label="probe_crossover_threshold",
    )
    _require(
        summary["probe_efficiency_threshold"] == efficiency,
        "incorrect 5 pp threshold forecast",
    )
    _require(
        summary["probe_crossover_threshold"] == crossover,
        "incorrect 2 pp threshold forecast",
    )
    _require(
        summary["completed_validation_cells"] == len(rows),
        "completed validation cell count differs",
    )
    _require(
        isinstance(summary["probe_wall_time_s"], int | float)
        and not isinstance(summary["probe_wall_time_s"], bool)
        and math.isfinite(summary["probe_wall_time_s"])
        and summary["probe_wall_time_s"] > 0
        and isinstance(summary["cache_size_bytes"], int)
        and summary["cache_size_bytes"] > 0,
        "probe wall time or cache size is invalid",
    )
    _require(
        summary["codec_totals"]
        == {
            "infeasible_count": sum(not row["feasible"] for row in rows),
            "decode_failure_count": sum(
                row["feasible"] and not row["decode_success"] for row in rows
            ),
        },
        "codec failure totals differ",
    )
    _verify_cache_manifest(cache_manifest, rows, design)
    _verify_no_tracked_cache_or_codestream(design)
    return {
        "status": "COMPLETE",
        "measurement_commit": summary["measurement_commit"],
        "cells": len(rows),
        "efficiency_budget": efficiency["result"]["budget_bytes"],
        "crossover_budget": crossover["result"]["budget_bytes"],
    }


def main() -> int:
    try:
        result = verify()
    except (OSError, VerificationError, ValueError) as exc:
        print(f"Transparency probe verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Transparency probe verification PASS: "
        f"commit={result['measurement_commit'][:12]}, "
        f"cells={result['cells']}, "
        f"5pp={result['efficiency_budget']} B, "
        f"2pp={result['crossover_budget']} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
