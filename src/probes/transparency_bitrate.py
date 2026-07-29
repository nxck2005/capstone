"""Frozen design, deterministic aggregation, and paired analysis for the probe."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config.params import REPO_ROOT, get

DIRECT_BPP_POINTS = (
    "0.25",
    "0.50",
    "0.75",
    "1.00",
    "1.25",
    "1.50",
    "1.67",
    "2.00",
    "2.50",
    "3.00",
    "3.33",
    "4.00",
)
PER_IMAGE_FIELDS = (
    "stable_sample_id",
    "label",
    "budget_source",
    "budget_bytes",
    "requested_bpp",
    "encode_axis",
    "cache_key",
    "codestream_sha256",
    "emitted_bytes",
    "realized_bpp",
    "search_iterations",
    "feasible",
    "decode_success",
    "predicted_class",
    "correct",
    "clean_predicted_class",
    "clean_correct",
    "psnr",
    "ssim",
)
AGGREGATE_FIELDS = (
    "budget_bytes",
    "requested_bpp",
    "budget_source",
    "encode_axis",
    "n_correct",
    "n_total",
    "top1_accuracy",
    "accuracy_difference_from_clean",
    "mean_emitted_bytes",
    "median_emitted_bytes",
    "maximum_emitted_bytes",
    "mean_realized_bpp",
    "infeasible_count",
    "decode_failure_count",
    "mean_psnr",
    "mean_ssim",
    "selected_point_estimate",
)
_GIT_SHA_HEX_LENGTH = hashlib.sha1().digest_size * 2


class ProbeDesignError(ValueError):
    """Frozen-design or deterministic-analysis failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),  # literal-ok: one-MiB evidence hash chunk
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _packet_budgets(repo_root: Path, selector: dict[str, Any]) -> dict[int, str]:
    path = repo_root / "spec/evidence/packetisation_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in record["configurations"]
        if row["dataset"] == selector["dataset"]
        and row["modulation"] == selector["modulation"]
        and row["nominal_rate_str"] == str(selector["nominal_rate"])
    ]
    ratios = get("bandwidth.ratios")
    if not isinstance(ratios, dict):
        raise ProbeDesignError("bandwidth ratio ladder is invalid")
    if {row["ratio"] for row in rows} != set(ratios):
        raise ProbeDesignError("packetisation budget selector misses a ratio rung")
    return {
        int(row["source_bytes"]): (
            f"packetisation:{selector['modulation']}:"
            f"{selector['nominal_rate']}:{row['ratio']}"
        )
        for row in rows
    }


def _expected_budget_grid(
    repo_root: Path, selector: dict[str, Any]
) -> list[dict[str, Any]]:
    image_size = get("datasets.imagenette160.image_size")
    pixels = int(image_size[0]) * int(image_size[1])
    by_budget: dict[int, list[str]] = defaultdict(list)
    for budget, label in _packet_budgets(repo_root, selector).items():
        by_budget[budget].append(label)
    for point in DIRECT_BPP_POINTS:
        byte_value = Decimal(point) * Decimal(pixels) / Decimal(
            np.iinfo(np.uint8).bits
        )
        if byte_value != byte_value.to_integral_value():
            raise ProbeDesignError(f"direct bpp point {point} is not byte integral")
        by_budget[int(byte_value)].append(f"direct_bpp:{point}")
    return [
        {
            "budget_bytes": budget,
            "requested_bpp": float(
                Decimal(budget)
                * Decimal(np.iinfo(np.uint8).bits)
                / Decimal(pixels)
            ),
            "sources": sources,
        }
        for budget, sources in sorted(by_budget.items())
    ]


def load_design(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    require_implementation_commit: bool = True,
) -> dict[str, Any]:
    try:
        design = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProbeDesignError(f"cannot read frozen probe design: {exc}") from None
    required = {
        "experiment",
        "dataset",
        "split",
        "classifier_variant",
        "classifier",
        "codec",
        "encode_axes_px",
        "classifier_batch_size",
        "bootstrap",
        "tie_breaking",
        "pilot",
        "packetisation_budget_selector",
        "budget_grid",
        "thresholds",
        "outputs",
        "cache_root",
        "shard_root",
        "implementation_commit",
    }
    if not isinstance(design, dict) or set(design) != required:
        raise ProbeDesignError("probe design top-level schema differs")
    expected_scalars = {
        "experiment": "transparency_bitrate_probe",
        "dataset": "imagenette160",
        "split": "validation",
        "classifier_variant": "clean",
        "codec": "jpeg2000",
        "tie_breaking": "highest_integer_n_correct_then_first_configured_axis",
    }
    for key, expected in expected_scalars.items():
        if design[key] != expected:
            raise ProbeDesignError(f"probe design {key} disagrees")
    axes = get("baseline.downsample_axis_px.imagenette160")
    if design["encode_axes_px"] != axes:
        raise ProbeDesignError("probe encode-axis order differs from parameters")
    expected_grid = _expected_budget_grid(
        repo_root, design["packetisation_budget_selector"]
    )
    if len(design["budget_grid"]) != len(expected_grid):
        raise ProbeDesignError("probe budget-grid length differs")
    for actual, expected in zip(design["budget_grid"], expected_grid, strict=True):
        if set(actual) != {"budget_bytes", "requested_bpp", "sources"}:
            raise ProbeDesignError("probe budget-grid entry schema differs")
        if (
            actual["budget_bytes"] != expected["budget_bytes"]
            or actual["sources"] != expected["sources"]
            or float(actual["requested_bpp"]) != expected["requested_bpp"]
        ):
            raise ProbeDesignError(
                f"probe budget grid differs at {actual.get('budget_bytes')}"
            )
    classifier = design["classifier"]
    if classifier != {
        "checkpoint_sha256": (
            "9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002"
        ),
        "config_hash": (
            "a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1"
        ),
    }:
        raise ProbeDesignError("probe classifier identity differs")
    bootstrap = design["bootstrap"]
    if (
        bootstrap.get("method")
        != "stable_id_trajectory_selection_aware_paired_bootstrap"
        or bootstrap.get("resamples") != get("evaluation.bootstrap_resamples")
        or bootstrap.get("quantile_method") != "lower"
        or bootstrap.get("one_sided_confidence") != 0.95
    ):
        raise ProbeDesignError("probe bootstrap design differs")
    if design["thresholds"] != {
        "probe_efficiency_threshold_max_drop_pp": get(
            "bandwidth.efficiency_ratio_threshold_pp"
        ),
        "probe_crossover_threshold_max_drop_pp": get(
            "bandwidth.crossover_ratio_threshold_pp"
        ),
    }:
        raise ProbeDesignError("probe threshold definitions differ")
    commit = design["implementation_commit"]
    if require_implementation_commit and (
        not isinstance(commit, str)
        or len(commit) != _GIT_SHA_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ProbeDesignError("probe implementation commit is not frozen")
    return design


def parameter_snapshot() -> dict[str, Any]:
    return {
        root: get(root)
        for root in (
            "project",
            "datasets",
            "preprocessing",
            "bandwidth",
            "baseline",
            "reference_classifier",
            "evaluation",
            "environment",
        )
    }


def design_fingerprint(design: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema_version": 1,
                "design": design,
                "parameters": parameter_snapshot(),
            }
        )
    ).hexdigest()


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return format(float(value), ".17g")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ProbeDesignError("CSV row schema differs")
            writer.writerow(row)


def read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ProbeDesignError(f"{path} CSV schema differs")
        return list(reader)


def best_axis(
    rows_by_axis: dict[int, list[dict[str, Any]]], axis_order: list[int]
) -> int:
    counts = {
        axis: sum(bool(row["correct"]) for row in rows_by_axis[axis])
        for axis in axis_order
    }
    return max(axis_order, key=lambda axis: (counts[axis], -axis_order.index(axis)))


def aggregate(
    rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    axes = [int(axis) for axis in design["encode_axes_px"]]
    clean_by_id: dict[str, bool] = {}
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stable_id = str(row["stable_sample_id"])
        clean = bool(row["clean_correct"])
        if stable_id in clean_by_id and clean_by_id[stable_id] != clean:
            raise ProbeDesignError("clean outcome changes across trajectory")
        clean_by_id[stable_id] = clean
        groups[(int(row["budget_bytes"]), int(row["encode_axis"]))].append(row)
    clean_accuracy = sum(clean_by_id.values()) / len(clean_by_id)
    selected: dict[int, int] = {}
    for budget in (int(item["budget_bytes"]) for item in design["budget_grid"]):
        rows_by_axis = {axis: groups[(budget, axis)] for axis in axes}
        selected[budget] = best_axis(rows_by_axis, axes)

    output: list[dict[str, Any]] = []
    for budget_item in design["budget_grid"]:
        budget = int(budget_item["budget_bytes"])
        for axis in axes:
            group = groups[(budget, axis)]
            total = len(group)
            correct = sum(bool(row["correct"]) for row in group)
            feasible = [row for row in group if row["feasible"]]
            emitted = [int(row["emitted_bytes"]) for row in feasible]
            realised = [float(row["realized_bpp"]) for row in feasible]
            psnr = [float(row["psnr"]) for row in feasible]
            ssim = [float(row["ssim"]) for row in feasible]
            output.append(
                {
                    "budget_bytes": budget,
                    "requested_bpp": float(budget_item["requested_bpp"]),
                    "budget_source": "|".join(budget_item["sources"]),
                    "encode_axis": axis,
                    "n_correct": correct,
                    "n_total": total,
                    "top1_accuracy": correct / total,
                    "accuracy_difference_from_clean": correct / total
                    - clean_accuracy,
                    "mean_emitted_bytes": float(np.mean(emitted)) if emitted else None,
                    "median_emitted_bytes": (
                        float(np.median(emitted)) if emitted else None
                    ),
                    "maximum_emitted_bytes": max(emitted) if emitted else None,
                    "mean_realized_bpp": (
                        float(np.mean(realised)) if realised else None
                    ),
                    "infeasible_count": total - len(feasible),
                    "decode_failure_count": sum(
                        bool(row["feasible"]) and not bool(row["decode_success"])
                        for row in group
                    ),
                    "mean_psnr": float(np.mean(psnr)) if psnr else None,
                    "mean_ssim": float(np.mean(ssim)) if ssim else None,
                    "selected_point_estimate": axis == selected[budget],
                }
            )
    return output, selected


def selection_aware_bootstrap(
    rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> dict[str, Any]:
    axes = [int(axis) for axis in design["encode_axes_px"]]
    stable_ids = sorted({str(row["stable_sample_id"]) for row in rows})
    id_index = {stable_id: index for index, stable_id in enumerate(stable_ids)}
    clean = np.zeros(len(stable_ids), dtype=np.int8)
    seen_clean: set[str] = set()
    by_cell: dict[tuple[int, int], np.ndarray] = {}
    for budget in (int(item["budget_bytes"]) for item in design["budget_grid"]):
        for axis in axes:
            by_cell[(budget, axis)] = np.zeros(len(stable_ids), dtype=np.int8)
    for row in rows:
        stable_id = str(row["stable_sample_id"])
        index = id_index[stable_id]
        clean_value = int(bool(row["clean_correct"]))
        if stable_id in seen_clean and clean[index] != clean_value:
            raise ProbeDesignError("clean trajectory is inconsistent")
        clean[index] = clean_value
        seen_clean.add(stable_id)
        by_cell[(int(row["budget_bytes"]), int(row["encode_axis"]))][index] = int(
            bool(row["correct"])
        )
    bootstrap = design["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    resamples = int(bootstrap["resamples"])
    draw = rng.integers(
        0,
        len(stable_ids),
        size=(resamples, len(stable_ids)),
        dtype=np.int32,
    )
    clean_accuracy = clean[draw].mean(axis=1)
    budget_results: list[dict[str, Any]] = []
    for budget_item in design["budget_grid"]:
        budget = int(budget_item["budget_bytes"])
        axis_accuracies = np.stack(
            [by_cell[(budget, axis)][draw].mean(axis=1) for axis in axes],
            axis=1,
        )
        selected_indices = np.argmax(axis_accuracies, axis=1)
        selected_accuracy = axis_accuracies[
            np.arange(resamples, dtype=np.int64), selected_indices
        ]
        differences = selected_accuracy - clean_accuracy
        lower = float(
            np.quantile(
                differences,
                1 - float(bootstrap["one_sided_confidence"]),
                method=str(bootstrap["quantile_method"]),
            )
        )
        point_counts = [
            int(by_cell[(budget, axis)].sum())
            for axis in axes
        ]
        point_index = int(np.argmax(np.asarray(point_counts)))
        point_axis = axes[point_index]
        point_accuracy = point_counts[point_index] / len(stable_ids)
        point_clean = float(clean.mean())
        emitted = [
            float(row["realized_bpp"])
            for row in rows
            if int(row["budget_bytes"]) == budget
            and int(row["encode_axis"]) == point_axis
            and bool(row["feasible"])
        ]
        budget_results.append(
            {
                "budget_bytes": budget,
                "requested_bpp": float(budget_item["requested_bpp"]),
                "selected_encode_axis": point_axis,
                "selected_n_correct": point_counts[point_index],
                "selected_accuracy": point_accuracy,
                "clean_accuracy": point_clean,
                "point_difference": point_accuracy - point_clean,
                "one_sided_95_lower_bound": lower,
                "mean_realized_bpp": float(np.mean(emitted)) if emitted else None,
                "meets_5pp": lower >= -0.05,
                "meets_2pp": lower >= -0.02,
            }
        )
    return {
        "method": bootstrap["method"],
        "resamples": resamples,
        "seed": int(bootstrap["seed"]),
        "identity": bootstrap["identity"],
        "one_sided_confidence": float(bootstrap["one_sided_confidence"]),
        "quantile_method": bootstrap["quantile_method"],
        "budgets": budget_results,
    }


def threshold_forecast(
    bootstrap_result: dict[str, Any],
    *,
    key: str,
    label: str,
) -> dict[str, Any]:
    budgets = bootstrap_result["budgets"]
    matching = [item for item in budgets if item[key]]
    if not matching:
        return {
            "label": label,
            "status": "not_reached_within_frozen_grid",
            "right_censored": True,
            "left_censored": False,
            "result": None,
        }
    chosen = matching[0]
    return {
        "label": label,
        "status": "left_censored" if chosen is budgets[0] else "measured",
        "right_censored": False,
        "left_censored": chosen is budgets[0],
        "result": chosen,
    }
