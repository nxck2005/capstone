#!/usr/bin/env python3
"""Run, shard, and merge the validation-only JPEG 2000 transparency probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import glymur
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.j2k import J2KCodec  # noqa: E402
from config.params import get  # noqa: E402
from data.preprocessing import (  # noqa: E402
    codec_downsample,
    codec_upsample,
    evaluation_input,
    reconstruction_input,
    reconstruction_metrics,
)
from data.registry import load_dataset  # noqa: E402
from env import assert_cuda, assert_j2k_runtime, loaded_openjpeg_version  # noqa: E402
from models.frozen_reference_classifier import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
    load_frozen_reference_classifier,
)
from probes.transparency_bitrate import (  # noqa: E402
    AGGREGATE_FIELDS,
    PER_IMAGE_FIELDS,
    aggregate,
    canonical_json,
    design_fingerprint,
    format_float,
    load_design,
    parameter_snapshot,
    read_csv,
    selection_aware_bootstrap,
    sha256_path,
    threshold_forecast,
    write_csv,
)

DEFAULT_CONFIG = REPO / "configs/transparency-bitrate-probe.yaml"
_DISCLAIMER = (
    "Validation-only engineering probe. "
    "Does not select or replace G-8 operating points."
)


class ProbeRunError(RuntimeError):
    """A hard-stop probe execution or evidence error."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ProbeRunError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _measurement_state() -> tuple[str, bool]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=all"))
    return commit, dirty


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_samples(indices: list[int] | None = None):
    source = load_dataset("imagenette160", "val", REPO)
    chosen = range(len(source)) if indices is None else indices
    return [(*source[index],) for index in chosen]


def _pilot_indices(stable_ids: list[str], pilot: dict[str, Any]) -> list[int]:
    identity = str(pilot["identity"])
    scored = [
        (
            hashlib.sha256(f"{identity}\0{stable_id}".encode("ascii")).hexdigest(),
            index,
        )
        for index, stable_id in enumerate(stable_ids)
    ]
    return [index for _, index in sorted(scored)[: int(pilot["size"])]]


def _score_clean(model, samples, *, batch_size: int, device: torch.device):
    predictions: dict[str, int] = {}
    correct: dict[str, bool] = {}
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        inputs = torch.stack([evaluation_input(product) for product, _ in batch])
        with torch.no_grad():
            logits = model(inputs.to(device))
        values = logits.argmax(dim=1).cpu().tolist()
        for (product, label), prediction in zip(batch, values, strict=True):
            predictions[product.stable_sample_id] = int(prediction)
            correct[product.stable_sample_id] = int(prediction) == int(label)
    return predictions, correct


def _flush_predictions(model, pending, *, device: torch.device) -> None:
    if not pending:
        return
    tensors = torch.stack([item[1] for item in pending])
    with torch.no_grad():
        predictions = model(tensors.to(device)).argmax(dim=1).cpu().tolist()
    for (row, _), prediction in zip(pending, predictions, strict=True):
        row["predicted_class"] = int(prediction)
        row["correct"] = int(prediction) == int(row["label"])
    pending.clear()


def _evaluate_axis(
    *,
    model,
    samples,
    clean_predictions: dict[str, int],
    clean_correct: dict[str, bool],
    design: dict[str, Any],
    axis: int,
    codec: J2KCodec,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    target_hw = tuple(int(value) for value in get("datasets.imagenette160.image_size")[:2])
    pixel_count = target_hw[0] * target_hw[1]
    downsampled = [
        codec_downsample(product.canonical_image, axis)
        for product, _ in samples
    ]
    canonical_hashes = [
        hashlib.sha256(product.canonical_image.tobytes()).hexdigest()
        for product, _ in samples
    ]
    rows: list[dict[str, Any]] = []
    cache_hits = 0
    for budget_item in design["budget_grid"]:
        budget = int(budget_item["budget_bytes"])
        pending: list[tuple[dict[str, Any], torch.Tensor]] = []
        for sample_index, ((product, label), encoded_image) in enumerate(
            zip(samples, downsampled, strict=True)
        ):
            result = codec.encode_to_budget(
                encoded_image,
                canonical_pixels_sha256=canonical_hashes[sample_index],
                budget_bytes=budget,
                encode_axis_px=axis,
            )
            cache_hits += int(result.cache_hit)
            row: dict[str, Any] = {
                "stable_sample_id": product.stable_sample_id,
                "label": int(label),
                "budget_source": "|".join(budget_item["sources"]),
                "budget_bytes": budget,
                "requested_bpp": float(budget_item["requested_bpp"]),
                "encode_axis": axis,
                "cache_key": result.cache_key,
                "codestream_sha256": result.codestream_sha256,
                "emitted_bytes": result.emitted_byte_count,
                "realized_bpp": (
                    result.emitted_byte_count
                    * np.iinfo(np.uint8).bits
                    / pixel_count
                    if result.emitted_byte_count is not None
                    else None
                ),
                "search_iterations": result.search_iterations,
                "feasible": result.feasible,
                "decode_success": result.decode_success,
                "predicted_class": None,
                "correct": False,
                "clean_predicted_class": clean_predictions[
                    product.stable_sample_id
                ],
                "clean_correct": clean_correct[product.stable_sample_id],
                "psnr": None,
                "ssim": None,
            }
            if result.feasible:
                if result.decoded_image is None or not result.decode_success:
                    raise ProbeRunError("feasible JPEG 2000 result did not decode")
                restored = codec_upsample(result.decoded_image, target_hw)
                reconstructed = reconstruction_input(restored)
                metrics = reconstruction_metrics(
                    evaluation_input(product), reconstructed
                )
                row["psnr"] = metrics.psnr_db
                row["ssim"] = metrics.ssim
                pending.append((row, reconstructed))
                if len(pending) == int(design["classifier_batch_size"]):
                    _flush_predictions(model, pending, device=device)
            rows.append(row)
        _flush_predictions(model, pending, device=device)
    return rows, {
        "wall_time_s": time.perf_counter() - started,
        "cache_hits": cache_hits,
        "cells": len(rows),
        "mean_search_iterations": float(
            np.mean([row["search_iterations"] for row in rows])
        ),
    }


def _serialise_per_image(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    float_fields = {"requested_bpp", "realized_bpp", "psnr", "ssim"}
    bool_fields = {"feasible", "decode_success", "correct", "clean_correct"}
    for row in rows:
        encoded: dict[str, str] = {}
        for field in PER_IMAGE_FIELDS:
            value = row[field]
            if value is None:
                encoded[field] = ""
            elif field in float_fields:
                encoded[field] = format_float(float(value))
            elif field in bool_fields:
                encoded[field] = "true" if bool(value) else "false"
            else:
                encoded[field] = str(value)
        output.append(encoded)
    return output


def _parse_per_image(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
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
    output: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for field, value in row.items():
            if value == "":
                parsed[field] = None
            elif field in integer_fields:
                parsed[field] = int(value)
            elif field in float_fields:
                parsed[field] = float(value)
            elif field in bool_fields:
                if value not in {"true", "false"}:
                    raise ProbeRunError(f"invalid boolean {field}={value!r}")
                parsed[field] = value == "true"
            else:
                parsed[field] = value
        output.append(parsed)
    return output


def _clean_check(correct: dict[str, bool], *, full: bool) -> dict[str, Any]:
    n_correct = sum(correct.values())
    total = len(correct)
    if full and (n_correct != 898 or total != 1000):
        raise ProbeRunError(
            f"frozen clean validation mismatch: {n_correct}/{total}, expected 898/1000"
        )
    return {
        "n_correct": n_correct,
        "n_total": total,
        "top1_accuracy": n_correct / total,
    }


def run_pilot(design: dict[str, Any]) -> dict[str, Any]:
    assert_j2k_runtime()
    assert_cuda()
    device = torch.device("cuda", 0)
    all_source = load_dataset("imagenette160", "val", REPO)
    stable_ids = [
        all_source.source_sample(index).stable_sample_id
        for index in range(len(all_source))
    ]
    indices = _pilot_indices(stable_ids, design["pilot"])
    samples = [(*all_source[index],) for index in indices]
    model = load_frozen_reference_classifier(device)
    clean_predictions, clean_correct = _score_clean(
        model,
        samples,
        batch_size=int(design["classifier_batch_size"]),
        device=device,
    )
    cache_root = REPO / design["cache_root"]
    before = sum(
        path.stat().st_size for path in cache_root.glob("*.j2kcache")
    ) if cache_root.exists() else 0
    codec = J2KCodec(cache_root)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    axis_stats: list[dict[str, Any]] = []
    for axis in design["encode_axes_px"]:
        axis_rows, stats = _evaluate_axis(
            model=model,
            samples=samples,
            clean_predictions=clean_predictions,
            clean_correct=clean_correct,
            design=design,
            axis=int(axis),
            codec=codec,
            device=device,
        )
        rows.extend(axis_rows)
        axis_stats.append({"axis": int(axis), **stats})
    elapsed = time.perf_counter() - started
    first_product, _ = samples[0]
    first_axis = int(design["encode_axes_px"][0])
    first_image = codec_downsample(first_product.canonical_image, first_axis)
    first_budget = int(design["budget_grid"][0]["budget_bytes"])
    cache_check = codec.encode_to_budget(
        first_image,
        canonical_pixels_sha256=hashlib.sha256(
            first_product.canonical_image.tobytes()
        ).hexdigest(),
        budget_bytes=first_budget,
        encode_axis_px=first_axis,
    )
    if not cache_check.cache_hit:
        raise ProbeRunError("pilot cache repeat did not hit")
    cache_files = tuple(cache_root.glob("*.j2kcache"))
    after = sum(path.stat().st_size for path in cache_files)
    full_cells = (
        int(get("datasets.imagenette160.val_images"))
        * len(design["budget_grid"])
        * len(design["encode_axes_px"])
    )
    pilot = {
        "schema_version": 1,
        "status": "PASS",
        "stable_sample_ids": [
            product.stable_sample_id for product, _ in samples
        ],
        "selection": design["pilot"],
        "codec_configuration_hash": codec.configuration_hash,
        "openjpeg_version": loaded_openjpeg_version(required=True),
        "glymur_version": glymur.__version__,
        "cells": len(rows),
        "feasible_cells": sum(bool(row["feasible"]) for row in rows),
        "budget_compliance": all(
            row["emitted_bytes"] is None
            or row["emitted_bytes"] <= row["budget_bytes"]
            for row in rows
        ),
        "decode_success": all(
            not row["feasible"] or row["decode_success"] for row in rows
        ),
        "average_search_iterations": float(
            np.mean([row["search_iterations"] for row in rows])
        ),
        "wall_time_s": elapsed,
        "wall_time_per_image_budget_axis_s": elapsed / len(rows),
        "projected_full_wall_time_s": elapsed / len(rows) * full_cells,
        "projected_full_wall_time_h": elapsed / len(rows) * full_cells / 3600,
        "cache_repeat_hit": cache_check.cache_hit,
        "cache_bytes_before": before,
        "cache_bytes_after": after,
        "cache_bytes_added": after - before,
        "cache_entries": len(cache_files),
        "maximum_cache_entry_bytes": max(
            (path.stat().st_size for path in cache_files), default=0
        ),
        "temporary_disk_use_bytes_upper_bound": after - before,
        "axis_stats": axis_stats,
        "test_split_accessed": False,
        "training_performed": False,
    }
    destination = REPO / design["shard_root"] / "pilot.json"
    _json_write(destination, pilot)
    return pilot


def run_axis(design: dict[str, Any], axis: int) -> dict[str, Any]:
    if axis not in design["encode_axes_px"]:
        raise ProbeRunError(f"axis {axis} is not in the frozen design")
    commit, dirty = _measurement_state()
    if dirty:
        raise ProbeRunError("full probe measurement requires a clean worktree")
    assert_j2k_runtime()
    assert_cuda()
    device = torch.device("cuda", 0)
    samples = _load_samples()
    model = load_frozen_reference_classifier(device)
    clean_predictions, clean_correct = _score_clean(
        model,
        samples,
        batch_size=int(design["classifier_batch_size"]),
        device=device,
    )
    clean_result = _clean_check(clean_correct, full=True)
    codec = J2KCodec(REPO / design["cache_root"])
    rows, stats = _evaluate_axis(
        model=model,
        samples=samples,
        clean_predictions=clean_predictions,
        clean_correct=clean_correct,
        design=design,
        axis=axis,
        codec=codec,
        device=device,
    )
    shard_root = REPO / design["shard_root"]
    csv_path = shard_root / f"axis-{axis}.csv"
    metadata_path = shard_root / f"axis-{axis}.json"
    write_csv(csv_path, PER_IMAGE_FIELDS, _serialise_per_image(rows))
    metadata = {
        "schema_version": 1,
        "axis": axis,
        "measurement_commit": commit,
        "git_dirty": dirty,
        "design_hash": design_fingerprint(design),
        "clean_validation": clean_result,
        "rows": len(rows),
        "expected_rows": int(get("datasets.imagenette160.val_images"))
        * len(design["budget_grid"]),
        "csv_sha256": sha256_path(csv_path),
        "codec_configuration_hash": codec.configuration_hash,
        "stats": stats,
        "test_split_accessed": False,
        "training_performed": False,
    }
    _json_write(metadata_path, metadata)
    return metadata


def _aggregate_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    float_fields = {
        "requested_bpp",
        "top1_accuracy",
        "accuracy_difference_from_clean",
        "mean_emitted_bytes",
        "median_emitted_bytes",
        "mean_realized_bpp",
        "mean_psnr",
        "mean_ssim",
    }
    output: list[dict[str, str]] = []
    for row in rows:
        output.append(
            {
                field: (
                    ""
                    if row[field] is None
                    else format_float(row[field])
                    if field in float_fields
                    else "true"
                    if field == "selected_point_estimate" and row[field]
                    else "false"
                    if field == "selected_point_estimate"
                    else str(row[field])
                )
                for field in AGGREGATE_FIELDS
            }
        )
    return output


def _cache_manifest(rows: list[dict[str, Any]], design: dict[str, Any]):
    cache_root = REPO / design["cache_root"]
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row["cache_key"])
        if key in seen:
            raise ProbeRunError(f"duplicate cache key in cells: {key}")
        seen.add(key)
        path = cache_root / f"{key}.j2kcache"
        if not path.is_file():
            raise ProbeRunError(f"cache entry is absent: {key}")
        entries.append(
            {
                "cache_key": key,
                "cache_path": path.relative_to(REPO).as_posix(),
                "cache_file_sha256": sha256_path(path),
                "codestream_sha256": row["codestream_sha256"],
                "emitted_bytes": row["emitted_bytes"],
                "feasible": bool(row["feasible"]),
            }
        )
    return {
        "schema_version": 1,
        "cache_root": (REPO / design["cache_root"]).relative_to(REPO).as_posix(),
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda item: item["cache_key"]),
    }


def _worklog(summary: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> str:
    selected = [row for row in aggregate_rows if row["selected_point_estimate"]]
    lines = [
        "# Validation-only JPEG 2000 transparency-bitrate probe",
        "",
        f"**Status:** {summary['probe_status']}",
        "",
        _DISCLAIMER,
        "",
        "No training ran, the classifier remained frozen, and the test split stayed sealed.",
        "G-8 remains unresolved; both threshold outputs below are probe forecasts only.",
        "",
        "## Fixed codec",
        "",
        "OpenJPEG 2.5.4 through Glymur 0.14.3; raw codestream, irreversible 9/7,",
        "RPCL, six resolutions, 64×64 code blocks, whole-image tile, and bounded",
        "compression-ratio bisection retaining the largest observed codestream at or",
        "below the byte budget.",
        "",
        "## Selected validation curve",
        "",
        "| Budget bytes | Requested bpp | Axis | Correct | Accuracy | Δ clean | Mean realised bpp |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['budget_bytes']} | {row['requested_bpp']:.7g} | "
            f"{row['encode_axis']} | {row['n_correct']}/{row['n_total']} | "
            f"{row['top1_accuracy']:.6f} | "
            f"{row['accuracy_difference_from_clean']:.6f} | "
            f"{row['mean_realized_bpp']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Forecasts",
            "",
            "```json",
            json.dumps(
                {
                    "probe_efficiency_threshold": summary[
                        "probe_efficiency_threshold"
                    ],
                    "probe_crossover_threshold": summary[
                        "probe_crossover_threshold"
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            f"Completed cells: {summary['completed_validation_cells']}. "
            f"Infeasible: {summary['codec_totals']['infeasible_count']}. "
            f"Decode failures: {summary['codec_totals']['decode_failure_count']}.",
            "",
        ]
    )
    return "\n".join(lines)


def merge(design: dict[str, Any]) -> dict[str, Any]:
    shard_root = REPO / design["shard_root"]
    metadata: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for axis in design["encode_axes_px"]:
        csv_path = shard_root / f"axis-{axis}.csv"
        metadata_path = shard_root / f"axis-{axis}.json"
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        if item["csv_sha256"] != sha256_path(csv_path):
            raise ProbeRunError(f"axis {axis} shard hash disagrees")
        if (
            item["axis"] != axis
            or item["git_dirty"]
            or item["rows"] != item["expected_rows"]
            or item["clean_validation"]
            != {"n_correct": 898, "n_total": 1000, "top1_accuracy": 0.898}
        ):
            raise ProbeRunError(f"axis {axis} shard metadata disagrees")
        metadata.append(item)
        rows.extend(_parse_per_image(read_csv(csv_path, PER_IMAGE_FIELDS)))
    commits = {item["measurement_commit"] for item in metadata}
    design_hashes = {item["design_hash"] for item in metadata}
    codec_hashes = {item["codec_configuration_hash"] for item in metadata}
    if len(commits) != 1 or len(design_hashes) != 1 or len(codec_hashes) != 1:
        raise ProbeRunError("shards disagree on commit, design, or codec identity")
    expected_cells = (
        int(get("datasets.imagenette160.val_images"))
        * len(design["budget_grid"])
        * len(design["encode_axes_px"])
    )
    keys = {
        (
            row["stable_sample_id"],
            row["budget_bytes"],
            row["encode_axis"],
        )
        for row in rows
    }
    if len(rows) != expected_cells or len(keys) != expected_cells:
        raise ProbeRunError("merged shard cells are missing or duplicated")
    budget_order = {
        int(item["budget_bytes"]): index
        for index, item in enumerate(design["budget_grid"])
    }
    axis_order = {
        int(axis): index for index, axis in enumerate(design["encode_axes_px"])
    }
    rows.sort(
        key=lambda row: (
            budget_order[int(row["budget_bytes"])],
            axis_order[int(row["encode_axis"])],
            str(row["stable_sample_id"]),
        )
    )
    outputs = {key: REPO / value for key, value in design["outputs"].items()}
    write_csv(outputs["per_image"], PER_IMAGE_FIELDS, _serialise_per_image(rows))
    aggregate_rows, selected_axes = aggregate(rows, design)
    write_csv(
        outputs["aggregate"],
        AGGREGATE_FIELDS,
        _aggregate_for_csv(aggregate_rows),
    )
    bootstrap = selection_aware_bootstrap(rows, design)
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
    manifest = _cache_manifest(rows, design)
    _json_write(outputs["cache_manifest"], manifest)
    measurement_commit = next(iter(commits))
    resolved = {
        "schema_version": 1,
        "source": "configs/transparency-bitrate-probe.yaml",
        "design": design,
        "parameters": parameter_snapshot(),
        "design_hash": design_fingerprint(design),
        "measurement_commit": measurement_commit,
        "measurement_git_dirty": False,
    }
    _json_write(outputs["resolved_config"], resolved)
    total_wall = sum(float(item["stats"]["wall_time_s"]) for item in metadata)
    cache_size = sum(
        (REPO / entry["cache_path"]).stat().st_size
        for entry in manifest["entries"]
    )
    summary = {
        "schema_version": 1,
        "probe_status": "COMPLETE",
        "prominent_declaration": _DISCLAIMER,
        "measurement_commit": measurement_commit,
        "git_dirty_state": False,
        "dataset": "imagenette160",
        "split": "validation",
        "dataset_identity": get("datasets.imagenette160.archive_sha256"),
        "archive_identity": get("datasets.imagenette160.archive_sha256"),
        "manifest_identity": get("datasets.imagenette160.manifest_sha256"),
        "classifier_checkpoint_identity": EXPECTED_CHECKPOINT_SHA256,
        "classifier_config_identity": EXPECTED_CONFIG_HASH,
        "classifier_variant": "clean",
        "clean_validation": {
            "n_correct": 898,
            "n_total": 1000,
            "top1_accuracy": 0.898,
        },
        "codec_configuration": J2KCodec(REPO / design["cache_root"]).snapshot,
        "codec_configuration_hash": next(iter(codec_hashes)),
        "openjpeg_version": loaded_openjpeg_version(required=True),
        "glymur_version": glymur.__version__,
        "budget_grid": design["budget_grid"],
        "encode_axis_order": design["encode_axes_px"],
        "bootstrap": bootstrap,
        "bootstrap_resamples": int(design["bootstrap"]["resamples"]),
        "threshold_definitions": design["thresholds"],
        "point_estimate_best_axes": [
            {"budget_bytes": budget, "encode_axis": selected_axes[budget]}
            for budget in (
                int(item["budget_bytes"]) for item in design["budget_grid"]
            )
        ],
        "probe_efficiency_threshold": efficiency,
        "probe_crossover_threshold": crossover,
        "test_isolation_declaration": {
            "test_split_sealed": True,
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
        },
        "cache_manifest_hash": sha256_path(outputs["cache_manifest"]),
        "per_image_file_hash": sha256_path(outputs["per_image"]),
        "aggregate_file_hash": sha256_path(outputs["aggregate"]),
        "resolved_config_hash": sha256_path(outputs["resolved_config"]),
        "commands_used": [
            ".venv/bin/python tools/run_transparency_bitrate_probe.py --pilot",
            *[
                ".venv/bin/python tools/run_transparency_bitrate_probe.py "
                f"--axis {axis}"
                for axis in design["encode_axes_px"]
            ],
            ".venv/bin/python tools/run_transparency_bitrate_probe.py --merge",
        ],
        "completed_validation_cells": len(rows),
        "codec_totals": {
            "infeasible_count": sum(not bool(row["feasible"]) for row in rows),
            "decode_failure_count": sum(
                bool(row["feasible"]) and not bool(row["decode_success"])
                for row in rows
            ),
        },
        "probe_wall_time_s": total_wall,
        "cache_size_bytes": cache_size,
        "provisional_bandwidth_parameters": {
            "crossover_ratio": get("bandwidth.crossover_ratio"),
            "crossover_ratio_status": get("bandwidth.crossover_ratio_status"),
            "efficiency_ratio": get("bandwidth.efficiency_ratio"),
            "efficiency_ratio_status": get("bandwidth.efficiency_ratio_status"),
            "low_ratio_operating_point": get(
                "bandwidth.low_ratio_operating_point"
            ),
            "low_ratio_operating_point_status": get(
                "bandwidth.low_ratio_operating_point_status"
            ),
        },
        "g8_status": "unresolved",
        "training_performed": False,
    }
    _json_write(outputs["summary"], summary)
    outputs["worklog"].parent.mkdir(parents=True, exist_ok=True)
    outputs["worklog"].write_text(
        _worklog(summary, aggregate_rows),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--axis", type=int)
    group.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    try:
        design = load_design(args.config.resolve(), repo_root=REPO)
        if args.pilot:
            result = run_pilot(design)
            print(
                "Transparency pilot PASS: "
                f"cells={result['cells']}, "
                f"mean_search={result['average_search_iterations']:.2f}, "
                f"projected={result['projected_full_wall_time_h']:.3f} h"
            )
        elif args.axis is not None:
            result = run_axis(design, args.axis)
            print(
                f"Transparency shard axis={args.axis} PASS: "
                f"rows={result['rows']}, "
                f"wall={result['stats']['wall_time_s']:.3f}s"
            )
        else:
            result = merge(design)
            print(
                "Transparency probe merge COMPLETE: "
                f"cells={result['completed_validation_cells']}"
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Transparency probe FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
