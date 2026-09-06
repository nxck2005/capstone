"""The sole model-facing W9-A/G-10 validation runner.

This module is imported only by the owner-authorized campaign entry point.  It
has one evaluation route: the three frozen W8 ``r_1_6`` checkpoints over the
21 frozen SNRs.  It has no training, test-split, classifier-selection, or
classical execution path.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from channels.awgn import keyed_complex_noise
from config.execution_profiles import authenticate_execution_profile
from config.run_config import config_hash, load_experiment
from data.djscc_validation import ValidationDJSCCDataset, validation_noise_id
from evaluation.g10_protocol import (
    AUTHORIZATION_PATH,
    EXPECTED_DATASET,
    EXPECTED_DENOMINATOR,
    EXPECTED_DEVICE,
    EXPECTED_GPU_NAME,
    EXPECTED_GPU_UUID,
    EXPECTED_PROFILE_ID,
    EXPECTED_RATIO,
    EXPECTED_SPLIT,
    EXPECTED_CELL_COUNT,
    G10ProtocolHold,
    RUNTIME_PREFIX,
    canonical_sha256,
    cell_key,
    expected_cell_keys,
    load_json,
    rendered_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_authorization,
)
from models.djscc import build_djscc


class G10ExecutionHold(G10ProtocolHold):
    """A fail-closed execution, custody, or exact-cell violation."""


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _atomic_exclusive(path: Path, raw: bytes) -> None:
    """Publish one immutable runtime object without replacing an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.link(temporary, path, follow_symlinks=False)
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise G10ExecutionHold(f"immutable G-10 runtime object already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_mutable(path: Path, value: dict[str, Any]) -> None:
    """Write coordination state atomically; scientific cell records are exclusive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(rendered_json(value))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _prediction_digest(
    ids: list[str], predictions: list[int], labels: list[int]
) -> str:
    return canonical_sha256(
        [
            {
                "stable_sample_id": stable_id,
                "prediction": int(prediction),
                "label": int(label),
                "correct": int(prediction) == int(label),
            }
            for stable_id, prediction, label in zip(ids, predictions, labels, strict=True)
        ]
    )


def _noise_ids(
    stable_ids: list[str], *, channel_seed: int, snr_db: int, ratio: str, k: int, root: Path
) -> list[str]:
    from config.params import get

    version_field = str(get("config.dataset_version_rule"))
    dataset_version = str(get(f"datasets.{EXPECTED_DATASET}.{version_field}"))
    manifest_hash = str(get(f"datasets.{EXPECTED_DATASET}.manifest_sha256"))
    return [
        validation_noise_id(
            stable_sample_id=stable_id,
            dataset_version=dataset_version,
            split_manifest_hash=manifest_hash,
            channel_seed=channel_seed,
            channel="awgn",
            ratio=ratio,
            k=k,
            snr_db=snr_db,
        )
        for stable_id in stable_ids
    ]


def _validation_dataset(root: Path) -> ValidationDJSCCDataset:
    dataset = ValidationDJSCCDataset(EXPECTED_DATASET, repo_root=root)
    require(len(dataset) == EXPECTED_DENOMINATOR, "G-10 validation denominator differs from 1000")
    identifiers = [str(dataset._source.source_sample(index).stable_sample_id) for index in range(len(dataset))]
    require(identifiers == sorted(identifiers) and len(set(identifiers)) == len(identifiers), "G-10 validation order/identity differs")
    return dataset


def _checkpoint_payload(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"W8 checkpoint is missing or unsafe: {path}")
    require(sha256_file(path) == checkpoint["checkpoint_sha256"], f"W8 checkpoint bytes differ: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "W8 checkpoint payload is not a mapping")
    require(payload.get("artifact_role") == "W8_FINAL_TRAINING_CHECKPOINT", "checkpoint is not a W8 final checkpoint")
    require(payload.get("completed_epoch") == checkpoint["epoch"], "checkpoint epoch differs from frozen W8 mapping")
    require(payload.get("run_id") == checkpoint["run_id"], "checkpoint run identity differs")
    lineage = payload.get("lineage")
    require(isinstance(lineage, dict), "W8 checkpoint lineage is missing")
    require(lineage.get("ratio") == EXPECTED_RATIO, "checkpoint ratio is not r_1_6")
    require(lineage.get("train_seed") == checkpoint["train_seed"] and lineage.get("channel_seed") == checkpoint["channel_seed"], "checkpoint seed pairing differs")
    protected = payload.get("protected_counters")
    require(isinstance(protected, dict), "W8 checkpoint protected counters are missing")
    require(all(int(value) == 0 for key, value in protected.items() if key in {"g10_adjudications", "er2_randomized_training", "er9_training", "learned_test_inference", "test_model_facing_access"}), "W8 checkpoint protected boundary moved")
    model_state = payload.get("model_state")
    require(isinstance(model_state, dict) and model_state, "W8 checkpoint model state is missing")
    return payload


def _cell_record(
    *,
    cell_index: int,
    checkpoint: dict[str, Any],
    snr_db: int,
    n_correct: int,
    n_total: int,
    stable_ids: list[str],
    labels: list[int],
    predictions: list[int],
    noise_ids: list[str],
    config_hash_value: str,
    protocol_sha256: str,
    source_commit: str,
    execution_checkout_commit: str,
    profile_binding: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    rows = [
        {
            "stable_sample_id": stable_id,
            "label": int(label),
            "prediction": int(prediction),
            "correct": int(prediction) == int(label),
            "noise_id": noise_id,
        }
        for stable_id, label, prediction, noise_id in zip(
            stable_ids, labels, predictions, noise_ids, strict=True
        )
    ]
    body = {
        "schema_version": 1,
        "artifact_role": "G10_LEARNED_VALIDATION_CELL",
        "status": "COMPLETE",
        "cell_index": cell_index,
        "cell_key": cell_key(checkpoint["train_seed"], checkpoint["channel_seed"], snr_db),
        "train_seed": checkpoint["train_seed"],
        "channel_seed": checkpoint["channel_seed"],
        "selected_epoch": checkpoint["epoch"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "ratio": EXPECTED_RATIO,
        "snr_db": snr_db,
        "dataset": EXPECTED_DATASET,
        "validation_split": EXPECTED_SPLIT,
        "validation_manifest_sha256": "224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889",
        "validation_denominator": n_total,
        "n_correct": n_correct,
        "top1_accuracy": n_correct / n_total,
        "top1_extraction": "torch.Tensor.argmax(dim=1)_first_index",
        "validation_order": "stable_manifest_order",
        "noise": {
            "policy": "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs",
            "identity_function": "data.djscc_validation.validation_noise_id",
            "rng_purpose": "channel_noise",
            "noise_id_count": len(noise_ids),
            "noise_id_digest": canonical_sha256(noise_ids),
            "noise_id_first": noise_ids[0],
            "noise_id_last": noise_ids[-1],
            "ambient_sequential_rng": False,
        },
        "prediction_digest": _prediction_digest(stable_ids, predictions, labels),
        "row_digest": canonical_sha256(rows),
        "rows": rows,
        "config_hash": config_hash_value,
        "protocol_sha256": protocol_sha256,
        "scientific_source_commit": source_commit,
        "execution_checkout_commit": execution_checkout_commit,
        "execution_profile": profile_binding,
        "authority_id": None,
        "runtime_root": str(runtime_root),
        "training": 0,
        "test_access": 0,
    }
    return body


def _identify_cell(body: dict[str, Any], authority_id: str) -> dict[str, Any]:
    value = dict(body)
    value["authority_id"] = authority_id
    value["artifact_id"] = "g10cell-" + canonical_sha256(value)
    value["artifact_content_sha256"] = canonical_sha256(value)
    return value


def _verify_cell(value: dict[str, Any], *, expected_index: int, expected_checkpoint: dict[str, Any], expected_snr: int) -> None:
    require(value.get("artifact_role") == "G10_LEARNED_VALIDATION_CELL" and value.get("status") == "COMPLETE", "G-10 cell status differs")
    without_content = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    require(value.get("artifact_content_sha256") == canonical_sha256(without_content), "G-10 cell content identity differs")
    without_id = {key: child for key, child in without_content.items() if key != "artifact_id"}
    require(value.get("artifact_id") == "g10cell-" + canonical_sha256(without_id), "G-10 cell identity differs")
    require(value.get("cell_index") == expected_index and value.get("snr_db") == expected_snr, "G-10 cell coordinates differ")
    for key in ("train_seed", "channel_seed", "selected_epoch", "checkpoint_id", "checkpoint_sha256"):
        expected_key = {"selected_epoch": "epoch"}.get(key, key)
        require(value.get(key) == expected_checkpoint[expected_key], f"G-10 cell {key} differs")
    require(value.get("ratio") == EXPECTED_RATIO and value.get("dataset") == EXPECTED_DATASET and value.get("validation_split") == EXPECTED_SPLIT, "G-10 cell scope differs")
    require(value.get("validation_denominator") == EXPECTED_DENOMINATOR and isinstance(value.get("n_correct"), int) and 0 <= value["n_correct"] <= EXPECTED_DENOMINATOR, "G-10 cell count differs")
    require(value.get("top1_accuracy") == value["n_correct"] / EXPECTED_DENOMINATOR, "G-10 top-1 is not count-derived")
    rows = value.get("rows")
    require(isinstance(rows, list) and len(rows) == EXPECTED_DENOMINATOR, "G-10 cell row denominator differs")
    require(value["row_digest"] == canonical_sha256(rows), "G-10 cell row digest differs")
    ids = [row["stable_sample_id"] for row in rows]
    noise = [row["noise_id"] for row in rows]
    require(ids == sorted(ids) and len(set(ids)) == EXPECTED_DENOMINATOR, "G-10 cell stable-ID coverage differs")
    require(len(set(noise)) == EXPECTED_DENOMINATOR and value["noise"]["noise_id_digest"] == canonical_sha256(noise), "G-10 cell noise schedule differs")
    derived = sum(int(row["correct"]) for row in rows)
    require(derived == value["n_correct"] and value["prediction_digest"] == _prediction_digest(ids, [int(row["prediction"]) for row in rows], [int(row["label"]) for row in rows]), "G-10 cell predictions/count differ")


def run_campaign(*, authorization_path: Path = AUTHORIZATION_PATH, runtime_root: Path, root: Path) -> dict[str, Any]:
    """Execute/resume only the authorized 3x21 matrix on one bound profile."""

    authorization = verify_authorization(root / authorization_path, root=root)
    require(runtime_root.is_absolute(), "G-10 runtime root must be absolute")
    if runtime_root.exists() and runtime_root.is_symlink():
        raise G10ExecutionHold("G-10 runtime root is a symlink")
    runtime_root.mkdir(parents=True, exist_ok=True)
    cells_dir = runtime_root / "cells"
    markers_dir = runtime_root / "started"
    cells_dir.mkdir(exist_ok=True)
    markers_dir.mkdir(exist_ok=True)
    protocol_sha = authorization["protocol"]["protocol_sha256"]
    live_config_hash = protocol_sha
    binding_path = runtime_root / "execution_profile_binding.json"
    if binding_path.exists():
        live_binding, _ = load_json(binding_path, "G-10 live profile binding")
        require(live_binding.get("authority_id") == authorization["authorization_id"], "G-10 runtime belongs to another authority")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == EXPECTED_GPU_UUID, "G-10 requires the frozen Titan Xp UUID in CUDA_VISIBLE_DEVICES")
        live_binding = authenticate_execution_profile(
            EXPECTED_PROFILE_ID,
            device=EXPECTED_DEVICE,
            config_hash=live_config_hash,
            require_openjpeg=False,
        )
        require(live_binding.get("gpu_uuid") == EXPECTED_GPU_UUID and live_binding.get("gpu_name") == EXPECTED_GPU_NAME, "live G-10 GPU differs from authority")
        live_binding = {"authority_id": authorization["authorization_id"], "binding": live_binding}
        _atomic_exclusive(binding_path, rendered_json(live_binding))
    state_path = runtime_root / "campaign_state.json"
    if state_path.exists():
        state, _ = load_json(state_path, "G-10 runtime state")
        require(state.get("authority_id") == authorization["authorization_id"] and state.get("status") in {"RUNNING", "MATRIX_READY_FOR_AGGREGATION"}, "G-10 runtime state belongs to another campaign")
    else:
        state = {
            "schema_version": 1,
            "artifact_role": "G10_RUNTIME_CAMPAIGN_STATE",
            "status": "RUNNING",
            "authority_id": authorization["authorization_id"],
            "expected_cell_count": EXPECTED_CELL_COUNT,
            "source_commit": authorization["scientific_source"]["commit"],
            "execution_checkout_commit": _git_head(root),
            "profile_id": EXPECTED_PROFILE_ID,
            "test": "SEALED",
        }
        _atomic_exclusive(state_path, rendered_json(state))
    execution_checkout_commit = _git_head(root)
    profile_binding = live_binding["binding"]
    expected_keys = expected_cell_keys()
    checkpoint_rows = authorization["checkpoints"]
    dataset = _validation_dataset(root)
    stable_ids = [str(dataset._source.source_sample(index).stable_sample_id) for index in range(len(dataset))]
    labels_by_id: dict[str, int] = {}
    # The source dataset is read once here only to bind label order; the actual
    # model-facing loop below reads the same committed validation view through
    # DataLoader and checks that order again.
    for index in range(len(dataset)):
        _, label, stable_id = dataset[index]
        labels_by_id[str(stable_id)] = int(label)
    require(len(labels_by_id) == EXPECTED_DENOMINATOR, "G-10 validation labels are not complete")
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted(cells_dir.glob("*.json")):
        value, _ = load_json(path, f"G-10 cell {path.name}")
        key = str(value.get("cell_key"))
        require(key in expected_keys, f"foreign G-10 cell exists: {key}")
        expected_index = expected_keys.index(key)
        checkpoint = checkpoint_rows[expected_index // len(authorization["snr_authority"]["resolved_ordered_values_db"])]
        snr = int(authorization["snr_authority"]["resolved_ordered_values_db"][expected_index % len(authorization["snr_authority"]["resolved_ordered_values_db"])])
        _verify_cell(value, expected_index=expected_index, expected_checkpoint=checkpoint, expected_snr=snr)
        completed[key] = value
    for marker in sorted(markers_dir.glob("*.json")):
        marker_value, _ = load_json(marker, f"G-10 cell marker {marker.name}")
        key = str(marker_value.get("cell_key"))
        require(key in expected_keys, f"foreign G-10 cell marker exists: {key}")
        require(key in completed, f"G-10 cell was started but has no successful immutable record: {key}")
    if len(completed) == EXPECTED_CELL_COUNT:
        state["status"] = "MATRIX_READY_FOR_AGGREGATION"
        _write_mutable(state_path, state)
        return _runtime_manifest(authorization, runtime_root, completed, profile_binding, root)
    device = torch.device(EXPECTED_DEVICE)
    for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
        config = load_experiment(
            root / "configs/learned-w8-final.yaml",
            train_seed=int(checkpoint["train_seed"]),
            channel_seed=int(checkpoint["channel_seed"]),
        )
        require(config.resolved["dataset"] == EXPECTED_DATASET and config.resolved["bw_ratio"] == EXPECTED_RATIO and config.resolved["channel_seed"] == checkpoint["channel_seed"] and config.resolved["train_seed"] == checkpoint["train_seed"], "G-10 resolved learned config differs")
        current_config_hash = config_hash(config)
        model = build_djscc(config, device=device)
        payload = _checkpoint_payload(Path(checkpoint["checkpoint_path"]), checkpoint)
        incompatible = model.load_state_dict(payload["model_state"], strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, "W8 checkpoint model state does not load strictly")
        model.eval()
        for snr_index, snr_value in enumerate(authorization["snr_authority"]["resolved_ordered_values_db"]):
            snr_db = int(snr_value)
            key = cell_key(checkpoint["train_seed"], checkpoint["channel_seed"], snr_db)
            if key in completed:
                continue
            expected_index = checkpoint_index * len(authorization["snr_authority"]["resolved_ordered_values_db"]) + snr_index
            marker = {
                "schema_version": 1,
                "artifact_role": "G10_CELL_STARTED_MARKER",
                "authority_id": authorization["authorization_id"],
                "cell_index": expected_index,
                "cell_key": key,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "snr_db": snr_db,
            }
            marker_path = markers_dir / f"{expected_index:03d}-{key}.json"
            if not marker_path.exists():
                _atomic_exclusive(marker_path, rendered_json(marker))
            else:
                marker_value, _ = load_json(marker_path, "G-10 cell marker")
                require(marker_value == marker, f"G-10 cell marker differs: {key}")
            noise_ids: list[str] = []
            all_ids: list[str] = []
            all_labels: list[int] = []
            all_predictions: list[int] = []
            correct = 0
            loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)  # literal-ok: frozen W8 validation batch size
            with torch.inference_mode():
                for inputs, labels, batch_ids in loader:
                    ids = [str(value) for value in batch_ids]
                    batch_labels = [int(value) for value in labels]
                    batch_noise_ids = _noise_ids(ids, channel_seed=int(checkpoint["channel_seed"]), snr_db=snr_db, ratio=EXPECTED_RATIO, k=int(config.resolved["k"]), root=root)
                    inputs_device = inputs.to(device, non_blocking=True)
                    noise = keyed_complex_noise(batch_noise_ids, int(config.resolved["k"]), dtype=torch.complex64, device=device)
                    output = model(inputs_device, snr_db, unit_noise=noise)
                    predictions = [int(value) for value in output.logits.argmax(dim=1).detach().cpu()]
                    correct += sum(int(prediction == label) for prediction, label in zip(predictions, batch_labels, strict=True))
                    all_ids.extend(ids)
                    all_labels.extend(batch_labels)
                    all_predictions.extend(predictions)
                    noise_ids.extend(batch_noise_ids)
            require(all_ids == stable_ids and len(all_ids) == EXPECTED_DENOMINATOR, f"G-10 validation order differs at {key}")
            require(len(set(noise_ids)) == EXPECTED_DENOMINATOR, f"G-10 noise schedule has duplicate IDs at {key}")
            body = _cell_record(
                cell_index=expected_index,
                checkpoint=checkpoint,
                snr_db=snr_db,
                n_correct=correct,
                n_total=len(all_ids),
                stable_ids=all_ids,
                labels=all_labels,
                predictions=all_predictions,
                noise_ids=noise_ids,
                config_hash_value=current_config_hash,
                protocol_sha256=protocol_sha,
                source_commit=authorization["scientific_source"]["commit"],
                execution_checkout_commit=execution_checkout_commit,
                profile_binding=profile_binding,
                runtime_root=runtime_root,
            )
            value = _identify_cell(body, authorization["authorization_id"])
            _verify_cell(value, expected_index=expected_index, expected_checkpoint=checkpoint, expected_snr=snr_db)
            output_path = cells_dir / f"{expected_index:03d}-{key}.json"
            _atomic_exclusive(output_path, rendered_json(value))
            completed[key] = value
    require(len(completed) == EXPECTED_CELL_COUNT, "G-10 runner did not complete exactly 63 cells")
    state["status"] = "MATRIX_READY_FOR_AGGREGATION"
    _write_mutable(state_path, state)
    return _runtime_manifest(authorization, runtime_root, completed, profile_binding, root)


def _runtime_manifest(
    authorization: dict[str, Any],
    runtime_root: Path,
    completed: dict[str, dict[str, Any]],
    profile_binding: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    expected_keys = expected_cell_keys()
    rows = []
    for index, key in enumerate(expected_keys):
        value = completed[key]
        path = runtime_root / "cells" / f"{index:03d}-{key}.json"
        rows.append({
            "cell_index": index,
            "cell_key": key,
            "artifact_id": value["artifact_id"],
            "file_path": str(path),
            "file_sha256": sha256_file(path),
            "n_correct": value["n_correct"],
            "n_total": value["validation_denominator"],
            "noise_id_digest": value["noise"]["noise_id_digest"],
            "row_digest": value["row_digest"],
            "prediction_digest": value["prediction_digest"],
        })
    body = {
        "schema_version": 1,
        "artifact_role": "G10_RUNTIME_MATRIX_MANIFEST",
        "status": "COMPLETE_MATRIX_READY_FOR_AGGREGATION",
        "authority_id": authorization["authorization_id"],
        "source_commit": authorization["scientific_source"]["commit"],
        "execution_checkout_commit": _git_head(root),
        "execution_profile": profile_binding,
        "runtime_root": str(runtime_root),
        "matrix_shape": {"checkpoints": 3, "snr_points": 21, "cells": EXPECTED_CELL_COUNT},  # literal-ok: AM-94 fixes 3 x 21 cells
        "cell_order": "train_seed_ascending_then_snr_grid_order",
        "cells": rows,
        "protected_counters": {"training": 0, "test_access": 0, "g10": EXPECTED_CELL_COUNT},
    }
    value = dict(body)
    value["runtime_manifest_id"] = RUNTIME_PREFIX + canonical_sha256(body)
    value["artifact_content_sha256"] = canonical_sha256(value)
    path = runtime_root / "runtime_manifest.json"
    if path.exists():
        existing, _ = load_json(path, "G-10 runtime manifest")
        require(existing == value, "existing G-10 runtime manifest differs")
    else:
        _atomic_exclusive(path, rendered_json(value))
    return value
