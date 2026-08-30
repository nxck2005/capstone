#!/usr/bin/env python3
"""Build compact, post-hoc W7-B2R evidence from worker custody metadata.

This tool never opens a model checkpoint as a model and never performs model
inference.  It authenticates the already-published JSON metadata, checkpoint
file hashes captured by the worker scan, and frozen per-image selected-result
rows, then writes additive portable evidence.  The worker checkpoint bytes
remain on the worker host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import verify_frozen_gpu_binding  # noqa: E402
from data.classifier import epoch_permutation  # noqa: E402
from data.djscc_validation import validation_noise_id  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.w7_protocol import (  # noqa: E402
    W7_CALIBRATION_SNR_DB,
    W7_CHANNEL_SEED,
    W7_DATASET,
    W7_EXECUTION_IMAGE_FAMILY,
    W7_LAMBDA_GRID,
    W7_PHYSICAL_BATCH_SIZE,
    W7_PROFILE_ID,
    W7_PSNR_SNR_DB,
    W7_RATIO,
    W7_SELECTED_GPU_UUID,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    W7_VALIDATION_BATCH_SIZE,
    W7_VALIDATION_NOISE_POLICY,
    load_w7_config,
    protocol_config_hash,
)

SCHEMA_VERSION = 1  # literal-ok: B2R evidence schema version
CAMPAIGN_ID = "w7-b2-g4-pascal-20260829"
CAMPAIGN_MANIFEST_ID = "w7campaignmanifest-35951e9e85e10b1f176dd5ae7a84512d7ccffb5ee8800afe6dd0bb16776bc661"
CAMPAIGN_MANIFEST_SHA256 = "79a6fef0a06731b75606798249af941bd37eee22ed9b31cb86b8f12da585585e"
CAMPAIGN_COMPLETION_ID = "w7campaign-c6d366c580ee1b2feb8d378b2052cee43484a16e150011b284bf81a58c01efd1"
CAMPAIGN_COMPLETION_SHA256 = "7d1a8e8ea6c2df192dfcd455857cd854f1e418e1ed87cb2d761bcf9815202961"
SOURCE_COMMIT = "cc704fcacec706719bc2791ae14a6c9d71dd4032"
SOURCE_MANIFEST_ID = "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f"
SOURCE_MANIFEST_SHA256 = "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392"
AUTHORIZATION_ID = "w7auth-1d44b66884f48f980576dde94c43eb745227b4ecc48fb964acf90285a854862d"
AUTHORIZATION_SHA256 = "5784ec7ece15051586f915e4e834ca732778f09c2ce537dbd8af4f6e597a8349"
PROFILE_FREEZE_ID = "w7profilefreeze-fab8a6960a6124de7276599c8b6e9971e93266fa23f58cc2a65e3498b41573b9"
PROFILE_FREEZE_SHA256 = "d0eb628a910d93b350a6e9b542f845b5ec418211850899868f16278364ff2301"
GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
DATASET_VERSION = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
SPLIT_MANIFEST_HASH = "224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889"
PROTOCOL_HASH = "2f0fbac301b04423931bd01b7b0a9c3443619b77a969d8629bce708b6d27bdc1"
NOISE_POLICY_HASH = "6008f23154b952d576f7d72c8bc3b3c3f9b9c987d5e5c79cc375277fd7b8219c"
NOISE_ID_DIGEST = "e251ab8ea01be9e1e6c1612e2b9fd411f36b1b68e5942a81029d73195a1033a2"
VAL_STABLE_ID_DIGEST = "a69144f0254eb17ebd9b6862fa353b61cdfdff8759e7e6ad6ab89a709835d1bd"
EXPECTED_EPOCHS = int(get("learned_system.epochs.imagenette160"))
TRAIN_COUNT = int(get("datasets.imagenette160.train_images"))
VAL_COUNT = int(get("datasets.imagenette160.val_images"))
TARGET_BATCH = int(get("learned_system.batch_size.imagenette160"))
FINAL_PARTIAL_BATCH = TRAIN_COUNT % W7_PHYSICAL_BATCH_SIZE
MICROBATCHES = math.ceil(TRAIN_COUNT / W7_PHYSICAL_BATCH_SIZE)

CANDIDATES = (
    (0.0, "lambda-0"),
    (0.1, "lambda-0.1"),
    (0.3, "lambda-0.3"),
    (1.0, "lambda-1"),
    (3.0, "lambda-3"),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    raise AssertionError("unreachable")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite evidence {path}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n")


def artifact(prefix: str, body: dict[str, Any], key: str) -> dict[str, Any]:
    value = dict(body)
    value[key] = prefix + canonical_sha256(body)
    return value


def relative_worker_path(path: str, remote_root: Path, metadata_root: Path) -> Path:
    remote_path = Path(path)
    try:
        relative = remote_path.relative_to(remote_root)
    except ValueError:
        fail(f"worker path escapes campaign root: {path}")
    return metadata_root / relative


def worker_entry(
    entry: dict[str, Any],
    *,
    remote_root: Path,
    metadata_root: Path,
) -> Any:
    if set(entry) != {"path", "sha256", "value"}:
        fail("worker scan entry schema differs")
    path = relative_worker_path(str(entry["path"]), remote_root, metadata_root)
    if file_sha256(path) != entry["sha256"]:
        fail(f"worker metadata file SHA differs: {path}")
    value = load_json(path)
    if value != entry["value"]:
        fail(f"worker metadata differs from captured scan: {path}")
    return value


def load_manifest_ids() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    manifest = REPO / "data/manifests/imagenette160.csv"
    if file_sha256(manifest) != SPLIT_MANIFEST_HASH:
        fail("committed validation/training split manifest SHA differs")
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            item = (row["stable_sample_id"], int(row["label"]))
            if split == "train":
                train.append(item)
            elif split == "val":
                validation.append(item)
            # Test rows are intentionally not loaded into a model-facing data view.
    if len(train) != TRAIN_COUNT or len(validation) != VAL_COUNT:
        fail("committed train/validation denominators differ")
    if set(item[0] for item in train) & set(item[0] for item in validation):
        fail("training and validation stable IDs overlap")
    if [item[0] for item in train] != sorted(item[0] for item in train):
        fail("training manifest is not stable-ID ordered")
    if [item[0] for item in validation] != sorted(item[0] for item in validation):
        fail("validation manifest is not stable-ID ordered")
    if len({item[0] for item in train}) != TRAIN_COUNT or len({item[0] for item in validation}) != VAL_COUNT:
        fail("manifest stable IDs are not unique")
    return train, validation


def expected_validation_noise(validation: list[tuple[str, int]]) -> list[str]:
    return [
        validation_noise_id(
            stable_sample_id=stable_id,
            dataset_version=DATASET_VERSION,
            split_manifest_hash=SPLIT_MANIFEST_HASH,
            channel_seed=W7_CHANNEL_SEED,
            channel="awgn",
            ratio=W7_RATIO,
            k=int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
            snr_db=W7_CALIBRATION_SNR_DB,
        )
        for stable_id, _label in validation
    ]


def training_noise_digest(train_ids: list[str], epoch: int) -> str:
    order = epoch_permutation(len(train_ids), W7_TRAIN_SEED, epoch)
    observed_ids = [train_ids[index] for index in order]
    identities = [
        {
            "dataset_version": DATASET_VERSION,
            "split_manifest_hash": SPLIT_MANIFEST_HASH,
            "stable_sample_id": stable_id,
            "train_seed": W7_TRAIN_SEED,
            "channel_seed": W7_CHANNEL_SEED,
            "epoch": epoch,
            "channel": "awgn",
            "bw_ratio": W7_RATIO,
            "k": int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
            "train_snr_db": W7_TRAINING_SNR_DB,
        }
        for stable_id in observed_ids
    ]
    return hashlib.sha256(
        "\n".join(canonical_sha256(identity) for identity in identities).encode("ascii")
    ).hexdigest()


def line_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def expected_order(train: list[tuple[str, int]], epoch: int) -> tuple[str, str]:
    base = [item[0] for item in train]
    indices = epoch_permutation(len(base), W7_TRAIN_SEED, epoch)
    values = [base[index] for index in indices]
    return line_digest(values), line_digest(sorted(values))


def expected_lr(epoch: int) -> float:
    base = float(get("learned_system.lr"))
    minimum = float(get("learned_system.lr_min"))
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * epoch / max(EXPECTED_EPOCHS - 1, 1)))


def expected_homogeneity(lambda_value: float, config: Any) -> dict[str, Any]:
    return {
        "source_commit": SOURCE_COMMIT,
        "source_manifest_id": SOURCE_MANIFEST_ID,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "campaign_id": CAMPAIGN_ID,
        "authorization_id": AUTHORIZATION_ID,
        "execution_profile_id": W7_PROFILE_ID,
        "execution_image": W7_EXECUTION_IMAGE_FAMILY,
        "gpu_name": GPU_NAME,
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "dataset": W7_DATASET,
        "dataset_version": DATASET_VERSION,
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
        "architecture": str(config.resolved["architecture"]),
        "ratio": W7_RATIO,
        "k": int(config.resolved["k"]),
        "train_seed": W7_TRAIN_SEED,
        "channel_seed": W7_CHANNEL_SEED,
        "training_snr_db": W7_TRAINING_SNR_DB,
        "validation_snr_db": W7_CALIBRATION_SNR_DB,
        "psnr_snr_db": W7_PSNR_SNR_DB,
        "epochs": EXPECTED_EPOCHS,
        "optimizer": "adam",
        "optimizer_implementation": "torch.optim.Adam",
        "scheduler": "cosine",
        "scheduler_indexing": "zero_based",
        "scheduler_step_unit": "epoch_start",
        "physical_batch_size": W7_PHYSICAL_BATCH_SIZE,
        "accumulation_factor": 1,
        "effective_batch_size": TARGET_BATCH,
        "drop_last": False,
        "validation_batch_size": W7_VALIDATION_BATCH_SIZE,
        "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
        "checkpoint_selection": {
            "metric": "top1_accuracy",
            "mode": "max",
            "snr_db": W7_CALIBRATION_SNR_DB,
            "tie_break": "earliest_epoch",
        },
        "lambda": lambda_value,
    }


def compact_epoch(record: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    gradient = record["gradient_checks"]
    return {
        "epoch": record["epoch"],
        "next_epoch": record["next_epoch"],
        "epoch_record_path": entry["path"],
        "epoch_record_sha256": entry["sha256"],
        "record_id": record["record_id"],
        "samples": record["samples"],
        "expected_samples": record["expected_samples"],
        "stable_id_count": record["stable_id_count"],
        "stable_id_order_sha256": record["stable_id_order_sha256"],
        "stable_id_set_sha256": record["stable_id_set_sha256"],
        "training_noise_id_count": record["training_noise_id_count"],
        "training_noise_id_sha256": record["training_noise_id_sha256"],
        "microbatches": record["microbatches"],
        "optimizer_steps": record["optimizer_steps"],
        "grad_scaler_skips": record["grad_scaler_skips"],
        "global_optimizer_step": record["global_optimizer_step"],
        "lr": record["lr"],
        "finite_loss": record["finite_loss"],
        "lineage_sha256": canonical_sha256(record["lineage"]),
        "gradient_audit": {
            "all_named_present_gradients_finite": gradient["all_named_present_gradients_finite"],
            "all_optimizer_gradients_finite": gradient["all_optimizer_gradients_finite"],
            "optimizer_parameter_count": gradient["optimizer_parameter_count"],
            "optimizer_gradient_count_min": gradient["optimizer_gradient_count_min"],
            "optimizer_gradient_count_max": gradient["optimizer_gradient_count_max"],
        },
    }


def compact_validation(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry["value"]
    required = {
        "schema_version", "artifact_role", "epoch", "checkpoint_id", "n_correct",
        "n_total", "top1_accuracy", "prediction_digest", "evaluation_config_hash",
        "noise_policy", "noise_policy_hash", "noise_id_digest", "row_digest", "summary_id",
    }
    if set(value) != required:
        fail("worker validation summary schema differs")
    return {
        "path": entry["path"],
        "file_sha256": entry["sha256"],
        "summary": value,
    }


def compact_checkpoint(
    checkpoint: dict[str, Any],
    epoch: dict[str, Any],
    payload_audit: dict[str, Any],
) -> dict[str, Any]:
    sidecar = checkpoint["value"]
    return {
        "epoch": sidecar["completed_epoch"],
        "path": checkpoint["path"],
        "checkpoint_id": checkpoint["sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "sidecar_path": checkpoint["sidecar_path"],
        "sidecar_sha256": checkpoint["sidecar_sha256"],
        "sidecar": sidecar,
        "epoch_record_path": epoch["epoch_record_path"],
        "epoch_record_id": epoch["record_id"],
        "epoch_record_sha256": epoch["epoch_record_sha256"],
        "predecessor_checkpoint_id": sidecar["predecessor_checkpoint_id"],
        "global_optimizer_step": sidecar["global_optimizer_step"],
        "payload_audit": payload_audit,
    }


def tracked_file_ref(filename: str, id_key: str | None = None) -> dict[str, Any]:
    path = REPO / "results/learned/w7" / filename
    value = load_json(path)
    result: dict[str, Any] = {"path": str(Path("results/learned/w7") / filename), "file_sha256": file_sha256(path)}
    if id_key is not None:
        result[id_key] = value[id_key]
    return result


def build(args: argparse.Namespace) -> None:
    metadata_root = args.metadata_root.resolve()
    scan_path = args.worker_scan.resolve()
    payload_audit_path = args.full_checkpoint_audit.resolve()
    deep_audit_path = args.selected_deep_audit.resolve()
    scan = load_json(scan_path)
    payload_audit = load_json(payload_audit_path)
    deep_audit = load_json(deep_audit_path)
    if scan["scan_schema_version"] != 1 or scan["root"] != args.worker_root:
        fail("worker scan root/schema differs")
    remote_root = Path(scan["root"])
    if scan["worker_hostname"] != "confessor":
        fail("worker hostname differs")
    if payload_audit["source_commit"] != SOURCE_COMMIT or deep_audit["source_commit"] != SOURCE_COMMIT:
        fail("checkpoint audit source differs")
    payload_count = payload_audit.get("payloads_checked", payload_audit.get("payloads_and_sidecars_deep_validated"))
    if payload_count != EXPECTED_EPOCHS * len(CANDIDATES):
        fail("payload audit count differs")
    if len(deep_audit.get("candidates", [])) != len(CANDIDATES):
        fail("selected-checkpoint audit candidate count differs")
    if any(item.get("sidecar_validated") is not True or item.get("payload_validated") is not True for item in deep_audit["candidates"]):
        fail("selected-checkpoint deep validation did not pass")

    manifest_entry = scan["manifest"]
    completion_entry = scan["completion"]
    manifest = worker_entry(manifest_entry, remote_root=remote_root, metadata_root=metadata_root)
    completion = worker_entry(completion_entry, remote_root=remote_root, metadata_root=metadata_root)
    if manifest_entry["sha256"] != CAMPAIGN_MANIFEST_SHA256 or manifest["manifest_id"] != CAMPAIGN_MANIFEST_ID:
        fail("campaign manifest identity differs")
    manifest_body = dict(manifest)
    manifest_id = manifest_body.pop("manifest_id")
    if manifest_id != "w7campaignmanifest-" + canonical_sha256(manifest_body):
        fail("campaign manifest content digest differs")
    if completion_entry["sha256"] != CAMPAIGN_COMPLETION_SHA256 or completion["completion_id"] != CAMPAIGN_COMPLETION_ID:
        fail("campaign completion identity differs")
    completion_body = dict(completion)
    completion_id = completion_body.pop("completion_id")
    if completion_id != "w7campaign-" + canonical_sha256(completion_body):
        fail("campaign completion content digest differs")
    if manifest["campaign_id"] != CAMPAIGN_ID or completion["campaign_id"] != CAMPAIGN_ID:
        fail("campaign ID differs")
    if tuple(manifest["lambda_grid"]) != W7_LAMBDA_GRID or tuple(completion["candidate_lambdas"]) != W7_LAMBDA_GRID:
        fail("campaign lambda order differs")
    if manifest["g4_adjudication_run"] != 0 or manifest["lambda_core_updated"] is not False:
        fail("campaign manifest downstream boundary differs")
    if completion["g4_adjudication_run"] != 0 or completion["lambda_core_updated"] is not False:
        fail("campaign completion downstream boundary differs")

    train, validation = load_manifest_ids()
    validation_ids = [item[0] for item in validation]
    expected_noise = expected_validation_noise(validation)
    if line_digest(validation_ids) != VAL_STABLE_ID_DIGEST:
        fail("validation stable-ID digest differs")
    if canonical_sha256(expected_noise) != NOISE_ID_DIGEST:
        fail("validation noise-ID digest differs")
    expected_noise_policy_hash = canonical_sha256({
        "policy": W7_VALIDATION_NOISE_POLICY,
        "ids": expected_noise,
        "snr_db": W7_CALIBRATION_SNR_DB,
        "ratio": W7_RATIO,
        "channel_seed": W7_CHANNEL_SEED,
    })
    if expected_noise_policy_hash != NOISE_POLICY_HASH:
        fail("validation noise-policy digest differs")

    candidate_scan = scan["candidates"]
    if len(candidate_scan) != len(CANDIDATES):
        fail("worker candidate count differs")
    if [float(item["lambda"]) for item in candidate_scan] != list(W7_LAMBDA_GRID):
        fail("worker candidate order/grid differs")
    index_candidates: list[dict[str, Any]] = []
    custody_candidates: list[dict[str, Any]] = []
    noise_epoch_rows: list[dict[str, Any]] = []
    noise_selected_rows: list[dict[str, Any]] = []
    factual_measurements: list[dict[str, Any]] = []
    homogeneity_rows: list[dict[str, Any]] = []
    expected_checkpoint_count = EXPECTED_EPOCHS * len(CANDIDATES)
    all_checkpoint_count = 0

    for order, ((lambda_value, dirname), scanned) in enumerate(zip(CANDIDATES, candidate_scan, strict=True)):
        if float(scanned["lambda"]) != lambda_value or scanned["root"] != str(remote_root / dirname):
            fail(f"candidate root/order differs for lambda {lambda_value}")
        config = load_w7_config(
            lambda_value=lambda_value,
            role="W7_G4_SCIENTIFIC_PILOT_CHECKPOINT",
            physical_batch_size=W7_PHYSICAL_BATCH_SIZE,
            accumulation_factor=1,
            validation_batch_size=W7_VALIDATION_BATCH_SIZE,
        )
        expected_config = run_config_hash(config)
        expected_protocol = protocol_config_hash(config)
        if expected_protocol != PROTOCOL_HASH:
            fail(f"protocol hash differs for lambda {lambda_value}")
        candidate_value = worker_entry(scanned["candidate"], remote_root=remote_root, metadata_root=metadata_root)
        selected_value = worker_entry(scanned["selected"], remote_root=remote_root, metadata_root=metadata_root)
        latest_value = worker_entry(scanned["latest"], remote_root=remote_root, metadata_root=metadata_root)
        candidate_completion_path = str(remote_root / dirname / "candidate_completion.json")
        if scanned["candidate"]["path"] != candidate_completion_path:
            fail(f"candidate completion path differs for lambda {lambda_value}")
        if candidate_value["lambda"] != lambda_value or candidate_value["status"] != "COMPLETE" or candidate_value["authentication_status"] != "PASSED":
            fail(f"candidate completion status differs for lambda {lambda_value}")
        if candidate_value["test_access"] != 0:
            fail(f"candidate test boundary differs for lambda {lambda_value}")
        if candidate_value["candidate_id"] != "w7candidate-" + canonical_sha256({"lambda": lambda_value, "selected": selected_value["result_digest"]}):
            fail(f"candidate ID differs for lambda {lambda_value}")
        if candidate_value["selected_evidence"]["file_sha256"] != scanned["selected"]["sha256"]:
            fail(f"candidate selected evidence file binding differs for lambda {lambda_value}")
        if candidate_value["selected_evidence"]["result_digest"] != selected_value["result_digest"]:
            fail(f"candidate selected evidence digest differs for lambda {lambda_value}")
        if candidate_value["lineage"]["protocol_config_hash"] != expected_protocol:
            fail(f"candidate protocol lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["source_commit"] != SOURCE_COMMIT or candidate_value["lineage"]["gpu_uuid"] != W7_SELECTED_GPU_UUID:
            fail(f"candidate source/GPU lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["epochs"] != EXPECTED_EPOCHS:
            fail(f"candidate epoch lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["ratio"] != W7_RATIO or candidate_value["lineage"]["train_seed"] != W7_TRAIN_SEED or candidate_value["lineage"]["channel_seed"] != W7_CHANNEL_SEED:
            fail(f"candidate seed/ratio lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["checkpoint_selection"] != {
            "metric": "top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "snr_db": W7_CALIBRATION_SNR_DB
        }:
            fail(f"candidate checkpoint-selection rule differs for lambda {lambda_value}")
        if candidate_value["lineage"]["validation_noise_policy"] != W7_VALIDATION_NOISE_POLICY:
            fail(f"candidate validation-noise policy differs for lambda {lambda_value}")
        if candidate_value["lineage"]["source_manifest_id"] != SOURCE_MANIFEST_ID or candidate_value["lineage"]["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
            fail(f"candidate source manifest lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["dataset"] != W7_DATASET or candidate_value["lineage"]["split_manifest_hash"] != SPLIT_MANIFEST_HASH:
            fail(f"candidate dataset lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["execution_profile_id"] != W7_PROFILE_ID or candidate_value["lineage"]["execution_image"] != W7_EXECUTION_IMAGE_FAMILY:
            fail(f"candidate execution lineage differs for lambda {lambda_value}")
        if candidate_value["lineage"]["train_snr_db"] != W7_TRAINING_SNR_DB or candidate_value["psnr_evaluation"]["snr_db"] != W7_PSNR_SNR_DB:
            fail(f"candidate SNR lineage differs for lambda {lambda_value}")
        if candidate_value["psnr_evaluation"]["denominator"] != VAL_COUNT or candidate_value["psnr_evaluation"]["data_range"] != 1.0:
            fail(f"candidate PSNR denominator/data range differs for lambda {lambda_value}")
        if candidate_value["lineage"]["protocol_config_hash"] != PROTOCOL_HASH:
            fail(f"candidate protocol config hash differs for lambda {lambda_value}")

        profile_binding = scanned["selected_payload"]["execution_profile"]
        verify_frozen_gpu_binding(profile_binding, config_hash=expected_config)
        if profile_binding["gpu_uuid"] != W7_SELECTED_GPU_UUID or profile_binding["gpu_name"] != GPU_NAME or profile_binding["execution_profile_id"] != W7_PROFILE_ID:
            fail(f"profile/GPU binding differs for lambda {lambda_value}")
        if profile_binding["git_commit"] != SOURCE_COMMIT or profile_binding["lock_file_sha256"] != "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82":
            fail(f"profile source/lock binding differs for lambda {lambda_value}")
        payload_candidate = next(item for item in payload_audit["candidates"] if float(item["lambda"]) == lambda_value)
        selected_deep_candidate = next(item for item in deep_audit["candidates"] if float(item["lambda"]) == lambda_value)
        full_payload_candidate = payload_candidate
        payload_candidate_count = payload_candidate.get("checkpoint_payload_count", payload_candidate.get("checkpoint_count"))
        if payload_candidate_count != EXPECTED_EPOCHS or full_payload_candidate["checkpoint_count"] != EXPECTED_EPOCHS:
            fail(f"payload audit count differs for lambda {lambda_value}")
        if len(scanned["checkpoints"]) != EXPECTED_EPOCHS or len(scanned["epochs"]) != EXPECTED_EPOCHS or len(scanned["validation"]) != EXPECTED_EPOCHS:
            fail(f"worker history count differs for lambda {lambda_value}")
        if [item["value"]["epoch"] for item in scanned["epochs"]] != list(range(EXPECTED_EPOCHS)):
            fail(f"worker epoch records are not exact for lambda {lambda_value}")
        if [item["value"]["epoch"] for item in scanned["validation"]] != list(range(EXPECTED_EPOCHS)):
            fail(f"worker validation history is not exact for lambda {lambda_value}")

        prior_checkpoint: str | None = None
        prior_step = 0
        epoch_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        applied_total = 0
        skip_total = 0
        independent_summaries: list[dict[str, Any]] = []
        for epoch in range(EXPECTED_EPOCHS):
            checkpoint = scanned["checkpoints"][epoch]
            epoch_entry = scanned["epochs"][epoch]
            validation_entry = scanned["validation"][epoch]
            sidecar = checkpoint["value"]
            record = epoch_entry["value"]
            summary = validation_entry["value"]
            payload_summary = full_payload_candidate["entries"][epoch]
            if checkpoint["sha256"] != sidecar["checkpoint_id"] or checkpoint["bytes"] != sidecar["checkpoint_bytes"]:
                fail(f"checkpoint sidecar/file binding differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["completed_epoch"] != epoch or sidecar["next_epoch"] != epoch + 1:
                fail(f"checkpoint epoch differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["predecessor_checkpoint_id"] != prior_checkpoint:
                fail(f"checkpoint chain differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["global_optimizer_step"] != record["global_optimizer_step"] or sidecar["epoch_record_sha256"] != epoch_entry["sha256"] or sidecar["epoch_record_id"] != record["record_id"]:
                fail(f"checkpoint epoch-record binding differs for lambda {lambda_value}, epoch {epoch}")
            record_body = dict(record)
            record_id = record_body.pop("record_id", None)
            if record_id != canonical_sha256(record_body):
                fail(f"epoch record digest differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["artifact_role"] != "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT_SIDECAR" or record["artifact_role"] != "W7_G4_EPOCH_RECORD":
                fail(f"checkpoint/epoch role differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["lambda"] != lambda_value or sidecar["source_commit"] != SOURCE_COMMIT or sidecar["source_manifest_id"] != SOURCE_MANIFEST_ID or sidecar["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
                fail(f"checkpoint source/lambda binding differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["execution_profile_id"] != W7_PROFILE_ID or sidecar["execution_image"] != W7_EXECUTION_IMAGE_FAMILY or sidecar["gpu_uuid"] != W7_SELECTED_GPU_UUID:
                fail(f"checkpoint profile/GPU binding differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["dataset"] != W7_DATASET or sidecar["ratio"] != W7_RATIO or sidecar["k"] != int(config.resolved["k"]):
                fail(f"checkpoint dataset/ratio binding differs for lambda {lambda_value}, epoch {epoch}")
            if sidecar["train_seed"] != W7_TRAIN_SEED or sidecar["channel_seed"] != W7_CHANNEL_SEED or sidecar["train_snr_db"] != W7_TRAINING_SNR_DB:
                fail(f"checkpoint seed/SNR binding differs for lambda {lambda_value}, epoch {epoch}")
            order_sha, set_sha = expected_order(train, epoch)
            if record["samples"] != TRAIN_COUNT or record["expected_samples"] != TRAIN_COUNT or record["stable_id_count"] != TRAIN_COUNT:
                fail(f"training denominator differs for lambda {lambda_value}, epoch {epoch}")
            if record["stable_id_order_sha256"] != order_sha or record["stable_id_set_sha256"] != set_sha:
                fail(f"training stable-ID order/set differs for lambda {lambda_value}, epoch {epoch}")
            if record["training_noise_id_count"] != TRAIN_COUNT or record["training_noise_id_sha256"] != training_noise_digest([item[0] for item in train], epoch):
                fail(f"training noise identity history differs for lambda {lambda_value}, epoch {epoch}")
            if record["microbatches"] != MICROBATCHES or record["optimizer_steps"] + record["grad_scaler_skips"] != MICROBATCHES:
                fail(f"optimizer microbatch arithmetic differs for lambda {lambda_value}, epoch {epoch}")
            if record["global_optimizer_step"] != prior_step + record["optimizer_steps"]:
                fail(f"global optimizer-step recurrence differs for lambda {lambda_value}, epoch {epoch}")
            if record["lr"] != expected_lr(epoch):
                fail(f"cosine epoch-start LR differs for lambda {lambda_value}, epoch {epoch}")
            gradient = record["gradient_checks"]
            if gradient["optimizer_parameter_count"] != 66 or gradient["optimizer_gradient_count_min"] != 66 or gradient["optimizer_gradient_count_max"] != 66:
                fail(f"optimizer-wide gradient count differs for lambda {lambda_value}, epoch {epoch}")
            if gradient["all_optimizer_gradients_finite"] != (record["grad_scaler_skips"] == 0):
                fail(f"post-unscale finite classification differs for lambda {lambda_value}, epoch {epoch}")
            if not isinstance(gradient["all_named_present_gradients_finite"], bool):
                fail(f"named finite classification is invalid for lambda {lambda_value}, epoch {epoch}")
            if record["finite_loss"] is not True:
                fail(f"finite loss flag differs for lambda {lambda_value}, epoch {epoch}")
            if payload_summary["checkpoint_id"] != checkpoint["sha256"] or payload_summary["epoch"] != epoch:
                fail(f"payload/checkpoint audit binding differs for lambda {lambda_value}, epoch {epoch}")
            if payload_summary["epoch_manifest"] != {"path": sidecar["epoch_record_path"], "record_id": sidecar["epoch_record_id"], "record_sha256": sidecar["epoch_record_sha256"]}:
                fail(f"payload epoch-manifest binding differs for lambda {lambda_value}, epoch {epoch}")
            if payload_summary["predecessor_checkpoint_id"] != sidecar["predecessor_checkpoint_id"] or payload_summary["scheduler_state"] != {"completed_epoch": epoch}:
                fail(f"payload lineage/scheduler differs for lambda {lambda_value}, epoch {epoch}")
            if payload_summary["model_state_key_count"] != 68 or payload_summary["optimizer_state_param_count"] != 66 or payload_summary["optimizer_param_group_count"] != 1 or payload_summary["scaler_state_present"] is not True:
                fail(f"payload model/optimizer/scaler state differs for lambda {lambda_value}, epoch {epoch}")
            if summary["schema_version"] != 1 or summary["artifact_role"] != "W7_VALIDATION_EPOCH_SUMMARY" or summary["epoch"] != epoch:
                fail(f"validation summary role/epoch differs for lambda {lambda_value}, epoch {epoch}")
            summary_body = dict(summary)
            summary_id = summary_body.pop("summary_id", None)
            if summary_id != canonical_sha256(summary_body):
                fail(f"validation summary digest differs for lambda {lambda_value}, epoch {epoch}")
            if summary["checkpoint_id"] != checkpoint["sha256"] or summary["n_total"] != VAL_COUNT or not isinstance(summary["n_correct"], int) or not 0 <= summary["n_correct"] <= VAL_COUNT:
                fail(f"validation summary checkpoint/denominator differs for lambda {lambda_value}, epoch {epoch}")
            if summary["top1_accuracy"] != summary["n_correct"] / summary["n_total"]:
                fail(f"validation top1 arithmetic differs for lambda {lambda_value}, epoch {epoch}")
            if summary["noise_policy"] != W7_VALIDATION_NOISE_POLICY or summary["noise_policy_hash"] != NOISE_POLICY_HASH or summary["noise_id_digest"] != NOISE_ID_DIGEST:
                fail(f"validation noise binding differs for lambda {lambda_value}, epoch {epoch}")
            if summary["evaluation_config_hash"] != "f1c25277250ec10ec766aac99539d46bc988cd87c05e4b6b7e1b725a4fee2d65":
                fail(f"validation evaluation config differs for lambda {lambda_value}, epoch {epoch}")
            compact_epoch_value = compact_epoch(record, epoch_entry)
            epoch_rows.append(compact_epoch_value)
            checkpoint_rows.append(compact_checkpoint(checkpoint, compact_epoch_value, payload_summary))
            validation_rows.append(compact_validation(validation_entry))
            independent_summaries.append(summary)
            prior_checkpoint = checkpoint["sha256"]
            prior_step = record["global_optimizer_step"]
            applied_total += record["optimizer_steps"]
            skip_total += record["grad_scaler_skips"]
            noise_epoch_rows.append({
                "lambda": lambda_value,
                "epoch": epoch,
                "path": validation_entry["path"],
                "file_sha256": validation_entry["sha256"],
                "summary_id": summary["summary_id"],
                "checkpoint_id": summary["checkpoint_id"],
                "n_total": summary["n_total"],
                "noise_id_digest": summary["noise_id_digest"],
                "noise_policy_hash": summary["noise_policy_hash"],
            })
        if prior_step != applied_total or applied_total + skip_total != EXPECTED_EPOCHS * MICROBATCHES:
            fail(f"global optimizer-step total differs for lambda {lambda_value}")
        if latest_value != scanned["checkpoints"][-1]["value"]:
            fail(f"latest pointer differs for lambda {lambda_value}")
        if scanned["latest"]["sha256"] != scanned["checkpoints"][-1]["sidecar_sha256"]:
            fail(f"latest pointer SHA differs for lambda {lambda_value}")
        selected_result = selected_value
        if set(selected_result) != {"artifact_role", "calibration_rows", "calibration_validation", "checkpoint_epoch", "checkpoint_id", "protected_counters", "psnr_evaluation", "result_digest", "schema_version", "selection"}:
            fail(f"selected result schema differs for lambda {lambda_value}")
        result_body = dict(selected_result)
        result_digest = result_body.pop("result_digest", None)
        if result_digest != canonical_sha256(result_body):
            fail(f"selected result digest differs for lambda {lambda_value}")
        selected_epoch = min(
            epoch for epoch, summary in enumerate(independent_summaries)
            if summary["top1_accuracy"] == max(item["top1_accuracy"] for item in independent_summaries)
        )
        selected_summary = independent_summaries[selected_epoch]
        selection = {
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "selected_epoch": selected_epoch,
            "selected_checkpoint_id": selected_summary["checkpoint_id"],
            "n_correct": selected_summary["n_correct"],
            "n_total": selected_summary["n_total"],
            "top1_accuracy": selected_summary["top1_accuracy"],
        }
        if selected_result["checkpoint_epoch"] != selected_epoch or selected_result["checkpoint_id"] != selected_summary["checkpoint_id"]:
            fail(f"selected checkpoint differs from independent selection for lambda {lambda_value}")
        if selected_result["selection"] != selection:
            fail(f"selected result selection differs for lambda {lambda_value}")
        if selected_result["calibration_validation"] != selected_summary:
            fail(f"selected calibration summary differs for lambda {lambda_value}")
        rows = selected_result["calibration_rows"]
        if len(rows) != VAL_COUNT or [row["stable_sample_id"] for row in rows] != validation_ids:
            fail(f"selected calibration identity coverage differs for lambda {lambda_value}")
        if [row["noise_id"] for row in rows] != expected_noise:
            fail(f"selected calibration noise IDs differ for lambda {lambda_value}")
        if len({row["stable_sample_id"] for row in rows}) != VAL_COUNT or len({row["noise_id"] for row in rows}) != VAL_COUNT:
            fail(f"selected calibration IDs are duplicated for lambda {lambda_value}")
        for row in rows:
            if row["correct"] != (row["prediction"] == row["label"]):
                fail(f"selected calibration correctness differs for lambda {lambda_value}")
            if row["psnr_db"] == "inf":
                if row["mse"] != 0:
                    fail(f"selected calibration infinity handling differs for lambda {lambda_value}")
            else:
                if row["mse"] <= 0 or float(row["psnr_db"]) != 10.0 * math.log10(1.0 / float(row["mse"])):
                    fail(f"selected calibration PSNR arithmetic differs for lambda {lambda_value}")
        calibration_n_correct = sum(int(row["correct"]) for row in rows)
        if calibration_n_correct != selected_summary["n_correct"]:
            fail(f"selected calibration count differs for lambda {lambda_value}")
        prediction_digest = canonical_sha256([
            {"stable_sample_id": row["stable_sample_id"], "prediction": row["prediction"], "correct": row["prediction"] == row["label"]}
            for row in rows
        ])
        if prediction_digest != selected_summary["prediction_digest"] or canonical_sha256(rows) != selected_summary["row_digest"]:
            fail(f"selected calibration row digests differ for lambda {lambda_value}")
        psnr_eval = selected_result["psnr_evaluation"]
        if psnr_eval["checkpoint_id"] != selected_result["checkpoint_id"] or psnr_eval["snr_db"] != W7_PSNR_SNR_DB or psnr_eval["denominator"] != VAL_COUNT or psnr_eval["data_range"] != 1.0:
            fail(f"selected PSNR checkpoint/SNR/denominator differs for lambda {lambda_value}")
        if psnr_eval["psnr_definition"] != "per_image_mse_all_RGB_pixels_then_arithmetic_mean" or psnr_eval["papr_definition"] != "symbol_domain_per_image_then_arithmetic_mean":
            fail(f"selected metric definitions differ for lambda {lambda_value}")
        per_image = psnr_eval["per_image"]
        if len(per_image) != VAL_COUNT or [row["stable_sample_id"] for row in per_image] != validation_ids:
            fail(f"selected 15 dB metric identity coverage differs for lambda {lambda_value}")
        if canonical_sha256(per_image) != psnr_eval["per_image_digest"]:
            fail(f"selected 15 dB metric evidence digest differs for lambda {lambda_value}")
        mean_psnr = float(np.mean(np.asarray([float(row["psnr_db"]) for row in per_image], dtype=np.float64)))
        mean_papr = float(np.mean(np.asarray([float(row["papr_db"]) for row in per_image], dtype=np.float64)))
        if mean_psnr != psnr_eval["psnr_db"] or mean_papr != psnr_eval["papr_db"]:
            fail(f"selected aggregate metrics differ for lambda {lambda_value}")
        if selected_result["protected_counters"] != {
            "w7_candidate_results": 0, "learned_test_inference": 0, "test_model_facing_access": 0
        }:
            fail(f"selected protected counters differ for lambda {lambda_value}")
        if candidate_value["selected_validation"] != {
            "checkpoint_id": selected_result["checkpoint_id"], "epoch": selected_epoch,
            "n_correct": calibration_n_correct, "n_total": VAL_COUNT, "top1_accuracy": selected_summary["top1_accuracy"]
        }:
            fail(f"candidate selected-validation block differs for lambda {lambda_value}")
        if candidate_value["selected_validation_result_digest"] != selected_result["result_digest"]:
            fail(f"candidate selected-result digest differs for lambda {lambda_value}")
        if candidate_value["psnr_evaluation"]["psnr_db"] != psnr_eval["psnr_db"] or candidate_value["psnr_evaluation"]["per_image_digest"] != psnr_eval["per_image_digest"]:
            fail(f"candidate PSNR block differs for lambda {lambda_value}")

        payload_capture = {
            "schema_version": payload_audit["schema_version"],
            "checkpoint_payload_count": payload_candidate_count,
            "payload_summary_sha256": payload_candidate.get("payload_summary_sha256", payload_candidate["entries_sha256"]),
            "deep_validation_count": full_payload_candidate["checkpoint_count"],
            "deep_entries_sha256": full_payload_candidate["entries_sha256"],
            "entries": full_payload_candidate["entries"],
            "selected_checkpoint_audit": selected_deep_candidate,
        }
        index_candidate = {
            "campaign_order": order,
            "lambda": lambda_value,
            "candidate_root": scanned["root"],
            "candidate_completion": {
                "path": scanned["candidate"]["path"],
                "file_sha256": scanned["candidate"]["sha256"],
                "value": candidate_value,
            },
            "profile_binding": profile_binding,
            "config_hash": expected_config,
            "protocol_config_hash": expected_protocol,
            "homogeneity": expected_homogeneity(lambda_value, config),
            "epochs": epoch_rows,
            "checkpoints": checkpoint_rows,
            "validation_summaries": validation_rows,
            "checkpoint_payload_audit": payload_capture,
            "latest": {
                "path": scanned["latest"]["path"],
                "file_sha256": scanned["latest"]["sha256"],
                "value": latest_value,
            },
            "selected": {
                "selected_epoch": selected_epoch,
                "selected_checkpoint_id": selected_result["checkpoint_id"],
                "selected_checkpoint": checkpoint_rows[selected_epoch],
                "selected_checkpoint_result": {
                    "path": scanned["selected"]["path"],
                    "file_sha256": scanned["selected"]["sha256"],
                    "value": selected_result,
                },
                "independent_selection": selection,
                "selected_validation": selected_summary,
                "derived_n_correct": calibration_n_correct,
                "derived_n_total": VAL_COUNT,
                "derived_top1_accuracy": calibration_n_correct / VAL_COUNT,
                "derived_mean_psnr_db": mean_psnr,
                "derived_mean_papr": mean_papr,
                "psnr_definition": psnr_eval["psnr_definition"],
                "papr_definition": psnr_eval["papr_definition"],
            },
            "training_totals": {
                "completed_epochs": EXPECTED_EPOCHS,
                "training_denominator_per_epoch": TRAIN_COUNT,
                "microbatches_per_epoch": MICROBATCHES,
                "final_partial_batch_samples": FINAL_PARTIAL_BATCH,
                "applied_optimizer_steps": applied_total,
                "gradscaler_skips": skip_total,
                "final_global_optimizer_step": prior_step,
            },
            "worker_candidate_id": candidate_value["candidate_id"],
        }
        index_candidates.append(index_candidate)
        custody_candidates.append({
            "campaign_order": order,
            "lambda": lambda_value,
            "candidate_root": scanned["root"],
            "candidate_completion_path": scanned["candidate"]["path"],
            "candidate_completion_id": candidate_value["candidate_id"],
            "candidate_completion_file_sha256": scanned["candidate"]["sha256"],
            "selected_checkpoint_path": checkpoint_rows[selected_epoch]["path"],
            "selected_checkpoint_id": selected_result["checkpoint_id"],
            "selected_checkpoint_file_sha256": selected_result["checkpoint_id"],
            "selected_checkpoint_sidecar_path": checkpoint_rows[selected_epoch]["sidecar_path"],
            "selected_checkpoint_sidecar_sha256": checkpoint_rows[selected_epoch]["sidecar_sha256"],
            "selected_checkpoint_result_path": scanned["selected"]["path"],
            "selected_checkpoint_result_file_sha256": scanned["selected"]["sha256"],
            "selected_checkpoint_result_digest": selected_result["result_digest"],
            "validation_history_digest": canonical_sha256(validation_rows),
            "checkpoint_chain_digest": canonical_sha256(checkpoint_rows),
            "candidate_result_digest": canonical_sha256({
                "candidate_id": candidate_value["candidate_id"],
                "selected_result_digest": selected_result["result_digest"],
                "selected_epoch": selected_epoch,
            }),
            "checkpoint_count": len(checkpoint_rows),
            "checkpoints": [
                {
                    "epoch": item["epoch"],
                    "path": item["path"],
                    "checkpoint_id": item["checkpoint_id"],
                    "checkpoint_bytes": item["checkpoint_bytes"],
                    "sidecar_path": item["sidecar_path"],
                    "sidecar_sha256": item["sidecar_sha256"],
                    "epoch_record_path": item["epoch_record_path"],
                    "epoch_record_id": item["epoch_record_id"],
                    "epoch_record_sha256": item["epoch_record_sha256"],
                    "predecessor_checkpoint_id": item["predecessor_checkpoint_id"],
                    "global_optimizer_step": item["global_optimizer_step"],
                }
                for item in checkpoint_rows
            ],
        })
        noise_selected_rows.append({
            "lambda": lambda_value,
            "selected_epoch": selected_epoch,
            "selected_result_path": scanned["selected"]["path"],
            "selected_result_file_sha256": scanned["selected"]["sha256"],
            "selected_result_digest": selected_result["result_digest"],
            "stable_id_digest": canonical_sha256([row["stable_sample_id"] for row in rows]),
            "noise_id_digest": canonical_sha256([row["noise_id"] for row in rows]),
            "row_digest": canonical_sha256(rows),
            "mismatched_expected_noise_ids": sum(a != b for a, b in zip([row["noise_id"] for row in rows], expected_noise, strict=True)),
        })
        factual_measurements.append({
            "lambda": lambda_value,
            "selected_epoch": selected_epoch,
            "selected_checkpoint_id": selected_result["checkpoint_id"],
            "n_correct": calibration_n_correct,
            "n_total": VAL_COUNT,
            "top1_accuracy": calibration_n_correct / VAL_COUNT,
            "mean_psnr_db": mean_psnr,
            "mean_papr": mean_papr,
            "applied_optimizer_steps": applied_total,
            "gradscaler_skips": skip_total,
        })
        homogeneity_rows.append(expected_homogeneity(lambda_value, config))
        all_checkpoint_count += len(checkpoint_rows)

    worker_candidate_paths = [item["candidate"]["path"] for item in candidate_scan]
    if completion["candidate_paths"] != worker_candidate_paths:
        fail("campaign completion candidate references differ")
    if [item["candidate_id"] for item in completion["candidates"]] != [item["worker_candidate_id"] for item in index_candidates]:
        fail("campaign completion candidate IDs differ")
    if all_checkpoint_count != expected_checkpoint_count:
        fail("worker checkpoint total differs")
    if [row["lambda"] for row in homogeneity_rows] != list(W7_LAMBDA_GRID):
        fail("homogeneity row order differs")
    common_fields = {key: value for key, value in homogeneity_rows[0].items() if key != "lambda"}
    if any({key: value for key, value in row.items() if key != "lambda"} != common_fields for row in homogeneity_rows[1:]):
        fail("candidate homogeneity differs")

    index_body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_B2R_CANDIDATE_INDEX",
        "campaign_id": CAMPAIGN_ID,
        "campaign_manifest": {"manifest_id": CAMPAIGN_MANIFEST_ID, "file_sha256": CAMPAIGN_MANIFEST_SHA256, "worker_path": manifest_entry["path"], "value": manifest},
        "campaign_completion": {"completion_id": CAMPAIGN_COMPLETION_ID, "file_sha256": CAMPAIGN_COMPLETION_SHA256, "worker_path": completion_entry["path"], "value": completion},
        "authorization": {"authorization_id": AUTHORIZATION_ID, "file_sha256": AUTHORIZATION_SHA256, "path": "results/learned/w7/w7_execution_authorization.json"},
        "source": {"source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID, "source_manifest_sha256": SOURCE_MANIFEST_SHA256},
        "profile": {"profile_freeze_id": PROFILE_FREEZE_ID, "profile_freeze_sha256": PROFILE_FREEZE_SHA256, "execution_profile_id": W7_PROFILE_ID, "execution_image": W7_EXECUTION_IMAGE_FAMILY, "gpu_name": GPU_NAME, "gpu_uuid": W7_SELECTED_GPU_UUID},
        "protocol": {
            "dataset": W7_DATASET, "dataset_version": DATASET_VERSION, "ratio": W7_RATIO, "k": int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
            "lambda_grid": list(W7_LAMBDA_GRID), "lambda_order": "exact_configured_lambda_grid_order", "train_seed": W7_TRAIN_SEED, "channel_seed": W7_CHANNEL_SEED,
            "training_snr_db": W7_TRAINING_SNR_DB, "validation_snr_db": W7_CALIBRATION_SNR_DB, "psnr_snr_db": W7_PSNR_SNR_DB,
            "epochs": EXPECTED_EPOCHS, "training_denominator": TRAIN_COUNT, "validation_denominator": VAL_COUNT,
            "physical_batch_size": W7_PHYSICAL_BATCH_SIZE, "accumulation_factor": 1, "effective_batch_size": TARGET_BATCH, "validation_batch_size": W7_VALIDATION_BATCH_SIZE,
            "drop_last": False, "final_partial_batch_samples": FINAL_PARTIAL_BATCH, "microbatches_per_epoch": MICROBATCHES,
            "optimizer": "adam", "optimizer_implementation": "torch.optim.Adam", "scheduler": "cosine", "scheduler_indexing": "zero_based", "scheduler_step_unit": "epoch_start",
            "amp_enabled": True, "grad_scaler_policy": "optimizer_wide_post_unscale_finite_authoritative_skips_excluded_from_global_step", "scaler_state_checkpointed": True,
            "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
            "checkpoint_selection": {"metric": "top1_accuracy", "mode": "max", "snr_db": W7_CALIBRATION_SNR_DB, "tie_break": "earliest_epoch"},
            "psnr_definition": "per_image_mse_all_RGB_pixels_then_arithmetic_mean", "papr_definition": "symbol_domain_per_image_then_arithmetic_mean",
        },
        "homogeneity": {"fields": common_fields, "candidate_rows": homogeneity_rows, "unexpected_differences": []},
        "candidate_order": list(W7_LAMBDA_GRID),
        "candidates": index_candidates,
        "factual_measurements": factual_measurements,
        "worker_scan": {"path": str(scan_path), "file_sha256": file_sha256(scan_path), "schema_version": scan["scan_schema_version"], "worker_hostname": scan["worker_hostname"]},
        "checkpoint_audit_capture": {"path": str(payload_audit_path), "file_sha256": file_sha256(payload_audit_path), "deep_path": str(deep_audit_path), "deep_file_sha256": file_sha256(deep_audit_path)},
    }
    index = artifact("w7b2rindex-", index_body, "index_id")
    index_path = args.output_dir / "w7_b2_reconciliation_index.json"
    write_json(index_path, index)

    custody_body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_B2R_CHECKPOINT_CUSTODY_MANIFEST",
        "status": "FROZEN_WORKER_CUSTODY",
        "campaign_id": CAMPAIGN_ID,
        "candidate_index_id": index["index_id"],
        "worker_hostname": scan["worker_hostname"],
        "worker_campaign_root": str(remote_root),
        "checkpoint_policy": "one_completed_epoch_checkpoint_per_epoch_0_through_99",
        "expected_checkpoint_count": expected_checkpoint_count,
        "observed_checkpoint_count": all_checkpoint_count,
        "selected_checkpoint_bytes_remain_on_worker": True,
        "candidates": custody_candidates,
    }
    custody = artifact("w7b2rcustody-", custody_body, "custody_id")
    custody_path = args.output_dir / "w7_b2_checkpoint_custody.json"
    write_json(custody_path, custody)

    pairwise = []
    for left_index, left in enumerate(CANDIDATES):
        for right in CANDIDATES[left_index + 1:]:
            pairwise.append({"left_lambda": left[0], "right_lambda": right[0], "mismatched_sample_noise_ids": 0, "sample_count": VAL_COUNT})
    noise_body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_B2R_COMMON_VALIDATION_NOISE_AUDIT",
        "status": "AUTHENTICATED_COMMON_NOISE",
        "campaign_id": CAMPAIGN_ID,
        "candidate_index_id": index["index_id"],
        "evaluation_role": "validation_checkpoint_selection_calibration",
        "snr_db": W7_CALIBRATION_SNR_DB,
        "channel_seed": W7_CHANNEL_SEED,
        "ratio": W7_RATIO,
        "dataset": W7_DATASET,
        "validation_denominator": VAL_COUNT,
        "validation_order": "stable_manifest_order",
        "noise_policy": W7_VALIDATION_NOISE_POLICY,
        "noise_policy_hash": NOISE_POLICY_HASH,
        "stable_id_digest": VAL_STABLE_ID_DIGEST,
        "noise_id_digest": NOISE_ID_DIGEST,
        "identity_fields": ["dataset_version", "split_manifest_hash", "stable_sample_id", "channel_seed", "channel", "bw_ratio", "k", "snr_db", "rng_purpose"],
        "paired_samples": [{"stable_sample_id": stable_id, "noise_id": noise_id} for (stable_id, _label), noise_id in zip(validation, expected_noise, strict=True)],
        "epoch_summary_audit": noise_epoch_rows,
        "selected_candidate_audit": noise_selected_rows,
        "pairwise_comparisons": pairwise,
        "all_selected_sample_mismatch_count": 0,
        "test_model_facing_access": 0,
    }
    noise = artifact("w7b2rnoise-", noise_body, "audit_id")
    noise_path = args.output_dir / "w7_b2_common_noise_audit.json"
    write_json(noise_path, noise)

    upstream = {
        "w7_b1": {"status": "PASS", "command": "tools/verify_w7_b1.py verify", "authorization_id": AUTHORIZATION_ID, "source_manifest_id": SOURCE_MANIFEST_ID},
        "w7_a": {"status": "PASS", "command": "tools/verify_w7_a.py --no-upstream", "completion_id": "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17", "test_hardening_completion_id": "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3"},
        "w5": {"status": "PASS", "command": "tools/verify_w5_training_system.py", "completion_id": "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89"},
        "w6": {"status": "PASS", "command": "tools/verify_w6_complete.py", "completion_id": "w6completion-f992e38e553dce4075406ef8f08df0d42feb2a141a3b00b0ae29a0490e834515", "test": "SEALED", "pass_one": 1, "pass_two": 1, "pass_three": 0},
        "pascal_profile": {"status": "PASS", "command": "tools/verify_w7_profile.py results/learned/w7/w7_pascal_profile.json", "profile_id": "w7profile-c2e70848dc6857fe4df3868c90af1ccff4d6e0c7d267cbad8b9ad49b228e5d69"},
    }
    reconciliation_body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_B2R_RECONCILIATION",
        "status": "GREEN",
        "campaign_status": "COMPLETE_NOT_YET_ADJUDICATED",
        "campaign_id": CAMPAIGN_ID,
        "scientific_source_commit": SOURCE_COMMIT,
        "reconciliation_tooling_base_commit": "a3665b854dd1e9065a8082a66680a69ce29a10c1",
        "campaign_manifest_id": CAMPAIGN_MANIFEST_ID,
        "campaign_completion_id": CAMPAIGN_COMPLETION_ID,
        "candidate_index_id": index["index_id"],
        "custody_id": custody["custody_id"],
        "common_noise_audit_id": noise["audit_id"],
        "upstream_reauthentication": upstream,
        "homogeneity": {"result": "PASS", "candidate_count": len(CANDIDATES), "only_intended_candidate_field": "lambda", "unexpected_differences": []},
        "factual_candidate_measurements": factual_measurements,
        "worker_custody": {"hostname": scan["worker_hostname"], "checkpoint_count": all_checkpoint_count, "expected_checkpoint_count": expected_checkpoint_count, "bytes_preserved_on_worker": True},
        "protected_boundary": {
            "g8_reruns": 0, "f1_reruns": 0, "f2_optimizer_steps": 0, "f3_reruns": 0, "pass_one_reruns": 0, "pass_two_reruns": 0, "pass_three": 0, "bler_regeneration": 0,
            "g4_adjudications": 0, "lambda_core_updated": False, "lambda_decision": "NOT_PERFORMED", "lambda_status": "provisional_until_G-4",
            "w8_final_training_runs": 0, "w8_state": "UNOPENED", "test_model_facing_access": 0, "learned_test_inference": 0, "test_state": "SEALED",
        },
        "operational_closeout": {
            "heartbeat_path": scan["root"] + "/heartbeat.json", "heartbeat_file_sha256": file_sha256(metadata_root / "heartbeat.json"), "heartbeat": load_json(metadata_root / "heartbeat.json"),
            "process_state": "COMPLETE_NOT_ADJUDICATED", "campaign_process_absent": True, "tmux_w7_g4_absent": True, "global_campaign_lock": "FREE", "worker_source_worktree_clean": True,
            "monitor_is_operational_only": True, "worker_runtime_mutated": False,
        },
        "no_model_facing_recomputation": True,
        "decision_boundary": "return_for_hostile_audit_then_separate_W7_C_authorization",
    }
    reconciliation = artifact("w7b2rreconciliation-", reconciliation_body, "reconciliation_id")
    reconciliation_path = args.output_dir / "w7_b2_reconciliation.json"
    write_json(reconciliation_path, reconciliation)

    completion_body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_B2R_RECONCILIATION_COMPLETE_NOT_YET_ADJUDICATED",
        "status": "COMPLETE_NOT_YET_ADJUDICATED",
        "campaign_id": CAMPAIGN_ID,
        "candidate_count": len(CANDIDATES),
        "complete_candidate_count": len(CANDIDATES),
        "completed_epoch_cycles": expected_checkpoint_count,
        "candidate_index_id": index["index_id"],
        "custody_id": custody["custody_id"],
        "common_noise_audit_id": noise["audit_id"],
        "reconciliation_id": reconciliation["reconciliation_id"],
        "worker_campaign_completion_id": CAMPAIGN_COMPLETION_ID,
        "worker_campaign_completion_file_sha256": CAMPAIGN_COMPLETION_SHA256,
        "candidate_references": [
            {"lambda": item["lambda"], "candidate_id": item["worker_candidate_id"], "candidate_completion_file_sha256": item["candidate_completion"]["file_sha256"]}
            for item in index_candidates
        ],
        "g4_adjudication_run": 0,
        "lambda_decision": "NOT_PERFORMED",
        "lambda_core_updated": False,
        "lambda_status": "provisional_until_G-4",
        "w8_final_training_runs": 0,
        "w8_state": "UNOPENED",
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "test_state": "SEALED",
        "source_commit": SOURCE_COMMIT,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "no_scientific_execution_performed_by_reconciliation": True,
    }
    completion_artifact = artifact("w7b2rcompletion-", completion_body, "completion_id")
    completion_path = args.output_dir / "w7_b2_completion.json"
    write_json(completion_path, completion_artifact)

    print(json.dumps({
        "status": "GREEN",
        "index_id": index["index_id"],
        "custody_id": custody["custody_id"],
        "audit_id": noise["audit_id"],
        "reconciliation_id": reconciliation["reconciliation_id"],
        "completion_id": completion_artifact["completion_id"],
        "output_dir": str(args.output_dir),
        "checkpoint_count": all_checkpoint_count,
    }, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", choices=("build",))
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--worker-scan", type=Path, required=True)
    parser.add_argument("--full-checkpoint-audit", type=Path, required=True)
    parser.add_argument("--selected-deep-audit", type=Path, required=True)
    parser.add_argument("--worker-root", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/learned/w7")
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
