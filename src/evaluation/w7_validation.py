"""Validation, checkpoint-selection and final metric evaluation for W7.

This module has no test-split path.  Its validation rows are deliberately
small enough to retain for a selected checkpoint while epoch selection uses
only count-derived compact summaries.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from channels.awgn import keyed_complex_noise
from config.params import get
from data.djscc_validation import ValidationDJSCCDataset, validation_noise_id
from data.preprocessing import clip_reconstruction_for_metrics, reconstruction_psnr
from training.deterministic_core import canonical_bytes, canonical_sha256
from training.w7_g4 import W7_CHECKPOINT_ROLE, W7Trainer, W7Hold
from training.w7_protocol import (
    W7_CALIBRATION_SNR_DB,
    W7_DATASET,
    W7_PSNR_SNR_DB,
    W7_VALIDATION_BATCH_SIZE,
    W7_VALIDATION_NOISE_POLICY,
    W7_VALIDATION_ORDER,
)


PSNR_INFINITY_TOKEN = "inf"
VALIDATION_SUMMARY_SCHEMA_VERSION = 1
SELECTED_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidationEvaluation:
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _finite_float(value: object, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise W7Hold(f"{label} is non-finite")
    return converted


def _canonical_metric_values(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> tuple[float, float | str]:
    """Use the project PSNR implementation and expose its finite JSON form."""

    target_hwc = np.transpose(target, (1, 2, 0))
    reconstruction_hwc = np.transpose(reconstruction, (1, 2, 0))
    psnr_value = reconstruction_psnr(target_hwc, reconstruction_hwc)
    psnr: float | str = (
        PSNR_INFINITY_TOKEN if math.isinf(psnr_value) else float(psnr_value)
    )
    clipped = clip_reconstruction_for_metrics(reconstruction_hwc)
    mse = _finite_float(
        np.mean(np.square(target_hwc - clipped), dtype=np.float64), "validation MSE"
    )
    return mse, psnr


def _mean_metric(values: Sequence[float | str]) -> float | str:
    if any(value == PSNR_INFINITY_TOKEN for value in values):
        return PSNR_INFINITY_TOKEN
    return float(np.mean(np.asarray([float(value) for value in values], dtype=np.float64)))


def _validation_noise_ids(trainer: W7Trainer, stable_ids: Sequence[str], snr_db: int | float) -> list[str]:
    resolved = trainer.config.resolved
    return [
        validation_noise_id(
            stable_sample_id=stable_id,
            dataset_version=str(resolved["dataset_version"]),
            split_manifest_hash=str(get(f"datasets.{resolved['dataset']}.manifest_sha256")),
            channel_seed=int(resolved["channel_seed"]),
            channel=str(resolved["channel"]),
            ratio=str(resolved["bw_ratio"]),
            k=int(resolved["k"]),
            snr_db=snr_db,
        )
        for stable_id in stable_ids
    ]


def evaluate_validation(
    trainer: W7Trainer,
    *,
    checkpoint_id: str,
    snr_db: int | float = W7_CALIBRATION_SNR_DB,
    batch_size: int | None = None,
    repo_root=None,
    retain_rows: bool = True,
) -> ValidationEvaluation:
    """Evaluate the complete committed validation split at one keyed SNR."""

    if trainer.policy.role != W7_CHECKPOINT_ROLE or not trainer.policy.validation_enabled:
        raise W7Hold("profile/non-scientific artifacts cannot enter W7 validation")
    if trainer.completed_epoch < 0:
        raise W7Hold("validation requires a completed checkpoint epoch")
    if snr_db != W7_CALIBRATION_SNR_DB:
        raise W7Hold("checkpoint selection validation SNR differs from frozen 7 dB")
    chosen_batch = W7_VALIDATION_BATCH_SIZE if batch_size is None else int(batch_size)
    if chosen_batch <= 0:
        raise W7Hold("validation batch size must be positive")
    dataset = ValidationDJSCCDataset(W7_DATASET, repo_root=repo_root)
    expected_total = int(get(f"datasets.{W7_DATASET}.val_images"))
    if len(dataset) != expected_total:
        raise W7Hold("validation denominator differs from committed manifest")
    loader = DataLoader(
        dataset,
        batch_size=chosen_batch,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=bool(trainer.device.type == "cuda"),
    )
    trainer.model.eval()
    correct = 0
    total = 0
    all_ids: list[str] = []
    all_predictions: list[int] = []
    all_noise_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels, stable_ids in loader:
            ids = [str(value) for value in stable_ids]
            noise_ids = _validation_noise_ids(trainer, ids, snr_db)
            inputs_device = inputs.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            labels_device = labels.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            unit_noise = keyed_complex_noise(
                noise_ids,
                int(trainer.config.resolved["k"]),
                dtype=torch.complex64,
                device=trainer.device,
            )
            output = trainer.model(
                inputs_device,
                snr_db,
                unit_noise=unit_noise,
            )
            predictions = output.logits.argmax(dim=1)
            reconstruction = output.reconstruction.detach().float().cpu().numpy()
            targets = inputs.detach().float().cpu().numpy()
            papr = output.papr_db.detach().float().cpu().numpy()
            for index, stable_id in enumerate(ids):
                prediction = int(predictions[index].item())
                label = int(labels[index].item())
                is_correct = prediction == label
                correct += int(is_correct)
                total += 1
                target = targets[index]
                reconstructed = reconstruction[index]
                mse, psnr = _canonical_metric_values(target, reconstructed)
                papr_value = _finite_float(papr[index], "validation PAPR")
                all_ids.append(stable_id)
                all_predictions.append(prediction)
                all_noise_ids.append(noise_ids[index])
                rows.append(
                    {
                        "stable_sample_id": stable_id,
                        "label": label,
                        "prediction": prediction,
                        "correct": is_correct,
                        "noise_id": noise_ids[index],
                        "mse": mse,
                        "psnr_db": psnr,
                        "papr_db": papr_value,
                    }
                )
    if total != expected_total or len(all_ids) != expected_total:
        raise W7Hold("validation did not process the complete denominator")
    if all_ids != sorted(all_ids) or len(set(all_ids)) != len(all_ids):
        raise W7Hold("validation order or identity coverage differs")
    if len(set(all_noise_ids)) != len(all_noise_ids):
        raise W7Hold("validation noise identities are duplicated")
    top1 = correct / total
    prediction_digest = canonical_sha256(
        [
            {"stable_sample_id": stable_id, "prediction": prediction, "correct": prediction == row["label"]}
            for stable_id, prediction, row in zip(all_ids, all_predictions, rows)
        ]
    )
    noise_policy_hash = canonical_sha256(
        {
            "policy": W7_VALIDATION_NOISE_POLICY,
            "ids": all_noise_ids,
            "snr_db": snr_db,
            "ratio": trainer.config.resolved["bw_ratio"],
            "channel_seed": trainer.config.resolved["channel_seed"],
        }
    )
    summary = {
        "schema_version": VALIDATION_SUMMARY_SCHEMA_VERSION,
        "artifact_role": "W7_VALIDATION_EPOCH_SUMMARY",
        "epoch": trainer.completed_epoch,
        "checkpoint_id": checkpoint_id,
        "n_correct": correct,
        "n_total": total,
        "top1_accuracy": top1,
        "prediction_digest": prediction_digest,
        "evaluation_config_hash": canonical_sha256(
            {
                "protocol_config_hash": trainer.protocol_hash,
                "snr_db": snr_db,
                "batch_size": chosen_batch,
                "order": W7_VALIDATION_ORDER,
            }
        ),
        "noise_policy": W7_VALIDATION_NOISE_POLICY,
        "noise_policy_hash": noise_policy_hash,
        "noise_id_digest": canonical_sha256(all_noise_ids),
        "row_digest": canonical_sha256(rows),
    }
    summary["summary_id"] = canonical_sha256(summary)
    return ValidationEvaluation(summary=summary, rows=tuple(rows) if retain_rows else tuple())


def select_checkpoint_epoch(summaries: Sequence[Mapping[str, Any]], *, expected_epochs: int) -> dict[str, Any]:
    """Apply max top-1 then earliest-epoch tie-breaking, independent of order."""

    if len(summaries) != expected_epochs:
        raise W7Hold("checkpoint-selection validation history is incomplete")
    by_epoch: dict[int, Mapping[str, Any]] = {}
    for summary in summaries:
        if set(summary) != {
            "schema_version", "artifact_role", "epoch", "checkpoint_id", "n_correct",
            "n_total", "top1_accuracy", "prediction_digest", "evaluation_config_hash",
            "noise_policy", "noise_policy_hash", "noise_id_digest", "row_digest", "summary_id",
        }:
            raise W7Hold("validation epoch summary schema differs")
        summary_body = dict(summary)
        summary_id = summary_body.pop("summary_id", None)
        if summary_id != canonical_sha256(summary_body):
            raise W7Hold("validation epoch summary digest differs")
        epoch = summary["epoch"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch in by_epoch:
            raise W7Hold("validation epoch history has duplicate/invalid epoch")
        if summary["artifact_role"] != "W7_VALIDATION_EPOCH_SUMMARY":
            raise W7Hold("validation summary role differs")
        expected_total = int(get(f"datasets.{W7_DATASET}.val_images"))
        if summary["n_total"] != expected_total:
            raise W7Hold("validation denominator differs from the committed split")
        if not isinstance(summary["n_correct"], int) or isinstance(summary["n_correct"], bool) or not 0 <= summary["n_correct"] <= expected_total:
            raise W7Hold("validation correct-count is invalid")
        if summary["top1_accuracy"] != summary["n_correct"] / summary["n_total"]:
            raise W7Hold("validation top-1 is not count-derived")
        by_epoch[epoch] = summary
    if sorted(by_epoch) != list(range(expected_epochs)):
        raise W7Hold("validation epoch history is not an exact prefix")
    maximum = max(float(summary["top1_accuracy"]) for summary in by_epoch.values())
    if not math.isfinite(maximum):
        raise W7Hold("validation top-1 is non-finite")
    selected_epoch = min(epoch for epoch, summary in by_epoch.items() if float(summary["top1_accuracy"]) == maximum)
    selected = dict(by_epoch[selected_epoch])
    return {
        "metric": "top1_accuracy",
        "mode": "max",
        "tie_break": "earliest_epoch",
        "selected_epoch": selected_epoch,
        "selected_checkpoint_id": selected["checkpoint_id"],
        "n_correct": selected["n_correct"],
        "n_total": selected["n_total"],
        "top1_accuracy": selected["top1_accuracy"],
    }


def selected_checkpoint_result(
    trainer: W7Trainer,
    *,
    selection: Mapping[str, Any],
    repo_root=None,
) -> dict[str, Any]:
    """Independently reload the selected checkpoint and retain final evidence."""

    epoch = int(selection["selected_epoch"])
    sidecar = trainer.load_checkpoint_epoch(epoch)
    if sidecar["checkpoint_id"] != selection["selected_checkpoint_id"]:
        raise W7Hold("selected checkpoint ID differs from validation selection")
    calibration = evaluate_validation(
        trainer,
        checkpoint_id=sidecar["checkpoint_id"],
        snr_db=W7_CALIBRATION_SNR_DB,
        repo_root=repo_root,
        retain_rows=True,
    )
    if calibration.summary["n_correct"] != selection["n_correct"] or calibration.summary["top1_accuracy"] != selection["top1_accuracy"]:
        raise W7Hold("selected checkpoint validation reauthentication differs")
    psnr_eval = evaluate_reconstruction_metrics(trainer, sidecar["checkpoint_id"], repo_root=repo_root)
    base = {
        "schema_version": SELECTED_VALIDATION_SCHEMA_VERSION,
        "artifact_role": "W7_SELECTED_CHECKPOINT_VALIDATION_RESULT",
        "checkpoint_id": sidecar["checkpoint_id"],
        "checkpoint_epoch": epoch,
        "selection": dict(selection),
        "calibration_validation": calibration.summary,
        "calibration_rows": list(calibration.rows),
        "psnr_evaluation": psnr_eval,
        "protected_counters": {
            "w7_candidate_results": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
        },
    }
    base["result_digest"] = canonical_sha256(base)
    return base


def evaluate_reconstruction_metrics(
    trainer: W7Trainer,
    checkpoint_id: str,
    *,
    repo_root=None,
) -> dict[str, Any]:
    """Evaluate 15 dB PSNR and symbol-domain PAPR on complete validation."""

    dataset = ValidationDJSCCDataset(W7_DATASET, repo_root=repo_root)
    expected_total = int(get(f"datasets.{W7_DATASET}.val_images"))
    loader = DataLoader(dataset, batch_size=W7_VALIDATION_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    psnr_values: list[float | str] = []
    papr_values: list[float] = []
    ids: list[str] = []
    with torch.inference_mode():
        for inputs, _labels, stable_ids in loader:
            ids_batch = [str(value) for value in stable_ids]
            noise_ids = _validation_noise_ids(trainer, ids_batch, W7_PSNR_SNR_DB)
            inputs_device = inputs.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            noise = keyed_complex_noise(noise_ids, int(trainer.config.resolved["k"]), dtype=torch.complex64, device=trainer.device)
            output = trainer.model(inputs_device, W7_PSNR_SNR_DB, unit_noise=noise)
            reconstruction = output.reconstruction.detach().float().cpu().numpy()
            targets = inputs.detach().float().cpu().numpy()
            papr = output.papr_db.detach().float().cpu().numpy()
            for index, stable_id in enumerate(ids_batch):
                reconstructed = reconstruction[index]
                _mse, psnr = _canonical_metric_values(targets[index], reconstructed)
                psnr_values.append(psnr)
                papr_values.append(_finite_float(papr[index], "PAPR"))
                ids.append(stable_id)
    if len(ids) != expected_total or ids != sorted(ids) or len(set(ids)) != len(ids):
        raise W7Hold("PSNR/PAPR validation coverage differs")
    per_image = [
        {"stable_sample_id": stable_id, "psnr_db": psnr, "papr_db": papr}
        for stable_id, psnr, papr in zip(ids, psnr_values, papr_values)
    ]
    return {
        "checkpoint_id": checkpoint_id,
        "snr_db": W7_PSNR_SNR_DB,
        "denominator": expected_total,
        "data_range": float(get("preprocessing.psnr_data_range")),
        "psnr_definition": "per_image_mse_all_RGB_pixels_then_arithmetic_mean",
        "psnr_db": _mean_metric(psnr_values),
        "papr_definition": "symbol_domain_per_image_then_arithmetic_mean",
        "papr_db": float(np.mean(np.asarray(papr_values, dtype=np.float64))),
        "per_image_digest": canonical_sha256(per_image),
        "per_image": per_image,
    }
