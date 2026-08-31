"""W8 validation-only checkpoint selection.

This module has no test-split route and no W7 artifact dependency.  It uses
one fixed validation SNR and the run's zipped channel seed for every epoch;
the keyed noise identity therefore remains comparable across epochs and
across DataLoader batchings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from channels.awgn import keyed_complex_noise
from config.params import get
from data.djscc_validation import ValidationDJSCCDataset, validation_noise_id
from training.deterministic_core import canonical_sha256
from training.w8_final import W8Hold, W8Trainer
from training.w8_protocol import (
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
    W8_DATASET,
    W8_EXPECTED_K,
    W8_EXPECTED_RATIOS,
    W8_SELECTED_ROLE,
    W8_TRAIN_SEEDS,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    eligibility_for_role,
    checkpoint_selection_snr_db,
)


W8_VALIDATION_SUMMARY_SCHEMA_VERSION = 1
W8_SELECTED_CHECKPOINT_SCHEMA_VERSION = 1
W8_VALIDATION_NOISE_POLICY = "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs"
W8_VALIDATION_ORDER = "stable_manifest_order"


class W8ValidationHold(W8Hold):
    """The validation path is incomplete, changed, or outside W8 scope."""


def evaluation_config_hash(
    trainer: W8Trainer, *, batch_size: int
) -> str:
    """Digest the non-result validation execution identity for one run."""

    return canonical_sha256(
        {
            "protocol_config_hash": trainer.protocol_hash,
            "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "snr_db": checkpoint_selection_snr_db(),
            "channel_seed": trainer.config.resolved["channel_seed"],
            "batch_size": int(batch_size),
            "order": W8_VALIDATION_ORDER,
        }
    )


@dataclass(frozen=True)
class ValidationEvaluation:
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def validation_stable_ids(*, repo_root=None) -> tuple[str, ...]:
    """Return the committed validation identity order without decoding images."""

    dataset = ValidationDJSCCDataset(W8_DATASET, repo_root=repo_root)
    source = getattr(dataset, "_source", None)
    source_sample = getattr(source, "source_sample", None)
    if not callable(source_sample):
        raise W8ValidationHold("W8 validation source does not expose stable identities")
    identifiers = tuple(str(source_sample(index).stable_sample_id) for index in range(len(dataset)))
    if len(identifiers) != W8_VALIDATION_SAMPLE_COUNT or len(set(identifiers)) != len(identifiers):
        raise W8ValidationHold("W8 validation identity denominator differs")
    if identifiers != tuple(sorted(identifiers)):
        raise W8ValidationHold("W8 validation identity order differs")
    return identifiers


def _finite(value: object, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise W8ValidationHold(f"{label} is not numeric") from None
    if not math.isfinite(converted):
        raise W8ValidationHold(f"{label} is non-finite")
    return converted


def _noise_ids_for_values(
    stable_ids: Sequence[str],
    *,
    ratio: str,
    k: int,
    channel_seed: int,
    snr_db: int | float,
) -> list[str]:
    version_field = str(get("config.dataset_version_rule"))
    dataset_version = str(get(f"datasets.{W8_DATASET}.{version_field}"))
    manifest_hash = str(get(f"datasets.{W8_DATASET}.manifest_sha256"))
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


def _noise_ids(trainer: W8Trainer, stable_ids: Sequence[str], snr_db: int | float) -> list[str]:
    resolved = trainer.config.resolved
    return _noise_ids_for_values(
        stable_ids,
        ratio=str(resolved["bw_ratio"]),
        k=int(resolved["k"]),
        channel_seed=int(resolved["channel_seed"]),
        snr_db=snr_db,
    )


def _prediction_digest(ids: Sequence[str], predictions: Sequence[int], labels: Sequence[int]) -> str:
    return canonical_sha256(
        [
            {
                "stable_sample_id": stable_id,
                "prediction": int(prediction),
                "label": int(label),
                "correct": int(prediction) == int(label),
            }
            for stable_id, prediction, label in zip(ids, predictions, labels)
        ]
    )


def evaluate_validation(
    trainer: W8Trainer,
    *,
    checkpoint_id: str,
    snr_db: int | float | None = None,
    batch_size: int | None = None,
    repo_root=None,
    retain_rows: bool = False,
) -> ValidationEvaluation:
    """Evaluate all 1000 validation images at the frozen W8 selection SNR."""

    if not trainer.policy.validation_enabled or not trainer.policy.scientific:
        raise W8ValidationHold("non-scientific W8 artifacts cannot enter validation")
    if trainer.completed_epoch < 0:
        raise W8ValidationHold("W8 validation requires a completed checkpoint epoch")
    selected_snr = checkpoint_selection_snr_db()
    chosen_snr = selected_snr if snr_db is None else snr_db
    if chosen_snr != selected_snr:
        raise W8ValidationHold("W8 checkpoint-selection validation SNR differs from its frozen parameter")
    chosen_batch = W8_VALIDATION_BATCH_SIZE if batch_size is None else int(batch_size)
    if chosen_batch <= 0:
        raise W8ValidationHold("W8 validation batch size must be positive")
    dataset = ValidationDJSCCDataset(W8_DATASET, repo_root=repo_root)
    expected_total = int(get(f"datasets.{W8_DATASET}.val_images"))
    if expected_total != W8_VALIDATION_SAMPLE_COUNT or len(dataset) != expected_total:
        raise W8ValidationHold("W8 validation denominator differs from the committed 1000-image split")
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
    all_labels: list[int] = []
    all_noise_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels, stable_ids in loader:
            ids = [str(value) for value in stable_ids]
            labels_list = [int(value) for value in labels]
            noise_ids = _noise_ids(trainer, ids, chosen_snr)
            inputs_device = inputs.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            labels_device = labels.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            noise = keyed_complex_noise(
                noise_ids,
                int(trainer.config.resolved["k"]),
                dtype=torch.complex64,
                device=trainer.device,
            )
            output = trainer.model(inputs_device, chosen_snr, unit_noise=noise)
            predictions = [int(value) for value in output.logits.argmax(dim=1).detach().cpu()]
            for stable_id, label, prediction, noise_id in zip(ids, labels_list, predictions, noise_ids):
                is_correct = prediction == label
                correct += int(is_correct)
                total += 1
                all_ids.append(stable_id)
                all_labels.append(label)
                all_predictions.append(prediction)
                all_noise_ids.append(noise_id)
                if retain_rows:
                    rows.append(
                        {
                            "stable_sample_id": stable_id,
                            "label": label,
                            "prediction": prediction,
                            "correct": is_correct,
                            "noise_id": noise_id,
                        }
                    )
    if total != expected_total or len(all_ids) != expected_total:
        raise W8ValidationHold("W8 validation did not process the complete denominator")
    if all_ids != sorted(all_ids) or len(set(all_ids)) != len(all_ids):
        raise W8ValidationHold("W8 validation order or identity coverage differs")
    if len(set(all_noise_ids)) != len(all_noise_ids):
        raise W8ValidationHold("W8 validation noise identities are duplicated")
    top1 = correct / total
    row_digest = canonical_sha256(
        rows if retain_rows else [
            {"stable_sample_id": stable_id, "label": label, "prediction": prediction, "correct": prediction == label, "noise_id": noise_id}
            for stable_id, label, prediction, noise_id in zip(all_ids, all_labels, all_predictions, all_noise_ids)
        ]
    )
    summary_body = {
        "schema_version": W8_VALIDATION_SUMMARY_SCHEMA_VERSION,
        "artifact_role": "W8_VALIDATION_EPOCH_SUMMARY",
        "eligibility": eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
        "campaign_id": trainer.campaign_id,
        "run_id": trainer.run_id,
        "ratio": trainer.config.resolved["bw_ratio"],
        "k": trainer.config.resolved["k"],
        "train_seed": trainer.config.resolved["train_seed"],
        "channel_seed": trainer.config.resolved["channel_seed"],
        "checkpoint_id": checkpoint_id,
        "epoch": trainer.completed_epoch,
        "validation_split": "val",
        "validation_order": W8_VALIDATION_ORDER,
        "validation_augmentation": False,
        "validation_batch_size": chosen_batch,
        "validation_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        "validation_snr_resolution": "params.channel.train_snr_db_fixed",
        "validation_snr_db": chosen_snr,
        "validation_channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
        "validation_channel_seed": trainer.config.resolved["channel_seed"],
        "validation_noise_policy": W8_VALIDATION_NOISE_POLICY,
        "validation_noise_id_digest": canonical_sha256(all_noise_ids),
        "validation_noise_id_count": len(all_noise_ids),
        "n_correct": correct,
        "n_total": total,
        "top1_accuracy": top1,
        "prediction_digest": _prediction_digest(all_ids, all_predictions, all_labels),
        "row_digest": row_digest,
        "evaluation_config_hash": evaluation_config_hash(
            trainer, batch_size=chosen_batch
        ),
        "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"],
        "test_model_facing_access": 0,
    }
    summary_body["summary_id"] = canonical_sha256(summary_body)
    return ValidationEvaluation(summary=summary_body, rows=tuple(rows))


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    expected_epoch: int | None = None,
    expected_evaluation_config_hash: str | None = None,
) -> None:
    required = {
        "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
        "ratio", "k", "train_seed", "channel_seed", "checkpoint_id", "epoch",
        "validation_split", "validation_order", "validation_augmentation",
        "validation_batch_size", "validation_snr_parameter", "validation_snr_resolution",
        "validation_snr_db", "validation_channel_seed_rule", "validation_channel_seed",
        "validation_noise_policy", "validation_noise_id_digest", "validation_noise_id_count",
        "n_correct", "n_total", "top1_accuracy", "prediction_digest", "row_digest",
        "evaluation_config_hash", "forbidden_selection_inputs", "test_model_facing_access",
        "summary_id",
    }
    if set(summary) != required:
        raise W8ValidationHold("W8 validation summary schema differs")
    body = dict(summary)
    digest = body.pop("summary_id")
    if digest != canonical_sha256(body):
        raise W8ValidationHold("W8 validation summary digest differs")
    if summary["schema_version"] != W8_VALIDATION_SUMMARY_SCHEMA_VERSION or summary["artifact_role"] != "W8_VALIDATION_EPOCH_SUMMARY":
        raise W8ValidationHold("W8 validation summary role/version differs")
    if summary["eligibility"] != eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"):
        raise W8ValidationHold("W8 validation summary eligibility differs")
    if (
        not isinstance(summary["ratio"], str)
        or summary["ratio"] not in W8_EXPECTED_RATIOS
        or not isinstance(summary["k"], int)
        or isinstance(summary["k"], bool)
        or summary["k"] != W8_EXPECTED_K[summary["ratio"]]
    ):
        raise W8ValidationHold("W8 validation ratio or k differs")
    if (
        not isinstance(summary["train_seed"], int)
        or isinstance(summary["train_seed"], bool)
        or summary["train_seed"] not in W8_TRAIN_SEEDS
        or not isinstance(summary["channel_seed"], int)
        or isinstance(summary["channel_seed"], bool)
        or summary["channel_seed"] != summary["train_seed"]
    ):
        raise W8ValidationHold("W8 validation seed pairing differs")
    if not isinstance(summary["checkpoint_id"], str) or len(summary["checkpoint_id"]) != 64 or any(character not in "0123456789abcdef" for character in summary["checkpoint_id"]):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 validation checkpoint ID is invalid")
    if not isinstance(summary["summary_id"], str) or len(summary["summary_id"]) != 64 or any(character not in "0123456789abcdef" for character in summary["summary_id"]):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 validation summary ID is invalid")
    if expected_epoch is not None and summary["epoch"] != expected_epoch:
        raise W8ValidationHold("W8 validation summary epoch order differs")
    if not isinstance(summary["epoch"], int) or isinstance(summary["epoch"], bool) or summary["epoch"] < 0:
        raise W8ValidationHold("W8 validation summary epoch is invalid")
    if not isinstance(summary["checkpoint_id"], str) or not summary["checkpoint_id"]:
        raise W8ValidationHold("W8 validation checkpoint ID is empty")
    if summary["validation_split"] != "val" or summary["validation_order"] != W8_VALIDATION_ORDER or summary["validation_augmentation"] is not False:
        raise W8ValidationHold("W8 validation view differs")
    if expected_evaluation_config_hash is not None and summary["evaluation_config_hash"] != expected_evaluation_config_hash:
        raise W8ValidationHold("W8 validation evaluation configuration digest differs")
    if (
        not isinstance(summary["validation_batch_size"], int)
        or isinstance(summary["validation_batch_size"], bool)
        or summary["validation_batch_size"] != W8_VALIDATION_BATCH_SIZE
    ):
        raise W8ValidationHold("W8 validation batch binding differs")
    if (
        summary["validation_snr_parameter"] != W8_CHECKPOINT_SELECTION_SNR_PARAMETER
        or summary["validation_snr_resolution"] != "params.channel.train_snr_db_fixed"
        or not isinstance(summary["validation_snr_db"], int | float)
        or isinstance(summary["validation_snr_db"], bool)
        or not math.isfinite(float(summary["validation_snr_db"]))
        or summary["validation_snr_db"] != checkpoint_selection_snr_db()
    ):
        raise W8ValidationHold("W8 validation SNR authority differs")
    if not isinstance(summary["validation_noise_id_digest"], str) or len(summary["validation_noise_id_digest"]) != 64 or any(character not in "0123456789abcdef" for character in summary["validation_noise_id_digest"]):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 validation noise digest is invalid")
    if (
        summary["validation_channel_seed_rule"] != W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE
        or not isinstance(summary["validation_channel_seed"], int)
        or isinstance(summary["validation_channel_seed"], bool)
        or summary["validation_channel_seed"] != summary["channel_seed"]
    ):
        raise W8ValidationHold("W8 validation channel-seed rule differs")
    if summary["validation_noise_policy"] != W8_VALIDATION_NOISE_POLICY:
        raise W8ValidationHold("W8 validation noise policy differs")
    if (
        not isinstance(summary["validation_noise_id_count"], int)
        or isinstance(summary["validation_noise_id_count"], bool)
        or summary["validation_noise_id_count"] != W8_VALIDATION_SAMPLE_COUNT
    ):
        raise W8ValidationHold("W8 validation noise denominator differs")
    if not isinstance(summary["n_total"], int) or isinstance(summary["n_total"], bool) or summary["n_total"] != W8_VALIDATION_SAMPLE_COUNT:
        raise W8ValidationHold("W8 validation denominator differs from 1000")
    if not isinstance(summary["n_correct"], int) or isinstance(summary["n_correct"], bool) or not 0 <= summary["n_correct"] <= W8_VALIDATION_SAMPLE_COUNT:
        raise W8ValidationHold("W8 validation correct count is invalid")
    if (
        not isinstance(summary["top1_accuracy"], int | float)
        or isinstance(summary["top1_accuracy"], bool)
        or not math.isfinite(float(summary["top1_accuracy"]))
        or summary["top1_accuracy"] != summary["n_correct"] / summary["n_total"]
    ):
        raise W8ValidationHold("W8 top-1 accuracy is not count-derived")
    if summary["test_model_facing_access"] != 0 or not isinstance(summary["test_model_facing_access"], int) or isinstance(summary["test_model_facing_access"], bool):
        raise W8ValidationHold("W8 validation summary claims test access")
    if summary["forbidden_selection_inputs"] != ["psnr", "papr", "reconstruction_loss"]:
        raise W8ValidationHold("W8 selection inputs are not fail-closed")
    if summary["test_model_facing_access"] != 0:
        raise W8ValidationHold("W8 validation summary claims test access")


def _validate_selection(selection: Mapping[str, Any]) -> None:
    """Authenticate the per-run selector object before it is consumed."""

    required = {
        "artifact_role", "eligibility", "campaign_id", "run_id", "ratio", "k",
        "train_seed", "channel_seed", "metric", "mode", "tie_break",
        "validation_snr_parameter", "validation_snr_resolution", "validation_snr_db",
        "validation_channel_seed_rule", "selected_epoch", "selected_checkpoint_id",
        "n_correct", "n_total", "top1_accuracy", "cross_seed_selection",
        "psnr_selected", "papr_selected", "reconstruction_loss_selected", "selection_id",
    }
    if set(selection) != required:
        raise W8ValidationHold("W8 selected-checkpoint schema differs")
    body = dict(selection)
    identifier = body.pop("selection_id")
    if not isinstance(identifier, str) or len(identifier) != 64 or any(character not in "0123456789abcdef" for character in identifier):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 selection ID is invalid")
    if identifier != canonical_sha256(body):
        raise W8ValidationHold("W8 selection digest differs")
    if selection["artifact_role"] != W8_SELECTED_ROLE:
        raise W8ValidationHold("W8 selected-checkpoint role differs")
    expected_eligibility = {
        **eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
        "artifact_role": W8_SELECTED_ROLE,
        "selection_eligibility": "ELIGIBLE_FOR_PER_RUN_PRE_TEST_VALIDATION_ONLY",
        "w8_eligibility": "SELECTED_W8_CHECKPOINT_PENDING_RECONCILIATION",
        "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
        "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
    }
    if selection["eligibility"] != expected_eligibility:
        raise W8ValidationHold("W8 selected-checkpoint eligibility differs")
    if (
        not isinstance(selection["ratio"], str)
        or selection["ratio"] not in W8_EXPECTED_RATIOS
        or not isinstance(selection["k"], int)
        or isinstance(selection["k"], bool)
        or selection["k"] != W8_EXPECTED_K[selection["ratio"]]
    ):
        raise W8ValidationHold("W8 selected-checkpoint ratio or k differs")
    if (
        not isinstance(selection["train_seed"], int)
        or isinstance(selection["train_seed"], bool)
        or selection["train_seed"] not in W8_TRAIN_SEEDS
        or not isinstance(selection["channel_seed"], int)
        or isinstance(selection["channel_seed"], bool)
        or selection["channel_seed"] != selection["train_seed"]
    ):
        raise W8ValidationHold("W8 selected-checkpoint seed pairing differs")
    if selection["metric"] != "validation_top1_accuracy" or selection["mode"] != "max" or selection["tie_break"] != "earliest_epoch":
        raise W8ValidationHold("W8 selected-checkpoint rule differs")
    if (
        selection["validation_snr_parameter"] != W8_CHECKPOINT_SELECTION_SNR_PARAMETER
        or selection["validation_snr_resolution"] != "params.channel.train_snr_db_fixed"
        or not isinstance(selection["validation_snr_db"], int | float)
        or isinstance(selection["validation_snr_db"], bool)
        or not math.isfinite(float(selection["validation_snr_db"]))
        or selection["validation_snr_db"] != checkpoint_selection_snr_db()
        or selection["validation_channel_seed_rule"] != W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE
    ):
        raise W8ValidationHold("W8 selected-checkpoint validation authority differs")
    if not isinstance(selection["selected_epoch"], int) or isinstance(selection["selected_epoch"], bool) or selection["selected_epoch"] < 0:
        raise W8ValidationHold("W8 selected epoch is invalid")
    if not isinstance(selection["selected_checkpoint_id"], str) or len(selection["selected_checkpoint_id"]) != 64 or any(character not in "0123456789abcdef" for character in selection["selected_checkpoint_id"]):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 selected checkpoint ID is invalid")
    if (
        not isinstance(selection["n_total"], int)
        or isinstance(selection["n_total"], bool)
        or selection["n_total"] != W8_VALIDATION_SAMPLE_COUNT
        or not isinstance(selection["n_correct"], int)
        or isinstance(selection["n_correct"], bool)
        or not 0 <= selection["n_correct"] <= W8_VALIDATION_SAMPLE_COUNT
        or not isinstance(selection["top1_accuracy"], int | float)
        or isinstance(selection["top1_accuracy"], bool)
        or not math.isfinite(float(selection["top1_accuracy"]))
        or selection["top1_accuracy"] != selection["n_correct"] / selection["n_total"]
    ):
        raise W8ValidationHold("W8 selected-checkpoint count is invalid")
    if selection["cross_seed_selection"] is not False or selection["psnr_selected"] is not False or selection["papr_selected"] is not False or selection["reconstruction_loss_selected"] is not False:
        raise W8ValidationHold("W8 selected-checkpoint used a forbidden selection input")


def select_checkpoint_epoch(
    summaries: Sequence[Mapping[str, Any]], *, expected_epochs: int
) -> dict[str, Any]:
    """Select max validation top-1, retaining the earliest exact tie.

    Unlike a permissive result consumer, this function requires the supplied
    list to be the ordered epoch prefix.  It also requires all summaries to be
    from one run, so three seeds can never be silently pooled for selection.
    """

    if not isinstance(expected_epochs, int) or isinstance(expected_epochs, bool) or expected_epochs <= 0:
        raise W8ValidationHold("W8 expected epoch count is invalid")
    if len(summaries) != expected_epochs:
        raise W8ValidationHold("W8 checkpoint-selection validation history is incomplete")
    validated: list[dict[str, Any]] = []
    for epoch, summary in enumerate(summaries):
        _validate_summary(summary, expected_epoch=epoch)
        validated.append(dict(summary))
    identity = (validated[0]["campaign_id"], validated[0]["run_id"], validated[0]["ratio"], validated[0]["train_seed"], validated[0]["channel_seed"], validated[0]["k"])
    if any((item["campaign_id"], item["run_id"], item["ratio"], item["train_seed"], item["channel_seed"], item["k"]) != identity for item in validated):
        raise W8ValidationHold("W8 cross-seed or cross-ratio checkpoint selection is forbidden")
    if any(item["validation_noise_id_digest"] != validated[0]["validation_noise_id_digest"] for item in validated[1:]):
        raise W8ValidationHold("W8 validation noise must remain fixed across epochs")
    maximum = max(_finite(item["top1_accuracy"], "W8 validation top-1") for item in validated)
    selected_epoch = next(index for index, item in enumerate(validated) if float(item["top1_accuracy"]) == maximum)
    selected = validated[selected_epoch]
    campaign_id, run_id, ratio, train_seed, channel_seed, k = identity
    body = {
        "artifact_role": W8_SELECTED_ROLE,
        "eligibility": {
            **eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
            "artifact_role": W8_SELECTED_ROLE,
            "selection_eligibility": "ELIGIBLE_FOR_PER_RUN_PRE_TEST_VALIDATION_ONLY",
            "w8_eligibility": "SELECTED_W8_CHECKPOINT_PENDING_RECONCILIATION",
            "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        },
        "campaign_id": campaign_id,
        "run_id": run_id,
        "ratio": ratio,
        "k": k,
        "train_seed": train_seed,
        "channel_seed": channel_seed,
        "metric": "validation_top1_accuracy",
        "mode": "max",
        "tie_break": "earliest_epoch",
        "validation_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        "validation_snr_resolution": "params.channel.train_snr_db_fixed",
        "validation_snr_db": checkpoint_selection_snr_db(),
        "validation_channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_id": selected["checkpoint_id"],
        "n_correct": selected["n_correct"],
        "n_total": selected["n_total"],
        "top1_accuracy": selected["top1_accuracy"],
        "cross_seed_selection": False,
        "psnr_selected": False,
        "papr_selected": False,
        "reconstruction_loss_selected": False,
    }
    body["selection_id"] = canonical_sha256(body)
    _validate_selection(body)
    return body


def validate_selected_checkpoint_result(
    result: Mapping[str, Any],
    *,
    expected_evaluation_config_hash: str | None = None,
    expected_validation_ids: Sequence[str] | None = None,
) -> None:
    """Validate a selected result without rerunning model-facing inference."""

    required = {
        "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
        "ratio", "train_seed", "channel_seed", "checkpoint_id", "checkpoint_epoch",
        "selection", "validation", "validation_rows", "test_model_facing_access",
        "result_id",
    }
    if set(result) != required:
        raise W8ValidationHold("W8 selected result schema differs")
    body = dict(result)
    identifier = body.pop("result_id")
    if not isinstance(identifier, str) or len(identifier) != 64 or any(character not in "0123456789abcdef" for character in identifier):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 selected result ID is invalid")
    if identifier != canonical_sha256(body):
        raise W8ValidationHold("W8 selected result digest differs")
    if result["schema_version"] != W8_SELECTED_CHECKPOINT_SCHEMA_VERSION or result["artifact_role"] != W8_SELECTED_ROLE:
        raise W8ValidationHold("W8 selected result role/version differs")
    expected_eligibility = {
        **eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
        "artifact_role": W8_SELECTED_ROLE,
        "selection_eligibility": "ELIGIBLE_FOR_PER_RUN_PRE_TEST_VALIDATION_ONLY",
        "w8_eligibility": "SELECTED_W8_CHECKPOINT_PENDING_RECONCILIATION",
        "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
        "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
    }
    if result["eligibility"] != expected_eligibility:
        raise W8ValidationHold("W8 selected result eligibility differs")
    if (
        not isinstance(result["ratio"], str)
        or result["ratio"] not in W8_EXPECTED_RATIOS
        or not isinstance(result["train_seed"], int)
        or isinstance(result["train_seed"], bool)
        or result["train_seed"] not in W8_TRAIN_SEEDS
        or not isinstance(result["channel_seed"], int)
        or isinstance(result["channel_seed"], bool)
        or result["channel_seed"] != result["train_seed"]
        or not isinstance(result["checkpoint_epoch"], int)
        or isinstance(result["checkpoint_epoch"], bool)
        or result["checkpoint_epoch"] < 0
    ):
        raise W8ValidationHold("W8 selected result identity differs")
    checkpoint_id = result["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_id):  # literal-ok: SHA-256 width
        raise W8ValidationHold("W8 selected result checkpoint ID is invalid")
    _validate_selection(result["selection"])
    selection = result["selection"]
    if selection["campaign_id"] != result["campaign_id"] or selection["run_id"] != result["run_id"] or selection["ratio"] != result["ratio"] or selection["train_seed"] != result["train_seed"] or selection["channel_seed"] != result["channel_seed"] or selection["selected_epoch"] != result["checkpoint_epoch"] or selection["selected_checkpoint_id"] != checkpoint_id:
        raise W8ValidationHold("W8 selected result selection binding differs")
    summary = result["validation"]
    if not isinstance(summary, Mapping):
        raise W8ValidationHold("W8 selected result validation is not a mapping")
    _validate_summary(
        summary,
        expected_epoch=result["checkpoint_epoch"],
        expected_evaluation_config_hash=expected_evaluation_config_hash,
    )
    if summary["campaign_id"] != result["campaign_id"] or summary["run_id"] != result["run_id"] or summary["ratio"] != result["ratio"] or summary["train_seed"] != result["train_seed"] or summary["channel_seed"] != result["channel_seed"] or summary["checkpoint_id"] != checkpoint_id:
        raise W8ValidationHold("W8 selected result validation binding differs")
    if any(summary[field] != selection[field] for field in ("n_correct", "n_total", "top1_accuracy")):
        raise W8ValidationHold("W8 selected result selection metric differs")
    rows = result["validation_rows"]
    if not isinstance(rows, list) or len(rows) != summary["n_total"]:
        raise W8ValidationHold("W8 selected result row denominator differs")
    row_keys = {"stable_sample_id", "label", "prediction", "correct", "noise_id"}
    ids: list[str] = []
    predictions: list[int] = []
    labels: list[int] = []
    noise_ids: list[str] = []
    class_count = int(get(f"datasets.{W8_DATASET}.classes"))
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise W8ValidationHold("W8 selected result row schema differs")
        stable_id = row["stable_sample_id"]
        if not isinstance(stable_id, str) or not stable_id:
            raise W8ValidationHold("W8 selected result stable ID is invalid")
        if not isinstance(row["label"], int) or isinstance(row["label"], bool) or not 0 <= row["label"] < class_count or not isinstance(row["prediction"], int) or isinstance(row["prediction"], bool) or not 0 <= row["prediction"] < class_count or row["correct"] is not (row["prediction"] == row["label"]):
            raise W8ValidationHold("W8 selected result row outcome is invalid")
        noise_id = row["noise_id"]
        if not isinstance(noise_id, str) or len(noise_id) != 64 or any(character not in "0123456789abcdef" for character in noise_id):  # literal-ok: SHA-256 width
            raise W8ValidationHold("W8 selected result noise identity is invalid")
        ids.append(stable_id)
        predictions.append(row["prediction"])
        labels.append(row["label"])
        noise_ids.append(noise_id)
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise W8ValidationHold("W8 selected result row identity order differs")
    if expected_validation_ids is not None and tuple(ids) != tuple(expected_validation_ids):
        raise W8ValidationHold("W8 selected result validation identity set differs")
    expected_noise_ids = _noise_ids_for_values(
        ids,
        ratio=str(result["ratio"]),
        k=W8_EXPECTED_K[str(result["ratio"])],
        channel_seed=int(result["channel_seed"]),
        snr_db=summary["validation_snr_db"],
    )
    if noise_ids != expected_noise_ids or summary["validation_noise_id_digest"] != canonical_sha256(expected_noise_ids):
        raise W8ValidationHold("W8 selected result validation noise identities differ")
    if canonical_sha256(rows) != summary["row_digest"] or _prediction_digest(ids, predictions, labels) != summary["prediction_digest"]:
        raise W8ValidationHold("W8 selected result row digest differs")
    if (
        not isinstance(result["test_model_facing_access"], int)
        or isinstance(result["test_model_facing_access"], bool)
        or result["test_model_facing_access"] != 0
    ):
        raise W8ValidationHold("W8 selected result claims test access")


def selected_checkpoint_result(
    trainer: W8Trainer,
    *,
    selection: Mapping[str, Any],
    repo_root=None,
) -> dict[str, Any]:
    """Reload and reauthenticate one run's selected validation checkpoint."""

    _validate_selection(selection)
    if selection.get("run_id") != trainer.run_id or selection.get("campaign_id") != trainer.campaign_id:
        raise W8ValidationHold("W8 selected checkpoint belongs to a different run")
    if (
        selection["ratio"] != trainer.config.resolved["bw_ratio"]
        or selection["k"] != trainer.config.resolved["k"]
        or selection["train_seed"] != trainer.config.resolved["train_seed"]
        or selection["channel_seed"] != trainer.config.resolved["channel_seed"]
    ):
        raise W8ValidationHold("W8 selected checkpoint config identity differs")
    selection_body = dict(selection)
    selection_id = selection_body.pop("selection_id", None)
    if selection_id != canonical_sha256(selection_body):
        raise W8ValidationHold("W8 selection digest differs")
    epoch = selection.get("selected_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0 or epoch >= int(get(f"learned_system.epochs.{W8_DATASET}")):
        raise W8ValidationHold("W8 selected epoch is invalid")
    sidecar = trainer.load_checkpoint_epoch(epoch)
    if sidecar["checkpoint_id"] != selection["selected_checkpoint_id"]:
        raise W8ValidationHold("W8 selected checkpoint ID differs")
    evaluation = evaluate_validation(
        trainer,
        checkpoint_id=sidecar["checkpoint_id"],
        snr_db=checkpoint_selection_snr_db(),
        repo_root=repo_root,
        retain_rows=True,
    )
    if evaluation.summary["n_correct"] != selection["n_correct"] or evaluation.summary["n_total"] != selection["n_total"] or evaluation.summary["top1_accuracy"] != selection["top1_accuracy"]:
        raise W8ValidationHold("W8 selected checkpoint validation reauthentication differs")
    result = {
        "schema_version": W8_SELECTED_CHECKPOINT_SCHEMA_VERSION,
        "artifact_role": W8_SELECTED_ROLE,
        "eligibility": dict(selection["eligibility"]),
        "campaign_id": trainer.campaign_id,
        "run_id": trainer.run_id,
        "ratio": trainer.config.resolved["bw_ratio"],
        "train_seed": trainer.config.resolved["train_seed"],
        "channel_seed": trainer.config.resolved["channel_seed"],
        "checkpoint_id": sidecar["checkpoint_id"],
        "checkpoint_epoch": epoch,
        "selection": dict(selection),
        "validation": evaluation.summary,
        "validation_rows": list(evaluation.rows),
        "test_model_facing_access": 0,
    }
    result["result_id"] = canonical_sha256(result)
    validate_selected_checkpoint_result(
        result,
        expected_evaluation_config_hash=evaluation_config_hash(
            trainer, batch_size=int(evaluation.summary["validation_batch_size"])
        ),
    )
    return result
