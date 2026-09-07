"""Validation-only ER-9 real-chain evaluation helpers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.params import get
from data.djscc_training import TrainingDJSCCDataset
from data.djscc_validation import ValidationDJSCCDataset, validation_noise_id
from evaluation.er9_protocol import (
    EncodedMessage,
    StaticEntropyModel,
    decode_message,
    encode_message,
    fit_static_entropy_model,
)
from evaluation.er9_search import configured_phy_candidates, select_phy
from evaluation.er9_transport import ER9TransportBatch
from models.er9_digital import ER9DigitalModel


@dataclass(frozen=True)
class ValidationFeatureBatch:
    indices: np.ndarray
    labels: np.ndarray
    stable_ids: tuple[str, ...]


def _split_manifest_hash(dataset: str) -> str:
    return str(get(f"datasets.{dataset}.manifest_sha256"))


@torch.no_grad()
def collect_validation_features(
    model: ER9DigitalModel,
    config: Any,
    *,
    device: torch.device | str,
    num_workers: int,
) -> tuple[ValidationFeatureBatch, ...]:
    """Encode the complete validation split before opening any PHY sessions."""

    dataset = ValidationDJSCCDataset(str(config.resolved["dataset"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config.resolved["validation_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=bool(config.parameters["learned_system"]["pin_memory"] and torch.device(device).type == "cuda"),
    )
    model.eval()
    batches: list[ValidationFeatureBatch] = []
    for inputs, labels, stable_ids in loader:
        inputs = inputs.to(device, non_blocking=torch.device(device).type == "cuda")
        indices = model.encode(inputs)[2].detach().cpu().numpy().astype(np.int64)
        batches.append(
            ValidationFeatureBatch(
                indices=indices,
                labels=labels.detach().cpu().numpy().astype(np.int64),
                stable_ids=tuple(str(value) for value in stable_ids),
            )
        )
    if sum(len(batch.stable_ids) for batch in batches) != len(dataset):
        raise RuntimeError("ER-9 validation feature collection did not cover the split")
    return tuple(batches)


@torch.no_grad()
def fit_entropy_model(
    model: ER9DigitalModel,
    config: Any,
    *,
    device: torch.device | str,
    num_workers: int,
) -> tuple[StaticEntropyModel, dict[str, Any]]:
    """Fit the static table on an explicitly train-only, deterministic view."""

    dataset = TrainingDJSCCDataset(
        str(config.resolved["dataset"]),
        int(config.resolved["train_seed"]),
        0,  # literal-ok: fixed first-epoch augmentation view for train-only fitting
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.resolved["validation_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=bool(config.parameters["learned_system"]["pin_memory"] and torch.device(device).type == "cuda"),
    )
    model.eval()
    chunks: list[np.ndarray] = []
    sample_count = 0
    for inputs, _labels, _stable_ids in loader:
        inputs = inputs.to(device, non_blocking=torch.device(device).type == "cuda")
        indices = model.encode(inputs)[2].detach().cpu().numpy().astype(np.int64)
        chunks.append(indices.reshape(-1))
        sample_count += int(indices.shape[0])
    all_indices = np.concatenate(chunks)
    entropy = fit_static_entropy_model(all_indices, model.quantiser.levels)
    return entropy, {
        "fit_split": "train",
        "fit_view": "TrainingDJSCCDataset_epoch_0_deterministic_augmentation",
        "sample_count": sample_count,
        "symbol_count": int(all_indices.size),
        "table": entropy.as_dict(),
    }


def _messages(
    batch: ValidationFeatureBatch,
    model: ER9DigitalModel,
    entropy: StaticEntropyModel,
    packet_payload_bits: int,
) -> tuple[tuple[EncodedMessage, ...], tuple[np.ndarray, ...]]:
    messages: list[EncodedMessage] = []
    payloads: list[np.ndarray] = []
    for row in batch.indices:
        message = encode_message(
            row,
            model.quantiser,
            entropy,
            packet_payload_bits=packet_payload_bits,
        )
        messages.append(message)
        payloads.append(message.message_bits)
    return tuple(messages), tuple(payloads)


def evaluate_candidate_at_snr(
    model: ER9DigitalModel,
    config: Any,
    *,
    entropy: StaticEntropyModel,
    dimension: int,
    quantiser_bits: int,
    snr_db: int,
    device: torch.device | str,
    num_workers: int,
    validation_features: tuple[ValidationFeatureBatch, ...] | None = None,
    include_per_image: bool = False,
) -> dict[str, Any]:
    """Tune every feasible configured PHY candidate on the full validation set."""

    if validation_features is None:
        validation_features = collect_validation_features(
            model, config, device=device, num_workers=num_workers
        )
    rows: list[dict[str, Any]] = []
    phy_candidates = configured_phy_candidates(int(config.resolved["k"]))
    dataset_name = str(config.resolved["dataset"])
    split_hash = _split_manifest_hash(dataset_name)
    selected_per_image: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for modulation, rate, packet in phy_candidates:
        session = ER9TransportBatch(packet, device=str(device))
        correct = 0
        total = 0
        delivered = 0
        range_count = 0
        raw_count = 0
        range_bits = 0
        raw_bits = 0
        energies: list[float] = []
        per_image: list[dict[str, Any]] = []
        for batch in validation_features:
            messages, payloads = _messages(
                batch,
                model,
                entropy,
                int(packet.segmentation.payload_bits),  # type: ignore[union-attr]
            )
            noise_ids = [
                validation_noise_id(
                    stable_sample_id=stable_id,
                    dataset_version=str(config.resolved["dataset_version"]),
                    split_manifest_hash=split_hash,
                    channel_seed=int(config.resolved["channel_seed"]),
                    channel=str(config.resolved["channel"]),
                    ratio=str(config.resolved["bw_ratio"]),
                    k=int(config.resolved["k"]),
                    snr_db=snr_db,
                )
                for stable_id in batch.stable_ids
            ]
            result = session.round_trip(payloads, snr_db=snr_db, noise_ids=noise_ids)
            energies.extend(result.realised_symbol_energy)
            for message, payload, label, stable_id in zip(
                messages, result.payloads, batch.labels, batch.stable_ids, strict=True
            ):
                total += 1
                if payload is None:
                    if include_per_image:
                        per_image.append({
                            "stable_sample_id": stable_id,
                            "correct": False,
                            "delivered": False,
                        })
                    continue
                delivered += 1
                try:
                    indices, branch = decode_message(
                        payload,
                        dimension=dimension,
                        quantiser=model.quantiser,
                        entropy_model=entropy,
                    )
                except ValueError:
                    delivered -= 1
                    if include_per_image:
                        per_image.append({
                            "stable_sample_id": stable_id,
                            "correct": False,
                            "delivered": False,
                            "decode_failure": True,
                        })
                    continue
                if branch == "range":
                    range_count += 1
                    range_bits += message.range_bit_count
                else:
                    raw_count += 1
                    raw_bits += message.raw_bit_count
                tensor = torch.as_tensor(indices[None, :], dtype=torch.long, device=device)
                prediction = int(model.logits_from_indices(tensor).argmax(dim=1).item())
                outcome = prediction == int(label)
                correct += int(outcome)
                if include_per_image:
                    per_image.append({
                        "stable_sample_id": stable_id,
                        "correct": bool(outcome),
                        "delivered": True,
                        "branch": branch,
                    })
        packet_metadata = packet.metadata()
        rows.append(
            {
                "transmit_dim": dimension,
                "quantiser_bits": quantiser_bits,
                "snr_db": snr_db,
                "modulation": modulation,
                "ldpc_rate": rate,
                "n_correct": correct,
                "n_total": total,
                "delivered": delivered,
                "decode_failures": total - delivered,
                "range_messages": range_count,
                "raw_messages": raw_count,
                "range_bit_count_sum": range_bits,
                "raw_bit_count_sum": raw_bits,
                "mean_realised_symbol_energy": float(np.mean(energies)),
                "packet": packet_metadata,
            }
        )
        if include_per_image:
            selected_per_image[(modulation, rate)] = per_image
        del session
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    selected = dict(select_phy(rows))
    per_image = selected_per_image.get(
        (str(selected["modulation"]), str(selected["ldpc_rate"])), []
    )
    if include_per_image and len(per_image) != int(selected["n_total"]):
        raise RuntimeError("ER-9 selected PHY per-image evidence does not cover validation")
    return {
        "transmit_dim": dimension,
        "quantiser_bits": quantiser_bits,
        "snr_db": snr_db,
        "selection_rule": "highest_exact_validation_n_correct_then_qm_ascending_rate_ascending_configured_id",
        "candidates": rows,
        "selected": selected,
        "validation_total": int(selected["n_total"]),
        "validation_n_correct": int(selected["n_correct"]),
        "validation_delivered": int(selected["delivered"]),
        "per_image": per_image if include_per_image else None,
        "test_access": 0,
    }


__all__ = [
    "ValidationFeatureBatch",
    "collect_validation_features",
    "evaluate_candidate_at_snr",
    "fit_entropy_model",
]
