"""Production W10 validation-side scientific evaluators (AM-98).

These functions own the per-unit science: load nothing implicitly, keep the
exact validation order, draw keyed noise from the one normative scheduled
identity, and return SR-18 rows plus a resolved binding.  Model construction
lives in :mod:`evaluation.w10_dispatch`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from baseline.classical.records import score_result
from config.params import get
from config.run_config import config_hash as run_config_hash
from evaluation.w10_evidence import (
    W10_PAPR_DOMAIN,
    per_image_relative_path,
    recompute_n_correct,
    rows_sha256,
    run_identity,
    scheduled_noise_id,
    sr18_row,
    unit_relative_path,
    validate_per_image,
)
from evaluation.w10_scope import W10_DATASET, W10_VALIDATION_DENOMINATOR
from models.frozen_reference_classifier import load_frozen_reference_classifier
from training.deterministic_core import canonical_sha256

NOT_APPLICABLE = "not_applicable"
DECODE_FAILURE = "decode_failure"
STRUCTURAL_INFEASIBILITY = "structural_infeasibility"
CODEC_INFEASIBILITY = "codec_infeasibility"
PAPR_BOUND_TOLERANCE_DB = 1e-4  # literal-ok: PeakPowerConstraint numerical bound tolerance


def papr_record(values: Sequence[float], *, denominator: int) -> dict[str, Any]:
    """Measured symbol-domain PAPR over the rows that actually transmitted.

    ``papr_measured_count`` is the number of transmissions whose realised
    symbols were measured; ``papr_denominator`` is the unit's complete row
    count.  Infeasible rows emit no symbols and are excluded from the mean but
    remain visible in the denominator, so a cell cannot report a clean PAPR by
    simply failing to transmit.
    """

    measured = [float(value) for value in values]
    if not measured:
        return {
            "mean_papr_db": None,
            "max_papr_db": None,
            "papr_measured_count": 0,
            "papr_denominator": int(denominator),
            "papr_domain": W10_PAPR_DOMAIN,
        }
    if not all(math.isfinite(value) for value in measured):
        raise RuntimeError("W10 PAPR measurement is non-finite")
    if len(measured) > int(denominator):
        raise RuntimeError("W10 PAPR measured count exceeds its denominator")
    return {
        "mean_papr_db": sum(measured) / len(measured),
        "max_papr_db": max(measured),
        "papr_measured_count": len(measured),
        "papr_denominator": int(denominator),
        "papr_domain": W10_PAPR_DOMAIN,
    }


@dataclass
class ValidationView:
    """The complete committed Imagenette-160 validation split in stable order."""

    dataset: str = W10_DATASET
    _source: Any = None
    stable_ids: list[str] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from data.registry import load_dataset

        self._source = load_dataset(self.dataset, "val")
        if len(self._source) != W10_VALIDATION_DENOMINATOR:
            raise RuntimeError("W10 validation denominator differs from 1000")
        pairs = []
        for index in range(len(self._source)):
            product, label = self._source[index]
            pairs.append((str(product.stable_sample_id), int(label)))
        pairs.sort()
        ids = [stable_id for stable_id, _ in pairs]
        if len(set(ids)) != W10_VALIDATION_DENOMINATOR:
            raise RuntimeError("W10 validation stable IDs are not unique")
        if ids != sorted(ids):
            raise RuntimeError("W10 validation stable IDs are not in manifest order")
        self.stable_ids = ids
        self.labels = dict(pairs)

    def index_of(self, stable_id: str) -> int:
        return self.stable_ids.index(stable_id)

    def product(self, stable_id: str) -> Any:
        return self._source[self.index_of(stable_id)][0]

    def label(self, stable_id: str) -> int:
        return int(self.labels[stable_id])

    def canonical_tensor(self, stable_id: str) -> torch.Tensor:
        from data.preprocessing import evaluation_input

        product = self.product(stable_id)
        return evaluation_input(product)


@dataclass
class W10Execution:
    """One execution context shared by every backend of one W10 run."""

    root: Any
    device: torch.device | str
    view: ValidationView | None = None
    models: dict[str, Any] = field(default_factory=dict)
    classifiers: dict[str, Any] = field(default_factory=dict)
    er9: dict[str, Any] = field(default_factory=dict)
    builders: dict[str, Callable[[str], Any]] = field(default_factory=dict)

    def validation(self) -> ValidationView:
        if self.view is None:
            self.view = ValidationView()
        return self.view

    def model(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self.models:
            self.models[key] = loader()
        return self.models[key]

    def classifier(self, variant: str) -> torch.nn.Module:
        if variant not in self.classifiers:
            model = load_frozen_reference_classifier(self.device, allow_download=True)
            model.eval()
            self.classifiers[variant] = model
        return self.classifiers[variant]

    def artifact_classifier(self, loader: Callable[[], torch.nn.Module]) -> torch.nn.Module:
        if "artifact_finetuned" not in self.classifiers:
            model = loader()
            model.eval()
            self.classifiers["artifact_finetuned"] = model
        return self.classifiers["artifact_finetuned"]


# ---------------------------------------------------------------------------
# Learned / reconstruction ablation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _learned_rows(
    context: W10Execution,
    *,
    model: torch.nn.Module,
    config: Any,
    checkpoint_id: str,
    unit: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    view = context.validation()
    identity = run_identity(
        system=unit["system"],
        bw_ratio=unit["bw_ratio"],
        snr_db=unit["snr_db"],
        config_hash=run_config_hash(config),
        checkpoint_id=checkpoint_id,
        classifier_variant=("clean" if mode == "recon_ablation" else "own_task_head"),
        ldpc_rate=NOT_APPLICABLE,
        modulation=NOT_APPLICABLE,
        quantiser_bits=None,
        transmit_dim=None,
        reconstruction_weight=3.0,  # literal-ok: AM-92 frozen lambda_core
    )
    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{unit['bw_ratio']}"))
    rows: list[dict[str, Any]] = []
    classifier = context.classifier("clean") if mode == "recon_ablation" else None
    papr_values: list[float] = []
    batch = 32  # literal-ok: frozen W8 validation batch size
    for start in range(0, len(view.stable_ids), batch):
        chunk = view.stable_ids[start : start + batch]
        inputs = torch.stack([view.canonical_tensor(stable_id) for stable_id in chunk]).to(context.device)
        noise_ids = [
            scheduled_noise_id(
                stable_sample_id=stable_id,
                bw_ratio=unit["bw_ratio"],
                test_snr_db=unit["snr_db"],
                k=k,
            )
            for stable_id in chunk
        ]
        from channels.awgn import keyed_complex_noise

        noise = keyed_complex_noise(tuple(noise_ids), k, dtype=torch.complex64, device=context.device)
        output = model(inputs, float(unit["snr_db"]), unit_noise=noise)
        papr_values.extend(float(value) for value in output.papr_db.detach().cpu())
        if mode == "recon_ablation":
            if classifier is None:
                raise RuntimeError("reconstruction ablation requires the frozen G-1 classifier")
            reconstruction = output.reconstruction
            logits = classifier(reconstruction)
            predictions = [int(value) for value in logits.argmax(dim=1).cpu()]
        else:
            predictions = [int(value) for value in output.logits.argmax(dim=1).cpu()]
        for stable_id, prediction in zip(chunk, predictions, strict=True):
            label = view.label(stable_id)
            rows.append(
                sr18_row(
                    identity=identity,
                    stable_sample_id=stable_id,
                    true_label=label,
                    pred_label=prediction,
                    correct=prediction == label,
                    outage=False,
                    outage_reason=None,
                    source_bytes=None,
                )
            )
    return rows, papr_values


def _apply_papr(aggregate: dict[str, Any], values: Sequence[float], *, denominator: int) -> None:
    aggregate.update(papr_record(values, denominator=denominator))


def learned_unit(
    context: W10Execution,
    unit: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    config: Any,
    checkpoint_id: str,
    papr_cap_db: float | None = None,
    protocol_config_hash: str | None = None,
) -> dict[str, Any]:
    rows, papr_values = _learned_rows(context, model=model, config=config, checkpoint_id=checkpoint_id, unit=unit, mode="learned")
    aggregate = _aggregate(rows, system=unit["system"])
    _apply_papr(aggregate, papr_values, denominator=len(rows))
    if papr_cap_db is not None:
        aggregate["papr_cap_db"] = float(papr_cap_db)
        aggregate["papr_cap_compliant"] = bool(
            aggregate["max_papr_db"] is not None
            and aggregate["max_papr_db"] <= float(papr_cap_db) + PAPR_BOUND_TOLERANCE_DB
        )
    aggregate["binding"] = {
        "kind": "learned_validation_evaluation",
        "checkpoint_id": checkpoint_id,
        "papr_cap_db": None if papr_cap_db is None else float(papr_cap_db),
        "papr_cap_compliance_required": papr_cap_db is not None,
    }
    if papr_cap_db is not None:
        if not isinstance(protocol_config_hash, str) or not protocol_config_hash:
            raise RuntimeError("PAPR evaluation requires its protocol config hash")
        aggregate["binding"].update(
            {
                "config_hash": run_config_hash(config),
                "protocol_config_hash": protocol_config_hash,
            }
        )
    return aggregate


def recon_ablation_unit(context: W10Execution, unit: Mapping[str, Any], *, model: torch.nn.Module, config: Any, checkpoint_id: str) -> dict[str, Any]:
    rows, papr_values = _learned_rows(context, model=model, config=config, checkpoint_id=checkpoint_id, unit=unit, mode="recon_ablation")
    aggregate = _aggregate(rows, system=unit["system"])
    _apply_papr(aggregate, papr_values, denominator=len(rows))
    aggregate["binding"] = {
        "kind": "er4_reconstruction_ablation",
        "checkpoint_id": checkpoint_id,
        "scorer": "frozen_g1_reference_classifier",
        "uses_djscc_logits": False,
    }
    return aggregate


# ---------------------------------------------------------------------------
# Randomized ER-2
# ---------------------------------------------------------------------------


@torch.inference_mode()
def er2_unit(context: W10Execution, unit: Mapping[str, Any], *, model: torch.nn.Module, config: Any, checkpoint_id: str, task_head_identity: str) -> dict[str, Any]:
    view = context.validation()
    identity = run_identity(
        system=unit["system"],
        bw_ratio=unit["bw_ratio"],
        snr_db=unit["snr_db"],
        config_hash=run_config_hash(config),
        checkpoint_id=checkpoint_id,
        classifier_variant="own_task_head",
        ldpc_rate=NOT_APPLICABLE,
        modulation=NOT_APPLICABLE,
        quantiser_bits=None,
        transmit_dim=None,
        reconstruction_weight=3.0,  # literal-ok: AM-92 frozen lambda_core
    )
    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{unit['bw_ratio']}"))
    from channels.awgn import keyed_complex_noise

    rows: list[dict[str, Any]] = []
    papr_values: list[float] = []
    batch = 32  # literal-ok: frozen W8 validation batch size
    for start in range(0, len(view.stable_ids), batch):
        chunk = view.stable_ids[start : start + batch]
        inputs = torch.stack([view.canonical_tensor(stable_id) for stable_id in chunk]).to(context.device)
        noise_ids = [
            scheduled_noise_id(stable_sample_id=stable_id, bw_ratio=unit["bw_ratio"], test_snr_db=unit["snr_db"], k=k)
            for stable_id in chunk
        ]
        noise = keyed_complex_noise(tuple(noise_ids), k, dtype=torch.complex64, device=context.device)
        output = model(inputs, float(unit["snr_db"]), unit_noise=noise)
        papr_values.extend(float(value) for value in output.papr_db.detach().cpu())
        predictions = [int(value) for value in output.logits.argmax(dim=1).cpu()]
        for stable_id, prediction in zip(chunk, predictions, strict=True):
            label = view.label(stable_id)
            rows.append(sr18_row(identity=identity, stable_sample_id=stable_id, true_label=label, pred_label=prediction, correct=prediction == label, outage=False, outage_reason=None, source_bytes=None))
    aggregate = _aggregate(rows, system=unit["system"])
    _apply_papr(aggregate, papr_values, denominator=len(rows))
    aggregate["binding"] = {
        "kind": "er2_randomized_validation_evaluation",
        "checkpoint_id": checkpoint_id,
        "task_head_identity": task_head_identity,
        "retrained": False,
    }
    return aggregate


# ---------------------------------------------------------------------------
# ER-9 digital control (frozen selected PHY per SNR)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def er9_unit(context: W10Execution, unit: Mapping[str, Any], *, assets: Mapping[str, Any], selection_phy: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    """Rehearse the frozen ER-9 selected transport on validation with SR-18 rows."""

    from evaluation.er9_campaign import authenticated_er9_outage_policy  # noqa: PLC0415
    from evaluation.er9_protocol import decode_message, encode_message  # noqa: PLC0415
    from evaluation.er9_transport import ER9TransportBatch  # noqa: PLC0415

    view = context.validation()
    model = assets["model"]
    config = assets["config"]
    entropy = assets["entropy"]
    dimension = int(assets["dimension"])
    quantiser_bits = int(assets["quantiser_bits"])
    checkpoint_id = str(assets["checkpoint_id"])
    candidates = {(str(item["modulation"]), str(item["ldpc_rate"])): item for item in assets["phy_candidates"]}
    point = selection_phy[float(unit["snr_db"])]
    phy = candidates[(str(point["modulation"]), str(point["ldpc_rate"]))]
    packet = phy["packet"]
    layout = packet.segmentation
    if layout is None:
        raise RuntimeError("ER-9 selected PHY has no packet segmentation")
    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{unit['bw_ratio']}"))
    config_hash = run_config_hash(config)
    identity = run_identity(
        system=unit["system"],
        bw_ratio=unit["bw_ratio"],
        snr_db=unit["snr_db"],
        config_hash=config_hash,
        checkpoint_id=checkpoint_id,
        classifier_variant="own_task_head",
        ldpc_rate=str(point["ldpc_rate"]),
        modulation=str(point["modulation"]),
        quantiser_bits=quantiser_bits,
        transmit_dim=dimension,
        reconstruction_weight=None,
    )
    policy = authenticated_er9_outage_policy(config)
    session = ER9TransportBatch(packet, device=str(context.device))
    rows: list[dict[str, Any]] = []
    papr_values: list[float] = []
    batch = 32  # literal-ok: frozen W8 validation batch size
    for start in range(0, len(view.stable_ids), batch):
        chunk = view.stable_ids[start : start + batch]
        inputs = torch.stack([view.canonical_tensor(stable_id) for stable_id in chunk]).to(context.device)
        indices = model.encode(inputs)[2].detach().cpu().numpy()
        payloads = [
            encode_message(row, model.quantiser, entropy, packet_payload_bits=int(layout.payload_bits)).message_bits
            for row in indices
        ]
        noise_ids = [scheduled_noise_id(stable_sample_id=stable_id, bw_ratio=unit["bw_ratio"], test_snr_db=unit["snr_db"], k=k) for stable_id in chunk]
        result = session.round_trip(payloads=payloads, snr_db=float(unit["snr_db"]), noise_ids=noise_ids)
        papr_values.extend(float(value) for value in result.papr_db)
        for stable_id, payload in zip(chunk, result.payloads, strict=True):
            true_label = view.label(stable_id)
            if payload is None:
                pred_label = int(policy.predict())
                correct = policy.is_correct(true_label)
                outage_reason = DECODE_FAILURE
            else:
                try:
                    decoded, _branch = decode_message(payload, dimension=dimension, quantiser=model.quantiser, entropy_model=entropy)
                except ValueError:
                    pred_label = int(policy.predict())
                    correct = policy.is_correct(true_label)
                    outage_reason = DECODE_FAILURE
                else:
                    tensor = torch.as_tensor(decoded[None, :], dtype=torch.long, device=context.device)
                    pred_label = int(model.logits_from_indices(tensor).argmax(dim=1).item())
                    correct = pred_label == true_label
                    outage_reason = None
            rows.append(sr18_row(identity=identity, stable_sample_id=stable_id, true_label=true_label, pred_label=pred_label, correct=correct, outage=outage_reason is not None, outage_reason=outage_reason, source_bytes=int(layout.payload_bits) // 8))  # literal-ok: bits-per-octet conversion
    aggregate = _aggregate(rows, system=unit["system"])
    _apply_papr(aggregate, papr_values, denominator=len(rows))
    aggregate["binding"] = {
        "kind": "er9_frozen_selected_phy_validation_evaluation",
        "checkpoint_id": checkpoint_id,
        "transmit_dim": dimension,
        "quantiser_bits": quantiser_bits,
        "modulation": str(point["modulation"]),
        "ldpc_rate": str(point["ldpc_rate"]),
        "retrained": False,
        "reselected": False,
    }
    return aggregate


# ---------------------------------------------------------------------------
# ER-12 label-transmission upper bound
# ---------------------------------------------------------------------------


def label_payload(label: int) -> np.ndarray:
    """The frozen one-byte frame: low nibble predicted label, high nibble zero."""

    bits = int(get("evaluation.w10_er12_label_bits"))
    classes = int(get(f"datasets.{W10_DATASET}.classes"))
    if not 0 <= int(label) < classes or int(label) >= (1 << bits):  # literal-ok: label-field width
        raise ValueError("the predicted label does not fit the frozen label field")
    return np.array([int(label) & 0x0F], dtype=np.uint8)  # literal-ok: low-nibble label frame


def decode_label_payload(payload: np.ndarray, *, payload_bits: int) -> int | None:
    """Recover the label field from decoded transport bits.

    Any out-of-range nibble or nonzero high nibble is a decode failure; the
    frozen frame is exactly one byte with the label in the low nibble.
    """

    bits = np.asarray(payload, dtype=np.uint8).reshape(-1)
    if bits.size < 8:  # literal-ok: one-byte frame
        return None
    byte = int(np.packbits(bits[:8])[0])  # literal-ok: one-byte frame
    if byte >> 4 != 0:  # literal-ok: high nibble must be zero
        return None
    value = byte & 0x0F  # literal-ok: low-nibble label field
    return value if 0 <= value <= 9 else None  # literal-ok: ten-class label range


@torch.inference_mode()
def label_bound_unit(context: W10Execution, unit: Mapping[str, Any], *, selection: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, Any]:
    """Transmit only the transmitter-predicted label over the frozen PHY."""

    view = context.validation()
    model = assets["model"]
    config = assets["config"]
    entropy = assets["entropy"]
    dimension = int(assets["dimension"])
    quantiser_bits = int(assets["quantiser_bits"])
    checkpoint_id = str(assets["checkpoint_id"])
    if entropy is None:
        raise RuntimeError("label-bound execution requires the fitted ER-9 entropy model")
    from evaluation.er9_transport import ER9TransportBatch  # noqa: PLC0415

    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{unit['bw_ratio']}"))
    point = _selection_point(selection, float(unit["snr_db"]))
    entries = {str(item["modulation"]) + "|" + str(item["ldpc_rate"]): item for item in assets["phy_candidates"]}
    phy = entries[str(point["modulation"]) + "|" + str(point["ldpc_rate"])]
    packet = phy["packet"]
    layout = packet.segmentation
    if layout is None:
        raise RuntimeError("label-bound selected PHY has no packet segmentation")
    payload_bytes = int(layout.payload_bits) // 8  # literal-ok: bits-per-octet conversion
    identity = run_identity(
        system=unit["system"],
        bw_ratio=unit["bw_ratio"],
        snr_db=unit["snr_db"],
        config_hash=run_config_hash(config),
        checkpoint_id=checkpoint_id,
        classifier_variant="predicted_label",
        ldpc_rate=str(point["ldpc_rate"]),
        modulation=str(point["modulation"]),
        quantiser_bits=None,
        transmit_dim=None,
        reconstruction_weight=None,
    )
    from evaluation.er9_campaign import authenticated_er9_outage_policy  # noqa: PLC0415

    policy = authenticated_er9_outage_policy(config)
    rows: list[dict[str, Any]] = []
    papr_values: list[float] = []
    batch = 32  # literal-ok: frozen W8 validation batch size
    for start in range(0, len(view.stable_ids), batch):
        chunk = view.stable_ids[start : start + batch]
        inputs = torch.stack([view.canonical_tensor(stable_id) for stable_id in chunk]).to(context.device)
        logits = model(inputs).logits
        predicted = [int(value) for value in logits.argmax(dim=1).cpu()]
        payloads = []
        for label in predicted:
            frame = np.zeros(payload_bytes, dtype=np.uint8)
            frame[:1] = label_payload(label)[0]
            payloads.append(np.unpackbits(frame))
        noise_ids = [scheduled_noise_id(stable_sample_id=stable_id, bw_ratio=unit["bw_ratio"], test_snr_db=unit["snr_db"], k=k) for stable_id in chunk]
        session = ER9TransportBatch(packet, device=str(context.device))
        result = session.round_trip(payloads=payloads, snr_db=float(unit["snr_db"]), noise_ids=noise_ids)
        papr_values.extend(float(value) for value in result.papr_db)
        for stable_id, payload in zip(chunk, result.payloads, strict=True):
            true_label = view.label(stable_id)
            if payload is None or decode_label_payload(payload, payload_bits=int(layout.payload_bits)) is None:
                pred_label = int(policy.predict())
                correct = policy.is_correct(true_label)
                outage_reason = DECODE_FAILURE
            else:
                pred_label = int(decode_label_payload(payload, payload_bits=int(layout.payload_bits)))
                correct = pred_label == true_label
                outage_reason = None
            rows.append(sr18_row(identity=identity, stable_sample_id=stable_id, true_label=true_label, pred_label=pred_label, correct=correct, outage=outage_reason is not None, outage_reason=outage_reason, source_bytes=payload_bytes))
    from evaluation.w10_selections import score_vector_digest

    if point.get("per_image_correct_digest") is not None and score_vector_digest(
        [bool(row["correct"]) for row in rows]
    ) != str(point["per_image_correct_digest"]):
        raise RuntimeError("W10 ER-12 arm does not reproduce its frozen selection candidate digest")
    aggregate = _aggregate(rows, system=unit["system"], primary_classifier_variant="predicted_label")
    _apply_papr(aggregate, papr_values, denominator=len(rows))
    aggregate["binding"] = {
        "kind": "er12_label_transmission_upper_bound",
        "checkpoint_id": checkpoint_id,
        "selection_id": selection.get("selection_id"),
        "payload_frame": str(get("evaluation.w10_er12_payload_frame")),
        "declared_role": str(get("evaluation.w10_er12_declared_role")),
        "modulation": str(point["modulation"]),
        "ldpc_rate": str(point["ldpc_rate"]),
        "true_label_in_payload": False,
    }
    return aggregate


def _selection_point(selection: Mapping[str, Any], snr_db: float) -> dict[str, Any]:
    for item in selection["selections"]:
        if float(item["snr_db"]) == float(snr_db):
            return item
    raise RuntimeError(f"W10 selection has no point at {snr_db} dB")


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    system: str,
    primary_classifier_variant: str = "own_task_head",
) -> dict[str, Any]:
    total = len(rows)
    correct = recompute_n_correct(rows)
    delivered = sum(1 for row in rows if not row["outage"])
    delivered_correct = sum(1 for row in rows if not row["outage"] and row["correct"])
    infeasible = sum(1 for row in rows if row["outage_reason"] in (STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY))
    decode_failures = sum(1 for row in rows if row["outage_reason"] == DECODE_FAILURE)
    return {
        "system": system,
        "n_correct": correct,
        "n_total": total,
        "coverage_rate": delivered / total if total else None,
        "acc_given_delivery": (delivered_correct / delivered) if delivered else None,
        "decode_failure_count": decode_failures,
        "infeasible_count": infeasible,
        "per_image": list(rows),
        "per_image_sha256": rows_sha256(rows),
        "primary_classifier_variant": primary_classifier_variant,
    }


__all__ = [
    "CODEC_INFEASIBILITY",
    "DECODE_FAILURE",
    "NOT_APPLICABLE",
    "STRUCTURAL_INFEASIBILITY",
    "papr_record",
    "W10Execution",
    "ValidationView",
    "decode_label_payload",
    "er2_unit",
    "er9_unit",
    "label_bound_unit",
    "label_payload",
    "learned_unit",
    "recon_ablation_unit",
]
