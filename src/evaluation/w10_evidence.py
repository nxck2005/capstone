"""W10 SR-18 per-image evidence and unit identity contracts (AM-98).

Every scientific W10 unit carries exactly 1000 normative SR-18 rows over the
complete Imagenette-160 validation split, and one system-wide scheduled
``noise_id`` convention so paired arms share ``pair_id`` (BR-7, ER-10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from baseline.classical.records import (
    RunIdentity,
    noise_identity,
    per_image_schema,
    require_system,
    validate_row,
)
from config.params import get
from training.deterministic_core import canonical_bytes, canonical_sha256

from evaluation.w10_scope import (
    W10_CHANNEL_SEED,
    W10_DATASET,
    W10_SPLIT,
    W10_TRAIN_SEED,
    W10_VALIDATION_DENOMINATOR,
)

W10_UNIT_ROLE = "W10_VALIDATION_REHEARSAL_UNIT"
W10_PER_IMAGE_ROLE = "W10_VALIDATION_REHEARSAL_PER_IMAGE"
W10_UNIT_PREFIX = "w10unit-"


def dataset_version() -> str:
    rule = str(get("config.dataset_version_rule"))
    return str(get(f"datasets.{W10_DATASET}.{rule}"))


def split_manifest_hash() -> str:
    return str(get(f"datasets.{W10_DATASET}.manifest_sha256"))


def scheduled_noise_id(
    *,
    stable_sample_id: str,
    bw_ratio: str,
    test_snr_db: int | float,
    channel_seed: int = W10_CHANNEL_SEED,
    k: int,
    block_index: int = 0,
) -> str:
    """The one normative channel realisation identity every W10 arm shares.

    ``params.artifacts.noise_id_key`` is the normative key, and ``block_index``
    is fixed at 0 for the row-level scheduled realisation.  Using the same
    function for the analog and digital arms is what makes ``pair_id`` join
    (BR-7); the campaign-local identity helpers used by closed G-10/ER-9 runs
    are historical and are not reused here.
    """

    return noise_identity(
        dataset_version=dataset_version(),
        split_manifest_hash=split_manifest_hash(),
        stable_sample_id=str(stable_sample_id),
        test_snr_db=float(test_snr_db),
        channel_seed=int(channel_seed),
        k=int(k),
        block_index=int(block_index),
    )


def run_identity(
    *,
    system: str,
    bw_ratio: str,
    snr_db: int | float,
    config_hash: str,
    checkpoint_id: str,
    classifier_variant: str,
    ldpc_rate: str,
    modulation: str,
    quantiser_bits: int | None,
    transmit_dim: int | None,
    reconstruction_weight: float | None,
) -> RunIdentity:
    return RunIdentity(
        system=system,
        dataset=W10_DATASET,
        dataset_version=dataset_version(),
        split=W10_SPLIT,
        split_manifest_hash=split_manifest_hash(),
        bw_ratio=bw_ratio,
        test_snr_db=float(snr_db),
        train_seed=W10_TRAIN_SEED,
        channel_seed=W10_CHANNEL_SEED,
        config_hash=config_hash,
        checkpoint_id=checkpoint_id,
        classifier_variant=classifier_variant,
        ldpc_rate=ldpc_rate,
        modulation=modulation,
        quantiser_bits=quantiser_bits,
        transmit_dim=transmit_dim,
        reconstruction_weight=reconstruction_weight,
        analysis_version=int(get("config.analysis_version")),
    )


def sr18_row(
    *,
    identity: RunIdentity,
    stable_sample_id: str,
    true_label: int,
    pred_label: int,
    correct: bool,
    outage: bool,
    outage_reason: str | None,
    source_bytes: int | None,
) -> dict[str, Any]:
    """Build one schema-exact SR-18 row for any W10 arm."""

    noise_id = scheduled_noise_id(
        stable_sample_id=stable_sample_id,
        bw_ratio=identity.bw_ratio,
        test_snr_db=identity.test_snr_db,
        channel_seed=identity.channel_seed,
        k=_k_for(identity.bw_ratio),
    )
    row = {
        "run_id": identity.run_id(),
        "pair_id": identity.pair_id(
            stable_sample_id=str(stable_sample_id), noise_id=noise_id
        ),
        "noise_id": noise_id,
        "analysis_cell_id": identity.analysis_cell_id(),
        "dataset": identity.dataset,
        "dataset_version": identity.dataset_version,
        "split": identity.split,
        "stable_sample_id": str(stable_sample_id),
        "bw_ratio": identity.bw_ratio,
        "test_snr_db": identity.test_snr_db,
        "true_label": int(true_label),
        "pred_label": int(pred_label),
        "correct": bool(correct),
        "outage": bool(outage),
        "outage_reason": outage_reason,
        "source_bytes": None if source_bytes is None else int(source_bytes),
    }
    return validate_row(row, per_image_schema())


def _k_for(bw_ratio: str) -> int:
    return int(get(f"bandwidth.k_symbols.{W10_DATASET}.{bw_ratio}"))


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Order-preserving content address over the per-image rows."""

    return canonical_sha256({"rows": [dict(row) for row in rows]})


def rows_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_bytes({"rows": [dict(row) for row in rows]})


def validate_per_image(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stable_ids: Sequence[str],
    system: str,
    bw_ratio: str,
    snr_db: int | float,
) -> None:
    """Fail closed unless the stream is the exact 1000-row validation evidence."""

    require_system(system)
    if len(rows) != W10_VALIDATION_DENOMINATOR:
        raise ValueError(
            f"W10 per-image denominator differs: {len(rows)} != {W10_VALIDATION_DENOMINATOR}"
        )
    ids = [str(row.get("stable_sample_id")) for row in rows]
    if ids != list(expected_stable_ids):
        raise ValueError("W10 per-image stable-ID order differs from the validation view")
    if len(set(ids)) != W10_VALIDATION_DENOMINATOR:
        raise ValueError("W10 per-image stable IDs are not unique")
    expected_columns = per_image_schema()
    for row in rows:
        if tuple(row) != tuple(expected_columns):
            raise ValueError("W10 per-image row is not schema-exact")
        if row.get("split") != "val":
            raise ValueError("W10 per-image row is not the validation split")
        if str(row.get("bw_ratio")) != bw_ratio:
            raise ValueError("W10 per-image row ratio differs")
        if float(row.get("test_snr_db")) != float(snr_db):
            raise ValueError("W10 per-image row SNR differs")
        if bool(row.get("outage")) is False and row.get("outage_reason") is not None:
            raise ValueError("delivered W10 per-image row carries an outage reason")
        if bool(row.get("outage")) is True and not row.get("outage_reason"):
            raise ValueError("outage W10 per-image row lacks its typed reason")
    run_ids = {str(row.get("run_id")) for row in rows}
    if len(run_ids) != 1:
        raise ValueError("W10 per-image rows do not share one run identity")


def recompute_n_correct(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(bool(row.get("correct"))) for row in rows)


def build_per_image_record(
    *,
    ordinal: int,
    unit_key: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": W10_PER_IMAGE_ROLE,
        "ordinal": int(ordinal),
        "unit_key": str(unit_key),
        "n_total": len(rows),
        "rows": [dict(row) for row in rows],
    }
    body["per_image_id"] = W10_PER_IMAGE_ROLE.lower() + "-" + canonical_sha256(body)
    return body


def unit_relative_path(unit: Mapping[str, Any], suffix: str) -> str:
    return (
        f"units/{int(unit['ordinal']):03d}-{unit['system']}-{unit['bw_ratio']}"
        f"-snr{int(unit['snr_db']):+03d}.{suffix}"
    )


def per_image_relative_path(unit: Mapping[str, Any], variant_index: int) -> str:
    suffix = "json" if variant_index == 0 else f"{variant_index}.json"
    return (
        f"per_image/{int(unit['ordinal']):03d}-{unit['system']}-{unit['bw_ratio']}"
        f"-snr{int(unit['snr_db']):+03d}.{suffix}"
    )


__all__ = [
    "W10_PER_IMAGE_ROLE",
    "W10_UNIT_PREFIX",
    "W10_UNIT_ROLE",
    "build_per_image_record",
    "dataset_version",
    "per_image_relative_path",
    "recompute_n_correct",
    "rows_bytes",
    "rows_sha256",
    "run_identity",
    "scheduled_noise_id",
    "split_manifest_hash",
    "sr18_row",
    "unit_relative_path",
    "validate_per_image",
]
