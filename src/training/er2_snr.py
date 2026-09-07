"""AM-95's deterministic SNR assignment for the randomized ER-2 variant.

The assignment is deliberately a pure function of the scientific identity of
one training sample in one epoch.  It is called once for that sample's forward
pass; callers must reuse its result for every channel use in the pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from artifacts.rng import keyed_generator
from config.params import get


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def snr_selection_identity(
    dataset_version: str,
    split_manifest_hash: str,
    stable_sample_id: str,
    train_seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Return the exact AM-95 identity supplied to the keyed RNG."""

    return {
        "dataset_version": _require_non_empty_text(dataset_version, "dataset_version"),
        "split_manifest_hash": _require_non_empty_text(
            split_manifest_hash, "split_manifest_hash"
        ),
        "stable_sample_id": _require_non_empty_text(stable_sample_id, "stable_sample_id"),
        "train_seed": _require_non_negative_int(train_seed, "train_seed"),
        "epoch": _require_non_negative_int(epoch, "epoch"),
    }


def _sampling_domain() -> tuple[int, ...]:
    domain = get("channel.train_snr_db_set")
    if (
        not isinstance(domain, list | tuple)
        or not domain
        or any(isinstance(value, bool) or not isinstance(value, int) for value in domain)
        or len(set(domain)) != len(domain)
    ):
        raise ValueError("channel.train_snr_db_set must be a non-empty list of distinct integers")
    return tuple(int(value) for value in domain)


def _validated_contract() -> tuple[str, tuple[int, ...]]:
    distribution = get("channel.train_snr_randomisation_distribution")
    if distribution != "discrete_uniform":
        raise ValueError(
            "AM-95 requires channel.train_snr_randomisation_distribution=discrete_uniform"
        )
    unit = get("channel.train_snr_randomisation_unit")
    if unit != "per_sample_per_epoch":
        raise ValueError(
            "AM-95 requires channel.train_snr_randomisation_unit=per_sample_per_epoch"
        )
    purpose = get("channel.train_snr_randomisation_rng_purpose")
    if purpose != "er2_snr_randomised_v1":
        raise ValueError(
            "AM-95 requires channel.train_snr_randomisation_rng_purpose="
            "er2_snr_randomised_v1"
        )
    if purpose not in get("artifacts.rng_purposes"):
        raise ValueError(f"AM-95 RNG purpose is not registered: {purpose!r}")
    return purpose, _sampling_domain()


def select_training_snr_db(
    dataset_version: str,
    split_manifest_hash: str,
    stable_sample_id: str,
    train_seed: int,
    epoch: int,
) -> int:
    """Select one configured SNR for one sample/epoch, independent of batching.

    The channel-noise stream is intentionally not touched here.  A caller that
    needs channel noise must construct its separately keyed AM-91 identity.
    """

    purpose, domain = _validated_contract()
    identity = snr_selection_identity(
        dataset_version,
        split_manifest_hash,
        stable_sample_id,
        train_seed,
        epoch,
    )
    index = int(keyed_generator(purpose, identity).integers(0, len(domain)))
    return domain[index]


def assignment_record(
    dataset_version: str,
    split_manifest_hash: str,
    stable_sample_id: str,
    train_seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Return an auditable assignment without hiding its keyed identity."""

    purpose, _ = _validated_contract()
    identity = snr_selection_identity(
        dataset_version,
        split_manifest_hash,
        stable_sample_id,
        train_seed,
        epoch,
    )
    return {
        "rng_purpose": purpose,
        "identity": identity,
        "train_snr_db": select_training_snr_db(
            dataset_version,
            split_manifest_hash,
            stable_sample_id,
            train_seed,
            epoch,
        ),
    }


def assignments_for_samples(
    sample_ids: Mapping[object, str] | list[str] | tuple[str, ...],
    *,
    dataset_version: str,
    split_manifest_hash: str,
    train_seed: int,
    epoch: int,
) -> dict[object, int]:
    """Build a sample-ID keyed assignment map; input order is not scientific state."""

    if isinstance(sample_ids, Mapping):
        values = sample_ids.items()
    else:
        values = ((sample_id, sample_id) for sample_id in sample_ids)
    return {
        key: select_training_snr_db(
            dataset_version,
            split_manifest_hash,
            stable_sample_id,
            train_seed,
            epoch,
        )
        for key, stable_sample_id in values
    }
