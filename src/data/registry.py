"""One small public model-facing dataset registry (SR-2, SR-17, SR-20)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from config.params import REPO_ROOT, get
from data.adapters import (
    DatasetAdapterError,
    SourceSample,
    _adapter,
    _validate_adapter_coverage,
)
from data.manifests import (
    ManifestError,
    manifest_path as _manifest_path,
    manifest_sha256 as _manifest_sha256,
    validate_manifest_bytes,
)
from data.preprocessing import CanonicalProduct, canonicalize_source
from data.provenance import ProvenanceError, dataset_root, verify_extracted_dataset


class DatasetRegistryError(RuntimeError):
    """Raised before model-facing use when a registry contract is not met."""


class CanonicalDataset(Dataset[tuple[CanonicalProduct, int]]):
    """Train/validation samples canonicalized only when indexed."""

    def __init__(
        self,
        dataset: str,
        split: str,
        samples: tuple[SourceSample, ...],
    ) -> None:
        self.dataset = dataset
        self.split = split
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def source_sample(self, index: int) -> SourceSample:
        """Return the immutable source-bound record before canonicalization."""

        return self._samples[index]

    def __getitem__(self, index: int) -> tuple[CanonicalProduct, int]:
        sample = self.source_sample(index)
        product = canonicalize_source(sample.source_bytes, sample.dataset)
        if product.stable_sample_id != sample.stable_sample_id:
            raise DatasetRegistryError(
                f"{sample.dataset}: canonical product changed source identity"
            )
        return product, sample.label


def available_datasets() -> tuple[str, ...]:
    """Configured dataset names, after exact adapter/config coverage validation."""

    return _validate_adapter_coverage()


def manifest_path(dataset: str, repo_root: Path = REPO_ROOT) -> Path:
    _require_known_dataset(dataset)
    return _manifest_path(dataset, repo_root)


def manifest_sha256(dataset: str, repo_root: Path = REPO_ROOT) -> str:
    _require_known_dataset(dataset)
    return _manifest_sha256(dataset, repo_root)


def load_dataset(
    dataset: str,
    split: str,
    repo_root: Path = REPO_ROOT,
) -> CanonicalDataset:
    """Load only a manifest-backed train or validation dataset.

    Model-facing test loading remains exclusively behind ``data.test_access``
    and G-12.  This registry intentionally has no test override.
    """

    _require_known_dataset(dataset)
    if split == "test":
        raise DatasetRegistryError(
            "model-facing test loading remains sealed behind SR-22 and G-12"
        )
    if split not in {"train", "val"}:
        raise ValueError(
            f"unsupported split {split!r}; public datasets support train or val"
        )

    try:
        verify_extracted_dataset(dataset, repo_root)
        manifest_file = _manifest_path(dataset, repo_root)
        rows = validate_manifest_bytes(dataset, manifest_file.read_bytes())
        _verify_manifest_pin(dataset, _manifest_sha256(dataset, repo_root))
        adapter = _adapter(dataset, dataset_root(dataset, repo_root))
        class_mapping = dict(adapter.class_mapping())
        samples_by_id: dict[str, SourceSample] = {}
        for sample in adapter.iter_source_samples("train"):
            if sample.stable_sample_id in samples_by_id:
                raise DatasetRegistryError(
                    f"{dataset}: duplicate source ID {sample.stable_sample_id}"
                )
            samples_by_id[sample.stable_sample_id] = sample
    except (OSError, DatasetAdapterError, ManifestError, ProvenanceError) as exc:
        raise DatasetRegistryError(f"{dataset}: cannot load {split}: {exc}") from exc

    expected_labels = list(range(int(_dataset_config(dataset)["classes"])))
    if sorted(class_mapping.values()) != expected_labels:
        raise DatasetRegistryError(
            f"{dataset}: authoritative class mapping is inconsistent"
        )
    chosen_rows = tuple(row for row in rows if row.split == split)
    chosen_samples: list[SourceSample] = []
    for row in chosen_rows:
        try:
            sample = samples_by_id[row.stable_sample_id]
        except KeyError:
            raise DatasetRegistryError(
                f"{dataset}: manifest ID {row.stable_sample_id} is absent "
                "from the published training source"
            ) from None
        if sample.label != row.label:
            raise DatasetRegistryError(
                f"{dataset}: manifest label for {row.stable_sample_id} "
                f"is {row.label}, source label is {sample.label}"
            )
        chosen_samples.append(sample)

    expected_count = int(_dataset_config(dataset)[f"{split}_images"])
    if len(chosen_samples) != expected_count:
        raise DatasetRegistryError(
            f"{dataset}/{split}: loaded {len(chosen_samples)}, expected "
            f"{expected_count}"
        )
    if len({sample.stable_sample_id for sample in chosen_samples}) != len(
        chosen_samples
    ):
        raise DatasetRegistryError(f"{dataset}/{split}: duplicate stable IDs")
    return CanonicalDataset(dataset, split, tuple(chosen_samples))


def _verify_manifest_pin(dataset: str, actual_sha256: str) -> None:
    expected = _dataset_config(dataset)["manifest_sha256"]
    if not isinstance(expected, str) or expected.startswith("pending_"):
        raise DatasetRegistryError(
            f"{dataset}: pending manifest SHA-256 cannot authorize dataset use"
        )
    if actual_sha256 != expected:
        raise DatasetRegistryError(
            f"{dataset}: manifest SHA-256 mismatch: "
            f"{actual_sha256} != {expected}"
        )


def _dataset_config(dataset: str) -> Mapping[str, Any]:
    config = get(f"datasets.{dataset}")
    if not isinstance(config, Mapping):
        raise DatasetRegistryError(f"invalid dataset configuration: {dataset}")
    return config


def _require_known_dataset(dataset: str) -> None:
    if dataset not in available_datasets():
        raise ValueError(f"unknown dataset {dataset!r}")
