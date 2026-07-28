"""Dataset-specific source-payload adapters hidden behind the public registry.

The adapters deliberately stop at immutable source records.  They do not
canonicalize, augment, normalize, or expose a model-facing test dataset.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, STL10, Imagenette

from config.params import get
from data.identity import stable_sample_id


class DatasetAdapterError(RuntimeError):
    """Raised when source data contradicts its configured dataset contract."""


@dataclass(frozen=True)
class SourceSample:
    """One immutable sample bound to its exact original payload bytes."""

    dataset: str
    stable_sample_id: str
    label: int
    source_bytes: bytes

    def __post_init__(self) -> None:
        if stable_sample_id(self.source_bytes) != self.stable_sample_id:
            raise DatasetAdapterError(
                f"{self.dataset}: source bytes do not match stable ID "
                f"{self.stable_sample_id}"
            )
        if not isinstance(self.label, int) or isinstance(self.label, bool):
            raise TypeError("source-sample labels must be integers")


class DatasetAdapter(Protocol):
    """Private interface implemented once for each configured dataset."""

    name: str
    loader_name: str

    def class_mapping(self) -> Mapping[str, int]: ...

    def iter_source_samples(self, published_split: str) -> Iterator[SourceSample]: ...


def _dataset_config(name: str) -> Mapping[str, Any]:
    config = get(f"datasets.{name}")
    if not isinstance(config, Mapping) or "loader" not in config:
        raise DatasetAdapterError(f"invalid dataset parameter entry: {name}")
    return config


def _validated_class_mapping(
    dataset: str,
    names: list[str],
) -> dict[str, int]:
    expected = int(_dataset_config(dataset)["classes"])
    if len(names) != expected:
        raise DatasetAdapterError(
            f"{dataset}: authoritative metadata has {len(names)} classes, "
            f"expected {expected}"
        )
    if len(set(names)) != len(names):
        raise DatasetAdapterError(
            f"{dataset}: authoritative class names are not unique"
        )
    mapping = {name: label for label, name in enumerate(names)}
    if sorted(mapping.values()) != list(range(expected)):
        raise DatasetAdapterError(f"{dataset}: class labels are not contiguous")
    return mapping


def _sample(dataset: str, label: int, source_bytes: bytes) -> SourceSample:
    config = _dataset_config(dataset)
    class_count = int(config["classes"])
    if label not in range(class_count):
        raise DatasetAdapterError(
            f"{dataset}: label {label} is outside [0, {class_count})"
        )
    return SourceSample(
        dataset=dataset,
        stable_sample_id=stable_sample_id(source_bytes),
        label=label,
        source_bytes=source_bytes,
    )


def _published_count(dataset: str, published_split: str) -> int:
    config = _dataset_config(dataset)
    if published_split == "train":
        return int(config["train_images"]) + int(config["val_images"])
    if published_split == "test":
        return int(config["test_images"])
    raise ValueError(
        f"unsupported published split {published_split!r}; expected train or test"
    )


def _validate_sample_count(
    dataset: str,
    published_split: str,
    actual: int,
) -> None:
    expected = _published_count(dataset, published_split)
    if actual != expected:
        raise DatasetAdapterError(
            f"{dataset}/{published_split}: found {actual} records, expected {expected}"
        )


class _ImagenetteAdapter:
    name = "imagenette160"
    loader_name = "torchvision_datasets_imagenette"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _loader(self, published_split: str) -> Imagenette:
        if published_split == "train":
            split = "train"
        elif published_split == "test":
            split = "val"
        else:
            raise ValueError(
                f"unsupported published split {published_split!r}; "
                "expected train or test"
            )
        config = _dataset_config(self.name)
        return Imagenette(
            self.root,
            split=split,
            size=str(config["loader_size_arg"]),
            download=False,
            loader=lambda path: path,
        )

    def class_mapping(self) -> Mapping[str, int]:
        loader = self._loader("train")
        identifiers = list(loader.wnids)
        if identifiers != sorted(identifiers):
            raise DatasetAdapterError(
                "imagenette160: authoritative directory identifiers are not sorted"
            )
        expected = _validated_class_mapping(self.name, identifiers)
        if dict(loader.wnid_to_idx) != expected:
            raise DatasetAdapterError(
                "imagenette160: Torchvision class mapping is inconsistent"
            )
        return expected

    def iter_source_samples(self, published_split: str) -> Iterator[SourceSample]:
        loader = self._loader(published_split)
        mapping = self.class_mapping()
        count = 0
        for path_text, loader_label in loader._samples:
            path = Path(path_text)
            class_identifier = path.parent.name
            try:
                label = mapping[class_identifier]
            except KeyError:
                raise DatasetAdapterError(
                    f"imagenette160: unknown class directory {class_identifier!r}"
                ) from None
            if int(loader_label) != label:
                raise DatasetAdapterError(
                    "imagenette160: sample label disagrees with directory mapping"
                )
            yield _sample(self.name, label, path.read_bytes())
            count += 1
        _validate_sample_count(self.name, published_split, count)


class _STL10Adapter:
    name = "stl10"
    loader_name = "torchvision_datasets_stl10"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.base = self.root / STL10.base_folder

    def _class_names(self) -> list[str]:
        path = self.base / STL10.class_names_file
        try:
            names = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DatasetAdapterError(
                f"stl10: cannot read authoritative class metadata {path}: {exc}"
            ) from exc
        return names

    def class_mapping(self) -> Mapping[str, int]:
        return _validated_class_mapping(self.name, self._class_names())

    def iter_source_samples(self, published_split: str) -> Iterator[SourceSample]:
        self.class_mapping()
        if published_split == "train":
            data_filename = STL10.train_list[0][0]
            label_filename = STL10.train_list[1][0]
        elif published_split == "test":
            data_filename = STL10.test_list[0][0]
            label_filename = STL10.test_list[1][0]
        else:
            raise ValueError(
                f"unsupported published split {published_split!r}; "
                "expected train or test"
            )

        data_path = self.base / data_filename
        label_path = self.base / label_filename
        try:
            labels = label_path.read_bytes()
            stream = data_path.open("rb")
        except OSError as exc:
            raise DatasetAdapterError(
                f"stl10: cannot open published {published_split} source: {exc}"
            ) from exc

        record_bytes = int(_dataset_config(self.name)["n"])
        count = 0
        with stream:
            for encoded_label in labels:
                source_bytes = stream.read(record_bytes)
                if len(source_bytes) != record_bytes:
                    raise DatasetAdapterError(
                        f"stl10: truncated source record {count} in {data_path}"
                    )
                label = int(encoded_label) - 1
                yield _sample(self.name, label, source_bytes)
                count += 1
            if stream.read(1):
                raise DatasetAdapterError(
                    f"stl10: source file contains more records than labels: {data_path}"
                )
        _validate_sample_count(self.name, published_split, count)


class _CIFAR10Adapter:
    name = "cifar10"
    loader_name = "torchvision_datasets_cifar10"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.base = self.root / CIFAR10.base_folder

    @staticmethod
    def _pickle(path: Path) -> Mapping[str, Any]:
        try:
            with path.open("rb") as stream:
                value = pickle.load(stream, encoding="latin1")
        except (OSError, pickle.UnpicklingError) as exc:
            raise DatasetAdapterError(f"cifar10: cannot read {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DatasetAdapterError(f"cifar10: expected mapping in {path}")
        return value

    def _class_names(self) -> list[str]:
        metadata = self._pickle(self.base / str(CIFAR10.meta["filename"]))
        names = metadata.get(str(CIFAR10.meta["key"]))
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise DatasetAdapterError(
                "cifar10: batches.meta has invalid authoritative label names"
            )
        return names

    def class_mapping(self) -> Mapping[str, int]:
        return _validated_class_mapping(self.name, self._class_names())

    def iter_source_samples(self, published_split: str) -> Iterator[SourceSample]:
        self.class_mapping()
        if published_split == "train":
            files = CIFAR10.train_list
        elif published_split == "test":
            files = CIFAR10.test_list
        else:
            raise ValueError(
                f"unsupported published split {published_split!r}; "
                "expected train or test"
            )

        expected_width = int(_dataset_config(self.name)["n"])
        count = 0
        for filename, _md5 in files:
            entry = self._pickle(self.base / filename)
            data = np.asarray(entry.get("data"), dtype=np.uint8)
            labels = entry.get("labels", entry.get("fine_labels"))
            if (
                data.ndim != 2
                or data.shape[1] != expected_width
                or not isinstance(labels, list)
                or len(labels) != data.shape[0]
            ):
                raise DatasetAdapterError(
                    f"cifar10: invalid data/label payload in {filename}"
                )
            for row, raw_label in zip(data, labels, strict=True):
                yield _sample(
                    self.name,
                    int(raw_label),
                    np.ascontiguousarray(row).tobytes(),
                )
                count += 1
        _validate_sample_count(self.name, published_split, count)


def _imagenette_decoder(source_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(source_bytes)) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _stl10_decoder(source_bytes: bytes) -> np.ndarray:
    config = _dataset_config("stl10")
    height, width, channels = (int(value) for value in config["image_size"])
    expected = int(config["n"])
    if len(source_bytes) != expected:
        raise DatasetAdapterError(
            f"stl10: source record has {len(source_bytes)} bytes, expected {expected}"
        )
    planar = np.frombuffer(source_bytes, dtype=np.uint8).reshape(
        channels,
        height,
        width,
    )
    return np.ascontiguousarray(np.transpose(planar, (2, 1, 0)))


def _cifar10_decoder(source_bytes: bytes) -> np.ndarray:
    config = _dataset_config("cifar10")
    height, width, channels = (int(value) for value in config["image_size"])
    expected = int(config["n"])
    if len(source_bytes) != expected:
        raise DatasetAdapterError(
            f"cifar10: source record has {len(source_bytes)} bytes, expected {expected}"
        )
    planar = np.frombuffer(source_bytes, dtype=np.uint8).reshape(
        channels,
        height,
        width,
    )
    return np.ascontiguousarray(np.transpose(planar, (1, 2, 0)))


_ADAPTER_TYPES: dict[str, type[DatasetAdapter]] = {
    "imagenette160": _ImagenetteAdapter,
    "stl10": _STL10Adapter,
    "cifar10": _CIFAR10Adapter,
}

_DECODERS: dict[str, Callable[[bytes], Image.Image | np.ndarray]] = {
    "imagenette160": _imagenette_decoder,
    "stl10": _stl10_decoder,
    "cifar10": _cifar10_decoder,
}


def _configured_dataset_names() -> set[str]:
    datasets = get("datasets")
    return {
        name
        for name, value in datasets.items()
        if isinstance(value, Mapping) and "loader" in value
    }


def _validate_adapter_coverage(
    adapter_types: Mapping[str, type[DatasetAdapter]] | None = None,
) -> tuple[str, ...]:
    selected = _ADAPTER_TYPES if adapter_types is None else adapter_types
    registered = set(selected)
    configured = _configured_dataset_names()
    if registered != configured:
        missing = sorted(configured - registered)
        extra = sorted(registered - configured)
        raise DatasetAdapterError(
            f"dataset adapter/config mismatch: missing={missing}, extra={extra}"
        )
    for name, adapter_type in selected.items():
        configured_loader = _dataset_config(name)["loader"]
        if adapter_type.loader_name != configured_loader:
            raise DatasetAdapterError(
                f"{name}: adapter loader {adapter_type.loader_name!r} does not "
                f"match configured loader {configured_loader!r}"
            )
    return tuple(sorted(configured))


def _adapter(dataset: str, root: Path) -> DatasetAdapter:
    _validate_adapter_coverage()
    try:
        adapter_type = _ADAPTER_TYPES[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}") from None
    return adapter_type(Path(root))


def _registered_source_decoders() -> dict[str, Callable[[bytes], Any]]:
    names = set(_validate_adapter_coverage())
    if set(_DECODERS) != names:
        raise DatasetAdapterError(
            "source decoder registrations do not match dataset adapters"
        )
    return dict(_DECODERS)
