"""Source-payload, decoder, class-mapping, and registry checks."""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config.params import REPO_ROOT, get
from data import adapters
from data.adapters import (
    DatasetAdapterError,
    _adapter,
    _cifar10_decoder,
    _imagenette_decoder,
    _stl10_decoder,
    _validate_adapter_coverage,
)
from data.identity import stable_sample_id
from data.registry import (
    DatasetRegistryError,
    available_datasets,
    load_dataset,
    manifest_path,
    manifest_sha256,
)
from data.provenance import dataset_root


def test_all_configured_datasets_have_exactly_one_adapter():
    configured = {
        name
        for name, value in get("datasets").items()
        if isinstance(value, dict) and "loader" in value
    }

    assert set(available_datasets()) == configured
    assert set(adapters._ADAPTER_TYPES) == configured
    assert set(adapters._registered_source_decoders()) == configured


def test_adapter_config_set_mismatch_fails_loudly():
    missing = dict(adapters._ADAPTER_TYPES)
    missing.pop("stl10")
    with pytest.raises(DatasetAdapterError, match="missing=.*stl10"):
        _validate_adapter_coverage(missing)

    extra = dict(adapters._ADAPTER_TYPES)
    extra["not-configured"] = adapters._ADAPTER_TYPES["stl10"]
    with pytest.raises(DatasetAdapterError, match="extra=.*not-configured"):
        _validate_adapter_coverage(extra)


@pytest.mark.parametrize("dataset", ["imagenette160", "stl10", "cifar10"])
def test_all_adapters_instantiate_through_one_path(
    synthetic_dataset_repo: Path,
    dataset: str,
):
    instance = _adapter(dataset, dataset_root(dataset, synthetic_dataset_repo))

    assert instance.name == dataset
    assert sorted(instance.class_mapping().values()) == list(range(10))


def test_unknown_dataset_and_unsupported_splits_fail(
    synthetic_dataset_repo: Path,
):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("unknown", "train", synthetic_dataset_repo)
    with pytest.raises(ValueError, match="unsupported split"):
        load_dataset("cifar10", "dev", synthetic_dataset_repo)
    with pytest.raises(DatasetRegistryError, match="SR-22 and G-12"):
        load_dataset("cifar10", "test", synthetic_dataset_repo)


@pytest.mark.parametrize("dataset", ["imagenette160", "stl10", "cifar10"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_public_train_and_val_paths_canonicalize_source_records(
    synthetic_dataset_repo: Path,
    dataset: str,
    split: str,
):
    loaded = load_dataset(dataset, split, synthetic_dataset_repo)
    source = loaded.source_sample(0)
    product, label = loaded[0]

    assert len(loaded) == 10
    assert source.dataset == dataset
    assert source.stable_sample_id == stable_sample_id(source.source_bytes)
    assert product.stable_sample_id == source.stable_sample_id
    assert product.canonical_image.shape == tuple(get(f"datasets.{dataset}.image_size"))
    assert label == source.label


def test_manifest_public_helpers_reproduce_pinned_hash(
    synthetic_dataset_repo: Path,
):
    path = manifest_path("stl10", synthetic_dataset_repo)
    actual = manifest_sha256("stl10", synthetic_dataset_repo)

    assert path.name == get("datasets.stl10.manifest_filename")
    assert actual == hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == get("datasets.stl10.manifest_sha256")


def test_source_payloads_are_exact_predecode_records(
    synthetic_dataset_repo: Path,
):
    for dataset in ("imagenette160", "stl10", "cifar10"):
        instance = _adapter(dataset, dataset_root(dataset, synthetic_dataset_repo))
        sample = next(instance.iter_source_samples("train"))
        assert sample.stable_sample_id == hashlib.sha256(
            sample.source_bytes
        ).hexdigest()[:16]

    imagenette = _adapter(
        "imagenette160",
        dataset_root("imagenette160", synthetic_dataset_repo),
    )
    imagenette_sample = next(imagenette.iter_source_samples("train"))
    assert imagenette_sample.source_bytes.startswith(b"\xff\xd8")

    stl = _adapter("stl10", dataset_root("stl10", synthetic_dataset_repo))
    stl_sample = next(stl.iter_source_samples("train"))
    assert len(stl_sample.source_bytes) == get("datasets.stl10.n")

    cifar = _adapter("cifar10", dataset_root("cifar10", synthetic_dataset_repo))
    cifar_sample = next(cifar.iter_source_samples("train"))
    assert len(cifar_sample.source_bytes) == get("datasets.cifar10.n")


def test_labels_and_preprocessing_do_not_enter_source_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"one exact container record"
    before = stable_sample_id(payload)

    assert before == stable_sample_id(payload)
    assert before != stable_sample_id(payload + b"-different")
    monkeypatch.setitem(
        get("preprocessing"),
        "canonical_image",
        "a-deliberately-different-preprocessing-policy",
    )
    assert stable_sample_id(payload) == before


def test_stl_decoder_reproduces_torchvision_axis_transpose():
    height, width, channels = get("datasets.stl10.image_size")
    planar = np.arange(channels * height * width, dtype=np.uint16)
    source = (planar % 256).astype(np.uint8).tobytes()

    actual = _stl10_decoder(source)
    expected = np.frombuffer(source, dtype=np.uint8).reshape(
        channels,
        height,
        width,
    ).transpose(2, 1, 0)

    np.testing.assert_array_equal(actual, expected)


def test_imagenette_decoder_reproduces_encoded_rgb_pixels():
    source_array = np.arange(8 * 11 * 3, dtype=np.uint8).reshape(8, 11, 3)
    encoded = io.BytesIO()
    Image.fromarray(source_array).save(encoded, format="PNG")
    source_bytes = encoded.getvalue()

    actual = _imagenette_decoder(source_bytes)
    with Image.open(io.BytesIO(source_bytes)) as image:
        expected = np.asarray(image.convert("RGB"), dtype=np.uint8)

    np.testing.assert_array_equal(actual, expected)


def test_cifar_decoder_reproduces_planar_chw_to_rgb_hwc():
    height, width, channels = get("datasets.cifar10.image_size")
    planar = np.arange(channels * height * width, dtype=np.uint16)
    source = (planar % 256).astype(np.uint8).tobytes()

    actual = _cifar10_decoder(source)
    expected = np.frombuffer(source, dtype=np.uint8).reshape(
        channels,
        height,
        width,
    ).transpose(1, 2, 0)

    np.testing.assert_array_equal(actual, expected)


def test_class_mappings_follow_authoritative_metadata_and_not_enumeration(
    synthetic_dataset_repo: Path,
):
    imagenette = _adapter(
        "imagenette160",
        dataset_root("imagenette160", synthetic_dataset_repo),
    )
    stl = _adapter("stl10", dataset_root("stl10", synthetic_dataset_repo))
    cifar = _adapter("cifar10", dataset_root("cifar10", synthetic_dataset_repo))

    assert list(imagenette.class_mapping()) == sorted(imagenette.class_mapping())
    assert list(stl.class_mapping()) == [f"class-{index}" for index in range(10)]
    assert list(cifar.class_mapping()) == [f"class-{index}" for index in range(10)]
    for instance in (imagenette, stl, cifar):
        first = dict(instance.class_mapping())
        second = dict(instance.class_mapping())
        assert first == second
        assert sorted(first.values()) == list(range(10))


def test_only_canonicalize_source_constructs_canonical_products():
    source_root = REPO_ROOT / "src"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        if path.name == "preprocessing.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"_CanonicalProduct", "_canonicalize_decoded"}:
                    violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == []
