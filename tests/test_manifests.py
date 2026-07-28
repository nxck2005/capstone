"""Deterministic split-carve and canonical manifest checks (SR-17)."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from config.params import get
from data import adapters, manifests
from data.adapters import _adapter
from data.manifests import (
    ManifestError,
    ManifestRow,
    canonical_manifest_bytes,
    check_manifest,
    manifest_path,
    materialize_manifest_bytes,
    validate_manifest_bytes,
)
from data.provenance import dataset_root


def test_canonical_csv_bytes_are_exact():
    rows = [
        ManifestRow("b" * 16, 2, "test"),
        ManifestRow("a" * 16, 1, "train"),
    ]

    assert canonical_manifest_bytes(rows) == (
        b"stable_sample_id,label,split\n"
        b"aaaaaaaaaaaaaaaa,1,train\n"
        b"bbbbbbbbbbbbbbbb,2,test\n"
    )


@pytest.mark.parametrize("dataset", ["imagenette160", "stl10", "cifar10"])
def test_synthetic_manifests_have_exact_counts_stratification_and_disjointness(
    synthetic_dataset_repo: Path,
    dataset: str,
):
    rows = validate_manifest_bytes(
        dataset,
        manifest_path(dataset, synthetic_dataset_repo).read_bytes(),
    )
    by_split = {
        split: {row.stable_sample_id for row in rows if row.split == split}
        for split in ("train", "val", "test")
    }

    assert {split: len(ids) for split, ids in by_split.items()} == {
        "train": 10,
        "val": 10,
        "test": 10,
    }
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])
    assert Counter(row.label for row in rows if row.split == "val") == Counter(
        range(10)
    )


def test_manifest_generation_is_invariant_to_reversed_and_shuffled_enumeration(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real = _adapter(
        "cifar10",
        dataset_root("cifar10", synthetic_dataset_repo),
    )
    train = list(real.iter_source_samples("train"))
    test = list(real.iter_source_samples("test"))

    class Reordered:
        def __init__(self, train_order, test_order):
            self.train_order = train_order
            self.test_order = test_order

        def class_mapping(self):
            return real.class_mapping()

        def iter_source_samples(self, split):
            return iter(self.train_order if split == "train" else self.test_order)

    monkeypatch.setattr(
        manifests,
        "_adapter",
        lambda _dataset, _root: Reordered(train, test),
    )
    forward = materialize_manifest_bytes("cifar10", synthetic_dataset_repo)

    shuffled_train = list(train)
    shuffled_test = list(test)
    random.Random(19).shuffle(shuffled_train)
    random.Random(23).shuffle(shuffled_test)
    monkeypatch.setattr(
        manifests,
        "_adapter",
        lambda _dataset, _root: Reordered(shuffled_train, shuffled_test),
    )
    shuffled = materialize_manifest_bytes("cifar10", synthetic_dataset_repo)
    monkeypatch.setattr(
        manifests,
        "_adapter",
        lambda _dataset, _root: Reordered(
            list(reversed(train)),
            list(reversed(test)),
        ),
    )
    reversed_payload = materialize_manifest_bytes(
        "cifar10",
        synthetic_dataset_repo,
    )

    assert forward == shuffled == reversed_payload


def test_validation_carve_matches_independent_pcg64_choice_by_hand(
    synthetic_dataset_repo: Path,
):
    adapter = _adapter(
        "stl10",
        dataset_root("stl10", synthetic_dataset_repo),
    )
    training = sorted(
        adapter.iter_source_samples("train"),
        key=lambda sample: sample.stable_sample_id,
    )
    rng = np.random.default_rng(get("evaluation.split_seed"))
    expected: set[str] = set()
    for label in range(10):
        candidates = [sample for sample in training if sample.label == label]
        chosen = rng.choice(len(candidates), size=1, replace=False)
        expected.add(candidates[int(chosen[0])].stable_sample_id)

    rows = validate_manifest_bytes(
        "stl10",
        manifest_path("stl10", synthetic_dataset_repo).read_bytes(),
    )
    actual = {row.stable_sample_id for row in rows if row.split == "val"}

    assert actual == expected


def test_preprocessing_changes_do_not_change_manifest(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    before = materialize_manifest_bytes("imagenette160", synthetic_dataset_repo)
    monkeypatch.setitem(
        get("preprocessing"),
        "resize_interpolation",
        "nearest",
    )
    after = materialize_manifest_bytes("imagenette160", synthetic_dataset_repo)

    assert before == after


def test_manifest_check_rejects_a_mismatched_pinned_hash(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        get("datasets.imagenette160"),
        "manifest_sha256",
        "0" * 64,
    )

    with pytest.raises(ManifestError, match="manifest SHA-256 mismatch"):
        check_manifest("imagenette160", synthetic_dataset_repo)


def test_duplicate_stable_ids_fail(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real = _adapter("stl10", dataset_root("stl10", synthetic_dataset_repo))
    train = list(real.iter_source_samples("train"))
    test = list(real.iter_source_samples("test"))
    train.append(train[0])

    class DuplicateAdapter:
        def class_mapping(self):
            return real.class_mapping()

        def iter_source_samples(self, split):
            return iter(train if split == "train" else test)

    monkeypatch.setattr(
        manifests,
        "_adapter",
        lambda _dataset, _root: DuplicateAdapter(),
    )
    with pytest.raises(ManifestError, match="duplicate stable sample ID"):
        materialize_manifest_bytes("stl10", synthetic_dataset_repo)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.replace(
                b"stable_sample_id,label,split",
                b"path,label,split",
                1,
            ),
            "header",
        ),
        (
            lambda payload: payload.replace(b",train\n", b",invalid\n", 1),
            "invalid split",
        ),
        (
            lambda payload: payload.replace(b",0,train\n", b",+0,train\n", 1),
            "label",
        ),
        (
            lambda payload: payload.replace(b"\n", b"\r\n"),
            "non-LF",
        ),
    ],
)
def test_malformed_manifest_rows_fail(
    synthetic_dataset_repo: Path,
    mutation,
    message: str,
):
    payload = manifest_path("cifar10", synthetic_dataset_repo).read_bytes()
    with pytest.raises(ManifestError, match=message):
        validate_manifest_bytes("cifar10", mutation(payload))


def test_incorrect_manifest_label_fails_registry_source_check(
    synthetic_dataset_repo: Path,
):
    path = manifest_path("cifar10", synthetic_dataset_repo)
    rows = list(validate_manifest_bytes("cifar10", path.read_bytes()))
    index = next(
        index
        for index, row in enumerate(rows)
        if row.split == "train" and row.label != 9
    )
    row = rows[index]
    rows[index] = ManifestRow(row.stable_sample_id, row.label + 1, row.split)
    path.write_bytes(canonical_manifest_bytes(rows))
    get("datasets.cifar10")["manifest_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    from data.registry import DatasetRegistryError, load_dataset

    with pytest.raises(DatasetRegistryError, match="manifest label"):
        load_dataset("cifar10", "train", synthetic_dataset_repo)


def test_manifest_contains_no_path_or_positional_index(
    synthetic_dataset_repo: Path,
):
    payload = manifest_path("imagenette160", synthetic_dataset_repo).read_bytes()
    header = payload.splitlines()[0]

    assert header == b"stable_sample_id,label,split"
    assert b"/" not in payload
    assert b"\\" not in payload
    assert b"source_index" not in payload
    assert b"path" not in header


def test_provenance_only_test_scan_calls_no_decoder_or_model_path(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = Counter()

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"forbidden provenance call: {name}")

        return fail

    monkeypatch.setattr(
        adapters,
        "_DECODERS",
        {name: forbidden(f"decoder:{name}") for name in adapters._DECODERS},
    )
    import data.preprocessing as preprocessing
    import data.registry as registry

    monkeypatch.setattr(
        preprocessing,
        "canonicalize_source",
        forbidden("canonicalize_source"),
    )
    monkeypatch.setattr(registry, "load_dataset", forbidden("model_loader"))

    payload = materialize_manifest_bytes("stl10", synthetic_dataset_repo)

    assert payload
    assert calls == Counter()


@pytest.mark.parametrize(
    ("dataset", "counts"),
    [
        ("imagenette160", {"train": 8469, "val": 1000, "test": 3925}),
        ("stl10", {"train": 4500, "val": 500, "test": 8000}),
        ("cifar10", {"train": 45000, "val": 5000, "test": 10000}),
    ],
)
def test_committed_manifest_counts_and_sha256_reproduce(dataset: str, counts):
    path = manifest_path(dataset)
    rows = validate_manifest_bytes(dataset, path.read_bytes())

    assert Counter(row.split for row in rows) == Counter(counts)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == get(
        f"datasets.{dataset}.manifest_sha256"
    )
