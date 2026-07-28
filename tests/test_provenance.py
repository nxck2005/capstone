"""Archive provenance and pre-freeze isolation checks (SR-20, SR-22)."""

from __future__ import annotations

import io
import hashlib
import shutil
import tarfile
from pathlib import Path

import pytest

from config.params import get
from data.provenance import (
    ProvenanceError,
    archive_path,
    dataset_root,
    extract_verified_archive,
    fetch_archive,
    measure_archive,
    verify_archive,
)
from data.registry import DatasetRegistryError, load_dataset
from data.manifests import materialize_manifest_bytes


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_fetch_requests_the_exact_normative_url(
    synthetic_dataset_repo: Path,
):
    dataset = "imagenette160"
    path = archive_path(dataset, synthetic_dataset_repo)
    path.unlink()
    requested: list[str] = []

    def opener(request):
        requested.append(request.full_url)
        return _Response(b"new measured archive bytes")

    measured = fetch_archive(dataset, synthetic_dataset_repo, opener=opener)

    assert requested == [get(f"datasets.{dataset}.source_url")]
    assert measured.filename == get(f"datasets.{dataset}.archive_filename")
    assert measured.byte_length == len(b"new measured archive bytes")
    assert measured.path.read_bytes() == b"new measured archive bytes"


def test_archive_mismatch_fails_before_adapter_or_sample_use(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import data.registry as registry

    path = archive_path("stl10", synthetic_dataset_repo)
    path.write_bytes(path.read_bytes() + b"corrupt")
    calls: list[str] = []
    monkeypatch.setattr(
        registry,
        "_adapter",
        lambda *_args: calls.append("adapter") or None,
    )

    with pytest.raises(DatasetRegistryError, match="archive byte length mismatch"):
        load_dataset("stl10", "train", synthetic_dataset_repo)
    assert calls == []


@pytest.mark.parametrize("field", ["archive_sha256", "archive_bytes"])
def test_no_pending_archive_provenance_passes(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    monkeypatch.setitem(
        get("datasets.cifar10"),
        field,
        "pending_first_fetch_at_W1",
    )

    with pytest.raises(ProvenanceError, match="pending archive provenance"):
        verify_archive("cifar10", synthetic_dataset_repo)


def test_pending_archive_blocks_manifest_scan_before_adapter_use(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import data.manifests as manifests

    monkeypatch.setitem(
        get("datasets.stl10"),
        "archive_sha256",
        "pending_first_fetch_at_W1",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        manifests,
        "_adapter",
        lambda *_args: calls.append("adapter") or None,
    )

    with pytest.raises(ProvenanceError, match="pending archive provenance"):
        materialize_manifest_bytes("stl10", synthetic_dataset_repo)
    assert calls == []


def test_measured_archive_hash_is_independent_of_configured_hash(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    measured = measure_archive("cifar10", synthetic_dataset_repo)
    monkeypatch.setitem(
        get("datasets.cifar10"),
        "archive_sha256",
        "0" * 64,
    )

    assert measure_archive("cifar10", synthetic_dataset_repo) == measured
    with pytest.raises(ProvenanceError, match="SHA-256 mismatch"):
        verify_archive("cifar10", synthetic_dataset_repo)


def test_verified_archive_extracts_only_after_exact_hash(
    synthetic_dataset_repo: Path,
    tmp_path: Path,
):
    dataset = "cifar10"
    archive = archive_path(dataset, synthetic_dataset_repo)
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("verified extraction fixture", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(payload_file, arcname="fixture/payload.txt")
    config = get(f"datasets.{dataset}")
    config["archive_bytes"] = archive.stat().st_size
    config["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = dataset_root(dataset, synthetic_dataset_repo)
    shutil.rmtree(destination)

    extracted = extract_verified_archive(dataset, synthetic_dataset_repo)

    assert (extracted / "fixture/payload.txt").read_text(encoding="utf-8") == (
        "verified extraction fixture"
    )
    assert (extracted / ".archive-sha256").read_text(encoding="ascii").strip() == (
        config["archive_sha256"]
    )


def test_missing_manifest_fails_public_loading(
    synthetic_dataset_repo: Path,
):
    path = (
        synthetic_dataset_repo
        / get("datasets.manifest_dir")
        / get("datasets.imagenette160.manifest_filename")
    )
    path.unlink()

    with pytest.raises(DatasetRegistryError, match="cannot load"):
        load_dataset("imagenette160", "train", synthetic_dataset_repo)


def test_pending_manifest_hash_cannot_authorize_loading(
    synthetic_dataset_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        get("datasets.stl10"),
        "manifest_sha256",
        "pending_manifest_materialization_at_W1",
    )

    with pytest.raises(DatasetRegistryError, match="pending manifest SHA-256"):
        load_dataset("stl10", "train", synthetic_dataset_repo)


def test_production_registry_has_no_parallel_test_loader():
    import data.registry as registry

    public_names = {
        name
        for name in dir(registry)
        if not name.startswith("_")
    }
    assert "load_test_dataset" not in public_names
    assert "load_test_sample" not in public_names
    assert "test_dataset" not in public_names
