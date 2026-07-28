"""Archive fetching, SHA-256 verification, and extraction provenance (SR-20)."""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from config.params import REPO_ROOT, get

_PENDING_PREFIX = "pending_"
_HASH_CHUNK_BYTES = 1024 * 1024  # literal-ok: one MiB streaming I/O chunk


class ProvenanceError(RuntimeError):
    """Raised before use when archive provenance is absent or inconsistent."""


@dataclass(frozen=True)
class ArchiveProvenance:
    dataset: str
    url: str
    filename: str
    byte_length: int
    sha256: str
    path: Path


def archive_directory(repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root) / str(get("datasets.archive_dir"))


def extracted_directory(repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root) / str(get("datasets.extracted_dir"))


def dataset_root(dataset: str, repo_root: Path = REPO_ROOT) -> Path:
    return extracted_directory(repo_root) / dataset


def archive_path(dataset: str, repo_root: Path = REPO_ROOT) -> Path:
    config = _dataset_config(dataset)
    return archive_directory(repo_root) / str(config["archive_filename"])


def _dataset_config(dataset: str) -> Mapping[str, Any]:
    try:
        config = get(f"datasets.{dataset}")
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}") from None
    if not isinstance(config, Mapping) or "loader" not in config:
        raise ValueError(f"unknown dataset {dataset!r}")
    return config


def _hash_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_length = 0
    while chunk := stream.read(_HASH_CHUNK_BYTES):
        byte_length += len(chunk)
        digest.update(chunk)
    return byte_length, digest.hexdigest()


def measure_archive(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> ArchiveProvenance:
    path = archive_path(dataset, repo_root)
    try:
        with path.open("rb") as stream:
            byte_length, sha256 = _hash_stream(stream)
    except OSError as exc:
        raise ProvenanceError(f"{dataset}: cannot read archive {path}: {exc}") from exc
    config = _dataset_config(dataset)
    return ArchiveProvenance(
        dataset=dataset,
        url=str(config["source_url"]),
        filename=str(config["archive_filename"]),
        byte_length=byte_length,
        sha256=sha256,
        path=path,
    )


def _is_pending(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_PENDING_PREFIX)


def verify_archive(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> ArchiveProvenance:
    config = _dataset_config(dataset)
    expected_hash = config["archive_sha256"]
    expected_bytes = config["archive_bytes"]
    if _is_pending(expected_hash) or _is_pending(expected_bytes):
        raise ProvenanceError(
            f"{dataset}: pending archive provenance cannot authorize dataset use"
        )
    measured = measure_archive(dataset, repo_root)
    if measured.byte_length != int(expected_bytes):
        raise ProvenanceError(
            f"{dataset}: archive byte length mismatch: "
            f"{measured.byte_length} != {expected_bytes}"
        )
    if measured.sha256 != str(expected_hash):
        raise ProvenanceError(
            f"{dataset}: archive SHA-256 mismatch: "
            f"{measured.sha256} != {expected_hash}"
        )
    return measured


def fetch_archive(
    dataset: str,
    repo_root: Path = REPO_ROOT,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> ArchiveProvenance:
    """Fetch exactly the configured normative URL and measure the resulting file."""

    config = _dataset_config(dataset)
    destination = archive_path(dataset, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return measure_archive(dataset, repo_root)

    request = urllib.request.Request(
        str(config["source_url"]),
        headers={"User-Agent": "capstone-dataset-provenance/1"},
    )
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with opener(request) as response, partial.open("wb") as output:
            while chunk := response.read(_HASH_CHUNK_BYTES):
                output.write(chunk)
        os.replace(partial, destination)
    except BaseException:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return measure_archive(dataset, repo_root)


def extract_verified_archive(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Extract only after SHA-256 and byte-length verification succeed."""

    provenance = verify_archive(dataset, repo_root)
    destination = dataset_root(dataset, repo_root)
    marker_name = ".archive-sha256"
    marker = destination / marker_name
    if destination.exists():
        if marker.is_file() and marker.read_text(encoding="ascii").strip() == (
            provenance.sha256
        ):
            return destination
        raise ProvenanceError(
            f"{dataset}: extracted destination exists without matching provenance "
            f"marker: {destination}"
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=parent) as temp_text:
        temp = Path(temp_text)
        try:
            with tarfile.open(provenance.path, mode="r:*") as archive:
                archive.extractall(temp, filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise ProvenanceError(
                f"{dataset}: cannot extract verified archive {provenance.path}: {exc}"
            ) from exc
        (temp / marker_name).write_text(
            f"{provenance.sha256}\n",
            encoding="ascii",
            newline="\n",
        )
        temp.rename(destination)
    return destination


def configured_datasets() -> tuple[str, ...]:
    datasets = get("datasets")
    return tuple(
        sorted(
            name
            for name, value in datasets.items()
            if isinstance(value, Mapping) and "loader" in value
        )
    )


def provision_archives(
    repo_root: Path = REPO_ROOT,
    *,
    measure_only: bool,
) -> Iterator[ArchiveProvenance]:
    """Fetch all normative archives, then optionally verify and extract them."""

    for dataset in configured_datasets():
        measured = fetch_archive(dataset, repo_root)
        if measure_only:
            yield measured
            continue
        verified = verify_archive(dataset, repo_root)
        extract_verified_archive(dataset, repo_root)
        yield verified
