"""Archive fetching, SHA-256 verification, and extraction provenance (SR-20)."""

from __future__ import annotations

import hashlib
import os
import re
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
_HTTP_OK = 200  # literal-ok: HTTP protocol status code, not an experiment setting
_HTTP_PARTIAL_CONTENT = 206  # literal-ok: HTTP protocol status code, not an experiment setting


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


def _required_extraction_paths(dataset: str) -> tuple[Path, ...]:
    """Return opaque top-level payload paths required before adapter use."""

    config = _dataset_config(dataset)
    loader = str(config["loader"])
    if loader == "torchvision_datasets_imagenette":
        base = Path("imagenette2-160")
        return (base / "train", base / "val")
    if loader == "torchvision_datasets_stl10":
        base = Path("stl10_binary")
        return (
            base / "class_names.txt",
            base / "train_X.bin",
            base / "train_y.bin",
            base / "test_X.bin",
            base / "test_y.bin",
        )
    if loader == "torchvision_datasets_cifar10":
        base = Path("cifar-10-batches-py")
        return (
            base / "batches.meta",
            *(base / f"data_batch_{index}" for index in range(1, 6)),  # literal-ok: CIFAR-10 archive has five fixed train members
            base / "test_batch",
        )
    raise ProvenanceError(f"{dataset}: unsupported extraction loader {loader!r}")


def verify_extracted_dataset(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> ArchiveProvenance:
    """Fail closed unless extraction is bound to the verified configured archive."""

    provenance = verify_archive(dataset, repo_root)
    root = dataset_root(dataset, repo_root)
    if not root.is_dir():
        raise ProvenanceError(
            f"{dataset}: extraction path is missing: {root}; expected archive SHA-256 "
            f"{provenance.sha256}"
        )
    marker = root / ".archive-sha256"
    try:
        observed = marker.read_bytes()
    except FileNotFoundError:
        raise ProvenanceError(
            f"{dataset}: extraction marker is missing: {marker}; expected archive "
            f"SHA-256 {provenance.sha256}"
        ) from None
    expected_marker = f"{provenance.sha256}\n".encode("ascii")
    if not re.fullmatch(rb"[0-9a-f]{64}\n", observed) or observed != expected_marker:
        display = observed.decode("ascii", errors="backslashreplace")
        raise ProvenanceError(
            f"{dataset}: extraction marker mismatch at {marker}: observed "
            f"{display!r}, expected archive SHA-256 {provenance.sha256}"
        )
    missing = [str(root / path) for path in _required_extraction_paths(dataset) if not (root / path).exists()]
    if missing:
        raise ProvenanceError(
            f"{dataset}: extraction path {root} is missing required content: "
            f"{missing}; expected archive SHA-256 {provenance.sha256}"
        )
    return provenance


def fetch_archive(
    dataset: str,
    repo_root: Path = REPO_ROOT,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> ArchiveProvenance:
    """Fetch exactly the configured URL with verified, resumable finalization."""

    config = _dataset_config(dataset)
    destination = archive_path(dataset, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if destination.exists():
        if not destination.is_file():
            raise ProvenanceError(f"{dataset}: archive destination is not a file: {destination}")
        return verify_archive(dataset, repo_root)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "capstone-dataset-provenance/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(str(config["source_url"]), headers=headers)
    expected_total = int(config["archive_bytes"])
    try:
        with opener(request) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = getattr(response, "getcode", lambda: _HTTP_OK)()
            response_headers = getattr(response, "headers", {})
            mode = "ab" if offset else "wb"
            expected_remaining = expected_total
            if offset:
                if status == _HTTP_OK:
                    mode = "wb"
                    offset = 0
                elif status == _HTTP_PARTIAL_CONTENT:
                    content_range = response_headers.get("Content-Range")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
                    if match is None or int(match.group(1)) != offset:
                        raise ProvenanceError(
                            f"{dataset}: resume Content-Range does not start at {offset}: {content_range!r}"
                        )
                    range_start, range_end, range_total = (int(value) for value in match.groups())
                    if range_total != expected_total:
                        raise ProvenanceError(
                            f"{dataset}: resume Content-Range total {range_total} != expected {expected_total}"
                        )
                    expected_remaining = expected_total - offset
                    if range_end < range_start:
                        raise ProvenanceError(
                            f"{dataset}: resume Content-Range end {range_end} precedes start {range_start}"
                        )
                    if range_end != expected_total - 1:
                        raise ProvenanceError(
                            f"{dataset}: resume Content-Range end {range_end} != expected {expected_total - 1}"
                        )
                    if range_end - range_start + 1 != expected_remaining:
                        raise ProvenanceError(
                            f"{dataset}: resume Content-Range length {range_end - range_start + 1} "
                            f"!= expected {expected_remaining}"
                        )
                else:
                    raise ProvenanceError(f"{dataset}: expected HTTP 206 or 200 for resume, got {status}")
            elif status != _HTTP_OK:
                raise ProvenanceError(
                    f"{dataset}: expected HTTP {_HTTP_OK} for fresh fetch, got {status}"
                )
            declared = response_headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_length = int(declared)
                except (TypeError, ValueError):
                    raise ProvenanceError(f"{dataset}: invalid Content-Length {declared!r}") from None
                if declared_length != expected_remaining:
                    raise ProvenanceError(
                        f"{dataset}: Content-Length {declared} != expected {expected_remaining}"
                    )
            with partial.open(mode) as output:
                written = 0
                while chunk := response.read(_HASH_CHUNK_BYTES):
                    written += len(chunk)
                    if offset + written > expected_total:
                        raise ProvenanceError(f"{dataset}: transfer exceeds expected byte length {expected_total}")
                    output.write(chunk)
            if written != expected_remaining or partial.stat().st_size != expected_total:
                raise ProvenanceError(
                    f"{dataset}: transfer ended at {partial.stat().st_size} bytes; expected {expected_total}"
                )
        with partial.open("rb") as stream:
            measured_bytes, measured_hash = _hash_stream(stream)
        if measured_bytes != expected_total or measured_hash != str(config["archive_sha256"]):
            raise ProvenanceError(
                f"{dataset}: completed partial does not match configured archive provenance"
            )
        os.replace(partial, destination)
    except BaseException:
        raise
    return verify_archive(dataset, repo_root)


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
        if marker.is_file():
            verify_extracted_dataset(dataset, repo_root)
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
    verify_extracted_dataset(dataset, repo_root)
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
