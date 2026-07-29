"""Canonical deterministic split-manifest construction and verification."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.params import REPO_ROOT, get
from data.adapters import _adapter
from data.identity import stable_sample_id_width
from data.provenance import dataset_root, verify_extracted_dataset


class ManifestError(RuntimeError):
    """Raised when manifest input or output violates the canonical contract."""


@dataclass(frozen=True, order=True)
class ManifestRow:
    stable_sample_id: str
    label: int
    split: str


def manifest_path(dataset: str, repo_root: Path = REPO_ROOT) -> Path:
    config = _dataset_config(dataset)
    return (
        Path(repo_root)
        / str(get("datasets.manifest_dir"))
        / str(config["manifest_filename"])
    )


def manifest_sha256(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    path = manifest_path(dataset, repo_root)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{dataset}: cannot read manifest {path}: {exc}") from exc
    validate_manifest_bytes(dataset, payload)
    return hashlib.sha256(payload).hexdigest()


def _dataset_config(dataset: str) -> Mapping[str, Any]:
    try:
        config = get(f"datasets.{dataset}")
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}") from None
    if not isinstance(config, Mapping) or "loader" not in config:
        raise ValueError(f"unknown dataset {dataset!r}")
    return config


def _manifest_columns() -> tuple[str, ...]:
    columns = tuple(str(value) for value in get("datasets.manifest_columns"))
    expected = ("stable_sample_id", "label", "split")
    if columns != expected:
        raise NotImplementedError(
            f"unsupported params.datasets.manifest_columns: {columns}"
        )
    return columns


def canonical_manifest_bytes(rows: Iterable[ManifestRow]) -> bytes:
    """Serialize exact UTF-8/LF CSV bytes in global stable-ID order."""

    if get("datasets.manifest_encoding") != "utf-8":
        raise NotImplementedError("only UTF-8 manifest encoding is supported")
    if get("datasets.manifest_newline") != "lf":
        raise NotImplementedError("only LF manifest newlines are supported")
    sorted_rows = sorted(rows, key=lambda row: row.stable_sample_id)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_manifest_columns())
    for row in sorted_rows:
        writer.writerow((row.stable_sample_id, str(row.label), row.split))
    return output.getvalue().encode("utf-8")


def validate_manifest_bytes(
    dataset: str,
    payload: bytes,
) -> tuple[ManifestRow, ...]:
    """Parse strictly and reject every non-canonical or inconsistent manifest."""

    if not isinstance(payload, bytes):
        raise TypeError("manifest payload must be bytes")
    if b"\r" in payload:
        raise ManifestError(f"{dataset}: manifest contains non-LF newlines")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{dataset}: manifest is not UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise ManifestError(f"{dataset}: manifest must end with LF")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = tuple(next(reader))
    except StopIteration:
        raise ManifestError(f"{dataset}: manifest is empty") from None
    if header != _manifest_columns():
        raise ManifestError(
            f"{dataset}: manifest header {header!r} is not {_manifest_columns()!r}"
        )

    config = _dataset_config(dataset)
    class_count = int(config["classes"])
    valid_splits = tuple(str(value) for value in get("datasets.manifest_splits"))
    expected_splits = ("train", "val", "test")
    if valid_splits != expected_splits:
        raise NotImplementedError(
            f"unsupported params.datasets.manifest_splits: {valid_splits}"
        )
    id_pattern = re.compile(rf"[0-9a-f]{{{stable_sample_id_width()}}}")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    for line_number, fields in enumerate(
        reader,
        start=2,  # literal-ok: CSV header occupies line one
    ):
        if len(fields) != len(header):
            raise ManifestError(
                f"{dataset}: line {line_number} has {len(fields)} fields"
            )
        sample_id, label_text, split = fields
        if id_pattern.fullmatch(sample_id) is None:
            raise ManifestError(
                f"{dataset}: line {line_number} has invalid stable sample ID"
            )
        if sample_id in seen:
            raise ManifestError(
                f"{dataset}: duplicate stable sample ID {sample_id}"
            )
        seen.add(sample_id)
        try:
            label = int(label_text)
        except ValueError:
            raise ManifestError(
                f"{dataset}: line {line_number} has invalid decimal label"
            ) from None
        if label_text != str(label) or label not in range(class_count):
            raise ManifestError(
                f"{dataset}: line {line_number} has non-canonical/out-of-range label"
            )
        if split not in valid_splits:
            raise ManifestError(
                f"{dataset}: line {line_number} has invalid split {split!r}"
            )
        rows.append(ManifestRow(sample_id, label, split))

    if [row.stable_sample_id for row in rows] != sorted(seen):
        raise ManifestError(
            f"{dataset}: rows are not globally sorted by stable_sample_id"
        )
    _validate_counts_and_stratification(dataset, rows)
    canonical = canonical_manifest_bytes(rows)
    if payload != canonical:
        raise ManifestError(f"{dataset}: manifest bytes are not canonical")
    return tuple(rows)


def _validate_counts_and_stratification(
    dataset: str,
    rows: Sequence[ManifestRow],
) -> None:
    config = _dataset_config(dataset)
    split_counts = Counter(row.split for row in rows)
    for split in ("train", "val", "test"):
        expected = int(config[f"{split}_images"])
        if split_counts[split] != expected:
            raise ManifestError(
                f"{dataset}: {split} count {split_counts[split]} != {expected}"
            )
    class_count = int(config["classes"])
    val_count = int(config["val_images"])
    if val_count % class_count:
        raise ManifestError(
            f"{dataset}: validation count is not divisible by class count"
        )
    quota = val_count // class_count
    val_labels = Counter(row.label for row in rows if row.split == "val")
    if val_labels != Counter({label: quota for label in range(class_count)}):
        raise ManifestError(
            f"{dataset}: validation split is not exactly stratified"
        )


def materialize_manifest_bytes(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> bytes:
    """Regenerate one manifest without decoding or canonicalizing any image."""

    verify_extracted_dataset(dataset, repo_root)
    adapter = _adapter(dataset, dataset_root(dataset, repo_root))
    mapping = dict(adapter.class_mapping())
    expected_labels = list(range(int(_dataset_config(dataset)["classes"])))
    if sorted(mapping.values()) != expected_labels:
        raise ManifestError(f"{dataset}: inconsistent authoritative class mapping")

    training_rows: list[ManifestRow] = []
    seen: set[str] = set()
    for sample in adapter.iter_source_samples("train"):
        if sample.stable_sample_id in seen:
            raise ManifestError(
                f"{dataset}: duplicate stable sample ID {sample.stable_sample_id}"
            )
        seen.add(sample.stable_sample_id)
        training_rows.append(
            ManifestRow(sample.stable_sample_id, sample.label, "train")
        )
    training_rows.sort(key=lambda row: row.stable_sample_id)

    config = _dataset_config(dataset)
    class_count = int(config["classes"])
    val_count = int(config["val_images"])
    if val_count % class_count:
        raise ManifestError(
            f"{dataset}: validation count {val_count} is not divisible by "
            f"{class_count} classes"
        )
    quota = val_count // class_count
    rng = np.random.default_rng(int(get("evaluation.split_seed")))
    if not isinstance(rng.bit_generator, np.random.PCG64):
        raise ManifestError("numpy.default_rng did not provide PCG64")
    validation_ids: set[str] = set()
    for label in range(class_count):
        candidates = [row for row in training_rows if row.label == label]
        if len(candidates) < quota:
            raise ManifestError(
                f"{dataset}: class {label} has {len(candidates)} records, "
                f"below validation quota {quota}"
            )
        chosen = rng.choice(len(candidates), size=quota, replace=False)
        validation_ids.update(candidates[int(index)].stable_sample_id for index in chosen)

    rows = [
        ManifestRow(
            row.stable_sample_id,
            row.label,
            "val" if row.stable_sample_id in validation_ids else "train",
        )
        for row in training_rows
    ]
    for sample in adapter.iter_source_samples("test"):
        if sample.stable_sample_id in seen:
            raise ManifestError(
                f"{dataset}: duplicate stable sample ID {sample.stable_sample_id}"
            )
        seen.add(sample.stable_sample_id)
        rows.append(ManifestRow(sample.stable_sample_id, sample.label, "test"))

    payload = canonical_manifest_bytes(rows)
    validate_manifest_bytes(dataset, payload)
    return payload


def write_manifest(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, str]:
    payload = materialize_manifest_bytes(dataset, repo_root)
    path = manifest_path(dataset, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise ManifestError(f"{dataset}: manifest re-read differs after write")
    validate_manifest_bytes(dataset, path.read_bytes())
    return path, hashlib.sha256(payload).hexdigest()


def check_manifest(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    path = manifest_path(dataset, repo_root)
    try:
        committed = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{dataset}: missing committed manifest {path}") from exc
    regenerated = materialize_manifest_bytes(dataset, repo_root)
    if committed != regenerated:
        raise ManifestError(
            f"{dataset}: committed manifest differs from deterministic regeneration"
        )
    validate_manifest_bytes(dataset, committed)
    actual_sha256 = hashlib.sha256(committed).hexdigest()
    expected_sha256 = _dataset_config(dataset)["manifest_sha256"]
    if (
        not isinstance(expected_sha256, str)
        or expected_sha256.startswith("pending_")
    ):
        raise ManifestError(
            f"{dataset}: pending manifest SHA-256 cannot pass --check"
        )
    if actual_sha256 != expected_sha256:
        raise ManifestError(
            f"{dataset}: manifest SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return actual_sha256
