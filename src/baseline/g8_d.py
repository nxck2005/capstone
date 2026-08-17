"""G8_D validation-measurement contracts and content identities.

This module is the pre-campaign boundary for G8_D.  It owns no dataset loader,
test-split path, classifier invocation, channel simulation, selection pass, or
training entry point.  The identities here are intentionally boring and
explicit: a cache object is useful only when its complete scientific inputs
are in the key and every archived record can be checked without guessing.

D2--D5 extend this module with the codec, reconstruction and resume seams.  A
small amount of the implementation is already useful in D1: strict identity
schemas, frozen G8_C/classifier/split bindings, and the content-addressed
contract generator consumed by the independent verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from baseline.classical import composition
from config.params import REPO_ROOT, get

__all__ = [
    "G8DContractError",
    "G8_D_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "CODEC_CACHE_SCHEMA_VERSION",
    "RECORD_SCHEMA_VERSION",
    "RESUME_SCHEMA_VERSION",
    "ValidationSplitIdentity",
    "ImageIdentity",
    "BudgetIdentity",
    "CodecConfigurationIdentity",
    "G8CTableIdentity",
    "ClassifierIdentity",
    "CodecSearchKey",
    "CandidateIdentity",
    "EmittedFileIdentity",
    "ReconstructionIdentity",
    "WorkUnitIdentity",
    "canonical_json",
    "rendered_json",
    "sha256_bytes",
    "sha256_file",
    "identity_id",
    "current_codec_snapshot",
    "build_g8_d_contract",
]


G8_D_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
CODEC_CACHE_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
RESUME_SCHEMA_VERSION = 1
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
PHASE_ORDER = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7")
VALIDATION_SPLIT = "val"


class G8DContractError(ValueError):
    """A G8_D identity, schema, cache or contract violation."""


def canonical_json(value: Any) -> bytes:
    """The byte form used by every G8_D content identity."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise G8DContractError(f"value is not canonical JSON: {exc}") from None


def rendered_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise G8DContractError(f"cannot hash {path}: {exc}") from exc


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_DIGEST.fullmatch(value) is None:
        raise G8DContractError(f"{label} is not a lowercase SHA-256")
    return value


def _strict_object(value: Any, fields: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G8DContractError(f"{label} is not an object")
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        raise G8DContractError(
            f"{label} schema differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(value)


def _identity_input(
    value: Any,
    fields: Sequence[str],
    label: str,
    identity_type: str,
) -> dict[str, Any]:
    """Accept a bare field object or its exact schema-wrapped form."""

    bare = set(fields)
    wrapped = bare | {"schema_version", "identity_type"}
    if not isinstance(value, Mapping) or set(value) not in (bare, wrapped):
        return _strict_object(value, fields, label)
    data = dict(value)
    if set(value) == wrapped:
        if data["schema_version"] != IDENTITY_SCHEMA_VERSION or data["identity_type"] != identity_type:
            raise G8DContractError(f"{label} schema wrapper differs")
        data.pop("schema_version")
        data.pop("identity_type")
    return data


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise G8DContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise G8DContractError(f"{label} must be a non-negative integer")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise G8DContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G8DContractError(f"{label} must be finite")
    return result


def _copy_json(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except G8DContractError:
        raise


def identity_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """Hash a complete identity payload; callers add no unbound fields later."""

    if not isinstance(prefix, str) or not prefix or not prefix.endswith("-"):
        raise G8DContractError("identity prefix must be a non-empty trailing-dash string")
    return prefix + sha256_bytes(canonical_json(payload))


@dataclass(frozen=True)
class _Identity:
    """Shared strict serialisation for the concrete identities below."""

    ID_PREFIX: ClassVar[str] = "g8d-"

    def payload(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def identity_id(self) -> str:
        return identity_id(self.ID_PREFIX, self.payload())

    def as_dict(self) -> dict[str, Any]:
        return _copy_json(self.payload())


@dataclass(frozen=True)
class ValidationSplitIdentity(_Identity):
    """Hash of the validation manifest and its dataset archive lineage only."""

    dataset: str
    split: str
    dataset_version: str
    manifest_sha256: str

    ID_PREFIX: ClassVar[str] = "g8dsplit-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset",
        "split",
        "dataset_version",
        "manifest_sha256",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise G8DContractError("validation dataset is empty")
        if self.split != VALIDATION_SPLIT:
            raise G8DContractError("G8_D measurement identities may only use split 'val'")
        _require_digest(self.dataset_version, "dataset_version")
        _require_digest(self.manifest_sha256, "manifest_sha256")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "validation_split",
            "dataset": self.dataset,
            "split": self.split,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationSplitIdentity":
        data = _identity_input(value, cls.FIELDS, "validation split identity", "validation_split")
        return cls(
            dataset=str(data["dataset"]),
            split=str(data["split"]),
            dataset_version=str(data["dataset_version"]),
            manifest_sha256=str(data["manifest_sha256"]),
        )

    @classmethod
    def from_current(cls, dataset: str, repo_root: Path = REPO_ROOT) -> "ValidationSplitIdentity":
        from data.manifests import manifest_sha256

        datasets = get("datasets")
        if not isinstance(datasets, Mapping) or dataset not in datasets:
            raise G8DContractError(f"unknown configured dataset {dataset!r}")
        version_rule = str(get("config.dataset_version_rule"))
        dataset_version = datasets[dataset].get(version_rule)
        if not isinstance(dataset_version, str):
            raise G8DContractError(f"{dataset}: dataset version is not pinned")
        return cls(
            dataset=dataset,
            split=VALIDATION_SPLIT,
            dataset_version=dataset_version,
            manifest_sha256=manifest_sha256(dataset, repo_root),
        )


@dataclass(frozen=True)
class ImageIdentity(_Identity):
    """Source and canonical-pixel identity for one validation image."""

    dataset: str
    split: str
    dataset_version: str
    manifest_sha256: str
    stable_sample_id: str
    source_bytes_sha256: str
    canonical_pixels_sha256: str
    canonical_shape: tuple[int, int, int]

    ID_PREFIX: ClassVar[str] = "g8dimage-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset",
        "split",
        "dataset_version",
        "manifest_sha256",
        "stable_sample_id",
        "source_bytes_sha256",
        "canonical_pixels_sha256",
        "canonical_shape",
    )

    def __post_init__(self) -> None:
        split = ValidationSplitIdentity(
            self.dataset, self.split, self.dataset_version, self.manifest_sha256
        )
        del split
        if not isinstance(self.stable_sample_id, str) or not self.stable_sample_id:
            raise G8DContractError("stable_sample_id is empty")
        _require_digest(self.source_bytes_sha256, "source_bytes_sha256")
        _require_digest(self.canonical_pixels_sha256, "canonical_pixels_sha256")
        if (
            not isinstance(self.canonical_shape, tuple)
            or len(self.canonical_shape) != 3
            or self.canonical_shape[2] != 3
            or any(_positive_int(item, "canonical shape") <= 0 for item in self.canonical_shape)
        ):
            raise G8DContractError("canonical_shape must be (height, width, 3)")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "validation_image",
            "dataset": self.dataset,
            "split": self.split,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "stable_sample_id": self.stable_sample_id,
            "source_bytes_sha256": self.source_bytes_sha256,
            "canonical_pixels_sha256": self.canonical_pixels_sha256,
            "canonical_shape": list(self.canonical_shape),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImageIdentity":
        data = _identity_input(value, cls.FIELDS, "image identity", "validation_image")
        shape = data["canonical_shape"]
        if not isinstance(shape, Sequence) or isinstance(shape, str):
            raise G8DContractError("canonical_shape is not a sequence")
        return cls(
            dataset=str(data["dataset"]),
            split=str(data["split"]),
            dataset_version=str(data["dataset_version"]),
            manifest_sha256=str(data["manifest_sha256"]),
            stable_sample_id=str(data["stable_sample_id"]),
            source_bytes_sha256=str(data["source_bytes_sha256"]),
            canonical_pixels_sha256=str(data["canonical_pixels_sha256"]),
            canonical_shape=tuple(int(item) for item in shape),
        )

    @classmethod
    def from_pixels(
        cls,
        *,
        split_identity: ValidationSplitIdentity,
        stable_sample_id: str,
        source_bytes: bytes,
        canonical_pixels: np.ndarray,
    ) -> "ImageIdentity":
        if not isinstance(source_bytes, bytes) or not source_bytes:
            raise G8DContractError("source_bytes must be non-empty bytes")
        if (
            not isinstance(canonical_pixels, np.ndarray)
            or canonical_pixels.dtype != np.uint8
            or canonical_pixels.ndim != 3
            or canonical_pixels.shape[2] != 3
        ):
            raise G8DContractError("canonical_pixels must be uint8 RGB HWC")
        return cls(
            dataset=split_identity.dataset,
            split=split_identity.split,
            dataset_version=split_identity.dataset_version,
            manifest_sha256=split_identity.manifest_sha256,
            stable_sample_id=stable_sample_id,
            source_bytes_sha256=sha256_bytes(source_bytes),
            canonical_pixels_sha256=sha256_bytes(np.ascontiguousarray(canonical_pixels).tobytes()),
            canonical_shape=tuple(int(item) for item in canonical_pixels.shape),
        )


@dataclass(frozen=True)
class BudgetIdentity(_Identity):
    """Complete source/transport byte budget and packet geometry identity."""

    bw_ratio: str
    bytes_sent: int
    payload_bytes: int
    packet_accounting: Mapping[str, Any]

    ID_PREFIX: ClassVar[str] = "g8dbudget-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "bw_ratio",
        "bytes_sent",
        "payload_bytes",
        "packet_accounting",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.bw_ratio, str) or not self.bw_ratio:
            raise G8DContractError("bw_ratio is empty")
        _positive_int(self.bytes_sent, "bytes_sent")
        _positive_int(self.payload_bytes, "payload_bytes")
        if not isinstance(self.packet_accounting, Mapping) or not self.packet_accounting:
            raise G8DContractError("packet_accounting must be a non-empty mapping")
        if "payload_bytes" in self.packet_accounting and self.packet_accounting["payload_bytes"] != self.payload_bytes:
            raise G8DContractError("packet accounting payload does not match budget")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "transport_budget",
            "bw_ratio": self.bw_ratio,
            "bytes_sent": self.bytes_sent,
            "payload_bytes": self.payload_bytes,
            "packet_accounting": _copy_json(self.packet_accounting),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetIdentity":
        data = _identity_input(value, cls.FIELDS, "budget identity", "transport_budget")
        return cls(
            bw_ratio=str(data["bw_ratio"]),
            bytes_sent=int(data["bytes_sent"]),
            payload_bytes=int(data["payload_bytes"]),
            packet_accounting=data["packet_accounting"],
        )


@dataclass(frozen=True)
class CodecConfigurationIdentity(_Identity):
    """Hash of every output-affecting JPEG 2000 and resize setting."""

    snapshot: Mapping[str, Any]
    configuration_hash: str
    runtime_version: str

    ID_PREFIX: ClassVar[str] = "g8dcodec-"
    FIELDS: ClassVar[tuple[str, ...]] = ("snapshot", "configuration_hash", "runtime_version")

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, Mapping) or set(self.snapshot) != {"baseline", "preprocessing", "environment"}:
            raise G8DContractError("codec snapshot top-level schema differs")
        _require_digest(self.configuration_hash, "codec configuration_hash")
        if sha256_bytes(canonical_json(self.snapshot)) != self.configuration_hash:
            raise G8DContractError("codec configuration_hash does not reproduce snapshot")
        if not isinstance(self.runtime_version, str) or not self.runtime_version:
            raise G8DContractError("codec runtime_version is empty")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "jpeg2000_configuration",
            "snapshot": _copy_json(self.snapshot),
            "configuration_hash": self.configuration_hash,
            "runtime_version": self.runtime_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodecConfigurationIdentity":
        data = _identity_input(value, cls.FIELDS, "codec configuration identity", "jpeg2000_configuration")
        return cls(data["snapshot"], str(data["configuration_hash"]), str(data["runtime_version"]))


@dataclass(frozen=True)
class G8CTableIdentity(_Identity):
    """The exact measured-only Pascal successor table binding."""

    campaign_id: str
    execution_profile_id: str
    measurement_source_commit: str
    production_contract_sha256: str
    table_id: str
    table_sha256: str
    merge_report_id: str
    merge_report_sha256: str
    closeout_id: str
    closeout_sha256: str
    curves: int
    measured_points: int
    trials_per_point: int
    predecessor_table_contribution: str

    ID_PREFIX: ClassVar[str] = "g8dtable-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "campaign_id", "execution_profile_id", "measurement_source_commit",
        "production_contract_sha256", "table_id", "table_sha256", "merge_report_id",
        "merge_report_sha256", "closeout_id", "closeout_sha256", "curves",
        "measured_points", "trials_per_point", "predecessor_table_contribution",
    )

    def __post_init__(self) -> None:
        for field in ("production_contract_sha256", "table_sha256", "merge_report_sha256", "closeout_sha256"):
            _require_digest(getattr(self, field), field)
        if not all(isinstance(getattr(self, field), str) and getattr(self, field) for field in ("campaign_id", "execution_profile_id", "measurement_source_commit", "table_id", "merge_report_id", "closeout_id")):
            raise G8DContractError("G8_C table binding contains an empty identity")
        if (self.curves, self.measured_points, self.trials_per_point) != (153, 3213, 5000):
            raise G8DContractError("G8_C table binding is not the frozen 153/3213/5000 table")
        if self.predecessor_table_contribution != "none":
            raise G8DContractError("predecessor evidence is bound into the successor table")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "g8_c_pascal_table"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "G8CTableIdentity":
        data = _identity_input(value, cls.FIELDS, "G8_C table identity", "g8_c_pascal_table")
        return cls(**data)


@dataclass(frozen=True)
class ClassifierIdentity(_Identity):
    """Frozen G-1 clean-classifier identity, without loading its checkpoint."""

    variant: str
    checkpoint_sha256: str
    classifier_config_sha256: str
    dataset: str
    split: str
    dataset_version: str
    manifest_sha256: str

    ID_PREFIX: ClassVar[str] = "g8dclassifier-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "variant", "checkpoint_sha256", "classifier_config_sha256", "dataset", "split",
        "dataset_version", "manifest_sha256",
    )

    def __post_init__(self) -> None:
        if self.variant != "clean":
            raise G8DContractError("G8_D only admits the frozen clean classifier")
        _require_digest(self.checkpoint_sha256, "checkpoint_sha256")
        _require_digest(self.classifier_config_sha256, "classifier_config_sha256")
        ValidationSplitIdentity(self.dataset, self.split, self.dataset_version, self.manifest_sha256)

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "g1_clean_classifier"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifierIdentity":
        return cls(**_identity_input(value, cls.FIELDS, "classifier identity", "g1_clean_classifier"))


@dataclass(frozen=True)
class CodecSearchKey(_Identity):
    """Cache key for rate-control work; SNR is deliberately absent."""

    image_identity_id: str
    budget_identity_id: str
    codec_configuration_id: str
    encode_axis_px: int

    ID_PREFIX: ClassVar[str] = "g8dsearch-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "image_identity_id", "budget_identity_id", "codec_configuration_id", "encode_axis_px"
    )

    def __post_init__(self) -> None:
        for field in self.FIELDS[:3]:
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise G8DContractError(f"{field} is empty")
        _positive_int(self.encode_axis_px, "encode_axis_px")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "codec_search_key"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result


@dataclass(frozen=True)
class CandidateIdentity(_Identity):
    """One complete validation candidate, including the exact BLER identity."""

    image_identity_id: str
    budget_identity_id: str
    codec_configuration_id: str
    g8_c_table_identity_id: str
    bler_identity: Mapping[str, Any]
    snr_db: float
    encode_axis_px: int

    ID_PREFIX: ClassVar[str] = "g8dcandidate-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "image_identity_id", "budget_identity_id", "codec_configuration_id",
        "g8_c_table_identity_id", "bler_identity", "snr_db", "encode_axis_px",
    )

    def __post_init__(self) -> None:
        for field in self.FIELDS[:4]:
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise G8DContractError(f"{field} is empty")
        composition.BlerIdentity.from_mapping(self.bler_identity)
        _finite_float(self.snr_db, "snr_db")
        _positive_int(self.encode_axis_px, "encode_axis_px")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "candidate"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result


@dataclass(frozen=True)
class EmittedFileIdentity(_Identity):
    """Identity of the bytes actually emitted by the codec."""

    codec_search_key_id: str
    codestream_sha256: str
    emitted_bytes: int
    payload_budget_bytes: int
    filler_bytes: int

    ID_PREFIX: ClassVar[str] = "g8demitted-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "codec_search_key_id", "codestream_sha256", "emitted_bytes", "payload_budget_bytes", "filler_bytes"
    )

    def __post_init__(self) -> None:
        if not self.codec_search_key_id:
            raise G8DContractError("codec_search_key_id is empty")
        _require_digest(self.codestream_sha256, "codestream_sha256")
        _positive_int(self.emitted_bytes, "emitted_bytes")
        _positive_int(self.payload_budget_bytes, "payload_budget_bytes")
        _nonnegative_int(self.filler_bytes, "filler_bytes")
        if self.emitted_bytes > self.payload_budget_bytes or self.filler_bytes != self.payload_budget_bytes - self.emitted_bytes:
            raise G8DContractError("emitted-byte/filler arithmetic does not reconcile")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "emitted_codestream"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result


@dataclass(frozen=True)
class ReconstructionIdentity(_Identity):
    """Identity of one decoded/upsampled reconstruction cache object."""

    image_identity_id: str
    emitted_file_identity_id: str
    codec_configuration_id: str
    output_shape: tuple[int, int, int]
    upsample_interpolation: str
    preserves_aspect: bool

    ID_PREFIX: ClassVar[str] = "g8drecon-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "image_identity_id", "emitted_file_identity_id", "codec_configuration_id", "output_shape",
        "upsample_interpolation", "preserves_aspect",
    )

    def __post_init__(self) -> None:
        for field in self.FIELDS[:3]:
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise G8DContractError(f"{field} is empty")
        if len(self.output_shape) != 3 or self.output_shape[2] != 3 or any(_positive_int(item, "output shape") <= 0 for item in self.output_shape):
            raise G8DContractError("output_shape must be (height, width, 3)")
        if not isinstance(self.upsample_interpolation, str) or not self.upsample_interpolation:
            raise G8DContractError("upsample_interpolation is empty")
        if not isinstance(self.preserves_aspect, bool):
            raise G8DContractError("preserves_aspect must be boolean")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": CODEC_CACHE_SCHEMA_VERSION, "identity_type": "reconstruction"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        result["output_shape"] = list(self.output_shape)
        return result


@dataclass(frozen=True)
class WorkUnitIdentity(_Identity):
    """Deterministic ordered resume unit; ordinal is part of its identity."""

    campaign_id: str
    ordinal: int
    candidate_identity_id: str
    record_schema_version: int = RECORD_SCHEMA_VERSION

    ID_PREFIX: ClassVar[str] = "g8dwork-"
    FIELDS: ClassVar[tuple[str, ...]] = (
        "campaign_id", "ordinal", "candidate_identity_id", "record_schema_version"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id:
            raise G8DContractError("campaign_id is empty")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise G8DContractError("work-unit ordinal must be a non-negative integer")
        if not isinstance(self.candidate_identity_id, str) or not self.candidate_identity_id:
            raise G8DContractError("candidate_identity_id is empty")
        if self.record_schema_version != RECORD_SCHEMA_VERSION:
            raise G8DContractError("unsupported work-unit record schema")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": RESUME_SCHEMA_VERSION, "identity_type": "work_unit"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result


def current_codec_snapshot() -> dict[str, Any]:
    """Read the already-frozen J2K snapshot without encoding an image."""

    from baseline.j2k import _codec_snapshot

    return _copy_json(_codec_snapshot())


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8DContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise G8DContractError(f"{label} is not an object")
    return value, raw


def _current_g8_c_binding(repo_root: Path) -> G8CTableIdentity:
    d0, _ = _read_json(repo_root / "results/baseline/g8_d/d0_open.json", "D0 opening artifact")
    data = d0["g8_c"]
    return G8CTableIdentity(
        campaign_id=data["campaign_id"],
        execution_profile_id=data["execution_profile_id"],
        measurement_source_commit=data["measurement_source_commit"],
        production_contract_sha256=data["production_contract_sha256"],
        table_id=data["table_id"],
        table_sha256=data["table_sha256"],
        merge_report_id=data["merge_report_id"],
        merge_report_sha256=data["merge_report_sha256"],
        closeout_id=data["closeout_provenance_id"],
        closeout_sha256=data["closeout_provenance_sha256"],
        curves=data["curves"],
        measured_points=data["measured_points"],
        trials_per_point=data["trials_per_point"],
        predecessor_table_contribution=data["predecessor_table_contribution"],
    )


def _current_classifier_binding(repo_root: Path) -> ClassifierIdentity:
    adjudication, _ = _read_json(repo_root / "results/reference_classifier/g1_adjudication.json", "G1 adjudication")
    split = ValidationSplitIdentity.from_current("imagenette160", repo_root)
    if adjudication.get("split_manifest_hash") != split.manifest_sha256:
        raise G8DContractError("G1 classifier and validation manifest bindings differ")
    return ClassifierIdentity(
        variant=str(adjudication["classifier_variant"]),
        checkpoint_sha256=str(adjudication["checkpoint_sha256"]),
        classifier_config_sha256=str(adjudication["config_hash"]),
        dataset=str(adjudication["dataset"]),
        split=VALIDATION_SPLIT,
        dataset_version=split.dataset_version,
        manifest_sha256=split.manifest_sha256,
    )


def _source_bindings(repo_root: Path) -> list[dict[str, str]]:
    paths = (
        ("src/baseline/g8_d.py", "g8_d_identity_and_contract_runtime"),
        ("tools/gen_g8_d_contract.py", "g8_d_contract_generator"),
        ("tools/verify_g8_d_contract.py", "g8_d_independent_verifier"),
        ("tools/verify_g8_d_open.py", "d0_upstream_verifier"),
        ("src/baseline/g8_pascal_merge.py", "frozen_successor_loader"),
        ("src/baseline/j2k.py", "frozen_jpeg2000_codec"),
        ("src/data/preprocessing.py", "frozen_codec_resize_contract"),
        ("src/baseline/classical/records.py", "frozen_br11_record_contract"),
    )
    return [
        {"path": path, "role": role, "sha256": sha256_file(repo_root / path)}
        for path, role in paths
    ]


def build_g8_d_contract(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build the deterministic D1 contract without touching image payloads."""

    repo_root = Path(repo_root).resolve()
    g8c = _current_g8_c_binding(repo_root)
    classifier = _current_classifier_binding(repo_root)
    split_bindings = [
        ValidationSplitIdentity.from_current(dataset, repo_root).as_dict()
        for dataset in ("cifar10", "imagenette160", "stl10")
    ]
    codec_snapshot = current_codec_snapshot()
    codec_hash = sha256_bytes(canonical_json(codec_snapshot))
    codec = CodecConfigurationIdentity(codec_snapshot, codec_hash, str(codec_snapshot["environment"]["openjpeg"]))
    d0_path = repo_root / "results/baseline/g8_d/d0_open.json"
    w4_path = repo_root / "results/baseline/w4/integration_adjudication.json"
    spec_path = repo_root / "spec/SPEC.md"
    params_path = repo_root / "spec/params.generated.yaml"
    d0, _ = _read_json(d0_path, "D0 opening artifact")
    w4, _ = _read_json(w4_path, "W4 integration adjudication")
    body: dict[str, Any] = {
        "schema_version": G8_D_SCHEMA_VERSION,
        "artifact_role": "g8_d_validation_measurement_contract",
        "phase": "G8_D",
        "checkpoint": "D1",
        "status": "tooling_ready",
        "contract_id": None,
        "campaign_id": None,
        "g8_c_binding": g8c.as_dict(),
        "d0_open_binding": {
            "artifact_id": d0["artifact_id"],
            "artifact_sha256": sha256_file(d0_path),
        },
        "validation_split_bindings": split_bindings,
        "classifier_binding": classifier.as_dict(),
        "codec_binding": codec.as_dict(),
        "upstream_bindings": {
            "w4_integration_adjudication_sha256": sha256_file(w4_path),
            "w4_selection_policy_sha256": w4["selection_machinery"]["selection_policy_sha256"],
            "spec_sha256": sha256_file(spec_path),
            "params_generated_sha256": sha256_file(params_path),
            "g1_adjudication_sha256": sha256_file(repo_root / "results/reference_classifier/g1_adjudication.json"),
        },
        "identity_schema": {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "validation_split": list(ValidationSplitIdentity.FIELDS),
            "validation_image": list(ImageIdentity.FIELDS),
            "transport_budget": list(BudgetIdentity.FIELDS),
            "jpeg2000_configuration": list(CodecConfigurationIdentity.FIELDS),
            "g8_c_pascal_table": list(G8CTableIdentity.FIELDS),
            "g1_clean_classifier": list(ClassifierIdentity.FIELDS),
            "codec_search_key": list(CodecSearchKey.FIELDS),
            "candidate": list(CandidateIdentity.FIELDS),
            "emitted_codestream": list(EmittedFileIdentity.FIELDS),
            "reconstruction": list(ReconstructionIdentity.FIELDS),
            "work_unit": list(WorkUnitIdentity.FIELDS),
        },
        "cache_schema": {
            "codec_cache_schema_version": CODEC_CACHE_SCHEMA_VERSION,
            "reconstruction_cache_schema_version": CODEC_CACHE_SCHEMA_VERSION,
            "key_excludes": ["snr_db", "requested_compression_ratio"],
            "emitted_bytes_authoritative": True,
            "content_addressed_objects": True,
            "source_bytes_bound": True,
            "canonical_pixels_bound": True,
            "budget_bound": True,
            "codec_configuration_bound": True,
            "downsample_axis_bound": True,
        },
        "record_schema": {
            "schema_version": RECORD_SCHEMA_VERSION,
            "measured_accuracy_requires_counts": True,
            "allowed_split": VALIDATION_SPLIT,
            "allowed_verdicts": [
                "structural_infeasibility", "codec_infeasibility", "decode_failure", "delivered"
            ],
            "infeasible_candidates_are_records": True,
        },
        "resume_schema": {
            "schema_version": RESUME_SCHEMA_VERSION,
            "ordering": ["ordinal", "work_unit_id", "candidate_id"],
            "completed_must_be_exact_prefix": True,
            "completed_records_immutable": True,
            "aggregate_must_reference_durable_records": True,
            "changed_contract_fails_closed": True,
            "duplicate_or_missing_work_unit_fails_closed": True,
        },
        "work_unit_ordering": [
            "dataset", "stable_sample_id", "bw_ratio", "k_symbols", "modulation",
            "ldpc_rate", "encode_axis_px", "snr_db", "candidate_id",
        ],
        "phase_order": list(PHASE_ORDER),
        "source_bindings": _source_bindings(repo_root),
        "safety": {
            "validation_campaign_started": False,
            "selection_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "training_started": False,
            "test_split_accessed": False,
            "test_access": 0,
            "inference": 0,
            "training": 0,
            "validation_decoding": 0,
            "g8_e_started": False,
        },
        "next_gate": "G8_D/D2",
    }
    campaign_basis = dict(body)
    campaign_basis.pop("campaign_id")
    campaign_basis.pop("contract_id")
    body["campaign_id"] = "g8d-" + sha256_bytes(canonical_json(campaign_basis))
    contract_basis = dict(body)
    contract_basis.pop("contract_id")
    body["contract_id"] = "g8dcontract-" + sha256_bytes(canonical_json(contract_basis))
    return body
