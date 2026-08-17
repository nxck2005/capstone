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
import base64
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
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
    "BR11Accounting",
    "BR11Aggregate",
    "account_br11",
    "aggregate_br11",
    "ReconstructionResult",
    "ReconstructionCache",
    "CleanClassifierMeasurementRecord",
    "canonical_json",
    "rendered_json",
    "sha256_bytes",
    "sha256_file",
    "identity_id",
    "current_codec_snapshot",
    "build_g8_d_contract",
    "STRUCTURAL_INFEASIBILITY",
    "CODEC_INFEASIBILITY",
    "CODEC_FEASIBLE",
    "CodecSearchResult",
    "publish_immutable_object",
    "CodecSearchEngine",
]


G8_D_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
CODEC_CACHE_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
RESUME_SCHEMA_VERSION = 1
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
PHASE_ORDER = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7")
VALIDATION_SPLIT = "val"
STRUCTURAL_INFEASIBILITY = "structural_infeasibility"
CODEC_INFEASIBILITY = "codec_infeasibility"
CODEC_FEASIBLE = "feasible"


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
            for chunk in iter(
                lambda: stream.read(1024 * 1024),  # literal-ok: one-MiB streaming I/O chunk
                b"",
            ):
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
        if self.bytes_sent != self.payload_bytes:
            raise G8DContractError(
                "bytes_sent must equal payload_bytes: AM-81 binds both to the complete A/8 transport capacity"
            )
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
        if (self.curves, self.measured_points, self.trials_per_point) != (153, 3213, 5000):  # literal-ok: frozen G8_C trial count
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodecSearchKey":
        return cls(**_identity_input(value, cls.FIELDS, "codec search key", "codec_search_key"))


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
        for field in self.FIELDS[:4]:  # literal-ok: four leading identity string fields
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise G8DContractError(f"{field} is empty")
        composition.BlerIdentity.from_mapping(self.bler_identity)
        _finite_float(self.snr_db, "snr_db")
        _positive_int(self.encode_axis_px, "encode_axis_px")

    def payload(self) -> dict[str, Any]:
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "candidate"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateIdentity":
        return cls(**_identity_input(value, cls.FIELDS, "candidate identity", "candidate"))


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EmittedFileIdentity":
        return cls(**_identity_input(value, cls.FIELDS, "emitted file identity", "emitted_codestream"))


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
        result = {"schema_version": IDENTITY_SCHEMA_VERSION, "identity_type": "reconstruction"}
        result.update({field: _copy_json(getattr(self, field)) for field in self.FIELDS})
        result["output_shape"] = list(self.output_shape)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReconstructionIdentity":
        data = _identity_input(value, cls.FIELDS, "reconstruction identity", "reconstruction")
        shape = data["output_shape"]
        if not isinstance(shape, Sequence) or isinstance(shape, str):
            raise G8DContractError("reconstruction output_shape is not a sequence")
        data["output_shape"] = tuple(int(item) for item in shape)
        return cls(**data)


BR11_EMITTED_VERDICTS = frozenset({"delivered", "decode_failure"})


@dataclass(frozen=True)
class BR11Accounting:
    """Exact AM-81 byte accounting for one emitted codestream.

    The object is deliberately constructed from the actual codestream bytes
    by :func:`account_br11`.  A requested compression ratio, a codec-reported
    estimate, or a delivered-only denominator cannot enter this record.
    """

    verdict: str
    emitted_file_identity: EmittedFileIdentity
    bytes_sent: int
    emitted_codestream_bytes: int
    header_bytes: int
    payload_bytes: int
    payload_filler_bytes: int
    codestream_sha256: str
    denominator: int = 1

    FIELDS: ClassVar[tuple[str, ...]] = (
        "verdict",
        "emitted_file_identity",
        "bytes_sent",
        "emitted_codestream_bytes",
        "header_bytes",
        "payload_bytes",
        "payload_filler_bytes",
        "codestream_sha256",
        "denominator",
    )

    def __post_init__(self) -> None:
        if self.verdict not in BR11_EMITTED_VERDICTS:
            raise G8DContractError(
                "BR-11 accounting requires delivered or decode_failure for an emitted codestream"
            )
        if not isinstance(self.emitted_file_identity, EmittedFileIdentity):
            raise G8DContractError("BR-11 accounting has no emitted-file identity")
        _positive_int(self.bytes_sent, "BR-11 bytes_sent")
        _positive_int(self.emitted_codestream_bytes, "BR-11 emitted_codestream_bytes")
        _nonnegative_int(self.header_bytes, "BR-11 header_bytes")
        _nonnegative_int(self.payload_bytes, "BR-11 payload_bytes")
        _nonnegative_int(self.payload_filler_bytes, "BR-11 payload_filler_bytes")
        _require_digest(self.codestream_sha256, "BR-11 codestream_sha256")
        if self.denominator != 1:
            raise G8DContractError("one emitted codestream must have denominator 1")
        if self.bytes_sent != self.emitted_file_identity.payload_budget_bytes:
            raise G8DContractError("BR-11 bytes_sent differs from emitted-file payload budget")
        if self.codestream_sha256 != self.emitted_file_identity.codestream_sha256:
            raise G8DContractError("BR-11 codestream hash differs from emitted-file identity")
        if self.emitted_codestream_bytes != self.emitted_file_identity.emitted_bytes:
            raise G8DContractError("BR-11 emitted length differs from emitted-file identity")
        if self.emitted_codestream_bytes != self.header_bytes + self.payload_bytes:
            raise G8DContractError("BR-11 header plus payload does not reconcile")
        if self.bytes_sent != self.emitted_codestream_bytes + self.payload_filler_bytes:
            raise G8DContractError("BR-11 emitted bytes plus filler does not reconcile")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "br11_emitted_codestream_accounting",
            "verdict": self.verdict,
            "emitted_file_identity": self.emitted_file_identity.payload(),
            "bytes_sent": self.bytes_sent,
            "emitted_codestream_bytes": self.emitted_codestream_bytes,
            "header_bytes": self.header_bytes,
            "payload_bytes": self.payload_bytes,
            "payload_filler_bytes": self.payload_filler_bytes,
            "codestream_sha256": self.codestream_sha256,
            "denominator": self.denominator,
            "accounting_rule": "AM-81",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BR11Accounting":
        fields = set(cls.FIELDS) | {"schema_version", "artifact_role", "accounting_rule"}
        data = _strict_object(value, tuple(fields), "BR-11 accounting")
        if data["schema_version"] != RECORD_SCHEMA_VERSION:
            raise G8DContractError("unsupported BR-11 accounting schema")
        if data["artifact_role"] != "br11_emitted_codestream_accounting" or data["accounting_rule"] != "AM-81":
            raise G8DContractError("BR-11 accounting role or rule differs")
        return cls(
            verdict=data["verdict"],
            emitted_file_identity=EmittedFileIdentity.from_mapping(data["emitted_file_identity"]),
            bytes_sent=data["bytes_sent"],
            emitted_codestream_bytes=data["emitted_codestream_bytes"],
            header_bytes=data["header_bytes"],
            payload_bytes=data["payload_bytes"],
            payload_filler_bytes=data["payload_filler_bytes"],
            codestream_sha256=data["codestream_sha256"],
            denominator=data["denominator"],
        )


def account_br11(
    codestream: bytes,
    *,
    emitted_file_identity: EmittedFileIdentity,
    bytes_sent: int,
    verdict: str,
) -> BR11Accounting:
    """Derive AM-81 columns from complete emitted JPEG 2000 bytes."""

    if not isinstance(codestream, bytes) or not codestream:
        raise G8DContractError("BR-11 codestream must be non-empty bytes")
    if not isinstance(emitted_file_identity, EmittedFileIdentity):
        raise G8DContractError("BR-11 requires an emitted-file identity")
    actual_hash = sha256_bytes(codestream)
    if len(codestream) != emitted_file_identity.emitted_bytes:
        raise G8DContractError("BR-11 codestream length differs from emitted-file identity")
    if actual_hash != emitted_file_identity.codestream_sha256:
        raise G8DContractError("BR-11 codestream bytes differ from emitted-file identity")
    if bytes_sent != emitted_file_identity.payload_budget_bytes:
        raise G8DContractError("BR-11 bytes_sent differs from emitted-file payload budget")
    try:
        from baseline.classical.records import codestream_byte_split

        header_bytes, payload_bytes = codestream_byte_split(codestream)
    except Exception as exc:
        raise G8DContractError(f"BR-11 codestream cannot be structurally accounted: {exc}") from None
    return BR11Accounting(
        verdict=verdict,
        emitted_file_identity=emitted_file_identity,
        bytes_sent=bytes_sent,
        emitted_codestream_bytes=len(codestream),
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        payload_filler_bytes=bytes_sent - len(codestream),
        codestream_sha256=actual_hash,
    )


@dataclass(frozen=True)
class BR11Aggregate:
    """Per-cell AM-81 sums and means over every emitted row."""

    bytes_sent: int | None
    emitted_row_count: int
    header_numerator: int
    payload_numerator: int
    emitted_codestream_numerator: int
    payload_filler_numerator: int
    denominator: int
    verdict_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.emitted_row_count != self.denominator:
            raise G8DContractError("BR-11 aggregate denominator is not emitted-row count")
        if self.emitted_row_count == 0 and self.bytes_sent is not None:
            raise G8DContractError("empty BR-11 aggregate has no bytes_sent")
        if self.emitted_row_count > 0 and self.bytes_sent is None:
            raise G8DContractError("non-empty BR-11 aggregate has no bytes_sent")
        for name in (
            "header_numerator",
            "payload_numerator",
            "emitted_codestream_numerator",
            "payload_filler_numerator",
        ):
            _nonnegative_int(getattr(self, name), f"BR-11 {name}")
        if self.emitted_codestream_numerator != self.header_numerator + self.payload_numerator:
            raise G8DContractError("BR-11 aggregate header plus payload does not reconcile")
        if self.bytes_sent is not None and self.bytes_sent * self.denominator != (
            self.emitted_codestream_numerator + self.payload_filler_numerator
        ):
            raise G8DContractError("BR-11 aggregate emitted bytes plus filler does not reconcile")

    def as_dict(self) -> dict[str, Any]:
        denominator = self.denominator
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "br11_aggregate",
            "accounting_rule": "AM-81",
            "bytes_sent": self.bytes_sent,
            "emitted_row_count": self.emitted_row_count,
            "denominator": denominator,
            "header_bytes_numerator": self.header_numerator,
            "payload_bytes_numerator": self.payload_numerator,
            "emitted_codestream_bytes_numerator": self.emitted_codestream_numerator,
            "payload_filler_bytes_numerator": self.payload_filler_numerator,
            "header_bytes": None if denominator == 0 else self.header_numerator / denominator,
            "payload_bytes": None if denominator == 0 else self.payload_numerator / denominator,
            "emitted_codestream_bytes": None if denominator == 0 else self.emitted_codestream_numerator / denominator,
            "payload_filler_bytes": None if denominator == 0 else self.payload_filler_numerator / denominator,
            "verdict_counts": _copy_json(self.verdict_counts),
        }


def aggregate_br11(accountings: Sequence[BR11Accounting]) -> BR11Aggregate:
    """Aggregate delivered and decode-failure rows, excluding no-emission rows."""

    if not isinstance(accountings, Sequence) or isinstance(accountings, (str, bytes)):
        raise G8DContractError("BR-11 aggregate input is not an ordered sequence")
    rows = tuple(accountings)
    if any(not isinstance(row, BR11Accounting) for row in rows):
        raise G8DContractError("BR-11 aggregate contains a non-accounting row")
    if not rows:
        return BR11Aggregate(
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            {verdict: 0 for verdict in sorted(BR11_EMITTED_VERDICTS)},
        )
    bytes_sent = rows[0].bytes_sent
    if any(row.bytes_sent != bytes_sent for row in rows):
        raise G8DContractError("BR-11 aggregate mixes transport budgets")
    counts = {verdict: 0 for verdict in sorted(BR11_EMITTED_VERDICTS)}
    for row in rows:
        counts[row.verdict] += row.denominator
    return BR11Aggregate(
        bytes_sent=bytes_sent,
        emitted_row_count=len(rows),
        header_numerator=sum(row.header_bytes for row in rows),
        payload_numerator=sum(row.payload_bytes for row in rows),
        emitted_codestream_numerator=sum(row.emitted_codestream_bytes for row in rows),
        payload_filler_numerator=sum(row.payload_filler_bytes for row in rows),
        denominator=sum(row.denominator for row in rows),
        verdict_counts=counts,
    )


@dataclass(frozen=True)
class ReconstructionResult:
    """A validated receiver reconstruction and its immutable cache provenance."""

    identity: ReconstructionIdentity
    reconstruction: np.ndarray
    cache_hit: bool
    cache_object_id: str


class ReconstructionCache:
    """Content-addressed decoded/upsampled reconstruction cache.

    Cache objects contain the final uint8 RGB reconstruction bytes and the
    complete image, emitted-file and codec identities.  They are immutable;
    the same-directory no-replace publisher is the only publication path.
    """

    _CACHE_FIELDS = (
        "schema_version",
        "reconstruction_identity",
        "image_identity",
        "emitted_file_identity",
        "codec_configuration",
        "input_codestream_sha256",
        "input_emitted_bytes",
        "decoded_pixels_b64",
        "decoded_pixels_sha256",
        "decoded_shape",
        "decoded_dtype",
        "cache_object_id",
    )

    def __init__(
        self,
        cache_root: Path,
        codec_identity: CodecConfigurationIdentity,
        decoder: Callable[[bytes], np.ndarray] | Any | None = None,
    ) -> None:
        if not isinstance(codec_identity, CodecConfigurationIdentity):
            raise G8DContractError("reconstruction cache requires codec configuration identity")
        self.cache_root = Path(cache_root).resolve()
        self.codec_identity = codec_identity
        if decoder is None:
            from baseline.j2k import J2KCodec

            decoder = J2KCodec(self.cache_root / "backend").decode_codestream
        elif hasattr(decoder, "decode_codestream"):
            decoder = decoder.decode_codestream
        if not callable(decoder):
            raise G8DContractError("reconstruction decoder is not callable")
        self.decoder: Callable[[bytes], np.ndarray] = decoder
        interpolation = get("preprocessing.codec_upsample_interpolation")
        preserves_aspect = get("preprocessing.codec_resize_preserves_aspect")
        if not isinstance(interpolation, str) or not interpolation:
            raise G8DContractError("codec upsample interpolation is not configured")
        if not isinstance(preserves_aspect, bool) or not preserves_aspect:
            raise G8DContractError("G8_D requires aspect-preserving codec resize")
        self.upsample_interpolation = interpolation
        self.preserves_aspect = preserves_aspect

    @staticmethod
    def _validated_pixels(pixels: Any, label: str) -> np.ndarray:
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise G8DContractError(f"{label} must be uint8 RGB HWC")
        if any(int(value) <= 0 for value in pixels.shape):
            raise G8DContractError(f"{label} has a non-positive shape")
        return np.ascontiguousarray(pixels)

    def _identity(
        self,
        image_identity: ImageIdentity,
        emitted_file_identity: EmittedFileIdentity,
        output_shape: tuple[int, int, int],
    ) -> ReconstructionIdentity:
        if emitted_file_identity.payload_budget_bytes <= 0:
            raise G8DContractError("reconstruction emitted-file budget is invalid")
        return ReconstructionIdentity(
            image_identity_id=image_identity.identity_id,
            emitted_file_identity_id=emitted_file_identity.identity_id,
            codec_configuration_id=self.codec_identity.identity_id,
            output_shape=tuple(int(value) for value in output_shape),
            upsample_interpolation=self.upsample_interpolation,
            preserves_aspect=self.preserves_aspect,
        )

    def _cache_path(self, identity: ReconstructionIdentity) -> Path:
        return self.cache_root / "reconstruction" / f"{identity.identity_id}.json"

    @staticmethod
    def _cache_object_id(metadata_without_id: Mapping[str, Any]) -> str:
        return "g8dreconobj-" + sha256_bytes(canonical_json(metadata_without_id))

    def _load_cache(
        self,
        path: Path,
        *,
        identity: ReconstructionIdentity,
        image_identity: ImageIdentity,
        emitted_file_identity: EmittedFileIdentity,
        codestream: bytes,
    ) -> tuple[np.ndarray, str]:
        try:
            metadata = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise G8DContractError(f"invalid reconstruction cache object: {exc}") from None
        if not isinstance(metadata, Mapping) or set(metadata) != set(self._CACHE_FIELDS):
            raise G8DContractError("reconstruction cache schema differs")
        if metadata["schema_version"] != CODEC_CACHE_SCHEMA_VERSION:
            raise G8DContractError("unsupported reconstruction cache schema")
        if metadata["reconstruction_identity"] != identity.payload():
            raise G8DContractError("reconstruction cache identity is stale")
        if metadata["image_identity"] != image_identity.payload():
            raise G8DContractError("reconstruction cache image identity differs")
        if metadata["emitted_file_identity"] != emitted_file_identity.payload():
            raise G8DContractError("reconstruction cache emitted-file identity differs")
        if metadata["codec_configuration"] != self.codec_identity.payload():
            raise G8DContractError("reconstruction cache codec configuration differs")
        actual_hash = sha256_bytes(codestream)
        if metadata["input_codestream_sha256"] != actual_hash or actual_hash != emitted_file_identity.codestream_sha256:
            raise G8DContractError("reconstruction cache input codestream differs")
        if metadata["input_emitted_bytes"] != len(codestream) or len(codestream) != emitted_file_identity.emitted_bytes:
            raise G8DContractError("reconstruction cache input length differs")
        try:
            pixels_bytes = base64.b64decode(metadata["decoded_pixels_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise G8DContractError(f"reconstruction cache pixels are invalid: {exc}") from None
        if sha256_bytes(pixels_bytes) != metadata["decoded_pixels_sha256"]:
            raise G8DContractError("reconstruction cache pixel hash differs")
        shape = metadata["decoded_shape"]
        if not isinstance(shape, list) or tuple(shape) != identity.output_shape:
            raise G8DContractError("reconstruction cache output shape differs")
        if metadata["decoded_dtype"] != "uint8":
            raise G8DContractError("reconstruction cache dtype differs")
        expected_bytes = int(np.prod(identity.output_shape, dtype=np.int64))
        if len(pixels_bytes) != expected_bytes:
            raise G8DContractError("reconstruction cache pixel byte length differs")
        pixels = np.frombuffer(pixels_bytes, dtype=np.uint8).reshape(identity.output_shape).copy()
        basis = dict(metadata)
        object_id = basis.pop("cache_object_id")
        if object_id != self._cache_object_id(basis):
            raise G8DContractError("reconstruction cache object ID differs")
        return pixels, str(object_id)

    def _decode_to_output(self, codestream: bytes, output_shape: tuple[int, int, int]) -> np.ndarray:
        try:
            decoded = self._validated_pixels(self.decoder(codestream), "decoded JPEG 2000 pixels")
        except G8DContractError:
            raise
        except Exception as exc:
            raise G8DContractError(f"JPEG 2000 reconstruction decode failed: {exc}") from None
        if tuple(decoded.shape) == output_shape:
            return decoded
        if decoded.shape[0] > output_shape[0] or decoded.shape[1] > output_shape[1]:
            raise G8DContractError("reconstruction resize would downsample decoded pixels")
        try:
            from data.preprocessing import codec_upsample

            result = codec_upsample(decoded, output_hw=output_shape[:2])
        except Exception as exc:
            raise G8DContractError(f"JPEG 2000 reconstruction upsample failed: {exc}") from None
        return self._validated_pixels(result, "upsampled reconstruction")

    def get_or_create(
        self,
        *,
        image_identity: ImageIdentity,
        emitted_file_identity: EmittedFileIdentity,
        codestream: bytes,
        output_shape: tuple[int, int, int],
    ) -> ReconstructionResult:
        if not isinstance(image_identity, ImageIdentity):
            raise G8DContractError("reconstruction requires an image identity")
        if not isinstance(emitted_file_identity, EmittedFileIdentity):
            raise G8DContractError("reconstruction requires an emitted-file identity")
        if not isinstance(codestream, bytes) or not codestream:
            raise G8DContractError("reconstruction codestream must be non-empty bytes")
        if len(codestream) != emitted_file_identity.emitted_bytes:
            raise G8DContractError("reconstruction codestream length differs from identity")
        if sha256_bytes(codestream) != emitted_file_identity.codestream_sha256:
            raise G8DContractError("reconstruction codestream hash differs from identity")
        if self.codec_identity.identity_id == "":
            raise G8DContractError("reconstruction codec identity is empty")
        identity = self._identity(image_identity, emitted_file_identity, output_shape)
        path = self._cache_path(identity)
        try:
            cache_stat = path.lstat()
        except FileNotFoundError:
            cache_stat = None
        if cache_stat is not None:
            if stat.S_ISLNK(cache_stat.st_mode) or not stat.S_ISREG(cache_stat.st_mode):
                raise G8DContractError("reconstruction cache path is not a regular file")
            pixels, object_id = self._load_cache(
                path,
                identity=identity,
                image_identity=image_identity,
                emitted_file_identity=emitted_file_identity,
                codestream=codestream,
            )
            return ReconstructionResult(identity, pixels, True, object_id)

        pixels = self._decode_to_output(codestream, identity.output_shape)
        metadata: dict[str, Any] = {
            "schema_version": CODEC_CACHE_SCHEMA_VERSION,
            "reconstruction_identity": identity.payload(),
            "image_identity": image_identity.payload(),
            "emitted_file_identity": emitted_file_identity.payload(),
            "codec_configuration": self.codec_identity.payload(),
            "input_codestream_sha256": sha256_bytes(codestream),
            "input_emitted_bytes": len(codestream),
            "decoded_pixels_b64": base64.b64encode(pixels.tobytes()).decode("ascii"),
            "decoded_pixels_sha256": sha256_bytes(pixels.tobytes()),
            "decoded_shape": list(pixels.shape),
            "decoded_dtype": str(pixels.dtype),
        }
        metadata["cache_object_id"] = self._cache_object_id(metadata)
        created = publish_immutable_object(path, rendered_json(metadata))
        if not created:
            pixels, object_id = self._load_cache(
                path,
                identity=identity,
                image_identity=image_identity,
                emitted_file_identity=emitted_file_identity,
                codestream=codestream,
            )
            return ReconstructionResult(identity, pixels, True, object_id)
        return ReconstructionResult(identity, pixels, False, str(metadata["cache_object_id"]))


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkUnitIdentity":
        return cls(**_identity_input(value, cls.FIELDS, "work-unit identity", "work_unit"))


@dataclass(frozen=True)
class CleanClassifierMeasurementRecord:
    """A count-authoritative clean-classifier validation measurement.

    This is a record seam, not a classifier runner.  The only constructor
    that derives counts from outcomes accepts booleans; it never accepts a
    caller-supplied accuracy float.  The current D4 tests use synthetic
    outcomes and explicitly mark the resulting object non-scientific and
    non-mergeable.
    """

    work_unit: WorkUnitIdentity
    candidate: CandidateIdentity
    image: ImageIdentity
    validation_split: ValidationSplitIdentity
    classifier: ClassifierIdentity
    g8_c_table: G8CTableIdentity
    reconstruction: ReconstructionIdentity
    reconstruction_cache_object_id: str
    correct_count: int
    total_count: int
    source: str
    scientific_evidence: bool = False
    merge_eligible: bool = False

    FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "artifact_role",
        "record_id",
        "work_unit",
        "candidate",
        "image",
        "validation_split",
        "classifier",
        "g8_c_table",
        "reconstruction",
        "reconstruction_cache_object_id",
        "correct_count",
        "total_count",
        "accuracy",
        "accuracy_derivation",
        "source",
        "validation_only",
        "test_access",
        "training",
        "scientific_evidence",
        "merge_eligible",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.work_unit, WorkUnitIdentity):
            raise G8DContractError("clean measurement has no work-unit identity")
        if not isinstance(self.candidate, CandidateIdentity):
            raise G8DContractError("clean measurement has no candidate identity")
        if not isinstance(self.image, ImageIdentity):
            raise G8DContractError("clean measurement has no image identity")
        if not isinstance(self.validation_split, ValidationSplitIdentity):
            raise G8DContractError("clean measurement has no validation split identity")
        if not isinstance(self.classifier, ClassifierIdentity):
            raise G8DContractError("clean measurement has no classifier identity")
        if not isinstance(self.g8_c_table, G8CTableIdentity):
            raise G8DContractError("clean measurement has no G8_C table identity")
        if not isinstance(self.reconstruction, ReconstructionIdentity):
            raise G8DContractError("clean measurement has no reconstruction identity")
        if self.validation_split.split != VALIDATION_SPLIT:
            raise G8DContractError("clean measurement is not validation-only")
        if self.validation_split.payload() != {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "validation_split",
            "dataset": self.image.dataset,
            "split": self.image.split,
            "dataset_version": self.image.dataset_version,
            "manifest_sha256": self.image.manifest_sha256,
        }:
            raise G8DContractError("clean measurement image and split identities differ")
        if self.classifier.dataset != self.validation_split.dataset or self.classifier.split != VALIDATION_SPLIT:
            raise G8DContractError("clean measurement classifier is not bound to this validation split")
        if self.classifier.dataset_version != self.validation_split.dataset_version or self.classifier.manifest_sha256 != self.validation_split.manifest_sha256:
            raise G8DContractError("clean measurement classifier manifest differs")
        if self.candidate.image_identity_id != self.image.identity_id:
            raise G8DContractError("clean measurement candidate and image identities differ")
        if self.work_unit.candidate_identity_id != self.candidate.identity_id:
            raise G8DContractError("clean measurement work unit and candidate identities differ")
        if self.candidate.g8_c_table_identity_id != self.g8_c_table.identity_id:
            raise G8DContractError("clean measurement candidate and G8_C table identities differ")
        if self.g8_c_table.payload() != _current_g8_c_binding(REPO_ROOT).payload():
            raise G8DContractError("clean measurement is not bound to the frozen Pascal successor table")
        if self.classifier.payload() != _current_classifier_binding(REPO_ROOT).payload():
            raise G8DContractError("clean measurement classifier differs from the frozen clean classifier")
        if self.reconstruction.image_identity_id != self.image.identity_id:
            raise G8DContractError("clean measurement reconstruction and image identities differ")
        if self.reconstruction.output_shape != self.image.canonical_shape:
            raise G8DContractError("clean measurement reconstruction output shape differs from canonical image")
        if self.reconstruction.codec_configuration_id != self.candidate.codec_configuration_id:
            raise G8DContractError("clean measurement reconstruction and candidate codec identities differ")
        if not isinstance(self.reconstruction_cache_object_id, str) or not self.reconstruction_cache_object_id.startswith("g8dreconobj-"):
            raise G8DContractError("clean measurement reconstruction cache object ID is invalid")
        _nonnegative_int(self.correct_count, "clean measurement correct_count")
        _positive_int(self.total_count, "clean measurement total_count")
        if self.correct_count > self.total_count:
            raise G8DContractError("clean measurement correct_count exceeds total_count")
        if not isinstance(self.source, str) or not self.source.strip():
            raise G8DContractError("clean measurement has no provenance source")
        if self.scientific_evidence is not False or self.merge_eligible is not False:
            raise G8DContractError("D4 synthetic clean measurements are non-scientific and non-mergeable")

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_count

    @property
    def record_id(self) -> str:
        return "g8drecord-" + sha256_bytes(canonical_json(self._payload_without_record_id()))

    def _payload_without_record_id(self) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "artifact_role": "g8_d_clean_classifier_measurement",
            "work_unit": self.work_unit.payload(),
            "candidate": self.candidate.payload(),
            "image": self.image.payload(),
            "validation_split": self.validation_split.payload(),
            "classifier": self.classifier.payload(),
            "g8_c_table": self.g8_c_table.payload(),
            "reconstruction": self.reconstruction.payload(),
            "reconstruction_cache_object_id": self.reconstruction_cache_object_id,
            "correct_count": self.correct_count,
            "total_count": self.total_count,
            "accuracy": self.accuracy,
            "accuracy_derivation": "correct_count / total_count",
            "source": self.source,
            "validation_only": True,
            "test_access": 0,
            "training": False,
            "scientific_evidence": self.scientific_evidence,
            "merge_eligible": self.merge_eligible,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload_without_record_id()
        payload["record_id"] = self.record_id
        return {field: payload[field] for field in self.FIELDS}

    def measured_accuracy(self) -> composition.MeasuredCodecAccuracy:
        return composition.MeasuredCodecAccuracy(
            correct=self.correct_count,
            total=self.total_count,
            split=self.validation_split.split,
            source=f"{self.source}:{self.record_id}",
        )

    @classmethod
    def from_outcomes(
        cls,
        *,
        work_unit: WorkUnitIdentity,
        candidate: CandidateIdentity,
        image: ImageIdentity,
        validation_split: ValidationSplitIdentity,
        classifier: ClassifierIdentity,
        g8_c_table: G8CTableIdentity,
        reconstruction: ReconstructionIdentity,
        reconstruction_cache_object_id: str,
        outcomes: Sequence[bool],
        source: str,
    ) -> "CleanClassifierMeasurementRecord":
        if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
            raise G8DContractError("clean measurement outcomes must be a finite boolean sequence")
        if not outcomes:
            raise G8DContractError("clean measurement outcomes are empty")
        if any(type(outcome) is not bool for outcome in outcomes):
            raise G8DContractError("clean measurement outcomes must contain booleans")
        return cls(
            work_unit=work_unit,
            candidate=candidate,
            image=image,
            validation_split=validation_split,
            classifier=classifier,
            g8_c_table=g8_c_table,
            reconstruction=reconstruction,
            reconstruction_cache_object_id=reconstruction_cache_object_id,
            correct_count=sum(outcomes),
            total_count=len(outcomes),
            source=source,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CleanClassifierMeasurementRecord":
        data = _strict_object(value, cls.FIELDS, "clean classifier measurement record")
        if data["schema_version"] != RECORD_SCHEMA_VERSION or data["artifact_role"] != "g8_d_clean_classifier_measurement":
            raise G8DContractError("clean measurement record schema differs")
        if data["accuracy_derivation"] != "correct_count / total_count":
            raise G8DContractError("clean measurement accuracy is not count-derived")
        if data["validation_only"] is not True or data["test_access"] != 0 or data["training"] is not False:
            raise G8DContractError("clean measurement split or safety binding differs")
        record = cls(
            work_unit=WorkUnitIdentity.from_mapping(data["work_unit"]),
            candidate=CandidateIdentity.from_mapping(data["candidate"]),
            image=ImageIdentity.from_mapping(data["image"]),
            validation_split=ValidationSplitIdentity.from_mapping(data["validation_split"]),
            classifier=ClassifierIdentity.from_mapping(data["classifier"]),
            g8_c_table=G8CTableIdentity.from_mapping(data["g8_c_table"]),
            reconstruction=ReconstructionIdentity.from_mapping(data["reconstruction"]),
            reconstruction_cache_object_id=data["reconstruction_cache_object_id"],
            correct_count=data["correct_count"],
            total_count=data["total_count"],
            source=data["source"],
            scientific_evidence=data["scientific_evidence"],
            merge_eligible=data["merge_eligible"],
        )
        if data["record_id"] != record.record_id:
            raise G8DContractError("clean measurement record ID differs")
        if data["accuracy"] != record.accuracy:
            raise G8DContractError("clean measurement accuracy differs from counts")
        return record


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
    """Build the deterministic pre-data contract without touching image payloads."""

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
        "checkpoint": "D4",
        "status": "clean_classifier_records_ready",
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
            "br11_accounting": list(BR11Accounting.FIELDS),
            "clean_classifier_measurement": list(CleanClassifierMeasurementRecord.FIELDS),
            "codec_search_result": [
                "search_key", "status", "reason", "payload_budget_bytes",
                "encoded_pixels_sha256", "emitted_identity", "requested_compression_ratio",
                "search_trace", "backend_cache_key", "cache_object_id",
            ],
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
            "codec_search_result_fields": [
                "search_key", "status", "reason", "payload_budget_bytes",
                "encoded_pixels_sha256", "emitted_identity", "requested_compression_ratio",
                "search_trace", "backend_cache_key", "cache_object_id",
            ],
            "structural_infeasibility_is_distinct": True,
            "codec_infeasibility_is_recorded": True,
            "reconstruction_identity_fields": list(ReconstructionIdentity.FIELDS),
            "reconstruction_cache_binds_emitted_bytes": True,
            "reconstruction_cache_binds_image_identity": True,
            "reconstruction_cache_binds_codec_identity": True,
            "reconstruction_cache_immutable": True,
            "br11_accounting_rule": "AM-81",
            "br11_header_is_structural_codestream_bytes": True,
            "br11_payload_is_all_tile_part_data": True,
            "br11_filler_is_separate": True,
            "br11_includes_decode_failures": True,
            "br11_infeasible_rows_excluded": True,
        },
        "record_schema": {
            "schema_version": RECORD_SCHEMA_VERSION,
            "measured_accuracy_requires_counts": True,
            "allowed_split": VALIDATION_SPLIT,
            "allowed_verdicts": [
                "structural_infeasibility", "codec_infeasibility", "decode_failure", "delivered"
            ],
            "infeasible_candidates_are_records": True,
            "br11_fields": list(BR11Accounting.FIELDS),
            "br11_counts_emitted_rows_only": True,
            "br11_count_derived_accuracy": True,
            "clean_classifier_record_fields": list(CleanClassifierMeasurementRecord.FIELDS),
            "clean_classifier_record_is_count_derived": True,
            "clean_classifier_record_validation_only": True,
            "clean_classifier_record_binds_classifier": True,
            "clean_classifier_record_binds_reconstruction_cache": True,
            "clean_classifier_record_merge_eligible": False,
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
        "next_gate": "G8_D/D5",
    }
    campaign_basis = dict(body)
    campaign_basis.pop("campaign_id")
    campaign_basis.pop("contract_id")
    body["campaign_id"] = "g8d-" + sha256_bytes(canonical_json(campaign_basis))
    contract_basis = dict(body)
    contract_basis.pop("contract_id")
    body["contract_id"] = "g8dcontract-" + sha256_bytes(canonical_json(contract_basis))
    return body


# ---------------------------------------------------------------------------
# D2 — emitted-byte-authoritative JPEG 2000 search
# ---------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    """Synchronise a directory or fail; durability is part of the cache claim."""

    flags = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, os.O_RDONLY | flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_object(path: Path, payload: bytes) -> bool:
    """Publish bytes once with a same-directory no-replace hard link.

    Returns ``True`` for a new object and ``False`` for exact idempotence.  A
    collision, dangling symlink, partial file or directory is an error.  This
    is used for content-addressed codec/reconstruction/record objects, where
    replacing a final pathname would turn a cache hit into silent evidence
    mutation.
    """

    if not isinstance(payload, bytes):
        raise G8DContractError("immutable object payload must be bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise G8DContractError(f"immutable object target is not a regular file: {path}")
        if path.read_bytes() == payload:
            return False
        raise G8DContractError(f"immutable object collision at {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            try:
                existing = path.lstat()
            except FileNotFoundError:
                raise G8DContractError(f"immutable object race lost without a target: {path}") from None
            if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
                raise G8DContractError(f"immutable object target is not a regular file: {path}")
            if path.read_bytes() != payload:
                raise G8DContractError(f"immutable object collision at {path}")
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)
        # The directory entry for the temporary name is not evidence, but its
        # removal is still made durable before the caller can report success.
        _fsync_directory(path.parent)


def _validated_codec_pixels(image: np.ndarray, axis: int, image_identity: ImageIdentity) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise G8DContractError("codec search input must be uint8 RGB HWC")
    if min(image.shape[:2]) != axis:
        raise G8DContractError("encoded image shorter side does not equal encode_axis_px")
    if axis > min(image_identity.canonical_shape[:2]):
        raise G8DContractError("downsample axis would upscale the canonical image")
    return np.ascontiguousarray(image)


def _search_trace_record(trace: Any) -> list[dict[str, Any]]:
    if trace is None:
        return []
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise G8DContractError("codec search trace is not a sequence")
    records: list[dict[str, Any]] = []
    fields = ("iteration", "compression_ratio", "emitted_bytes", "within_budget")
    for point in trace:
        if isinstance(point, Mapping):
            data = _strict_object(point, fields, "codec search trace point")
        else:
            data = {field: getattr(point, field, None) for field in fields}
            if any(value is None for value in data.values()):
                raise G8DContractError("codec search trace point is incomplete")
        if isinstance(data["within_budget"], bool) is False:
            raise G8DContractError("codec search trace within_budget is not boolean")
        records.append({
            "iteration": _nonnegative_int(data["iteration"], "codec search iteration"),
            "compression_ratio": _finite_float(data["compression_ratio"], "compression_ratio"),
            "emitted_bytes": _nonnegative_int(data["emitted_bytes"], "trace emitted_bytes"),
            "within_budget": data["within_budget"],
        })
    return records


@dataclass(frozen=True)
class CodecSearchResult:
    """One explicit codec-search outcome; infeasibility is never omitted."""

    search_key: CodecSearchKey
    status: str
    reason: str | None
    payload_budget_bytes: int
    encoded_pixels_sha256: str
    emitted_codestream: bytes | None
    emitted_identity: EmittedFileIdentity | None
    requested_compression_ratio: float | None
    search_trace: tuple[dict[str, Any], ...]
    backend_cache_key: str | None
    cache_hit: bool
    cache_object_id: str

    def __post_init__(self) -> None:
        if self.status not in {STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY, CODEC_FEASIBLE}:
            raise G8DContractError(f"unknown codec-search status {self.status!r}")
        _positive_int(self.payload_budget_bytes, "payload_budget_bytes")
        _require_digest(self.encoded_pixels_sha256, "encoded_pixels_sha256")
        if not isinstance(self.cache_hit, bool) or not self.cache_object_id:
            raise G8DContractError("codec cache provenance is incomplete")
        if self.status == CODEC_FEASIBLE:
            if not isinstance(self.emitted_codestream, bytes) or self.emitted_identity is None:
                raise G8DContractError("feasible codec result has no emitted bytes")
            if len(self.emitted_codestream) != self.emitted_identity.emitted_bytes:
                raise G8DContractError("emitted identity length differs from bytes")
        elif self.emitted_codestream is not None or self.emitted_identity is not None:
            raise G8DContractError("infeasible codec result carries emitted bytes")

    @property
    def feasible(self) -> bool:
        return self.status == CODEC_FEASIBLE

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CACHE_SCHEMA_VERSION,
            "search_key": self.search_key.payload(),
            "status": self.status,
            "reason": self.reason,
            "payload_budget_bytes": self.payload_budget_bytes,
            "encoded_pixels_sha256": self.encoded_pixels_sha256,
            "emitted_identity": None if self.emitted_identity is None else self.emitted_identity.payload(),
            "requested_compression_ratio": self.requested_compression_ratio,
            "search_trace": list(self.search_trace),
            "backend_cache_key": self.backend_cache_key,
            "cache_object_id": self.cache_object_id,
            "cache_hit": self.cache_hit,
        }


class CodecSearchEngine:
    """Run/cache the frozen JPEG2000 search without treating requested ratio as truth."""

    _CACHE_FIELDS = (
        "schema_version", "search_key", "status", "reason", "payload_budget_bytes",
        "encoded_pixels_sha256", "emitted_identity", "codestream_b64",
        "requested_compression_ratio", "search_trace", "backend_cache_key", "cache_object_id",
    )

    def __init__(
        self,
        cache_root: Path,
        *,
        backend: Any | None = None,
        codec_identity: CodecConfigurationIdentity | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        if backend is None:
            from baseline.j2k import J2KCodec

            backend = J2KCodec(self.cache_root / "backend")
        self.backend = backend
        if codec_identity is None:
            snapshot = getattr(backend, "snapshot", None)
            configuration_hash = getattr(backend, "configuration_hash", None)
            if not isinstance(snapshot, Mapping) or not isinstance(configuration_hash, str):
                raise G8DContractError("codec backend exposes no authenticated snapshot/hash")
            runtime_version = str(snapshot.get("environment", {}).get("openjpeg", ""))
            codec_identity = CodecConfigurationIdentity(snapshot, configuration_hash, runtime_version)
        self.codec_identity = codec_identity

    def _cache_path(self, key: CodecSearchKey) -> Path:
        return self.cache_root / "codec_search" / f"{key.identity_id}.json"

    @staticmethod
    def _cache_object_id(metadata_without_id: Mapping[str, Any]) -> str:
        return "g8dcodecobj-" + sha256_bytes(canonical_json(metadata_without_id))

    def _load_cache(
        self,
        path: Path,
        *,
        key: CodecSearchKey,
        encoded_pixels_sha256: str,
        budget: BudgetIdentity,
    ) -> CodecSearchResult:
        try:
            raw = path.read_bytes()
            metadata = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise G8DContractError(f"invalid codec cache object: {exc}") from None
        if not isinstance(metadata, Mapping) or set(metadata) != set(self._CACHE_FIELDS):
            raise G8DContractError("codec cache schema differs")
        if metadata["schema_version"] != CODEC_CACHE_SCHEMA_VERSION or metadata["search_key"] != key.payload():
            raise G8DContractError("codec cache key is stale")
        if metadata["encoded_pixels_sha256"] != encoded_pixels_sha256:
            raise G8DContractError("codec cache encoded pixels differ")
        if metadata["payload_budget_bytes"] != budget.payload_bytes:
            raise G8DContractError("codec cache budget differs")
        codestream: bytes | None = None
        if metadata["codestream_b64"] is not None:
            if not isinstance(metadata["codestream_b64"], str):
                raise G8DContractError("codec cache codestream is not base64")
            try:
                codestream = base64.b64decode(metadata["codestream_b64"], validate=True)
            except (ValueError, TypeError) as exc:
                raise G8DContractError(f"codec cache codestream is invalid: {exc}") from None
        if metadata["status"] == CODEC_FEASIBLE:
            if not codestream or metadata["emitted_identity"] is None:
                raise G8DContractError("feasible codec cache has no codestream")
            emitted = EmittedFileIdentity.from_mapping(metadata["emitted_identity"])
            if len(codestream) != emitted.emitted_bytes or sha256_bytes(codestream) != emitted.codestream_sha256:
                raise G8DContractError("codec cache emitted bytes do not match identity")
            if emitted.payload_budget_bytes != budget.payload_bytes:
                raise G8DContractError("codec cache emitted budget differs")
        else:
            emitted = None
            if codestream is not None or metadata["emitted_identity"] is not None:
                raise G8DContractError("infeasible codec cache carries bytes")
            if metadata["status"] != CODEC_INFEASIBILITY:
                raise G8DContractError("codec cache status is not an allowed infeasibility")
        basis = dict(metadata)
        object_id = basis.pop("cache_object_id")
        expected_object_id = self._cache_object_id(basis)
        if object_id != expected_object_id:
            raise G8DContractError("codec cache object ID differs")
        trace = tuple(_search_trace_record(metadata["search_trace"]))
        ratio = metadata["requested_compression_ratio"]
        if ratio is not None:
            ratio = _finite_float(ratio, "requested_compression_ratio")
        return CodecSearchResult(
            search_key=key,
            status=str(metadata["status"]),
            reason=None if metadata["reason"] is None else str(metadata["reason"]),
            payload_budget_bytes=budget.payload_bytes,
            encoded_pixels_sha256=encoded_pixels_sha256,
            emitted_codestream=codestream,
            emitted_identity=emitted,
            requested_compression_ratio=ratio,
            search_trace=trace,
            backend_cache_key=None if metadata["backend_cache_key"] is None else str(metadata["backend_cache_key"]),
            cache_hit=True,
            cache_object_id=object_id,
        )

    def _write_cache(
        self,
        path: Path,
        *,
        key: CodecSearchKey,
        status: str,
        reason: str | None,
        budget: BudgetIdentity,
        encoded_pixels_sha256: str,
        codestream: bytes | None,
        emitted_identity: EmittedFileIdentity | None,
        requested_ratio: float | None,
        trace: list[dict[str, Any]],
        backend_cache_key: str | None,
    ) -> str:
        metadata: dict[str, Any] = {
            "schema_version": CODEC_CACHE_SCHEMA_VERSION,
            "search_key": key.payload(),
            "status": status,
            "reason": reason,
            "payload_budget_bytes": budget.payload_bytes,
            "encoded_pixels_sha256": encoded_pixels_sha256,
            "emitted_identity": None if emitted_identity is None else emitted_identity.payload(),
            "codestream_b64": None if codestream is None else base64.b64encode(codestream).decode("ascii"),
            "requested_compression_ratio": requested_ratio,
            "search_trace": trace,
            "backend_cache_key": backend_cache_key,
        }
        metadata["cache_object_id"] = self._cache_object_id(metadata)
        publish_immutable_object(path, rendered_json(metadata))
        return str(metadata["cache_object_id"])

    def search(
        self,
        *,
        image_identity: ImageIdentity,
        encoded_image: np.ndarray,
        budget: BudgetIdentity,
        encode_axis_px: int,
        structurally_feasible: bool = True,
        structural_reason: str | None = None,
    ) -> CodecSearchResult:
        """Search one configured axis and preserve every infeasible outcome."""

        pixels = _validated_codec_pixels(encoded_image, encode_axis_px, image_identity)
        encoded_pixels_sha256 = sha256_bytes(pixels.tobytes())
        key = CodecSearchKey(
            image_identity.identity_id,
            budget.identity_id,
            self.codec_identity.identity_id,
            encode_axis_px,
        )
        if not structurally_feasible:
            object_id = "g8dcodecobj-" + sha256_bytes(canonical_json({"status": STRUCTURAL_INFEASIBILITY, "key": key.payload()}))
            return CodecSearchResult(
                search_key=key,
                status=STRUCTURAL_INFEASIBILITY,
                reason=structural_reason or "packetisation_infeasible",
                payload_budget_bytes=budget.payload_bytes,
                encoded_pixels_sha256=encoded_pixels_sha256,
                emitted_codestream=None,
                emitted_identity=None,
                requested_compression_ratio=None,
                search_trace=(),
                backend_cache_key=None,
                cache_hit=False,
                cache_object_id=object_id,
            )

        cache_path = self._cache_path(key)
        try:
            cache_stat = cache_path.lstat()
        except FileNotFoundError:
            cache_stat = None
        if cache_stat is not None:
            if stat.S_ISLNK(cache_stat.st_mode) or not stat.S_ISREG(cache_stat.st_mode):
                raise G8DContractError(f"codec cache path is not a regular file: {cache_path}")
            return self._load_cache(cache_path, key=key, encoded_pixels_sha256=encoded_pixels_sha256, budget=budget)

        try:
            backend_result = self.backend.encode_to_budget(
                pixels,
                canonical_pixels_sha256=image_identity.canonical_pixels_sha256,
                budget_bytes=budget.payload_bytes,
                encode_axis_px=encode_axis_px,
            )
        except Exception as exc:
            # Backend/configuration failure is still a codec infeasibility
            # record.  The exception text is retained; it is never swallowed.
            status = CODEC_INFEASIBILITY
            reason = f"codec_configuration_error: {exc}"
            trace: list[dict[str, Any]] = []
            requested_ratio = None
            backend_cache_key = None
            codestream = None
            emitted_identity = None
        else:
            feasible_flag = getattr(backend_result, "feasible", None)
            if not isinstance(feasible_flag, bool):
                raise G8DContractError("codec backend result has no boolean feasible flag")
            codestream = getattr(backend_result, "codestream", None)
            if feasible_flag and not isinstance(codestream, bytes):
                raise G8DContractError("codec backend marked feasible without codestream bytes")
            if not feasible_flag and codestream is not None:
                raise G8DContractError("codec backend marked infeasible while returning bytes")
            trace = _search_trace_record(getattr(backend_result, "search_trace", ()))
            requested_ratio = getattr(backend_result, "compression_ratio_argument", None)
            if requested_ratio is not None:
                requested_ratio = _finite_float(requested_ratio, "requested_compression_ratio")
            backend_cache_key = getattr(backend_result, "cache_key", None)
            if feasible_flag:
                emitted_bytes = len(codestream)
                if emitted_bytes > budget.payload_bytes:
                    raise G8DContractError("actual emitted codestream exceeds payload budget")
                reported = getattr(backend_result, "emitted_byte_count", emitted_bytes)
                if reported is not None and reported != emitted_bytes:
                    raise G8DContractError("backend emitted-byte count disagrees with actual bytes")
                emitted_identity = EmittedFileIdentity(
                    codec_search_key_id=key.identity_id,
                    codestream_sha256=sha256_bytes(codestream),
                    emitted_bytes=emitted_bytes,
                    payload_budget_bytes=budget.payload_bytes,
                    filler_bytes=budget.payload_bytes - emitted_bytes,
                )
                status = CODEC_FEASIBLE
                reason = None
            else:
                emitted_identity = None
                status = CODEC_INFEASIBILITY
                reason = "budget_exceeded"

        object_id = self._write_cache(
            cache_path,
            key=key,
            status=status,
            reason=reason,
            budget=budget,
            encoded_pixels_sha256=encoded_pixels_sha256,
            codestream=codestream,
            emitted_identity=emitted_identity,
            requested_ratio=requested_ratio,
            trace=trace,
            backend_cache_key=None if backend_cache_key is None else str(backend_cache_key),
        )
        return CodecSearchResult(
            search_key=key,
            status=status,
            reason=reason,
            payload_budget_bytes=budget.payload_bytes,
            encoded_pixels_sha256=encoded_pixels_sha256,
            emitted_codestream=codestream,
            emitted_identity=emitted_identity,
            requested_compression_ratio=requested_ratio,
            search_trace=tuple(trace),
            backend_cache_key=None if backend_cache_key is None else str(backend_cache_key),
            cache_hit=False,
            cache_object_id=object_id,
        )

    def search_with_packet_plan(
        self,
        *,
        image_identity: ImageIdentity,
        encoded_image: np.ndarray,
        budget: BudgetIdentity,
        encode_axis_px: int,
        k_symbols: int,
        modulation: str,
        ldpc_rate: str,
    ) -> CodecSearchResult:
        """Bind codec feasibility to the real packet plan before encoding."""

        from baseline.classical.channel_transport import build_accounting
        from baseline.ldpc.transport import build_packet_plan

        packet = build_packet_plan(k_symbols, modulation, ldpc_rate)
        if not packet.feasible:
            return self.search(
                image_identity=image_identity,
                encoded_image=encoded_image,
                budget=budget,
                encode_axis_px=encode_axis_px,
                structurally_feasible=False,
                structural_reason=packet.reason or "packetisation_infeasible",
            )
        accounting = build_accounting(packet)
        if accounting.payload_bytes != budget.payload_bytes:
            raise G8DContractError("budget identity does not match packet accounting payload")
        declared = dict(budget.packet_accounting)
        if declared and declared != accounting.as_dict():
            raise G8DContractError("budget identity packet accounting differs from frozen packet plan")
        return self.search(
            image_identity=image_identity,
            encoded_image=encoded_image,
            budget=budget,
            encode_axis_px=encode_axis_px,
        )
