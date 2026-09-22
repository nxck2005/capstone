"""Lossless publication and loading of the completed W10 JPEG selection.

The scientific JSON is intentionally kept as the logical artifact, but its
133 MiB representation is too large for an ordinary Git blob.  This module
publishes and authenticates a deterministic XZ carrier without parsing and
re-rendering the JSON.  Loading never materializes the raw logical pathname.
"""

from __future__ import annotations

import hashlib
import json
import lzma
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.deterministic_core import canonical_bytes, canonical_sha256

JPEG_SELECTION_PATH = "results/learned/w10/jpeg_validation_selection.json"
JPEG_CARRIER_PATH = "results/learned/w10/jpeg_validation_selection.json.xz"
JPEG_CARRIER_DESCRIPTOR_PATH = "results/learned/w10/jpeg_validation_selection_carrier.json"
JPEG_CARRIER_ID_PREFIX = "w10jpegcarrier-"
JPEG_CARRIER_SCHEMA_VERSION = 1
JPEG_CARRIER_ARTIFACT_ROLE = "W10_JPEG_SECONDARY_SELECTION_LOSSLESS_CARRIER"
JPEG_CARRIER_STATUS = "PUBLISHED_LOSSLESS_CARRIER"
JPEG_CARRIER_COMPRESSION = "xz"
JPEG_CARRIER_COMPRESSION_SETTINGS = {
    "format": "xz",
    "check": "crc64",
    "preset": 9,  # literal-ok: fixed deterministic XZ compression preset
    "extreme": True,
}
JPEG_CARRIER_MAX_BYTES_EXCLUSIVE = 95 * 1024 * 1024  # literal-ok: publication safety ceiling and MiB-to-byte conversion

JPEG_SELECTION_PREFIX = "w10jpegselection-"
JPEG_SELECTION_ID = "w10jpegselection-ff847b73e8f117bf6061b74f21d33fa48db4fbedd7399a105e8a5733b039601d"
JPEG_SELECTION_CONTRACT_SHA256 = "2ff0c39c63b7a522a6756c93614723a183641a115b16fa3083e1dd88cfd54870"
JPEG_SELECTION_RAW_SHA256 = "8618cbc638ac430778f35c771948d779a8ead91ecbe4182156097cc499cf952e"
JPEG_SELECTION_RAW_BYTES = 133540835
JPEG_SELECTION_SOURCE_RECORD = {
    "path": "results/learned/w10/w10_downstream_source_manifest_v5.json",
    "manifest_id": "w10downstreamsourcev5-00e19a1c1fbf0a6052347a58db57d7236e1efde9914a978e3b45408844339652",
    "sha256": "4fe35441999bc5771fa6fea10692378eab76ce6e5ac626d68202d80802c81ae3",
}


class JpegCarrierHold(RuntimeError):
    """The JPEG raw artifact, carrier or descriptor is not authentic."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JpegCarrierHold(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_bytes(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise JpegCarrierHold(f"{label} cannot be read: {exc}") from None


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JpegCarrierHold(f"{label} is corrupt: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _validate_selection(raw: bytes) -> dict[str, Any]:
    _require(len(raw) == JPEG_SELECTION_RAW_BYTES, "JPEG raw logical byte length differs")
    _require(_sha256(raw) == JPEG_SELECTION_RAW_SHA256, "JPEG raw logical SHA-256 differs")
    value = _parse_object(raw, "JPEG selection")
    body = dict(value)
    identifier = body.pop("selection_id", None)
    _require(identifier == JPEG_SELECTION_ID, "JPEG selection ID differs")
    _require(identifier == JPEG_SELECTION_PREFIX + canonical_sha256(body), "JPEG selection ID is not content-addressed")
    _require(value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256, "JPEG selection contract SHA-256 differs")
    _require(value.get("source_epoch") == JPEG_SELECTION_SOURCE_RECORD, "JPEG selection source epoch differs")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "JPEG selection crossed the test boundary")
    selections = value.get("selections")
    score_table = value.get("candidate_scores")
    _require(isinstance(selections, list) and len(selections) == 21, "JPEG selection does not cover 21 SNRs")  # literal-ok: frozen completed JPEG selection SNR cardinality
    _require(isinstance(score_table, list) and len(score_table) == 21, "JPEG score table does not cover 21 SNRs")  # literal-ok: frozen completed JPEG selection SNR cardinality
    counts = []
    for point in score_table:
        _require(isinstance(point, dict) and isinstance(point.get("candidates"), list), "JPEG score table point is malformed")
        counts.append(len(point["candidates"]))
    _require(counts == [912] * 21, "JPEG score table candidate cardinality differs")  # literal-ok: frozen completed JPEG selection SNR cardinality
    _require(sum(counts) == 19152, "JPEG score table record count differs")
    return value


def _validate_descriptor(value: dict[str, Any]) -> None:
    body = dict(value)
    identifier = body.pop("carrier_id", None)
    _require(identifier == JPEG_CARRIER_ID_PREFIX + canonical_sha256(body), "JPEG carrier descriptor ID differs")
    _require(value.get("schema_version") == JPEG_CARRIER_SCHEMA_VERSION, "JPEG carrier descriptor schema differs")
    _require(value.get("artifact_role") == JPEG_CARRIER_ARTIFACT_ROLE, "JPEG carrier descriptor role differs")
    _require(value.get("status") == JPEG_CARRIER_STATUS, "JPEG carrier descriptor status differs")
    _require(value.get("compression") == JPEG_CARRIER_COMPRESSION, "JPEG carrier compression differs")
    _require(value.get("compression_settings") == JPEG_CARRIER_COMPRESSION_SETTINGS, "JPEG carrier compression settings differ")
    _require(value.get("carrier_path") == JPEG_CARRIER_PATH, "JPEG carrier path differs")
    _require(value.get("raw_logical_path") == JPEG_SELECTION_PATH, "JPEG raw logical path differs")
    _require(value.get("raw_sha256") == JPEG_SELECTION_RAW_SHA256, "JPEG descriptor raw SHA-256 differs")
    _require(value.get("raw_bytes") == JPEG_SELECTION_RAW_BYTES, "JPEG descriptor raw byte length differs")
    _require(value.get("selection_id") == JPEG_SELECTION_ID, "JPEG descriptor selection ID differs")
    _require(value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256, "JPEG descriptor contract SHA-256 differs")
    _require(value.get("source_epoch") == JPEG_SELECTION_SOURCE_RECORD, "JPEG descriptor source epoch differs")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "JPEG descriptor crossed the test boundary")
    _require(isinstance(value.get("carrier_sha256"), str) and len(value["carrier_sha256"]) == 64, "JPEG carrier SHA-256 is malformed")  # literal-ok: SHA-256 hexadecimal width
    _require(isinstance(value.get("carrier_bytes"), int) and value["carrier_bytes"] > 0, "JPEG carrier byte length is malformed")


def compress_jpeg_selection(raw: bytes) -> bytes:
    """Compress exact raw bytes with the fixed deterministic XZ settings."""

    return lzma.compress(
        raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=lzma.PRESET_EXTREME | 9,  # literal-ok: fixed deterministic XZ compression preset
    )


def decompress_jpeg_carrier(carrier: bytes) -> bytes:
    try:
        return lzma.decompress(carrier, format=lzma.FORMAT_XZ)
    except (EOFError, lzma.LZMAError, ValueError) as exc:
        raise JpegCarrierHold(f"JPEG XZ carrier cannot be decompressed: {exc}") from None


@dataclass(frozen=True)
class LoadedJpegSelection:
    value: dict[str, Any]
    raw_bytes: bytes
    provenance: dict[str, Any]


def _provenance(root: Path, value: dict[str, Any], raw: bytes, *, descriptor: dict[str, Any] | None, descriptor_path: Path | None, carrier: bytes | None) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "logical_artifact_path": JPEG_SELECTION_PATH,
        "selection_id": value["selection_id"],
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "source_epoch": dict(value["source_epoch"]),
        "selection_digest": canonical_sha256(value),
        "carrier_path": None,
        "carrier_sha256": None,
        "carrier_bytes": None,
        "carrier_descriptor_path": None,
        "carrier_descriptor_id": None,
        "carrier_descriptor_sha256": None,
    }
    if descriptor is not None and descriptor_path is not None and carrier is not None:
        provenance.update(
            {
                "carrier_path": str(descriptor["carrier_path"]),
                "carrier_sha256": _sha256(carrier),
                "carrier_bytes": len(carrier),
                "carrier_descriptor_path": str(descriptor_path.relative_to(root)),
                "carrier_descriptor_id": str(descriptor["carrier_id"]),
                "carrier_descriptor_sha256": _sha256(_regular_bytes(descriptor_path, "JPEG carrier descriptor")),
            }
        )
    return provenance


def load_jpeg_selection_artifact(root: Path, *, require_carrier: bool = False) -> LoadedJpegSelection:
    """Load exact JPEG bytes from raw custody or the tracked carrier.

    When both representations exist, the bytes must compare exactly.  The raw
    logical pathname is never written as part of verification.
    """

    root = Path(root).resolve()
    raw_path = root / JPEG_SELECTION_PATH
    carrier_path = root / JPEG_CARRIER_PATH
    descriptor_path = root / JPEG_CARRIER_DESCRIPTOR_PATH
    raw_present = raw_path.exists() or raw_path.is_symlink()
    carrier_present = carrier_path.exists() or carrier_path.is_symlink()
    descriptor_present = descriptor_path.exists() or descriptor_path.is_symlink()
    if require_carrier:
        _require(carrier_present and descriptor_present, "JPEG tracked carrier is missing")
    if carrier_present or descriptor_present:
        _require(carrier_present and descriptor_present, "JPEG carrier and descriptor must be published together")
        descriptor_raw = _regular_bytes(descriptor_path, "JPEG carrier descriptor")
        descriptor = _parse_object(descriptor_raw, "JPEG carrier descriptor")
        _validate_descriptor(descriptor)
        carrier = _regular_bytes(carrier_path, "JPEG XZ carrier")
        _require(len(carrier) == int(descriptor["carrier_bytes"]), "JPEG carrier byte length differs")
        _require(_sha256(carrier) == descriptor["carrier_sha256"], "JPEG carrier SHA-256 differs")
        raw = decompress_jpeg_carrier(carrier)
        _require(len(raw) == int(descriptor["raw_bytes"]), "JPEG decompressed byte length differs")
        _require(_sha256(raw) == descriptor["raw_sha256"], "JPEG decompressed SHA-256 differs")
        value = _validate_selection(raw)
        _require(value["selection_id"] == descriptor["selection_id"], "JPEG carrier selection ID differs")
        _require(value["contract_sha256"] == descriptor["contract_sha256"], "JPEG carrier contract SHA-256 differs")
        _require(value["source_epoch"] == descriptor["source_epoch"], "JPEG carrier source epoch differs")
        if raw_present:
            local_raw = _regular_bytes(raw_path, "JPEG raw selection")
            _require(local_raw == raw, "JPEG raw and carrier bytes differ")
        return LoadedJpegSelection(
            value=value,
            raw_bytes=raw,
            provenance=_provenance(root, value, raw, descriptor=descriptor, descriptor_path=descriptor_path, carrier=carrier),
        )
    _require(raw_present, "JPEG selection artifact is missing")
    raw = _regular_bytes(raw_path, "JPEG raw selection")
    value = _validate_selection(raw)
    _require(not require_carrier, "JPEG tracked carrier is missing")
    return LoadedJpegSelection(
        value=value,
        raw_bytes=raw,
        provenance=_provenance(root, value, raw, descriptor=None, descriptor_path=None, carrier=None),
    )


def _write_immutable(path: Path, raw: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw, f"{label} differs from existing bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def build_jpeg_carrier(root: Path, *, raw_path: Path | None = None) -> LoadedJpegSelection:
    """Build the carrier and immutable descriptor from exact raw bytes."""

    root = Path(root).resolve()
    source = Path(raw_path) if raw_path is not None else root / JPEG_SELECTION_PATH
    raw = _regular_bytes(source, "JPEG raw selection")
    carrier = compress_jpeg_selection(raw)
    _require(len(carrier) < JPEG_CARRIER_MAX_BYTES_EXCLUSIVE, "JPEG compressed carrier is at or above the 95 MiB publication gate")
    # Compression consumes the exact bytes first; parsing only authenticates the
    # resulting package and supplies descriptor fields.
    value = _validate_selection(raw)
    carrier_path = root / JPEG_CARRIER_PATH
    descriptor_path = root / JPEG_CARRIER_DESCRIPTOR_PATH
    _write_immutable(carrier_path, carrier, "JPEG XZ carrier")
    descriptor_body: dict[str, Any] = {
        "schema_version": JPEG_CARRIER_SCHEMA_VERSION,
        "artifact_role": JPEG_CARRIER_ARTIFACT_ROLE,
        "status": JPEG_CARRIER_STATUS,
        "compression": JPEG_CARRIER_COMPRESSION,
        "compression_settings": dict(JPEG_CARRIER_COMPRESSION_SETTINGS),
        "carrier_path": JPEG_CARRIER_PATH,
        "carrier_sha256": _sha256(carrier),
        "carrier_bytes": len(carrier),
        "raw_logical_path": JPEG_SELECTION_PATH,
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "selection_id": value["selection_id"],
        "contract_sha256": value["contract_sha256"],
        "source_epoch": dict(value["source_epoch"]),
        "test": value["test"],
        "test_access": value["test_access"],
    }
    descriptor = dict(descriptor_body)
    descriptor["carrier_id"] = JPEG_CARRIER_ID_PREFIX + canonical_sha256(descriptor_body)
    _write_immutable(descriptor_path, canonical_bytes(descriptor), "JPEG carrier descriptor")
    return load_jpeg_selection_artifact(root, require_carrier=True)


def check_jpeg_carrier(root: Path) -> LoadedJpegSelection:
    """Read-only carrier verification, including parse and identity checks."""

    return load_jpeg_selection_artifact(root, require_carrier=True)


__all__ = [
    "JPEG_CARRIER_ARTIFACT_ROLE",
    "JPEG_CARRIER_COMPRESSION",
    "JPEG_CARRIER_COMPRESSION_SETTINGS",
    "JPEG_CARRIER_DESCRIPTOR_PATH",
    "JPEG_CARRIER_ID_PREFIX",
    "JPEG_CARRIER_MAX_BYTES_EXCLUSIVE",
    "JPEG_CARRIER_PATH",
    "JPEG_CARRIER_SCHEMA_VERSION",
    "JPEG_CARRIER_STATUS",
    "JPEG_SELECTION_CONTRACT_SHA256",
    "JPEG_SELECTION_ID",
    "JPEG_SELECTION_PATH",
    "JPEG_SELECTION_RAW_BYTES",
    "JPEG_SELECTION_RAW_SHA256",
    "JPEG_SELECTION_SOURCE_RECORD",
    "JpegCarrierHold",
    "LoadedJpegSelection",
    "build_jpeg_carrier",
    "check_jpeg_carrier",
    "compress_jpeg_selection",
    "decompress_jpeg_carrier",
    "load_jpeg_selection_artifact",
]
