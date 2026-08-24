"""Fail-closed G8_F/F1 assignment loading and artifact materialization.

F1 is deliberately not authorized by importing this module.  The production
CLI requires both the frozen F0 handoff and a later, separate owner-issued F1
launch authorization.  This module never imports the guarded test loader and
contains no classifier, optimizer, training, selection, or pass-two path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence

import numpy as np

from baseline.g8_f_sampler_plan import (
    AM87_PLAN_FILE_SHA256,
    AM87_PLAN_ID,
    EXPECTED_ATTEMPTS,
    EXPECTED_SEED,
    EXPECTED_VARIANTS,
    PLAN_PATH as SAMPLER_PLAN_PATH,
    _load_am87_support,
    _training_membership,
    canonical_json,
    derive_assignments,
    sha256_bytes,
)
from config.params import REPO_ROOT, get
from data.adapters import SourceSample
from data.preprocessing import canonicalize_source, codec_downsample, codec_input, codec_upsample

SCHEMA_VERSION = 2
REQUEST_ROLE = "g8_f_f1_assigned_pair_request"
RESULT_ROLE = "g8_f_f1_artifact_result"
SAMPLER_PLAN_ID = "g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6"
SAMPLER_PLAN_SHA256 = "eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23"
ORDERED_PAIR_SHA256 = "c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229"
PAIR_SET_SHA256 = "255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e"
CODEC_CONFIGURATION_ID = "g8dcodec-39f14b7eaba4f727c70759eb1c5250e8e13f7d5e871c0831aa6b602aef706858"
CODEC_CONFIGURATION_HASH = "2daf597fd914f56eb9e59df7bc20a88b02816522b3b0b4fd3f2db14d7451a0fa"
MANIFEST_SHA256 = "224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889"
DATASET_ARCHIVE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
CANONICAL_SHAPE = list(get("datasets.imagenette160.image_size"))
SHA256_HEX_LENGTH = 64  # literal-ok: SHA-256 hexadecimal identity width
STABLE_ID_HEX_LENGTH = 16  # literal-ok: frozen stable-sample-ID truncation width
ORDINAL_FILENAME_WIDTH = 5  # literal-ok: sufficient fixed width for 50,814 AM-88 ordinals


class G8FMaterializationHold(RuntimeError):
    """A condition that must stop F1 rather than become an omission."""


class CodecResult(Protocol):
    feasible: bool
    codestream: bytes | None
    emitted_byte_count: int | None
    codec_configuration_hash: str
    decoded_image: np.ndarray | None
    decode_success: bool
    codestream_sha256: str | None
    cache_key: str


class CodecBackend(Protocol):
    configuration_hash: str

    def encode_to_budget(
        self,
        image: np.ndarray,
        *,
        canonical_pixels_sha256: str,
        budget_bytes: int,
        encode_axis_px: int,
    ) -> CodecResult: ...


@dataclass(frozen=True)
class F1Assignment:
    ordinal: int
    stable_sample_id: str
    label: int
    quality_id: str
    payload_budget_bytes: int
    encode_axis_px: int
    codec_configuration_id: str

    @property
    def assignment_id(self) -> str:
        return "g8fassign-" + sha256_bytes(canonical_json(self.as_dict()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "stable_sample_id": self.stable_sample_id,
            "label": self.label,
            "quality_id": self.quality_id,
            "payload_budget_bytes": self.payload_budget_bytes,
            "encode_axis_px": self.encode_axis_px,
            "codec_configuration_id": self.codec_configuration_id,
        }


def rendered_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FMaterializationHold(message)


def _pair_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return sha256_bytes(canonical_json([[stable_id, quality_id] for stable_id, quality_id in pairs]))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_HEX_LENGTH and all(character in "0123456789abcdef" for character in value)


def _safe_file(runtime_root: Path, relative: str, *, expected_relative: str) -> Path:
    """Resolve one canonical runtime-relative file without following symlinks."""

    _require(relative == expected_relative, f"runtime object path is not canonical: {relative!r}")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and pure.parts and all(part not in ("", ".", "..") for part in pure.parts), "unsafe runtime object path")
    root = Path(runtime_root)
    _require(not root.is_symlink(), "F1 runtime root may not be a symlink")
    root = root.resolve()
    candidate = root.joinpath(*pure.parts)
    _require(candidate.parent.resolve(strict=False).is_relative_to(root), "runtime object escapes frozen root")
    current = root
    for part in pure.parts:
        current = current / part
        _require(not current.is_symlink(), f"runtime path may not be a symlink: {relative}")
    _require(candidate.is_file(), f"runtime object is missing or not a regular file: {relative}")
    return candidate


def _canonical_json_object(runtime_root: Path, relative: str, *, description: str) -> tuple[dict[str, Any], bytes]:
    path = _safe_file(runtime_root, relative, expected_relative=relative)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FMaterializationHold(f"cannot authenticate {description}: {exc}") from None
    _require(isinstance(value, dict), f"{description} is not a JSON object")
    _require(raw == rendered_json(value), f"{description} is not canonical rendered JSON")
    return value, raw


def authenticate_request(
    runtime_root: Path,
    assignment: F1Assignment,
    *,
    expected_scientific: bool,
) -> dict[str, Any]:
    """Authenticate one immutable request against its exact frozen assignment."""

    relative = f"requests/{assignment.ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json"
    value, _raw = _canonical_json_object(runtime_root, relative, description="existing F1 request")
    expected_keys = {
        "schema_version", "artifact_role", "scientific", "assignment_id", "assignment",
        "source", "data_identity", "codec", "outcome_semantics", "request_id",
    }
    _require(set(value) == expected_keys, "existing F1 request schema differs")
    _require(value["schema_version"] == SCHEMA_VERSION and value["artifact_role"] == REQUEST_ROLE, "existing F1 request header differs")
    _require(value["scientific"] is expected_scientific, "existing F1 request scientific flag differs")
    body = dict(value)
    request_id = body.pop("request_id")
    _require(request_id == "g8frequest-" + sha256_bytes(canonical_json(body)), "existing F1 request identity differs")
    _require(value["assignment_id"] == assignment.assignment_id, "existing F1 request belongs to another assignment")
    _require(value["assignment"] == assignment.as_dict(), "existing F1 request assignment body differs")

    source = value["source"]
    _require(isinstance(source, dict) and set(source) == {
        "dataset", "split", "source_bytes_sha256", "canonical_pixels_sha256",
        "canonical_shape", "encoded_pixels_sha256", "encoded_shape",
    }, "existing F1 request source schema differs")
    _require(source["dataset"] == "imagenette160" and source["split"] == "train", "existing F1 request is not Imagenette train")
    for key in ("source_bytes_sha256", "canonical_pixels_sha256", "encoded_pixels_sha256"):
        _require(_is_sha256(source[key]), f"existing F1 request has invalid {key}")
    _require(source["source_bytes_sha256"][:STABLE_ID_HEX_LENGTH] == assignment.stable_sample_id, "existing F1 request source bytes do not reproduce stable ID")
    _require(source["canonical_shape"] == CANONICAL_SHAPE, "existing F1 request canonical shape differs")
    _require(source["encoded_shape"] == [assignment.encode_axis_px, assignment.encode_axis_px, 3], "existing F1 request encoded shape differs")

    _require(value["data_identity"] == {
        "dataset": "imagenette160",
        "split": "train",
        "training_manifest_sha256": MANIFEST_SHA256,
        "published_archive_sha256": DATASET_ARCHIVE_SHA256,
    }, "existing F1 request data identity differs")
    _require(value["codec"] == {
        "codec_configuration_id": CODEC_CONFIGURATION_ID,
        "configuration_hash": CODEC_CONFIGURATION_HASH,
        "payload_budget_bytes": assignment.payload_budget_bytes,
        "encode_axis_px": assignment.encode_axis_px,
    }, "existing F1 request codec identity differs")
    _require(
        value["outcome_semantics"] == "typed_codec_infeasibility_omits_exact_pair_without_replacement_or_resampling;all_other_failures_hold",
        "existing F1 request outcome semantics differ",
    )
    return value


def _authenticate_object(runtime_root: Path, record: Any, *, kind: str, request: dict[str, Any]) -> None:
    _require(isinstance(record, dict), f"materialized {kind} record is not an object")
    if kind == "codestream":
        _require(set(record) == {"sha256", "bytes", "path", "backend_cache_key"}, "codestream record schema differs")
        _require(_is_sha256(record["sha256"]), "codestream record SHA-256 is invalid")
        _require(isinstance(record["bytes"], int) and 0 < record["bytes"] <= request["codec"]["payload_budget_bytes"], "codestream byte count is invalid")
        _require(isinstance(record["backend_cache_key"], str), "codestream backend cache key is invalid")
        expected_path = f"objects/codestream/{record['sha256']}.j2k"
    else:
        _require(set(record) == {"sha256", "bytes", "shape", "dtype", "path"}, "reconstruction record schema differs")
        _require(_is_sha256(record["sha256"]), "reconstruction record SHA-256 is invalid")
        _require(record["shape"] == request["source"]["canonical_shape"] and record["dtype"] == "uint8", "reconstruction shape/dtype differs")
        expected_bytes = int(np.prod(record["shape"], dtype=np.int64))
        _require(record["bytes"] == expected_bytes, "reconstruction byte count does not reconcile with shape/dtype")
        expected_path = f"objects/reconstruction/{record['sha256']}.rgb"
    _require(isinstance(record["path"], str), f"{kind} path is invalid")
    path = _safe_file(runtime_root, record["path"], expected_relative=expected_path)
    raw = path.read_bytes()
    _require(len(raw) == record["bytes"], f"{kind} object byte count differs")
    _require(hashlib.sha256(raw).hexdigest() == record["sha256"], f"{kind} object SHA-256 differs")


def authenticate_completed_result(
    runtime_root: Path,
    assignment: F1Assignment,
    *,
    expected_scientific: bool,
) -> dict[str, Any]:
    """Authenticate request, result, outcome, and referenced object bytes once."""

    request = authenticate_request(runtime_root, assignment, expected_scientific=expected_scientific)
    relative = f"results/{assignment.ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json"
    value, _raw = _canonical_json_object(runtime_root, relative, description="existing F1 result")
    expected_keys = {
        "schema_version", "artifact_role", "scientific", "request_id", "assignment_id",
        "assignment_ordinal", "stable_sample_id", "class_label", "quality_id",
        "source_bytes_sha256", "canonical_pixels_sha256", "encoded_pixels_sha256",
        "payload_budget_bytes", "encode_axis_px", "codec_configuration_id",
        "codec_configuration_hash", "replacement_assignment", "resampled", "outcome",
        "codestream", "reconstruction", "omission_semantics", "result_id",
    }
    _require(set(value) == expected_keys, "existing F1 result schema differs")
    _require(value["schema_version"] == SCHEMA_VERSION and value["artifact_role"] == RESULT_ROLE, "existing F1 result header differs")
    _require(value["scientific"] is expected_scientific, "existing F1 result scientific flag differs")
    body = dict(value)
    result_id = body.pop("result_id")
    _require(result_id == "g8fresult-" + sha256_bytes(canonical_json(body)), "existing F1 result identity differs")
    _require(value["request_id"] == request["request_id"], "existing F1 result belongs to another request")
    _require(value["assignment_id"] == assignment.assignment_id, "existing F1 result belongs to another assignment")
    reconciled = {
        "assignment_ordinal": assignment.ordinal,
        "stable_sample_id": assignment.stable_sample_id,
        "class_label": assignment.label,
        "quality_id": assignment.quality_id,
        "source_bytes_sha256": request["source"]["source_bytes_sha256"],
        "canonical_pixels_sha256": request["source"]["canonical_pixels_sha256"],
        "encoded_pixels_sha256": request["source"]["encoded_pixels_sha256"],
        "payload_budget_bytes": assignment.payload_budget_bytes,
        "encode_axis_px": assignment.encode_axis_px,
        "codec_configuration_id": CODEC_CONFIGURATION_ID,
        "codec_configuration_hash": CODEC_CONFIGURATION_HASH,
    }
    _require(all(value[key] == expected for key, expected in reconciled.items()), "existing F1 result scientific body differs")
    _require(value["replacement_assignment"] is None and value["resampled"] is False, "existing F1 result resampled or replaced its assignment")

    if value["outcome"] == "typed_image_codec_infeasibility":
        _require(value["codestream"] is None and value["reconstruction"] is None, "typed codec infeasibility carries an object")
        _require(value["omission_semantics"] == "record_omitted_assigned_pair_no_replacement_no_resampling", "typed codec infeasibility omission semantics differ")
    elif value["outcome"] == "materialized_verified_artifact":
        _require(value["omission_semantics"] is None, "materialized result carries omission semantics")
        _authenticate_object(runtime_root, value["codestream"], kind="codestream", request=request)
        _authenticate_object(runtime_root, value["reconstruction"], kind="reconstruction", request=request)
    else:
        raise G8FMaterializationHold(f"existing F1 result outcome is not permitted: {value['outcome']!r}")
    return value


def load_frozen_assignments() -> tuple[F1Assignment, ...]:
    """Reconstruct only the exact AM-88 assignment; outcomes cannot alter it."""

    sampler_raw = SAMPLER_PLAN_PATH.read_bytes()
    _require(sha256_bytes(sampler_raw) == SAMPLER_PLAN_SHA256, "AM-88 sampler-plan file SHA differs")
    sampler = json.loads(sampler_raw)
    _require(sampler.get("plan_id") == SAMPLER_PLAN_ID, "AM-88 sampler-plan ID differs")
    support, quality_ids, quality_rows = _load_am87_support()
    _require(support.get("plan_id") == AM87_PLAN_ID, "AM-87 support-plan ID differs")
    _require(sha256_bytes((REPO_ROOT / "results/baseline/g8_f/corpus_plan.json").read_bytes()) == AM87_PLAN_FILE_SHA256, "AM-87 support-plan SHA differs")
    training_ids, labels, ids_by_class, split_by_id = _training_membership()
    pairs, _cycle = derive_assignments(
        quality_ids,
        ids_by_class,
        seed=EXPECTED_SEED,
        variants_per_image=EXPECTED_VARIANTS,
    )
    _require(len(pairs) == EXPECTED_ATTEMPTS, "F1 assignment count is not AM-88")
    _require(_pair_digest(pairs) == ORDERED_PAIR_SHA256, "AM-88 ordered-pair digest differs")
    _require(_pair_digest(sorted(pairs)) == PAIR_SET_SHA256, "AM-88 pair-set digest differs")
    _require(len(set(pairs)) == len(pairs), "AM-88 assignment contains a duplicate pair")
    _require(set(stable_id for stable_id, _ in pairs) == set(training_ids), "AM-88 does not cover the exact train membership")
    _require(all(split_by_id[stable_id] == "train" for stable_id, _ in pairs), "validation/test entered F1 assignments")

    assignments: list[F1Assignment] = []
    for ordinal, (stable_id, quality_id) in enumerate(pairs):
        identity = quality_rows[quality_id]["identity"]
        _require(identity["codec_configuration_id"] == CODEC_CONFIGURATION_ID, "foreign codec configuration in AM-87 support")
        assignments.append(
            F1Assignment(
                ordinal=ordinal,
                stable_sample_id=stable_id,
                label=labels[stable_id],
                quality_id=quality_id,
                payload_budget_bytes=identity["payload_budget_bytes"],
                encode_axis_px=identity["encode_axis_px"],
                codec_configuration_id=identity["codec_configuration_id"],
            )
        )
    return tuple(assignments)


def _publish_immutable(path: Path, payload: bytes) -> bool:
    """Publish exact bytes once, or authenticate the existing exact object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        _require(existing == payload, f"immutable object differs: {path}")
        return False
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            created = False
        if not created:
            _require(path.read_bytes() == payload, f"concurrent immutable object differs: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return created
    finally:
        temporary.unlink(missing_ok=True)


def _sha_pixels(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


class F1Materializer:
    """Materialize one fixed assignment at a time with exact-prefix resume."""

    def __init__(self, runtime_root: Path, backend: CodecBackend, *, scientific: bool) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        self.backend = backend
        self.scientific = scientific
        _require(getattr(backend, "configuration_hash", None) == CODEC_CONFIGURATION_HASH, "wrong JPEG 2000 codec configuration")

    def materialize(self, assignment: F1Assignment, source: SourceSample, *, split: str) -> dict[str, Any]:
        _require(split == "train", "F1 materializer admits only the train split")
        _require(source.dataset == "imagenette160", "F1 materializer admits only Imagenette")
        _require(source.stable_sample_id == assignment.stable_sample_id, "source stable ID differs from assignment")
        _require(source.label == assignment.label, "source class differs from assignment")
        _require(assignment.codec_configuration_id == CODEC_CONFIGURATION_ID, "assignment codec configuration differs")

        product = canonicalize_source(source.source_bytes, source.dataset)
        _require(product.stable_sample_id == assignment.stable_sample_id, "canonical source identity differs")
        canonical_pixels = codec_input(product)
        canonical_sha = _sha_pixels(canonical_pixels)
        encoded = codec_downsample(canonical_pixels, assignment.encode_axis_px)
        encoded_sha = _sha_pixels(encoded)
        request = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": REQUEST_ROLE,
            "scientific": self.scientific,
            "assignment_id": assignment.assignment_id,
            "assignment": assignment.as_dict(),
            "source": {
                "dataset": source.dataset,
                "split": split,
                "source_bytes_sha256": hashlib.sha256(source.source_bytes).hexdigest(),
                "canonical_pixels_sha256": canonical_sha,
                "canonical_shape": list(canonical_pixels.shape),
                "encoded_pixels_sha256": encoded_sha,
                "encoded_shape": list(encoded.shape),
            },
            "data_identity": {
                "dataset": "imagenette160",
                "split": "train",
                "training_manifest_sha256": MANIFEST_SHA256,
                "published_archive_sha256": DATASET_ARCHIVE_SHA256,
            },
            "codec": {
                "codec_configuration_id": CODEC_CONFIGURATION_ID,
                "configuration_hash": CODEC_CONFIGURATION_HASH,
                "payload_budget_bytes": assignment.payload_budget_bytes,
                "encode_axis_px": assignment.encode_axis_px,
            },
            "outcome_semantics": "typed_codec_infeasibility_omits_exact_pair_without_replacement_or_resampling;all_other_failures_hold",
        }
        request["request_id"] = "g8frequest-" + sha256_bytes(canonical_json(request))
        request_path = self.runtime_root / "requests" / f"{assignment.ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json"
        _publish_immutable(request_path, rendered_json(request))
        authenticate_request(self.runtime_root, assignment, expected_scientific=self.scientific)

        result_path = self.runtime_root / "results" / f"{assignment.ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json"
        if result_path.exists() or result_path.is_symlink():
            return self._load_existing_result(assignment)

        try:
            codec_result = self.backend.encode_to_budget(
                encoded,
                canonical_pixels_sha256=canonical_sha,
                budget_bytes=assignment.payload_budget_bytes,
                encode_axis_px=assignment.encode_axis_px,
            )
        except Exception as exc:
            raise G8FMaterializationHold(f"unexpected codec/runtime failure: {exc}") from exc

        feasible = getattr(codec_result, "feasible", None)
        _require(isinstance(feasible, bool), "codec result has no typed feasible flag")
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": RESULT_ROLE,
            "scientific": self.scientific,
            "request_id": request["request_id"],
            "assignment_id": assignment.assignment_id,
            "assignment_ordinal": assignment.ordinal,
            "stable_sample_id": assignment.stable_sample_id,
            "class_label": assignment.label,
            "quality_id": assignment.quality_id,
            "source_bytes_sha256": request["source"]["source_bytes_sha256"],
            "canonical_pixels_sha256": canonical_sha,
            "encoded_pixels_sha256": encoded_sha,
            "payload_budget_bytes": assignment.payload_budget_bytes,
            "encode_axis_px": assignment.encode_axis_px,
            "codec_configuration_id": CODEC_CONFIGURATION_ID,
            "codec_configuration_hash": CODEC_CONFIGURATION_HASH,
            "replacement_assignment": None,
            "resampled": False,
        }
        if not feasible:
            _require(getattr(codec_result, "codestream", None) is None, "typed codec infeasibility returned bytes")
            _require(getattr(codec_result, "decoded_image", None) is None, "typed codec infeasibility returned reconstruction")
            result.update({
                "outcome": "typed_image_codec_infeasibility",
                "codestream": None,
                "reconstruction": None,
                "omission_semantics": "record_omitted_assigned_pair_no_replacement_no_resampling",
            })
        else:
            codestream = getattr(codec_result, "codestream", None)
            decoded = getattr(codec_result, "decoded_image", None)
            _require(isinstance(codestream, bytes) and codestream, "feasible codec result has no codestream")
            _require(isinstance(decoded, np.ndarray) and decoded.dtype == np.uint8 and decoded.ndim == 3 and decoded.shape[2] == 3, "feasible codec result has no verified RGB reconstruction")
            _require(getattr(codec_result, "decode_success", None) is True, "feasible codec result was not decoded")
            _require(getattr(codec_result, "codec_configuration_hash", None) == CODEC_CONFIGURATION_HASH, "codec result configuration differs")
            stream_sha = hashlib.sha256(codestream).hexdigest()
            _require(getattr(codec_result, "codestream_sha256", None) == stream_sha, "codec codestream identity differs")
            _require(getattr(codec_result, "emitted_byte_count", None) == len(codestream), "codec emitted length differs")
            _require(len(codestream) <= assignment.payload_budget_bytes, "codec exceeded assigned payload budget")
            reconstructed = decoded if decoded.shape == canonical_pixels.shape else codec_upsample(decoded, tuple(canonical_pixels.shape[:2]))
            reconstruction_sha = _sha_pixels(reconstructed)
            codestream_rel = f"objects/codestream/{stream_sha}.j2k"
            reconstruction_rel = f"objects/reconstruction/{reconstruction_sha}.rgb"
            _publish_immutable(self.runtime_root / codestream_rel, codestream)
            _publish_immutable(self.runtime_root / reconstruction_rel, reconstructed.tobytes())
            result.update({
                "outcome": "materialized_verified_artifact",
                "codestream": {
                    "sha256": stream_sha,
                    "bytes": len(codestream),
                    "path": codestream_rel,
                    "backend_cache_key": str(getattr(codec_result, "cache_key", "")),
                },
                "reconstruction": {
                    "sha256": reconstruction_sha,
                    "bytes": reconstructed.nbytes,
                    "shape": list(reconstructed.shape),
                    "dtype": str(reconstructed.dtype),
                    "path": reconstruction_rel,
                },
                "omission_semantics": None,
            })
        result["result_id"] = "g8fresult-" + sha256_bytes(canonical_json(result))
        _publish_immutable(result_path, rendered_json(result))
        return authenticate_completed_result(
            self.runtime_root,
            assignment,
            expected_scientific=self.scientific,
        )

    def _load_existing_result(self, assignment: F1Assignment) -> dict[str, Any]:
        return authenticate_completed_result(
            self.runtime_root,
            assignment,
            expected_scientific=self.scientific,
        )


def _ordinal_namespace(root: Path, description: str) -> list[Path]:
    if not root.exists():
        _require(not root.is_symlink(), f"{description} root may not be a symlink")
        return []
    _require(root.is_dir() and not root.is_symlink(), f"{description} root is not a regular directory")
    paths = sorted(root.iterdir(), key=lambda path: path.name)
    _require(
        all(path.suffix == ".json" and len(path.stem) == ORDINAL_FILENAME_WIDTH and path.stem.isdigit() and path.is_file() and not path.is_symlink() for path in paths),
        f"{description} contains a foreign path",
    )
    return paths


def validate_exact_result_prefix(
    runtime_root: Path,
    assignments: Sequence[F1Assignment],
    *,
    expected_scientific: bool = True,
) -> int:
    """Authenticate an exact completed prefix once, plus one legal orphan request."""

    runtime_root = Path(runtime_root)
    _require(not runtime_root.is_symlink(), "F1 runtime root may not be a symlink")
    result_paths = _ordinal_namespace(runtime_root / "results", "F1 results")
    request_paths = _ordinal_namespace(runtime_root / "requests", "F1 requests")
    _require(len(result_paths) <= len(assignments), "F1 result count exceeds supplied frozen assignment")
    result_names = [f"{ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json" for ordinal in range(len(result_paths))]
    _require([path.name for path in result_paths] == result_names, "F1 results are not an exact ordinal prefix")
    _require(len(request_paths) in {len(result_paths), len(result_paths) + 1}, "F1 requests are not the completed prefix plus at most one orphan")
    _require(len(request_paths) <= len(assignments), "F1 request count exceeds supplied frozen assignment")
    request_names = [f"{ordinal:0{ORDINAL_FILENAME_WIDTH}d}.json" for ordinal in range(len(request_paths))]
    _require([path.name for path in request_paths] == request_names, "F1 requests are not an exact ordinal prefix")

    for ordinal in range(len(result_paths)):
        authenticate_completed_result(
            runtime_root,
            assignments[ordinal],
            expected_scientific=expected_scientific,
        )
    if len(request_paths) == len(result_paths) + 1:
        authenticate_request(
            runtime_root,
            assignments[len(result_paths)],
            expected_scientific=expected_scientific,
        )
    return len(result_paths)
