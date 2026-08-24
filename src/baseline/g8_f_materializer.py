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
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

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
from config.params import REPO_ROOT
from data.adapters import SourceSample
from data.preprocessing import canonicalize_source, codec_downsample, codec_input, codec_upsample

SCHEMA_VERSION = 1
REQUEST_ROLE = "g8_f_f1_assigned_pair_request"
RESULT_ROLE = "g8_f_f1_artifact_result"
SAMPLER_PLAN_ID = "g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6"
SAMPLER_PLAN_SHA256 = "eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23"
ORDERED_PAIR_SHA256 = "c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229"
PAIR_SET_SHA256 = "255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e"
CODEC_CONFIGURATION_ID = "g8dcodec-39f14b7eaba4f727c70759eb1c5250e8e13f7d5e871c0831aa6b602aef706858"
CODEC_CONFIGURATION_HASH = "2daf597fd914f56eb9e59df7bc20a88b02816522b3b0b4fd3f2db14d7451a0fa"
MANIFEST_SHA256 = "224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889"


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
            "codec": {
                "codec_configuration_id": CODEC_CONFIGURATION_ID,
                "configuration_hash": CODEC_CONFIGURATION_HASH,
                "payload_budget_bytes": assignment.payload_budget_bytes,
                "encode_axis_px": assignment.encode_axis_px,
            },
            "outcome_semantics": "typed_codec_infeasibility_omits_exact_pair_without_replacement_or_resampling;all_other_failures_hold",
        }
        request["request_id"] = "g8frequest-" + sha256_bytes(canonical_json(request))
        request_path = self.runtime_root / "requests" / f"{assignment.ordinal:05d}.json"
        _publish_immutable(request_path, rendered_json(request))

        result_path = self.runtime_root / "results" / f"{assignment.ordinal:05d}.json"
        if result_path.exists():
            return self._load_existing_result(result_path, request)

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
        return result

    @staticmethod
    def _load_existing_result(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise G8FMaterializationHold(f"cannot authenticate existing F1 result: {exc}") from None
        _require(raw == rendered_json(value), "existing F1 result is not canonical rendered JSON")
        body = dict(value)
        result_id = body.pop("result_id", None)
        _require(result_id == "g8fresult-" + sha256_bytes(canonical_json(body)), "existing F1 result identity differs")
        _require(value.get("request_id") == request["request_id"], "existing F1 result belongs to another request")
        _require(value.get("assignment_id") == request["assignment_id"], "existing F1 result belongs to another assignment")
        _require(value.get("replacement_assignment") is None and value.get("resampled") is False, "existing F1 result resampled")
        return value


def validate_exact_result_prefix(runtime_root: Path, assignments: Sequence[F1Assignment]) -> int:
    """Reject holes, foreign ordinals, or results beyond the exact AM-88 list."""

    results_root = Path(runtime_root) / "results"
    if not results_root.exists():
        return 0
    paths = sorted(results_root.glob("*.json"))
    expected_names = [f"{ordinal:05d}.json" for ordinal in range(len(paths))]
    _require([path.name for path in paths] == expected_names, "F1 results are not an exact ordinal prefix")
    _require(len(paths) <= len(assignments), "F1 result count exceeds supplied frozen assignment")
    for ordinal, path in enumerate(paths):
        value = json.loads(path.read_bytes())
        _require(value.get("assignment_id") == assignments[ordinal].assignment_id, "F1 prefix assignment differs from AM-88")
    return len(paths)
