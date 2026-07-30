"""Budget-bounded raw JPEG 2000 source coding through Glymur/OpenJPEG."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import glymur
import numpy as np

from config.params import get
from env import assert_j2k_runtime, loaded_openjpeg_version

_RAW_CODESTREAM_SOC = b"\xff\x4f"
_JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
_CACHE_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2


class J2KCodecError(ValueError):
    """Fail-closed codec, search, decode, or cache error."""


@dataclass(frozen=True)
class SearchPoint:
    iteration: int
    compression_ratio: float
    emitted_bytes: int
    within_budget: bool


@dataclass(frozen=True)
class J2KResult:
    feasible: bool
    codestream: bytes | None
    emitted_byte_count: int | None
    requested_budget_bytes: int
    compression_ratio_argument: float | None
    encode_axis_px: int
    codec_configuration_hash: str
    search_iterations: int
    search_trace: tuple[SearchPoint, ...]
    decoded_image: np.ndarray | None
    decode_success: bool
    codestream_sha256: str | None
    cache_key: str
    cache_hit: bool


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("JPEG 2000 input must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("JPEG 2000 input must be uint8")
    if image.ndim != 3 or image.shape[2] != len("RGB"):
        raise ValueError("JPEG 2000 input must be RGB HWC")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("JPEG 2000 input dimensions must be positive")
    return np.ascontiguousarray(image)


def _codec_snapshot() -> dict[str, Any]:
    baseline = get("baseline")
    preprocessing = get("preprocessing")
    if not isinstance(baseline, dict) or not isinstance(preprocessing, dict):
        raise J2KCodecError("resolved parameter snapshot lacks codec configuration")
    j2k_keys = (
        "source_codec",
        "j2k_rate_control",
        "j2k_rate_control_method",
        "j2k_emitted_size_authoritative",
        "j2k_container",
        "j2k_impl",
        "j2k_impl_version",
        "j2k_binding",
        "j2k_wavelet",
        "j2k_progression_order",
        "j2k_resolutions",
        "j2k_code_block_size",
        "j2k_tile_size",
        "j2k_search_method",
        "j2k_search_bounds",
        "j2k_search_tolerance_bytes",
        "j2k_search_max_iters",
        "j2k_cache_key",
        "j2k_nonmonotone_policy",
        "downsample_axis_px",
        "downsample_axis_never_upscales",
    )
    preprocessing_keys = (
        "codec_input",
        "codec_downsample_interpolation",
        "codec_upsample_interpolation",
        "codec_resize_preserves_aspect",
    )
    return {
        "baseline": {key: baseline[key] for key in j2k_keys},
        "preprocessing": {key: preprocessing[key] for key in preprocessing_keys},
        "environment": {
            "glymur": get("environment.glymur"),
            "openjpeg": get("environment.openjpeg"),
        },
    }


def _validate_codec_snapshot(snapshot: dict[str, Any]) -> None:
    baseline = snapshot["baseline"]
    expected = {
        "source_codec": "jpeg2000",
        "j2k_rate_control": "largest_codestream_within_budget",
        "j2k_rate_control_method": "cached_search_over_compression_ratio",
        "j2k_emitted_size_authoritative": True,
        "j2k_container": "raw_codestream",
        "j2k_impl": "openjpeg",
        "j2k_impl_version": "2.5.4",
        "j2k_binding": "glymur",
        "j2k_wavelet": "irreversible_9_7",
        "j2k_progression_order": "RPCL",
        "j2k_tile_size": "whole_image",
        "j2k_search_method": "bisection_on_compression_ratio",
        "j2k_nonmonotone_policy": "keep_largest_codestream_at_or_below_budget",
    }
    for key, value in expected.items():
        if baseline[key] != value:
            raise J2KCodecError(f"unsupported JPEG 2000 setting {key}={baseline[key]!r}")
    if snapshot["environment"] != {"glymur": glymur.__version__, "openjpeg": "2.5.4"}:
        raise J2KCodecError("loaded JPEG 2000 binding versions disagree")


class J2KCodec:
    """Search, cache, and decode exact raw codestream bytes under a byte budget."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root.resolve()
        self.snapshot = _codec_snapshot()
        _validate_codec_snapshot(self.snapshot)
        self.configuration_hash = _sha256_bytes(_canonical_json(self.snapshot))
        bounds = self.snapshot["baseline"]["j2k_search_bounds"]
        self.search_lower = float(bounds[0])
        self.search_upper = float(bounds[1])
        self.byte_tolerance = int(
            self.snapshot["baseline"]["j2k_search_tolerance_bytes"]
        )
        self.max_iterations = int(
            self.snapshot["baseline"]["j2k_search_max_iters"]
        )
        if (
            not math.isfinite(self.search_lower)
            or not math.isfinite(self.search_upper)
            or self.search_lower <= 0
            or self.search_lower >= self.search_upper
        ):
            raise J2KCodecError("invalid JPEG 2000 search bounds")
        if self.byte_tolerance < 0 or self.max_iterations < 2:
            raise J2KCodecError("invalid JPEG 2000 search tolerance or iteration limit")

    def _cache_identity(
        self,
        *,
        canonical_pixels_sha256: str,
        budget_bytes: int,
        encode_axis_px: int,
    ) -> tuple[str, dict[str, Any]]:
        identity = {
            "canonical_pixels_sha256": canonical_pixels_sha256,
            "budget_bytes": budget_bytes,
            "encode_axis_px": encode_axis_px,
            "codec_config_hash": self.configuration_hash,
            "openjpeg_version": loaded_openjpeg_version(required=True),
        }
        return _sha256_bytes(_canonical_json(identity)), identity

    def encode_to_budget(
        self,
        image: np.ndarray,
        *,
        canonical_pixels_sha256: str,
        budget_bytes: int,
        encode_axis_px: int,
    ) -> J2KResult:
        """Return the largest observed raw codestream not exceeding ``budget_bytes``."""

        assert_j2k_runtime()
        source = _validated_image(image)
        if not isinstance(budget_bytes, int) or isinstance(budget_bytes, bool):
            raise TypeError("budget_bytes must be an integer")
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        if (
            not isinstance(encode_axis_px, int)
            or isinstance(encode_axis_px, bool)
            or encode_axis_px != min(source.shape[:2])
        ):
            raise ValueError("encode_axis_px must equal the encoded image shorter side")
        if (
            not isinstance(canonical_pixels_sha256, str)
            or len(canonical_pixels_sha256) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in canonical_pixels_sha256)
        ):
            raise ValueError("canonical_pixels_sha256 must be a lowercase SHA-256")

        cache_key, cache_identity = self._cache_identity(
            canonical_pixels_sha256=canonical_pixels_sha256,
            budget_bytes=budget_bytes,
            encode_axis_px=encode_axis_px,
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_root / f"{cache_key}.j2kcache"
        if cache_path.exists():
            return self._load_cache(
                cache_path,
                cache_key=cache_key,
                cache_identity=cache_identity,
                expected_shape=source.shape,
            )

        trace: list[SearchPoint] = []

        def observe(ratio: float) -> bytes:
            codestream = self._encode_once(source, ratio)
            trace.append(
                SearchPoint(
                    iteration=len(trace),
                    compression_ratio=ratio,
                    emitted_bytes=len(codestream),
                    within_budget=len(codestream) <= budget_bytes,
                )
            )
            return codestream

        observations: list[tuple[float, bytes]] = []
        lower = self.search_lower
        upper = self.search_upper
        observations.append((lower, observe(lower)))
        if len(observations[-1][1]) > budget_bytes:
            observations.append((upper, observe(upper)))
            if len(observations[-1][1]) <= budget_bytes:
                while len(trace) < self.max_iterations:
                    midpoint = (lower + upper) / 2
                    codestream = observe(midpoint)
                    observations.append((midpoint, codestream))
                    if len(codestream) <= budget_bytes:
                        upper = midpoint
                        if budget_bytes - len(codestream) <= self.byte_tolerance:
                            break
                    else:
                        lower = midpoint

        feasible = [
            (index, ratio, codestream)
            for index, (ratio, codestream) in enumerate(observations)
            if len(codestream) <= budget_bytes
        ]
        if not feasible:
            result = J2KResult(
                feasible=False,
                codestream=None,
                emitted_byte_count=None,
                requested_budget_bytes=budget_bytes,
                compression_ratio_argument=None,
                encode_axis_px=encode_axis_px,
                codec_configuration_hash=self.configuration_hash,
                search_iterations=len(trace),
                search_trace=tuple(trace),
                decoded_image=None,
                decode_success=False,
                codestream_sha256=None,
                cache_key=cache_key,
                cache_hit=False,
            )
            self._write_cache(cache_path, cache_identity, result)
            return result

        _, selected_ratio, selected = max(
            feasible,
            key=lambda item: (len(item[2]), -item[0]),
        )
        if len(selected) > budget_bytes:
            raise J2KCodecError("selected JPEG 2000 codestream exceeds budget")
        decoded = self._decode_codestream(selected)
        if decoded.shape != source.shape:
            raise J2KCodecError(
                f"decoded JPEG 2000 shape {decoded.shape} differs from {source.shape}"
            )
        result = J2KResult(
            feasible=True,
            codestream=selected,
            emitted_byte_count=len(selected),
            requested_budget_bytes=budget_bytes,
            compression_ratio_argument=selected_ratio,
            encode_axis_px=encode_axis_px,
            codec_configuration_hash=self.configuration_hash,
            search_iterations=len(trace),
            search_trace=tuple(trace),
            decoded_image=decoded,
            decode_success=True,
            codestream_sha256=_sha256_bytes(selected),
            cache_key=cache_key,
            cache_hit=False,
        )
        self._write_cache(cache_path, cache_identity, result)
        return result

    def decode_codestream(self, codestream: bytes) -> np.ndarray:
        """Decode raw codestream bytes that arrived over the channel.

        The receiver must decode what it actually received, not the encoder's
        cached image, so this exposes the same validated decode path publicly.
        """

        assert_j2k_runtime()
        return self._decode_codestream(codestream)

    def _encode_once(self, image: np.ndarray, compression_ratio: float) -> bytes:
        with tempfile.TemporaryDirectory(prefix="j2k-encode-") as temporary:
            path = Path(temporary) / "image.j2k"
            height, width = image.shape[:2]
            try:
                glymur.Jp2k(
                    path,
                    data=image,
                    cratios=(compression_ratio,),
                    irreversible=True,
                    prog=str(self.snapshot["baseline"]["j2k_progression_order"]),
                    numres=int(self.snapshot["baseline"]["j2k_resolutions"]),
                    cbsize=tuple(self.snapshot["baseline"]["j2k_code_block_size"]),
                    tilesize=(height, width),
                )
            except Exception as exc:
                raise J2KCodecError(
                    f"JPEG 2000 encoder failed at {height}x{width} "
                    f"cratio={compression_ratio}: {exc}"
                ) from exc
            codestream = path.read_bytes()
        self._validate_raw_codestream(codestream)
        return codestream

    @staticmethod
    def _validate_raw_codestream(codestream: bytes) -> None:
        if codestream.startswith(_JP2_SIGNATURE):
            raise J2KCodecError("JPEG 2000 output is a JP2 container, not a raw codestream")
        if not codestream.startswith(_RAW_CODESTREAM_SOC):
            raise J2KCodecError("JPEG 2000 output lacks the raw codestream SOC marker")

    def _decode_codestream(self, codestream: bytes) -> np.ndarray:
        self._validate_raw_codestream(codestream)
        with tempfile.TemporaryDirectory(prefix="j2k-decode-") as temporary:
            path = Path(temporary) / "image.j2k"
            path.write_bytes(codestream)
            try:
                decoded = glymur.Jp2k(path)[:]
            except Exception as exc:
                raise J2KCodecError(f"JPEG 2000 decoder failed: {exc}") from exc
        return _validated_image(np.asarray(decoded))

    def _write_cache(
        self,
        cache_path: Path,
        identity: dict[str, Any],
        result: J2KResult,
    ) -> None:
        metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "identity": identity,
            "feasible": result.feasible,
            "emitted_bytes": result.emitted_byte_count,
            "compression_ratio_argument": result.compression_ratio_argument,
            "search_iterations": result.search_iterations,
            "search_trace": [asdict(point) for point in result.search_trace],
            "codestream_sha256": result.codestream_sha256,
            "decode_success": result.decode_success,
            "decoded_shape": (
                list(result.decoded_image.shape)
                if result.decoded_image is not None
                else None
            ),
            "decoded_dtype": (
                str(result.decoded_image.dtype)
                if result.decoded_image is not None
                else None
            ),
            "decoded_image_sha256": (
                _sha256_bytes(result.decoded_image.tobytes())
                if result.decoded_image is not None
                else None
            ),
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("metadata.json", _canonical_json(metadata) + b"\n")
                if result.codestream is not None:
                    archive.writestr("codestream.j2k", result.codestream)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_cache(
        self,
        cache_path: Path,
        *,
        cache_key: str,
        cache_identity: dict[str, Any],
        expected_shape: tuple[int, ...],
    ) -> J2KResult:
        try:
            with zipfile.ZipFile(cache_path, "r") as archive:
                names = set(archive.namelist())
                if "metadata.json" not in names:
                    raise J2KCodecError("cache metadata is absent")
                metadata = json.loads(archive.read("metadata.json"))
                expected_names = (
                    {"metadata.json", "codestream.j2k"}
                    if metadata.get("feasible")
                    else {"metadata.json"}
                )
                if names != expected_names:
                    raise J2KCodecError("cache entries are stale or partial")
                codestream = (
                    archive.read("codestream.j2k")
                    if "codestream.j2k" in names
                    else None
                )
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise J2KCodecError(f"invalid JPEG 2000 cache entry: {exc}") from None
        required = {
            "schema_version",
            "identity",
            "feasible",
            "emitted_bytes",
            "compression_ratio_argument",
            "search_iterations",
            "search_trace",
            "codestream_sha256",
            "decode_success",
            "decoded_shape",
            "decoded_dtype",
            "decoded_image_sha256",
        }
        if set(metadata) != required:
            raise J2KCodecError("cache metadata schema differs")
        if (
            metadata["schema_version"] != _CACHE_SCHEMA_VERSION
            or metadata["identity"] != cache_identity
        ):
            raise J2KCodecError("cache identity is stale")
        trace = tuple(SearchPoint(**point) for point in metadata["search_trace"])
        if codestream is None:
            if any(
                (
                    metadata["feasible"],
                    metadata["emitted_bytes"] is not None,
                    metadata["compression_ratio_argument"] is not None,
                    metadata["codestream_sha256"] is not None,
                    metadata["decode_success"],
                    metadata["decoded_shape"] is not None,
                    metadata["decoded_dtype"] is not None,
                    metadata["decoded_image_sha256"] is not None,
                )
            ):
                raise J2KCodecError("infeasible cache status is inconsistent")
            return J2KResult(
                feasible=False,
                codestream=None,
                emitted_byte_count=None,
                requested_budget_bytes=int(cache_identity["budget_bytes"]),
                compression_ratio_argument=None,
                encode_axis_px=int(cache_identity["encode_axis_px"]),
                codec_configuration_hash=self.configuration_hash,
                search_iterations=int(metadata["search_iterations"]),
                search_trace=trace,
                decoded_image=None,
                decode_success=False,
                codestream_sha256=None,
                cache_key=cache_key,
                cache_hit=True,
            )
        if (
            len(codestream) != metadata["emitted_bytes"]
            or len(codestream) > cache_identity["budget_bytes"]
            or _sha256_bytes(codestream) != metadata["codestream_sha256"]
        ):
            raise J2KCodecError("cached codestream identity or budget disagrees")
        decoded = self._decode_codestream(codestream)
        if (
            list(decoded.shape) != metadata["decoded_shape"]
            or tuple(decoded.shape) != expected_shape
            or str(decoded.dtype) != metadata["decoded_dtype"]
            or _sha256_bytes(decoded.tobytes()) != metadata["decoded_image_sha256"]
            or metadata["decode_success"] is not True
        ):
            raise J2KCodecError("cached decoded-image verification disagrees")
        return J2KResult(
            feasible=True,
            codestream=codestream,
            emitted_byte_count=len(codestream),
            requested_budget_bytes=int(cache_identity["budget_bytes"]),
            compression_ratio_argument=float(
                metadata["compression_ratio_argument"]
            ),
            encode_axis_px=int(cache_identity["encode_axis_px"]),
            codec_configuration_hash=self.configuration_hash,
            search_iterations=int(metadata["search_iterations"]),
            search_trace=trace,
            decoded_image=decoded,
            decode_success=True,
            codestream_sha256=str(metadata["codestream_sha256"]),
            cache_key=cache_key,
            cache_hit=True,
        )
