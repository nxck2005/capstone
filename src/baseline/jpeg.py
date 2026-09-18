"""Secondary JPEG codec (DEC-9 / BR-1) driven exactly as the frozen parameters fix it."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from config.params import get

JPEG_EOI = b"\xff\xd9"


class JpegCodecError(ValueError):
    """A JPEG encode could not follow the frozen Pillow configuration."""


@dataclass(frozen=True)
class JpegResult:
    feasible: bool
    quality: int
    emitted_byte_count: int | None
    codestream: bytes | None
    cache_key: str
    search_points: tuple[int, ...]


def _subsampling_value() -> int:
    text = str(get("baseline.jpeg_chroma_subsampling"))
    mapping = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
    if text not in mapping:
        raise JpegCodecError(f"unsupported JPEG chroma subsampling: {text}")
    return mapping[text]


class JpegCodec:
    """A deterministic Pillow JPEG encode/search over ``params.baseline.jpeg_quality_grid``."""

    def __init__(self) -> None:
        if str(get("baseline.jpeg_impl")) != "pillow":
            raise JpegCodecError("params.baseline.jpeg_impl is not pillow")
        self.subsampling = _subsampling_value()
        self.optimize = bool(get("baseline.jpeg_optimise_huffman"))
        self.quality_grid = tuple(int(value) for value in get("baseline.jpeg_quality_grid"))

    def encode_at_quality(self, image: np.ndarray, *, quality: int) -> bytes:
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:  # literal-ok: RGB image rank/channels
            raise JpegCodecError("JPEG input must be an RGB uint8 image")
        if int(quality) not in self.quality_grid:
            raise JpegCodecError(f"JPEG quality {quality} is not in the configured grid")
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(
            buffer,
            format="JPEG",
            quality=int(quality),
            optimize=self.optimize,
            subsampling=self.subsampling,
        )
        return buffer.getvalue()

    def encode_to_budget(
        self,
        image: np.ndarray,
        *,
        canonical_pixels_sha256: str,
        budget_bytes: int,
        encode_axis_px: int,
    ) -> JpegResult:
        """The highest-configured quality whose emitted bytes fit the budget."""

        if budget_bytes <= 0:
            raise JpegCodecError("JPEG budget must be positive")
        search: list[int] = []
        best: tuple[int, bytes] | None = None
        for quality in sorted(self.quality_grid, reverse=True):
            search.append(quality)
            codestream = self.encode_at_quality(image, quality=quality)
            if len(codestream) <= budget_bytes:
                best = (quality, codestream)
                break
        cache_key = hashlib.sha256(
            (
                canonical_pixels_sha256
                + "|"
                + str(int(budget_bytes))
                + "|"
                + str(int(encode_axis_px))
                + "|jpeg|"
                + str(self.subsampling)
                + "|"
                + str(self.optimize)
            ).encode("ascii")
        ).hexdigest()
        if best is None:
            return JpegResult(False, -1, None, None, cache_key, tuple(search))
        return JpegResult(True, best[0], len(best[1]), best[1], cache_key, tuple(search))


def decode_codestream(payload: bytes) -> np.ndarray:
    """Decode a JPEG codestream recovered from the transport payload."""

    end = payload.rfind(JPEG_EOI)
    if end < 0:
        raise JpegCodecError("recovered payload contains no JPEG EOI marker")
    codestream = payload[: end + len(JPEG_EOI)]
    with Image.open(io.BytesIO(codestream)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


__all__ = ["JPEG_EOI", "JpegCodec", "JpegCodecError", "JpegResult", "decode_codestream"]
