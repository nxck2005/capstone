"""Canonical pixel, augmentation, codec-resampling, and metric contract (SR-19).

All resize operations in this module are PIL-backed torchvision functional
operations.  They never resize tensors: PIL and tensor backends produce
different pixels for the same nominal interpolation.  Interpolation and
antialiasing are passed explicitly from ``params.preprocessing``.

Canonicalisation and augmentation are deliberately separate.  A
``canonicalize_source`` is the sole supported factory.  It passes the original
source bytes to a private data-layer decoder registry, then returns the
read-only ``CanonicalProduct`` protocol backed by an unexported implementation.
The encoder tensor is derived only from its canonical bytes by conversion to
float32 CHW and division by 255; the codec receives a byte-identical uint8 copy.
Training augmentation starts from that same canonical image and obtains every
random draw from the keyed ``augmentation`` Philox stream.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode

from artifacts.rng import keyed_generator
from config.params import get

_RGB_CHANNELS = len("RGB")
_UINT8_MAX = np.iinfo(np.uint8).max
_SAMPLE_ID_RULE = re.compile(
    r"sha256_of_original_per_sample_source_bytes_truncated_(?P<width>\d+)_hex"
)


class CanonicalProduct(Protocol):
    """Read-only view of a source-bound canonical product."""

    @property
    def stable_sample_id(self) -> str: ...

    @property
    def canonical_image(self) -> np.ndarray: ...


@dataclass(frozen=True)
class _CanonicalProduct:
    """Private implementation; products are created only from source bytes."""

    stable_sample_id: str
    canonical_image: np.ndarray

    def __post_init__(self) -> None:
        canonical = _validated_uint8_rgb(self.canonical_image).copy()
        canonical.setflags(write=False)
        object.__setattr__(self, "canonical_image", canonical)


@dataclass(frozen=True)
class ReconstructionMetrics:
    """The two SR-19 reconstruction metrics after contract-mandated clipping."""

    psnr_db: float
    ssim: float


_SourceDecoder = Callable[[bytes], Image.Image | np.ndarray]
_SOURCE_DECODERS: dict[str, _SourceDecoder] = {}


def stable_sample_id(source_bytes: bytes) -> str:
    """Hash the original per-sample payload bytes, before decode or preprocessing."""

    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    if not source_bytes:
        raise ValueError("source_bytes must not be empty")

    rule = get("datasets.stable_sample_id_rule")
    match = _SAMPLE_ID_RULE.fullmatch(rule)
    if match is None:
        raise NotImplementedError(
            f"unsupported params.datasets.stable_sample_id_rule: {rule}"
        )
    width = int(match.group("width"))
    return hashlib.sha256(source_bytes).hexdigest()[:width]


def canonicalize_source(
    source_bytes: bytes,
    dataset: str,
) -> CanonicalProduct:
    """Decode source bytes once and create the canonical product.

    Dataset-loader implementations register their byte decoders privately in
    this data-layer module. Real registrations land with the loader batch;
    contract tests inject a private fixture decoder.
    """

    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    if not source_bytes:
        raise ValueError("source_bytes must not be empty")
    try:
        decoder = _SOURCE_DECODERS[dataset]
    except KeyError:
        raise ValueError(f"no source decoder registered for dataset {dataset!r}") from None
    decoded = decoder(source_bytes)
    return _canonicalize_decoded(
        decoded,
        dataset=dataset,
        source_bytes=source_bytes,
    )


def _canonicalize_decoded(
    image: Image.Image | np.ndarray,
    *,
    dataset: str,
    source_bytes: bytes,
) -> _CanonicalProduct:
    """Private decoded-pixel boundary used only by ``canonicalize_source``."""

    _validate_pipeline_contract()
    target_hw = _dataset_hw(dataset)
    pil_image = _as_pil_rgb(image)
    interpolation = _interpolation("resize_interpolation")
    antialias = _antialias()

    if get("preprocessing.canonical_image") != (
        "resize_shorter_side_then_crop_to_dataset_image_size"
    ):
        raise NotImplementedError(
            "unsupported params.preprocessing.canonical_image: "
            f"{get('preprocessing.canonical_image')}"
        )
    if get("preprocessing.eval_crop") != "center_crop":
        raise NotImplementedError(
            f"unsupported params.preprocessing.eval_crop: "
            f"{get('preprocessing.eval_crop')}"
        )

    shorter_side = min(target_hw)
    resized = vision_f.resize(
        pil_image,
        [shorter_side],
        interpolation=interpolation,
        antialias=antialias,
    )
    cropped = vision_f.center_crop(resized, list(target_hw))
    canonical = _pil_to_uint8_rgb(cropped)
    if canonical.shape[:2] != target_hw:
        raise RuntimeError(
            f"canonical image has shape {canonical.shape}, expected "
            f"{(*target_hw, _RGB_CHANNELS)}"
        )
    return _CanonicalProduct(
        stable_sample_id=stable_sample_id(source_bytes),
        canonical_image=canonical,
    )


def codec_input(product: CanonicalProduct) -> np.ndarray:
    """Return a writable, byte-identical copy of the shared canonical pixels."""

    if get("preprocessing.codec_input") != "canonical_8bit_pixels":
        raise NotImplementedError(
            f"unsupported params.preprocessing.codec_input: "
            f"{get('preprocessing.codec_input')}"
        )
    return _require_product(product).canonical_image.copy()


def _canonical_uint8_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB HWC to float32 RGB CHW by exactly ``image / 255``."""

    _validate_pipeline_contract()
    canonical = _validated_uint8_rgb(image)
    return (
        torch.from_numpy(canonical.copy())
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div(_UINT8_MAX)
    )


def encoder_input(product: CanonicalProduct) -> torch.Tensor:
    """Return the unnormalised model input derived only from canonical pixels."""

    return _canonical_uint8_to_tensor(_require_product(product).canonical_image)


def evaluation_input(product: CanonicalProduct) -> torch.Tensor:
    """Return the deterministic evaluation tensor; eval augmentation is forbidden."""

    if get("preprocessing.eval_augmentation_permitted"):
        raise NotImplementedError("evaluation augmentation is not supported")
    return encoder_input(product)


def training_input(
    product: CanonicalProduct,
    rng_identity: Mapping[str, Any],
) -> torch.Tensor:
    """Apply the configured training augmentation with the keyed Philox stream."""

    if not rng_identity:
        raise ValueError("augmentation RNG identity must not be empty")
    _validate_pipeline_contract()
    actual_product = _require_product(product)
    identity = dict(rng_identity)
    supplied_sample_id = identity.get("stable_sample_id")
    if supplied_sample_id != actual_product.stable_sample_id:
        raise ValueError(
            "augmentation RNG stable_sample_id does not match the canonical product"
        )
    rng = keyed_generator("augmentation", identity)

    if get("preprocessing.train_crop") != "random_resized_crop":
        raise NotImplementedError(
            f"unsupported params.preprocessing.train_crop: "
            f"{get('preprocessing.train_crop')}"
        )
    top, left, height, width = _random_resized_crop_box(
        actual_product.canonical_image.shape[:2], rng
    )
    output_hw = actual_product.canonical_image.shape[:2]
    augmented = vision_f.resized_crop(
        Image.fromarray(actual_product.canonical_image),
        top,
        left,
        height,
        width,
        list(output_hw),
        interpolation=_interpolation("resize_interpolation"),
        antialias=_antialias(),
    )
    if rng.random() < float(get("preprocessing.train_hflip_p")):
        augmented = vision_f.hflip(augmented)
    return _canonical_uint8_to_tensor(_pil_to_uint8_rgb(augmented))


def channel_normalisation_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Expose the normalisation constants for the later model-owned layer."""

    if (
        get("preprocessing.channel_normalisation")
        != "inside_model_never_in_the_pipeline"
    ):
        raise NotImplementedError(
            "unsupported params.preprocessing.channel_normalisation: "
            f"{get('preprocessing.channel_normalisation')}"
        )
    mean = tuple(float(value) for value in get("preprocessing.channel_mean"))
    std = tuple(float(value) for value in get("preprocessing.channel_std"))
    if len(mean) != _RGB_CHANNELS or len(std) != _RGB_CHANNELS:
        raise ValueError("channel_mean and channel_std must each have three values")
    return mean, std


def codec_downsample(image: np.ndarray, shorter_side: int) -> np.ndarray:
    """Downsample uint8 RGB pixels to a shorter side while preserving aspect."""

    source = _validated_uint8_rgb(image)
    if not isinstance(shorter_side, int) or isinstance(shorter_side, bool):
        raise TypeError("shorter_side must be an integer")
    if shorter_side <= 0:
        raise ValueError("shorter_side must be positive")
    if shorter_side > min(source.shape[:2]):
        raise ValueError("codec_downsample must not upscale")
    _require_aspect_preservation()
    resized = vision_f.resize(
        Image.fromarray(source),
        [shorter_side],
        interpolation=_interpolation("codec_downsample_interpolation"),
        antialias=_antialias(),
    )
    return _pil_to_uint8_rgb(resized)


def codec_upsample(
    image: np.ndarray,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Upsample uint8 RGB pixels to an aspect-compatible receiver size."""

    source = _validated_uint8_rgb(image)
    target_hw = _validated_hw(output_hw)
    if target_hw[0] < source.shape[0] or target_hw[1] < source.shape[1]:
        raise ValueError("codec_upsample must not downsample")
    _require_aspect_preservation()
    if source.shape[0] * target_hw[1] != source.shape[1] * target_hw[0]:
        raise ValueError("codec resize must preserve the source aspect ratio")
    resized = vision_f.resize(
        Image.fromarray(source),
        list(target_hw),
        interpolation=_interpolation("codec_upsample_interpolation"),
        antialias=_antialias(),
    )
    return _pil_to_uint8_rgb(resized)


def clip_reconstruction_for_metrics(
    reconstruction: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convert to float HWC and apply the configured pre-metric clipping."""

    array = _metric_array(reconstruction)
    if get("preprocessing.reconstruction_clipped_before_metrics"):
        data_range = float(get("preprocessing.psnr_data_range"))
        return np.clip(array, 0.0, data_range)
    return array.copy()


def reconstruction_psnr(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
) -> float:
    """Compute PSNR after applying SR-19's reconstruction clipping rule."""

    reference_array, clipped = _metric_pair(reference, reconstruction)
    mse = float(np.mean(np.square(reference_array - clipped), dtype=np.float64))
    if mse == 0:
        return math.inf
    data_range = float(get("preprocessing.psnr_data_range"))
    return (
        10  # literal-ok: PSNR definition uses ten times the base-10 logarithm
        * math.log10((data_range * data_range) / mse)
    )


def reconstruction_ssim(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
) -> float:
    """Compute the configured scikit-image SSIM after reconstruction clipping."""

    if get("preprocessing.ssim_impl") != "skimage_structural_similarity":
        raise NotImplementedError(
            f"unsupported params.preprocessing.ssim_impl: "
            f"{get('preprocessing.ssim_impl')}"
        )
    reference_array, clipped = _metric_pair(reference, reconstruction)
    return float(
        structural_similarity(
            reference_array,
            clipped,
            data_range=float(get("preprocessing.psnr_data_range")),
            channel_axis=int(get("preprocessing.ssim_channel_axis")),
            gaussian_weights=bool(get("preprocessing.ssim_gaussian_weights")),
            sigma=float(get("preprocessing.ssim_sigma")),
            K1=float(get("preprocessing.ssim_k1")),
            K2=float(get("preprocessing.ssim_k2")),
            use_sample_covariance=bool(
                get("preprocessing.ssim_use_sample_covariance")
            ),
        )
    )


def reconstruction_metrics(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
) -> ReconstructionMetrics:
    """Return PSNR and SSIM under the one shared reconstruction-score contract."""

    return ReconstructionMetrics(
        psnr_db=reconstruction_psnr(reference, reconstruction),
        ssim=reconstruction_ssim(reference, reconstruction),
    )


def _validate_pipeline_contract() -> None:
    if not get("preprocessing.canonical_and_augmentation_are_separate_stages"):
        raise NotImplementedError("canonicalisation and augmentation must be separate")
    if get("preprocessing.colour_space").lower() != "rgb":
        raise NotImplementedError(
            f"unsupported params.preprocessing.colour_space: "
            f"{get('preprocessing.colour_space')}"
        )
    if int(get("preprocessing.bit_depth")) != np.iinfo(np.uint8).bits:
        raise NotImplementedError(
            f"unsupported params.preprocessing.bit_depth: "
            f"{get('preprocessing.bit_depth')}"
        )
    if get("preprocessing.tensor_range") != "unit_interval":
        raise NotImplementedError(
            f"unsupported params.preprocessing.tensor_range: "
            f"{get('preprocessing.tensor_range')}"
        )
    if (
        get("preprocessing.channel_normalisation")
        != "inside_model_never_in_the_pipeline"
    ):
        raise NotImplementedError(
            "channel normalisation belongs inside the later model"
        )


def _dataset_hw(dataset: str) -> tuple[int, int]:
    size = get(f"datasets.{dataset}.image_size")
    if len(size) != _RGB_CHANNELS or int(size[-1]) != _RGB_CHANNELS:
        raise ValueError(f"params.datasets.{dataset}.image_size must be HWC RGB")
    return _validated_hw((int(size[0]), int(size[1])))


def _validated_hw(output_hw: tuple[int, int]) -> tuple[int, int]:
    if len(output_hw) != 2:
        raise ValueError("output size must contain height and width")
    height, width = output_hw
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
    ):
        raise TypeError("output height and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("output height and width must be positive")
    return height, width


def _as_pil_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("source arrays must have dtype uint8")
    if array.ndim not in (2, 3):
        raise ValueError("source arrays must be HW, HWC RGB, or HWC RGBA")
    if array.ndim == 3 and array.shape[-1] not in (
        1,
        _RGB_CHANNELS,
        len("RGBA"),
    ):
        raise ValueError("source arrays must be HW, HWC RGB, or HWC RGBA")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def _pil_to_uint8_rgb(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _validated_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("canonical and codec images must have dtype uint8")
    if array.ndim != 3 or array.shape[-1] != _RGB_CHANNELS:
        raise ValueError("canonical and codec images must be RGB HWC")
    return np.ascontiguousarray(array)


def _require_product(product: CanonicalProduct) -> _CanonicalProduct:
    if not isinstance(product, _CanonicalProduct):
        raise TypeError(
            "canonical products must be obtained from canonicalize_source"
        )
    return product


def _interpolation(parameter: str) -> InterpolationMode:
    value = get(f"preprocessing.{parameter}")
    try:
        return InterpolationMode(value)
    except ValueError:
        raise NotImplementedError(
            f"unsupported params.preprocessing.{parameter}: {value}"
        ) from None


def _antialias() -> bool:
    value = get("preprocessing.antialias")
    if not isinstance(value, bool):
        raise TypeError("params.preprocessing.antialias must be boolean")
    return value


def _require_aspect_preservation() -> None:
    if not get("preprocessing.codec_resize_preserves_aspect"):
        raise NotImplementedError("non-aspect-preserving codec resize is unsupported")


def _random_resized_crop_box(
    image_hw: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Torchvision-0.28-compatible proposal and deterministic fallback geometry."""

    image_height, image_width = image_hw
    area = image_height * image_width
    scale_min, scale_max = (
        float(value) for value in get("preprocessing.train_crop_scale")
    )
    ratio_min, ratio_max = (
        float(value) for value in get("preprocessing.train_crop_ratio")
    )
    if not (0 < scale_min <= scale_max):
        raise ValueError("train_crop_scale must be positive and ordered")
    if not (0 < ratio_min <= ratio_max):
        raise ValueError("train_crop_ratio must be positive and ordered")
    log_ratio_min, log_ratio_max = math.log(ratio_min), math.log(ratio_max)

    for _ in range(
        10  # literal-ok: fixed proposal count in torchvision RandomResizedCrop
    ):
        target_area = area * rng.uniform(scale_min, scale_max)
        aspect_ratio = math.exp(rng.uniform(log_ratio_min, log_ratio_max))
        width = int(round(math.sqrt(target_area * aspect_ratio)))
        height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < width <= image_width and 0 < height <= image_height:
            top = int(rng.integers(0, image_height - height + 1))
            left = int(rng.integers(0, image_width - width + 1))
            return top, left, height, width

    input_ratio = image_width / image_height
    if input_ratio < ratio_min:
        width = image_width
        height = int(round(width / ratio_min))
    elif input_ratio > ratio_max:
        height = image_height
        width = int(round(height * ratio_max))
    else:
        width = image_width
        height = image_height
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    return top, left, height, width


def _metric_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
        if tensor.ndim != 3 or tensor.shape[0] != _RGB_CHANNELS:
            raise ValueError("metric tensors must be RGB CHW")
        array = tensor.permute(1, 2, 0).numpy()
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != _RGB_CHANNELS:
            raise ValueError("metric arrays must be RGB HWC")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric inputs must contain only finite values")
    return array


def _metric_pair(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = _metric_array(reference)
    reconstruction_array = _metric_array(reconstruction)
    if reference_array.shape != reconstruction_array.shape:
        raise ValueError("reference and reconstruction shapes must match")
    data_range = float(get("preprocessing.psnr_data_range"))
    if np.any(reference_array < 0) or np.any(reference_array > data_range):
        raise ValueError("reference pixels must lie inside the metric data range")
    clipped = clip_reconstruction_for_metrics(reconstruction_array)
    return reference_array, clipped
