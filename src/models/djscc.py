"""The config-derived ``djscc_residual_v1`` W2 model skeleton."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from artifacts.rng import keyed_torch_seed
from channels.power import (
    PeakPowerConstraint,
    normalize_unit_average_power,
    symbol_papr_db,
)
from channels.registry import build_channel
from config.run_config import RunConfig
from models.reference_classifier import build_reference_classifier
from models.task_heads import DEFAULT_TASK_HEAD, build_task_head

# These constants define the named djscc_residual_v1 topology. They are
# architectural implementation details fixed by that name, not free experiment
# settings. Widths, block count, output channels, and downsampling remain config.
_DOWNSAMPLE_KERNEL = 5  # literal-ok: djscc_residual_v1 topology
_DOWNSAMPLE_STRIDE = 2
_DOWNSAMPLE_PADDING = 2
_RESIDUAL_KERNEL = 3
_RESIDUAL_PADDING = 1
_PROJECTION_KERNEL = 3
_PROJECTION_PADDING = 1
_UPSAMPLE_KERNEL = 4  # literal-ok: djscc_residual_v1 topology
_UPSAMPLE_STRIDE = 2
_UPSAMPLE_PADDING = 1
_GROUP_NORM_GROUPS = 8  # literal-ok: djscc_residual_v1 topology for widths 64/128
_RGB_CHANNELS = 3
_COMPLEX_COMPONENTS = 2
_MODEL_COMPONENT_PATH = "djscc.djscc_residual_v1"


@dataclass(frozen=True)
class DJSCCOutput:
    transmitted_symbols: torch.Tensor
    received_symbols: torch.Tensor
    reconstruction: torch.Tensor
    logits: torch.Tensor
    papr_db: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            width, width, kernel_size=_RESIDUAL_KERNEL, padding=_RESIDUAL_PADDING
        )
        self.norm1 = _group_norm(width)
        self.activation1 = nn.PReLU(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=_RESIDUAL_KERNEL, padding=_RESIDUAL_PADDING
        )
        self.norm2 = _group_norm(width)
        self.activation2 = nn.PReLU(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.activation1(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return self.activation2(inputs + residual)


def _group_norm(width: int) -> nn.GroupNorm:
    if width % _GROUP_NORM_GROUPS != 0:
        raise ValueError(
            f"djscc_residual_v1 width {width} is not divisible by "
            f"{_GROUP_NORM_GROUPS} GroupNorm groups"
        )
    return nn.GroupNorm(_GROUP_NORM_GROUPS, width)


def pack_complex_symbols(projection: torch.Tensor) -> torch.Tensor:
    """Pack adjacent real/imaginary channels, then flatten C, row, column."""

    if projection.ndim != 4 or projection.shape[1] % _COMPLEX_COMPONENTS:  # literal-ok: BCHW tensor rank
        raise ValueError("projection must have shape [B, 2C, h, w]")
    if projection.is_complex() or not torch.is_floating_point(projection):
        raise TypeError("projection must be a real floating-point tensor")
    batch, doubled_channels, height, width = projection.shape
    paired = projection.reshape(
        batch,
        doubled_channels // _COMPLEX_COMPONENTS,
        _COMPLEX_COMPONENTS,
        height,
        width,
    )
    # Native complex half remains experimental in PyTorch. Keep the channel
    # interface at complex64 under AMP; decoder convolutions still autocast.
    if paired.dtype in (torch.float16, torch.bfloat16):
        paired = paired.float()
    symbols = torch.complex(paired[:, :, 0], paired[:, :, 1])
    return symbols.flatten(start_dim=1)


def unpack_complex_symbols(
    symbols: torch.Tensor,
    *,
    complex_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Exact inverse of :func:`pack_complex_symbols`."""

    if symbols.ndim != 2 or not symbols.is_complex():
        raise TypeError("symbols must be a native complex tensor with shape [B, k]")
    expected = complex_channels * height * width
    if symbols.shape[1] != expected:
        raise ValueError(
            f"symbol count {symbols.shape[1]} does not match C*h*w={expected}"
        )
    grid = symbols.reshape(symbols.shape[0], complex_channels, height, width)
    paired = torch.stack((grid.real, grid.imag), dim=2)
    return paired.reshape(
        symbols.shape[0],
        complex_channels * _COMPLEX_COMPONENTS,
        height,
        width,
    )


class DJSCCEncoder(nn.Module):
    def __init__(
        self,
        *,
        stem_channels: int,
        body_channels: int,
        residual_blocks: int,
        complex_channels: int,
        peak_constraint: PeakPowerConstraint | None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                _RGB_CHANNELS,
                stem_channels,
                kernel_size=_DOWNSAMPLE_KERNEL,
                stride=_DOWNSAMPLE_STRIDE,
                padding=_DOWNSAMPLE_PADDING,
            ),
            _group_norm(stem_channels),
            nn.PReLU(stem_channels),
        )
        self.body_entry = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                body_channels,
                kernel_size=_DOWNSAMPLE_KERNEL,
                stride=_DOWNSAMPLE_STRIDE,
                padding=_DOWNSAMPLE_PADDING,
            ),
            _group_norm(body_channels),
            nn.PReLU(body_channels),
        )
        self.residual = nn.Sequential(
            *(ResidualBlock(body_channels) for _ in range(residual_blocks))
        )
        self.projection = nn.Conv2d(
            body_channels,
            complex_channels * _COMPLEX_COMPONENTS,
            kernel_size=_PROJECTION_KERNEL,
            padding=_PROJECTION_PADDING,
        )
        self.peak_constraint = peak_constraint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.residual(self.body_entry(self.stem(inputs)))
        symbols = pack_complex_symbols(self.projection(features))
        symbols = normalize_unit_average_power(symbols)
        if self.peak_constraint is not None:
            symbols = self.peak_constraint(symbols)
        return symbols


class DJSCCDecoder(nn.Module):
    def __init__(
        self,
        *,
        stem_channels: int,
        body_channels: int,
        residual_blocks: int,
        complex_channels: int,
        latent_height: int,
        latent_width: int,
        classes: int,
        task_head_name: str,
    ) -> None:
        super().__init__()
        self.complex_channels = complex_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.ingress = nn.Sequential(
            nn.Conv2d(
                complex_channels * _COMPLEX_COMPONENTS,
                body_channels,
                kernel_size=_PROJECTION_KERNEL,
                padding=_PROJECTION_PADDING,
            ),
            _group_norm(body_channels),
            nn.PReLU(body_channels),
        )
        self.residual = nn.Sequential(
            *(ResidualBlock(body_channels) for _ in range(residual_blocks))
        )
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(
                body_channels,
                stem_channels,
                kernel_size=_UPSAMPLE_KERNEL,
                stride=_UPSAMPLE_STRIDE,
                padding=_UPSAMPLE_PADDING,
            ),
            _group_norm(stem_channels),
            nn.PReLU(stem_channels),
            nn.ConvTranspose2d(
                stem_channels,
                _RGB_CHANNELS,
                kernel_size=_UPSAMPLE_KERNEL,
                stride=_UPSAMPLE_STRIDE,
                padding=_UPSAMPLE_PADDING,
            ),
            nn.Sigmoid(),
        )
        self.task_head = build_task_head(
            task_head_name,
            input_channels=body_channels,
            classes=classes,
        )

    def forward(self, symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = unpack_complex_symbols(
            symbols,
            complex_channels=self.complex_channels,
            height=self.latent_height,
            width=self.latent_width,
        )
        features = self.residual(self.ingress(packed))
        return self.reconstruction_head(features), self.task_head(features)


class DJSCC(nn.Module):
    def __init__(
        self,
        *,
        encoder: DJSCCEncoder,
        channel: nn.Module,
        decoder: DJSCCDecoder,
        image_height: int,
        image_width: int,
        k: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder
        self.image_height = image_height
        self.image_width = image_width
        self.k = k
        self.total_parameter_count = sum(
            parameter.numel() for parameter in self.parameters()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        snr_db: float | torch.Tensor,
        *,
        unit_noise: torch.Tensor | None = None,
    ) -> DJSCCOutput:
        if (
            inputs.ndim != 4  # literal-ok: BCHW tensor rank
            or inputs.shape[1:] != (_RGB_CHANNELS, self.image_height, self.image_width)
        ):
            raise ValueError(
                "DJSCC input must have configured shape "
                f"[B, 3, {self.image_height}, {self.image_width}]"
            )
        if not torch.is_floating_point(inputs) or inputs.is_complex():
            raise TypeError("DJSCC input must be a real floating-point tensor")
        if not torch.isfinite(inputs).all() or torch.any(inputs < 0) or torch.any(inputs > 1):
            raise ValueError("DJSCC input must contain finite RGB values in [0, 1]")
        transmitted = self.encoder(inputs)
        if transmitted.shape != (inputs.shape[0], self.k):
            raise RuntimeError(
                f"encoder emitted {tuple(transmitted.shape)}, expected "
                f"{(inputs.shape[0], self.k)}"
            )
        received = self.channel(transmitted, snr_db, unit_noise=unit_noise)
        reconstruction, logits = self.decoder(received)
        return DJSCCOutput(
            transmitted_symbols=transmitted,
            received_symbols=received,
            reconstruction=reconstruction,
            logits=logits,
            papr_db=symbol_papr_db(transmitted),
        )


@functools.cache
def _reference_classifier_parameter_count(dataset: str) -> int:
    model = build_reference_classifier(dataset)
    return sum(parameter.numel() for parameter in model.parameters())


def enforce_parameter_cap(
    model: DJSCC,
    *,
    max_params_millions: int | float,
    reference_parameter_count: int,
) -> None:
    if max_params_millions <= 0:
        raise ValueError("max_params_millions must be positive")
    absolute_cap = int(float(max_params_millions) * 1_000_000)
    count = model.total_parameter_count
    if count > absolute_cap:
        raise ValueError(
            f"DJSCC parameter count {count} exceeds absolute cap {absolute_cap}"
        )
    if count > reference_parameter_count:
        raise ValueError(
            f"DJSCC parameter count {count} exceeds reference classifier "
            f"count {reference_parameter_count}"
        )


def _mapping(value: Any, *, label: str) -> Any:
    if not hasattr(value, "__getitem__"):
        raise TypeError(f"{label} must be a mapping")
    return value


def build_djscc(
    config: RunConfig,
    *,
    task_head_name: str = DEFAULT_TASK_HEAD,
    peak_constraint: PeakPowerConstraint | None = None,
    device: torch.device | str | None = None,
) -> DJSCC:
    """Build the resolved W2 model without mutating ambient Torch RNG."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a resolved RunConfig")
    resolved = config.resolved
    parameters = config.parameters
    learned = _mapping(parameters["learned_system"], label="learned_system")
    datasets = _mapping(parameters["datasets"], label="datasets")
    bandwidth = _mapping(parameters["bandwidth"], label="bandwidth")

    dataset = resolved["dataset"]
    ratio = resolved["bw_ratio"]
    architecture = resolved.get("architecture", learned["encoder_arch"])
    if architecture != learned["encoder_arch"] or architecture != "djscc_residual_v1":
        raise ValueError(f"unsupported DJSCC architecture: {architecture}")
    if learned["decoder_arch"] != "mirror_of_encoder_with_two_heads":
        raise ValueError("unsupported DJSCC decoder architecture")
    if learned["encoder_complex_packing"] != "channel_pairs_to_real_imag":
        raise ValueError("unsupported DJSCC complex packing convention")
    if tuple(learned["decoder_heads"]) != ("reconstruction", "classification"):
        raise ValueError("configured DJSCC decoder heads differ")
    if bandwidth["power_constraint"] != "unit_average_power":
        raise ValueError("unsupported learned-system power constraint")

    dataset_parameters = _mapping(datasets[dataset], label=f"datasets.{dataset}")
    image_height, image_width, image_channels = dataset_parameters["image_size"]
    if image_channels != _RGB_CHANNELS:
        raise ValueError("djscc_residual_v1 requires RGB input")
    downsample = int(learned["encoder_downsample_factor"])
    if image_height % downsample or image_width % downsample:
        raise ValueError("configured image size is not divisible by encoder downsampling")
    latent_height = image_height // downsample
    latent_width = image_width // downsample
    complex_channels = int(learned["encoder_output_complex_channels"][ratio])
    expected_k = complex_channels * latent_height * latent_width
    configured_k = int(resolved["k"])
    parameter_k = int(bandwidth["k_symbols"][dataset][ratio])
    if expected_k != configured_k or expected_k != parameter_k:
        raise ValueError(
            "configured symbol budget disagrees: "
            f"C*h*w={expected_k}, resolved={configured_k}, params={parameter_k}"
        )

    train_seed = resolved["train_seed"]
    if not isinstance(train_seed, int) or isinstance(train_seed, bool):
        raise TypeError("resolved train_seed must be an integer")
    seed = keyed_torch_seed(
        {"train_seed": train_seed, "component_path": _MODEL_COMPONENT_PATH}
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = DJSCCEncoder(
            stem_channels=int(learned["encoder_stem_channels"]),
            body_channels=int(learned["encoder_body_channels"]),
            residual_blocks=int(learned["encoder_residual_blocks"]),
            complex_channels=complex_channels,
            peak_constraint=peak_constraint,
        )
        decoder = DJSCCDecoder(
            stem_channels=int(learned["encoder_stem_channels"]),
            body_channels=int(learned["encoder_body_channels"]),
            residual_blocks=int(learned["encoder_residual_blocks"]),
            complex_channels=complex_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            classes=int(dataset_parameters["classes"]),
            task_head_name=task_head_name,
        )
        model = DJSCC(
            encoder=encoder,
            channel=build_channel(str(resolved["channel"])),
            decoder=decoder,
            image_height=image_height,
            image_width=image_width,
            k=configured_k,
        )

    enforce_parameter_cap(
        model,
        max_params_millions=learned["max_params_millions"],
        reference_parameter_count=_reference_classifier_parameter_count(dataset),
    )
    return model.to(device) if device is not None else model
