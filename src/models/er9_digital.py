"""The AM-96 task-aware digital control model (ER-9)."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from artifacts.rng import keyed_torch_seed
from config.run_config import RunConfig
from models.djscc import ResidualBlock, _group_norm
from models.reference_classifier import ChannelNormalisation, build_reference_classifier
from models.task_heads import DEFAULT_TASK_HEAD, ImageClassificationHead
from evaluation.er9_protocol import (
    FeatureFactorisation,
    UniformScalarQuantizer,
    factorisation_for_dimension,
    flatten_features,
    unflatten_features,
)


_RGB_CHANNELS = 3  # literal-ok: RGB input channel count
_DOWNSAMPLE_KERNEL = 5  # literal-ok: djscc_residual_v1 topology
_DOWNSAMPLE_STRIDE = 2
_DOWNSAMPLE_PADDING = 2
_RESIDUAL_KERNEL = 3
_RESIDUAL_PADDING = 1
_INTERFACE_KERNEL = 3
_INTERFACE_PADDING = 1
_GROUP_NORM_GROUPS = 8  # literal-ok: djscc_residual_v1 topology
_MODEL_COMPONENT_ROOT = "er9.digital_control"


@dataclass(frozen=True)
class ER9DigitalOutput:
    """Training/evaluation tensors before any non-differentiable transport."""

    trunk_features: torch.Tensor
    transmitted_features: torch.Tensor
    quantised_indices: torch.Tensor
    dequantised_features: torch.Tensor
    logits: torch.Tensor


class ER9ResidualTrunk(nn.Module):
    """The shared DJSCC residual trunk ending before analog projection."""

    def __init__(
        self,
        *,
        stem_channels: int,
        body_channels: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        self.input_normalisation = ChannelNormalisation()
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.residual(
            self.body_entry(self.stem(self.input_normalisation(inputs)))
        )


@functools.cache
def _reference_parameter_count(dataset: str) -> int:
    model = build_reference_classifier(dataset)
    return sum(parameter.numel() for parameter in model.parameters())


class ER9DigitalModel(nn.Module):
    """Task-only ER-9 model with a fixed, protocol-counted interface."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        classes: int,
        stem_channels: int,
        body_channels: int,
        residual_blocks: int,
        factorisation: FeatureFactorisation,
        quantiser: UniformScalarQuantizer,
        task_head_name: str = DEFAULT_TASK_HEAD,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.factorisation = factorisation
        self.quantiser = quantiser
        self.trunk = ER9ResidualTrunk(
            stem_channels=stem_channels,
            body_channels=body_channels,
            residual_blocks=residual_blocks,
        )
        self.channel_interface = nn.Conv2d(
            body_channels,
            factorisation.output_channels,
            kernel_size=_INTERFACE_KERNEL,
            padding=_INTERFACE_PADDING,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (factorisation.pool_height, factorisation.pool_width)
        )
        if task_head_name != DEFAULT_TASK_HEAD:
            raise ValueError(f"ER-9 only supports the configured task head {DEFAULT_TASK_HEAD!r}")
        self.task_head = ImageClassificationHead(
            input_channels=factorisation.output_channels,
            classes=classes,
        )
        self.total_parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        expected = (_RGB_CHANNELS, self.image_height, self.image_width)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:  # literal-ok: BCHW tensor rank
            raise TypeError("ER-9 input must be a rank-4 tensor")
        if tuple(inputs.shape[1:]) != expected:
            raise ValueError(f"ER-9 input shape {tuple(inputs.shape[1:])} differs from {expected}")
        if not torch.is_floating_point(inputs) or inputs.is_complex():
            raise TypeError("ER-9 input must be a real floating-point tensor")
        if not torch.isfinite(inputs).all() or torch.any(inputs < 0) or torch.any(inputs > 1):  # literal-ok: model input range
            raise ValueError("ER-9 input must contain finite RGB values in [0, 1]")

    def trunk_output(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        return self.trunk(inputs)

    def interface_output(self, trunk_features: torch.Tensor) -> torch.Tensor:
        return self.adaptive_pool(self.channel_interface(trunk_features))

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(trunk, pooled_features, exact integer indices)``."""

        trunk = self.trunk_output(inputs)
        transmitted = self.interface_output(trunk)
        indices = self.quantiser.quantise(transmitted)
        return trunk, transmitted, flatten_features(indices, self.factorisation)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        values = self.quantiser.dequantise(indices)
        return unflatten_features(values, self.factorisation)

    def forward(self, inputs: torch.Tensor) -> ER9DigitalOutput:
        trunk = self.trunk_output(inputs)
        transmitted = self.interface_output(trunk)
        dequantised, indices_grid = self.quantiser.straight_through(transmitted)
        indices = flatten_features(indices_grid, self.factorisation)
        logits = self.task_head(dequantised)
        return ER9DigitalOutput(
            trunk_features=trunk,
            transmitted_features=transmitted,
            quantised_indices=indices,
            dequantised_features=dequantised,
            logits=logits,
        )

    def logits_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.task_head(self.decode_indices(indices))


def build_er9_model(
    config: RunConfig,
    *,
    transmit_dim: int,
    quantiser_bits: int,
    device: torch.device | str | None = None,
) -> ER9DigitalModel:
    """Build one fresh ER-9 candidate from its keyed configuration identity."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a resolved RunConfig")
    if config.resolved.get("system") != "er9_digital":
        raise ValueError("ER-9 model requires the er9_digital system choice")
    factorisation = factorisation_for_dimension(transmit_dim)
    quantiser = UniformScalarQuantizer(quantiser_bits)
    dataset = str(config.resolved["dataset"])
    dataset_parameters = config.parameters["datasets"][dataset]
    learned = config.parameters["learned_system"]
    image_height, image_width, image_channels = dataset_parameters["image_size"]
    if image_channels != _RGB_CHANNELS:
        raise ValueError("ER-9 requires RGB input")
    component_path = (
        f"{_MODEL_COMPONENT_ROOT}.transmit_dim_{transmit_dim}."
        f"quantiser_bits_{quantiser_bits}"
    )
    seed = keyed_torch_seed(
        {"train_seed": int(config.resolved["train_seed"]), "component_path": component_path}
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ER9DigitalModel(
            image_height=int(image_height),
            image_width=int(image_width),
            classes=int(dataset_parameters["classes"]),
            stem_channels=int(learned["encoder_stem_channels"]),
            body_channels=int(learned["encoder_body_channels"]),
            residual_blocks=int(learned["encoder_residual_blocks"]),
            factorisation=factorisation,
            quantiser=quantiser,
        )
    absolute_cap = int(float(learned["max_params_millions"]) * 1_000_000)
    if model.total_parameter_count > absolute_cap:
        raise ValueError("ER-9 model exceeds the configured absolute parameter cap")
    if model.total_parameter_count > _reference_parameter_count(dataset):
        raise ValueError("ER-9 model exceeds the frozen reference-classifier parameter count")
    return model.to(device) if device is not None else model


__all__ = [
    "ER9DigitalModel",
    "ER9DigitalOutput",
    "ER9ResidualTrunk",
    "build_er9_model",
]
