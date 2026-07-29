"""Config-derived DJSCC classification-plus-reconstruction objective (SR-8)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from config.run_config import RunConfig
from models.djscc import DJSCCOutput


@dataclass(frozen=True)
class DJSCCLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    reconstruction_mse: torch.Tensor


class DJSCCObjective(nn.Module):
    """``CE + lambda * MSE`` with lambda retained as data, including zero."""

    def __init__(self, reconstruction_weight: float) -> None:
        super().__init__()
        if isinstance(reconstruction_weight, bool) or not isinstance(
            reconstruction_weight, int | float
        ):
            raise TypeError("reconstruction weight must be numeric")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction weight must be non-negative")
        self.reconstruction_weight = float(reconstruction_weight)

    @classmethod
    def from_config(cls, config: RunConfig) -> DJSCCObjective:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a resolved RunConfig")
        if config.parameters["learned_system"]["loss"] != "CE + lambda * MSE":
            raise ValueError("unsupported configured learned-system loss")
        return cls(config.resolved["lambda"])

    def forward(
        self,
        output: DJSCCOutput,
        targets: torch.Tensor,
        augmented_inputs: torch.Tensor,
    ) -> DJSCCLoss:
        if output.reconstruction.shape != augmented_inputs.shape:
            raise ValueError(
                "reconstruction MSE target must be the same augmented input tensor shape"
            )
        cross_entropy = F.cross_entropy(output.logits, targets)
        reconstruction_mse = F.mse_loss(output.reconstruction, augmented_inputs)
        total = cross_entropy + self.reconstruction_weight * reconstruction_mse
        return DJSCCLoss(
            total=total,
            cross_entropy=cross_entropy,
            reconstruction_mse=reconstruction_mse,
        )
