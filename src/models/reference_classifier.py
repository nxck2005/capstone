"""From-scratch deterministic Torchvision reference classifier (DEC-15)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models

from artifacts.rng import keyed_torch_seed
from config.params import get
from data.preprocessing import channel_normalisation_stats


_RESNET_CONSTRUCTORS: dict[str, Callable[..., models.ResNet]] = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


class ChannelNormalisation(nn.Module):
    """The required model-owned RGB normalisation, with persistent buffers."""

    def __init__(self) -> None:
        super().__init__()
        mean, std = channel_normalisation_stats()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) / self.std


class ReferenceClassifier(nn.Module):
    """A Torchvision ResNet receiving unnormalised RGB tensors in ``[0, 1]``."""

    def __init__(self, architecture: str, classes: int) -> None:
        super().__init__()
        try:
            constructor = _RESNET_CONSTRUCTORS[architecture]
        except KeyError:
            raise ValueError(f"unsupported reference classifier architecture: {architecture}") from None
        self.architecture = architecture
        self.normalisation = ChannelNormalisation()
        self.network = constructor(weights=None, num_classes=classes)
        self.total_parameter_count = sum(parameter.numel() for parameter in self.parameters())
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(self.normalisation(inputs))


def build_reference_classifier(
    dataset: str,
    *,
    architecture: str | None = None,
    train_seed: int | None = None,
    weights: object | None = None,
    device: torch.device | str | None = None,
) -> ReferenceClassifier:
    """Construct a CPU-initialised classifier without mutating ambient Torch RNG."""

    if get("reference_classifier.implementation") != "torchvision_models_resnet":
        raise NotImplementedError("unsupported reference classifier implementation")
    if get("reference_classifier.weights") is not None or weights is not None:
        raise ValueError("reference classifier weights must be null")
    selected_architecture = architecture or get("reference_classifier.arch")
    if not isinstance(selected_architecture, str):
        raise TypeError("reference classifier architecture must be a string")
    dataset_params = get(f"datasets.{dataset}")
    if not isinstance(dataset_params, dict) or "classes" not in dataset_params:
        raise ValueError(f"unknown reference classifier dataset: {dataset}")
    classes = int(dataset_params["classes"])
    selected_seed = (
        get("reference_classifier.clean_train_seed")
        if train_seed is None
        else train_seed
    )
    if not isinstance(selected_seed, int) or isinstance(selected_seed, bool):
        raise TypeError("reference classifier train_seed must be an integer")
    template = get("reference_classifier.init_component_path_template")
    component_path = str(template).format(arch=selected_architecture)
    seed = keyed_torch_seed(
        {"train_seed": selected_seed, "component_path": component_path}
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ReferenceClassifier(selected_architecture, classes)
    return model.to(device) if device is not None else model
