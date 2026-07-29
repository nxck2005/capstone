"""Reference-classifier construction and model-owned normalisation tests."""

from __future__ import annotations

import torch
import pytest
from torch import nn

from artifacts.rng import keyed_torch_seed
from config.params import get
from data.preprocessing import channel_normalisation_stats
from models.reference_classifier import build_reference_classifier
import models.reference_classifier as reference_classifier


def _state(dataset: str = "cifar10", **kwargs: object) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in build_reference_classifier(dataset, **kwargs).state_dict().items()
    }


def test_identical_init_identity_is_exact_and_ambient_torch_rng_is_restored():
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    first = _state(train_seed=17)
    after = torch.get_rng_state()
    second = _state(train_seed=17)

    assert torch.equal(before, after)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_init_identity_changes_for_seed_and_architecture():
    first = _state(train_seed=17)
    changed_seed = _state(train_seed=18)
    changed_architecture = _state(architecture="resnet34", train_seed=17)

    assert not torch.equal(first["network.conv1.weight"], changed_seed["network.conv1.weight"])
    assert keyed_torch_seed(
        {"train_seed": 17, "component_path": "reference_classifier.resnet18"}
    ) != keyed_torch_seed(
        {"train_seed": 17, "component_path": "reference_classifier.resnet34"}
    )
    assert len(changed_architecture) != len(first)


def test_model_normalises_inside_the_first_model_operation():
    model = build_reference_classifier("cifar10")
    captured: dict[str, torch.Tensor] = {}

    class Capture(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            captured["value"] = value
            return torch.zeros((value.shape[0], 10), dtype=value.dtype)

    model.network = Capture()
    mean, std = channel_normalisation_stats()
    inputs = torch.tensor(mean).view(1, 3, 1, 1).expand(2, 3, 4, 4)
    output = model(inputs)

    assert output.shape == (2, 10)
    assert torch.equal(captured["value"], torch.zeros_like(captured["value"]))
    assert torch.equal(model.normalisation.mean.flatten(), torch.tensor(mean))
    assert torch.equal(model.normalisation.std.flatten(), torch.tensor(std))


def test_configured_image_shapes_return_ten_raw_logits():
    model = build_reference_classifier("imagenette160").eval()
    for dataset in ("imagenette160", "stl10", "cifar10"):
        height, width, _channels = get(f"datasets.{dataset}.image_size")
        output = model(torch.rand(2, 3, height, width))
        assert output.shape == (2, get(f"datasets.{dataset}.classes"))


def test_weights_and_unsupported_architectures_fail_closed():
    try:
        build_reference_classifier("cifar10", weights="DEFAULT")
    except ValueError as error:
        assert "weights must be null" in str(error)
    else:
        raise AssertionError("pretrained weights request unexpectedly accepted")
    try:
        build_reference_classifier("cifar10", architecture="tiny-resnet")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unsupported architecture unexpectedly accepted")


@pytest.mark.parametrize("architecture", ("resnet18", "resnet34", "resnet50"))
def test_torchvision_constructor_is_always_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
):
    original = reference_classifier._RESNET_CONSTRUCTORS[architecture]
    observed: dict[str, object] = {}

    def constructor(**kwargs: object):
        observed.update(kwargs)
        return original(**kwargs)

    monkeypatch.setitem(reference_classifier._RESNET_CONSTRUCTORS, architecture, constructor)
    build_reference_classifier("cifar10", architecture=architecture)

    assert observed["weights"] is None
    assert observed["num_classes"] == 10


def test_parameter_counts_are_recorded_and_trainable():
    model = build_reference_classifier("cifar10")

    assert model.total_parameter_count == sum(item.numel() for item in model.parameters())
    assert model.trainable_parameter_count == sum(
        item.numel() for item in model.parameters() if item.requires_grad
    )
    assert model.total_parameter_count == model.trainable_parameter_count
