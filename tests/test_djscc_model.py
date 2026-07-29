"""Config-derived DJSCC topology, shapes, identity, and parameter caps."""

from __future__ import annotations

import gc

import pytest
import torch
from torch import nn

from channels.awgn import keyed_complex_noise
from config.params import get
from models.djscc import (
    build_djscc,
    pack_complex_symbols,
    unpack_complex_symbols,
)
from models.task_heads import (
    DEFAULT_TASK_HEAD,
    build_task_head,
    register_task_head,
    task_head_names,
)


def test_complex_packing_round_trip_and_documented_order():
    projection = torch.arange(2 * 6 * 2 * 3, dtype=torch.float32).reshape(2, 6, 2, 3)
    symbols = pack_complex_symbols(projection)

    expected_first = torch.complex(projection[0, 0], projection[0, 1]).flatten()
    assert torch.equal(symbols[0, : expected_first.numel()], expected_first)
    unpacked = unpack_complex_symbols(
        symbols, complex_channels=3, height=2, width=3
    )
    assert torch.equal(unpacked, projection)


@pytest.mark.parametrize("dataset", ["imagenette160", "stl10", "cifar10"])
@pytest.mark.parametrize("ratio", list(get("bandwidth.ratios")))
def test_every_dataset_ratio_emits_exact_complex_symbol_budget_and_meets_caps(
    run_config_factory,
    dataset,
    ratio,
):
    config = run_config_factory(dataset, ratio)
    model = build_djscc(config)
    height, width, _channels = get(f"datasets.{dataset}.image_size")
    inputs = torch.rand(1, 3, height, width)

    with torch.no_grad():
        symbols = model.encoder(inputs)

    assert symbols.is_complex()
    assert symbols.shape == (1, get(f"bandwidth.k_symbols.{dataset}.{ratio}"))
    torch.testing.assert_close(
        symbols.abs().square().mean(dim=1),
        torch.ones(1),
        atol=1e-3,
        rtol=0,
    )
    assert model.total_parameter_count <= get("learned_system.max_params_millions") * 1_000_000
    assert model.total_parameter_count <= 11_181_642
    del model
    gc.collect()


def test_complete_output_shapes_and_configured_awgn(run_config_factory):
    config = run_config_factory("cifar10", "r_1_6")
    model = build_djscc(config).eval()
    inputs = torch.rand(2, 3, 32, 32)
    noise = keyed_complex_noise(
        ["shape-a", "shape-b"], config.resolved["k"], dtype=torch.complex64
    )

    output = model(inputs, config.resolved["train_snr_db"], unit_noise=noise)

    assert output.transmitted_symbols.shape == (2, config.resolved["k"])
    assert output.received_symbols.shape == output.transmitted_symbols.shape
    assert output.reconstruction.shape == inputs.shape
    assert output.logits.shape == (2, get("datasets.cifar10.classes"))
    assert output.papr_db.shape == (2,)
    assert model.channel.__class__.__name__ == "AWGN"


class _StubHead(nn.Module):
    def __init__(self, input_channels: int, classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(input_channels, classes + 1)

    def forward(self, features):
        return self.projection(self.pool(features).flatten(1))


def test_task_head_registry_stub_and_failures_do_not_change_encoder_or_channel(
    run_config_factory,
):
    name = "unit_test_extra_classification"
    register_task_head(name, _StubHead)
    assert DEFAULT_TASK_HEAD in task_head_names()
    assert isinstance(
        build_task_head(name, input_channels=8, classes=10), _StubHead
    )
    with pytest.raises(ValueError, match="already registered"):
        register_task_head(name, _StubHead)
    with pytest.raises(ValueError, match="unknown task head"):
        build_task_head("missing-task-head")

    config = run_config_factory()
    default_model = build_djscc(config)
    stub_model = build_djscc(config, task_head_name=name)
    for key, value in default_model.encoder.state_dict().items():
        assert torch.equal(value, stub_model.encoder.state_dict()[key])
    assert type(default_model.channel) is type(stub_model.channel)


def test_keyed_initialization_is_seeded_and_does_not_mutate_ambient_rng(
    run_config_factory,
):
    config = run_config_factory(train_seed=get("evaluation.train_seeds")[0])
    other = run_config_factory(train_seed=get("evaluation.train_seeds")[1])
    torch.manual_seed(9981)
    before = torch.random.get_rng_state().clone()
    first = build_djscc(config)
    after = torch.random.get_rng_state()
    second = build_djscc(config)
    changed = build_djscc(other)

    assert torch.equal(before, after)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
    assert any(
        not torch.equal(value, changed.state_dict()[key])
        for key, value in first.state_dict().items()
        if value.is_floating_point()
    )


def test_model_rejects_non_configured_input(run_config_factory):
    model = build_djscc(run_config_factory()).eval()
    with pytest.raises(ValueError, match="configured shape"):
        model(torch.rand(1, 3, 96, 96), 0)
