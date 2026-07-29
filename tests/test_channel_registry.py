"""Channel construction seam coverage for SR-5."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from channels import build_channel, channel_names, register_channel
from config.params import get


class _IdentityChannel(nn.Module):
    def forward(self, symbols, snr_db, **kwargs):
        return symbols


def test_every_configured_channel_is_pre_registered():
    assert set(get("channel.models_supported")).issubset(channel_names())
    assert isinstance(build_channel("awgn"), nn.Module)


def test_stub_channel_registers_and_builds_without_model_changes():
    name = "unit_test_identity_channel"
    register_channel(name, _IdentityChannel)

    symbols = torch.ones(2, 3, dtype=torch.complex64)
    assert torch.equal(build_channel(name)(symbols, 0), symbols)

    with pytest.raises(ValueError, match="already registered"):
        register_channel(name, _IdentityChannel)


def test_unknown_channel_fails_closed():
    with pytest.raises(ValueError, match="unknown channel"):
        build_channel("not-a-channel")
