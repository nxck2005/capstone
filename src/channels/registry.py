"""Name-based construction seam for differentiable channel models (SR-5)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from channels.awgn import AWGN
from config.params import get


@runtime_checkable
class Channel(Protocol):
    """The minimal channel interface used by learned-system code."""

    def forward(
        self,
        symbols: torch.Tensor,
        snr_db: float | torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...


ChannelFactory = Callable[..., nn.Module]
_CHANNELS: dict[str, ChannelFactory] = {}


def register_channel(name: str, factory: ChannelFactory) -> None:
    """Register one channel factory, rejecting ambiguity."""

    if not isinstance(name, str) or not name:
        raise ValueError("channel name must be a non-empty string")
    if not callable(factory):
        raise TypeError("channel factory must be callable")
    if name in _CHANNELS:
        raise ValueError(f"channel already registered: {name}")
    _CHANNELS[name] = factory


def build_channel(name: str, **kwargs: Any) -> nn.Module:
    """Construct a registered channel by configured string name."""

    try:
        factory = _CHANNELS[name]
    except KeyError:
        raise ValueError(
            f"unknown channel {name!r}; registered={sorted(_CHANNELS)}"
        ) from None
    channel = factory(**kwargs)
    if not isinstance(channel, nn.Module):
        raise TypeError(f"channel factory {name!r} did not return torch.nn.Module")
    return channel


def channel_names() -> tuple[str, ...]:
    return tuple(sorted(_CHANNELS))


_BUILTIN_CHANNELS: dict[str, ChannelFactory] = {"awgn": AWGN}
_configured = tuple(get("channel.models_supported"))
if set(_configured) != set(_BUILTIN_CHANNELS):
    raise RuntimeError(
        "implemented built-in channels differ from params.channel.models_supported: "
        f"implemented={sorted(_BUILTIN_CHANNELS)}, configured={sorted(_configured)}"
    )
for _name in _configured:
    register_channel(_name, _BUILTIN_CHANNELS[_name])
