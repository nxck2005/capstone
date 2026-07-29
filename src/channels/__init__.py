"""Complex-symbol channel, power, noise, and PAPR foundations (W2)."""

from channels.awgn import AWGN, keyed_complex_noise
from channels.power import (
    PeakPowerConstraint,
    normalize_unit_average_power,
    symbol_papr_db,
)
from channels.registry import (
    Channel,
    build_channel,
    channel_names,
    register_channel,
)

__all__ = [
    "AWGN",
    "Channel",
    "PeakPowerConstraint",
    "build_channel",
    "channel_names",
    "keyed_complex_noise",
    "normalize_unit_average_power",
    "register_channel",
    "symbol_papr_db",
]
