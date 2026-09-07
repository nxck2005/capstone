"""AM-96 ER-9 digital-interface primitives.

This module contains only the frozen interface mechanics: the deterministic
``D -> C x 8 x 8`` factorisation, the protocol quantiser, the static range
coder and the counted raw-escape message.  The physical channel remains owned
by ``baseline.classical.channel_transport`` and the LDPC implementation remains
owned by ``baseline.ldpc``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from config.params import get


class ER9ProtocolHold(RuntimeError):
    """The AM-96 ER-9 interface contract is not executable as configured."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ER9ProtocolHold(message)


@dataclass(frozen=True)
class FeatureFactorisation:
    """One exact, canonical transmitted feature geometry."""

    requested_dimension: int
    output_channels: int
    pool_height: int
    pool_width: int
    flatten_order: str
    reconstruction_order: str
    exactly_realised: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.output_channels, self.pool_height, self.pool_width)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_dimension": self.requested_dimension,
            "output_channels": self.output_channels,
            "pool_height": self.pool_height,
            "pool_width": self.pool_width,
            "flatten_order": self.flatten_order,
            "reconstruction_order": self.reconstruction_order,
            "exactly_realised": self.exactly_realised,
        }


def factorisation_for_dimension(dimension: int) -> FeatureFactorisation:
    """Return AM-96's unique channel-major ``D/64 x 8 x 8`` geometry."""

    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("transmit dimension must be an integer")
    configured = tuple(int(value) for value in get("digital_semantic_control.transmit_dim_grid"))
    if dimension not in configured:
        raise ValueError(f"transmit dimension is not configured: {dimension}")
    if get("digital_semantic_control.transmit_dim_realised_by") != (
        "output_channel_count_and_adaptive_pooling"
    ):
        raise ER9ProtocolHold("ER-9 transmit-dimension mechanism differs from AM-96")
    divisor = 64  # literal-ok: AM-96 D=C*8*8 factorisation
    pool_height = int(get("digital_semantic_control.factorisation_pool_height"))
    pool_width = int(get("digital_semantic_control.factorisation_pool_width"))
    if dimension % divisor or pool_height != 8 or pool_width != 8:  # literal-ok: AM-96 8x8 pool
        raise ER9ProtocolHold("ER-9 configured factorisation is not the AM-96 8x8 rule")
    channels = dimension // divisor
    body_channels = int(get("learned_system.encoder_body_channels"))
    if channels > body_channels:
        raise ER9ProtocolHold(
            f"ER-9 dimension {dimension} needs {channels} channels, above trunk capacity {body_channels}"
        )
    flatten_order = str(get("digital_semantic_control.factorisation_flatten_order"))
    expected_order = "channel_major_row_major_column_major_nchw_contiguous"
    if flatten_order != expected_order:
        raise ER9ProtocolHold("ER-9 flatten order differs from AM-96")
    if get("digital_semantic_control.factorisation_channel_rule") != "transmit_dim_div_64":
        raise ER9ProtocolHold("ER-9 channel factorisation rule differs from AM-96")
    return FeatureFactorisation(
        requested_dimension=dimension,
        output_channels=channels,
        pool_height=pool_height,
        pool_width=pool_width,
        flatten_order=flatten_order,
        reconstruction_order="reshape_NCHW_C_8_8_contiguous",
        exactly_realised=channels * pool_height * pool_width == dimension,
    )


def flatten_features(features: torch.Tensor, factorisation: FeatureFactorisation) -> torch.Tensor:
    """Flatten ``[B,C,8,8]`` in ordinary contiguous NCHW order."""

    if not isinstance(features, torch.Tensor) or features.ndim != 4:  # literal-ok: BCHW tensor rank
        raise TypeError("ER-9 features must be a rank-4 torch tensor")
    expected = (factorisation.output_channels, factorisation.pool_height, factorisation.pool_width)
    if tuple(features.shape[1:]) != expected:
        raise ValueError(f"feature shape {tuple(features.shape[1:])} differs from {expected}")
    return features.contiguous().flatten(start_dim=1)


def unflatten_features(values: torch.Tensor, factorisation: FeatureFactorisation) -> torch.Tensor:
    """Exact inverse of :func:`flatten_features`."""

    if not isinstance(values, torch.Tensor) or values.ndim != 2:  # literal-ok: feature matrix rank
        raise TypeError("ER-9 flattened features must have shape [B,D]")
    if values.shape[1] != factorisation.requested_dimension:
        raise ValueError("ER-9 flattened feature dimension differs from factorisation")
    return values.contiguous().reshape(
        values.shape[0],
        factorisation.output_channels,
        factorisation.pool_height,
        factorisation.pool_width,
    )


class UniformScalarQuantizer:
    """AM-96 fixed-range uniform scalar quantiser with identity STE."""

    def __init__(self, bits: int) -> None:
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise TypeError("quantiser bits must be an integer")
        configured = tuple(int(value) for value in get("digital_semantic_control.quantiser_bits_grid"))
        if bits not in configured:
            raise ValueError(f"quantiser bits are not configured: {bits}")
        if get("digital_semantic_control.quantiser") != "uniform_scalar":
            raise ER9ProtocolHold("ER-9 quantiser family differs from AM-96")
        if get("digital_semantic_control.quantiser_range") != "tanh_to_minus_one_plus_one":
            raise ER9ProtocolHold("ER-9 quantiser range differs from AM-96")
        if get("digital_semantic_control.quantiser_scale") != "fixed_protocol_no_image_adaptive_scale":
            raise ER9ProtocolHold("ER-9 quantiser scale differs from AM-96")
        if int(get("digital_semantic_control.quantiser_scale_side_information_bits")) != 0:  # literal-ok: AM-96 no scale metadata
            raise ER9ProtocolHold("AM-96 requires zero quantiser scale side information")
        self.bits = bits
        self.levels = 2**bits  # literal-ok: AM-96 L=2**b
        self.step = 2.0 / (self.levels - 1)  # literal-ok: fixed [-1,+1] endpoint span

    def bounded(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(features)

    def quantise(self, features: torch.Tensor) -> torch.Tensor:
        bounded = self.bounded(features)
        scaled = (bounded + 1.0) / self.step * 1.0  # literal-ok: AM-96 [-1,+1] offset
        return torch.round(scaled).clamp(0, self.levels - 1).to(torch.long)  # literal-ok: index lower bound

    def dequantise(self, indices: torch.Tensor) -> torch.Tensor:
        if not isinstance(indices, torch.Tensor) or not indices.dtype == torch.long:
            raise TypeError("quantiser indices must be torch.long")
        if torch.any(indices < 0) or torch.any(indices >= self.levels):  # literal-ok: index bounds
            raise ValueError("quantiser index is outside the configured level range")
        return indices.to(dtype=torch.float32) * self.step - 1.0  # literal-ok: AM-96 lower endpoint

    def straight_through(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the true tanh-forward/identity-rounding STE pair."""

        bounded = self.bounded(features)
        indices = self.quantise(features)
        dequantised = self.dequantise(indices).to(dtype=features.dtype, device=features.device)
        straight_through = bounded + (dequantised - bounded).detach()
        return straight_through, indices

    def raw_bits(self, indices: np.ndarray) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        if values.size == 0 or np.any(values < 0) or np.any(values >= self.levels):
            raise ValueError("raw quantiser indices are outside the configured range")
        output: list[int] = []
        for value in values:
            output.extend((int(value) >> shift) & 1 for shift in range(self.bits - 1, -1, -1))
        return np.asarray(output, dtype=np.uint8)

    def raw_indices(self, bits: np.ndarray, count: int) -> np.ndarray:
        values = np.asarray(bits, dtype=np.uint8).reshape(-1)
        required = count * self.bits
        if values.size < required or np.any(values[:required] > 1):
            raise ValueError("raw quantiser bit stream is too short or non-binary")
        indices = []
        for offset in range(0, required, self.bits):
            value = 0
            for bit in values[offset : offset + self.bits]:
                value = (value << 1) | int(bit)
            indices.append(value)
        return np.asarray(indices, dtype=np.int64)


@dataclass(frozen=True)
class StaticEntropyModel:
    """A train-only static cumulative-frequency table."""

    levels: int
    frequencies: tuple[int, ...]
    cumulative: tuple[int, ...]
    total: int
    fit_split: str
    fit_rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "levels": self.levels,
            "frequencies": list(self.frequencies),
            "cumulative": list(self.cumulative),
            "total": self.total,
            "fit_split": self.fit_split,
            "fit_rule": self.fit_rule,
        }


def fit_static_entropy_model(indices: np.ndarray, levels: int) -> StaticEntropyModel:
    """Fit add-one static frequencies using train-split symbols only."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0 or np.any(values < 0) or np.any(values >= levels):
        raise ValueError("entropy-model training indices are invalid")
    frequencies = np.bincount(values, minlength=levels).astype(np.int64) + 1  # literal-ok: Laplace add-one
    cumulative = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(frequencies)))
    total = int(cumulative[-1])
    return StaticEntropyModel(
        levels=levels,
        frequencies=tuple(int(value) for value in frequencies),
        cumulative=tuple(int(value) for value in cumulative),
        total=total,
        fit_split="train",
        fit_rule="add_one_symbol_frequency_static_range_coder",
    )


_RANGE_STATE_BITS = 32  # literal-ok: fixed 32-bit integer arithmetic coder
_RANGE_TOP = (1 << _RANGE_STATE_BITS) - 1
_RANGE_HALF = 1 << (_RANGE_STATE_BITS - 1)
_RANGE_FIRST_QUARTER = 1 << (_RANGE_STATE_BITS - 2)
_RANGE_THIRD_QUARTER = 3 * _RANGE_FIRST_QUARTER  # literal-ok: arithmetic-coder third quarter


def _emit_bit(output: list[int], bit: int, pending: int) -> None:
    output.append(bit)
    output.extend((1 - bit for _ in range(pending)))  # literal-ok: binary complement


def _range_encode(symbols: np.ndarray, model: StaticEntropyModel) -> np.ndarray:
    low = 0
    high = _RANGE_TOP
    pending = 0
    output: list[int] = []
    for symbol in np.asarray(symbols, dtype=np.int64).reshape(-1):
        index = int(symbol)
        if index < 0 or index >= model.levels:
            raise ValueError("range-coder symbol is outside the entropy alphabet")
        span = high - low + 1
        high = low + (span * model.cumulative[index + 1] // model.total) - 1
        low = low + (span * model.cumulative[index] // model.total)
        while True:
            if high < _RANGE_HALF:
                _emit_bit(output, 0, pending)  # literal-ok: arithmetic-coder low-half emission
                pending = 0
            elif low >= _RANGE_HALF:
                _emit_bit(output, 1, pending)  # literal-ok: arithmetic-coder high-half emission
                pending = 0
                low -= _RANGE_HALF
                high -= _RANGE_HALF
            elif low >= _RANGE_FIRST_QUARTER and high < _RANGE_THIRD_QUARTER:
                pending += 1  # literal-ok: arithmetic-coder underflow counter
                low -= _RANGE_FIRST_QUARTER
                high -= _RANGE_FIRST_QUARTER
            else:
                break
            low = low * 2  # literal-ok: arithmetic-coder renormalisation
            high = high * 2 + 1  # literal-ok: arithmetic-coder renormalisation
    pending += 1  # literal-ok: arithmetic-coder termination
    _emit_bit(output, 0 if low < _RANGE_FIRST_QUARTER else 1, pending)  # literal-ok: terminal code point and follow bits
    return np.asarray(output, dtype=np.uint8)


class _BitReader:
    def __init__(self, bits: np.ndarray) -> None:
        self.bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        self.offset = 0

    def read(self) -> int:
        if self.offset >= self.bits.size:
            return 0  # literal-ok: zero padding after the exact symbol stream
        value = int(self.bits[self.offset])
        self.offset += 1
        if value not in (0, 1):  # literal-ok: binary bit validation
            raise ValueError("range-coder input is not binary")
        return value


def _range_decode(bits: np.ndarray, count: int, model: StaticEntropyModel) -> np.ndarray:
    if count <= 0:  # literal-ok: positive message symbol count
        raise ValueError("range-coder decode count must be positive")
    reader = _BitReader(bits)
    low = 0
    high = _RANGE_TOP
    code = 0
    for _ in range(_RANGE_STATE_BITS):
        code = code * 2 + reader.read()  # literal-ok: binary arithmetic-coder input
    decoded: list[int] = []
    for _ in range(count):
        span = high - low + 1
        scaled = ((code - low + 1) * model.total - 1) // span
        index = bisect.bisect_right(model.cumulative, scaled) - 1
        if index < 0 or index >= model.levels:
            raise ValueError("range-coder decoded symbol is outside the alphabet")
        decoded.append(index)
        high = low + (span * model.cumulative[index + 1] // model.total) - 1
        low = low + (span * model.cumulative[index] // model.total)
        while True:
            if high < _RANGE_HALF:
                pass
            elif low >= _RANGE_HALF:
                low -= _RANGE_HALF
                high -= _RANGE_HALF
                code -= _RANGE_HALF
            elif low >= _RANGE_FIRST_QUARTER and high < _RANGE_THIRD_QUARTER:
                low -= _RANGE_FIRST_QUARTER
                high -= _RANGE_FIRST_QUARTER
                code -= _RANGE_FIRST_QUARTER
            else:
                break
            low = low * 2  # literal-ok: arithmetic-coder renormalisation
            high = high * 2 + 1  # literal-ok: arithmetic-coder renormalisation
            code = code * 2 + reader.read()  # literal-ok: arithmetic-coder renormalisation
    return np.asarray(decoded, dtype=np.int64)


@dataclass(frozen=True)
class EncodedMessage:
    selector: str
    message_bits: np.ndarray
    range_bit_count: int
    raw_bit_count: int
    counted_bits: int
    packet_payload_bits: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "range_bit_count": self.range_bit_count,
            "raw_bit_count": self.raw_bit_count,
            "counted_bits": self.counted_bits,
            "packet_payload_bits": self.packet_payload_bits,
        }


def protocol_metadata_bits() -> int:
    selector = int(get("digital_semantic_control.framing_selector_bits"))
    scale = int(get("digital_semantic_control.quantiser_scale_side_information_bits"))
    if selector != 1:  # literal-ok: AM-96 selector framing bit
        raise ER9ProtocolHold("AM-96 requires one framing selector bit")
    if get("digital_semantic_control.entropy_stream_length_field") != (
        "none_decode_exact_transmit_dim_symbols"
    ):
        raise ER9ProtocolHold("ER-9 entropy stream length rule differs from AM-96")
    return selector + scale


def raw_bound_bits(dimension: int, bits: int) -> int:
    factorisation_for_dimension(dimension)
    UniformScalarQuantizer(bits)
    return dimension * bits + protocol_metadata_bits()


def encode_message(
    indices: np.ndarray,
    quantiser: UniformScalarQuantizer,
    entropy_model: StaticEntropyModel,
    *,
    packet_payload_bits: int,
) -> EncodedMessage:
    """Select range/raw, count the selector, and zero-pad to packet payload."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    raw = quantiser.raw_bits(values)
    coded = _range_encode(values, entropy_model)
    if not bool(get("digital_semantic_control.raw_escape_required")):
        raise ER9ProtocolHold("AM-96 requires raw escape")
    use_range = coded.size <= raw.size
    selector = "range" if use_range else "raw"
    body = coded if use_range else raw
    selector_bit = np.asarray([0 if use_range else 1], dtype=np.uint8)  # literal-ok: selector code points
    message = np.concatenate((selector_bit, body))
    if message.size > packet_payload_bits:
        raise ValueError("selected ER-9 message exceeds exact packet payload")
    padded = np.pad(message, (0, packet_payload_bits - message.size), constant_values=0)
    return EncodedMessage(
        selector=selector,
        message_bits=padded,
        range_bit_count=int(coded.size),
        raw_bit_count=int(raw.size),
        counted_bits=int(message.size),
        packet_payload_bits=packet_payload_bits,
    )


def decode_message(
    payload_bits: np.ndarray,
    *,
    dimension: int,
    quantiser: UniformScalarQuantizer,
    entropy_model: StaticEntropyModel,
) -> tuple[np.ndarray, str]:
    """Decode exactly ``D`` symbols; range-branch padding is never a symbol."""

    values = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    if values.size <= 1 or np.any(values > 1):  # literal-ok: selector plus binary payload
        raise ValueError("ER-9 payload is not a non-empty binary stream")
    if values[0] == 0:
        return _range_decode(values[1:], dimension, entropy_model), "range"
    if values[0] == 1:
        return quantiser.raw_indices(values[1:], dimension), "raw"
    raise ValueError("ER-9 selector is invalid")


def expected_level_values(bits: int) -> np.ndarray:
    quantiser = UniformScalarQuantizer(bits)
    return np.linspace(-1.0, 1.0, quantiser.levels, dtype=np.float32)  # literal-ok: AM-96 fixed endpoints


__all__ = [
    "EncodedMessage",
    "ER9ProtocolHold",
    "FeatureFactorisation",
    "StaticEntropyModel",
    "UniformScalarQuantizer",
    "decode_message",
    "encode_message",
    "expected_level_values",
    "factorisation_for_dimension",
    "fit_static_entropy_model",
    "flatten_features",
    "protocol_metadata_bits",
    "raw_bound_bits",
    "unflatten_features",
]
