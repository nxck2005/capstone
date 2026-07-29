"""Configured BPSK/QPSK/16-QAM mapping, demapping and bit interleaving."""

from __future__ import annotations

import math

import numpy as np

from config.params import get

_MOD_NAMES = {"bpsk": 1, "qpsk": 2, "qam16": 4}  # literal-ok: modulation-order definitions


def bits_per_symbol(modulation: str) -> int:
    try:
        q_m = _MOD_NAMES[modulation]
    except KeyError as exc:
        raise ValueError(f"unsupported modulation: {modulation}") from exc
    if modulation not in get("baseline.modulations"):
        raise ValueError(f"modulation is not configured: {modulation}")
    return q_m


def interleaver_indices(length: int, q_m: int) -> np.ndarray:
    if q_m not in _MOD_NAMES.values() or length <= 0 or length % q_m:
        raise ValueError("interleaver length must be a positive multiple of Qm")
    columns = length // q_m
    return np.fromiter(
        (i * columns + j for j in range(columns) for i in range(q_m)),
        dtype=np.int64,
        count=length,
    )


def interleave(bits: np.ndarray, q_m: int) -> np.ndarray:
    source = np.asarray(bits)
    return source[..., interleaver_indices(source.shape[-1], q_m)]


def deinterleave(values: np.ndarray, q_m: int) -> np.ndarray:
    source = np.asarray(values)
    return source[..., np.argsort(interleaver_indices(source.shape[-1], q_m))]


def map_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    q_m = bits_per_symbol(modulation)
    source = np.asarray(bits, dtype=np.uint8)
    if source.shape[-1] % q_m or np.any(source > 1):
        raise ValueError("mapping input must be binary and divisible by Qm")
    grouped = source.reshape(*source.shape[:-1], source.shape[-1] // q_m, q_m)
    signed = 1.0 - 2.0 * grouped
    if q_m == 1:
        return signed[..., 0].astype(np.complex64)
    if q_m == 2:
        return ((signed[..., 0] + 1j * signed[..., 1]) / math.sqrt(2)).astype(np.complex64)
    real = signed[..., 0] * (2.0 - signed[..., 2])
    imag = signed[..., 1] * (2.0 - signed[..., 3])
    return ((real + 1j * imag) / math.sqrt(10)).astype(np.complex64)  # literal-ok: TS 38.211 16-QAM normalization


def constellation(modulation: str) -> tuple[np.ndarray, np.ndarray]:
    q_m = bits_per_symbol(modulation)
    labels = np.array(
        [[(value >> shift) & 1 for shift in range(q_m - 1, -1, -1)]
         for value in range(1 << q_m)],
        dtype=np.uint8,
    )
    return labels, map_bits(labels.reshape(-1), modulation)


def max_log_llr(symbols: np.ndarray, modulation: str, n0: float) -> np.ndarray:
    """Return log P(bit=1)/P(bit=0), the configured Sionna convention."""
    if n0 <= 0:
        raise ValueError("N0 must be positive")
    q_m = bits_per_symbol(modulation)
    labels, points = constellation(modulation)
    received = np.asarray(symbols)
    distances = np.abs(received[..., None] - points) ** 2
    outputs = []
    for bit_index in range(q_m):
        d0 = distances[..., labels[:, bit_index] == 0].min(axis=-1)
        d1 = distances[..., labels[:, bit_index] == 1].min(axis=-1)
        outputs.append((d0 - d1) / n0)
    return np.stack(outputs, axis=-1).reshape(*received.shape[:-1], -1).astype(np.float32)


def esn0_from_ebn0_db(ebn0_db: float, rate: float, q_m: int) -> float:
    if rate <= 0 or q_m not in _MOD_NAMES.values():
        raise ValueError("invalid rate or Qm")
    return float(ebn0_db + 10.0 * math.log10(rate * q_m))  # literal-ok: power-decibel definition


def n0_from_esn0_db(esn0_db: float) -> float:
    return float(10.0 ** (-esn0_db / 10.0))  # literal-ok: inverse power-decibel definition


def realised_symbol_energy(symbols: np.ndarray) -> float:
    values = np.asarray(symbols)
    if values.size == 0:
        raise ValueError("cannot measure an empty packet")
    return float(np.mean(np.abs(values) ** 2))
