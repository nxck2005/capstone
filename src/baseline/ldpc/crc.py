"""Configured CRC16/CRC24A/CRC24B operations (TS 38.212 5.1)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from config.params import get


def _spec(name: str) -> dict:
    try:
        return get("baseline.crc_spec")[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported configured CRC: {name}") from exc


def remainder(bits: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    """Return the configured MSB-first polynomial remainder."""
    spec = _spec(name)
    width = int(spec["width"])
    poly = int(spec["poly_hex"], 16) & ((1 << width) - 1)  # literal-ok: hexadecimal radix
    register = int(spec["init"])
    mask = (1 << width) - 1
    for raw in np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits).reshape(-1):
        bit = int(raw)
        if bit not in (0, 1):
            raise ValueError("CRC input must contain only bits")
        feedback = ((register >> (width - 1)) & 1) ^ bit
        register = (register << 1) & mask
        if feedback:
            register ^= poly
    register ^= int(spec["final_xor"])
    return np.fromiter(
        ((register >> shift) & 1 for shift in range(width - 1, -1, -1)),
        dtype=np.uint8,
        count=width,
    )


def attach(bits: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    payload = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return np.concatenate((payload, remainder(payload, name)))


def check(codeword: Iterable[int] | np.ndarray, name: str) -> bool:
    word = np.asarray(codeword, dtype=np.uint8).reshape(-1)
    width = int(_spec(name)["width"])
    if word.size < width:
        return False
    return bool(np.array_equal(remainder(word[:-width], name), word[-width:]))


def strip(codeword: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    word = np.asarray(codeword, dtype=np.uint8).reshape(-1)
    if not check(word, name):
        raise ValueError(f"{name.upper()} failure")
    return word[: -int(_spec(name)["width"])]
