"""Per-code-block rate-matching budgets and concatenation."""

from __future__ import annotations

import math


def distribute(total_bits: int, blocks: int, q_m: int, layers: int = 1) -> list[int]:
    unit = layers * q_m
    if total_bits <= 0 or blocks <= 0 or total_bits % unit:
        raise ValueError("rate-matching budget must be a positive multiple of N_L*Qm")
    gamma = (total_bits // unit) % blocks
    low = unit * ((total_bits // unit) // blocks)
    high = unit * math.ceil((total_bits // unit) / blocks)
    result = [low if index < blocks - gamma else high for index in range(blocks)]
    if sum(result) != total_bits or any(value % unit for value in result):
        raise AssertionError("rate-matching distribution lost bits")
    return result
