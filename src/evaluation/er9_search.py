"""Deterministic AM-96 ER-9 feasibility and two-stage search policy."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from baseline.ldpc.transport import PacketPlan, build_packet_plan
from baseline.ldpc.modulation import bits_per_symbol
from config.params import get
from evaluation.er9_protocol import factorisation_for_dimension, raw_bound_bits


@dataclass(frozen=True)
class SearchCandidate:
    transmit_dim: int
    quantiser_bits: int

    def as_dict(self) -> dict[str, int]:
        return {
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
        }


@dataclass(frozen=True)
class PacketFloor:
    k_symbols: int
    modulation: str
    nominal_rate: str
    payload_bits: int
    source_bytes: int
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_symbols": self.k_symbols,
            "modulation": self.modulation,
            "nominal_rate": self.nominal_rate,
            "payload_bits": self.payload_bits,
            "source_bytes": self.source_bytes,
            "metadata": dict(self.metadata),
        }


def packetisation_floor(k_symbols: int) -> PacketFloor:
    """Solve the exact conservative BPSK/rate-1/3 packet payload."""

    modulation = "bpsk"
    nominal_rate = "1/3"
    packet = build_packet_plan(k_symbols, modulation, nominal_rate)
    if not packet.feasible:
        raise RuntimeError(f"ER-9 packet floor is infeasible: {packet.reason}")
    metadata = packet.metadata()
    return PacketFloor(
        k_symbols=int(k_symbols),
        modulation=modulation,
        nominal_rate=nominal_rate,
        payload_bits=int(metadata["A"]),
        source_bytes=int(packet.source_bytes),
        metadata=metadata,
    )


def all_configured_pairs() -> tuple[SearchCandidate, ...]:
    dimensions = tuple(int(value) for value in get("digital_semantic_control.transmit_dim_grid"))
    widths = tuple(int(value) for value in get("digital_semantic_control.quantiser_bits_grid"))
    pairs = []
    for dimension in dimensions:
        factorisation_for_dimension(dimension)
        for width in widths:
            raw_bound_bits(dimension, width)
            pairs.append(SearchCandidate(dimension, width))
    return tuple(pairs)


def feasible_pairs(payload_bits: int) -> tuple[SearchCandidate, ...]:
    """Return every configured pair satisfying the proven raw bound."""

    return tuple(
        candidate
        for candidate in all_configured_pairs()
        if raw_bound_bits(candidate.transmit_dim, candidate.quantiser_bits) <= payload_bits
    )


def stage1_candidates(payload_bits: int) -> tuple[SearchCandidate, ...]:
    minimum_width = int(get("digital_semantic_control.stage1_quantiser_bits"))
    if minimum_width != 2:  # literal-ok: AM-96 stage-1 minimum quantiser width
        raise ValueError("AM-96 stage 1 must use two quantiser bits")
    candidates = tuple(
        candidate
        for candidate in feasible_pairs(payload_bits)
        if candidate.quantiser_bits == minimum_width
    )
    if tuple(candidate.transmit_dim for candidate in candidates) != tuple(
        sorted(candidate.transmit_dim for candidate in candidates)
    ):
        raise RuntimeError("ER-9 stage-1 dimensions are not ascending")
    return candidates


def stage2_candidates(payload_bits: int, selected_dimension: int) -> tuple[SearchCandidate, ...]:
    factorisation_for_dimension(selected_dimension)
    candidates = tuple(
        candidate
        for candidate in feasible_pairs(payload_bits)
        if candidate.transmit_dim == selected_dimension
    )
    if tuple(candidate.quantiser_bits for candidate in candidates) != tuple(
        sorted(candidate.quantiser_bits for candidate in candidates)
    ):
        raise RuntimeError("ER-9 stage-2 widths are not ascending")
    return candidates


def _score_value(row: Mapping[str, Any]) -> int:
    value = row.get("n_correct")
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("ER-9 search n_correct must be an exact integer")
    return value


def select_stage1(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select highest exact validation count, then smallest dimension."""

    if not rows:
        raise ValueError("ER-9 stage 1 has no candidate results")
    ordered = sorted(
        rows,
        key=lambda row: (-_score_value(row), int(row["transmit_dim"])),
    )
    if len({int(row["transmit_dim"]) for row in ordered}) != len(ordered):
        raise ValueError("ER-9 stage-1 result rows duplicate a dimension")
    return ordered[0]


def select_stage2(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select highest exact validation count, then smallest bit width."""

    if not rows:
        raise ValueError("ER-9 stage 2 has no candidate results")
    ordered = sorted(
        rows,
        key=lambda row: (-_score_value(row), int(row["quantiser_bits"])),
    )
    if len({int(row["quantiser_bits"]) for row in ordered}) != len(ordered):
        raise ValueError("ER-9 stage-2 result rows duplicate a quantiser width")
    return ordered[0]


def configured_phy_candidates(k_symbols: int) -> tuple[tuple[str, str, PacketPlan], ...]:
    """Return configured modulation/rate packet plans in canonical order."""

    result = []
    for modulation in get("baseline.modulations"):
        bits_per_symbol(str(modulation))
        for rate in get("baseline.ldpc_rates"):
            packet = build_packet_plan(k_symbols, str(modulation), str(rate))
            if packet.feasible:
                result.append((str(modulation), str(rate), packet))
    return tuple(result)


def phy_tie_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deterministic physical-PHY tie rule, independent of validation floats."""

    rate = Fraction(str(row["ldpc_rate"]))
    return (
        -_score_value(row),
        bits_per_symbol(str(row["modulation"])),
        rate,
        str(row["modulation"]),
        str(row["ldpc_rate"]),
    )


def select_phy(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("ER-9 physical transport selection has no candidates")
    return sorted(rows, key=phy_tie_key)[0]


__all__ = [
    "PacketFloor",
    "SearchCandidate",
    "all_configured_pairs",
    "configured_phy_candidates",
    "feasible_pairs",
    "packetisation_floor",
    "phy_tie_key",
    "select_phy",
    "select_stage1",
    "select_stage2",
    "stage1_candidates",
    "stage2_candidates",
]
