"""Executable packetisation solver and end-to-end transport integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config.params import get

from .adapter import SionnaLDPCAdapter
from .modulation import bits_per_symbol
from .rate_matching import distribute
from .segmentation import Segmentation, concatenate, plan, segment


@dataclass(frozen=True)
class PacketPlan:
    feasible: bool
    reason: str | None
    source_bytes: int | None
    nominal_rate: float
    nominal_rate_str: str
    q_m: int
    channel_bits: int
    clamped: bool
    segmentation: Segmentation | None
    e_r: tuple[int, ...]

    def metadata(self) -> dict:
        result = asdict(self)
        layout = result.pop("segmentation")
        if not self.feasible:
            return result
        assert layout is not None
        result.update({
            "A": layout["payload_bits"],
            "tb_crc_type": layout["tb_crc_name"],
            "tb_crc_bits": layout["tb_crc_bits"],
            "B": layout["payload_bits"] + layout["tb_crc_bits"],
            "base_graph": layout["base_graph"],
            "code_block_max_bits": get("baseline.code_block_max_bits")[f"bg{layout['base_graph']}"],
            "num_codeblocks": layout["code_blocks"],
            "cb_crc_bits_total": layout["code_blocks"] * layout["code_block_crc_bits"],
            "B_prime": layout["b_prime"],
            "K_prime": layout["k_prime"],
            "K_b_for_lifting": layout["k_b"],
            "lifting_size": layout["lifting_size"],
            "K": layout["k"],
            "filler_bits_per_block": layout["filler_bits_per_block"],
            "filler_bits_total": layout["filler_bits_per_block"] * layout["code_blocks"],
            "E": list(self.e_r),
            "E_sum": sum(self.e_r),
            "B_nominal": int(self.channel_bits * self.nominal_rate),
            "min_block_code_rate": layout["k_prime"] / max(self.e_r),
            "max_block_code_rate": layout["k_prime"] / min(self.e_r),
            "effective_code_rate": round(layout["k_prime"] / max(self.e_r), 6),  # literal-ok: evidence display precision
        })
        return result


def _minimum_rate(bg: int) -> float:
    return float(get("baseline.ldpc_bg1_min_coderate") if bg == 1 else get("baseline.bg2_min_coderate"))


def _candidate(payload_bits: int, channel_bits: int, q_m: int, rate: float) -> tuple[Segmentation, tuple[int, ...]] | None:
    layout = plan(payload_bits, rate)
    if layout is None:
        return None
    budgets = tuple(distribute(channel_bits, layout.code_blocks, q_m))
    if layout.k_prime / min(budgets) > 0.95:  # literal-ok: Sionna 2.0.1 structural maximum
        return None
    return layout, budgets


def build_packet_plan(k_symbols: int, modulation: str, nominal_rate: str) -> PacketPlan:
    q_m = bits_per_symbol(modulation)
    channel_bits = int(k_symbols) * q_m
    numerator, denominator = (int(value) for value in nominal_rate.split("/"))
    rate = numerator / denominator
    ceiling = None
    nominal_bits = int(channel_bits * rate)
    for payload_bits in range((nominal_bits // 8) * 8, 0, -8):  # literal-ok: octet framing step
        candidate = _candidate(payload_bits, channel_bits, q_m, rate)
        if candidate and payload_bits + candidate[0].tb_crc_bits <= nominal_bits:
            ceiling = candidate
            break
    if ceiling is None:
        return PacketPlan(False, "no_legal_byte_aligned_A_within_nominal_budget", None,
                          rate, nominal_rate, q_m, channel_bits, False, None, ())
    layout, budgets = ceiling
    if layout.k_prime / max(budgets) >= _minimum_rate(layout.base_graph) * (1 - 1e-9):
        return PacketPlan(True, None, layout.payload_bits // 8, rate, nominal_rate,  # literal-ok: bits per octet
                          q_m, channel_bits, False, layout, budgets)
    payload_bits = layout.payload_bits + 8  # literal-ok: octet framing step
    for _ in range(8192):  # literal-ok: bounded structural search, inherited from checked solver
        candidate = _candidate(payload_bits, channel_bits, q_m, rate)
        if candidate:
            layout, budgets = candidate
            if layout.k_prime / max(budgets) >= _minimum_rate(layout.base_graph) * (1 - 1e-9):
                return PacketPlan(True, None, layout.payload_bits // 8, rate, nominal_rate,  # literal-ok: bits per octet
                                  q_m, channel_bits, True, layout, budgets)
        payload_bits += 8  # literal-ok: octet framing step
    return PacketPlan(False, "min_coderate_clamp_did_not_converge", None,
                      rate, nominal_rate, q_m, channel_bits, False, None, ())


def transmit_transport(payload_bits: np.ndarray, packet: PacketPlan, device: str = "cpu") -> list[np.ndarray]:
    if not packet.feasible or packet.segmentation is None:
        raise ValueError("cannot transmit a structurally infeasible packet")
    blocks = segment(payload_bits, packet.segmentation)
    outputs = []
    for block, e_r in zip(blocks, packet.e_r, strict=True):
        adapter = SionnaLDPCAdapter(
            packet.segmentation.k, e_r, packet.q_m,
            packet.segmentation.base_graph, device,
        )
        if adapter.lifting_size != packet.segmentation.lifting_size:
            raise ValueError("Sionna selected a lifting size that differs from packetisation")
        outputs.append(adapter.encode(block[None, :])[0])
    if sum(value.size for value in outputs) != packet.channel_bits:
        raise AssertionError("encoded channel bits do not reconcile")
    return outputs


def receive_transport(llrs: list[np.ndarray], packet: PacketPlan, device: str = "cpu") -> np.ndarray:
    if not packet.feasible or packet.segmentation is None or len(llrs) != len(packet.e_r):
        raise ValueError("received blocks do not match packet plan")
    decoded = []
    for values, e_r in zip(llrs, packet.e_r, strict=True):
        if np.asarray(values).size != e_r:
            raise ValueError("LLR block length does not equal E_r")
        adapter = SionnaLDPCAdapter(
            packet.segmentation.k, e_r, packet.q_m,
            packet.segmentation.base_graph, device,
        )
        decoded.append(adapter.decode(np.asarray(values)[None, :])[0])
    return concatenate(decoded, packet.segmentation)
