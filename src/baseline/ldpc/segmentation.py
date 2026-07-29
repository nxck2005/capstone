"""Transport-block CRC, base-graph selection and code-block segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config.params import get

from . import crc


@dataclass(frozen=True)
class Segmentation:
    payload_bits: int
    tb_crc_name: str
    tb_crc_bits: int
    base_graph: int
    code_blocks: int
    code_block_crc_bits: int
    b_prime: int
    k_prime: int
    k_b: int
    lifting_size: int
    k: int
    filler_bits_per_block: int


def tb_crc_name(payload_bits: int) -> str:
    cfg = get("baseline.tb_crc")
    return cfg["large_polynomial"] if payload_bits > cfg["threshold_payload_bits"] else cfg["small_polynomial"]


def select_base_graph(payload_bits: int, rate: float) -> int:
    t = get("baseline.ldpc_base_graph_thresholds")
    if (payload_bits <= t["small_payload_bits"]
            or (payload_bits <= t["medium_payload_bits"] and rate <= t["medium_max_rate"])
            or rate <= t["robust_max_rate"]):
        return 2
    return 1


def select_kb(base_graph: int, b_bits: int) -> int:
    if base_graph == 1:
        return 22  # literal-ok: TS 38.212 BG1 K_b
    t = get("baseline.ldpc_bg2_kb_thresholds")
    if b_bits > t["kb10_above_bits"]:
        return 10  # literal-ok: TS 38.212 BG2 K_b branch
    if b_bits > t["kb9_above_bits"]:
        return 9  # literal-ok: TS 38.212 BG2 K_b branch
    if b_bits > t["kb8_above_bits"]:
        return 8  # literal-ok: TS 38.212 BG2 K_b branch
    return 6  # literal-ok: TS 38.212 BG2 K_b branch


def plan(payload_bits: int, rate: float) -> Segmentation | None:
    if payload_bits <= 0 or payload_bits % 8:  # literal-ok: octet framing definition
        return None
    name = tb_crc_name(payload_bits)
    tb_width = int(get("baseline.crc_spec")[name]["width"])
    b_bits = payload_bits + tb_width
    bg = select_base_graph(payload_bits, rate)
    max_bits = int(get("baseline.code_block_max_bits")[f"bg{bg}"])
    cb_width = int(get("baseline.cb_crc_bits"))
    if b_bits <= max_bits:
        blocks, b_prime, per_cb_crc = 1, b_bits, 0
    else:
        blocks = math.ceil(b_bits / (max_bits - cb_width))
        b_prime, per_cb_crc = b_bits + blocks * cb_width, cb_width
    if b_prime % blocks:
        return None
    k_prime = b_prime // blocks
    k_b = select_kb(bg, b_bits)
    z = next((value for value in get("baseline.ldpc_lifting_sizes") if k_b * value >= k_prime), None)
    if z is None:
        return None
    k = (22 if bg == 1 else 10) * z  # literal-ok: TS 38.212 systematic-column counts
    if k < k_prime:
        return None
    return Segmentation(
        payload_bits, name, tb_width, bg, blocks, per_cb_crc, b_prime, k_prime,
        k_b, z, k, k - k_prime,
    )


def segment(payload: np.ndarray, layout: Segmentation) -> list[np.ndarray]:
    source = np.asarray(payload, dtype=np.uint8).reshape(-1)
    if source.size != layout.payload_bits:
        raise ValueError("payload length does not match segmentation plan")
    transport = crc.attach(source, layout.tb_crc_name)
    if layout.code_blocks == 1:
        chunks = [transport]
    else:
        chunks = [
            crc.attach(chunk, get("baseline.cb_crc_polynomial"))
            for chunk in np.split(transport, layout.code_blocks)
        ]
    filler = np.zeros(layout.filler_bits_per_block, dtype=np.uint8)
    return [np.concatenate((chunk, filler)) for chunk in chunks]


def concatenate(blocks: list[np.ndarray], layout: Segmentation) -> np.ndarray:
    if len(blocks) != layout.code_blocks:
        raise ValueError("wrong number of decoded code blocks")
    restored = []
    for block in blocks:
        data = np.asarray(block, dtype=np.uint8).reshape(-1)[:layout.k_prime]
        if layout.code_blocks > 1:
            data = crc.strip(data, get("baseline.cb_crc_polynomial"))
        restored.append(data)
    return crc.strip(np.concatenate(restored), layout.tb_crc_name)
