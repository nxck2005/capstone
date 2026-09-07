"""Batched ER-9 transport adapter over the existing project PHY.

The batching here changes only execution granularity.  Packetisation, the
Sionna LDPC adapter, constellation mapper/demapper, shared AWGN and CRC
semantics are the existing baseline implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from baseline.classical.channel_transport import build_accounting, modulate
from baseline.ldpc import crc
from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import max_log_llr, n0_from_esn0_db
from baseline.ldpc.segmentation import segment
from baseline.ldpc.transport import PacketPlan
from channels.awgn import AWGN, keyed_complex_noise
from channels.registry import build_channel
from config.params import get


@dataclass(frozen=True)
class TransportBatchResult:
    payloads: tuple[np.ndarray | None, ...]
    crc_ok: tuple[bool, ...]
    symbols_per_packet: int
    realised_symbol_energy: tuple[float, ...]


class ER9TransportBatch:
    """One cached packet plan's exact baseline physical chain."""

    def __init__(self, packet: PacketPlan, *, device: str) -> None:
        if not packet.feasible or packet.segmentation is None:
            raise ValueError("ER-9 transport requires a feasible packet")
        accounting = build_accounting(packet)
        if not accounting.reconciles:
            raise ValueError("ER-9 packet accounting does not reconcile")
        self.packet = packet
        self.device = device
        self.accounting = accounting
        self.modulation = accounting.modulation
        self._adapters = tuple(
            SionnaLDPCAdapter(
                packet.segmentation.k_prime,
                e_r,
                packet.q_m,
                packet.segmentation.base_graph,
                device,
            )
            for e_r in packet.e_r
        )
        if any(
            adapter.lifting_size != packet.segmentation.lifting_size
            for adapter in self._adapters
        ):
            raise RuntimeError("ER-9 Sionna lifting size differs from packet plan")
        channel = build_channel("awgn")
        if not isinstance(channel, AWGN):
            raise RuntimeError("ER-9 must use the shared AWGN implementation")
        channel.eval()
        self.channel = channel

    def _encode(self, payloads: Sequence[np.ndarray]) -> np.ndarray:
        layout = self.packet.segmentation
        assert layout is not None
        segmented = [segment(payload, layout) for payload in payloads]
        encoded_blocks = []
        for block_index, adapter in enumerate(self._adapters):
            source = np.stack(
                [row[block_index][: layout.k_prime] for row in segmented], axis=0
            )
            encoded_blocks.append(np.asarray(adapter.encode(source), dtype=np.uint8))
        return np.concatenate(encoded_blocks, axis=1)

    def _decode(self, llrs: np.ndarray) -> tuple[tuple[np.ndarray | None, ...], tuple[bool, ...]]:
        layout = self.packet.segmentation
        assert layout is not None
        decoded_blocks = []
        offset = 0
        for adapter, e_r in zip(self._adapters, self.packet.e_r, strict=True):
            block_llrs = llrs[:, offset : offset + e_r]
            if block_llrs.shape[1] != e_r:
                raise ValueError("ER-9 LLR block length differs from packet plan")
            decoded_blocks.append(np.asarray(adapter.decode(block_llrs), dtype=np.uint8))
            offset += e_r
        if offset != llrs.shape[1]:
            raise ValueError("ER-9 LLR blocks do not cover the packet")

        cb_name = get("baseline.cb_crc_polynomial")
        cb_width = int(get("baseline.crc_spec")[cb_name]["width"])
        payloads: list[np.ndarray | None] = []
        verdicts: list[bool] = []
        for row in range(llrs.shape[0]):
            restored: list[np.ndarray] = []
            cb_ok = True
            for block_index, block in enumerate(decoded_blocks):
                data = np.asarray(block[row], dtype=np.uint8).reshape(-1)[: layout.k_prime]
                if layout.code_blocks > 1:
                    this_ok = bool(crc.check(data, cb_name))
                    cb_ok = cb_ok and this_ok
                    data = data[:-cb_width]
                restored.append(data)
            transport = np.concatenate(restored)
            tb_ok = bool(crc.check(transport, layout.tb_crc_name))
            ok = cb_ok and tb_ok
            verdicts.append(ok)
            payloads.append(transport[: layout.payload_bits] if ok else None)
        return tuple(payloads), tuple(verdicts)

    def round_trip(
        self,
        payloads: Sequence[np.ndarray],
        *,
        snr_db: float,
        noise_ids: Sequence[str],
    ) -> TransportBatchResult:
        if len(payloads) == 0 or len(payloads) != len(noise_ids):
            raise ValueError("ER-9 payload and noise batches must be non-empty and paired")
        layout = self.packet.segmentation
        assert layout is not None
        source = [np.asarray(payload, dtype=np.uint8).reshape(-1) for payload in payloads]
        if any(values.size != layout.payload_bits for values in source):
            raise ValueError("ER-9 payload length differs from packet plan")
        encoded = self._encode(source)
        symbols = np.stack(
            [modulate([row], self.modulation) for row in encoded], axis=0
        ).astype(np.complex64)
        if symbols.shape[1] != self.accounting.k_symbols:
            raise ValueError("ER-9 symbol count differs from packet plan")
        transmitted = torch.as_tensor(symbols, dtype=torch.complex64, device=self.device)
        unit_noise = keyed_complex_noise(
            tuple(str(value) for value in noise_ids),
            self.accounting.k_symbols,
            dtype=torch.complex64,
            device=self.device,
        )
        received = self.channel(transmitted, float(snr_db), unit_noise=unit_noise)
        llrs = max_log_llr(
            received.detach().cpu().numpy(),
            self.modulation,
            n0_from_esn0_db(float(snr_db)),
        )
        decoded, verdicts = self._decode(llrs)
        energy = tuple(float(np.mean(np.abs(row) ** 2)) for row in symbols)  # literal-ok: packet energy statistic
        return TransportBatchResult(
            payloads=decoded,
            crc_ok=verdicts,
            symbols_per_packet=int(symbols.shape[1]),
            realised_symbol_energy=energy,
        )


__all__ = ["ER9TransportBatch", "TransportBatchResult"]
