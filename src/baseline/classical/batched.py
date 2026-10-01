"""Batched classical arm: the per-image pipeline's exact semantics, batched at the LDPC.

``run_classical_pipeline`` builds a fresh Sionna encoder and decoder for every
code block of every image and decodes one packet at a time, which dominates the
classical cost (about 1–4 s per image per SNR in W10).  Within one evaluation
unit every image shares a packet plan, so the adapters can be built once and the
packets encoded and decoded as one batch.  Nothing else changes: the source
coding, payload filler, keyed noise identity, shared AWGN, demapper, CRC rules,
codestream recovery and verdicts are the per-image ones, and
``tests/test_classical_batched.py`` plus the G-12 equivalence check against the
committed W10 rows hold the two paths to identical outputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from baseline.classical.channel_transport import (
    TransportOutcome,
    _require_interleaver,
    _shared_channel,
    build_accounting,
    demodulate,
    modulate,
)
from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    ChannelIdentity,
    ClassicalPipelineError,
    ClassicalResult,
    _encode_source,
    _strip_payload_filler,
)
from baseline.classical.jpeg_pipeline import jpeg_source_coding
from baseline.j2k import J2KCodec
from baseline.jpeg import JpegCodec, decode_codestream as decode_jpeg_codestream
from baseline.ldpc import crc
from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import n0_from_esn0_db, realised_symbol_energy
from baseline.ldpc.segmentation import segment
from baseline.ldpc.transport import PacketPlan, ReceivedTransport, build_packet_plan
from channels.awgn import keyed_complex_noise
from channels.power import symbol_papr_db
from config.params import get
from data.preprocessing import CanonicalProduct, codec_input, codec_upsample

_CHANNEL_MODEL = "awgn"


class BatchedTransport:
    """One packet plan's adapters, built once, driving whole batches of packets."""

    def __init__(self, packet: PacketPlan, *, device: str) -> None:
        if not packet.feasible or packet.segmentation is None:
            raise ValueError("batched transport requires a feasible packet")
        self.packet = packet
        self.layout = packet.segmentation
        self.device = device
        self.accounting = build_accounting(packet)
        if not self.accounting.reconciles:
            raise AssertionError(f"packet accounting does not reconcile: {self.accounting}")
        self.adapters = tuple(
            SionnaLDPCAdapter(self.layout.k_prime, e_r, packet.q_m, self.layout.base_graph, device)
            for e_r in packet.e_r
        )
        for adapter in self.adapters:
            if adapter.lifting_size != self.layout.lifting_size:
                raise ValueError("Sionna selected a lifting size that differs from packetisation")
        self.channel = _shared_channel()

    def _encode(self, payloads: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        segmented = [segment(payload, self.layout) for payload in payloads]
        per_block = []
        for index, adapter in enumerate(self.adapters):
            source = np.stack([row[index][: self.layout.k_prime] for row in segmented], axis=0)
            per_block.append(adapter.encode(source))
        encoded = [[per_block[block][row] for block in range(len(self.adapters))] for row in range(len(payloads))]
        for blocks in encoded:
            if sum(block.size for block in blocks) != self.packet.channel_bits:
                raise AssertionError("encoded channel bits do not reconcile")
        return encoded

    def _decode(self, llr_rows: Sequence[list[np.ndarray]]) -> list[ReceivedTransport]:
        layout = self.layout
        decoded_blocks = []
        for index, (adapter, e_r) in enumerate(zip(self.adapters, self.packet.e_r, strict=True)):
            stacked = np.stack([np.asarray(row[index]) for row in llr_rows], axis=0)
            if stacked.shape[1] != e_r:
                raise ValueError("LLR block length does not equal E_r")
            decoded_blocks.append(adapter.decode(stacked))
        cb_name = get("baseline.cb_crc_polynomial")
        cb_width = int(get("baseline.crc_spec")[cb_name]["width"])
        received = []
        for row in range(len(llr_rows)):
            code_block_crc_ok: list[bool] = []
            restored = []
            for block in decoded_blocks:
                data = np.asarray(block[row], dtype=np.uint8).reshape(-1)[: layout.k_prime]
                if layout.code_blocks > 1:
                    code_block_crc_ok.append(bool(crc.check(data, cb_name)))
                    data = data[:-cb_width]
                restored.append(data)
            transport = np.concatenate(restored)
            tb_crc_ok = bool(crc.check(transport, layout.tb_crc_name))
            crc_ok = tb_crc_ok and all(code_block_crc_ok)
            received.append(ReceivedTransport(
                crc_ok=crc_ok,
                tb_crc_ok=tb_crc_ok,
                code_block_crc_ok=tuple(code_block_crc_ok),
                payload_bits=transport[: layout.payload_bits] if crc_ok else None,
            ))
        return received

    def round_trip(self, payloads: Sequence[np.ndarray], *, snr_db: float, noise_ids: Sequence[str]) -> list[TransportOutcome]:
        """``transport_round_trip`` for a batch of packets sharing this plan."""

        if len(payloads) != len(noise_ids):
            raise ValueError("one noise identity is required per packet")
        if not payloads:
            return []
        accounting = self.accounting
        modulation = accounting.modulation
        encoded = self._encode(payloads)
        symbols = [modulate(blocks, modulation) for blocks in encoded]
        for row in symbols:
            if row.size != accounting.k_symbols:
                raise AssertionError("modulated symbol count differs from the exact k")
        transmitted = torch.as_tensor(np.stack(symbols, axis=0), dtype=torch.complex64)
        energies = [realised_symbol_energy(row) for row in symbols]
        paprs = [float(value) for value in symbol_papr_db(transmitted)]
        noise = torch.cat([keyed_complex_noise(noise_id, accounting.k_symbols, dtype=transmitted.dtype) for noise_id in noise_ids], dim=0)
        received = self.channel(transmitted, float(snr_db), unit_noise=noise)
        n0 = n0_from_esn0_db(float(snr_db))
        llr_rows = [demodulate(received[row].reshape(-1).numpy(), modulation, n0, accounting.rate_matched_bits) for row in range(len(payloads))]
        decoded = self._decode(llr_rows)
        interleaver = _require_interleaver()
        return [
            TransportOutcome(
                accounting=accounting,
                snr_db=float(snr_db),
                n0=n0,
                channel_model=_CHANNEL_MODEL,
                noise_id=noise_ids[row],
                realised_symbol_energy=energies[row],
                papr_db=paprs[row],
                per_packet_power_rescaling_applied=False,
                interleaver=interleaver,
                unit_noise_sha256=hashlib.sha256(noise[row : row + 1].numpy().tobytes()).hexdigest(),
                received=decoded[row],
            )
            for row in range(len(payloads))
        ]


def _result(product: CanonicalProduct, **fields: Any) -> ClassicalResult:
    return ClassicalResult(stable_sample_id=product.stable_sample_id, **fields)


def run_classical_batch(
    products: Sequence[CanonicalProduct],
    *,
    dataset: str,
    k_symbols: int,
    modulation: str,
    ldpc_rate: str,
    snr_db: float,
    codec: J2KCodec | JpegCodec,
    channel_identity: ChannelIdentity | Mapping[str, Any],
    encode_axis_px: int | None = None,
    block_index: int = 0,
    device: str = "cpu",
    transport: BatchedTransport | None = None,
    batch_size: int = 256,  # literal-ok: decode batch size; outputs are batch-invariant
    codec_kind: str = "jpeg2000",
    quality: int | None = None,
) -> list[ClassicalResult]:
    """``run_classical_pipeline`` (or ``run_jpeg_pipeline``) over many images, in order."""

    if codec_kind not in ("jpeg2000", "jpeg"):
        raise ValueError(f"unsupported batched codec: {codec_kind}")
    if codec_kind == "jpeg":
        if quality is None or int(quality) not in codec.quality_grid:
            raise ClassicalPipelineError("JPEG execution requires its frozen configured quality")
    elif ldpc_rate not in get("baseline.ldpc_rates"):
        raise ValueError(f"unconfigured LDPC rate: {ldpc_rate}")
    if isinstance(channel_identity, Mapping):
        channel_identity = ChannelIdentity(**channel_identity)
    common = {"dataset": dataset, "k_symbols": k_symbols, "modulation": modulation, "ldpc_rate": ldpc_rate, "snr_db": float(snr_db)}
    packet = build_packet_plan(k_symbols, modulation, ldpc_rate)
    if not packet.feasible:
        return [
            _result(product, verdict=STRUCTURAL_INFEASIBILITY, noise_id=None, packet_feasible=False,
                    structural_reason=packet.reason, accounting=None, source_coding=None, transport=None,
                    codestream_recovered_exactly=None, decoded_image=None, **common)
            for product in products
        ]
    accounting = build_accounting(packet)
    if accounting.k_symbols != k_symbols:
        raise ClassicalPipelineError("packet plan does not carry the requested k")
    if transport is None:
        transport = BatchedTransport(packet, device=device)
    elif transport.packet != packet:
        raise ValueError("cached batched transport belongs to a different packet plan")

    results: list[ClassicalResult | None] = [None] * len(products)
    pending: list[tuple[int, Any, Any, np.ndarray, str, np.ndarray]] = []
    for index, product in enumerate(products):
        canonical_image = codec_input(product)
        if codec_kind == "jpeg":
            source_coding, j2k = jpeg_source_coding(
                canonical_image,
                dataset=dataset,
                accounting=accounting,
                quality=int(quality),
                codec=codec,
                encode_axis_px=encode_axis_px,
            )
        else:
            source_coding, j2k, _ = _encode_source(
                codec=codec,
                canonical_image=canonical_image,
                canonical_pixels_sha256=hashlib.sha256(canonical_image.tobytes()).hexdigest(),
                dataset=dataset,
                payload_capacity_bytes=accounting.payload_bytes,
                encode_axis_px=encode_axis_px,
            )
        if not source_coding.feasible or j2k is None or j2k.codestream is None:
            results[index] = _result(product, verdict=CODEC_INFEASIBILITY, noise_id=None, packet_feasible=True,
                                     structural_reason=None, accounting=accounting, source_coding=source_coding,
                                     transport=None, codestream_recovered_exactly=None, decoded_image=None, **common)
            continue
        assert source_coding.payload_filler_bytes is not None
        payload_bytes = j2k.codestream + b"\x00" * source_coding.payload_filler_bytes
        if len(payload_bytes) != accounting.payload_bytes:
            raise ClassicalPipelineError("padded payload does not fill the transport block")
        payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        if payload_bits.size != accounting.payload_bits:
            raise ClassicalPipelineError("payload bit count does not match the packet plan")
        noise_id = channel_identity.noise_id(
            stable_sample_id=product.stable_sample_id,
            test_snr_db=float(snr_db),
            k=k_symbols,
            block_index=block_index,
        )
        pending.append((index, source_coding, j2k, payload_bits, noise_id, canonical_image))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        payloads = [payload_bits for _index, _coding, _j2k, payload_bits, _noise, _image in chunk]
        noise_ids = [noise_id for _index, _coding, _j2k, _bits, noise_id, _image in chunk]
        outcomes = transport.round_trip(payloads, snr_db=float(snr_db), noise_ids=noise_ids)
        for (index, source_coding, j2k, _bits, noise_id, canonical_image), outcome in zip(chunk, outcomes, strict=True):
            product = products[index]
            if not outcome.crc_ok or outcome.payload_bits is None:
                results[index] = _result(product, verdict=DECODE_FAILURE, noise_id=noise_id, packet_feasible=True,
                                         structural_reason=None, accounting=accounting, source_coding=source_coding,
                                         transport=outcome, codestream_recovered_exactly=None, decoded_image=None, **common)
                continue
            received_payload = np.packbits(outcome.payload_bits).tobytes()
            if codec_kind == "jpeg":
                # run_jpeg_pipeline decodes the whole padded payload and reports exact recovery.
                decoded = decode_jpeg_codestream(received_payload)
                exact = True
            else:
                recovered = _strip_payload_filler(received_payload)
                if recovered is None:
                    raise ClassicalPipelineError("CRC-clean payload carries no JPEG 2000 end-of-codestream marker")
                decoded = codec.decode_codestream(recovered)
                exact = recovered == j2k.codestream
            restored = codec_upsample(decoded, tuple(int(v) for v in canonical_image.shape[:2]))
            results[index] = _result(product, verdict=DELIVERED, noise_id=noise_id, packet_feasible=True,
                                     structural_reason=None, accounting=accounting, source_coding=source_coding,
                                     transport=outcome, codestream_recovered_exactly=exact,
                                     decoded_image=restored, **common)
    if any(result is None for result in results):
        raise AssertionError("batched classical arm left an image without a verdict")
    return list(results)  # type: ignore[arg-type]


__all__ = ["BatchedTransport", "run_classical_batch"]
