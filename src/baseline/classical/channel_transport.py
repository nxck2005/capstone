"""Bits to symbols to AWGN to LLRs to bits, with exact packet accounting.

This module owns only the channel-facing half of the classical arm.  It never
constructs a channel of its own: it builds the *shared* project channel through
``channels.registry.build_channel``, and it draws noise only from the keyed
``channels.awgn.keyed_complex_noise`` counter-based stream, so the classical and
learned arms observe the same realisation for the same noise identity.

Two rules here are prohibitions rather than features:

* per-packet power rescaling is forbidden (``baseline.per_packet_power_rescaling_permitted``).
  The only normalisation applied is the fixed constellation normalisation, so the
  realised per-packet symbol energy is a *measurement*, not a constant;
* the modulation bit interleaver (``baseline.modulation_bit_interleaver``) is
  mandatory whenever ``baseline.modulation_bit_interleaver_required`` holds, and
  it is applied per code block, before mapping.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from baseline.ldpc.modulation import (
    bits_per_symbol,
    deinterleave,
    interleave,
    map_bits,
    max_log_llr,
    n0_from_esn0_db,
    realised_symbol_energy,
)
from baseline.ldpc.transport import (
    PacketPlan,
    ReceivedTransport,
    receive_transport_verified,
    transmit_transport,
)
from channels.awgn import AWGN, keyed_complex_noise
from channels.power import symbol_papr_db
from channels.registry import build_channel
from config.params import get

_CHANNEL_MODEL = "awgn"


@dataclass(frozen=True)
class TransportAccounting:
    """Every bit the packet plan promises, counted and reconciled."""

    k_symbols: int
    q_m: int
    modulation: str
    nominal_rate: str
    channel_bits: int
    payload_bits: int
    payload_bytes: int
    tb_crc_name: str
    tb_crc_bits: int
    code_blocks: int
    cb_crc_bits_per_block: int
    cb_crc_bits_total: int
    base_graph: int
    lifting_size: int
    k_prime: int
    systematic_bits_per_block: int
    ldpc_filler_bits_per_block: int
    ldpc_filler_bits_total: int
    systematic_bits_total: int
    rate_matched_bits: tuple[int, ...]
    rate_matched_bits_total: int
    channel_uses_exact: bool
    channel_bits_equal_k_times_qm: bool
    systematic_reconciles: bool
    channel_reconciles: bool
    reconciles: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TransportOutcome:
    """One packet's realised transmission, measurements and CRC verdicts."""

    accounting: TransportAccounting
    snr_db: float
    n0: float
    channel_model: str
    noise_id: str
    realised_symbol_energy: float
    papr_db: float
    per_packet_power_rescaling_applied: bool
    interleaver: str
    unit_noise_sha256: str
    received: ReceivedTransport

    @property
    def crc_ok(self) -> bool:
        return self.received.crc_ok

    @property
    def payload_bits(self) -> np.ndarray | None:
        return self.received.payload_bits


def build_accounting(packet: PacketPlan) -> TransportAccounting:
    """Reconcile a feasible packet plan's bits exactly, or raise."""

    if not packet.feasible or packet.segmentation is None:
        raise ValueError("cannot account for a structurally infeasible packet")
    layout = packet.segmentation
    bits_per_byte = np.iinfo(np.uint8).bits
    cb_crc_total = layout.code_blocks * layout.code_block_crc_bits
    filler_total = layout.code_blocks * layout.filler_bits_per_block
    systematic_total = layout.code_blocks * layout.k
    rate_matched_total = int(sum(packet.e_r))

    channel_uses_exact = packet.channel_bits % packet.q_m == 0
    k_symbols = packet.channel_bits // packet.q_m
    channel_bits_equal_k_times_qm = k_symbols * packet.q_m == packet.channel_bits
    systematic_reconciles = (
        layout.payload_bits + layout.tb_crc_bits + cb_crc_total + filler_total
        == systematic_total
    )
    channel_reconciles = rate_matched_total == packet.channel_bits

    return TransportAccounting(
        k_symbols=k_symbols,
        q_m=packet.q_m,
        modulation=_modulation_of(packet.q_m),
        nominal_rate=packet.nominal_rate_str,
        channel_bits=packet.channel_bits,
        payload_bits=layout.payload_bits,
        payload_bytes=layout.payload_bits // bits_per_byte,
        tb_crc_name=layout.tb_crc_name,
        tb_crc_bits=layout.tb_crc_bits,
        code_blocks=layout.code_blocks,
        cb_crc_bits_per_block=layout.code_block_crc_bits,
        cb_crc_bits_total=cb_crc_total,
        base_graph=layout.base_graph,
        lifting_size=layout.lifting_size,
        k_prime=layout.k_prime,
        systematic_bits_per_block=layout.k,
        ldpc_filler_bits_per_block=layout.filler_bits_per_block,
        ldpc_filler_bits_total=filler_total,
        systematic_bits_total=systematic_total,
        rate_matched_bits=tuple(int(value) for value in packet.e_r),
        rate_matched_bits_total=rate_matched_total,
        channel_uses_exact=channel_uses_exact,
        channel_bits_equal_k_times_qm=channel_bits_equal_k_times_qm,
        systematic_reconciles=systematic_reconciles,
        channel_reconciles=channel_reconciles,
        reconciles=(
            channel_uses_exact
            and channel_bits_equal_k_times_qm
            and systematic_reconciles
            and channel_reconciles
        ),
    )


def _modulation_of(q_m: int) -> str:
    for name in get("baseline.modulations"):
        if bits_per_symbol(name) == q_m:
            return name
    raise ValueError(f"no configured modulation carries {q_m} bits per symbol")


def _require_interleaver() -> str:
    name = get("baseline.modulation_bit_interleaver")
    if not get("baseline.modulation_bit_interleaver_required"):
        raise NotImplementedError(
            "params.baseline.modulation_bit_interleaver_required is no longer set; "
            "the classical arm has no unbroken-interleaver mode"
        )
    if name != "ts_38212_5_4_2_2":
        raise NotImplementedError(f"unsupported bit interleaver: {name}")
    return name


def _require_fixed_normalisation() -> None:
    if get("baseline.per_packet_power_rescaling_permitted"):
        raise NotImplementedError(
            "per-packet power rescaling is prohibited by params.baseline"
        )
    convention = get("baseline.constellation_normalisation")
    if convention != "fixed_unit_average_energy_under_uniform_labels":
        raise NotImplementedError(f"unsupported constellation normalisation: {convention}")
    if get("baseline.constellation_mapping") != "ts_38211_gray":
        raise NotImplementedError("unsupported constellation mapping")


def modulate(blocks: list[np.ndarray], modulation: str) -> np.ndarray:
    """Interleave each code block, concatenate, and Gray-map to complex symbols.

    No power normalisation is applied here beyond the fixed constellation
    normalisation baked into ``map_bits``.
    """

    _require_interleaver()
    _require_fixed_normalisation()
    q_m = bits_per_symbol(modulation)
    interleaved = [
        interleave(np.asarray(block, dtype=np.uint8).reshape(-1), q_m)
        for block in blocks
    ]
    return map_bits(np.concatenate(interleaved), modulation)


def demodulate(
    symbols: np.ndarray,
    modulation: str,
    n0: float,
    block_lengths: tuple[int, ...],
) -> list[np.ndarray]:
    """Max-log-APP demap the whole packet, then split and deinterleave blocks."""

    _require_interleaver()
    if get("baseline.demapper") != "max_log_app":
        raise NotImplementedError(f"unsupported demapper: {get('baseline.demapper')}")
    if get("baseline.demapper_noise_variance_convention") != (
        "two_sided_n0_over_2_per_real_dimension"
    ):
        raise NotImplementedError("unsupported demapper noise-variance convention")
    if get("baseline.ldpc_llr_convention") != "log_p1_over_p0":
        raise NotImplementedError("unsupported LLR convention")
    q_m = bits_per_symbol(modulation)
    llrs = max_log_llr(np.asarray(symbols).reshape(1, -1), modulation, n0).reshape(-1)
    if llrs.size != int(sum(block_lengths)):
        raise ValueError("demapped LLR count does not match the packet plan")
    blocks = []
    offset = 0
    for length in block_lengths:
        blocks.append(deinterleave(llrs[offset : offset + length], q_m))
        offset += length
    return blocks


def _shared_channel() -> torch.nn.Module:
    """Build the one project channel, and refuse anything that is not it.

    The classical and learned arms must share the same AWGN implementation and
    the same SNR definition.  A second implementation — even a correct-looking
    one — silently breaks every paired comparison, so it is rejected here rather
    than discovered in the results.
    """

    if _CHANNEL_MODEL not in get("channel.models_supported"):
        raise NotImplementedError(
            f"{_CHANNEL_MODEL} is not in params.channel.models_supported"
        )
    channel = build_channel(_CHANNEL_MODEL)
    if not isinstance(channel, AWGN):
        raise RuntimeError(
            "the classical arm must use the shared channels.awgn.AWGN, not "
            f"{type(channel).__module__}.{type(channel).__qualname__}"
        )
    channel.eval()
    return channel


def transport_round_trip(
    payload_bits: np.ndarray,
    packet: PacketPlan,
    *,
    snr_db: float,
    noise_id: str,
    device: str = "cpu",
) -> TransportOutcome:
    """Run one packet through encode, map, the shared AWGN, demap and decode."""

    accounting = build_accounting(packet)
    if not accounting.reconciles:
        raise AssertionError(f"packet accounting does not reconcile: {accounting}")
    modulation = accounting.modulation
    encoded = transmit_transport(payload_bits, packet, device)
    symbols = modulate(encoded, modulation)
    if symbols.size != accounting.k_symbols:
        raise AssertionError("modulated symbol count differs from the exact k")

    transmitted = torch.as_tensor(symbols, dtype=torch.complex64).reshape(1, -1)
    energy = realised_symbol_energy(symbols)
    papr = float(symbol_papr_db(transmitted)[0])

    unit_noise = keyed_complex_noise(
        noise_id, accounting.k_symbols, dtype=transmitted.dtype
    )
    channel = _shared_channel()
    received = channel(transmitted, float(snr_db), unit_noise=unit_noise)

    n0 = n0_from_esn0_db(float(snr_db))
    llr_blocks = demodulate(
        received.reshape(-1).numpy(), modulation, n0, accounting.rate_matched_bits
    )
    decoded = receive_transport_verified(llr_blocks, packet, device)

    return TransportOutcome(
        accounting=accounting,
        snr_db=float(snr_db),
        n0=n0,
        channel_model=_CHANNEL_MODEL,
        noise_id=noise_id,
        realised_symbol_energy=energy,
        papr_db=papr,
        per_packet_power_rescaling_applied=False,
        interleaver=_require_interleaver(),
        unit_noise_sha256=hashlib.sha256(
            unit_noise.numpy().tobytes()
        ).hexdigest(),
        received=decoded,
    )
