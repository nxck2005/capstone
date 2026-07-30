"""Classical arm end to end: canonical image in, decoded image plus a verdict out.

The segment implemented here is exactly

    canonical source image -> configured downsample -> JPEG 2000 budget search
    -> exact packet plan -> TB CRC -> code-block segmentation and CRC -> filler
    and rate matching -> modulation bit interleaving -> BPSK/QPSK/16-QAM mapping
    -> shared project AWGN -> soft demapping -> LDPC decode -> CRC check and
    decode-failure/outage classification -> JPEG 2000 decode -> configured
    upsample

Classification into an outage label, reconstruction and task metrics, record
emission and BR-4 operating-point selection are deliberately *not* here.  This
module returns a structured result that those layers consume.

Every invocation returns a verdict.  A configuration that cannot be run is
reported, never skipped, and the three ways it can fail stay distinct:

``structural_infeasibility``
    no legal TS 38.212 packetisation exists for ``(k, modulation, rate)``.
    Detected before any encoding happens.
``codec_infeasibility``
    a packetisation exists, but JPEG 2000 cannot emit a codestream at or below
    the payload budget for this image at any configured downsample axis.
``decode_failure``
    the packet was transmitted and a CRC failed.
``delivered``
    the transport block was recovered and the image decoded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from artifacts.ids import make_noise_id
from baseline.classical.channel_transport import (
    TransportAccounting,
    TransportOutcome,
    build_accounting,
    transport_round_trip,
)
from baseline.j2k import J2KCodec, J2KCodecError, J2KResult
from baseline.ldpc.transport import build_packet_plan
from config.params import get
from data.preprocessing import (
    CanonicalProduct,
    codec_downsample,
    codec_input,
    codec_upsample,
)

STRUCTURAL_INFEASIBILITY = "structural_infeasibility"
CODEC_INFEASIBILITY = "codec_infeasibility"
DECODE_FAILURE = "decode_failure"
DELIVERED = "delivered"
VERDICTS = (STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY, DECODE_FAILURE, DELIVERED)

#: Per-axis reasons recorded inside a ``codec_infeasibility`` verdict.  They are
#: sub-reasons of one verdict, not a fourth verdict class.
BUDGET_EXCEEDED = "budget_exceeded"
CODEC_CONFIGURATION_ERROR = "codec_configuration_error"

_CHANNEL_MODEL = "awgn"
_RNG_PURPOSE = "channel_noise"
_J2K_EOC = b"\xff\xd9"


class ClassicalPipelineError(RuntimeError):
    """A contract violation inside the classical path, never a link outcome."""


@dataclass(frozen=True)
class ChannelIdentity:
    """The run-level half of ``params.artifacts.noise_id_key``.

    The per-invocation half — ``stable_sample_id``, ``test_snr_db``, ``k`` and
    ``block_index`` — is supplied by the pipeline call itself, so a caller
    cannot accidentally key two different packets to the same realisation.
    """

    dataset_version: str
    split_manifest_hash: str
    channel_seed: int

    def noise_id(
        self,
        *,
        stable_sample_id: str,
        test_snr_db: float,
        k: int,
        block_index: int,
    ) -> str:
        return make_noise_id(
            {
                "dataset_version": self.dataset_version,
                "split_manifest_hash": self.split_manifest_hash,
                "stable_sample_id": stable_sample_id,
                "test_snr_db": test_snr_db,
                "channel_seed": self.channel_seed,
                "channel": _CHANNEL_MODEL,
                "k": k,
                "block_index": block_index,
                "rng_purpose": _RNG_PURPOSE,
            }
        )


@dataclass(frozen=True)
class SourceCoding:
    """What the JPEG 2000 stage did against the packet plan's payload budget."""

    feasible: bool
    encode_axis_px: int | None
    axes_attempted: tuple[int, ...]
    axis_reasons: tuple[tuple[int, str], ...]
    payload_capacity_bytes: int
    emitted_bytes: int | None
    payload_filler_bytes: int | None
    payload_filler_bits: int | None
    codestream_sha256: str | None
    cache_key: str | None
    cache_hit: bool | None
    search_iterations: int | None
    #: The exact emitted codestream bytes.  Retained so the record layer can
    #: split them into JPEG 2000 container bytes and entropy-coded data bytes
    #: for BR-11's ``header_bytes``/``payload_bytes`` columns without
    #: re-encoding.  ``None`` whenever no codestream was emitted.
    emitted_codestream: bytes | None = None


@dataclass(frozen=True)
class ClassicalResult:
    """One invocation's verdict, accounting, measurements and decoded image."""

    verdict: str
    dataset: str
    k_symbols: int
    modulation: str
    ldpc_rate: str
    snr_db: float
    stable_sample_id: str
    noise_id: str | None
    packet_feasible: bool
    structural_reason: str | None
    accounting: TransportAccounting | None
    source_coding: SourceCoding | None
    transport: TransportOutcome | None
    codestream_recovered_exactly: bool | None
    decoded_image: np.ndarray | None

    @property
    def delivered(self) -> bool:
        return self.verdict == DELIVERED

    def summary(self) -> dict[str, Any]:
        """A flat, record-shaped view; the emitted schema itself is PB_2's."""

        return {
            "verdict": self.verdict,
            "dataset": self.dataset,
            "k": self.k_symbols,
            "modulation": self.modulation,
            "ldpc_rate": self.ldpc_rate,
            "test_snr_db": self.snr_db,
            "stable_sample_id": self.stable_sample_id,
            "noise_id": self.noise_id,
            "structural_reason": self.structural_reason,
            "accounting": None if self.accounting is None else self.accounting.as_dict(),
            "encode_axis_px": (
                None if self.source_coding is None else self.source_coding.encode_axis_px
            ),
            "emitted_bytes": (
                None if self.source_coding is None else self.source_coding.emitted_bytes
            ),
            "payload_filler_bytes": (
                None
                if self.source_coding is None
                else self.source_coding.payload_filler_bytes
            ),
            "axis_reasons": (
                None if self.source_coding is None else list(self.source_coding.axis_reasons)
            ),
            "realised_symbol_energy": (
                None if self.transport is None else self.transport.realised_symbol_energy
            ),
            "papr_db": None if self.transport is None else self.transport.papr_db,
            "crc_ok": None if self.transport is None else self.transport.crc_ok,
        }


def configured_axes(dataset: str, canonical_shorter_side: int) -> tuple[int, ...]:
    """Configured encode axes, largest first, never upscaling the canonical image."""

    axes = get("baseline.downsample_axis_px")[dataset]
    if not get("baseline.downsample_axis_never_upscales"):
        raise NotImplementedError(
            "params.baseline.downsample_axis_never_upscales is no longer set"
        )
    permitted = sorted(
        {int(axis) for axis in axes if int(axis) <= canonical_shorter_side}, reverse=True
    )
    if not permitted:
        raise ClassicalPipelineError(
            f"no configured downsample axis for {dataset} fits a "
            f"{canonical_shorter_side}px canonical image"
        )
    return tuple(permitted)


def _strip_payload_filler(payload: bytes) -> bytes | None:
    """Recover the codestream from a zero-filled payload at its EOC marker.

    Filler is zero bytes, so the last EOC in the payload is the real one.  No
    length is signalled: the raw codestream is self-terminating, which keeps
    ``params.baseline.control_plane_policy`` honest.
    """

    end = payload.rfind(_J2K_EOC)
    if end < 0:
        return None
    return payload[: end + len(_J2K_EOC)]


def _encode_source(
    *,
    codec: J2KCodec,
    canonical_image: np.ndarray,
    canonical_pixels_sha256: str,
    dataset: str,
    payload_capacity_bytes: int,
    encode_axis_px: int | None,
) -> tuple[SourceCoding, J2KResult | None, np.ndarray | None]:
    canonical_shorter_side = int(min(canonical_image.shape[:2]))
    if encode_axis_px is None:
        axes = configured_axes(dataset, canonical_shorter_side)
    else:
        # An explicit axis is a *selection* from the configured ladder, not a
        # second configuration source.  PB_1 only checked that it did not
        # upscale, which let an unconfigured axis (say 48 px for CIFAR-10) reach
        # the codec and produce cache keys and evidence for a configuration the
        # spec never authorised.  Membership is checked before any encoding runs.
        requested = int(encode_axis_px)
        if requested > canonical_shorter_side:
            raise ClassicalPipelineError("requested encode axis would upscale the source")
        permitted = configured_axes(dataset, canonical_shorter_side)
        if requested not in permitted:
            raise ClassicalPipelineError(
                f"requested encode axis {requested}px is not configured for "
                f"{dataset}: params.baseline.downsample_axis_px permits {list(permitted)}"
            )
        axes = (requested,)

    attempted: list[int] = []
    reasons: list[tuple[int, str]] = []
    for axis in axes:
        attempted.append(axis)
        downsampled = codec_downsample(canonical_image, axis)
        try:
            result = codec.encode_to_budget(
                downsampled,
                canonical_pixels_sha256=canonical_pixels_sha256,
                budget_bytes=payload_capacity_bytes,
                encode_axis_px=axis,
            )
        except J2KCodecError as exc:
            # The axis produced no codestream at all under the frozen codec
            # configuration.  Recorded verbatim rather than swallowed: this is
            # not the same failure as "the codestream did not fit the budget".
            reasons.append((axis, f"{CODEC_CONFIGURATION_ERROR}: {exc}"))
            continue
        if not result.feasible:
            reasons.append((axis, BUDGET_EXCEEDED))
            continue
        if result.codestream is None or result.emitted_byte_count is None:
            raise ClassicalPipelineError("feasible JPEG 2000 result carries no codestream")
        if result.emitted_byte_count > payload_capacity_bytes:
            raise ClassicalPipelineError(
                "emitted codestream exceeds the payload capacity"
            )
        filler_bytes = payload_capacity_bytes - result.emitted_byte_count
        return (
            SourceCoding(
                feasible=True,
                encode_axis_px=axis,
                axes_attempted=tuple(attempted),
                axis_reasons=tuple(reasons),
                payload_capacity_bytes=payload_capacity_bytes,
                emitted_bytes=result.emitted_byte_count,
                payload_filler_bytes=filler_bytes,
                payload_filler_bits=filler_bytes * np.iinfo(np.uint8).bits,
                codestream_sha256=result.codestream_sha256,
                cache_key=result.cache_key,
                cache_hit=result.cache_hit,
                search_iterations=result.search_iterations,
                emitted_codestream=result.codestream,
            ),
            result,
            downsampled,
        )

    return (
        SourceCoding(
            feasible=False,
            encode_axis_px=None,
            axes_attempted=tuple(attempted),
            axis_reasons=tuple(reasons),
            payload_capacity_bytes=payload_capacity_bytes,
            emitted_bytes=None,
            payload_filler_bytes=None,
            payload_filler_bits=None,
            codestream_sha256=None,
            cache_key=None,
            cache_hit=None,
            search_iterations=None,
        ),
        None,
        None,
    )


def run_classical_pipeline(
    product: CanonicalProduct,
    *,
    dataset: str,
    k_symbols: int,
    modulation: str,
    ldpc_rate: str,
    snr_db: float,
    codec: J2KCodec,
    channel_identity: ChannelIdentity | Mapping[str, Any],
    encode_axis_px: int | None = None,
    block_index: int = 0,
    device: str = "cpu",
) -> ClassicalResult:
    """Run one image through the classical arm and return its verdict."""

    if ldpc_rate not in get("baseline.ldpc_rates"):
        raise ValueError(f"unconfigured LDPC rate: {ldpc_rate}")
    if isinstance(channel_identity, Mapping):
        channel_identity = ChannelIdentity(**channel_identity)

    canonical_image = codec_input(product)
    stable_sample_id = product.stable_sample_id

    packet = build_packet_plan(k_symbols, modulation, ldpc_rate)
    if not packet.feasible:
        return ClassicalResult(
            verdict=STRUCTURAL_INFEASIBILITY,
            dataset=dataset,
            k_symbols=k_symbols,
            modulation=modulation,
            ldpc_rate=ldpc_rate,
            snr_db=float(snr_db),
            stable_sample_id=stable_sample_id,
            noise_id=None,
            packet_feasible=False,
            structural_reason=packet.reason,
            accounting=None,
            source_coding=None,
            transport=None,
            codestream_recovered_exactly=None,
            decoded_image=None,
        )

    accounting = build_accounting(packet)
    if accounting.k_symbols != k_symbols:
        raise ClassicalPipelineError("packet plan does not carry the requested k")

    source_coding, j2k, _ = _encode_source(
        codec=codec,
        canonical_image=canonical_image,
        canonical_pixels_sha256=hashlib.sha256(canonical_image.tobytes()).hexdigest(),
        dataset=dataset,
        payload_capacity_bytes=accounting.payload_bytes,
        encode_axis_px=encode_axis_px,
    )
    if not source_coding.feasible or j2k is None or j2k.codestream is None:
        return ClassicalResult(
            verdict=CODEC_INFEASIBILITY,
            dataset=dataset,
            k_symbols=k_symbols,
            modulation=modulation,
            ldpc_rate=ldpc_rate,
            snr_db=float(snr_db),
            stable_sample_id=stable_sample_id,
            noise_id=None,
            packet_feasible=True,
            structural_reason=None,
            accounting=accounting,
            source_coding=source_coding,
            transport=None,
            codestream_recovered_exactly=None,
            decoded_image=None,
        )

    assert source_coding.payload_filler_bytes is not None
    payload_bytes = j2k.codestream + b"\x00" * source_coding.payload_filler_bytes
    if len(payload_bytes) != accounting.payload_bytes:
        raise ClassicalPipelineError("padded payload does not fill the transport block")
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    if payload_bits.size != accounting.payload_bits:
        raise ClassicalPipelineError("payload bit count does not match the packet plan")

    noise_id = channel_identity.noise_id(
        stable_sample_id=stable_sample_id,
        test_snr_db=float(snr_db),
        k=k_symbols,
        block_index=block_index,
    )
    transport = transport_round_trip(
        payload_bits, packet, snr_db=float(snr_db), noise_id=noise_id, device=device
    )

    if not transport.crc_ok or transport.payload_bits is None:
        return ClassicalResult(
            verdict=DECODE_FAILURE,
            dataset=dataset,
            k_symbols=k_symbols,
            modulation=modulation,
            ldpc_rate=ldpc_rate,
            snr_db=float(snr_db),
            stable_sample_id=stable_sample_id,
            noise_id=noise_id,
            packet_feasible=True,
            structural_reason=None,
            accounting=accounting,
            source_coding=source_coding,
            transport=transport,
            codestream_recovered_exactly=None,
            decoded_image=None,
        )

    received_payload = np.packbits(transport.payload_bits).tobytes()
    recovered = _strip_payload_filler(received_payload)
    if recovered is None:
        raise ClassicalPipelineError(
            "CRC-clean payload carries no JPEG 2000 end-of-codestream marker"
        )
    decoded = codec.decode_codestream(recovered)
    restored = codec_upsample(decoded, tuple(int(v) for v in canonical_image.shape[:2]))

    return ClassicalResult(
        verdict=DELIVERED,
        dataset=dataset,
        k_symbols=k_symbols,
        modulation=modulation,
        ldpc_rate=ldpc_rate,
        snr_db=float(snr_db),
        stable_sample_id=stable_sample_id,
        noise_id=noise_id,
        packet_feasible=True,
        structural_reason=None,
        accounting=accounting,
        source_coding=source_coding,
        transport=transport,
        codestream_recovered_exactly=recovered == j2k.codestream,
        decoded_image=restored,
    )
