"""The secondary JPEG classical segment: canonical image in, verdict plus decoded image out.

This mirrors :mod:`baseline.classical.pipeline` exactly except for the source
codec.  It shares the packet plan, accounting, keyed AWGN, demapping and LDPC
decode with the JPEG 2000 arm, so the two curves differ only in the codec.
"""

from __future__ import annotations

import hashlib

import numpy as np

from baseline.classical.channel_transport import build_accounting, transport_round_trip
from baseline.classical.pipeline import (
    BUDGET_EXCEEDED,
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    ChannelIdentity,
    ClassicalPipelineError,
    ClassicalResult,
    SourceCoding,
    configured_axes,
)
from baseline.jpeg import JpegCodec, JpegCodecError, decode_codestream
from baseline.ldpc.transport import build_packet_plan
from data.preprocessing import CanonicalProduct, codec_downsample, codec_input, codec_upsample


def run_jpeg_pipeline(
    product: CanonicalProduct,
    *,
    dataset: str,
    k_symbols: int,
    modulation: str,
    ldpc_rate: str,
    snr_db: float,
    quality: int,
    codec: JpegCodec,
    channel_identity: ChannelIdentity,
    encode_axis_px: int | None = None,
    block_index: int = 0,
    device: str = "cpu",
) -> ClassicalResult:
    """Run one image through the JPEG secondary arm and return its verdict."""

    packet = build_packet_plan(k_symbols, modulation, ldpc_rate)
    canonical_image = codec_input(product)
    stable_sample_id = product.stable_sample_id
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

    canonical_shorter_side = int(min(canonical_image.shape[:2]))
    if encode_axis_px is None:
        axes = configured_axes(dataset, canonical_shorter_side)
    else:
        requested = int(encode_axis_px)
        permitted = configured_axes(dataset, canonical_shorter_side)
        if requested not in permitted:
            raise ClassicalPipelineError(
                f"requested encode axis {requested}px is not configured for {dataset}"
            )
        axes = (requested,)

    attempted: list[int] = []
    reasons: list[tuple[int, str]] = []
    result = None
    downsampled = None
    selected_axis = None
    for axis in axes:
        attempted.append(axis)
        candidate = codec_downsample(canonical_image, axis)
        try:
            encoded = codec.encode_to_budget(
                candidate,
                canonical_pixels_sha256=hashlib.sha256(canonical_image.tobytes()).hexdigest(),
                budget_bytes=accounting.payload_bytes,
                encode_axis_px=axis,
            )
        except JpegCodecError as exc:
            reasons.append((axis, f"codec_configuration_error: {exc}"))
            continue
        if not encoded.feasible or encoded.codestream is None:
            reasons.append((axis, BUDGET_EXCEEDED))
            continue
        if encoded.emitted_byte_count is None or encoded.emitted_byte_count > accounting.payload_bytes:
            raise ClassicalPipelineError("JPEG codestream exceeds the payload capacity")
        result = encoded
        downsampled = candidate
        selected_axis = axis
        break

    if result is None:
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
            source_coding=SourceCoding(
                feasible=False,
                encode_axis_px=None,
                axes_attempted=tuple(attempted),
                axis_reasons=tuple(reasons),
                payload_capacity_bytes=accounting.payload_bytes,
                emitted_bytes=None,
                payload_filler_bytes=None,
                payload_filler_bits=None,
                codestream_sha256=None,
                cache_key=None,
                cache_hit=None,
                search_iterations=None,
            ),
            transport=None,
            codestream_recovered_exactly=None,
            decoded_image=None,
        )

    assert result.codestream is not None and result.emitted_byte_count is not None
    filler = accounting.payload_bytes - result.emitted_byte_count
    source_coding = SourceCoding(
        feasible=True,
        encode_axis_px=int(selected_axis) if selected_axis is not None else None,
        axes_attempted=tuple(attempted),
        axis_reasons=tuple(reasons),
        payload_capacity_bytes=accounting.payload_bytes,
        emitted_bytes=result.emitted_byte_count,
        payload_filler_bytes=filler,
        payload_filler_bits=filler * np.iinfo(np.uint8).bits,
        codestream_sha256=hashlib.sha256(result.codestream).hexdigest(),
        cache_key=result.cache_key,
        cache_hit=None,
        search_iterations=len(result.search_points),
        emitted_codestream=None,
    )
    padded = result.codestream + b"\x00" * filler
    payload_bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8))
    if payload_bits.size != accounting.payload_bits:
        raise ClassicalPipelineError("JPEG payload bit count does not match the packet plan")
    noise_id = channel_identity.noise_id(
        stable_sample_id=stable_sample_id,
        test_snr_db=float(snr_db),
        k=k_symbols,
        block_index=block_index,
    )
    transport = transport_round_trip(payload_bits, packet, snr_db=float(snr_db), noise_id=noise_id, device=device)
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
    decoded = decode_codestream(received_payload)
    restored = codec_upsample(decoded, tuple(int(value) for value in canonical_image.shape[:2]))
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
        codestream_recovered_exactly=True,
        decoded_image=restored,
    )


__all__ = ["run_jpeg_pipeline"]
