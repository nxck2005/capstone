"""Classical arm: JPEG 2000 source coding over the shared LDPC/AWGN transport."""

from .channel_transport import (
    TransportAccounting,
    TransportOutcome,
    build_accounting,
    demodulate,
    modulate,
    transport_round_trip,
)
from .pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    VERDICTS,
    ChannelIdentity,
    ClassicalPipelineError,
    ClassicalResult,
    SourceCoding,
    configured_axes,
    run_classical_pipeline,
)

__all__ = [
    "CODEC_INFEASIBILITY",
    "DECODE_FAILURE",
    "DELIVERED",
    "STRUCTURAL_INFEASIBILITY",
    "VERDICTS",
    "ChannelIdentity",
    "ClassicalPipelineError",
    "ClassicalResult",
    "SourceCoding",
    "TransportAccounting",
    "TransportOutcome",
    "build_accounting",
    "configured_axes",
    "demodulate",
    "modulate",
    "run_classical_pipeline",
    "transport_round_trip",
]
