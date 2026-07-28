"""Stable per-sample identity derived only from original source payload bytes."""

from __future__ import annotations

import hashlib
import re

from config.params import get

_SAMPLE_ID_RULE = re.compile(
    r"sha256_of_original_per_sample_source_bytes_truncated_(?P<width>\d+)_hex"
)


def stable_sample_id_width() -> int:
    """Return the configured hexadecimal prefix width."""

    rule = get("datasets.stable_sample_id_rule")
    match = _SAMPLE_ID_RULE.fullmatch(rule)
    if match is None:
        raise NotImplementedError(
            f"unsupported params.datasets.stable_sample_id_rule: {rule}"
        )
    return int(match.group("width"))


def stable_sample_id(source_bytes: bytes) -> str:
    """Hash the original per-sample payload bytes, before decode or preprocessing."""

    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    if not source_bytes:
        raise ValueError("source_bytes must not be empty")

    return hashlib.sha256(source_bytes).hexdigest()[:stable_sample_id_width()]
