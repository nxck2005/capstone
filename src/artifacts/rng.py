"""Counter-based keyed random streams whose draws do not depend on control flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from config.params import get
from config.run_config import canonical_sha256


def keyed_generator(
    purpose: str,
    identity: Mapping[str, Any],
) -> np.random.Generator:
    """Return a fresh Philox generator keyed only by purpose and identity."""

    stream = get("artifacts.rng_stream")
    if stream != "counter_based_keyed_not_sequential":
        raise NotImplementedError(f"unsupported params.artifacts.rng_stream: {stream}")
    purposes = get("artifacts.rng_purposes")
    if purpose not in purposes:
        raise ValueError(
            f"unknown RNG purpose {purpose!r}; expected one of {purposes}"
        )
    if not identity:
        raise ValueError("RNG identity must not be empty")

    seed_material = {
        "purpose": purpose,
        "identity": dict(identity),
    }
    seed = int.from_bytes(bytes.fromhex(canonical_sha256(seed_material)), "big")
    return np.random.Generator(np.random.Philox(seed))


def keyed_standard_normal(
    purpose: str,
    identity: Mapping[str, Any],
    size: int | tuple[int, ...],
) -> np.ndarray:
    """Draw a standard-normal array as a pure function of its complete key."""

    return keyed_generator(purpose, identity).standard_normal(size=size)
