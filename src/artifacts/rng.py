"""Counter-based keyed random streams whose draws do not depend on control flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from config.params import get
from config.run_config import canonical_sha256


def _validate_identity(purpose: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    fields = get(f"artifacts.rng_identity_fields.{purpose}")
    expected = set(fields)
    actual = set(identity)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"RNG identity for {purpose!r} differs: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    values = dict(identity)
    empty = [
        field
        for field, value in values.items()
        if value is None or value == ""
    ]
    if empty:
        raise ValueError(
            f"RNG identity for {purpose!r} has empty fields: {sorted(empty)}"
        )
    if purpose == "init":
        component_path = values["component_path"]
        if (
            not isinstance(component_path, str)
            or "." not in component_path
            or component_path.startswith(".")
            or component_path.endswith(".")
            or any(not part for part in component_path.split("."))
        ):
            raise ValueError(
                "init component_path must be a stable model-qualified name"
            )
    return values


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
    validated_identity = _validate_identity(purpose, identity)

    seed_material = {
        "purpose": purpose,
        "identity": validated_identity,
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
