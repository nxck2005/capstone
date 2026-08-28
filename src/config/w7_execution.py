"""W7-specific execution-profile authentication and UUID binding.

The historical profile authenticator accepts a process-visible ``cuda:N``.  W7
adds the missing outer binding: the worker process must have been launched with
one exact registered GPU UUID, and the authenticated CUDA ordinal must resolve
back to that same UUID.  Ordinals are therefore observational only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from config.execution_profiles import ProfileAuthenticationError, profile_definition
from config.params import get
from env import profile_environment_record
from training.deterministic_core import canonical_sha256
from training.w7_protocol import W7_PROFILE_ID, validate_profile_binding


class W7ExecutionHold(RuntimeError):
    """A W7 process is not bound to its frozen physical execution profile."""


def _normalise_uuid(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise W7ExecutionHold("GPU UUID must be a non-empty string")
    return value if value.startswith("GPU-") else f"GPU-{value}"


def _torch_uuid(device: str) -> tuple[str, str, str]:
    import torch

    if device != "cuda:0":
        raise W7ExecutionHold("W7 UUID authentication requires the process-visible cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise W7ExecutionHold("W7 requires CUDA_VISIBLE_DEVICES bound to one exact GPU UUID")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise W7ExecutionHold("W7 requires exactly one process-visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    uuid = _normalise_uuid(getattr(properties, "uuid", None))
    return uuid, str(properties.name), f"{properties.major}.{properties.minor}"


def authenticate_w7_gpu(
    *,
    config_hash: str,
    expected_gpu_uuid: str,
    device: str = "cuda:0",
    require_openjpeg: bool = False,
) -> dict[str, Any]:
    """Authenticate the exact UUID-bound Pascal process for a W7 run."""

    expected = _normalise_uuid(expected_gpu_uuid)
    profile = profile_definition(W7_PROFILE_ID)
    if expected not in profile["allowed_gpu_uuids"]:
        raise W7ExecutionHold("requested W7 UUID is not registered for confessor_pascal_cu126")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected:
        raise W7ExecutionHold(
            "CUDA_VISIBLE_DEVICES must equal the selected physical GPU UUID; "
            f"observed {visible!r}, expected {expected!r}"
        )
    try:
        uuid, name, capability = _torch_uuid(device)
        if uuid != expected:
            raise W7ExecutionHold("Torch process-visible UUID differs from selected UUID")
        base = profile_environment_record(
            W7_PROFILE_ID,
            device=device,
            config_hash=config_hash,
            require_openjpeg=require_openjpeg,
        )
    except (ProfileAuthenticationError, RuntimeError, ValueError) as exc:
        if isinstance(exc, W7ExecutionHold):
            raise
        raise W7ExecutionHold(f"Pascal profile authentication failed: {exc}") from None
    if base.get("gpu_uuid") != expected or base.get("gpu_name") != name:
        raise W7ExecutionHold("authenticated Torch and profile GPU identity disagree")
    if str(base.get("gpu_compute_capability")) != capability:
        raise W7ExecutionHold("authenticated GPU compute capability differs")
    binding = {
        "schema_version": 1,
        "authentication_status": "PASSED",
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": expected,
        "gpu_name": name,
        "gpu_compute_capability": capability,
        "cuda_visible_devices": visible,
        "device": device,
        "profile_environment": base,
        "lock_file_sha256": str(profile["lock_file_sha256"]),
        "git_commit": str(base["git_commit"]),
        "config_hash": config_hash,
    }
    validate_profile_binding(binding)
    binding["binding_sha256"] = canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    return binding


def verify_frozen_gpu_binding(binding: Mapping[str, Any], *, config_hash: str) -> None:
    """Verify a stored binding without probing a GPU (for local custody checks)."""

    value = dict(binding)
    digest = value.pop("binding_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(value):
        raise W7ExecutionHold("W7 GPU binding digest differs")
    if value.get("config_hash") != config_hash:
        raise W7ExecutionHold("W7 GPU binding config differs")
    validate_profile_binding(value)
    if value.get("cuda_visible_devices") != value.get("gpu_uuid"):
        raise W7ExecutionHold("stored W7 GPU binding is not UUID-bound")
    if value.get("device") != "cuda:0":
        raise W7ExecutionHold("stored W7 GPU binding device differs")


def profile_gpu_candidates() -> tuple[dict[str, str], ...]:
    """Return registered candidates in the owner-mandated GTX-first order."""

    profile = profile_definition(W7_PROFILE_ID)
    names = {
        "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b": "NVIDIA GeForce GTX 1080 Ti",
        "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a": "NVIDIA TITAN Xp",
    }
    ordered = (
        "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
        "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
    )
    allowed = set(profile["allowed_gpu_uuids"])
    if set(ordered) != allowed:
        raise W7ExecutionHold("registered Pascal GPU set differs from frozen W7 ladder")
    return tuple(
        {"gpu_uuid": uuid, "gpu_name": names[uuid], "compute_capability": str(profile["compute_capability"])}
        for uuid in ordered
    )


def profile_lock_sha256() -> str:
    return str(get("environment.execution_profiles.confessor_pascal_cu126.lock_file_sha256"))
