"""Exact UUID-bound execution authentication for the W8 Pascal campaign."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from config.execution_profiles import (
    ProfileAuthenticationError,
    authenticate_execution_profile,
    profile_definition,
)
from config.params import get
from training.deterministic_core import canonical_sha256

W8_PROFILE_ID = "confessor_pascal_cu126"
W8_GPU_UUID = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"
W8_GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
W8_DEVICE = "cuda:0"
W8_LOCK_SHA256 = "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82"


class W8ExecutionHold(RuntimeError):
    """The W8 process is not bound to the frozen physical execution profile."""


def authenticate_w8_gpu(
    *,
    config_hash: str,
    expected_gpu_uuid: str = W8_GPU_UUID,
    device: str = W8_DEVICE,
    require_openjpeg: bool = False,
) -> dict[str, Any]:
    """Authenticate the one permitted W8 device, rejecting the TITAN fallback."""

    if expected_gpu_uuid != W8_GPU_UUID:
        raise W8ExecutionHold("W8 permits only the qualified GTX 1080 Ti UUID")
    if device != W8_DEVICE:
        raise W8ExecutionHold("W8 requires process-visible cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != W8_GPU_UUID:
        raise W8ExecutionHold(
            "W8 requires CUDA_VISIBLE_DEVICES bound to the exact GTX UUID"
        )
    profile = profile_definition(W8_PROFILE_ID)
    if W8_LOCK_SHA256 != str(profile["lock_file_sha256"]):
        raise W8ExecutionHold("W8 Pascal lock binding differs from the profile registry")
    if W8_GPU_UUID not in profile["allowed_gpu_uuids"] or W8_GPU_NAME not in profile["allowed_gpu_names"]:
        raise W8ExecutionHold("W8 GTX UUID/name is not registered in the Pascal profile")
    try:
        environment = authenticate_execution_profile(
            W8_PROFILE_ID,
            device=device,
            config_hash=config_hash,
            require_openjpeg=require_openjpeg,
        )
    except (ProfileAuthenticationError, RuntimeError, ValueError) as exc:
        raise W8ExecutionHold(f"W8 profile authentication failed: {exc}") from None
    if environment.get("gpu_uuid") != W8_GPU_UUID or environment.get("gpu_name") != W8_GPU_NAME:
        raise W8ExecutionHold("authenticated GPU is not the frozen GTX 1080 Ti")
    if environment.get("gpu_compute_capability") != str(profile["compute_capability"]):
        raise W8ExecutionHold("authenticated W8 compute capability differs")
    if environment.get("git_dirty") is not False:
        raise W8ExecutionHold("W8 scientific checkout is dirty")
    binding = {
        "schema_version": 1,
        "authentication_status": "PASSED",
        "execution_profile_id": W8_PROFILE_ID,
        "gpu_uuid": W8_GPU_UUID,
        "gpu_name": W8_GPU_NAME,
        "gpu_compute_capability": str(profile["compute_capability"]),
        "cuda_visible_devices": W8_GPU_UUID,
        "device": device,
        "profile_environment": environment,
        "lock_file": str(profile["lock_file"]),
        "lock_file_sha256": W8_LOCK_SHA256,
        "git_commit": str(environment["git_commit"]),
        "git_dirty": False,
        "config_hash": config_hash,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    verify_frozen_w8_gpu_binding(binding, config_hash=config_hash)
    return binding


def verify_frozen_w8_gpu_binding(
    binding: Mapping[str, Any], *, config_hash: str, source_commit: str | None = None
) -> None:
    """Verify a stored binding without probing CUDA or selecting another device."""

    value = dict(binding)
    digest = value.pop("binding_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(value):
        raise W8ExecutionHold("W8 GPU binding digest differs")
    required = {
        "schema_version", "authentication_status", "execution_profile_id", "gpu_uuid",
        "gpu_name", "gpu_compute_capability", "cuda_visible_devices", "device",
        "profile_environment", "lock_file", "lock_file_sha256", "git_commit",
        "git_dirty", "config_hash",
    }
    if set(value) != required:
        raise W8ExecutionHold("W8 GPU binding schema differs")
    if value["schema_version"] != 1 or value["authentication_status"] != "PASSED":
        raise W8ExecutionHold("W8 GPU binding status differs")
    if value["execution_profile_id"] != W8_PROFILE_ID or value["gpu_uuid"] != W8_GPU_UUID or value["gpu_name"] != W8_GPU_NAME:
        raise W8ExecutionHold("W8 GPU binding profile/UUID differs")
    if value["gpu_compute_capability"] != "6.1":  # literal-ok: Pascal compute capability
        raise W8ExecutionHold("W8 GPU compute capability differs")
    if value["cuda_visible_devices"] != W8_GPU_UUID or value["device"] != W8_DEVICE:
        raise W8ExecutionHold("W8 CUDA visibility/device binding differs")
    if value["lock_file"] != str(get("environment.execution_profiles.confessor_pascal_cu126.lock_file")) or value["lock_file_sha256"] != W8_LOCK_SHA256:
        raise W8ExecutionHold("W8 Pascal lock binding differs")
    if value["git_dirty"] is not False or not isinstance(value["git_commit"], str) or len(value["git_commit"]) != 40:  # literal-ok: Git SHA-1 width
        raise W8ExecutionHold("W8 source checkout binding is invalid")
    if source_commit is not None and value["git_commit"] != source_commit:
        raise W8ExecutionHold("W8 profile binding source commit differs")
    if value["config_hash"] != config_hash:
        raise W8ExecutionHold("W8 profile binding config hash differs")
    environment = value["profile_environment"]
    if not isinstance(environment, Mapping):
        raise W8ExecutionHold("W8 nested profile environment differs")
    expected_environment = {
        "execution_profile_id": W8_PROFILE_ID,
        "lock_file": value["lock_file"],
        "lock_file_sha256": W8_LOCK_SHA256,
        "gpu_uuid": W8_GPU_UUID,
        "gpu_name": W8_GPU_NAME,
        "gpu_compute_capability": "6.1",  # literal-ok: Pascal compute capability
        "git_commit": value["git_commit"],
        "git_dirty": False,
        "config_hash": config_hash,
    }
    if any(environment.get(key) != expected for key, expected in expected_environment.items()):
        raise W8ExecutionHold("W8 nested profile environment differs")
