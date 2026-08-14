"""Named execution-profile selection and fail-closed authentication (SR-23).

Execution profiles are provenance, not interchangeable device aliases.  New
scientific artifacts bind one profile before measurement; qualification may
authenticate a registry entry whose role is still pending without making it
production eligible.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT, get
from config.run_config import FrozenMap, RunConfig, canonical_sha256


class ProfileAuthenticationError(RuntimeError):
    """The live runtime is not the execution profile it claims to be."""


def profile_definition(profile_id: str) -> Mapping[str, Any]:
    profiles = get("environment.execution_profiles")
    if not isinstance(profile_id, str) or profile_id not in profiles:
        raise ValueError(f"unknown execution profile: {profile_id!r}")
    profile = profiles[profile_id]
    if not isinstance(profile, Mapping):
        raise TypeError(f"execution profile {profile_id!r} must be a mapping")
    return profile


def bind_execution_profile(config: RunConfig, profile_id: str) -> RunConfig:
    """Return a new schema-current config with one frozen profile identity."""

    profile = profile_definition(profile_id)
    resolved = config.resolved.to_dict()
    prior = resolved.get("execution_profile_id")
    if prior is not None and prior != profile_id:
        raise ValueError(
            f"execution profile is already frozen as {prior}; cannot switch to {profile_id}"
        )
    if profile["role"] != "eligible_production_execution_profile":
        raise ValueError(f"execution profile is not production eligible: {profile_id}")
    resolved["execution_profile_id"] = profile_id
    return replace(
        config,
        fingerprint_schema_version=int(get("config.fingerprint_schema_version")),
        resolved=FrozenMap.from_mapping(resolved),
    )


def selection_record(
    *,
    scope_id: str,
    scope_kind: str,
    profile_id: str,
    git_commit: str,
    config_hash: str,
) -> dict[str, Any]:
    """Canonical freeze record written before a run's first measurement."""

    if not scope_id or not scope_kind:
        raise ValueError("execution-profile selection scope must be non-empty")
    if len(git_commit) != 40 or len(config_hash) != 64:
        raise ValueError("selection record requires full git and config hashes")
    profile = profile_definition(profile_id)
    if profile["role"] != "eligible_production_execution_profile":
        raise ValueError(f"execution profile is not production eligible: {profile_id}")
    record = {
        "schema_version": 1,
        "status": "frozen_before_first_scientific_measurement",
        "scope_id": scope_id,
        "scope_kind": scope_kind,
        "execution_profile_id": profile_id,
        "lock_file": profile["lock_file"],
        "lock_file_sha256": profile["lock_file_sha256"],
        "git_commit": git_commit,
        "config_hash": config_hash,
    }
    record["selection_sha256"] = canonical_sha256(record)
    return record


def verify_selection_record(
    record: Mapping[str, Any], *, expected_scope_id: str | None = None
) -> None:
    required = {
        "schema_version",
        "status",
        "scope_id",
        "scope_kind",
        "execution_profile_id",
        "lock_file",
        "lock_file_sha256",
        "git_commit",
        "config_hash",
        "selection_sha256",
    }
    if set(record) != required:
        raise ValueError("execution-profile selection record schema differs")
    if record["schema_version"] != 1:
        raise ValueError("unsupported execution-profile selection schema")
    if record["status"] != "frozen_before_first_scientific_measurement":
        raise ValueError("execution-profile selection is not frozen")
    if expected_scope_id is not None and record["scope_id"] != expected_scope_id:
        raise ValueError("execution-profile selection scope differs")
    profile = profile_definition(str(record["execution_profile_id"]))
    if record["lock_file"] != profile["lock_file"]:
        raise ValueError("selection lock path differs from profile")
    if record["lock_file_sha256"] != profile["lock_file_sha256"]:
        raise ValueError("selection lock hash differs from profile")
    unhashed = {key: value for key, value in record.items() if key != "selection_sha256"}
    if record["selection_sha256"] != canonical_sha256(unhashed):
        raise ValueError("execution-profile selection digest differs")


def _package_versions() -> dict[str, str]:
    import numpy
    import torch
    import torchvision
    from sionna import __version__ as sionna_version

    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(numpy.__version__),
        "sionna_version": str(sionna_version),
    }


def _gpu_inventory() -> dict[int, dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,  # literal-ok: subprocess safety timeout
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProfileAuthenticationError(f"cannot query GPU inventory: {exc}") from None
    inventory: dict[int, dict[str, str]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ProfileAuthenticationError("unexpected nvidia-smi inventory format")
        index = int(fields[0])
        if index in inventory:
            raise ProfileAuthenticationError("duplicate GPU index in inventory")
        inventory[index] = {
            "gpu_uuid": fields[1],
            "gpu_name": fields[2],
            "driver_version": fields[3],
            "gpu_vram_mib": fields[4],
        }
    return inventory


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,  # literal-ok: subprocess safety timeout
    ).stdout.strip()


def authenticate_execution_profile(
    profile_id: str,
    *,
    device: str,
    config_hash: str,
    require_openjpeg: bool = False,
    allow_pending_qualification: bool = False,
) -> dict[str, Any]:
    """Authenticate exact software, lock, device and source identity."""

    import torch
    from env import loaded_openjpeg_version, set_deterministic_backend

    profile = profile_definition(profile_id)
    allowed_roles = {"eligible_production_execution_profile"}
    if allow_pending_qualification:
        allowed_roles.add("eligible_production_execution_profile_pending_qualification")
    if profile["role"] not in allowed_roles:
        raise ProfileAuthenticationError(
            f"profile {profile_id} has ineligible role {profile['role']}"
        )
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ProfileAuthenticationError("new profile-aware CUDA work requires explicit cuda:N")
    gpu_index = int(device[5:])
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise ProfileAuthenticationError(f"CUDA device is unavailable: {device}")

    expected_versions = {
        key: str(profile[key])
        for key in (
            "python_version",
            "torch_version",
            "torch_cuda_build",
            "torchvision_version",
            "numpy_version",
            "sionna_version",
        )
    }
    actual_versions = _package_versions()
    for key, expected in expected_versions.items():
        if actual_versions[key] != expected:
            raise ProfileAuthenticationError(
                f"{key} mismatch for {profile_id}: {actual_versions[key]!r} != {expected!r}"
            )

    lock_path = REPO_ROOT / str(profile["lock_file"])
    if not lock_path.is_file():
        raise ProfileAuthenticationError(f"profile lock is missing: {lock_path}")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if lock_sha != profile["lock_file_sha256"]:
        raise ProfileAuthenticationError("profile lock SHA-256 mismatch")

    inventory = _gpu_inventory()
    if gpu_index not in inventory:
        raise ProfileAuthenticationError(f"nvidia-smi did not enumerate GPU {gpu_index}")
    properties = torch.cuda.get_device_properties(gpu_index)
    torch_uuid = str(properties.uuid)
    torch_uuid = torch_uuid if torch_uuid.startswith("GPU-") else f"GPU-{torch_uuid}"
    gpu = next((item for item in inventory.values() if item["gpu_uuid"] == torch_uuid), None)
    if gpu is None:
        raise ProfileAuthenticationError("Torch UUID is absent from nvidia-smi inventory")
    if gpu["gpu_uuid"] not in profile["allowed_gpu_uuids"]:
        raise ProfileAuthenticationError("GPU UUID is not allowed by the profile")
    if gpu["gpu_name"] not in profile["allowed_gpu_names"]:
        raise ProfileAuthenticationError("GPU name is not allowed by the profile")
    compute_capability = f"{properties.major}.{properties.minor}"
    if compute_capability != str(profile["compute_capability"]):
        raise ProfileAuthenticationError("GPU compute capability differs from profile")
    if properties.name != gpu["gpu_name"]:
        raise ProfileAuthenticationError("Torch and nvidia-smi GPU enumeration disagree")

    set_deterministic_backend()
    settings = get("environment.deterministic_backend")
    if dict(settings) != dict(profile["deterministic_backend"]):
        raise ProfileAuthenticationError("profile deterministic backend differs")
    openjpeg = loaded_openjpeg_version(required=require_openjpeg)
    if openjpeg is not None and openjpeg != str(profile["openjpeg_version"]):
        raise ProfileAuthenticationError("OpenJPEG differs from profile")

    git_commit = _git_value("rev-parse", "HEAD")
    git_dirty = bool(_git_value("status", "--porcelain", "--untracked-files=all"))
    if len(config_hash) != 64:
        raise ProfileAuthenticationError("config hash must be a full SHA-256")
    return {
        "execution_profile_id": profile_id,
        "lock_file": str(profile["lock_file"]),
        "lock_file_sha256": lock_sha,
        **actual_versions,
        "openjpeg_version": openjpeg,
        "deterministic_backend": dict(settings),
        "amp": bool(profile["amp"]),
        **gpu,
        "gpu_compute_capability": compute_capability,
        "gpu_index": gpu_index,
        "nvidia_smi_index": int(next(index for index, item in inventory.items() if item is gpu)),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash,
    }


_ADDITIVE_COMPUTE_KEYS = {"primary_device_scope", "execution_profile_policy"}
_ADDITIVE_ENVIRONMENT_KEYS = {
    "execution_profile_registry_schema_version",
    "execution_profile_id_required_for_new_science",
    "execution_profile_record_fields",
    "execution_profiles",
    "qualification",
    "historical_profile_compatibility",
    "scientific_writer_authentication",
}


def verify_historical_local_compatibility(config: RunConfig) -> None:
    """Admit only exact schema-1 snapshots plus AM-83's additive registry."""

    if config.fingerprint_schema_version != 1:
        raise ValueError("historical compatibility applies only to fingerprint schema 1")
    snapshots = config.parameters.to_dict()
    roots = list(get("config.fingerprint_parameter_roots"))
    if set(snapshots) != set(roots):
        raise ValueError("historical snapshot root set differs")
    for root in roots:
        current = get(root)
        archived = snapshots[root]
        if root == "compute":
            current = {k: v for k, v in current.items() if k not in _ADDITIVE_COMPUTE_KEYS}
        elif root == "environment":
            current = {
                k: v for k, v in current.items() if k not in _ADDITIVE_ENVIRONMENT_KEYS
            }
        if archived != current:
            raise ValueError(f"historical params.{root} snapshot has unrelated drift")
    if get("environment.historical_profile_compatibility.archived_profile") != (
        "local_4060_cu130"
    ):
        raise ValueError("historical profile reinterpretation is forbidden")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode() + b"\n"
