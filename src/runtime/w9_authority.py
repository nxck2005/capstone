"""Authority-derived runtime roots and live Pascal authentication for W9 v4."""

from __future__ import annotations

import json
import platform
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.execution_profiles import ProfileAuthenticationError, authenticate_execution_profile
from runtime.source_guard import SourceGuardHold, assert_clean_source_closure


class W9AuthorityHold(RuntimeError):
    """The W9 authority cannot be used for a live process."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W9AuthorityHold(message)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"authority is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W9AuthorityHold(f"authority is corrupt: {exc}") from None
    _require(isinstance(value, dict), "authority is not an object")
    return value


def load_authority(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    value = _read(Path(path))
    if kind is not None:
        _require(value.get("authority_kind") == kind, "authority kind differs")
    runtime_root = value.get("runtime_root")
    _require(isinstance(runtime_root, str) and runtime_root and not Path(runtime_root).is_absolute(), "authority runtime root must be repository-relative")
    root = Path(runtime_root)
    _require(".." not in root.parts, "authority runtime root escapes repository")
    return value


def resolve_runtime_root(repo: Path, authority: Mapping[str, Any]) -> Path:
    runtime_root = authority.get("runtime_root")
    _require(isinstance(runtime_root, str) and runtime_root, "authority has no runtime root")
    relative = Path(runtime_root)
    _require(not relative.is_absolute() and ".." not in relative.parts, "authority runtime root is not repository-contained")
    resolved = (Path(repo).resolve() / relative).resolve()
    _require(Path(repo).resolve() in resolved.parents, "authority runtime root escapes repository")
    return resolved


def resolve_authority_pair(repo: Path, authority: Mapping[str, Any], verifier_authority: Mapping[str, Any]) -> Path:
    producer = resolve_runtime_root(repo, authority)
    verifier = resolve_runtime_root(repo, verifier_authority)
    _require(producer == verifier, "producer and verifier resolve different authority runtime roots")
    return producer


def authenticate_live_w9_pascal(
    repo: Path,
    authority: Mapping[str, Any],
    *,
    config_hash: str,
    require_openjpeg: bool = False,
) -> dict[str, Any]:
    """Authenticate host, source closure and exact GPU immediately pre-model."""

    _require(authority.get("execution_profile_id") == "confessor_pascal_cu126", "W9 v4 requires the Confessor Pascal profile")
    _require(authority.get("host") == "confessor", "W9 authority host differs")
    live_hosts = {socket.gethostname(), platform.node()}
    _require("confessor" in live_hosts or any(host.startswith("confessor") for host in live_hosts), "live host is not Confessor")
    _require(authority.get("gpu_uuid"), "W9 authority has no exact GPU UUID")
    _require(authority.get("device") == "cuda:0", "W9 authority must bind logical cuda:0")
    _require(
        authority.get("cuda_visible_devices", authority.get("gpu_uuid")) == authority.get("gpu_uuid"),
        "W9 authority must expose exactly its frozen GPU UUID",
    )
    try:
        source_report = assert_clean_source_closure(repo, authority["source_binding"])
        environment = authenticate_execution_profile(
            "confessor_pascal_cu126",
            device=str(authority["device"]),
            config_hash=config_hash,
            require_openjpeg=require_openjpeg,
            expected_gpu_uuid=str(authority["gpu_uuid"]),
            require_cuda_visible_mapping=True,
        )
    except (SourceGuardHold, ProfileAuthenticationError, RuntimeError, ValueError) as exc:
        raise W9AuthorityHold(f"live W9 Pascal authentication failed: {exc}") from None
    _require(environment.get("gpu_uuid") == authority["gpu_uuid"], "live GPU UUID differs from frozen W9 authority")
    _require(environment.get("gpu_name") == authority["gpu_name"], "live GPU name differs from frozen W9 authority")
    _require(environment.get("gpu_compute_capability") == authority["compute_capability"], "live GPU compute capability differs")
    _require(environment.get("gpu_index") == int(str(authority["device"])[5:]), "live CUDA mapping differs")  # literal-ok: cuda:N device suffix
    mapping = environment.get("cuda_mapping")
    _require(isinstance(mapping, Mapping), "live CUDA mapping evidence is missing")
    _require(mapping.get("logical_device") == "cuda:0", "live CUDA logical device differs")
    _require(mapping.get("cuda_visible_devices") == authority.get("cuda_visible_devices", authority["gpu_uuid"]), "live CUDA_VISIBLE_DEVICES differs")
    _require(mapping.get("cuda0_gpu_uuid") == authority["gpu_uuid"], "live cuda:0 UUID differs")
    _require(mapping.get("cuda0_gpu_name") == authority["gpu_name"], "live cuda:0 GPU name differs")
    _require(mapping.get("cuda0_compute_capability") == authority["compute_capability"], "live cuda:0 compute capability differs")
    _require(environment.get("git_dirty") is False, "live scientific checkout is dirty")
    _require(environment.get("config_hash") == config_hash, "live config hash differs")
    return {"authority": dict(authority), "source_guard": source_report, "environment": environment}


__all__ = ["W9AuthorityHold", "authenticate_live_w9_pascal", "load_authority", "resolve_authority_pair", "resolve_runtime_root"]
