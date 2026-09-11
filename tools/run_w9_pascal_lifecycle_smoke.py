#!/usr/bin/env python3
"""Run the real W9 v4 trainer lifecycle with deterministic synthetic tensors.

The only tensors made by this script are the two-sample fixture tensors passed
to ``ER9V4CandidateTrainer`` through its synthetic provider.  The script does
not import a dataset or a loader, and its runtime namespace and evidence are
permanently ineligible for scientific selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.er9_v4 import ER9V4CandidateTrainer  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256, state_tree_sha256  # noqa: E402
from runtime.w9_authority import W9AuthorityHold, authenticate_live_w9_pascal, resolve_runtime_root  # noqa: E402
from verify_er9_pascal_v4 import assert_synthetic_smoke_ineligible, verify_stage1_authority  # noqa: E402


FIXTURE_ID = "w9_pascal_v4_lifecycle_smoke_v1"
RUNTIME_ROOT = REPO / "checkpoints/smoke/w9_pascal_v4_fixture"
EVIDENCE_PATH = REPO / "results/learned/w9/w9_pascal_v4_lifecycle_smoke.json"
PROFILE_ID = "confessor_pascal_cu126"


def _nvidia_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"synthetic Pascal smoke cannot authenticate GPU inventory: {exc}") from None
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError("synthetic Pascal smoke found malformed GPU inventory")
        result.append(
            {
                "index": fields[0],
                "gpu_uuid": fields[1],
                "gpu_name": fields[2],
                "driver_version": fields[3],
                "gpu_vram_mib": fields[4],
            }
        )
    if not result:
        raise RuntimeError("synthetic Pascal smoke found no GPU")
    return result


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise RuntimeError(f"synthetic smoke evidence already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("xb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    finally:
        descriptor.close()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _identity_overrides() -> dict[str, Any]:
    return {
        "fixture_id": FIXTURE_ID,
        "eligibility": {
            "NON_SCIENTIFIC": True,
            "SYNTHETIC_ONLY": True,
            "INELIGIBLE_FOR_SELECTION": True,
            "INELIGIBLE_FOR_ER9_SEARCH": True,
            "INELIGIBLE_FOR_ER2_RESULT": True,
            "TEST_NOT_ACCESSED": True,
        },
    }


def _synthetic_epoch(epoch: int, _device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[str, str]]:
    generator = torch.Generator(device="cpu").manual_seed(9400 + epoch)
    inputs = torch.rand((2, 3, 160, 160), generator=generator, dtype=torch.float32)  # literal-ok: fixed synthetic Imagenette-shaped fixture tensor
    labels = torch.tensor([epoch % 10, (epoch + 1) % 10], dtype=torch.long)  # literal-ok: fixed synthetic ten-class fixture labels
    ids = (f"{FIXTURE_ID}-epoch-{epoch}-sample-0", f"{FIXTURE_ID}-epoch-{epoch}-sample-1")
    return inputs, labels, ids


def run(
    *,
    require_pascal: bool,
    expected_gpu_uuid: str | None = None,
    runtime_root: Path = RUNTIME_ROOT,
    evidence_path: Path = EVIDENCE_PATH,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic Pascal smoke requires actual CUDA")
    inventory = _nvidia_inventory()
    host = socket.gethostname()
    node = platform.node()
    if expected_gpu_uuid is None:
        raise RuntimeError("synthetic Pascal smoke requires an exact frozen GPU UUID")
    matching = [item for item in inventory if item["gpu_uuid"] == expected_gpu_uuid]
    if len(matching) != 1:
        raise RuntimeError(f"synthetic Pascal smoke exact GPU UUID is unavailable: {expected_gpu_uuid}")
    gpu = matching[0]
    if require_pascal and not (
        host == "confessor"
        or node == "confessor"
        or host.startswith("confessor")
        or node.startswith("confessor")
    ):
        raise RuntimeError(f"synthetic Pascal smoke host is not Confessor: {host}/{node}")
    if require_pascal and gpu["gpu_name"] not in {
        "NVIDIA GeForce GTX 1080 Ti",
        "NVIDIA TITAN Xp",
    }:
        raise RuntimeError(f"synthetic Pascal smoke GPU is not registered Pascal: {gpu['gpu_name']}")
    if not runtime_root.is_absolute():
        runtime_root = REPO / runtime_root
    if runtime_root.exists() or runtime_root.is_symlink():
        raise RuntimeError(
            f"synthetic smoke runtime already exists; preserve it and choose a new fixture root: {runtime_root}"
        )

    try:
        manifest, authority, config, _solver = verify_stage1_authority()
    except (W9AuthorityHold, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"synthetic smoke active v4 authority is not authenticated: {exc}") from None
    if authority["gpu_uuid"] != gpu["gpu_uuid"] or authority["gpu_name"] != gpu["gpu_name"]:
        raise RuntimeError("synthetic smoke GPU differs from the active v4 Stage-1 authority")
    scientific_runtime = resolve_runtime_root(REPO, authority)
    if runtime_root.resolve() == scientific_runtime:
        raise RuntimeError("synthetic smoke cannot use the scientific Stage-1 runtime root")
    if require_pascal:
        # The caller launches this process with CUDA_VISIBLE_DEVICES set to the
        # exact UUID.  The authenticator proves it is logical cuda:0 before the
        # shared trainer can construct a model, optimizer, scaler, or dataset.
        live = authenticate_live_w9_pascal(REPO, authority, config_hash=authority["config_hash"])
    else:
        live = {"authority": authority, "environment": {"git_dirty": False}}
    device = torch.device("cuda:0")

    first = ER9V4CandidateTrainer(
        config,
        transmit_dim=64,
        quantiser_bits=2,
        runtime_root=runtime_root,
        source_binding=manifest,
        campaign_id=FIXTURE_ID,
        run_id=FIXTURE_ID,
        live_authentication=live,
        device=device,
        synthetic_provider=_synthetic_epoch,
        identity_overrides=_identity_overrides(),
        total_epochs=2,
        runtime_role="W9_SYNTHETIC_ONLY_ER9",
        resume=False,
    )
    first.loop.run(max_epochs=1)
    committed_epoch_zero = first.runtime.inspect()[-1]
    state_after_epoch_zero = state_tree_sha256(
        {
            "model": first.model.state_dict(),
            "optimizer": first.optimizer.state_dict(),
            "scaler": first.scaler.state_dict(),
        }
    )

    # A fresh adapter instance represents the process/restart boundary.  Its
    # constructor authenticates the committed prefix and restores the exact
    # latest model/optimizer/scaler state before epoch 1 is eligible.
    second = ER9V4CandidateTrainer(
        config,
        transmit_dim=64,
        quantiser_bits=2,
        runtime_root=runtime_root,
        source_binding=manifest,
        campaign_id=FIXTURE_ID,
        run_id=FIXTURE_ID,
        live_authentication=live,
        device=device,
        synthetic_provider=_synthetic_epoch,
        identity_overrides=_identity_overrides(),
        total_epochs=2,
        runtime_role="W9_SYNTHETIC_ONLY_ER9",
        resume=True,
    )
    state_after_restart = state_tree_sha256(
        {
            "model": second.model.state_dict(),
            "optimizer": second.optimizer.state_dict(),
            "scaler": second.scaler.state_dict(),
        }
    )
    if second.loop.completed_epoch != 0 or state_after_restart != state_after_epoch_zero:
        raise RuntimeError("synthetic smoke did not restore the exact epoch-0 state")
    terminal = second.loop.run(max_epochs=1)
    if terminal is None:
        raise RuntimeError("synthetic smoke did not complete epoch 1")
    committed = second.runtime.inspect()
    if [item.epoch for item in committed] != [0, 1]:
        raise RuntimeError("synthetic smoke committed an unexpected epoch prefix")

    stale_pointer = second.runtime.store._pointer_body(committed_epoch_zero)
    (runtime_root / "latest.json").write_bytes(canonical_bytes(stale_pointer))
    repaired = second.runtime.inspect()
    repaired_pointer = json.loads((runtime_root / "latest.json").read_bytes())
    if repaired[-1].epoch != 1 or repaired_pointer.get("epoch") != 1:
        raise RuntimeError("synthetic smoke did not repair the stale pointer")

    terminal_again, reused_again = second.runtime.terminalize(
        selected_epoch=int(terminal["selected_epoch"]),
        selection_metric=terminal["selection_metric"],
        tie_break=terminal["tie_break"],
    )
    if not reused_again or terminal_again != terminal:
        raise RuntimeError("synthetic smoke terminalization is not idempotent")

    evidence = {
        "schema_version": 1,
        "artifact_role": "W9_PASCAL_V4_SYNTHETIC_LIFECYCLE_SMOKE",
        "status": "NON_SCIENTIFIC",
        "fixture_id": FIXTURE_ID,
        "fixture_identity": first.runtime.identity,
        "eligibility": first.runtime.identity["eligibility"],
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "source_tree_hashes": manifest["tree_hashes"],
        "pascal_lock_sha256": manifest["requirements_pascal_lock_sha256"],
        "host": host,
        "platform_node": node,
        "execution_profile": PROFILE_ID if require_pascal else "synthetic_cuda_development_only",
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": "cuda:0",
        "cuda_mapping": live.get("environment", {}).get("cuda_mapping"),
        "runtime_root": str(runtime_root.relative_to(REPO)) if REPO in runtime_root.resolve().parents else str(runtime_root.resolve()),
        "committed_epoch_count": 2,
        "restart_resume": {
            "fresh_adapter_reconstruction": True,
            "restored_epoch": 0,
            "exact_model_optimizer_scaler_restore": True,
            "state_after_epoch_zero_sha256": state_after_epoch_zero,
            "state_after_restart_sha256": state_after_restart,
            "epoch_one_committed_once": True,
        },
        "stale_pointer_repair": True,
        "terminal_id": terminal["terminal_id"],
        "terminal_sha256": hashlib.sha256(canonical_bytes(terminal)).hexdigest(),
        "terminal_idempotent": True,
        "scientific_data_accesses": 0,
        "test_access": 0,
        "test": "SEALED",
    }
    evidence_body = dict(evidence)
    evidence["evidence_id"] = "w9pascalv4smoke-" + canonical_sha256(evidence_body)
    assert_synthetic_smoke_ineligible(evidence)
    _write_immutable(evidence_path, evidence)
    evidence["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-nonconfessor-development", action="store_true", help="permit local CUDA mechanism testing; output remains ineligible")
    parser.add_argument("--gpu-uuid", required=True, help="exact registered GPU UUID selected for the Pascal smoke")
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--evidence-path", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args(argv)
    evidence = run(
        require_pascal=not args.allow_nonconfessor_development,
        expected_gpu_uuid=args.gpu_uuid,
        runtime_root=args.runtime_root,
        evidence_path=args.evidence_path,
    )
    print(f"synthetic W9 v4 lifecycle smoke PASS: {evidence['evidence_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
