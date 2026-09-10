#!/usr/bin/env python3
"""Run the synthetic-only CUDA lifecycle smoke for the W9 v4 custody layer.

This script deliberately does not import a dataset or instantiate a data
loader.  It initializes the actual ER-9 model and optimizer with deterministic
synthetic tensors, then exercises transaction publication, process restart,
stale-pointer repair, resume, terminalization and terminal idempotence.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.run_config import config_hash, load_experiment  # noqa: E402
from models.er9_digital import build_er9_model  # noqa: E402
from runtime.transactional_epochs import TransactionalEpochStore, canonical_bytes, sha256_bytes  # noqa: E402


FIXTURE_ID = "w9_pascal_v4_lifecycle_smoke_v1"
RUNTIME_ROOT = REPO / "checkpoints/smoke/w9_pascal_v4_fixture"
EVIDENCE_PATH = REPO / "results/learned/w9/w9_pascal_v4_lifecycle_smoke.json"


def _nvidia_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"synthetic Pascal smoke cannot authenticate GPU inventory: {exc}") from None
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            result.append({"index": fields[0], "gpu_uuid": fields[1], "gpu_name": fields[2], "driver_version": fields[3]})
    if not result:
        raise RuntimeError("synthetic Pascal smoke found no GPU")
    return result


def _save_model_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any, epoch: int) -> bytes:
    buffer = io.BytesIO()
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scaler_state": None if scaler is None else scaler.state_dict(), "test_access": 0}, buffer)
    return buffer.getvalue()


def _restore_model_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any, path: Path) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("test_access") != 0:
        raise RuntimeError("synthetic smoke checkpoint is not authenticated")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return payload


def _train_one(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any, device: torch.device, epoch: int) -> tuple[bytes, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(9400 + epoch)
    inputs = torch.rand((2, 3, 160, 160), generator=generator, dtype=torch.float32).to(device)
    labels = torch.tensor([epoch % 10, (epoch + 1) % 10], dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output = model(inputs)
        loss = F.cross_entropy(output.logits, labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError("synthetic smoke loss is non-finite")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
    if not gradients_finite:
        raise RuntimeError("synthetic smoke post-unscale gradients are non-finite")
    scaler.step(optimizer)
    scaler.update()
    correct = int((output.logits.argmax(dim=1) == labels).sum().item())
    checkpoint = _save_model_state(model, optimizer, scaler, epoch)
    return checkpoint, {
        "optimizer_opportunities": 1,
        "applied_optimizer_steps": 1,
        "grad_scaler_skips": 0,
        "validation_n_correct": correct,
        "synthetic_loss": float(loss.detach().float().item()),
        "fixture_epoch": epoch,
        "test_access": 0,
    }


def run(*, require_pascal: bool, expected_gpu_uuid: str | None = None, runtime_root: Path = RUNTIME_ROOT, evidence_path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic Pascal smoke requires actual CUDA")
    inventory = _nvidia_inventory()
    host = socket.gethostname()
    node = platform.node()
    if expected_gpu_uuid is not None:
        matching = [item for item in inventory if item["gpu_uuid"] == expected_gpu_uuid]
        if len(matching) != 1:
            raise RuntimeError(f"synthetic Pascal smoke exact GPU UUID is unavailable: {expected_gpu_uuid}")
        gpu = matching[0]
    else:
        gpu = inventory[0]
    if require_pascal:
        if expected_gpu_uuid is None:
            raise RuntimeError("synthetic Pascal smoke requires a prospectively frozen exact GPU UUID")
        if not (host == "confessor" or node == "confessor" or host.startswith("confessor") or node.startswith("confessor")):
            raise RuntimeError(f"synthetic Pascal smoke host is not Confessor: {host}/{node}")
        if gpu["gpu_name"] not in {"NVIDIA GeForce GTX 1080 Ti", "NVIDIA TITAN Xp"}:
            raise RuntimeError(f"synthetic Pascal smoke GPU is not registered Pascal: {gpu['gpu_name']}")
    device = torch.device("cuda:0")
    config = load_experiment("configs/er9-digital-pascal-v4.yaml", train_seed=0, channel_seed=0)
    model = build_er9_model(config, transmit_dim=64, quantiser_bits=2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    identity = {"fixture_id": FIXTURE_ID, "run_id": FIXTURE_ID, "source_binding": "synthetic-only", "config_hash": config_hash(config)}
    store = TransactionalEpochStore(runtime_root, identity=identity, total_epochs=2, role="W9_SYNTHETIC_ONLY_ER9")
    if runtime_root.exists():
        raise RuntimeError(f"synthetic smoke runtime already exists; preserve it and choose a new fixture root: {runtime_root}")
    store.initialise()
    checkpoint, record = _train_one(model, optimizer, scaler, device, 0)
    store.publish_epoch(0, checkpoint, record)
    # A new process reconstructs the model/optimizer/scaler from the committed
    # checkpoint. No data loader or scientific dataset is involved.
    model = build_er9_model(config, transmit_dim=64, quantiser_bits=2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    first = store.inspect()[-1]
    _restore_model_state(model, optimizer, scaler, first.checkpoint_path)
    checkpoint, record = _train_one(model, optimizer, scaler, device, 1)
    store.publish_epoch(1, checkpoint, record)
    second = store.inspect()[-1]
    stale_pointer = {
        "schema_version": 1,
        "artifact_role": "W9_SYNTHETIC_ONLY_ER9",
        "identity": identity,
        "epoch": 0,
        "next_epoch": 1,
        "epoch_path": "epochs/epoch-0000",
        "chain_sha256": store.inspect()[0].chain_sha256,
    }
    (runtime_root / "latest.json").write_bytes(canonical_bytes(stale_pointer))
    repaired = store.inspect()[-1]
    if repaired.epoch != 1 or json.loads((runtime_root / "latest.json").read_bytes())["epoch"] != 1:
        raise RuntimeError("synthetic smoke did not repair stale pointer")
    terminal, reused = store.terminalize(selected_epoch=1, selection_metric={"validation_n_correct": record["validation_n_correct"]}, tie_break="earliest_epoch")
    terminal_again, reused_again = store.terminalize(selected_epoch=1, selection_metric={"validation_n_correct": record["validation_n_correct"]}, tie_break="earliest_epoch")
    if reused or not reused_again or terminal_again != terminal:
        raise RuntimeError("synthetic smoke terminalization is not idempotent")
    evidence = {
        "schema_version": 1,
        "artifact_role": "W9_PASCAL_V4_SYNTHETIC_LIFECYCLE_SMOKE",
        "status": "NON_SCIENTIFIC",
        "fixture_identity": FIXTURE_ID,
        "eligibility": {
            "SYNTHETIC_ONLY": True,
            "INELIGIBLE_FOR_SELECTION": True,
            "INELIGIBLE_FOR_ER9_SEARCH": True,
            "INELIGIBLE_FOR_ER2_RESULT": True,
            "TEST_NOT_ACCESSED": True,
        },
        "source_commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip(),
        "host": host,
        "platform_node": node,
        "execution_profile": "confessor_pascal_cu126" if require_pascal else "synthetic_cuda_development_only",
        "gpu": gpu,
        "cuda_device": "cuda:0",
        "runtime_root": str(runtime_root.relative_to(REPO)) if REPO in runtime_root.resolve().parents else str(runtime_root.resolve()),
        "committed_epoch_count": 2,
        "restart_resume": "new-process model/optimizer/scaler restored from epoch-0000; epoch-0001 committed exactly once",
        "stale_pointer_repair": True,
        "terminal_id": terminal["terminal_id"],
        "terminal_sha256": sha256_bytes(canonical_bytes(terminal)),
        "terminal_idempotent": True,
        "scientific_data_accesses": 0,
        "test_access": 0,
        "test": "SEALED",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    if evidence_path.exists():
        if json.loads(evidence_path.read_bytes()) != evidence:
            raise RuntimeError("synthetic smoke evidence already exists with different bytes")
    else:
        evidence_path.write_bytes((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("ascii"))
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-nonconfessor-development", action="store_true", help="permit local CUDA mechanism testing; output remains ineligible")
    parser.add_argument("--gpu-uuid", default=None, help="exact registered GPU UUID selected for the Pascal smoke")
    args = parser.parse_args(argv)
    evidence = run(require_pascal=not args.allow_nonconfessor_development, expected_gpu_uuid=args.gpu_uuid)
    print(f"synthetic W9 v4 lifecycle smoke PASS: {evidence['terminal_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
