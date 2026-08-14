#!/usr/bin/env python3
"""Non-scientific synthetic qualification for one profile and one GPU (SR-23)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for root in (SRC, REPO / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from baseline.ldpc.adapter import SionnaLDPCAdapter  # noqa: E402
from baseline.ldpc.modulation import map_bits, max_log_llr  # noqa: E402
from config.execution_profiles import (  # noqa: E402
    authenticate_execution_profile,
    canonical_json_bytes,
)
from config.run_config import FrozenMap, config_hash  # noqa: E402
from models.djscc import build_djscc  # noqa: E402
from training.djscc_loss import DJSCCObjective  # noqa: E402
from profile_djscc_g7 import load_profile_config  # noqa: E402


def _synthetic_djscc_config():
    config = load_profile_config(REPO / "configs/learned-g7-profile.yaml")
    resolved = config.resolved.to_dict()
    resolved.update(
        dataset="cifar10",
        bw_ratio="r_1_48",
        k=int(config.parameters["bandwidth"]["k_symbols"]["cifar10"]["r_1_48"]),
    )
    return replace(config, resolved=FrozenMap.from_mapping(resolved))


def _temperature(device_index: int) -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,  # literal-ok: subprocess safety timeout
        ).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _crash_restart_probe(device: str) -> dict[str, bool]:
    program = (
        "import torch,time; d=torch.device(%r); "
        "x=torch.ones((512,512),device=d); print('READY',flush=True); time.sleep(30)"
        % device
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = child.stdout.readline().strip() == "READY" if child.stdout else False
    child.terminate()
    try:
        child.wait(timeout=10)  # literal-ok: process-control timeout
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)  # literal-ok: process-control timeout
    restart = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import torch; x=torch.ones(8,device={device!r}); "
            "assert float(x.sum().cpu()) == 8.0",
        ],
        capture_output=True,
        text=True,
        timeout=30,  # literal-ok: process-control timeout
    ).returncode == 0
    return {"child_reached_cuda": ready, "terminated_child": child.returncode is not None, "fresh_cuda_after_kill": restart}


def _ldpc(device: str) -> dict[str, object]:
    bits = np.random.Generator(np.random.Philox(20260815)).integers(
        0, 2, size=(8, 128), dtype=np.uint8
    )
    adapter = SionnaLDPCAdapter(128, 256, 2, 2, device)
    encoded = adapter.encode(bits)
    symbols = map_bits(encoded, "qpsk")
    llr = max_log_llr(symbols, "qpsk", 0.001)
    decoded = adapter.decode(llr)
    return {
        "encode_pass": encoded.shape == (8, 256),
        "decode_pass": bool(np.array_equal(decoded, bits)),
        "lifting_size": adapter.lifting_size,
        "encoded_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
    }


def _modulation() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, q_m in (("bpsk", 1), ("qpsk", 2), ("qam16", 4)):
        bits = np.tile(np.arange(2**q_m, dtype=np.uint8)[:, None], (1, q_m))
        shifts = np.arange(q_m - 1, -1, -1, dtype=np.uint8)
        labels = ((np.arange(2**q_m, dtype=np.uint8)[:, None] >> shifts) & 1).reshape(-1)
        symbols = map_bits(labels, name)
        hard = (max_log_llr(symbols[None, :], name, 0.001) > 0).astype(np.uint8).reshape(-1)
        result[name] = bool(np.array_equal(hard, labels))
    return result


def _djscc(device: torch.device, *, batch_size: int, iterations: int) -> dict[str, object]:
    config = _synthetic_djscc_config()
    model = build_djscc(config, device=device).train()
    objective = DJSCCObjective.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(20260815)
    inputs = torch.rand((batch_size, 3, 32, 32), generator=generator).to(device)
    targets = torch.arange(batch_size, device=device) % 10
    noise_generator = torch.Generator(device=device).manual_seed(20260815)
    unit_noise = torch.randn(
        (batch_size, int(config.resolved["k"])),
        device=device,
        dtype=torch.complex64,
        generator=noise_generator,
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs, 7.0, unit_noise=unit_noise)
    loss = objective(output, targets, inputs).total
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite_gradients = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero_gradients = any(bool(torch.count_nonzero(g)) for g in gradients)
    optimizer.step()

    with tempfile.TemporaryDirectory(prefix="capstone-profile-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        restored = build_djscc(config, device=device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
        payload = torch.load(path, map_location=device, weights_only=True)
        restored.load_state_dict(payload["model"])
        restored_optimizer.load_state_dict(payload["optimizer"])
        checkpoint_reload = all(
            torch.equal(a, b)
            for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=True)
        )

    model.eval()
    with torch.no_grad():
        first = model(inputs, 7.0, unit_noise=unit_noise).reconstruction
        second = model(inputs, 7.0, unit_noise=unit_noise).reconstruction
    deterministic_repeat = bool(torch.equal(first, second))

    model.train()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        candidate = model(inputs, 7.0, unit_noise=unit_noise)
        candidate_loss = objective(candidate, targets, inputs).total
        candidate_loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "forward_pass": bool(torch.isfinite(output.reconstruction).all()),
        "backward_pass": finite_gradients and nonzero_gradients,
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": finite_gradients,
        "optimizer_step": True,
        "checkpoint_save": True,
        "checkpoint_reload": bool(checkpoint_reload),
        "checkpoint_sha256": checkpoint_sha,
        "same_seed_repeatability": deterministic_repeat,
        "safe_batch_size": batch_size,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "sustained_iterations": iterations,
        "sustained_elapsed_s": elapsed,
        "throughput_images_per_s": batch_size * iterations / elapsed,
    }


def qualify(profile_id: str, device_name: str, batch_size: int, iterations: int) -> dict[str, object]:
    device = torch.device(device_name)
    config = _synthetic_djscc_config()
    digest = config_hash(config)
    environment = authenticate_execution_profile(
        profile_id,
        device=device_name,
        config_hash=digest,
        require_openjpeg=False,
        allow_pending_qualification=True,
    )
    before = _temperature(device.index or 0)
    tensor = torch.arange(1024, device=device, dtype=torch.float32).reshape(32, 32)
    cuda_pass = bool(torch.isfinite(tensor @ tensor.T).all())
    ldpc = _ldpc(device_name)
    modulation = _modulation()
    djscc = _djscc(device, batch_size=batch_size, iterations=iterations)
    crash = _crash_restart_probe(device_name)
    after = _temperature(device.index or 0)
    return {
        "schema_version": 1,
        "artifact_kind": "execution_profile_qualification",
        "scientific_status": "NON-SCIENTIFIC",
        "g8_coverage": 0,
        "test_access_count": 0,
        "validation_count": 0,
        "selection_count": 0,
        "training_campaign_count": 0,
        "synthetic_data_only": True,
        "execution_profile_id": profile_id,
        "device": device_name,
        "environment": environment,
        "checks": {
            "real_cuda_tensor_operations": cuda_pass,
            "ldpc": ldpc,
            "project_modulation_demodulation": modulation,
            "djscc": djscc,
            "crash_kill_restart": crash,
            "temperature_c": {"before": before, "after": after},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0:
        parser.error("batch size and iterations must be positive")
    report = qualify(args.profile, args.device, args.batch_size, args.iterations)
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"output": str(args.output), "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
