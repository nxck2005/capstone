#!/usr/bin/env python3
"""Read-only host monitor for detached G8_F/F2 BR-12 training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path("/home/nick/projects/capstone")
RUNTIME = REPO / "results/baseline/g8_f/f2_runtime"
OPS = Path("/home/nick/g8-f-f2-ops")
SESSION = "g8f-f2"
TOTAL_EPOCHS = 20
TOTAL_STEPS = 6900
STALL_SECONDS = 20 * 60


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _tmux_alive() -> bool:
    command = shutil.which("tmux") or "/usr/bin/tmux"
    return subprocess.run([command, "has-session", "-t", SESSION], capture_output=True, check=False).returncode == 0


def state() -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    progress_path = RUNTIME / "progress.json"
    progress = _read_json(progress_path) or {}
    pid = _read_int(OPS / "worker.pid")
    exit_status = _read_int(OPS / "exit.status")
    age = None
    if progress_path.is_file():
        age = max(0.0, now.timestamp() - progress_path.stat().st_mtime)
    completed_steps = int(progress.get("total_optimizer_steps", 0) or 0)
    worker_alive = _alive(pid)
    completed = progress.get("status") == "COMPLETED" and completed_steps == TOTAL_STEPS and exit_status == 0
    unexpected_exit = not worker_alive and exit_status not in (None, 0) and not completed
    stalled = worker_alive and age is not None and age > STALL_SECONDS
    disk = shutil.disk_usage(REPO)
    return {
        "campaign": "G8_F / F2 / BR-12",
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "worker_alive": worker_alive,
        "tmux_session": SESSION,
        "tmux_alive": _tmux_alive(),
        "pid": pid,
        "gpu_profile": "confessor_pascal_cu126 / cuda:0 / NVIDIA TITAN Xp",
        "completed_epoch": progress.get("completed_epoch", -1),
        "current_epoch": progress.get("current_epoch"),
        "step_in_epoch": progress.get("step_in_epoch", 0),
        "steps_per_epoch": progress.get("steps_per_epoch", 345),
        "total_optimizer_steps": completed_steps,
        "expected_optimizer_steps": TOTAL_STEPS,
        "latest_train_loss": progress.get("latest_train_loss"),
        "latest_validation_top1": progress.get("latest_validation_top1"),
        "best_validation_top1": progress.get("best_validation_top1"),
        "best_epoch": progress.get("best_epoch"),
        "latest_checkpoint": progress.get("latest_checkpoint"),
        "progress_age_seconds": age,
        "stalled": stalled,
        "unexpected_exit": unexpected_exit,
        "exit_status": exit_status,
        "completed": completed,
        "free_disk_gib": disk.free / (1024 ** 3),
        "protected_actions": "monitor is read-only; no restart, checkpoint selection, fallback, F3, or pass two",
    }


def _message(current: dict[str, Any], prior: dict[str, Any] | None) -> str | None:
    if current["completed"] and not (prior or {}).get("completed"):
        return f"F2 COMPLETED — authenticated worker exit 0; {TOTAL_STEPS}/{TOTAL_STEPS} optimizer steps. F3/pass two remain closed."
    if current["unexpected_exit"] and not (prior or {}).get("unexpected_exit"):
        return f"F2 UNEXPECTED EXIT / HOLD — PID {current['pid']}, status {current['exit_status']}; monitor will not restart it."
    if current["stalled"] and not (prior or {}).get("stalled"):
        return f"F2 STALLED / HOLD — no progress for {current['progress_age_seconds']:.0f}s; monitor will not restart it."
    if current["worker_alive"] and not (prior or {}).get("worker_alive"):
        return f"F2 STARTED — PID {current['pid']}, tmux {current['tmux_session']}, Pascal cuda:0; contract frozen; pass two/test closed."
    epoch = current.get("completed_epoch")
    if current["worker_alive"] and epoch is not None and epoch != (prior or {}).get("completed_epoch") and epoch >= 0:
        return f"F2 progress — epoch {epoch + 1}/{TOTAL_EPOCHS} completed, steps {current['total_optimizer_steps']}/{TOTAL_STEPS}, val={current['latest_validation_top1']}, best={current['best_validation_top1']} at epoch {current['best_epoch']}."
    return None


def _notify(message: str) -> int:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("G8_DISCORD_WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("Discord webhook environment variable is absent")
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"content": message}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "capstone-g8-f2-monitor"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return int(response.status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--state-file", type=Path, default=Path("/home/nick/g8-discord-monitor/f2_state.json"))
    args = parser.parse_args()
    current = state()
    prior = _read_json(args.state_file)
    message = _message(current, prior)
    if args.notify and message:
        current["notification"] = {"message": message, "http_status": _notify(message)}
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.state_file)
    print(json.dumps(current, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
