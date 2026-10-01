#!/usr/bin/env python3
"""Read-only Discord monitor for the single G-12 test campaign on Confessor.

Posts rich embeds (progress bar, ETA, per-arm progress, worker health, GPU
load, latest units) to ``DISCORD_WEBHOOK_URL``.  It only reads the campaign's
runtime files and process table; it never writes into the repository and never
restarts a worker.

    python3 monitor.py            # run as a service
    python3 monitor.py --preview  # post one embed now and exit
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("G12_REPO", "/home/nick/projects/capstone"))
RUNTIME = REPO / "checkpoints/g12_campaign"
UNITS = RUNTIME / "units"
MARKER = RUNTIME / "campaign.json"
STATE = Path(os.environ.get("G12_MONITOR_STATE", str(Path(__file__).resolve().parent / "monitor-state.json")))
POLL_SECONDS = int(os.environ.get("G12_MONITOR_POLL_SECONDS", "30"))
REPORT_SECONDS = int(os.environ.get("G12_MONITOR_REPORT_SECONDS", "900"))
TOTAL_UNITS = 441

# role -> (label, units)
ARMS = [
    ("er1_headline_learned", "DJSCC (AI link) · 3 seeds", 63),
    ("er1_headline_classical", "JPEG 2000 + LDPC · 3 seeds", 63),
    ("er9_control_h4", "Task-aware digital · 3 seeds", 63),
    ("br16_fixed_mcs_h2", "Fixed-MCS (H2) · 3 seeds", 63),
    ("er11_efficiency_learned", "DJSCC r=1/24", 21),
    ("er11_efficiency_classical", "JPEG 2000 r=1/24", 21),
    ("er2_snr_randomised", "SNR-randomised DJSCC", 21),
    ("sr16_papr_secondary", "PAPR-capped DJSCC", 21),
    ("br9_fixed_modulation", "Fixed modulation", 21),
    ("dec9_jpeg_secondary", "JPEG secondary", 21),
    ("er12_label_upper_bound", "Label-only bound", 21),
    ("er4_reconstruction_ablation", "Reconstruction ablation", 21),
    ("am100_low_rate_secondary", "Low-rate digital (AM-100)", 21),
]
STEM = re.compile(r"^(\d{3})-(.+)-(r_1_\d+)-c(\d)-snr([+-]\d{2})\.json\.gz$")
BLUE, GREEN, RED, AMBER = 0x2A78D6, 0x1BAF7A, 0xD64545, 0xE8A33D


def now() -> datetime:
    return datetime.now(timezone.utc)


def log(message: str) -> None:
    print(f"{now().isoformat()} {message}", flush=True)


def load_state() -> dict:
    try:
        value = json.loads(STATE.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".state-", dir=STATE.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, sort_keys=True)
    os.replace(name, STATE)


def role_of(stem_system: str, ordinal: int) -> str:
    cumulative = 0
    for role, _label, count in ARMS:
        cumulative += count
        if ordinal < cumulative:
            return role
    return "unknown"


def completed_units() -> list[dict]:
    units = []
    if not UNITS.is_dir():
        return units
    for path in UNITS.glob("*.json.gz"):
        match = STEM.match(path.name)
        if not match:
            continue
        ordinal = int(match.group(1))
        units.append({
            "path": path,
            "ordinal": ordinal,
            "system": match.group(2),
            "ratio": match.group(3),
            "cell": int(match.group(4)),
            "snr": int(match.group(5)),
            "role": role_of(match.group(2), ordinal),
            "mtime": path.stat().st_mtime,
        })
    return sorted(units, key=lambda unit: unit["mtime"])


def unit_accuracy(path: Path) -> tuple[int, int] | None:
    try:
        with gzip.open(path, "rb") as handle:
            value = json.loads(handle.read())
        stream = value["streams"][0]
        return int(stream["n_correct"]), len(stream["rows"])
    except (OSError, ValueError, KeyError):
        return None


def workers() -> list[str]:
    try:
        result = subprocess.run(["pgrep", "-fa", "[r]un_g12_campaign.py run"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in result.stdout.splitlines():
        match = re.search(r"--worker (\d+).*--device (\S+)", line)
        if match:
            found.append(f"w{match.group(1)}@{match.group(2)}")
    return sorted(found)


def gpus() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ["GPU query unavailable"]
    lines = []
    for row in result.stdout.splitlines():
        index, name, util, used, total, temp = [field.strip() for field in row.split(",")]
        short = name.replace("NVIDIA ", "").replace("GeForce ", "")
        lines.append(f"`{index}` {short} — {util}% · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GB · {temp}°C")
    return lines


def bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def eta_text(units: list[dict], started: float | None) -> str:
    if len(units) < 3 or started is None:
        return "estimating…"
    recent = units[-30:]
    span = recent[-1]["mtime"] - recent[0]["mtime"]
    rate = (len(recent) - 1) / span if span > 0 else 0.0
    remaining = TOTAL_UNITS - len(units)
    if rate <= 0 or remaining <= 0:
        return "—"
    seconds = remaining / rate
    finish = datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).astimezone()
    return f"~{seconds / 3600:.1f} h (≈ {finish:%a %H:%M})"


def duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m"


def build_embed(event: str, units: list[dict], alive: list[str], state: dict) -> dict:
    done = len(units)
    fraction = done / TOTAL_UNITS
    marker = {}
    try:
        marker = json.loads(MARKER.read_bytes())
    except (OSError, ValueError):
        pass
    started = state.get("first_seen")
    elapsed = duration(time.time() - started) if started else "—"
    by_role: dict[str, int] = {}
    for unit in units:
        by_role[unit["role"]] = by_role.get(unit["role"], 0) + 1
    arm_lines = []
    for role, label, count in ARMS:
        have = by_role.get(role, 0)
        icon = "✅" if have == count else ("🔄" if have else "⏳")
        arm_lines.append(f"{icon} {label} — {have}/{count}")
    latest = []
    for unit in reversed(units[-4:]):
        accuracy = unit_accuracy(unit["path"])
        score = f"{accuracy[0] / accuracy[1]:.1%}" if accuracy else "?"
        latest.append(f"`{unit['ordinal']:03d}` {unit['system']} {unit['ratio']} c{unit['cell']} {unit['snr']:+d} dB → **{score}**")
    if event == "DONE":
        colour, title = GREEN, "✅ G-12 test campaign complete"
    elif event == "WORKER_LOST":
        colour, title = RED, "⚠️ G-12 worker stopped"
    elif event == "START":
        colour, title = AMBER, "🚀 G-12 test campaign started"
    else:
        colour, title = BLUE, "🧪 G-12 test campaign"
    description = (
        f"`{bar(fraction)}` **{done}/{TOTAL_UNITS}** ({fraction:.0%})\n"
        f"⏱️ elapsed {elapsed} · ETA {eta_text(units, started)}"
    )
    fields = [
        {"name": "Arms", "value": "\n".join(arm_lines[:4]), "inline": True},
        {"name": "​", "value": "\n".join(arm_lines[4:]), "inline": True},
        {"name": f"Workers ({len(alive)} alive)", "value": " ".join(f"`{name}`" for name in alive) or "none running", "inline": False},
        {"name": "GPUs", "value": "\n".join(gpus()), "inline": False},
    ]
    if latest:
        fields.append({"name": "Latest units (primary scorer, test split)", "value": "\n".join(latest), "inline": False})
    footer = f"freeze {str(marker.get('freeze_id', '—'))[:22]}… · commit {str(marker.get('code_commit', '—'))[:10]} · read-only monitor"
    return {"title": title, "description": description, "color": colour, "fields": fields, "footer": {"text": footer}, "timestamp": now().isoformat()}


def send(embed: dict) -> bool:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        log("discord_error=DISCORD_WEBHOOK_URL_missing")
        return False
    payload = json.dumps({"username": "G-12 campaign", "embeds": [embed]}).encode()
    request = urllib.request.Request(webhook, data=payload, method="POST", headers={"Content-Type": "application/json", "User-Agent": "capstone-g12-read-only-monitor"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            log(f"discord_status={response.status}")
            return 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001 - never log the webhook URL
        log(f"discord_error={type(exc).__name__}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    state = load_state()
    if args.preview:
        units = completed_units()
        send(build_embed("PROGRESS", units, workers(), state))
        return 0
    state.setdefault("last_report", 0.0)
    state.setdefault("last_decile", -1)
    state.setdefault("arms_done", [])
    while True:
        units = completed_units()
        alive = workers()
        if alive and not state.get("first_seen"):
            state["first_seen"] = time.time()
            send(build_embed("START", units, alive, state))
            state["last_report"] = time.time()
        if state.get("first_seen"):
            decile = int(10 * len(units) / TOTAL_UNITS)
            counts: dict[str, int] = {}
            for unit in units:
                counts[unit["role"]] = counts.get(unit["role"], 0) + 1
            finished_arms = sorted(role for role, _label, count in ARMS if counts.get(role, 0) == count)
            new_arm = set(finished_arms) - set(state["arms_done"])
            if len(units) >= TOTAL_UNITS and not state.get("done_sent"):
                state["done_sent"] = send(build_embed("DONE", units, alive, state))
            elif not alive and len(units) < TOTAL_UNITS and not state.get("lost_sent"):
                state["lost_sent"] = send(build_embed("WORKER_LOST", units, alive, state))
            elif decile > state["last_decile"] or new_arm or time.time() - state["last_report"] >= REPORT_SECONDS:
                send(build_embed("PROGRESS", units, alive, state))
                state["last_report"] = time.time()
            if alive:
                state["lost_sent"] = False
            state["last_decile"] = decile
            state["arms_done"] = finished_arms
        save_state(state)
        if state.get("done_sent"):
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
