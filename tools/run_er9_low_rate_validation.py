#!/usr/bin/env python3
"""AM-100: run the fixed low-rate ER-9 variant once on the full validation split.

The variant is the closed Stage-1 v4 D1024/b2 run at its own selected epoch,
sent at BPSK rate 1/5 (base graph 2) at every SNR in the test grid.  Nothing is
selected: the checkpoint, width, quantiser, modulation and rate are all fixed
by AM-100 before this runs.  The per-SNR evaluation reuses the W10 ER-9
backend unchanged, so rows, noise identities and outage handling are exactly
the W10 ones.  Test stays sealed; this tool never reaches a test loader.

    python tools/run_er9_low_rate_validation.py run --device cuda:0
    python tools/run_er9_low_rate_validation.py verify
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402

OUT_DIR = REPO / "results/learned/er9_low_rate"
SUMMARY_PATH = OUT_DIR / "validation_summary.json"
ROWS_PATH = OUT_DIR / "validation_per_image.jsonl.gz"
STAGE1_EVALUATION = REPO / "results/learned/er9/stage1_v4_evaluations/D1024_b2.json"
STAGE1_TERMINAL_INDEX = REPO / "results/learned/er9/stage1_v4_terminal_index.json"
RUNTIME_ROOT = "checkpoints/er9_pascal_v4/stage1/D1024_b2"
ARTIFACT_ROLE = "AM100_ER9_LOW_RATE_VALIDATION"


def _params() -> dict[str, Any]:
    keys = (
        "system", "role", "transmit_dim", "quantiser_bits", "checkpoint_run",
        "checkpoint_epoch", "checkpoint_sha256", "modulation", "ldpc_rate",
        "payload_bits", "transport_selection", "retraining",
    )
    return {key: get(f"digital_semantic_control.low_rate_variant_{key}") for key in keys}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"AM-100 low-rate validation HOLD: {message}")


def _stage1_binding(params: dict[str, Any]) -> dict[str, Any]:
    """Bind the checkpoint to the closed Stage-1 evidence, not to a local path alone."""

    evaluation = json.loads(STAGE1_EVALUATION.read_bytes())
    _require(evaluation["candidate"] == {"quantiser_bits": params["quantiser_bits"], "transmit_dim": params["transmit_dim"]}, "Stage-1 candidate differs")
    _require(int(evaluation["selected_epoch"]) == int(params["checkpoint_epoch"]), "Stage-1 selected epoch differs")
    _require(evaluation["runtime_root"] == RUNTIME_ROOT, "Stage-1 runtime root differs")
    index_text = STAGE1_TERMINAL_INDEX.read_text()
    _require(params["checkpoint_sha256"] in index_text, "checkpoint SHA-256 is not in the closed Stage-1 terminal index")
    terminal = REPO / RUNTIME_ROOT / "run_terminal.json"
    _require(_sha256(terminal) == evaluation["terminal_sha256"], "Stage-1 run terminal bytes differ")
    terminal_value = json.loads(terminal.read_bytes())
    _require(terminal_value["selected_checkpoint_sha256"] == params["checkpoint_sha256"], "Stage-1 terminal selects a different checkpoint")
    _require(int(terminal_value["selected_epoch"]) == int(params["checkpoint_epoch"]), "Stage-1 terminal selects a different epoch")
    return {
        "stage1_evaluation_path": str(STAGE1_EVALUATION.relative_to(REPO)),
        "stage1_evaluation_sha256": _sha256(STAGE1_EVALUATION),
        "stage1_evaluation_id": evaluation["evaluation_id"],
        "stage1_validation_n_correct_at_7db": int(evaluation["real_chain_validation"]["validation_n_correct"]),
        "run_terminal_sha256": evaluation["terminal_sha256"],
        "entropy_table": evaluation["entropy_model"]["table"],
    }


def run(device: str) -> int:
    import torch

    from baseline.ldpc.transport import build_packet_plan
    from config.run_config import config_hash as run_config_hash, load_experiment
    from evaluation.er9_campaign import fit_entropy_model
    from evaluation.er9_protocol import raw_bound_bits
    from evaluation.w10_backends import W10Execution, er9_unit
    from evaluation.w10_dispatch import ER9_CONFIG, _load_checkpoint_state
    from evaluation.w10_scope import snr_grid
    from models.er9_digital import build_er9_model

    params = _params()
    _require(params["system"] == "er9_digital_low_rate", "variant system value differs")
    _require(params["transport_selection"] == "none_fixed_at_every_snr" and params["retraining"] == "none", "variant is not fixed")
    _require(not SUMMARY_PATH.exists(), f"{SUMMARY_PATH.relative_to(REPO)} already exists; AM-100 runs validation once")
    binding = _stage1_binding(params)

    config = load_experiment(ER9_CONFIG, train_seed=0, channel_seed=0)
    k = int(config.resolved["k"])
    packet = build_packet_plan(k, str(params["modulation"]), str(params["ldpc_rate"]))
    layout = packet.segmentation
    _require(packet.feasible and layout is not None, "BPSK rate 1/5 packet is infeasible")
    _require(int(layout.payload_bits) == int(params["payload_bits"]), "low-rate payload differs from AM-100")
    _require(int(layout.base_graph) == 2 and int(layout.code_blocks) == 1, "low-rate packet is not one base-graph-2 block")
    bound = raw_bound_bits(int(params["transmit_dim"]), int(params["quantiser_bits"]))
    _require(bound <= int(layout.payload_bits), "raw bound does not fit the low-rate payload")

    model = build_er9_model(config, transmit_dim=int(params["transmit_dim"]), quantiser_bits=int(params["quantiser_bits"]), device=device)
    checkpoint = REPO / RUNTIME_ROOT / "epochs" / f"epoch-{int(params['checkpoint_epoch']):04d}" / "checkpoint.pt"
    state = _load_checkpoint_state(checkpoint, str(params["checkpoint_sha256"]))
    incompatible = model.load_state_dict(state, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "checkpoint did not load strictly")
    model.eval()

    entropy, entropy_record = fit_entropy_model(model, config, device=device, num_workers=0)
    _require(entropy_record["table"] == binding["entropy_table"], "refit entropy table differs from the closed Stage-1 table")

    candidate = {"modulation": str(params["modulation"]), "ldpc_rate": str(params["ldpc_rate"]), "packet": packet}
    assets = {
        "model": model,
        "config": config,
        "entropy": entropy,
        "dimension": int(params["transmit_dim"]),
        "quantiser_bits": int(params["quantiser_bits"]),
        "checkpoint_id": str(params["checkpoint_sha256"]),
        "phy_candidates": [candidate],
    }
    context = W10Execution(root=REPO, device=device)
    points: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for snr_db in snr_grid():
            unit = {"system": params["system"], "bw_ratio": "r_1_6", "snr_db": snr_db}
            selection = {float(snr_db): {"modulation": candidate["modulation"], "ldpc_rate": candidate["ldpc_rate"]}}
            aggregate = er9_unit(context, unit, assets=assets, selection_phy=selection)
            rows = aggregate.pop("per_image")
            all_rows.extend(rows)
            points.append({"snr_db": snr_db, **aggregate})
            print(f"{snr_db:>4} dB  {aggregate['n_correct']}/{aggregate['n_total']}  delivered {aggregate['coverage_rate']:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_bytes = b"".join(_canonical(row) for row in all_rows)
    with gzip.GzipFile(ROWS_PATH, "wb", mtime=0) as handle:
        handle.write(rows_bytes)
    summary = {
        "artifact_role": ARTIFACT_ROLE,
        "schema_version": 1,
        "amendment": "AM-100",
        "authorization": "owner decision 2026-10-02: Option B, one validation run of the fixed low-rate variant before G-12",
        "variant": params,
        "config_path": ER9_CONFIG,
        "config_hash": run_config_hash(config),
        "k_symbols": k,
        "packet": packet.metadata(),
        "raw_bound_bits": bound,
        "stage1_binding": {key: value for key, value in binding.items() if key != "entropy_table"},
        "entropy_table_matches_stage1": True,
        "split": "val",
        "seed_cell": {"train_seed": 0, "channel_seed": 0},
        "snr_grid_db": list(snr_grid()),
        "points": points,
        "per_image_path": str(ROWS_PATH.relative_to(REPO)),
        "per_image_rows": len(all_rows),
        "per_image_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "device": str(device),
        "torch": torch.__version__,
        "test": "SEALED",
        "test_access": 0,
    }
    SUMMARY_PATH.write_bytes(json.dumps(summary, indent=1, sort_keys=True).encode() + b"\n")
    print(f"wrote {SUMMARY_PATH.relative_to(REPO)} and {ROWS_PATH.relative_to(REPO)}")
    return 0


def verify() -> int:
    summary = json.loads(SUMMARY_PATH.read_bytes())
    _require(summary["artifact_role"] == ARTIFACT_ROLE, "artifact role differs")
    _require(summary["variant"] == _params(), "summary variant differs from the AM-100 parameters")
    _require(summary["test"] == "SEALED" and summary["test_access"] == 0, "test boundary differs")
    with gzip.open(ROWS_PATH, "rb") as handle:
        raw = handle.read()
    _require(hashlib.sha256(raw).hexdigest() == summary["per_image_sha256"], "per-image bytes differ")
    rows = [json.loads(line) for line in raw.splitlines()]
    denominator = int(get("evaluation.w10_validation_denominator"))
    _require(len(rows) == denominator * len(summary["snr_grid_db"]), "per-image row count differs")
    for point in summary["points"]:
        subset = [row for row in rows if float(row["test_snr_db"]) == float(point["snr_db"])]
        _require(len(subset) == denominator, f"{point['snr_db']} dB denominator differs")
        _require(sum(1 for row in subset if row["correct"]) == point["n_correct"], f"{point['snr_db']} dB n_correct differs")
        _require(all(row["split"] == "val" for row in subset), "a row is not on the validation split")
    print(f"AM-100 low-rate validation PASS: {len(rows)} rows, commit {summary['git_commit']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--device", default="cuda:0")
    sub.add_parser("verify")
    args = parser.parse_args()
    return run(args.device) if args.command == "run" else verify()


if __name__ == "__main__":
    raise SystemExit(main())
