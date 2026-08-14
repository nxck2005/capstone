#!/usr/bin/env python3
"""Run preregistered paired G8 diagnostics on one explicit CUDA device.

The command is deliberately diagnostic-only: it never uses the production
runner, never publishes a request/result/state artifact, and never touches a
dataset.  It replays the old contract's paired information/noise stimuli and
records per-trial hard block-error indicators for later comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for root in (REPO / "src", REPO / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from baseline.g8_bler_contract import (  # noqa: E402
    PURPOSE_AWGN_IMAG,
    PURPOSE_AWGN_REAL,
    PURPOSE_INFORMATION_BITS,
    derive_seed,
    information_bit_stream,
)
from baseline.g8_campaign import (  # noqa: E402
    load_campaign_manifest,
    load_required_bler_identities,
    rendered_json,
)
from baseline.ldpc.adapter import SionnaLDPCAdapter  # noqa: E402
from baseline.ldpc.modulation import (  # noqa: E402
    bits_per_symbol,
    map_bits,
    max_log_llr,
    n0_from_esn0_db,
)
from config.execution_profiles import (  # noqa: E402
    authenticate_execution_profile,
    canonical_json_bytes,
)
from config.params import get  # noqa: E402

PARITY_PLAN = REPO / "results/baseline/g8/execution_profile_parity_plan.json"


def _load_plan() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = PARITY_PLAN.read_bytes()
    plan = json.loads(raw)
    if raw != rendered_json(plan) or plan.get("scientific_status") != "NON-SCIENTIFIC":
        raise RuntimeError("parity plan is not canonical non-scientific evidence")
    required = load_required_bler_identities()["required_bler_work_units"]
    bindings = plan.get("selected_identity_bindings")
    if not isinstance(bindings, list):
        raise RuntimeError("parity plan has no selected identity bindings")
    return plan, required, bindings


def _run_cell(
    *,
    campaign_id: str,
    unit: dict[str, Any],
    device: str,
    trials: int,
    batch_size: int,
) -> dict[str, Any]:
    identity = unit["identity"]
    k, n = (int(value) for value in identity["k_and_n"])
    q_m = bits_per_symbol(identity["modulation"])
    symbols_per_trial = n // q_m
    info_seed = derive_seed(campaign_id, unit["work_unit_id"], PURPOSE_INFORMATION_BITS)
    real_seed = derive_seed(campaign_id, unit["work_unit_id"], PURPOSE_AWGN_REAL)
    imag_seed = derive_seed(campaign_id, unit["work_unit_id"], PURPOSE_AWGN_IMAG)
    real_rng = np.random.Generator(np.random.Philox(key=real_seed))
    imag_rng = np.random.Generator(np.random.Philox(key=imag_seed))
    noise_scale = math.sqrt(n0_from_esn0_db(float(unit["snr_db"])) / 2.0)
    adapter = SionnaLDPCAdapter(k, n, q_m, int(identity["base_graph"]), device=device)
    if adapter.lifting_size != int(identity["lifting_size"]):
        raise RuntimeError("LDPC lifting size differs from the required identity")
    block_errors: list[int] = []
    bit_errors = 0
    started = time.perf_counter()
    for start in range(0, trials, batch_size):
        count = min(batch_size, trials - start)
        information = information_bit_stream(info_seed, start * k, count * k).reshape(count, k)
        encoded = np.asarray(adapter.encode(information), dtype=np.uint8)
        symbols = map_bits(encoded, identity["modulation"])
        real = real_rng.standard_normal((count, symbols_per_trial))
        imag = imag_rng.standard_normal((count, symbols_per_trial))
        received = symbols + noise_scale * (real + 1j * imag)
        llr = np.asarray(max_log_llr(received, identity["modulation"], n0_from_esn0_db(float(unit["snr_db"]))), dtype=np.float32)
        decoded = np.asarray(adapter.decode(llr), dtype=np.uint8)
        differences = np.not_equal(decoded, information)
        per_trial_bits = np.count_nonzero(differences, axis=1).astype(np.int64)
        bit_errors += int(np.sum(per_trial_bits))
        block_errors.extend((per_trial_bits > 0).astype(np.uint8).tolist())
    elapsed = time.perf_counter() - started
    return {
        "ordinal": unit["ordinal"],
        "work_unit_id": unit["work_unit_id"],
        "identity": identity,
        "snr_db": unit["snr_db"],
        "trials": trials,
        "information_bits": trials * k,
        "bit_errors": bit_errors,
        "block_errors": int(sum(block_errors)),
        "bler": float(sum(block_errors) / trials),
        "ber": float(bit_errors / (trials * k)),
        "block_error_indicators": block_errors,
        "elapsed_s": elapsed,
        "throughput_trials_per_s": float(trials / elapsed) if elapsed else 0.0,
        "stream_seeds": {
            "information_bits": info_seed,
            "awgn_real": real_seed,
            "awgn_imag": imag_seed,
        },
    }


def audit(profile_id: str, device: str, *, trials: int, batch_size: int) -> dict[str, Any]:
    plan, required, bindings = _load_plan()
    if trials != int(plan["paired_trial_count_per_cell"]):
        raise ValueError("final parity audit must use the preregistered trial count")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    manifest = load_campaign_manifest()
    config_hash = hashlib.sha256(canonical_json_bytes({"audit": "execution_profile_parity", "plan_sha256": hashlib.sha256(PARITY_PLAN.read_bytes()).hexdigest()})).hexdigest()
    environment = authenticate_execution_profile(
        profile_id,
        device=device,
        config_hash=config_hash,
        require_openjpeg=False,
        allow_pending_qualification=True,
    )
    units: list[dict[str, Any]] = []
    for binding in bindings:
        ordinal = int(binding["ordinal"])
        source = required[ordinal]
        if binding["work_unit_id"] != source["work_unit_id"] or binding["identity"] != source["identity"] or binding["snr_db"] != source["snr_db"]:
            raise RuntimeError(f"parity binding differs from required identity {ordinal}")
        unit = dict(source)
        unit["ordinal"] = ordinal
        units.append(unit)
    cells = []
    for unit in units:
        cells.append(_run_cell(campaign_id=manifest["campaign_id"], unit=unit, device=device, trials=trials, batch_size=batch_size))
        torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_paired_numerical_parity",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile_id,
        "device": device,
        "environment": environment,
        "campaign_id": manifest["campaign_id"],
        "parity_plan_sha256": hashlib.sha256(PARITY_PLAN.read_bytes()).hexdigest(),
        "required_identity_count": 3213,
        "selected_cell_count": len(cells),
        "paired_trial_count_per_cell": trials,
        "diagnostic_only": True,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
        "cells": cells,
        "criterion": {
            "per_cell_disagreement_rate_max": 0.02,
            "aggregate_disagreement_rate_max": 0.01,
            "waterfall_displacement_db_max": 0.5,
            "interpretation": "qualification-only paired diagnostic; not a hypothesis test",
        },
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    report = audit(args.profile, args.device, trials=args.trials, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"profile": args.profile, "device": args.device, "cells": len(report["cells"]), "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
