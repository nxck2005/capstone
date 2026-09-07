#!/usr/bin/env python3
"""Freeze the exact ER-9 Stage-1 candidates before optimizer steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402


DEFAULT_OUTPUT = REPO / "results/learned/er9/er9_stage1_execution_authorization_v2.json"
DEFAULT_MANIFEST = REPO / "results/learned/er9/er_execution_source_manifest_v2.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _immutable(value: dict[str, Any], path: Path) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable ER-9 authorization exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_authorization(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("ER source manifest must be a JSON object")
    implementation_commit = str(manifest["source_commit"])
    current_commit = _git("rev-parse", "HEAD")
    if current_commit != implementation_commit:
        raise ValueError("Stage-1 authorization must be generated at the implementation source commit")
    config = load_experiment(
        "configs/er9-digital.yaml",
        train_seed=0,  # literal-ok: first preregistered train seed
        channel_seed=0,  # literal-ok: first preregistered channel seed
    )
    resolved = config.resolved
    train_seeds = tuple(int(value) for value in get("evaluation.train_seeds"))
    channel_seeds = tuple(int(value) for value in get("evaluation.channel_seeds"))
    zipped = tuple(zip(train_seeds, channel_seeds, strict=True))
    if not zipped or zipped[0] != (0, 0):  # literal-ok: first zipped seed cell
        raise ValueError("the first configured zipped seed cell is not train/channel 0/0")
    floor = packetisation_floor(int(resolved["k"]))
    admissible = feasible_pairs(floor.payload_bits)
    stage1 = stage1_candidates(floor.payload_bits)
    all_pairs = all_configured_pairs()
    config_digest = config_hash(config)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    body: dict[str, Any] = {
        "schema_version": 1,  # literal-ok: Stage-1 authorization schema
        "artifact_role": "ER9_STAGE1_EXECUTION_AUTHORIZATION",
        "status": "FROZEN_BEFORE_FIRST_ER9_OPTIMIZER_STEP",
        "authorization_scope": "ER9_STAGE1_ONLY",
        "source_commit": implementation_commit,
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": manifest_sha,
        "config_path": "configs/er9-digital.yaml",
        "config_hash": config_digest,
        "execution_profile_selection": {
            "execution_profile_id": "local_4060_cu130",
            "device": "cuda:0",
            "status": "selected_before_first_scientific_measurement",
            "sole_writer": True,
        },
        "dataset": resolved["dataset"],
        "split": "train",
        "ratio": resolved["bw_ratio"],
        "k_symbols": int(resolved["k"]),
        "evaluation_snr_db": int(resolved["train_snr_db"]),
        "search_seed_cell": {"train_seed": 0, "channel_seed": 0},  # literal-ok: first zipped seed cell
        "packet_floor": floor.as_dict(),
        "metadata_bits": 1,  # literal-ok: AM-96 framing selector bit
        "configured_pair_count": len(all_pairs),
        "admissible_pair_count": len(admissible),
        "admissible_pairs": [candidate.as_dict() for candidate in admissible],
        "rejected_pairs": [
            {
                **candidate.as_dict(),
                "reason": "raw_bound_exceeds_A_floor",
            }
            for candidate in all_pairs
            if candidate not in admissible
        ],
        "stage1_rule": {
            "quantiser_bits": 2,  # literal-ok: AM-96 stage-1 minimum width
            "candidate_order": "ascending_numeric_transmit_dim",
            "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
            "tie_break": "smallest_transmit_dim",
            "cross_product": False,
        },
        "stage1_candidates": [candidate.as_dict() for candidate in stage1],
        "stage1_training_count": len(stage1),
        "stage2_authorization": "created_only_after_terminal_stage1_selection",
        "final_seed_scope": "all_three_existing_zipped_seed_cells_after_pair_selection",
        "pre_execution_counters": {
            "er9_training": 0,
            "randomized_er2_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "test": "SEALED",
    }
    body["authorization_id"] = "er9stage1auth-" + canonical_sha256(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    output = args.output if args.output.is_absolute() else REPO / args.output
    value = build_authorization(manifest)
    _immutable(value, output)
    print(f"ER-9 Stage-1 authorization written: {value['authorization_id']}")
    print(f"Stage-1 candidate count: {value['stage1_training_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
