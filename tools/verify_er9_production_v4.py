#!/usr/bin/env python3
"""Verify final ER-9 production authority, runtimes, and validation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.run_config import load_experiment  # noqa: E402
from evaluation.downstream_v4 import (  # noqa: E402
    FINAL_PAIR, PRODUCTION_CELLS, PRODUCTION_RUNTIME_ROOT, read_json,
    sha256_file, validate_final_er9_cell, verify_production_authority,
)
from runtime.transactional_epochs import TransactionalEpochStore  # noqa: E402
from training.er9_v4 import ER9V4CandidateTrainer, expected_er9_v4_identity  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

PRODUCTION_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"


def verify_runtime(authority: dict, cell: tuple[int, int]) -> dict:
    train_seed, channel_seed = cell
    record = next(item for item in authority["seed_cells"] if (item["train_seed"], item["channel_seed"]) == cell)
    config = load_experiment(authority["config_path"], train_seed=train_seed, channel_seed=channel_seed)
    identity = expected_er9_v4_identity(
        config=config,
        transmit_dim=FINAL_PAIR["transmit_dim"],
        quantiser_bits=FINAL_PAIR["quantiser_bits"],
        source_binding=authority["source_binding"],
        campaign_id="er9_production_v4",
        run_id=f"er9-production-v4-train{train_seed}-channel{channel_seed}",
        runtime_role=ER9V4CandidateTrainer.PRODUCTION_ROLE,
    )
    runtime = REPO / record["runtime_root"]
    store = TransactionalEpochStore(runtime, identity=identity, total_epochs=100, role=ER9V4CandidateTrainer.PRODUCTION_ROLE)  # literal-ok: AM-96 epochs
    committed = store.inspect()
    if len(committed) != 100:  # literal-ok: AM-96 epochs
        raise ValueError(f"production cell {cell} does not contain 100 committed epochs")
    terminal = read_json(runtime / "run_terminal.json", f"production terminal {cell}")
    if terminal.get("selected_checkpoint_sha256") != committed[int(terminal["selected_epoch"])].checkpoint_sha256:
        raise ValueError("production selected checkpoint differs")
    if terminal.get("applied_optimizer_steps", 0) + terminal.get("grad_scaler_skips", 0) != terminal.get("optimizer_opportunities"):
        raise ValueError("production optimizer/scaler counters differ")
    return terminal


def verify_closeout(authority: dict, terminals: dict[tuple[int, int], dict]) -> dict:
    value = read_json(PRODUCTION_CLOSEOUT, "ER-9 production closeout")
    body = dict(value)
    identifier = body.pop("closeout_id", None)
    if identifier != "er9productionv4closeout-" + canonical_sha256(body):
        raise ValueError("production closeout ID differs")
    if value.get("authority_id") != authority["authority_id"] or value.get("source_commit") != authority["source_commit"]:
        raise ValueError("production closeout authority/source differs")
    if value.get("selected_pair") != FINAL_PAIR or value.get("training_count") != 3:  # literal-ok: exact production cells
        raise ValueError("production closeout scope differs")
    if value.get("stage1_promotion_count") != 0 or value.get("best_seed_selection") is not False:
        raise ValueError("production closeout promoted/selected a seed")
    records = value.get("seed_cells")
    if not isinstance(records, list) or [(item.get("train_seed"), item.get("channel_seed")) for item in records] != list(PRODUCTION_CELLS):
        raise ValueError("production closeout cells differ")
    for cell, record in zip(PRODUCTION_CELLS, records, strict=True):
        terminal = terminals[cell]
        if record.get("promoted_stage1") is not False or record.get("test_access") != 0:
            raise ValueError("production closeout cell policy differs")
        if record.get("selected_checkpoint_sha256") != terminal["selected_checkpoint_sha256"]:
            raise ValueError("production closeout selected checkpoint differs")
        validation_path = REPO / record["validation_path"]
        if sha256_file(validation_path) != record["validation_sha256"]:
            raise ValueError("production closeout validation hash differs")
        validation = read_json(validation_path, f"ER-9 final validation {cell}")
        validate_final_er9_cell(validation, cell=cell)
        if validation["checkpoint"]["sha256"] != terminal["selected_checkpoint_sha256"]:
            raise ValueError("production validation checkpoint is not selected checkpoint")
    if value.get("test") != "SEALED" or value.get("test_access") != 0:
        raise ValueError("production closeout crossed test boundary")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    authority = verify_production_authority(REPO)
    if args.terminal:
        terminals = {}
        for cell in PRODUCTION_CELLS:
            terminals[cell] = verify_runtime(authority, cell)
            path = REPO / f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
            value = read_json(path, f"ER-9 final validation {cell}")
            validate_final_er9_cell(value, cell=cell)
        verify_closeout(authority, terminals)
    print("ER-9 production v4 verifier PASS" + (": three terminal cells and full validation" if args.terminal else ": authority only"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
