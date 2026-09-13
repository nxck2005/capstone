#!/usr/bin/env python3
"""Run a CPU-only, synthetic, permanently ineligible downstream lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import EXPECTED_SNR_GRID, FINAL_PAIR, PRODUCTION_CELLS, immutable_write, validate_final_er9_cell  # noqa: E402
from training.deterministic_core import canonical_sha256, state_tree_sha256  # noqa: E402
from training.w9_v4 import W9V4TrainingLoop, W9V4TrainingRuntime  # noqa: E402

RUNTIME = REPO / "checkpoints/smoke/final_downstream_v4"
EVIDENCE = REPO / "results/learned/w9/downstream_successor_synthetic_smoke.json"


def _one_cell(cell: tuple[int, int]) -> dict:
    root = RUNTIME / f"train{cell[0]}_channel{cell[1]}"
    identity = {"fixture": "final_downstream_v4", "cell": list(cell), "eligibility": "SYNTHETIC_ONLY_INELIGIBLE", "test_access": 0}

    def construct(resume: bool):
        torch.manual_seed(700 + cell[0])
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        runtime = W9V4TrainingRuntime(root, identity=identity, total_epochs=2, role="W9_FINAL_DOWNSTREAM_SYNTHETIC")

        def epoch(index: int):
            inputs = torch.tensor([[1.0, float(index)], [0.0, 1.0]])
            labels = torch.tensor([0, 1])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), labels)
            loss.backward(); optimizer.step()
            return {"epoch": index, "optimizer_opportunities": 1, "applied_optimizer_steps": 1, "grad_scaler_skips": 0, "validation_n_correct": 1 + (index % 2), "test_access": 0}

        loop = W9V4TrainingLoop(runtime, model=model, optimizer=optimizer, scaler=None, total_epochs=2, train_epoch=epoch, resume=resume)
        return model, optimizer, runtime, loop

    model, optimizer, runtime, loop = construct(False)
    if loop.run(max_epochs=1) is not None:
        raise RuntimeError("synthetic interruption did not stop after one epoch")
    before = state_tree_sha256({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    model2, optimizer2, runtime2, loop2 = construct(True)
    after = state_tree_sha256({"model": model2.state_dict(), "optimizer": optimizer2.state_dict()})
    if before != after or loop2.completed_epoch != 0:
        raise RuntimeError("synthetic exact resume failed")
    terminal = loop2.run(max_epochs=1)
    if terminal is None:
        raise RuntimeError("synthetic terminalization failed")
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    runtime2.restore_epoch(int(terminal["selected_epoch"]), restored, restored_optimizer, None)
    return {"cell": list(cell), "runtime_root": str(root.relative_to(REPO)), "resume_state_sha256": before, "selected_epoch": terminal["selected_epoch"], "selected_checkpoint_sha256": terminal["selected_checkpoint_sha256"], "selected_checkpoint_loaded": True}


def _validation_fixture(checkpoint_id: str) -> dict:
    ids = [f"synthetic-{index:04d}" for index in range(1000)]
    points = []
    for snr in EXPECTED_SNR_GRID:
        rows = [{"stable_sample_id": stable, "split": "val", "test_snr_db": snr, "run_id": f"synthetic-run-{snr}", "correct": False} for stable in ids]
        points.append({
            "snr_db": snr,
            "validation_total": 1000,
            "validation_n_correct": 0,
            "run_id": f"synthetic-run-{snr}",
            "checkpoint_id": checkpoint_id,
            "task_head_identity": "er9taskhead-" + "f" * 64,
            "row_binding": {
                "checkpoint_id": checkpoint_id,
                "train_seed": 0,
                "channel_seed": 0,
                "transmit_dim": 2048,
                "quantiser_bits": 2,
                "task_head_identity": "er9taskhead-" + "f" * 64,
                "test_access": 0,
            },
            "per_image": rows,
            "test_access": 0,
        })
    return {"artifact_role": "ER9_FINAL_PRODUCTION_VALIDATION_CELL_V4", "train_seed": 0, "channel_seed": 0, "selected_pair": FINAL_PAIR, "checkpoint": {"sha256": checkpoint_id}, "task_head": {"kind": "own_task_head", "identity": "er9taskhead-" + "f" * 64}, "snr_grid_db": list(EXPECTED_SNR_GRID), "points": points, "test": "SEALED", "test_access": 0}


def main() -> int:
    if RUNTIME.exists() or RUNTIME.is_symlink() or EVIDENCE.exists() or EVIDENCE.is_symlink():
        raise SystemExit("synthetic downstream smoke custody already exists; it is immutable")
    cells = [_one_cell(cell) for cell in PRODUCTION_CELLS]
    if len({item["runtime_root"] for item in cells}) != 3 or len({item["selected_checkpoint_sha256"] for item in cells}) != 3:
        raise RuntimeError("synthetic production identities are not isolated")
    validation = _validation_fixture(cells[0]["selected_checkpoint_sha256"])
    validate_final_er9_cell(validation, cell=(0, 0))
    body = {
        "schema_version": 1, "artifact_role": "FINAL_DOWNSTREAM_V4_SYNTHETIC_LIFECYCLE_SMOKE",
        "status": "NON_SCIENTIFIC", "eligibility": {"synthetic_only": True, "scientific_use": False, "selection_use": False},
        "production_cells": cells, "fresh_start": True, "transactional_checkpoint": True, "exact_resume": True,
        "terminalization": True, "selected_checkpoint_loading": True, "final_validation_plumbing": True,
        "er2_assignment_plumbing": "covered_by_keyed_selector_synthetic_tests", "scientific_dataset_reads": 0,
        "scientific_training": 0, "test": "SEALED", "test_access": 0,
    }
    body["smoke_id"] = "downstreamsmoke-" + canonical_sha256(body)
    immutable_write(EVIDENCE, body)
    print(f"synthetic downstream lifecycle smoke PASS: {body['smoke_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
