#!/usr/bin/env python3
"""Verify the final v4 randomized ER-2 lifecycle, never the legacy reader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import load_experiment  # noqa: E402
from evaluation.downstream_v4 import ER2_RUNTIME_ROOT, EXPECTED_SNR_GRID, read_json, require, sha256_file, verify_er2_authority  # noqa: E402
from runtime.transactional_epochs import TransactionalEpochStore  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.er2_v4 import ER2V4RandomizedTrainer, er2_v4_assignment_audit, expected_er2_v4_identity  # noqa: E402

ROOT = REPO / "results/learned/er2_randomized"


def verify_selected(path: Path, authority: dict, terminal: dict) -> dict:
    value = read_json(path, "ER-2 selected checkpoint")
    body = dict(value)
    identifier = body.pop("selection_id", None)
    require(identifier == "er2selectedv4-" + canonical_sha256(body), "ER-2 selection ID differs")
    require(value.get("artifact_role") == "ER2_RANDOMIZED_SELECTED_CHECKPOINT_V4", "ER-2 selection role differs")
    require(value.get("authority_id") == authority["authority_id"] and value.get("source_commit") == authority["source_commit"], "ER-2 selection authority/source differs")
    require((value.get("train_seed"), value.get("channel_seed")) == (0, 0), "ER-2 selection seed differs")
    require(value.get("selected_epoch") == terminal["selected_epoch"] and value.get("checkpoint_sha256") == terminal["selected_checkpoint_sha256"], "ER-2 selected checkpoint differs from terminal")
    require(sha256_file(REPO / value["checkpoint_path"]) == value["checkpoint_sha256"], "ER-2 selected checkpoint bytes differ")
    require(str(value.get("task_head_identity", "")).startswith("er9taskhead-"), "ER-2 own task-head identity differs")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "ER-2 selection accessed test")
    return value


def verify_audit(path: Path, authority: dict, *, recompute: bool) -> dict:
    value = read_json(path, "ER-2 v4 assignment audit")
    body = dict(value)
    identifier = body.pop("audit_id", None)
    require(identifier == "er2assignmentv4-" + canonical_sha256(body), "ER-2 audit ID differs")
    require(value.get("artifact_role") == "ER2_RANDOMIZED_SNR_ASSIGNMENT_AUDIT_V4", "ER-2 audit role differs")
    require(value.get("domain") == [1, 4, 7, 13, 19] and value.get("distribution") == "discrete_uniform", "ER-2 audit domain/distribution differs")  # literal-ok: AM-95 exact domain
    require(value.get("unit") == "per_sample_per_epoch" and value.get("batching_independent") is True and value.get("channel_noise_separately_keyed") is True, "ER-2 assignment semantics differ")
    require(value.get("assignment_count") == value.get("sample_count") * value.get("epoch_count"), "ER-2 assignment count differs")
    require(value.get("authority_id") == authority["authority_id"] and value.get("source_commit") == authority["source_commit"], "ER-2 audit authority/source differs")
    require(value.get("test_access") == 0, "ER-2 audit accessed test")
    if recompute:
        config = load_experiment(authority["config_path"], train_seed=0, channel_seed=0)
        expected = er2_v4_assignment_audit(config)
        for key, child in expected.items():
            require(value.get(key) == child, f"ER-2 assignment recomputation differs at {key}")
    return value


def verify_validation(path: Path, authority: dict, selected: dict) -> dict:
    value = read_json(path, "ER-2 v4 validation")
    body = dict(value)
    identifier = body.pop("validation_id", None)
    require(identifier == "er2validationv4-" + canonical_sha256(body), "ER-2 validation ID differs")
    require(value.get("artifact_role") == "ER2_RANDOMIZED_VALIDATION_ONLY_EVIDENCE_V4", "ER-2 validation role differs")
    require(value.get("authority_id") == authority["authority_id"] and value.get("source_commit") == authority["source_commit"], "ER-2 validation authority/source differs")
    require(value.get("selected_checkpoint") == {"path": str((ROOT / "er2_selected_checkpoint_v4.json").relative_to(REPO)), "sha256": sha256_file(ROOT / "er2_selected_checkpoint_v4.json"), "checkpoint_sha256": selected["checkpoint_sha256"]}, "ER-2 validation selection binding differs")
    require(value.get("snr_grid_db") == list(EXPECTED_SNR_GRID), "ER-2 validation grid differs")
    require(value.get("scientific_training_run_count") == 1 and value.get("validation_only") is True, "ER-2 validation scope differs")
    curves = value.get("curves")
    require(isinstance(curves, list) and [row.get("snr_db") for row in curves] == list(EXPECTED_SNR_GRID), "ER-2 curve order differs")
    stable_ids = None
    for curve in curves:
        outcomes = curve.get("outcomes")
        require(isinstance(outcomes, list) and len(outcomes) == int(get("datasets.imagenette160.val_images")), "ER-2 validation denominator differs")
        ids = [row.get("stable_sample_id") for row in outcomes]
        require(len(set(ids)) == len(ids) and ids == sorted(ids), "ER-2 stable IDs differ")
        if stable_ids is None:
            stable_ids = ids
        require(ids == stable_ids, "ER-2 trajectories are incomplete")
        require(sum(int(row.get("correct")) for row in outcomes) == curve.get("n_correct") and curve.get("n_total") == len(outcomes), "ER-2 correctness/count differs")
        require(all(row.get("checkpoint_id") == selected["checkpoint_sha256"] and row.get("task_head_identity") == selected["task_head_identity"] and row.get("test_access") == 0 for row in outcomes), "ER-2 outcomes are not selected-checkpoint/task-head/test bound")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "ER-2 validation test boundary differs")
    return value


def verify_completion(path: Path, authority: dict) -> dict:
    value = read_json(path, "ER-2 v4 completion")
    body = dict(value)
    identifier = body.pop("completion_id", None)
    require(identifier == "er2completionv4-" + canonical_sha256(body), "ER-2 completion ID differs")
    require(value.get("artifact_role") == "ER2_RANDOMIZED_V4_COMPLETION" and value.get("scientific_training_run_count") == 1, "ER-2 completion scope differs")
    require(value.get("authority_id") == authority["authority_id"] and value.get("source_commit") == authority["source_commit"] and value.get("runtime_root") == ER2_RUNTIME_ROOT, "ER-2 completion authority/source/runtime differs")
    for field in ("terminal", "selected_checkpoint", "assignment_audit", "validation"):
        record = value.get(field)
        require(isinstance(record, dict), f"ER-2 completion lacks {field}")
        require(sha256_file(REPO / record["path"]) == record["sha256"], f"ER-2 completion {field} hash differs")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "ER-2 completion test boundary differs")
    return value


def verify_runtime(authority: dict) -> dict:
    config = load_experiment(authority["config_path"], train_seed=0, channel_seed=0)
    identity = expected_er2_v4_identity(config=config, source_binding=authority["source_binding"], campaign_id="er2_randomized_v4", run_id="er2-randomized-v4-train0-channel0")
    epochs = int(config.parameters["learned_system"]["epochs"][config.resolved["dataset"]])
    store = TransactionalEpochStore(REPO / ER2_RUNTIME_ROOT, identity=identity, total_epochs=epochs, role=ER2V4RandomizedTrainer.ROLE)
    committed = store.inspect()
    require(len(committed) == epochs, "ER-2 committed epoch count differs")
    terminal = read_json(REPO / ER2_RUNTIME_ROOT / "run_terminal.json", "ER-2 terminal")
    require(terminal.get("selected_checkpoint_sha256") == committed[int(terminal["selected_epoch"])].checkpoint_sha256, "ER-2 selected checkpoint differs")
    require(terminal.get("applied_optimizer_steps") + terminal.get("grad_scaler_skips") == terminal.get("optimizer_opportunities"), "ER-2 optimizer/scaler counters differ")
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-only", action="store_true")
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args(argv)
    authority = verify_er2_authority(REPO)
    if not args.authority_only:
        terminal = verify_runtime(authority)
        selected = verify_selected(ROOT / "er2_selected_checkpoint_v4.json", authority, terminal)
        verify_audit(ROOT / "er2_snr_assignment_audit_v4.json", authority, recompute=not args.skip_recompute)
        verify_validation(ROOT / "er2_randomized_validation_v4.json", authority, selected)
        verify_completion(ROOT / "er2_randomized_completion_v4.json", authority)
    print("randomized ER-2 v4 verifier PASS" + (": authority only" if args.authority_only else ": one run and full validation"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
