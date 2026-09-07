#!/usr/bin/env python3
"""Verify the single validation-only AM-95 randomized ER-2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from training.er2_randomized import snr_assignment_audit  # noqa: E402


RESULT_ROOT = REPO / "results/learned/er2_randomized"
COMPLETION_DEFAULT = RESULT_ROOT / "er2_randomized_completion.json"
AUDIT_DEFAULT = RESULT_ROOT / "er2_snr_assignment_audit.json"
VALIDATION_DEFAULT = RESULT_ROOT / "er2_randomized_validation.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(path == root.resolve() or root.resolve() in path.parents, "ER-2 artifact escapes its runtime")
    return path


def verify_completion(path: Path, audit_path: Path) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("system") == "learned_snr_randomised", "ER-2 system differs")
    _require(value.get("scientific_training_run_count") == 1, "ER-2 scientific run count differs")
    _require(value.get("training_run_count") == 1, "ER-2 training run count differs")
    _require(value.get("test_access") == 0, "ER-2 completion records test access")
    _require(value.get("runtime_root"), "ER-2 runtime root is missing")
    runtime = _safe_relative(REPO, str(value["runtime_root"]))
    runtime_completion = runtime / "run_completion.json"
    runtime_audit = runtime / "snr_assignment_audit.json"
    _require(runtime_completion.is_file() and not runtime_completion.is_symlink(), "ER-2 runtime completion is missing or unsafe")
    _require(runtime_audit.is_file() and not runtime_audit.is_symlink(), "ER-2 runtime assignment audit is missing or unsafe")
    _require(value.get("completion_sha256") == _sha(runtime_completion), "ER-2 runtime completion hash differs")
    _require(value.get("audit_sha256") == _sha(runtime_audit) if "audit_sha256" in value else True, "ER-2 audit hash differs")
    runtime_value = _read(runtime_completion)
    _require(runtime_value.get("training_run_count") == 1, "runtime ER-2 run count differs")
    _require(runtime_value.get("system") == "learned_snr_randomised", "runtime ER-2 system differs")
    selected = runtime_value.get("selected_checkpoint")
    _require(isinstance(selected, dict), "ER-2 selected checkpoint record is missing")
    selected_path = _safe_relative(runtime, str(selected["checkpoint_path"]))
    _require(selected_path.is_file() and not selected_path.is_symlink(), "ER-2 selected checkpoint is missing or unsafe")
    _require(selected.get("checkpoint_id") == _sha(selected_path), "ER-2 selected checkpoint hash differs")
    audit = _read(audit_path)
    _require(value.get("runtime_root") == audit.get("runtime_root"), "ER-2 completion/audit runtime differs")
    _require(audit.get("audit_sha256") == _sha(runtime_audit), "ER-2 outer audit hash differs")
    _require(audit.get("test_access") == 0 and audit.get("test") in (None, "SEALED"), "ER-2 audit test boundary differs")
    return value


def verify_audit(path: Path, *, recompute: bool = True) -> dict[str, Any]:
    value = _read(path)
    expected_domain = [1, 4, 7, 13, 19]  # literal-ok: AM-95 configured domain
    _require(value.get("system") == "learned_snr_randomised", "ER-2 audit system differs")
    _require(value.get("rng_purpose") == "er2_snr_randomised_v1", "ER-2 audit purpose differs")
    _require(
        value.get("identity_fields") == [
            "dataset_version",
            "split_manifest_hash",
            "stable_sample_id",
            "train_seed",
            "epoch",
        ],
        "ER-2 audit identity fields differ",
    )
    config = load_experiment("configs/learned-er2-randomized.yaml", train_seed=0, channel_seed=0)  # literal-ok: sole authorized ER-2 cell
    _require(value.get("config_hash") == config_hash(config), "ER-2 audit config hash differs")
    _require(value.get("dataset_version") == config.resolved["dataset_version"], "ER-2 audit dataset version differs")
    _require(
        value.get("split_manifest_hash") == str(get(f"datasets.{config.resolved['dataset']}.manifest_sha256")),
        "ER-2 audit split manifest differs",
    )
    _require(value.get("domain") == expected_domain, "ER-2 audit domain differs")
    _require(value.get("distribution") == "discrete_uniform", "ER-2 distribution differs")
    _require(value.get("unit") == "per_sample_per_epoch", "ER-2 sampling unit differs")
    _require(value.get("sample_count") == int(get("datasets.imagenette160.train_images")), "ER-2 sample count differs")
    _require(value.get("epoch_count") == int(get("learned_system.epochs.imagenette160")), "ER-2 epoch count differs")
    _require(value.get("assignment_count") == value["sample_count"] * value["epoch_count"], "ER-2 assignment count differs")
    _require(value.get("batching_independent") is True, "ER-2 batching-independence record differs")
    _require(value.get("channel_noise_separately_keyed") is True, "ER-2 noise-keying record differs")
    global_counts = value.get("global_counts")
    _require(isinstance(global_counts, dict) and set(global_counts) == {str(item) for item in expected_domain}, "ER-2 global count keys differ")
    _require(sum(int(item) for item in global_counts.values()) == value["assignment_count"], "ER-2 global counts do not reconcile")
    for row in value.get("per_epoch_counts", []):
        _require(set(row.get("counts", {})) == set(global_counts), "ER-2 epoch count keys differ")
        _require(sum(int(item) for item in row["counts"].values()) == value["sample_count"], "ER-2 epoch count does not reconcile")
    if recompute:
        expected = snr_assignment_audit(config)
        for key in (
            "rng_purpose", "identity_fields", "config_hash", "domain", "distribution", "unit",
            "sample_count", "epoch_count", "assignment_count", "assignment_digest",
            "global_counts", "per_epoch_counts", "batching_independent",
            "channel_noise_separately_keyed",
        ):
            _require(value.get(key) == expected.get(key), f"ER-2 audit recomputation differs at {key}")
        _require(value.get("config_hash", config_hash(config)) == config_hash(config), "ER-2 audit config hash differs")
    return value


def verify_validation(path: Path) -> dict[str, Any]:
    value = _read(path)
    expected_grid = [int(item) for item in get("channel.test_snr_grid_db")]
    _require(value.get("artifact_role") == "ER2_RANDOMIZED_VALIDATION_ONLY_EVIDENCE", "ER-2 validation role differs")
    _require(value.get("snr_grid_db") == expected_grid, "ER-2 validation SNR grid differs")
    _require(value.get("scientific_training_run_count") == 1, "ER-2 validation run count differs")
    _require(value.get("validation_only") is True and value.get("test_access") == 0, "ER-2 validation scope differs")
    _require(value.get("test") == "SEALED", "ER-2 validation test boundary differs")
    seen_snr: list[int] = []
    for curve in value.get("curves", []):
        _require(curve.get("system") == "learned_snr_randomised", "ER-2 curve system differs")
        snr = int(curve["snr_db"])
        seen_snr.append(snr)
        outcomes = curve.get("outcomes")
        _require(isinstance(outcomes, list) and len(outcomes) == int(get("datasets.imagenette160.val_images")), "ER-2 validation denominator differs")
        _require(sum(bool(row["correct"]) for row in outcomes) == int(curve["n_correct"]), "ER-2 validation count is not outcome-derived")
        _require(curve["n_total"] == len(outcomes) and curve.get("test_access") == 0, "ER-2 curve accounting differs")
        ids = [str(row["stable_sample_id"]) for row in outcomes]
        _require(len(set(ids)) == len(ids), "ER-2 validation IDs are duplicated")
    _require(seen_snr == expected_grid, "ER-2 validation curve order/coverage differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, default=COMPLETION_DEFAULT)
    parser.add_argument("--audit", type=Path, default=AUDIT_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args(argv)
    paths = [path if path.is_absolute() else REPO / path for path in (args.completion, args.audit, args.validation)]
    completion, audit, validation = paths
    verify_completion(completion, audit)
    verify_audit(audit, recompute=not args.skip_recompute)
    verify_validation(validation)
    print("randomized ER-2 verifier PASS: one AM-95 run, validation only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
