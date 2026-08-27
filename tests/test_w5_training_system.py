"""W5 contract/schema/completion verifier mutation boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import verify_w5_training_system as verifier
from training.djscc import ELIGIBILITY, PROTECTED_COUNTERS


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _smoke() -> dict:
    result = {
        "schema_version": 1,
        "artifact_role": "w5_djscc_smoke_result",
        "eligibility": copy.deepcopy(ELIGIBILITY),
        "lineage": {"source_commit": "a" * 40, "source_manifest_id": "w5source-test", "source_manifest_sha256": "b" * 64, "config_hash": "c" * 64},
        "scope": {
            "role": "NON_SCIENTIFIC_W5_PLUMBING_ONLY",
            "dataset": "cifar10",
            "lambda": 1.0,
            "lambda_status": "provisional_until_G-4",
            "smoke_only_max_microbatches_per_epoch": 1,
            "accuracy_recorded": False,
            "selection_performed": False,
        },
        "environment": {"execution_profile_id": "local_4060_cu130"},
        "training": {
            "w5_non_scientific_optimizer_steps": 6,
            "samples_across_all_physical_smoke_trajectories": 18,
            "finite_total_ce_mse": True,
            "optimizer_step_accounting": {
                "definition": "actual GradScaler step iff all optimizer-owned gradients are finite and no backoff occurs",
                "all_optimizer_owned_gradients_covered": True,
                "trajectories": {
                    name: {
                        "actual_applied_optimizer_steps": steps,
                        "grad_scaler_skips": 0,
                        "global_optimizer_step_matches_trace": True,
                        "optimizer_wide_finiteness_matches_applied_markers": True,
                        "optimizer_parameter_counts": [42],
                    }
                    for name, steps in {
                        "cifar_uninterrupted": 2,
                        "cifar_resumed": 2,
                        "imagenette_r_1_6": 1,
                        "imagenette_r_1_24": 1,
                    }.items()
                },
                "actual_applied_optimizer_steps": 6,
            },
        },
        "gradients": {"encoder_finite_nonzero": True, "reconstruction_head_finite_nonzero": True, "task_head_finite_nonzero": True},
        "checkpoint_resume": {"process_boundary": True, "fresh_process_resume": True, "exact": True, "comparison": {"model": True}},
        "selected_ratio_plumbing": {
            "r_1_6": {"dataset": "imagenette160", "k": 12800, "steps": 1, "samples": 1, "gradient_checks": {head: {"finite": True, "nonzero": True} for head in ("encoder", "reconstruction_head", "task_head")}},
            "r_1_24": {"dataset": "imagenette160", "k": 3200, "steps": 1, "samples": 1, "gradient_checks": {head: {"finite": True, "nonzero": True} for head in ("encoder", "reconstruction_head", "task_head")}},
        },
        "data_isolation": {"test_model_facing_loads": 0, "test_decoding_or_preprocessing": 0, "test_inference": 0, "test_accuracy_computation": 0, "learned_validation_selection": 0},
        "protected_counters": copy.deepcopy(PROTECTED_COUNTERS),
    }
    result["smoke_id"] = "w5smoke-" + hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _write(path: Path, value: dict) -> None:
    path.write_bytes(_canonical(value))


def test_w5_contract_schema_and_g8_lineage_verify():
    assert verifier.verify_schema()["schema_version"] == 1
    assert verifier.verify_contract()["contract_id"].startswith("w5contract-")
    assert verifier.verify_g8_lineage()["closeout_id"].startswith("g8closeout-")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["eligibility"].__setitem__("selection_eligibility", "ELIGIBLE"), "eligibility"),
        (lambda value: value["protected_counters"].__setitem__("test_access", 1), "protected counters"),
        (lambda value: value["data_isolation"].__setitem__("test_inference", 1), "data-isolation"),
        (lambda value: value["checkpoint_resume"].__setitem__("exact", False), "resume proof"),
        (lambda value: value["selected_ratio_plumbing"]["r_1_6"].__setitem__("k", 1), "selected-ratio k"),
        (lambda value: value["scope"].__setitem__("selection_performed", True), "scope"),
        (lambda value: value["training"]["optimizer_step_accounting"].__setitem__("all_optimizer_owned_gradients_covered", False), "optimizer-step accounting"),
    ],
)
def test_smoke_verifier_rejects_scope_schema_and_counter_mutations(tmp_path: Path, monkeypatch, mutation, match):
    value = _smoke()
    mutation(value)
    # Re-sign canonical identity so this exercises the inner invariant rather
    # than stopping at the outer content-addressed ID.
    value.pop("smoke_id")
    value["smoke_id"] = "w5smoke-" + hashlib.sha256(_canonical(value)).hexdigest()
    path = tmp_path / "smoke.json"
    _write(path, value)
    monkeypatch.setattr(verifier, "SMOKE_PATH", path)
    with pytest.raises(ValueError, match=match):
        verifier.verify_smoke()


def test_historical_completion_is_exactly_bounded_and_rejects_byte_drift(
    tmp_path: Path, monkeypatch
):
    historical = verifier.verify_historical_completion()
    assert historical["completion_id"] == verifier.HISTORICAL_COMPLETION_ID
    raw = verifier.HISTORICAL_COMPLETION_PATH.read_bytes()
    mutated = tmp_path / "w5_completion.json"
    mutated.write_bytes(raw + b" ")
    monkeypatch.setattr(verifier, "HISTORICAL_COMPLETION_PATH", mutated)
    with pytest.raises(ValueError, match="historical W5 completion bytes"):
        verifier.verify_historical_completion()


def test_smoke_verifier_rejects_identity_drift(tmp_path: Path, monkeypatch):
    value = _smoke()
    value["smoke_id"] = "w5smoke-" + "0" * 64
    path = tmp_path / "smoke.json"
    _write(path, value)
    monkeypatch.setattr(verifier, "SMOKE_PATH", path)
    with pytest.raises(ValueError, match="smoke_id"):
        verifier.verify_smoke()
