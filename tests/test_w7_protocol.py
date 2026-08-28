"""W7-A protocol and role-boundary regressions (no data or optimizer work)."""

from __future__ import annotations

import copy

import pytest

from config.run_config import load_experiment
from training.w7_g4 import NON_SCIENTIFIC_PROFILE_POLICY, W7_G4_PILOT_POLICY
from training.w7_protocol import (
    W7_CALIBRATION_SNR_DB,
    W7_CHANNEL_SEED,
    W7_LAMBDA_GRID,
    W7_PSNR_SNR_DB,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    load_w7_config,
    protocol_descriptor,
    validate_w7_config,
)


def test_protocol_is_result_independent_and_exact():
    descriptor = protocol_descriptor()
    assert descriptor["dataset"] == "imagenette160"
    assert descriptor["ratio"] == "r_1_6"
    assert descriptor["lambda_grid"] == list(W7_LAMBDA_GRID)
    assert descriptor["lambda_order"] == "exact_configured_lambda_grid_order"
    assert (descriptor["train_seed"], descriptor["channel_seed"]) == (W7_TRAIN_SEED, W7_CHANNEL_SEED)
    assert descriptor["training_snr_db"] == W7_TRAINING_SNR_DB
    assert descriptor["calibration_snr_db"] == W7_CALIBRATION_SNR_DB
    assert descriptor["psnr_eval_snr_db"] == W7_PSNR_SNR_DB
    assert descriptor["validation"]["batch_size"] == 32  # literal-ok: owner-frozen W7-A validation batch
    assert descriptor["validation"]["denominator"] == 1000  # literal-ok: committed Imagenette validation denominator
    assert descriptor["checkpoint_selection"]["tie_break"] == "earliest_epoch"
    assert descriptor["psnr"]["aggregation"].startswith("arithmetic_mean")
    assert descriptor["training"]["drop_last"] is False


def test_each_lambda_resolves_only_the_grid_entry():
    configs = [load_w7_config(lambda_value=value) for value in W7_LAMBDA_GRID]
    assert [config.resolved["lambda"] for config in configs] == list(W7_LAMBDA_GRID)
    assert [config.resolved["lambda_grid_index"] for config in configs] == list(range(len(W7_LAMBDA_GRID)))
    assert len({config.resolved["k"] for config in configs}) == 1
    assert all(config.resolved["artifact_role"] == "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT" for config in configs)


def test_profile_policy_is_non_scientific_and_w5_is_not_admitted():
    profile = load_w7_config(lambda_value=1.0, role="NON_SCIENTIFIC_PROFILE")
    NON_SCIENTIFIC_PROFILE_POLICY.validate(profile)
    pilot = load_w7_config(lambda_value=0.0)
    W7_G4_PILOT_POLICY.validate(pilot)
    old = load_experiment("configs/learned-w5-smoke.yaml")
    with pytest.raises(ValueError):
        validate_w7_config(old)


def test_protocol_validator_does_not_mutate_resolved_config():
    config = load_w7_config(lambda_value=0.3)
    before = copy.deepcopy(config.to_dict())
    validate_w7_config(config)
    assert config.to_dict() == before
