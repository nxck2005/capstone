"""The prospective AM-98 PAPR-constrained learned successor (one run, not executed).

The constrained variant differs from the frozen W8 ``r_1_6`` model only by
``PeakPowerConstraint(params.evaluation.w10_papr_cap_db)`` installed on the
shared encoder path.  Training itself reuses the frozen W8 recipe and the W8
trainer's transactional epoch store through the narrow ``_build_model`` seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from channels.power import PeakPowerConstraint
from config.params import get
from config.run_config import RunConfig, load_experiment
from models.djscc import DJSCC, build_djscc
from training.w8_final import W8Trainer

PAPR_CONFIG = "configs/learned-papr-constrained-r1-6.yaml"
PAPR_RATIO = "r_1_6"
PAPR_TRAIN_SEED = 0
PAPR_CHANNEL_SEED = 0
PAPR_TRAINING_RUNS = 1
PAPR_AUTHORITY_PATH = "results/learned/w10/papr_training_authorization.json"
PAPR_SELECTED_CHECKPOINT_PATH = "results/learned/w10/papr_selected_checkpoint.json"
PAPR_RUNTIME_ROOT = "checkpoints/papr_constrained_pascal_v4"
# The constrained successor reuses the frozen W8 GTX-bound trainer/profile
# authentication; W10 execution itself remains bound to the TITAN Xp.
PAPR_GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
PAPR_GPU_UUID = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"
PAPR_CAMPAIGN_ID = "papr-constrained-r1-6-2026"
PAPR_RUN_ID = "papr-constrained-r1-6-train0-channel0"


def papr_cap_db() -> float:
    return float(get("evaluation.w10_papr_cap_db"))


def build_papr_model(config: RunConfig, device: torch.device | str) -> DJSCC:
    if config.resolved.get("system") != "learned":
        raise ValueError("the PAPR-constrained variant is a learned-system run")
    if str(config.resolved.get("bw_ratio")) != PAPR_RATIO:
        raise ValueError("the PAPR-constrained variant is frozen at the headline ratio")
    return build_djscc(config, peak_constraint=PeakPowerConstraint(papr_cap_db()), device=device)


def load_papr_config() -> RunConfig:
    return load_experiment(PAPR_CONFIG, train_seed=PAPR_TRAIN_SEED, channel_seed=PAPR_CHANNEL_SEED)


class PaprConstrainedTrainer(W8Trainer):
    """The W8 trainer with the frozen symbol-domain PAPR projection installed."""

    @staticmethod
    def _build_model(config: RunConfig, device: torch.device) -> DJSCC:
        return build_papr_model(config, device)


def protected_counters() -> dict[str, int]:
    return {
        "papr_constrained_training": 0,
        "production_training": 0,
        "randomized_er2_training": 0,
        "er9_training": 0,
        "g10_adjudications": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
    }


def papr_protocol(root: Path, *, source_record_value: dict, config_hash_value: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": int(get("evaluation.w10_papr_protocol_version")),
        "system": "learned_papr_constrained",
        "dataset": "imagenette160",
        "bw_ratio": PAPR_RATIO,
        "train_seed": PAPR_TRAIN_SEED,
        "channel_seed": PAPR_CHANNEL_SEED,
        "fresh_initialization": True,
        "transfer_initialization_permitted": False,
        "training_runs": PAPR_TRAINING_RUNS,
        "additional_seeds_permitted": False,
        "outcome_conditioned_tuning_permitted": False,
        "papr_cap_db": papr_cap_db(),
        "papr_domain": "symbol_domain_not_oversampled_waveform",
        "architecture": str(get("learned_system.encoder_arch")),
        "recipe": "frozen_w8_r1_6_recipe",
        "lambda_core": float(get("learned_system.lambda_core")),
        "train_snr_db": int(get("channel.train_snr_db_fixed")),
        "epochs": int(get("learned_system.epochs.imagenette160")),
        "checkpoint_selection": {
            "metric": "validation_n_correct",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_db": int(get("channel.train_snr_db_fixed")),
        },
        "runtime_root": PAPR_RUNTIME_ROOT,
        "config_path": PAPR_CONFIG,
        "config_hash": config_hash_value,
        "source": dict(source_record_value),
        "protected_counters": protected_counters(),
        "test": "SEALED",
        "test_access": 0,
    }


__all__ = [
    "PAPR_AUTHORITY_PATH",
    "PAPR_CAMPAIGN_ID",
    "PAPR_CHANNEL_SEED",
    "PAPR_CONFIG",
    "PAPR_GPU_NAME",
    "PAPR_GPU_UUID",
    "PAPR_RATIO",
    "PAPR_RUN_ID",
    "PAPR_RUNTIME_ROOT",
    "PAPR_SELECTED_CHECKPOINT_PATH",
    "PAPR_TRAIN_SEED",
    "PAPR_TRAINING_RUNS",
    "PaprConstrainedTrainer",
    "build_papr_model",
    "load_papr_config",
    "papr_cap_db",
    "papr_protocol",
    "protected_counters",
]
