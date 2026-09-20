"""The prospective AM-98 PAPR-constrained learned successor (one run, not executed).

The constrained variant differs from the frozen W8 ``r_1_6`` model only by
``PeakPowerConstraint(params.evaluation.w10_papr_cap_db)`` installed on the
shared encoder path.  Training deliberately inherits the frozen W8 recipe
mechanics through an explicit PAPR protocol projection: it never adopts a W8
artifact role, eligibility record or protected-counter namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from channels.power import PeakPowerConstraint
from config.params import get
from config.run_config import RunConfig, config_hash as run_config_hash, load_experiment
from training.deterministic_core import canonical_sha256
from models.djscc import DJSCC, build_djscc
from training.w8_final import W8Trainer, W8TrainingPolicy
from training.w8_protocol import (
    W8_ACCUMULATION_FACTOR,
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
    W8_COMPONENT_PATH,
    W8_DATASET,
    W8_EPOCHS,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_EXPECTED_K,
    W8_EXPECTED_LAMBDA,
    W8_EXPECTED_RATIOS,
    W8_FINAL_PARTIAL_BATCH,
    W8_EXPECTED_MICROBATCHES,
    W8_PHYSICAL_BATCH_SIZE,
    W8_PROFILE_ID,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    checkpoint_selection_snr_db,
    unique_core_ratios,
)

PAPR_CONFIG = "configs/learned-papr-constrained-r1-6.yaml"
PAPR_RATIO = "r_1_6"
PAPR_TRAIN_SEED = 0
PAPR_CHANNEL_SEED = 0
PAPR_TRAINING_RUNS = 1
PAPR_EPOCHS = W8_EPOCHS
PAPR_K = W8_EXPECTED_K[PAPR_RATIO]
PAPR_AUTHORITY_PATH = "results/learned/w10/papr_training_authorization.json"
PAPR_SELECTED_CHECKPOINT_PATH = "results/learned/w10/papr_selected_checkpoint.json"
PAPR_COMPLETION_PATH = "results/learned/w10/papr_training_completion.json"
PAPR_RUNTIME_ROOT = "checkpoints/papr_constrained_pascal_v4"
PAPR_RUN_DIRECTORY = "train0_channel0"
# The constrained successor reuses the frozen W8 GTX-bound trainer/profile
# authentication; W10 execution itself remains bound to the TITAN Xp.
PAPR_GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
PAPR_GPU_UUID = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"
PAPR_CAMPAIGN_ID = "papr-constrained-r1-6-2026"
PAPR_RUN_ID = "papr-constrained-r1-6-train0-channel0"

PAPR_CORE_ROLE = "PAPR_CONSTRAINED_TRAINING_RUN"
PAPR_EPOCH_ROLE = "PAPR_CONSTRAINED_TRAINING_EPOCH_RECORD"
PAPR_CHECKPOINT_ROLE = "PAPR_CONSTRAINED_TRAINING_CHECKPOINT"
PAPR_SIDECAR_ROLE = "PAPR_CONSTRAINED_TRAINING_CHECKPOINT_SIDECAR"
PAPR_SELECTED_ROLE = "PAPR_CONSTRAINED_SELECTED_CHECKPOINT"
PAPR_COMPLETION_ROLE = "PAPR_CONSTRAINED_TRAINING_COMPLETION"
PAPR_VALIDATION_ROLE = "PAPR_CONSTRAINED_VALIDATION_EPOCH_SUMMARY"
PAPR_PROTOCOL_PREFIX = "papr-constrained-pre-execution-v"
PAPR_SELECTED_PREFIX = "paprselected-"
PAPR_COMPLETION_PREFIX = "paprcompletion-"
PAPR_AUTHORITY_PREFIX = "paprtrainingauth-"
PAPR_SCHEMA_VERSION_AUTHORITY = 2

PAPR_ELIGIBILITY = {
    "artifact_role": PAPR_CORE_ROLE,
    "scientific_status": "SCIENTIFIC_PAPR_CONSTRAINED_TRAINING",
    "selection_eligibility": "PER_RUN_VALIDATION_ONLY",
    "reporting_eligibility": "NOT_ELIGIBLE_UNTIL_W10_REHEARSAL",
    "w10_eligibility": "ELIGIBLE_FOR_W10_PAPR_SECONDARY_ONLY",
    "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
    "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
}

PAPR_SELECTED_ELIGIBILITY = {
    **PAPR_ELIGIBILITY,
    "artifact_role": PAPR_SELECTED_ROLE,
    "selection_eligibility": "ELIGIBLE_FOR_W10_PAPR_EVALUATION_ONLY",
    "w10_eligibility": "SELECTED_PAPR_CHECKPOINT_PENDING_W10_REHEARSAL",
    "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
}

_EXECUTION_COMMIT_RULE = "commit_that_adds_this_authority_file"


def papr_cap_db() -> float:
    return float(get("evaluation.w10_papr_cap_db"))


def papr_protocol_version() -> str:
    value = get("evaluation.w10_papr_protocol_version")
    return f"{PAPR_PROTOCOL_PREFIX}{int(value)}"


def pre_execution_protected_counters() -> dict[str, int]:
    """The authority's zero-work counters; the live run increments only PAPR."""

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


def active_protected_counters() -> dict[str, int]:
    """The counters every PAPR training artifact must carry while it runs."""

    return {**pre_execution_protected_counters(), "papr_constrained_training": 1}


@dataclass(frozen=True)
class PaprTrainingPolicy(W8TrainingPolicy):
    """The PAPR-constrained lifecycle's explicit artifact namespace."""

    def validate(self, config: RunConfig) -> None:
        validate_papr_config(config)
        if config.resolved.get("artifact_role") != PAPR_CORE_ROLE:
            raise ValueError("PAPR policy/config role differs")
        if any(config.resolved.get(key) != value for key, value in PAPR_ELIGIBILITY.items()):
            raise ValueError("PAPR policy eligibility is not exact")
        if not (self.validation_enabled and self.scientific):
            raise ValueError("PAPR policy is not scientific")
        if dict(self.protected_counters) != active_protected_counters():
            raise ValueError("PAPR policy protected counters differ")
        if self.papr_cap_db != papr_cap_db():
            raise ValueError("PAPR policy cap differs from the frozen parameter")
        if self.protocol_version != papr_protocol_version():
            raise ValueError("PAPR policy protocol version differs")
        if self.eligibility is None or dict(self.eligibility) != PAPR_ELIGIBILITY:
            raise ValueError("PAPR policy eligibility record differs")
        if self.role != PAPR_CORE_ROLE:
            raise ValueError("PAPR policy role differs")


PROTECTED_PAPR_ROLES = (
    PAPR_EPOCH_ROLE,
    PAPR_CHECKPOINT_ROLE,
    PAPR_SIDECAR_ROLE,
    PAPR_SELECTED_ROLE,
    PAPR_COMPLETION_ROLE,
    PAPR_VALIDATION_ROLE,
)


def papr_training_policy() -> PaprTrainingPolicy:
    return PaprTrainingPolicy(
        role=PAPR_CORE_ROLE,
        validation_enabled=True,
        scientific=True,
        protected_counters=active_protected_counters(),
        protocol_version=papr_protocol_version(),
        epoch_role=PAPR_EPOCH_ROLE,
        checkpoint_role=PAPR_CHECKPOINT_ROLE,
        sidecar_role=PAPR_SIDECAR_ROLE,
        eligibility=PAPR_ELIGIBILITY,
        papr_cap_db=papr_cap_db(),
    )


PAPR_TRAINING_POLICY = papr_training_policy()


def validate_papr_config(config: RunConfig) -> None:
    """Validate the exact W8-derived recipe AM-98 freezes for PAPR.

    This is not a weakened ``validate_w8_config``: the W8 artifact role and W8
    eligibility are deliberately absent, and every shared recipe fact is
    re-asserted here against the normative parameters.
    """

    if not isinstance(config, RunConfig):
        raise TypeError("PAPR requires a resolved RunConfig")
    resolved = config.resolved
    if resolved.get("system") != "learned" or resolved.get("dataset") != W8_DATASET:
        raise ValueError("PAPR system/dataset differs from the frozen protocol")
    if resolved.get("split") != "train" or resolved.get("channel") != "awgn":
        raise ValueError("PAPR train split or channel differs from the frozen protocol")
    if resolved.get("bw_ratio") != PAPR_RATIO or PAPR_RATIO not in W8_EXPECTED_RATIOS or PAPR_RATIO not in unique_core_ratios():
        raise ValueError("PAPR ratio is not the frozen headline ratio")
    if resolved.get("k") != PAPR_K or int(get(f"bandwidth.k_symbols.{W8_DATASET}.{PAPR_RATIO}")) != PAPR_K:
        raise ValueError("PAPR symbol budget differs from the frozen ratio budget")
    if resolved.get("train_seed") != PAPR_TRAIN_SEED or resolved.get("channel_seed") != PAPR_CHANNEL_SEED:
        raise ValueError("PAPR seed cell is not the frozen (0, 0)")
    if float(resolved.get("lambda")) != W8_EXPECTED_LAMBDA:
        raise ValueError("PAPR lambda is not the selected G-4 lambda")
    if float(get("learned_system.lambda_core")) != W8_EXPECTED_LAMBDA or get("learned_system.lambda_status") != "selected_at_G-4":
        raise ValueError("current normative G-4 lambda state is not selected_at_G-4 = 3.0")
    if resolved.get("architecture") != get("learned_system.encoder_arch") or resolved.get("architecture") != "djscc_residual_v1":
        raise ValueError("PAPR architecture differs from djscc_residual_v1")
    if resolved.get("train_snr_db") != get("channel.train_snr_db_fixed"):
        raise ValueError("PAPR training SNR does not resolve to train_snr_db_fixed")
    if resolved.get("checkpoint_selection_snr_db") != checkpoint_selection_snr_db():
        raise ValueError("PAPR checkpoint selection SNR resolution differs")
    if resolved.get("checkpoint_selection_snr_parameter") != W8_CHECKPOINT_SELECTION_SNR_PARAMETER:
        raise ValueError("PAPR checkpoint selection SNR parameter binding differs")
    if resolved.get("execution_profile_id") != W8_PROFILE_ID:
        raise ValueError("PAPR execution profile is not confessor_pascal_cu126")
    if resolved.get("physical_batch_size") != W8_PHYSICAL_BATCH_SIZE or resolved.get("accumulation_factor") != W8_ACCUMULATION_FACTOR or resolved.get("effective_batch_size") != W8_EFFECTIVE_BATCH_SIZE:
        raise ValueError("PAPR Pascal batch binding differs")
    if resolved.get("validation_batch_size") != W8_VALIDATION_BATCH_SIZE:
        raise ValueError("PAPR validation batch binding differs")
    if int(get(f"learned_system.batch_size.{W8_DATASET}")) != W8_EFFECTIVE_BATCH_SIZE:
        raise ValueError("PAPR effective batch differs from the learned-system parameter")
    if int(get(f"learned_system.epochs.{W8_DATASET}")) != PAPR_EPOCHS:
        raise ValueError("PAPR epoch schedule differs")
    if get("learned_system.checkpoint_selection_split") != "validation" or get("learned_system.checkpoint_selection_metric") != "top1_accuracy" or get("learned_system.checkpoint_selection_mode") != "max" or get("learned_system.checkpoint_selection_tie_break") != "earliest_epoch":
        raise ValueError("PAPR checkpoint-selection rule differs")
    if get("learned_system.papr_report_required") is not True or get("learned_system.papr_constrained_variant_required") is not True:
        raise ValueError("PAPR reporting boundary was weakened")
    if int(get("learned_system.papr_constrained_variant_seeds")) != PAPR_TRAINING_RUNS:
        raise ValueError("PAPR seed count is not the frozen single run")
    if int(get("evaluation.w10_papr_training_runs")) != PAPR_TRAINING_RUNS:
        raise ValueError("PAPR authority run count is not the frozen single run")
    cap = papr_cap_db()
    if cap != 3.0 or cap <= 0:  # literal-ok: AM-98 frozen cap
        raise ValueError("PAPR cap differs from the frozen 3.0 dB")
    if resolved.get("artifact_role") != PAPR_CORE_ROLE:
        raise ValueError("PAPR config carries a foreign artifact role")
    for key, value in PAPR_ELIGIBILITY.items():
        if resolved.get(key) != value:
            raise ValueError(f"PAPR eligibility {key} differs")


def _papr_projection(config: RunConfig) -> RunConfig:
    """Project the PAPR namespace onto the frozen W8 recipe mechanics."""

    value = config.to_dict()
    choices = dict(value["choices"])
    resolved = dict(value["resolved"])
    resolved.update(
        {
            "k": PAPR_K,
            "artifact_role": PAPR_CORE_ROLE,
            "execution_profile_id": W8_PROFILE_ID,
            "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
            "accumulation_factor": W8_ACCUMULATION_FACTOR,
            "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
            "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
            "checkpoint_selection_snr_db": checkpoint_selection_snr_db(),
            "checkpoint_selection_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        }
    )
    choices["artifact_role"] = PAPR_CORE_ROLE
    for key, selected in PAPR_ELIGIBILITY.items():
        choices[key] = selected
        resolved[key] = selected
    value["choices"] = choices
    value["resolved"] = resolved
    return RunConfig.from_dict(value)


def load_papr_config() -> RunConfig:
    base = load_experiment(PAPR_CONFIG, train_seed=PAPR_TRAIN_SEED, channel_seed=PAPR_CHANNEL_SEED)
    config = _papr_projection(base)
    validate_papr_config(config)
    return config


def build_papr_model(config: RunConfig, device: torch.device | str) -> DJSCC:
    if config.resolved.get("system") != "learned":
        raise ValueError("the PAPR-constrained variant is a learned-system run")
    if str(config.resolved.get("bw_ratio")) != PAPR_RATIO:
        raise ValueError("the PAPR-constrained variant is frozen at the headline ratio")
    return build_djscc(config, peak_constraint=PeakPowerConstraint(papr_cap_db()), device=device)


def papr_protocol_descriptor(config: RunConfig) -> dict[str, Any]:
    """The complete, result-independent PAPR protocol projection."""

    validate_papr_config(config)
    resolved = config.resolved
    return {
        "protocol_version": papr_protocol_version(),
        "papr_protocol_version_parameter": "params.evaluation.w10_papr_protocol_version",
        "system": "learned_papr_constrained",
        "dataset": W8_DATASET,
        "bw_ratio": PAPR_RATIO,
        "k": PAPR_K,
        "train_seed": PAPR_TRAIN_SEED,
        "channel_seed": PAPR_CHANNEL_SEED,
        "fresh_initialization": True,
        "transfer_initialization_permitted": False,
        "initialization_rule": "fresh_keyed_init_same_component_as_w8_headline",
        "component_path": W8_COMPONENT_PATH,
        "training_runs": PAPR_TRAINING_RUNS,
        "additional_seeds_permitted": False,
        "outcome_conditioned_tuning_permitted": False,
        "papr_cap_db": papr_cap_db(),
        "papr_cap_parameter": "params.evaluation.w10_papr_cap_db",
        "papr_domain": "symbol_domain_not_oversampled_waveform",
        "papr_bound_tolerance_db": 1e-4,  # literal-ok: PeakPowerConstraint numerical bound tolerance
        "architecture": str(get("learned_system.encoder_arch")),
        "recipe": "frozen_w8_r1_6_recipe",
        "lambda_core": float(get("learned_system.lambda_core")),
        "lambda_status": "selected_at_G-4",
        "train_snr_db": int(get("channel.train_snr_db_fixed")),
        "train_snr_parameter": "params.channel.train_snr_db_fixed",
        "epochs": PAPR_EPOCHS,
        "checkpoint_selection": {
            "split": "validation",
            "metric": "validation_top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "snr_resolution": "params.channel.train_snr_db_fixed",
            "snr_db": checkpoint_selection_snr_db(),
            "channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
            "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
            "fixed_noise_across_epochs": True,
            "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"],
            "cross_seed_selection": False,
        },
        "batch": {
            "physical": W8_PHYSICAL_BATCH_SIZE,
            "accumulation": W8_ACCUMULATION_FACTOR,
            "effective": W8_EFFECTIVE_BATCH_SIZE,
            "validation": W8_VALIDATION_BATCH_SIZE,
            "train_samples": int(get(f"datasets.{W8_DATASET}.train_images")),
            "microbatches": W8_EXPECTED_MICROBATCHES,
            "final_physical_batch": W8_FINAL_PARTIAL_BATCH,
        },
        "resume_semantics": "exact_authenticated_completed_epoch_model_optimizer_scaler",
        "runtime_root": PAPR_RUNTIME_ROOT,
        "run_directory": PAPR_RUN_DIRECTORY,
        "config_path": PAPR_CONFIG,
        "config_hash": run_config_hash(config),
        "protected_counters": pre_execution_protected_counters(),
        "test": "SEALED",
        "test_access": 0,
    }


def papr_protocol_config_hash(config: RunConfig) -> str:
    """The PAPR protocol/config fingerprint recorded in every artifact lineage."""

    return canonical_sha256({"protocol": papr_protocol_descriptor(config), "config": config.to_dict()})


def papr_authority_protocol(config: RunConfig, *, source_record_value: dict[str, Any]) -> dict[str, Any]:
    """The authority protocol: the descriptor plus its source/execution custody."""

    descriptor = papr_protocol_descriptor(config)
    return {
        **descriptor,
        "protocol_config_hash": papr_protocol_config_hash(config),
        "source": dict(source_record_value),
        "execution": {
            "execution_commit_rule": _EXECUTION_COMMIT_RULE,
            "source_epoch_rule": "protected_source_at_execution_commit_equals_frozen_epoch",
            "runtime_root": f"{PAPR_RUNTIME_ROOT}/{PAPR_RUN_DIRECTORY}",
            "sole_writer": True,
        },
    }


class PaprConstrainedTrainer(W8Trainer):
    """The W8 trainer mechanics with the frozen symbol-domain PAPR projection."""

    @staticmethod
    def _build_model(config: RunConfig, device: torch.device) -> DJSCC:
        return build_papr_model(config, device)

    def _protocol_config_hash(self, config: RunConfig) -> str:
        return papr_protocol_config_hash(config)


def protected_counters() -> dict[str, int]:
    """Backward-compatible alias for the pre-execution authority counters."""

    return pre_execution_protected_counters()


def papr_protocol(root: Path, *, source_record_value: dict, config_hash_value: str) -> dict[str, Any]:
    """Backward-compatible authority protocol builder over the projected config."""

    config = load_papr_config()
    if papr_protocol_config_hash(config) != config_hash_value:
        raise ValueError("PAPR protocol config hash differs from the supplied projection")
    return papr_authority_protocol(config, source_record_value=source_record_value)


__all__ = [
    "PAPR_AUTHORITY_PATH",
    "PAPR_AUTHORITY_PREFIX",
    "PAPR_CAMPAIGN_ID",
    "PAPR_CHANNEL_SEED",
    "PAPR_CHECKPOINT_ROLE",
    "PAPR_COMPLETION_PATH",
    "PAPR_COMPLETION_PREFIX",
    "PAPR_COMPLETION_ROLE",
    "PAPR_CONFIG",
    "PAPR_CORE_ROLE",
    "PAPR_ELIGIBILITY",
    "PAPR_EPOCH_ROLE",
    "PAPR_EPOCHS",
    "PAPR_GPU_NAME",
    "PAPR_GPU_UUID",
    "PAPR_K",
    "PAPR_RATIO",
    "PAPR_RUN_DIRECTORY",
    "PAPR_RUN_ID",
    "PAPR_RUNTIME_ROOT",
    "PAPR_SCHEMA_VERSION_AUTHORITY",
    "PAPR_SELECTED_CHECKPOINT_PATH",
    "PAPR_SELECTED_ELIGIBILITY",
    "PAPR_SELECTED_PREFIX",
    "PAPR_SELECTED_ROLE",
    "PAPR_SIDECAR_ROLE",
    "PAPR_TRAINING_RUNS",
    "PAPR_TRAIN_SEED",
    "PAPR_TRAINING_POLICY",
    "PAPR_VALIDATION_ROLE",
    "PaprConstrainedTrainer",
    "PaprTrainingPolicy",
    "active_protected_counters",
    "build_papr_model",
    "load_papr_config",
    "papr_authority_protocol",
    "papr_cap_db",
    "papr_protocol",
    "papr_protocol_config_hash",
    "papr_protocol_descriptor",
    "papr_protocol_version",
    "papr_training_policy",
    "pre_execution_protected_counters",
    "protected_counters",
    "validate_papr_config",
]
