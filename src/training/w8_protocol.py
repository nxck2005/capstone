"""Frozen, result-independent W8 final multi-seed training protocol.

W8 is intentionally a separate protocol namespace.  It does not inherit a W7
artifact role or a W7 checkpoint loader: W7 is upstream authority only, while
this module constructs exactly the six fresh final-training cells authorized by
W8-A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.params import REPO_ROOT, get
from config.run_config import RunConfig, canonical_sha256, load_experiment

W8_PROTOCOL_VERSION = "w8-final-multi-seed-pre-execution-v1"
W8_PROFILE_ID = "confessor_pascal_cu126"
W8_EXECUTION_IMAGE_FAMILY = "pascal-cu126-requirements-pascal-lock-v1"
W8_SELECTED_GPU_UUID = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"
W8_SELECTED_GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
W8_DATASET = "imagenette160"
W8_CHANNEL = "awgn"
W8_TRAIN_SEEDS = (0, 1, 2)
W8_CHANNEL_SEEDS = (0, 1, 2)
W8_SEED_PAIRING = "zipped_not_cross_product"
W8_EXPECTED_RATIOS = ("r_1_6", "r_1_24")
W8_EXPECTED_K = {"r_1_6": 12800, "r_1_24": 3200}  # literal-ok: owner-frozen W8 ratio budgets
W8_EXPECTED_LAMBDA = 3.0  # literal-ok: owner-frozen G-4 selection
W8_PHYSICAL_BATCH_SIZE = 32  # literal-ok: qualified Pascal profile batch
W8_ACCUMULATION_FACTOR = 1  # literal-ok: qualified Pascal profile accumulation
W8_EFFECTIVE_BATCH_SIZE = 32  # literal-ok: owner-frozen effective batch
W8_VALIDATION_BATCH_SIZE = 32  # literal-ok: owner-frozen validation batch
W8_TRAIN_SAMPLE_COUNT = 8469  # literal-ok: committed Imagenette train denominator
W8_VALIDATION_SAMPLE_COUNT = 1000  # literal-ok: committed Imagenette validation denominator
W8_FINAL_PARTIAL_BATCH = 21  # literal-ok: 8469 modulo the physical batch
W8_EXPECTED_MICROBATCHES = 265  # literal-ok: ceil(8469 / 32)
W8_MIN_FREE_SPACE_GIB = 25  # literal-ok: owner-required pre-launch free-space floor
W8_EPOCHS = 100  # literal-ok: owner-frozen Imagenette final schedule
W8_COMPONENT_PATH = "djscc.djscc_residual_v1"
W8_TRAIN_SNR_PARAMETER = "params.channel.train_snr_db_fixed"
W8_CHECKPOINT_SELECTION_SNR_PARAMETER = "params.learned_system.checkpoint_selection_snr_db"
W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE = "run_channel_seed"
W8_CORE_ROLE = "W8_FINAL_MULTI_SEED_RUN"
W8_SMOKE_ROLE = "W8_NON_SCIENTIFIC_SMOKE"
W8_CHECKPOINT_ROLE = "W8_FINAL_TRAINING_CHECKPOINT"
W8_CHECKPOINT_SIDECAR_ROLE = "W8_FINAL_TRAINING_CHECKPOINT_SIDECAR"
W8_TRAINING_EPOCH_ROLE = "W8_FINAL_TRAINING_EPOCH_RECORD"
W8_EPOCH_ROLE = "W8_VALIDATION_EPOCH_SUMMARY"
W8_SELECTED_ROLE = "W8_SELECTED_CHECKPOINT"
W8_CAMPAIGN_MANIFEST_ROLE = "W8_CAMPAIGN_MANIFEST"
W8_CAMPAIGN_COMPLETION_ROLE = "W8_CAMPAIGN_COMPLETION"


@dataclass(frozen=True)
class W8RunCell:
    """One immutable ratio/train-seed/channel-seed cell in campaign order."""

    run_index: int
    ratio: str
    train_seed: int
    channel_seed: int
    k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "ratio": self.ratio,
            "k": self.k,
            "train_seed": self.train_seed,
            "channel_seed": self.channel_seed,
        }


def _base_config_path() -> Any:
    return REPO_ROOT / "configs/learned-w8-final.yaml"


def unique_core_ratios() -> tuple[str, ...]:
    """Resolve DEC-11's named ratios and retain first-occurrence order."""

    names = ("crossover_ratio", "efficiency_ratio", "low_ratio_operating_point")
    values: list[str] = []
    for name in names:
        status = get(f"bandwidth.{name}_status")
        if status != "selected_at_G-8":
            raise ValueError(f"W8 ratio {name} is not frozen at G-8")
        ratio = get(f"bandwidth.{name}")
        if ratio not in get("bandwidth.ratios"):
            raise ValueError(f"W8 ratio {name} is not a configured ratio: {ratio!r}")
        if ratio not in values:
            values.append(str(ratio))
    resolved = tuple(values)
    if resolved != W8_EXPECTED_RATIOS:
        raise ValueError(
            "W8 DEC-11 unique-ratio resolution differs: "
            f"{resolved!r} != {W8_EXPECTED_RATIOS!r}"
        )
    return resolved


def run_cells() -> tuple[W8RunCell, ...]:
    """Build the exact seed-major, ratio-minor six-cell order."""

    ratios = unique_core_ratios()
    cells: list[W8RunCell] = []
    index = 0
    for train_seed, channel_seed in zip(W8_TRAIN_SEEDS, W8_CHANNEL_SEEDS):
        for ratio in ratios:
            index += 1
            k = int(get(f"bandwidth.k_symbols.{W8_DATASET}.{ratio}"))
            if k != W8_EXPECTED_K[ratio]:
                raise ValueError(f"W8 k differs for {ratio}: {k}")
            cells.append(W8RunCell(index, ratio, train_seed, channel_seed, k))
    result = tuple(cells)
    if len(result) != 6:  # literal-ok: six authorized core cells
        raise ValueError("W8 run matrix does not contain exactly six cells")
    return result


def validate_run_order(cells: tuple[W8RunCell, ...] | list[W8RunCell]) -> None:
    expected = run_cells()
    actual = tuple(cells)
    if actual != expected:
        raise ValueError("W8 campaign order differs from the seed-major frozen matrix")


def eligibility_for_role(role: str) -> dict[str, str]:
    if role == W8_CORE_ROLE:
        return {
            "artifact_role": role,
            "scientific_status": "SCIENTIFIC_W8_FINAL_TRAINING",
            "selection_eligibility": "PER_RUN_VALIDATION_ONLY",
            "reporting_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
            "w8_eligibility": "ELIGIBLE_FOR_W8_FINAL_TRAINING_ONLY",
            "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        }
    if role == W8_SMOKE_ROLE:
        return {
            "artifact_role": role,
            "scientific_status": "NON_SCIENTIFIC",
            "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
            "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
            "w8_eligibility": "NOT_ELIGIBLE_FOR_W8_RESULT",
            "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        }
    raise ValueError(f"unknown W8 artifact role: {role!r}")


def checkpoint_selection_snr_db() -> int | float:
    """Resolve the explicit W8 selection-SNR parameter, fail closed if changed."""

    selector = get("learned_system.checkpoint_selection_snr_db")
    if selector != "train_snr_db_fixed":
        raise ValueError(
            "W8 checkpoint selection SNR must resolve through "
            "params.channel.train_snr_db_fixed"
        )
    value = get("channel.train_snr_db_fixed")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("W8 checkpoint selection SNR is not numeric")
    return value


def fresh_initialization_identity(train_seed: int) -> dict[str, Any]:
    fields = get("artifacts.rng_identity_fields.init")
    if fields != ["train_seed", "component_path"]:
        raise ValueError("W8 init identity fields are not the frozen two-field contract")
    if not isinstance(train_seed, int) or isinstance(train_seed, bool) or train_seed not in W8_TRAIN_SEEDS:
        raise ValueError("W8 train seed is outside the frozen seed set")
    return {
        "purpose": "init",
        "identity": {"train_seed": train_seed, "component_path": W8_COMPONENT_PATH},
        "mode": "fresh_keyed_init",
        "predecessor_checkpoint_id": None,
        "w7_checkpoint_transfer": False,
        "optimizer_state_transfer": False,
        "scheduler_state_transfer": False,
        "scaler_state_transfer": False,
    }


def _update_config(config: RunConfig, *, ratio: str, train_seed: int, channel_seed: int, role: str) -> RunConfig:
    value = config.to_dict()
    choices = dict(value["choices"])
    resolved = dict(value["resolved"])
    choices["bw_ratio"] = ratio
    choices["artifact_role"] = role
    resolved.update(
        {
            "bw_ratio": ratio,
            "train_seed": train_seed,
            "channel_seed": channel_seed,
            "k": int(get(f"bandwidth.k_symbols.{W8_DATASET}.{ratio}")),
            "artifact_role": role,
            "execution_profile_id": W8_PROFILE_ID,
            "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
            "accumulation_factor": W8_ACCUMULATION_FACTOR,
            "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
            "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
            "checkpoint_selection_snr_db": checkpoint_selection_snr_db(),
            "checkpoint_selection_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        }
    )
    for key, selected in eligibility_for_role(role).items():
        choices[key] = selected
        resolved[key] = selected
    value["choices"] = choices
    value["resolved"] = resolved
    return RunConfig.from_dict(value)


def load_w8_config(
    ratio: str,
    train_seed: int,
    channel_seed: int,
    *,
    role: str = W8_CORE_ROLE,
) -> RunConfig:
    """Resolve one authorized W8 cell without changing the committed config."""

    if ratio not in unique_core_ratios():
        raise ValueError(f"ratio {ratio!r} is not one of the frozen W8 ratios")
    if train_seed not in W8_TRAIN_SEEDS or channel_seed not in W8_CHANNEL_SEEDS:
        raise ValueError("W8 seed is outside the frozen evaluation seed set")
    if W8_TRAIN_SEEDS.index(train_seed) != W8_CHANNEL_SEEDS.index(channel_seed):
        raise ValueError("W8 train/channel seeds must be zipped, not crossed")
    config = load_experiment(
        _base_config_path(), train_seed=train_seed, channel_seed=channel_seed
    )
    result = _update_config(
        config,
        ratio=ratio,
        train_seed=train_seed,
        channel_seed=channel_seed,
        role=role,
    )
    validate_w8_config(result)
    return result


def validate_w8_config(config: RunConfig) -> None:
    if not isinstance(config, RunConfig):
        raise TypeError("W8 requires a resolved RunConfig")
    resolved = config.resolved
    role = resolved.get("artifact_role")
    if role not in {W8_CORE_ROLE, W8_SMOKE_ROLE}:
        raise ValueError("W8 config has an unknown artifact role")
    if resolved.get("system") != "learned" or resolved.get("dataset") != W8_DATASET:
        raise ValueError("W8 system/dataset differs from the frozen protocol")
    if resolved.get("split") != "train" or resolved.get("channel") != W8_CHANNEL:
        raise ValueError("W8 train split or channel differs from the frozen protocol")
    ratios = unique_core_ratios()
    ratio = resolved.get("bw_ratio")
    if ratio not in ratios:
        raise ValueError("W8 config ratio is outside the exact unique-ratio set")
    expected_k = int(get(f"bandwidth.k_symbols.{W8_DATASET}.{ratio}"))
    if resolved.get("k") != expected_k or expected_k != W8_EXPECTED_K[ratio]:
        raise ValueError("W8 config k does not match the ratio budget")
    if float(resolved.get("lambda")) != W8_EXPECTED_LAMBDA:
        raise ValueError("W8 lambda is not the selected G-4 lambda")
    if float(get("learned_system.lambda_core")) != W8_EXPECTED_LAMBDA or get("learned_system.lambda_status") != "selected_at_G-4":
        raise ValueError("current normative G-4 lambda state is not selected_at_G-4 = 3.0")
    if resolved.get("architecture") != get("learned_system.encoder_arch") or resolved.get("architecture") != "djscc_residual_v1":
        raise ValueError("W8 architecture differs from djscc_residual_v1")
    train_snr = get("channel.train_snr_db_fixed")
    if resolved.get("train_snr_db") != train_snr:
        raise ValueError("W8 training SNR does not resolve to train_snr_db_fixed")
    selected_snr = checkpoint_selection_snr_db()
    if resolved.get("checkpoint_selection_snr_db") != selected_snr:
        raise ValueError("W8 checkpoint selection SNR resolution differs")
    if resolved.get("checkpoint_selection_snr_parameter") != W8_CHECKPOINT_SELECTION_SNR_PARAMETER:
        raise ValueError("W8 checkpoint selection SNR parameter binding differs")
    if resolved.get("execution_profile_id") != W8_PROFILE_ID:
        raise ValueError("W8 execution profile is not confessor_pascal_cu126")
    if resolved.get("train_seed") not in W8_TRAIN_SEEDS or resolved.get("channel_seed") not in W8_CHANNEL_SEEDS:
        raise ValueError("W8 seed is outside the frozen set")
    if W8_TRAIN_SEEDS.index(resolved["train_seed"]) != W8_CHANNEL_SEEDS.index(resolved["channel_seed"]):
        raise ValueError("W8 seed pairing is not zipped")
    if resolved.get("physical_batch_size") != W8_PHYSICAL_BATCH_SIZE or resolved.get("accumulation_factor") != W8_ACCUMULATION_FACTOR or resolved.get("effective_batch_size") != W8_EFFECTIVE_BATCH_SIZE:
        raise ValueError("W8 Pascal batch binding differs")
    if resolved.get("validation_batch_size") != W8_VALIDATION_BATCH_SIZE:
        raise ValueError("W8 validation batch binding differs")
    if int(get(f"learned_system.batch_size.{W8_DATASET}")) != W8_EFFECTIVE_BATCH_SIZE:
        raise ValueError("W8 effective batch differs from learned-system parameter")
    if int(get(f"learned_system.epochs.{W8_DATASET}")) != W8_EPOCHS:
        raise ValueError("W8 epoch schedule differs")
    if get("learned_system.checkpoint_selection_split") != "validation" or get("learned_system.checkpoint_selection_metric") != "top1_accuracy" or get("learned_system.checkpoint_selection_mode") != "max" or get("learned_system.checkpoint_selection_tie_break") != "earliest_epoch":
        raise ValueError("W8 checkpoint-selection rule differs")
    if get("learned_system.papr_report_required") is not True or get("learned_system.papr_constrained_variant_required") is not True:
        raise ValueError("W8 PAPR boundary was weakened")
    if role == W8_SMOKE_ROLE and resolved.get("scientific_status") != "NON_SCIENTIFIC":
        raise ValueError("W8 smoke is not machine-marked NON_SCIENTIFIC")
    expected_eligibility = eligibility_for_role(role)
    for key, expected in expected_eligibility.items():
        if resolved.get(key) != expected:
            raise ValueError(f"W8 eligibility {key} differs for {role}")


def protocol_descriptor() -> dict[str, Any]:
    """Return the complete W8-A source/authorization protocol projection."""

    cells = run_cells()
    return {
        "protocol_version": W8_PROTOCOL_VERSION,
        "dataset": W8_DATASET,
        "ratio_parameter": "params.bandwidth.crossover_ratio + params.bandwidth.efficiency_ratio + params.bandwidth.low_ratio_operating_point",
        "unique_ratios": list(unique_core_ratios()),
        "run_matrix": [cell.to_dict() for cell in cells],
        "train_seeds": list(W8_TRAIN_SEEDS),
        "channel_seeds": list(W8_CHANNEL_SEEDS),
        "seed_pairing": W8_SEED_PAIRING,
        "train_snr_parameter": W8_TRAIN_SNR_PARAMETER,
        "train_snr_db": get("channel.train_snr_db_fixed"),
        "lambda": W8_EXPECTED_LAMBDA,
        "lambda_parameter": "params.learned_system.lambda_core",
        "lambda_status": "selected_at_G-4",
        "architecture": get("learned_system.encoder_arch"),
        "epochs_per_run": W8_EPOCHS,
        "optimizer_recipe": {
            field: get(f"learned_system.{field}")
            for field in (
                "optimizer", "optimizer_implementation", "adam_beta1", "adam_beta2",
                "adam_epsilon", "adam_weight_decay", "adam_amsgrad", "adam_maximize",
                "adam_foreach", "adam_capturable", "adam_differentiable", "adam_fused",
                "lr", "lr_schedule", "lr_schedule_equation", "lr_min", "lr_warmup_epochs",
                "scheduler_step_unit", "scheduler_epoch_indexing", "scheduler_resume_state",
                "amp", "amp_device_type", "amp_dtype", "grad_scaler_enabled",
                "grad_scaler_init_scale", "grad_scaler_growth_factor",
                "grad_scaler_backoff_factor", "grad_scaler_growth_interval",
                "augmentation", "batch_order", "grad_accumulation_allowed", "drop_last", "dataloader_workers",
                "pin_memory", "batch_size_policy", "accumulation_gradient_rule",
                "final_partial_accumulation", "scheduler_steps_under_accumulation",
                "checkpoint_every_epochs", "checkpoint_timing", "checkpoint_resume_unit",
                "corrupt_latest_checkpoint_policy", "incomplete_epoch_policy",
                "checkpoint_schema_version", "w5_checkpoint_selection", "loss",
            )
        },
        "batch": {
            "physical": W8_PHYSICAL_BATCH_SIZE,
            "accumulation": W8_ACCUMULATION_FACTOR,
            "effective": W8_EFFECTIVE_BATCH_SIZE,
            "validation": W8_VALIDATION_BATCH_SIZE,
            "drop_last": get("learned_system.drop_last"),
            "train_samples": W8_TRAIN_SAMPLE_COUNT,
            "microbatches": W8_EXPECTED_MICROBATCHES,
            "final_physical_batch": W8_FINAL_PARTIAL_BATCH,
        },
        "checkpoint_selection": {
            "split": get("learned_system.checkpoint_selection_split"),
            "metric": get("learned_system.checkpoint_selection_metric"),
            "mode": get("learned_system.checkpoint_selection_mode"),
            "tie_break": get("learned_system.checkpoint_selection_tie_break"),
            "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "snr_resolution": W8_TRAIN_SNR_PARAMETER,
            "snr_db": checkpoint_selection_snr_db(),
            "channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
            "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
            "fixed_noise_across_epochs": True,
            "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"],
            "cross_seed_selection": False,
        },
        "initialization": fresh_initialization_identity(0),
        "profile": {
            "execution_profile_id": W8_PROFILE_ID,
            "execution_image_family": W8_EXECUTION_IMAGE_FAMILY,
            "gpu_name": W8_SELECTED_GPU_NAME,
            "gpu_uuid": W8_SELECTED_GPU_UUID,
            "lock_file": "requirements-pascal.lock",
            "lock_file_sha256": get("environment.execution_profiles.confessor_pascal_cu126.lock_file_sha256"),
        },
        "roles": {
            "epoch_record": W8_TRAINING_EPOCH_ROLE,
            "epoch_checkpoint": W8_CHECKPOINT_ROLE,
            "epoch_summary": W8_EPOCH_ROLE,
            "selected_checkpoint": W8_SELECTED_ROLE,
            "campaign_manifest": W8_CAMPAIGN_MANIFEST_ROLE,
            "campaign_completion": W8_CAMPAIGN_COMPLETION_ROLE,
            "w7_initialization": "FORBIDDEN",
            "test_release": "G-12_ONLY",
        },
        "scope": {
            "core_runs": len(cells),
            "er2_randomized_training": "NOT_AUTHORIZED",
            "papr_constrained_training": "NOT_AUTHORIZED_BOUND_REQUIRES_DOWNSTREAM_PROTOCOL_ITEM",
            "er9_training": "NOT_AUTHORIZED",
            "g10": "NOT_AUTHORIZED",
            "test": "SEALED",
        },
        "protocol_hash": canonical_sha256({"protocol_version": W8_PROTOCOL_VERSION, "cells": [cell.to_dict() for cell in cells], "checkpoint_selection_snr_db": checkpoint_selection_snr_db()}),
    }


def protocol_config_hash(config: RunConfig) -> str:
    validate_w8_config(config)
    return canonical_sha256({"protocol": protocol_descriptor(), "config": config.to_dict()})
