"""The frozen, result-independent W7/G-4 protocol and run configuration.

This module reads all existing scientific values through ``params``.  The
owner-supplied pre-result clarifications (seed pair, validation noise policy,
PSNR aggregation and checkpoint rule) are represented in the W7-A contract and
not inferred from a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.params import REPO_ROOT, get
from config.run_config import FrozenMap, RunConfig, canonical_sha256, load_experiment

W7_CONTRACT_VERSION = "w7-g4-pre-execution-v1"
W7_PROFILE_ID = "confessor_pascal_cu126"
W7_EXECUTION_IMAGE_FAMILY = "pascal-cu126-requirements-pascal-lock-v1"
W7_DATASET = "imagenette160"
W7_RATIO = "r_1_6"
W7_TRAIN_SEED = 0
W7_CHANNEL_SEED = 0
W7_TRAINING_SNR_DB = 7  # literal-ok: owner-frozen W7-A training SNR
W7_CALIBRATION_SNR_DB = 7  # literal-ok: owner-frozen W7-A calibration SNR
W7_PSNR_SNR_DB = 15  # literal-ok: owner-frozen W7-A PSNR evaluation SNR
W7_VALIDATION_BATCH_SIZE = 32  # literal-ok: owner-frozen W7-A validation batch
W7_PHYSICAL_BATCH_SIZE = 32  # literal-ok: profile ladder first attempt
W7_ACCUMULATION_FACTOR = 1  # literal-ok: profile ladder first attempt
W7_LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)  # literal-ok: owner-frozen W7-A lambda grid
W7_VALIDATION_NOISE_POLICY = "keyed_channel_noise_same_per_image_across_lambda"
W7_VALIDATION_ORDER = "stable_manifest_order"
W7_PSNR_ZERO_MSE = "positive_infinity_canonical_string"
W7_PSNR_AGGREGATION = "arithmetic_mean_over_complete_validation_denominator"
W7_PAPR_AGGREGATION = "arithmetic_mean_over_complete_validation_denominator"
W7_CAMPAIGN_ORDER = "exact_configured_lambda_grid_order"
W7_INCOMPLETE_POLICY = "replay_from_latest_authenticated_completed_epoch"
W7_CORRUPT_LATEST_POLICY = "hold_no_older_fallback"


def _full_sha(value: object, width: int) -> bool:
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _expected_lambda_grid() -> tuple[float, ...]:
    value = tuple(float(item) for item in get("learned_system.lambda_grid"))
    if value != W7_LAMBDA_GRID:
        raise ValueError(f"configured lambda grid differs from frozen W7 grid: {value}")
    return value


def _base_config_path() -> Path:
    return REPO_ROOT / "configs/learned-w7-g4-pilot.yaml"


def load_w7_config(
    lambda_value: float | None = None,
    *,
    role: str | None = None,
    physical_batch_size: int | None = None,
    accumulation_factor: int | None = None,
    validation_batch_size: int | None = None,
) -> RunConfig:
    """Resolve the committed W7 choices, changing only the requested λ entry.

    A future campaign obtains one instance per grid entry from this function.
    The committed config remains a choices file; no result can alter the
    configured grid or the provisional ``lambda_core`` parameter.
    """

    _expected_lambda_grid()
    config = load_experiment(_base_config_path())
    resolved = config.resolved.to_dict()
    choices = config.choices.to_dict()
    if lambda_value is not None:
        value = float(lambda_value)
        if value not in W7_LAMBDA_GRID:
            raise ValueError(f"lambda {value!r} is outside the frozen W7 grid")
        resolved["lambda"] = value
        resolved["lambda_grid_index"] = W7_LAMBDA_GRID.index(value)
    else:
        resolved["lambda_grid_index"] = W7_LAMBDA_GRID.index(float(resolved["lambda"]))
    if role is not None:
        choices["artifact_role"] = role
        resolved["artifact_role"] = role
        eligibility = eligibility_for_role(role)
        for key, value in eligibility.items():
            if key != "artifact_role":
                choices[key] = value
                resolved[key] = value
    if physical_batch_size is not None:
        resolved["physical_batch_size"] = int(physical_batch_size)
    if accumulation_factor is not None:
        resolved["accumulation_factor"] = int(accumulation_factor)
    if validation_batch_size is not None:
        resolved["validation_batch_size"] = int(validation_batch_size)
    updated = config.to_dict()
    updated["choices"] = choices
    updated["resolved"] = resolved
    result = RunConfig.from_dict(updated)
    validate_w7_config(result)
    return result


def eligibility_for_role(role: str) -> dict[str, str]:
    if role == "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT":
        return {
            "artifact_role": role,
            "selection_eligibility": "ELIGIBLE_FOR_OWN_G4_ONLY",
            "reporting_eligibility": "NOT_ELIGIBLE_UNTIL_G4",
            "w7_g4_eligibility": "ELIGIBLE_FOR_G4_CANDIDATE",
            "w8_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        }
    if role == "NON_SCIENTIFIC_PROFILE":
        return {
            "artifact_role": role,
            "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
            "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
            "w7_g4_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
            "w8_eligibility": "NOT_ELIGIBLE_FOR_W8",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        }
    raise ValueError(f"unknown W7 artifact role: {role!r}")


def validate_w7_config(config: RunConfig) -> None:
    if not isinstance(config, RunConfig):
        raise TypeError("W7 requires a resolved RunConfig")
    resolved = config.resolved
    if resolved.get("system") != "learned":
        raise ValueError("W7 system must be learned")
    if resolved.get("dataset") != W7_DATASET:
        raise ValueError("W7/G-4 dataset is imagenette160")
    if resolved.get("split") != "train":
        raise ValueError("W7 training config must use split=train")
    if resolved.get("bw_ratio") != W7_RATIO:
        raise ValueError("W7/G-4 ratio must resolve to r_1_6")
    headline_selector = get("bandwidth.headline_ratio")
    if headline_selector != "crossover_ratio" or get(f"bandwidth.{headline_selector}") != W7_RATIO:
        raise ValueError("W7 ratio is not the frozen headline/crossover ratio")
    if resolved.get("channel") != "awgn":
        raise ValueError("W7 channel must be the registered AWGN implementation")
    if get("learned_system.lambda_calibration_snr_db") != W7_CALIBRATION_SNR_DB:
        raise ValueError("W7 calibration SNR differs from the frozen 7 dB protocol")
    if get("learned_system.lambda_psnr_eval_snr_db") != W7_PSNR_SNR_DB:
        raise ValueError("W7 PSNR SNR differs from the frozen 15 dB protocol")
    if float(get("learned_system.lambda_acc_tolerance_pp")) != 1.0:  # literal-ok: owner-frozen W7-A accuracy tolerance
        raise ValueError("W7 accuracy tolerance differs from the frozen 1 pp protocol")
    if float(get("learned_system.lambda_psnr_floor_db")) != 20.0:  # literal-ok: owner-frozen W7-A primary floor
        raise ValueError("W7 primary PSNR floor differs from the frozen 20 dB protocol")
    if float(get("learned_system.lambda_psnr_floor_relaxed_db")) != 16.0:  # literal-ok: owner-frozen W7-A relaxed floor
        raise ValueError("W7 relaxed PSNR floor differs from the frozen 16 dB protocol")
    if resolved.get("train_snr_db") != W7_TRAINING_SNR_DB:
        raise ValueError("W7 training SNR differs from the frozen 7 dB protocol")
    if resolved.get("train_seed") != W7_TRAIN_SEED:
        raise ValueError("W7 train seed differs from the frozen seed pair")
    if resolved.get("channel_seed") != W7_CHANNEL_SEED:
        raise ValueError("W7 channel seed differs from the frozen seed pair")
    if float(resolved.get("lambda")) not in _expected_lambda_grid():
        raise ValueError("W7 lambda is outside the configured grid")
    if resolved.get("execution_profile_id") != W7_PROFILE_ID:
        raise ValueError("W7 execution profile is not confessor_pascal_cu126")
    if resolved.get("architecture") != get("learned_system.encoder_arch"):
        raise ValueError("W7 architecture differs from the frozen DJSCC architecture")
    expected_k = get(f"bandwidth.k_symbols.{W7_DATASET}.{W7_RATIO}")
    if resolved.get("k") != expected_k:
        raise ValueError("W7 k does not match the selected ratio")
    target_batch = int(get(f"learned_system.batch_size.{W7_DATASET}"))
    physical = resolved.get("physical_batch_size")
    accumulation = resolved.get("accumulation_factor")
    validation = resolved.get("validation_batch_size")
    if not isinstance(physical, int) or isinstance(physical, bool) or physical <= 0:
        raise ValueError("W7 physical batch size must be a positive integer")
    if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation <= 0:
        raise ValueError("W7 accumulation factor must be a positive integer")
    if physical * accumulation != target_batch:
        raise ValueError("W7 physical batch × accumulation must equal effective batch 32")
    if physical not in {target_batch // factor for factor in (1, 2, 4, 8, 16)}:  # literal-ok: owner-frozen W7 accumulation ladder
        raise ValueError("W7 physical batch is outside the predetermined accumulation ladder")
    if not isinstance(validation, int) or isinstance(validation, bool) or validation <= 0:
        raise ValueError("W7 validation batch size must be a positive integer")
    role = resolved.get("artifact_role")
    if role not in {
        "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT",
        "NON_SCIENTIFIC_PROFILE",
    }:
        raise ValueError("W7 config has an unknown artifact role")
    expected_eligibility = eligibility_for_role(role)
    for key, value in expected_eligibility.items():
        if resolved.get(key) != value:
            raise ValueError(f"W7 config eligibility {key} differs for {role}")
    epochs = get(f"learned_system.epochs.{W7_DATASET}")
    if epochs != 100:  # literal-ok: owner-frozen W7 epoch count
        raise ValueError("W7 protocol requires the configured 100-epoch Imagenette schedule")
    if get("learned_system.checkpoint_selection_split") != "validation":
        raise ValueError("W7 checkpoint selection must use validation")
    if get("learned_system.checkpoint_selection_metric") != "top1_accuracy":
        raise ValueError("W7 checkpoint selection metric differs")
    if get("learned_system.checkpoint_selection_mode") != "max":
        raise ValueError("W7 checkpoint selection mode differs")
    if get("learned_system.checkpoint_selection_tie_break") != "earliest_epoch":
        raise ValueError("W7 checkpoint tie-break differs")


def protocol_config_hash(config: RunConfig) -> str:
    """Hash the protocol with λ normalised, so candidates compare like-for-like."""

    validate_w7_config(config)
    value = config.to_dict()
    value["choices"] = dict(value["choices"])
    value["resolved"] = dict(value["resolved"])
    value["choices"]["lambda"] = "lambda_grid_entry"
    value["resolved"]["lambda"] = "lambda_grid_entry"
    value["resolved"].pop("lambda_grid_index", None)
    return canonical_sha256(value)


def protocol_descriptor() -> dict[str, Any]:
    """Return the complete result-independent G-4 decision protocol."""

    grid = list(_expected_lambda_grid())
    calibration_ratio = get("learned_system.lambda_calibration_ratio")
    if calibration_ratio != "headline_ratio":
        raise ValueError("lambda calibration ratio is no longer the headline selector")
    headline_selector = get("bandwidth.headline_ratio")
    ratio = get(f"bandwidth.{headline_selector}")
    if ratio != W7_RATIO:
        raise ValueError("lambda calibration ratio does not resolve to r_1_6")
    return {
        "protocol_version": W7_CONTRACT_VERSION,
        "dataset": W7_DATASET,
        "split": "validation_for_selection_and_metrics",
        "ratio": ratio,
        "ratio_parameter": "params.learned_system.lambda_calibration_ratio",
        "ratio_selector_resolution": {
            "parameter": "params.bandwidth.headline_ratio",
            "selector": get("bandwidth.headline_ratio"),
            "resolved": ratio,
        },
        "lambda_grid": grid,
        "lambda_order": W7_CAMPAIGN_ORDER,
        "train_seed": W7_TRAIN_SEED,
        "channel_seed": W7_CHANNEL_SEED,
        "seed_pairing": "one_compound_pair_same_for_every_lambda",
        "training_snr_db": W7_TRAINING_SNR_DB,
        "calibration_snr_db": int(get("learned_system.lambda_calibration_snr_db")),
        "psnr_eval_snr_db": int(get("learned_system.lambda_psnr_eval_snr_db")),
        "accuracy_tolerance_pp": float(get("learned_system.lambda_acc_tolerance_pp")),
        "primary_psnr_floor_db": float(get("learned_system.lambda_psnr_floor_db")),
        "relaxed_psnr_floor_db": float(get("learned_system.lambda_psnr_floor_relaxed_db")),
        "checkpoint_selection": {
            "after": "each_completed_published_epoch",
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_db": W7_CALIBRATION_SNR_DB,
            "no_psnr_selection": True,
        },
        "validation": {
            "denominator": int(get(f"datasets.{W7_DATASET}.val_images")),
            "order": W7_VALIDATION_ORDER,
            "augmentation": False,
            "drop_last": False,
            "batch_size": W7_VALIDATION_BATCH_SIZE,
            "noise_policy": W7_VALIDATION_NOISE_POLICY,
            "test_access": "structurally_rejected",
        },
        "psnr": {
            "definition": "RGB reconstruction against canonical target tensor",
            "data_range": float(get("preprocessing.psnr_data_range")),
            "per_image": "10*log10(data_range^2/per_image_mse)",
            "mse": "mean_over_all_RGB_pixels",
            "aggregation": W7_PSNR_AGGREGATION,
            "zero_mse": W7_PSNR_ZERO_MSE,
            "clipping": bool(get("preprocessing.reconstruction_clipped_before_metrics")),
        },
        "papr": {
            "domain": "symbol_domain",
            "aggregation": W7_PAPR_AGGREGATION,
        },
        "training": {
            "epochs": int(get(f"learned_system.epochs.{W7_DATASET}")),
            "effective_batch_size": int(get(f"learned_system.batch_size.{W7_DATASET}")),
            "physical_batch_ladder": [32, 16, 8, 4, 2],  # literal-ok: owner-frozen W7-A batch ladder
            "accumulation_ladder": [1, 2, 4, 8, 16],  # literal-ok: owner-frozen W7 accumulation ladder
            "drop_last": False,
            "incomplete_epoch": W7_INCOMPLETE_POLICY,
            "corrupt_latest": W7_CORRUPT_LATEST_POLICY,
        },
        "execution_profile_id": W7_PROFILE_ID,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "decision": {
            "baseline": "lambda_0_selected_checkpoint_validation_top1",
            "accuracy": "candidate_top1 >= baseline_top1 - tolerance_pp/100",
            "primary": "numeric_minimum_lambda_accuracy_ok_and_psnr_at_least_primary_floor",
            "relaxed": "only_if_primary_empty_numeric_minimum_lambda_accuracy_ok_and_psnr_at_least_relaxed_floor",
            "no_solution": "G4_HOLD_DEC2_REVERSAL_REPLAN_REQUIRED",
        },
    }


def validate_profile_binding(binding: dict[str, Any]) -> None:
    required = {
        "authentication_status",
        "execution_profile_id",
        "gpu_uuid",
        "gpu_name",
        "gpu_compute_capability",
        "lock_file_sha256",
        "git_commit",
        "config_hash",
    }
    missing = required - set(binding)
    if missing:
        raise ValueError(f"authenticated W7 profile binding is missing {sorted(missing)}")
    if binding["authentication_status"] != "PASSED":
        raise ValueError("W7 profile binding was not authenticated")
    if binding["execution_profile_id"] != W7_PROFILE_ID:
        raise ValueError("W7 profile binding is not confessor_pascal_cu126")
    profile = get(f"environment.execution_profiles.{W7_PROFILE_ID}")
    if binding["gpu_uuid"] not in profile["allowed_gpu_uuids"]:
        raise ValueError("W7 GPU UUID is not registered for the selected profile")
    if binding["gpu_name"] not in profile["allowed_gpu_names"]:
        raise ValueError("W7 GPU name is not registered for the selected profile")
    if str(binding["gpu_compute_capability"]) != str(profile["compute_capability"]):
        raise ValueError("W7 GPU compute capability differs")
    if binding["lock_file_sha256"] != profile["lock_file_sha256"]:
        raise ValueError("W7 Pascal lock SHA differs")
    if not _full_sha(binding["git_commit"], 40) or not _full_sha(binding["config_hash"], 64):  # literal-ok: Git/SHA-256 widths
        raise ValueError("W7 profile binding has an invalid source/config digest")
