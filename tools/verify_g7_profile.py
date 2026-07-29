#!/usr/bin/env python3
"""Offline fail-closed verifier for the committed G-7 profiling report."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from config.run_config import RunConfig, config_hash  # noqa: E402
from models.djscc import build_djscc  # noqa: E402
from models.reference_classifier import build_reference_classifier  # noqa: E402
from profile_djscc_g7 import load_profile_config  # noqa: E402

REPORT_PATH = REPO / "results/profiling/g7_djscc_profile.json"
EXPECTED_IMPLEMENTATION_COMMIT = "26b631ede27a6f88f1d004a66b845c52a658e07c"
_TOP_FIELDS = {
    "schema_version",
    "gate",
    "verdict",
    "profiled_at_utc",
    "implementation_commit",
    "git_dirty",
    "profile_config_path",
    "profile_config_sha256",
    "config_hash",
    "resolved_config",
    "dataset",
    "environment",
    "model",
    "training",
    "memory",
    "limits",
    "conditions",
    "data_isolation",
}
_DATASET_FIELDS = {
    "name",
    "split",
    "dataset_version",
    "archive_filename",
    "archive_bytes",
    "manifest_path",
    "manifest_sha256",
    "configured_train_examples",
}
_ENVIRONMENT_FIELDS = {
    "run_metadata",
    "torch_cuda_available",
    "real_cuda",
    "device_index",
    "device_name",
    "driver_version",
    "total_memory_bytes",
    "compute_capability",
}
_MODEL_FIELDS = {
    "architecture",
    "bw_ratio",
    "k",
    "complex_channels",
    "parameter_count",
    "absolute_parameter_cap",
    "reference_classifier_parameter_count",
    "peak_constraint_enabled",
}
_TRAINING_FIELDS = {
    "optimizer",
    "amp",
    "batch_size",
    "num_workers",
    "warmup_steps",
    "epochs_completed",
    "num_batches",
    "num_examples",
    "epoch_time_s",
    "images_per_second",
    "epoch_mean_loss",
    "projected_epochs",
    "projected_training_hours",
}
_MEMORY_FIELDS = {
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "peak_allocated_gb",
    "peak_reserved_gb",
}
_LIMIT_FIELDS = {
    "configured_batch_size",
    "vram_budget_gb",
    "max_wall_clock_hours_per_run",
    "max_params",
    "reference_classifier_params",
}
_CONDITION_FIELDS = {
    "primary_architecture_full_epoch",
    "configured_batch_size_32",
    "peak_reserved_vram",
    "projected_training_wall_time",
    "absolute_parameter_cap",
    "reference_parameter_cap",
    "clean_implementation_commit",
    "real_cuda",
    "training_split_only",
    "report_consistency",
}
_ISOLATION_FIELDS = {
    "scope",
    "test_split_accessed",
    "test_inference",
    "test_accuracy_computed",
}


class VerificationError(ValueError):
    """A closed G-7 evidence-verification failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    missing = expected - set(value)
    unexpected = set(value) - expected
    _require(
        not missing and not unexpected,
        f"{label} fields differ: missing={sorted(missing)}, "
        f"unexpected={sorted(unexpected)}",
    )
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid G-7 report: {exc}") from None
    return _exact_fields(value, _TOP_FIELDS, "report")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = _load(path)
    dataset = _exact_fields(report["dataset"], _DATASET_FIELDS, "dataset")
    environment = _exact_fields(
        report["environment"], _ENVIRONMENT_FIELDS, "environment"
    )
    model = _exact_fields(report["model"], _MODEL_FIELDS, "model")
    training = _exact_fields(report["training"], _TRAINING_FIELDS, "training")
    memory = _exact_fields(report["memory"], _MEMORY_FIELDS, "memory")
    limits = _exact_fields(report["limits"], _LIMIT_FIELDS, "limits")
    conditions = _exact_fields(
        report["conditions"], _CONDITION_FIELDS, "conditions"
    )
    isolation = _exact_fields(
        report["data_isolation"], _ISOLATION_FIELDS, "data_isolation"
    )

    _require(report["schema_version"] == 1, "unsupported G-7 schema version")
    _require(report["gate"] == "G-7", "report is not for G-7")
    _require(report["implementation_commit"] == EXPECTED_IMPLEMENTATION_COMMIT, "wrong implementation commit")
    _require(report["git_dirty"] is False, "implementation state was dirty")
    _require(
        report["profile_config_path"] == "configs/learned-g7-profile.yaml",
        "wrong profile configuration path",
    )
    config_path = REPO / report["profile_config_path"]
    _require(config_path.is_file(), "profile configuration is absent")
    _require(
        report["profile_config_sha256"] == _sha256(config_path),
        "profile configuration file hash disagrees",
    )
    expected_config = load_profile_config(config_path)
    try:
        archived_config = RunConfig.from_dict(report["resolved_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"archived resolved config is invalid: {exc}") from None
    _require(
        archived_config.to_dict() == expected_config.to_dict(),
        "archived resolved config disagrees with committed profile config",
    )
    _require(
        report["config_hash"] == config_hash(archived_config),
        "config hash disagrees with resolved config",
    )
    resolved = archived_config.resolved
    parameters = archived_config.parameters
    _require(
        resolved["implementation_commit"] == EXPECTED_IMPLEMENTATION_COMMIT,
        "config binds the wrong implementation commit",
    )

    expected_dataset = "imagenette160"
    expected_split = "train"
    expected_ratio = "r_1_2"
    expected_architecture = "djscc_residual_v1"
    expected_k = parameters["bandwidth"]["k_symbols"][expected_dataset][expected_ratio]
    _require(dataset["name"] == expected_dataset == resolved["dataset"], "wrong dataset")
    _require(dataset["split"] == expected_split == resolved["split"], "wrong split")
    _require(model["bw_ratio"] == expected_ratio == resolved["bw_ratio"], "wrong ratio")
    _require(
        model["architecture"] == expected_architecture == resolved["architecture"],
        "wrong architecture",
    )
    _require(model["k"] == expected_k == resolved["k"], "wrong complex-symbol budget k")
    expected_channels = parameters["learned_system"]["encoder_output_complex_channels"][
        expected_ratio
    ]
    _require(model["complex_channels"] == expected_channels, "wrong latent complex-channel count")

    dataset_parameters = parameters["datasets"][expected_dataset]
    _require(
        dataset["dataset_version"] == resolved["dataset_version"] == dataset_parameters["archive_sha256"],
        "dataset identity disagrees",
    )
    _require(dataset["archive_filename"] == dataset_parameters["archive_filename"], "archive filename disagrees")
    _require(dataset["archive_bytes"] == dataset_parameters["archive_bytes"], "archive length disagrees")
    _require(
        dataset["configured_train_examples"] == dataset_parameters["train_images"],
        "configured training-example count disagrees",
    )
    expected_manifest_path = (
        Path(parameters["datasets"]["manifest_dir"])
        / dataset_parameters["manifest_filename"]
    )
    _require(dataset["manifest_path"] == expected_manifest_path.as_posix(), "manifest path disagrees")
    _require(
        dataset["manifest_sha256"] == dataset_parameters["manifest_sha256"],
        "manifest identity disagrees with config",
    )
    _require(
        _sha256(REPO / expected_manifest_path) == dataset["manifest_sha256"],
        "manifest bytes disagree",
    )

    expected_run_metadata = set(parameters["environment"]["record_in_run_metadata"])
    run_metadata = _exact_fields(
        environment["run_metadata"], expected_run_metadata, "environment.run_metadata"
    )
    for field in (
        "python_version",
        "torch_version",
        "cuda_version",
        "driver_version",
        "device_name",
        "lock_file_sha256",
    ):
        _require(isinstance(run_metadata[field], str) and run_metadata[field], f"missing CUDA environment data: {field}")
    _require(environment["torch_cuda_available"] is True, "profile did not use CUDA")
    _require(environment["real_cuda"] is True, "profile is a CPU projection")
    _require(environment["device_index"] == 0, "profile did not use primary CUDA device")
    _require(
        environment["device_name"] == run_metadata["device_name"]
        and isinstance(environment["device_name"], str)
        and environment["device_name"],
        "CUDA device identity disagrees",
    )
    _require(
        environment["driver_version"] == run_metadata["driver_version"],
        "CUDA driver identity disagrees",
    )
    _require(
        isinstance(environment["total_memory_bytes"], int)
        and environment["total_memory_bytes"] > 0,
        "missing CUDA memory data",
    )
    _require(
        isinstance(environment["compute_capability"], list)
        and len(environment["compute_capability"]) == 2
        and all(isinstance(value, int) for value in environment["compute_capability"]),
        "missing CUDA compute capability",
    )

    rebuilt_model = build_djscc(archived_config)
    expected_parameter_count = rebuilt_model.total_parameter_count
    reference_model = build_reference_classifier(expected_dataset)
    expected_reference_count = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    absolute_cap = int(
        float(parameters["learned_system"]["max_params_millions"]) * 1_000_000
    )
    _require(model["parameter_count"] == expected_parameter_count, "incorrect model parameter count")
    _require(model["absolute_parameter_cap"] == absolute_cap, "incorrect absolute parameter cap")
    _require(
        model["reference_classifier_parameter_count"] == expected_reference_count,
        "incorrect reference parameter cap",
    )
    _require(model["peak_constraint_enabled"] is False, "profile unexpectedly used PAPR constraint")
    _require(expected_parameter_count <= absolute_cap, "model exceeds absolute parameter cap")
    _require(expected_parameter_count <= expected_reference_count, "model exceeds reference parameter cap")

    configured_batch = parameters["learned_system"]["batch_size"][expected_dataset]
    _require(training["optimizer"] == parameters["learned_system"]["optimizer"], "wrong optimizer")
    _require(training["amp"] is parameters["learned_system"]["amp"], "wrong AMP setting")
    _require(
        training["batch_size"] == configured_batch
        and training["batch_size"] >= 32,
        "batch size below configured 32",
    )
    _require(training["num_workers"] == resolved["num_workers"] == 0, "wrong worker count")
    _require(training["warmup_steps"] == resolved["warmup_steps"], "wrong warm-up count")
    expected_examples = dataset_parameters["train_images"]
    expected_batches = math.ceil(expected_examples / configured_batch)
    _require(training["epochs_completed"] == 1, "incomplete epoch count")
    _require(
        training["num_examples"] == expected_examples
        and training["num_batches"] == expected_batches,
        "incomplete epoch examples or batches",
    )
    for field in (
        "epoch_time_s",
        "images_per_second",
        "epoch_mean_loss",
        "projected_training_hours",
    ):
        _require(
            isinstance(training[field], int | float)
            and not isinstance(training[field], bool)
            and math.isfinite(training[field])
            and training[field] > 0,
            f"invalid measured training field: {field}",
        )
    _require(
        math.isclose(
            training["images_per_second"],
            training["num_examples"] / training["epoch_time_s"],
            rel_tol=1e-12,
        ),
        "images-per-second arithmetic disagrees",
    )
    expected_epochs = parameters["learned_system"]["epochs"][expected_dataset]
    _require(training["projected_epochs"] == expected_epochs, "wrong projected epoch count")
    _require(
        math.isclose(
            training["projected_training_hours"],
            training["epoch_time_s"] * expected_epochs / 3600,
            rel_tol=1e-12,
        ),
        "projected runtime arithmetic disagrees",
    )

    gib = 1024**3
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        _require(isinstance(memory[field], int) and memory[field] > 0, f"invalid {field}")
    _require(memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"], "allocated VRAM exceeds reserved VRAM")
    _require(
        math.isclose(memory["peak_allocated_gb"], memory["peak_allocated_bytes"] / gib, rel_tol=1e-12)
        and math.isclose(memory["peak_reserved_gb"], memory["peak_reserved_bytes"] / gib, rel_tol=1e-12),
        "VRAM byte/GB arithmetic disagrees",
    )
    vram_limit = float(parameters["compute"]["vram_budget_gb"])
    runtime_limit = float(parameters["compute"]["max_wall_clock_hours_per_run"])
    _require(limits["configured_batch_size"] == configured_batch, "batch-size limit disagrees")
    _require(limits["vram_budget_gb"] == vram_limit, "VRAM limit disagrees")
    _require(limits["max_wall_clock_hours_per_run"] == runtime_limit, "runtime limit disagrees")
    _require(limits["max_params"] == absolute_cap, "parameter limit disagrees")
    _require(limits["reference_classifier_params"] == expected_reference_count, "reference limit disagrees")
    _require(memory["peak_reserved_gb"] <= vram_limit, "peak VRAM above configured limit")
    _require(
        training["projected_training_hours"] <= runtime_limit,
        "projected runtime above configured limit",
    )

    _require(
        isolation
        == {
            "scope": "training_only",
            "test_split_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
        },
        "report contains a test-split claim",
    )
    expected_conditions = {
        "primary_architecture_full_epoch": "PASS",
        "configured_batch_size_32": "PASS",
        "peak_reserved_vram": "PASS",
        "projected_training_wall_time": "PASS",
        "absolute_parameter_cap": "PASS",
        "reference_parameter_cap": "PASS",
        "clean_implementation_commit": "PASS",
        "real_cuda": "PASS",
        "training_split_only": "PASS",
        "report_consistency": "PASS",
    }
    _require(conditions == expected_conditions, "G-7 component verdict is inconsistent")
    _require(report["verdict"] == "PASS", "overall G-7 PASS verdict is inconsistent")

    return {
        "gate": "G-7",
        "verdict": "PASS",
        "implementation_commit": report["implementation_commit"],
        "parameter_count": model["parameter_count"],
        "epoch_time_s": training["epoch_time_s"],
        "peak_reserved_gb": memory["peak_reserved_gb"],
        "projected_training_hours": training["projected_training_hours"],
    }


def main() -> int:
    try:
        result = verify()
    except VerificationError as exc:
        print(f"G-7 profile verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "G-7 profile verification PASS: "
        f"commit={result['implementation_commit'][:12]}, "
        f"params={result['parameter_count']}, "
        f"epoch={result['epoch_time_s']:.3f}s, "
        f"reserved={result['peak_reserved_gb']:.3f} GB, "
        f"projected={result['projected_training_hours']:.3f} h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
