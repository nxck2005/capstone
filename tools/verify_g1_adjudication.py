#!/usr/bin/env python3
"""Verify the committed G-1 adjudication without network or test-split access."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.run_config import RunConfig, config_hash  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = "9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002"
EXPECTED_CONFIG_HASH = "a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1"
EXPECTED_DATASET_VERSION = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
EXPECTED_MANIFEST_HASH = "224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889"
EXPECTED_TRAINING_SOURCE_COMMIT = "89a3af48c48a91d6d272ba62337f890c59bb40a5"
EXPECTED_EVIDENCE_PARENT_COMMIT = "98f62ec925f00d099b78c3fe4f9628a7cd76f3c3"
EXPECTED_CHECKPOINT_PATH = "checkpoints/reference_classifier/epoch-99.pt"
EXPECTED_CHECKPOINT_BYTES = 92_121_803
EXPECTED_EPOCHS = list(range(100))
EXPECTED_STEPS = 67
EXPECTED_CORRECT = 898
EXPECTED_TOTAL = 1000
EXPECTED_ACCURACY = EXPECTED_CORRECT / EXPECTED_TOTAL

REQUIRED_ADJUDICATION_FIELDS = {
    "schema_version",
    "gate",
    "verdict",
    "adjudicated_on",
    "training_source_commit",
    "evidence_parent_commit",
    "training_command",
    "repository_clean_at_training",
    "dataset",
    "split",
    "classifier_variant",
    "architecture",
    "pretrained_weights",
    "train_seed",
    "parameter_count",
    "metric",
    "floor",
    "best_epoch",
    "best_n_correct",
    "best_n_total",
    "best_accuracy",
    "final_epoch",
    "final_n_correct",
    "final_n_total",
    "final_accuracy",
    "checkpoint_sha256",
    "checkpoint_repository_path",
    "checkpoint_external_artifact",
    "config_hash",
    "dataset_version",
    "split_manifest_hash",
    "environment",
    "lineage",
    "test_isolation",
    "preflight_checks",
    "final_checks",
}

REQUIRED_PREFLIGHT_CHECKS = {
    "archive_provenance",
    "cpu_lock_clean_install",
    "cuda_device_matmul",
    "dataset_verification",
    "documentation_consistency",
    "literal_lint",
    "manifest_materialization",
    "packetisation",
    "pytest_250",
    "spec_views",
}

REQUIRED_FINAL_CHECKS = {
    "archive_provenance",
    "dataset_verification",
    "documentation_consistency",
    "git_diff_check",
    "literal_lint",
    "manifest_materialization",
    "packetisation",
    "pytest_250",
    "spec_views",
}


class VerificationError(ValueError):
    """A closed verification failure with an actionable reason."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid JSON from {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    _require("\\" not in value, f"{label} must use POSIX separators")
    _require(not value.startswith("/"), f"{label} must be relative, not absolute")
    _require(re.match(r"^[A-Za-z]:", value) is None, f"{label} must not contain a drive root")
    path = PurePosixPath(value)
    _require(".." not in path.parts, f"{label} must remain inside the repository")
    _require(path.as_posix() == value, f"{label} must be canonical repository-relative POSIX")
    return value


def _epoch_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot read epoch evidence {path}: {exc}") from None
    _require(len(lines) == len(EXPECTED_EPOCHS), "epoch evidence must contain exactly 100 records")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid epoch JSON at line {line_number}: {exc}") from None
        _require(isinstance(value, dict), f"epoch line {line_number} must be an object")
        _require(set(value) == {"training", "validation"}, f"epoch line {line_number} schema differs")
        _require(isinstance(value["training"], dict), f"training line {line_number} must be an object")
        _require(isinstance(value["validation"], dict), f"validation line {line_number} must be an object")
        records.append(value)
    return records


def _verify_git_lineage(repo: Path) -> None:
    if not (repo / ".git").exists():
        return
    for commit in (EXPECTED_TRAINING_SOURCE_COMMIT, EXPECTED_EVIDENCE_PARENT_COMMIT):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        _require(result.returncode == 0, f"lineage commit is absent: {commit}")
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_TRAINING_SOURCE_COMMIT,
            EXPECTED_EVIDENCE_PARENT_COMMIT,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(result.returncode == 0, "training source commit is not an ancestor of evidence commit")


def verify(repo: Path = REPO) -> dict[str, Any]:
    """Fail closed on any disagreement and return a compact success summary."""

    repo = repo.resolve()
    results = repo / "results/reference_classifier"
    adjudication = _load_json(results / "g1_adjudication.json")
    config_value = _load_json(results / "resolved_config.json")
    summary = _load_json(results / "validation_summary.json")
    checkpoint_metadata = _load_json(results / "best_checkpoint.json")
    records = _epoch_records(results / "epochs.jsonl")

    missing = REQUIRED_ADJUDICATION_FIELDS - set(adjudication)
    _require(not missing, f"adjudication fields missing: {sorted(missing)}")

    training_epochs: list[int] = []
    validation_epochs: list[int] = []
    validation_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        training = record["training"]
        validation = record["validation"]
        _require(
            set(training) == {"epoch", "loss", "lr", "steps"},
            f"training epoch {index} schema differs",
        )
        _require(_is_int(training["epoch"]), f"training epoch {index} is not an integer")
        _require(
            _is_int(training["steps"]) and training["steps"] == EXPECTED_STEPS,
            f"training epoch {index} must contain exactly {EXPECTED_STEPS} steps",
        )
        training_epochs.append(training["epoch"])

        _require(
            set(validation) == {"epoch", "n_correct", "n_total", "top1_accuracy"},
            f"validation epoch {index} schema differs",
        )
        _require(_is_int(validation["epoch"]), f"validation epoch {index} is not an integer")
        _require(_is_int(validation["n_correct"]), f"validation epoch {index} n_correct is not integral")
        _require(_is_int(validation["n_total"]), f"validation epoch {index} n_total is not integral")
        _require(validation["n_total"] == EXPECTED_TOTAL, f"validation epoch {index} total is not 1000")
        _require(
            0 <= validation["n_correct"] <= validation["n_total"],
            f"validation epoch {index} count is outside its total",
        )
        expected_accuracy = validation["n_correct"] / validation["n_total"]
        _require(
            type(validation["top1_accuracy"]) is float
            and validation["top1_accuracy"] == expected_accuracy,
            f"validation epoch {index} accuracy is inconsistent with integral counts",
        )
        validation_epochs.append(validation["epoch"])
        validation_records.append(validation)

    _require(
        training_epochs == EXPECTED_EPOCHS,
        "training epochs must be contiguous, ordered, and unique from 0 through 99",
    )
    _require(
        validation_epochs == EXPECTED_EPOCHS,
        "validation epochs must be contiguous, ordered, and unique from 0 through 99",
    )

    best_count = max(record["n_correct"] for record in validation_records)
    best_epoch = min(
        record["epoch"] for record in validation_records if record["n_correct"] == best_count
    )
    best_record = validation_records[best_epoch]
    final_record = validation_records[-1]
    _require(best_epoch == 99, "recomputed earliest-tie best epoch is not 99")
    _require(final_record["epoch"] == 99, "final epoch is not 99")
    for label, record in (("best", best_record), ("final", final_record)):
        _require(
            record["n_correct"] == EXPECTED_CORRECT
            and record["n_total"] == EXPECTED_TOTAL
            and record["top1_accuracy"] == EXPECTED_ACCURACY,
            f"{label} result is not 898/1000 = 0.898",
        )

    try:
        run_config = RunConfig.from_dict(config_value)
        computed_config_hash = config_hash(run_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"resolved configuration is invalid: {exc}") from None
    resolved = config_value.get("resolved")
    parameters = config_value.get("parameters")
    _require(isinstance(resolved, dict) and isinstance(parameters, dict), "resolved configuration shape differs")
    try:
        dataset_parameters = parameters["datasets"]["imagenette160"]
        classifier_parameters = parameters["reference_classifier"]
        environment_parameters = parameters["environment"]
    except (KeyError, TypeError):
        raise VerificationError("resolved configuration lacks required G-1 parameter roots") from None

    floor = dataset_parameters.get("clean_acc_floor")
    _require(type(floor) is float, "configured Imagenette floor must be a float")
    _require(adjudication["floor"] == floor, "adjudication floor does not match configuration")
    _require(EXPECTED_ACCURACY > floor, "committed G-1 result does not clear the configured floor")
    _require(computed_config_hash == EXPECTED_CONFIG_HASH, "resolved configuration hash changed")

    adjudication_path = _relative_posix(
        adjudication["checkpoint_repository_path"],
        label="adjudication checkpoint_repository_path",
    )
    _require(adjudication_path == EXPECTED_CHECKPOINT_PATH, "adjudication checkpoint path changed")

    expected_adjudication = {
        "schema_version": 1,
        "gate": "G-1",
        "verdict": "PASS",
        "adjudicated_on": "2026-07-29",
        "training_source_commit": EXPECTED_TRAINING_SOURCE_COMMIT,
        "evidence_parent_commit": EXPECTED_EVIDENCE_PARENT_COMMIT,
        "repository_clean_at_training": True,
        "dataset": "imagenette160",
        "split": "validation",
        "classifier_variant": "clean",
        "architecture": "resnet18",
        "pretrained_weights": None,
        "train_seed": 0,
        "parameter_count": 11_181_642,
        "metric": "top1_accuracy",
        "best_epoch": best_epoch,
        "best_n_correct": best_record["n_correct"],
        "best_n_total": best_record["n_total"],
        "best_accuracy": best_record["top1_accuracy"],
        "final_epoch": final_record["epoch"],
        "final_n_correct": final_record["n_correct"],
        "final_n_total": final_record["n_total"],
        "final_accuracy": final_record["top1_accuracy"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_hash": EXPECTED_CONFIG_HASH,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_manifest_hash": EXPECTED_MANIFEST_HASH,
    }
    for key, expected in expected_adjudication.items():
        _require(adjudication[key] == expected, f"adjudication {key} disagrees with committed evidence")

    _require(
        adjudication["training_command"]
        == ".venv/bin/python tools/train_reference_classifier.py --config "
        "configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --full-run",
        "training command differs from the worklog",
    )
    _require(resolved.get("dataset") == adjudication["dataset"], "dataset differs across config and adjudication")
    _require(resolved.get("dataset_version") == adjudication["dataset_version"], "dataset hash differs across artifacts")
    _require(
        resolved.get("split_manifest_hash") == adjudication["split_manifest_hash"],
        "manifest hash differs across artifacts",
    )
    _require(resolved.get("architecture") == adjudication["architecture"], "architecture differs across artifacts")
    _require(
        resolved.get("classifier_variant") == adjudication["classifier_variant"],
        "classifier variant differs across artifacts",
    )
    _require(resolved.get("train_seed") == adjudication["train_seed"], "train seed differs across artifacts")
    _require(classifier_parameters.get("weights") is None, "configuration does not record null pretrained weights")
    _require(
        classifier_parameters.get("pretrained_weights_permitted") is False,
        "configuration permits pretrained weights",
    )

    _require(summary.get("config_hash") == computed_config_hash, "summary config hash disagrees")
    _require(summary.get("run_complete") is True, "summary does not record a complete run")
    _require(summary.get("g1_eligible") is True, "summary does not record G-1 eligibility")
    _require(summary.get("best_epoch") == best_epoch, "summary best epoch disagrees")
    _require(summary.get("best_validation_top1") == EXPECTED_ACCURACY, "summary best accuracy disagrees")
    _require(summary.get("final_epoch") == final_record["epoch"], "summary final epoch disagrees")
    _require(
        summary.get("final_checkpoint_id") == EXPECTED_CHECKPOINT_SHA256,
        "summary checkpoint identity disagrees",
    )

    for key in ("best_checkpoint", "final_checkpoint"):
        value = _relative_posix(checkpoint_metadata.get(key), label=f"best_checkpoint.json {key}")
        _require(value == adjudication_path, f"best_checkpoint.json {key} path disagrees")
    for key in ("best_checkpoint_id", "final_checkpoint_id"):
        _require(
            checkpoint_metadata.get(key) == EXPECTED_CHECKPOINT_SHA256,
            f"best_checkpoint.json {key} disagrees",
        )
    _require(checkpoint_metadata.get("best_epoch") == best_epoch, "checkpoint metadata best epoch disagrees")
    _require(checkpoint_metadata.get("final_epoch") == final_record["epoch"], "checkpoint metadata final epoch disagrees")
    _require(
        checkpoint_metadata.get("best_validation_top1") == EXPECTED_ACCURACY,
        "checkpoint metadata accuracy disagrees",
    )

    external = adjudication["checkpoint_external_artifact"]
    _require(isinstance(external, dict), "checkpoint external artifact must be an object")
    external_required = {"provider", "repository", "release_tag", "asset_name", "sha256", "bytes"}
    external_missing = external_required - set(external)
    _require(not external_missing, f"external artifact identity missing: {sorted(external_missing)}")
    _require(external["provider"] == "github_release", "external artifact provider differs")
    _require(external["repository"] == "nxck2005/capstone", "external artifact repository differs")
    _require(
        external["release_tag"] == "g1-reference-classifier-2026-07-29",
        "external artifact release tag differs",
    )
    _require(
        external["asset_name"]
        == "reference-classifier-imagenette160-epoch99-9c37362347a0203597d6e8e9d9a58fde.pt",
        "external artifact asset name differs",
    )
    _require(external["sha256"] == EXPECTED_CHECKPOINT_SHA256, "external artifact hash disagrees")
    _require(
        _is_int(external["bytes"]) and external["bytes"] == EXPECTED_CHECKPOINT_BYTES,
        "external artifact byte size disagrees",
    )
    if "verified_download_url" in external:
        _require(
            external["verified_download_url"]
            == "https://github.com/nxck2005/capstone/releases/download/"
            "g1-reference-classifier-2026-07-29/"
            "reference-classifier-imagenette160-epoch99-9c37362347a0203597d6e8e9d9a58fde.pt",
            "verified external download URL differs",
        )

    local_checkpoint = repo / adjudication_path
    local_verified = local_checkpoint.is_file()
    if local_verified:
        _require(local_checkpoint.stat().st_size == EXPECTED_CHECKPOINT_BYTES, "local checkpoint size disagrees")
        _require(_sha256(local_checkpoint) == EXPECTED_CHECKPOINT_SHA256, "local checkpoint hash disagrees")

    environment = adjudication["environment"]
    _require(isinstance(environment, dict), "environment record must be an object")
    expected_environment = {
        "python_version": "3.14.6",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "driver_version": "592.82",
        "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "lock_file": "requirements.lock",
        "lock_file_sha256": "b87c3e45089a29caf8178879ea4bf069fcd75f16d507a1fa04c1752c6633cd5f",
    }
    _require(environment == expected_environment, "adjudicated environment differs from recorded evidence")
    _require(
        environment_parameters.get("python_version") == environment["python_version"],
        "environment Python version differs from configuration",
    )
    _require(
        environment_parameters.get("torch") == environment["torch_version"],
        "environment Torch version differs from configuration",
    )
    lock_path = repo / environment["lock_file"]
    _require(lock_path.is_file(), "recorded environment lockfile is absent")
    _require(_sha256(lock_path) == environment["lock_file_sha256"], "environment lockfile hash disagrees")

    lineage = adjudication["lineage"]
    _require(
        lineage
        == {
            "training_source_is_ancestor_of_evidence_parent": True,
            "hardening_commit_identity": "recorded_by_git_history_not_self_referenced",
        },
        "lineage declaration differs",
    )
    _verify_git_lineage(repo)

    isolation = adjudication["test_isolation"]
    _require(
        isolation
        == {
            "test_split_sealed": True,
            "model_facing_test_access": False,
            "test_inference": False,
            "test_accuracy_computation": False,
            "published_test_provenance_scan_only": True,
            "provenance_scan_decoder_calls": 0,
            "provenance_scan_canonicalization_calls": 0,
        },
        "test-isolation record permits or claims test evaluation/access",
    )
    for label, field, required in (
        ("preflight", "preflight_checks", REQUIRED_PREFLIGHT_CHECKS),
        ("final", "final_checks", REQUIRED_FINAL_CHECKS),
    ):
        checks = adjudication[field]
        _require(isinstance(checks, dict) and checks, f"{label} checks are absent")
        missing = required - set(checks)
        unexpected = set(checks) - required
        _require(
            not missing and not unexpected,
            f"{label} check keys differ: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )
        _require(
            all(value == "pass" for value in checks.values()),
            f"{label} checks contain a non-pass result",
        )

    return {
        "gate": "G-1",
        "verdict": "PASS",
        "epochs": len(records),
        "best": f"{EXPECTED_CORRECT}/{EXPECTED_TOTAL}",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "local_checkpoint_verified": local_verified,
    }


def main() -> int:
    try:
        result = verify()
    except VerificationError as exc:
        print(f"G-1 adjudication verification FAILED: {exc}", file=sys.stderr)
        return 1
    local = "verified" if result["local_checkpoint_verified"] else "absent (external identity retained)"
    print(
        "G-1 adjudication verification PASS: "
        f"{result['epochs']} epochs, best={result['best']}, local checkpoint={local}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
