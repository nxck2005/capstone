"""Inference-only loader for the adjudicated epoch-99 G-1 classifier."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from config.params import REPO_ROOT, get
from config.run_config import RunConfig, config_hash
from models.reference_classifier import ReferenceClassifier, build_reference_classifier

EXPECTED_CHECKPOINT_SHA256 = (
    "9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002"
)
EXPECTED_CONFIG_HASH = (
    "a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1"
)
EXPECTED_CHECKPOINT_BYTES = 92_121_803
_CHECKPOINT_FIELDS = {
    "checkpoint_schema_version",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "completed_epoch",
    "next_epoch",
    "best_validation_top1",
    "best_epoch",
    "resolved_run_config",
    "config_hash",
    "dataset",
    "dataset_version",
    "split_manifest_hash",
    "classifier_variant",
    "architecture",
    "train_seed",
    "model_total_parameter_count",
    "model_trainable_parameter_count",
    "training_history",
    "validation_history",
    "checkpoint_history",
    "execution_mode",
    "smoke_steps",
    "smoke_val_batches",
    "full_run_requested",
    "run_complete",
    "g1_eligible",
    "lineage_g1_eligible",
}


class FrozenClassifierError(ValueError):
    """Fail-closed frozen-checkpoint validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenClassifierError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),  # literal-ok: one-MiB streaming I/O chunk
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenClassifierError(f"cannot read {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _verify_g1_adjudication(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools/verify_g1_adjudication.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        result.returncode == 0,
        "G-1 adjudication verification failed: "
        f"{result.stderr.strip() or result.stdout.strip()}",
    )


def _download_checkpoint(
    *,
    destination: Path,
    url: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            while chunk := response.read(
                1024 * 1024  # literal-ok: one-MiB streaming I/O chunk
            ):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _require(
            temporary.stat().st_size == expected_bytes,
            "downloaded checkpoint byte length disagrees",
        )
        _require(
            _sha256(temporary) == expected_sha256,
            "downloaded checkpoint SHA-256 disagrees",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_checkpoint(
    *,
    repo_root: Path,
    adjudication: Mapping[str, Any],
    checkpoint_path: Path | None,
    allow_download: bool,
) -> Path:
    recorded = repo_root / str(adjudication["checkpoint_repository_path"])
    selected = recorded if checkpoint_path is None else checkpoint_path.resolve()
    if selected.is_file():
        return selected
    _require(
        checkpoint_path is None,
        "an explicitly selected checkpoint is absent",
    )
    _require(allow_download, "frozen checkpoint is absent and download is disabled")
    external = adjudication.get("checkpoint_external_artifact")
    _require(isinstance(external, Mapping), "external checkpoint identity is absent")
    required = {"release_tag", "asset_name", "bytes", "sha256", "repository"}
    _require(set(external) >= required, "external checkpoint identity is incomplete")
    _require(
        external["repository"] == "nxck2005/capstone",
        "unexpected checkpoint release repository",
    )
    url = (
        "https://github.com/nxck2005/capstone/releases/download/"
        f"{external['release_tag']}/{external['asset_name']}"
    )
    _download_checkpoint(
        destination=recorded,
        url=url,
        expected_bytes=int(external["bytes"]),
        expected_sha256=str(external["sha256"]),
    )
    return recorded


def _validate_history(
    payload: Mapping[str, Any],
    final_epoch: int,
    adjudication: Mapping[str, Any],
) -> None:
    training = payload["training_history"]
    validation = payload["validation_history"]
    checkpoints = payload["checkpoint_history"]
    _require(
        isinstance(training, list)
        and isinstance(validation, list)
        and isinstance(checkpoints, list),
        "checkpoint histories must be lists",
    )
    expected_epochs = list(range(final_epoch + 1))
    _require(
        [item.get("epoch") for item in training if isinstance(item, Mapping)]
        == expected_epochs
        and len(training) == len(expected_epochs),
        "incomplete full-run training lineage",
    )
    _require(
        [item.get("epoch") for item in validation if isinstance(item, Mapping)]
        == expected_epochs
        and len(validation) == len(expected_epochs),
        "incomplete full-run validation lineage",
    )
    final_validation = validation[-1]
    _require(
        final_validation.get("n_correct") == adjudication["final_n_correct"]
        and final_validation.get("n_total") == adjudication["final_n_total"]
        and final_validation.get("top1_accuracy") == adjudication["final_accuracy"],
        "checkpoint final validation result is not the adjudicated G-1 result",
    )
    _require(
        len(checkpoints) == final_epoch
        and [item.get("epoch") for item in checkpoints if isinstance(item, Mapping)]
        == list(range(final_epoch)),
        "incomplete checkpoint lineage",
    )


def _validate_payload(
    payload: object,
    *,
    adjudication: Mapping[str, Any],
    committed_config: Mapping[str, Any],
) -> ReferenceClassifier:
    _require(isinstance(payload, Mapping), "checkpoint payload must be an object")
    missing = _CHECKPOINT_FIELDS - set(payload)
    unexpected = set(payload) - _CHECKPOINT_FIELDS
    _require(
        not missing and not unexpected,
        f"checkpoint schema differs: missing={sorted(missing)}, "
        f"unexpected={sorted(unexpected)}",
    )
    try:
        checkpoint_config = RunConfig.from_dict(payload["resolved_run_config"])
        expected_config = RunConfig.from_dict(committed_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenClassifierError(f"checkpoint configuration is invalid: {exc}") from None
    _require(
        checkpoint_config.to_dict() == expected_config.to_dict(),
        "checkpoint resolved configuration disagrees",
    )
    _require(
        config_hash(checkpoint_config)
        == payload["config_hash"]
        == adjudication["config_hash"]
        == EXPECTED_CONFIG_HASH,
        "checkpoint config hash disagrees",
    )
    expected = {
        "checkpoint_schema_version": get(
            "reference_classifier.checkpoint_schema_version"
        ),
        "dataset": adjudication["dataset"],
        "dataset_version": adjudication["dataset_version"],
        "split_manifest_hash": adjudication["split_manifest_hash"],
        "classifier_variant": adjudication["classifier_variant"],
        "architecture": adjudication["architecture"],
        "train_seed": adjudication["train_seed"],
        "model_total_parameter_count": adjudication["parameter_count"],
        "model_trainable_parameter_count": adjudication["parameter_count"],
        "completed_epoch": adjudication["final_epoch"],
        "next_epoch": adjudication["final_epoch"] + 1,
        "best_epoch": adjudication["best_epoch"],
        "best_validation_top1": adjudication["best_accuracy"],
        "execution_mode": "full",
        "smoke_steps": None,
        "smoke_val_batches": None,
        "full_run_requested": True,
        "run_complete": True,
        "g1_eligible": True,
        "lineage_g1_eligible": True,
    }
    for key, value in expected.items():
        _require(payload[key] == value, f"checkpoint {key} disagrees")
    _require(
        payload["scheduler_state"] == {"completed_epoch": adjudication["final_epoch"]},
        "checkpoint scheduler completion disagrees",
    )
    _require(isinstance(payload["optimizer_state"], Mapping), "optimizer state is invalid")
    _validate_history(
        payload,
        int(adjudication["final_epoch"]),
        adjudication,
    )

    model = build_reference_classifier(
        str(payload["dataset"]),
        architecture=str(payload["architecture"]),
        train_seed=int(payload["train_seed"]),
    )
    _require(
        model.total_parameter_count == payload["model_total_parameter_count"],
        "constructed classifier parameter count disagrees",
    )
    model_state = payload["model_state"]
    _require(isinstance(model_state, Mapping), "checkpoint model_state is invalid")
    expected_keys = set(model.state_dict())
    missing_state = expected_keys - set(model_state)
    unexpected_state = set(model_state) - expected_keys
    _require(
        not missing_state and not unexpected_state,
        f"checkpoint model state differs: missing={sorted(missing_state)}, "
        f"unexpected={sorted(unexpected_state)}",
    )
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise FrozenClassifierError(f"checkpoint model state is invalid: {exc}") from None
    return model


def load_frozen_reference_classifier(
    device: torch.device | str,
    *,
    repo_root: Path = REPO_ROOT,
    checkpoint_path: Path | None = None,
    allow_download: bool = True,
) -> ReferenceClassifier:
    """Verify, CPU-load, validate, freeze, and return the exact G-1 classifier."""

    repo_root = repo_root.resolve()
    _verify_g1_adjudication(repo_root)
    adjudication = _json(repo_root / "results/reference_classifier/g1_adjudication.json")
    committed_config = _json(repo_root / "results/reference_classifier/resolved_config.json")
    selected = _resolve_checkpoint(
        repo_root=repo_root,
        adjudication=adjudication,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
    )
    _require(
        selected.stat().st_size
        == adjudication["checkpoint_external_artifact"]["bytes"]
        == EXPECTED_CHECKPOINT_BYTES,
        "checkpoint byte length disagrees",
    )
    _require(
        _sha256(selected)
        == adjudication["checkpoint_sha256"]
        == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint SHA-256 disagrees",
    )
    try:
        payload = torch.load(selected, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FrozenClassifierError(f"cannot CPU-load frozen checkpoint: {exc}") from None
    model = _validate_payload(
        payload,
        adjudication=adjudication,
        committed_config=committed_config,
    )
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    _require(not model.training, "frozen classifier did not remain in evaluation mode")
    _require(
        all(not parameter.requires_grad for parameter in model.parameters()),
        "frozen classifier retains trainable parameters",
    )
    return model
