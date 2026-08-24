"""Dedicated G8_F/F2 BR-12 artifact-classifier dataset and trainer (AM-89).

This module has no test-split, pass-two, codec, or learned-system entry point.
Production training consumes one logical item per materialized frozen F1
assignment row and initializes a distinct ResNet18 descendant from exact G1.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode

from baseline.g8_f_closeout import (
    COMPLETION_PATH,
    INFEASIBLE,
    MANIFEST_FIELDS,
    MANIFEST_PATH,
    MATERIALIZED,
    OMISSION,
    RUNTIME_PATH,
    verify_closeout,
)
from artifacts.rng import keyed_generator
from baseline.g8_f_materializer import load_frozen_assignments
from config.params import REPO_ROOT, get
from data.classifier import EpochPermutationSampler, ValidationClassifierDataset
from models.frozen_reference_classifier import (
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_CHECKPOINT_SHA256,
    load_frozen_reference_classifier,
)
from training.reference_classifier import ValidationResult, atomic_torch_save, validate

F2_SCHEMA_VERSION = 1
F2_VARIANT = "artifact_finetuned"
F2_SCOPE = "G8_F_F2_BR12_TRAINING_ONLY"
F1_COMPLETION_ID = "g8ff1completion-b5bb834a1767f639406e5589022e813a624a4f8ccd9ad4885c455c10fce24412"
F1_COMPLETION_SHA256 = "d4f9d44a01dbf53de96fb9126364d651ca999b35117b7d51c55f76f1a13d888b"
F1_CORPUS_ID = "g8fcorpus-adeae50779a45e9e856af3ff47e84671b237b344867a562978599170912135c2"
F1_MANIFEST_SHA256 = "792cce92bd8a72f99b7ddee58511d1b5b7e908a4d0cd4178bbb08b9e1ba2d144"
G1_ADJUDICATION_ID = "g1adjudication-a1d6a59ce45fb6006271f60df9487a0c07c65fba562a1987cc51526d70a53efd"
G1_ADJUDICATION_SHA256 = "a1d6a59ce45fb6006271f60df9487a0c07c65fba562a1987cc51526d70a53efd"
G1_CHECKPOINT_ID = EXPECTED_CHECKPOINT_SHA256
G1_CHECKPOINT_SHA256 = EXPECTED_CHECKPOINT_SHA256
G1_CHECKPOINT_BYTES = EXPECTED_CHECKPOINT_BYTES
EXPECTED_ASSIGNMENTS = 50_814
EXPECTED_MATERIALIZED = 44_039
EXPECTED_OMISSIONS = 6_775
EXPECTED_UNIQUE_RECONSTRUCTIONS = 42_932
EXPECTED_EPOCHS = 20  # literal-ok: AM-89 exact F2 recipe
EXPECTED_BATCH_SIZE = 128  # literal-ok: AM-89 exact F2 recipe
EXPECTED_STEPS_PER_EPOCH = 345
EXPECTED_OPTIMIZER_STEPS = 6_900
RECONSTRUCTION_BYTES = 160 * 160 * 3  # literal-ok: Imagenette canonical RGB shape


class F2Hold(RuntimeError):
    """A fail-closed F2 input, lineage, or resume violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F2Hold(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _durable_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    """Publish one epoch checkpoint atomically and durably before returning."""

    checkpoint_id = atomic_torch_save(payload, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return checkpoint_id


def _parse_manifest(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise F2Hold("F1 corpus manifest is not ASCII") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, "F1 corpus manifest schema differs")
    rows = list(reader)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _require(output.getvalue().encode("ascii") == raw, "F1 corpus manifest is not canonical CSV")
    return rows


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError):
        raise F2Hold(f"F1 row {field} is not an integer") from None
    return value


@dataclass(frozen=True)
class F2Example:
    f1_ordinal: int
    assignment_id: str
    stable_sample_id: str
    class_label: int
    quality_id: str
    request_id: str
    result_id: str
    reconstruction_path: str
    reconstruction_sha256: str
    reconstruction_bytes: int
    f1_corpus_id: str
    f1_corpus_sha256: str


@dataclass(frozen=True)
class F2DatasetSummary:
    assignment_rows: int
    materialized_rows: int
    omitted_rows: int
    unexpected_rows: int
    distinct_materialized_assignments: int
    unique_reconstruction_sha256: int
    validation_ids: int
    test_ids: int


def _safe_object(root: Path, relative: str, expected_sha: str, expected_bytes: int) -> bytes:
    _require(root.is_dir() and not root.is_symlink(), "F1 runtime root is missing or symlinked")
    candidate_rel = Path(relative)
    _require(not candidate_rel.is_absolute() and ".." not in candidate_rel.parts, "F1 reconstruction path traverses runtime")
    _require(relative == f"objects/reconstruction/{expected_sha}.rgb", "F1 reconstruction path is not content-addressed")
    cursor = root
    for part in candidate_rel.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "F1 reconstruction path contains a symlink")
    _require(cursor.is_file(), "F1 reconstruction object is missing or non-regular")
    raw = cursor.read_bytes()
    _require(len(raw) == expected_bytes == RECONSTRUCTION_BYTES, "F1 reconstruction byte length differs")
    _require(sha256_bytes(raw) == expected_sha, "F1 reconstruction SHA-256 differs")
    return raw


def _f2_crop_box(image_hw: tuple[int, int], rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Exact AM-78/Torchvision-0.28 proposal and fallback geometry."""

    image_height, image_width = image_hw
    area = image_height * image_width
    scale_min, scale_max = (float(value) for value in get("preprocessing.train_crop_scale"))
    ratio_min, ratio_max = (float(value) for value in get("preprocessing.train_crop_ratio"))
    _require(0 < scale_min <= scale_max and 0 < ratio_min <= ratio_max, "F2 crop bounds differ")
    log_ratio_min, log_ratio_max = math.log(ratio_min), math.log(ratio_max)
    for _ in range(10):  # literal-ok: frozen Torchvision RandomResizedCrop proposal count
        target_area = area * rng.uniform(scale_min, scale_max)
        aspect_ratio = math.exp(rng.uniform(log_ratio_min, log_ratio_max))
        width = int(round(math.sqrt(target_area * aspect_ratio)))
        height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < width <= image_width and 0 < height <= image_height:
            top = int(rng.integers(0, image_height - height + 1))
            left = int(rng.integers(0, image_width - width + 1))
            return top, left, height, width
    input_ratio = image_width / image_height
    if input_ratio < ratio_min:
        width, height = image_width, int(round(image_width / ratio_min))
    elif input_ratio > ratio_max:
        height, width = image_height, int(round(image_height * ratio_max))
    else:
        width, height = image_width, image_height
    return (image_height - height) // 2, (image_width - width) // 2, height, width


def artifact_training_input(
    image: np.ndarray,
    *,
    stable_id: str,
    train_seed: int,
    epoch: int,
) -> torch.Tensor:
    """Apply the frozen keyed clean transform to exact reconstruction pixels."""

    pixels = np.asarray(image)
    _require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[-1] == 3, "F2 reconstruction pixels are not uint8 RGB")
    pixels = np.ascontiguousarray(pixels)
    identity = {"stable_sample_id": stable_id, "train_seed": train_seed, "epoch": epoch}
    rng = keyed_generator("augmentation", identity)
    _require(get("preprocessing.train_crop") == "random_resized_crop", "F2 crop semantics differ")
    top, left, height, width = _f2_crop_box(pixels.shape[:2], rng)
    try:
        interpolation = InterpolationMode(get("preprocessing.resize_interpolation"))
    except ValueError:
        raise F2Hold("F2 interpolation semantics differ") from None
    antialias = get("preprocessing.antialias")
    _require(isinstance(antialias, bool), "F2 antialias semantics differ")
    augmented = vision_f.resized_crop(
        Image.fromarray(pixels),
        top,
        left,
        height,
        width,
        list(pixels.shape[:2]),
        interpolation=interpolation,
        antialias=antialias,
    )
    if rng.random() < float(get("preprocessing.train_hflip_p")):
        augmented = vision_f.hflip(augmented)
    output = np.ascontiguousarray(np.asarray(augmented.convert("RGB"), dtype=np.uint8))
    return torch.from_numpy(output.copy()).permute(2, 0, 1).to(dtype=torch.float32).div(255)  # literal-ok: exact 8-bit unit-interval conversion


class F2ArtifactDataset(Dataset[tuple[torch.Tensor, int]]):
    """An epoch-specific deterministic view over materialized assignment rows."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        runtime_root: Path,
        assignment_authority: Sequence[Mapping[str, Any]],
        epoch: int,
        train_seed: int,
        corpus_id: str,
        corpus_sha256: str,
        expected_assignment_rows: int,
        expected_materialized_rows: int,
        expected_omitted_rows: int,
        expected_corpus_id: str | None = None,
        expected_corpus_sha256: str | None = None,
        authenticate_objects: bool = True,
    ) -> None:
        _require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0, "F2 epoch is invalid")
        _require(isinstance(train_seed, int) and not isinstance(train_seed, bool), "F2 train seed is invalid")
        _require(len(rows) == len(assignment_authority) == expected_assignment_rows, "F2 assignment authority length differs")
        _require(corpus_id == (corpus_id if expected_corpus_id is None else expected_corpus_id), "F2 corpus ID differs")
        _require(corpus_sha256 == (corpus_sha256 if expected_corpus_sha256 is None else expected_corpus_sha256), "F2 corpus SHA-256 differs")
        self.runtime_root = Path(runtime_root)
        self.epoch = epoch
        self.train_seed = train_seed
        self.corpus_id = corpus_id
        self.corpus_sha256 = corpus_sha256
        examples: list[F2Example] = []
        omitted = 0
        assignment_ids: set[str] = set()
        reconstruction_shas: set[str] = set()
        for ordinal, (row, authority) in enumerate(zip(rows, assignment_authority, strict=True)):
            _require(_integer(row, "ordinal") == ordinal, "F2 manifest ordinal is missing, duplicate, or reordered")
            expected = {
                "assignment_id": str(authority["assignment_id"]),
                "stable_sample_id": str(authority["stable_sample_id"]),
                "class_label": str(authority["class_label"]),
                "quality_id": str(authority["quality_id"]),
            }
            _require(all(row[key] == value for key, value in expected.items()), "F2 manifest row differs from frozen assignment authority")
            _require(str(authority.get("split", "train")) == "train", "validation or test identity entered F2 training authority")
            _require(row["assignment_id"] not in assignment_ids, "duplicate F2 assignment row")
            assignment_ids.add(row["assignment_id"])
            outcome = row["outcome"]
            if outcome == INFEASIBLE:
                omitted += 1
                _require(row["omission_state"] == OMISSION, "F2 omission semantics differ")
                _require(not any(row[name] for name in ("reconstruction_path", "reconstruction_sha256", "reconstruction_bytes")), "F2 omission was presented as materialized")
                continue
            _require(outcome == MATERIALIZED, "F2 manifest contains unexpected outcome")
            _require(not row["omission_state"], "materialized F2 row carries omission state")
            reconstruction_bytes = _integer(row, "reconstruction_bytes")
            example = F2Example(
                f1_ordinal=ordinal,
                assignment_id=row["assignment_id"],
                stable_sample_id=row["stable_sample_id"],
                class_label=_integer(row, "class_label"),
                quality_id=row["quality_id"],
                request_id=row["request_id"],
                result_id=row["result_id"],
                reconstruction_path=row["reconstruction_path"],
                reconstruction_sha256=row["reconstruction_sha256"],
                reconstruction_bytes=reconstruction_bytes,
                f1_corpus_id=corpus_id,
                f1_corpus_sha256=corpus_sha256,
            )
            _require(example.reconstruction_sha256, "materialized F2 row has no reconstruction identity")
            if authenticate_objects:
                _safe_object(self.runtime_root, example.reconstruction_path, example.reconstruction_sha256, example.reconstruction_bytes)
            reconstruction_shas.add(example.reconstruction_sha256)
            examples.append(example)
        _require(len(examples) == expected_materialized_rows, "F2 materialized-row count differs")
        _require(omitted == expected_omitted_rows, "F2 omission count differs")
        self._examples = tuple(examples)
        self.summary = F2DatasetSummary(
            assignment_rows=len(rows),
            materialized_rows=len(examples),
            omitted_rows=omitted,
            unexpected_rows=len(rows) - len(examples) - omitted,
            distinct_materialized_assignments=len({item.assignment_id for item in examples}),
            unique_reconstruction_sha256=len(reconstruction_shas),
            validation_ids=0,
            test_ids=0,
        )

    @classmethod
    def production(
        cls,
        *,
        epoch: int,
        runtime_root: Path = RUNTIME_PATH,
        repo_root: Path = REPO_ROOT,
        authenticate_objects: bool = True,
    ) -> "F2ArtifactDataset":
        completion_path = repo_root / COMPLETION_PATH.relative_to(REPO_ROOT)
        manifest_path = repo_root / MANIFEST_PATH.relative_to(REPO_ROOT)
        _require(sha256_bytes(completion_path.read_bytes()) == F1_COMPLETION_SHA256, "F1 completion file SHA-256 differs")
        _require(sha256_bytes(manifest_path.read_bytes()) == F1_MANIFEST_SHA256, "F1 corpus manifest SHA-256 differs")
        completion = verify_closeout(completion_path, manifest_path)
        _require(completion["completion_id"] == F1_COMPLETION_ID and completion["corpus_id"] == F1_CORPUS_ID, "F1 completion/corpus identity differs")
        rows = _parse_manifest(manifest_path.read_bytes())
        assignments = load_frozen_assignments()
        authority = [
            {
                "assignment_id": item.assignment_id,
                "stable_sample_id": item.stable_sample_id,
                "class_label": item.label,
                "quality_id": item.quality_id,
                "split": "train",
            }
            for item in assignments
        ]
        dataset = cls(
            rows,
            runtime_root=runtime_root,
            assignment_authority=authority,
            epoch=epoch,
            train_seed=int(f2_recipe()["train_seed"]),
            corpus_id=F1_CORPUS_ID,
            corpus_sha256=F1_MANIFEST_SHA256,
            expected_corpus_id=F1_CORPUS_ID,
            expected_corpus_sha256=F1_MANIFEST_SHA256,
            expected_assignment_rows=EXPECTED_ASSIGNMENTS,
            expected_materialized_rows=EXPECTED_MATERIALIZED,
            expected_omitted_rows=EXPECTED_OMISSIONS,
            authenticate_objects=authenticate_objects,
        )
        _require(dataset.summary.unique_reconstruction_sha256 == EXPECTED_UNIQUE_RECONSTRUCTIONS, "F2 unique reconstruction count differs")
        return dataset

    def for_epoch(self, epoch: int) -> "F2ArtifactDataset":
        clone = object.__new__(type(self))
        clone.runtime_root = self.runtime_root
        clone.epoch = epoch
        clone.train_seed = self.train_seed
        clone.corpus_id = self.corpus_id
        clone.corpus_sha256 = self.corpus_sha256
        clone._examples = self._examples
        clone.summary = self.summary
        return clone

    def __len__(self) -> int:
        return len(self._examples)

    def trace(self, index: int) -> F2Example:
        return self._examples[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self._examples[index]
        raw = _safe_object(self.runtime_root, item.reconstruction_path, item.reconstruction_sha256, item.reconstruction_bytes)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(160, 160, 3)  # literal-ok: Imagenette canonical RGB shape
        tensor = artifact_training_input(
            pixels,
            stable_id=item.stable_sample_id,
            train_seed=self.train_seed,
            epoch=self.epoch,
        )
        return tensor, item.class_label


def f2_recipe() -> dict[str, Any]:
    recipe = dict(get("reference_classifier.artifact_finetune_recipe"))
    required = {
        "classifier_variant": F2_VARIANT,
        "initialization": "exact_frozen_g1_best_clean_checkpoint",
        "parent_variant": "clean",
        "train_seed": 0,
        "optimizer": "sgd_momentum",
        "optimizer_implementation": "torch.optim.SGD",
        "loss": "cross_entropy",
        "lr": 0.01,  # literal-ok: AM-89 exact F2 recipe
        "momentum": 0.9,  # literal-ok: AM-89 exact F2 recipe
        "nesterov": False,
        "weight_decay": 0.0005,  # literal-ok: AM-89 exact F2 recipe
        "lr_schedule": "cosine",
        "lr_warmup_epochs": 5,  # literal-ok: AM-89 exact F2 recipe
        "lr_warmup_schedule": "linear",
        "lr_warmup_start_factor": 0.1,  # literal-ok: AM-89 exact F2 recipe
        "lr_min": 0.0,
        "scheduler_step_unit": "epoch_start",
        "scheduler_epoch_indexing": "zero_based",
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "label_smoothing": 0.1,  # literal-ok: AM-89 exact F2 recipe
        "augmentation": ["random_resized_crop", "horizontal_flip"],
        "augmentation_input": "exact_authenticated_f1_reconstruction_pixels",
        "batch_order": "keyed_philox_permutation_per_epoch_over_materialized_assignment_rows",
        "drop_last": False,
        "mixed_precision": False,
        "dataloader_workers": 4,  # literal-ok: AM-89 exact F2 recipe
        "pin_memory": True,
        "validation_split": "imagenette160_validation_clean_canonical_view",
        "validation_every_epochs": 1,
        "checkpoint_every_epochs": 1,
        "checkpoint_metric": "validation_top1_accuracy",
        "checkpoint_mode": "max",
        "checkpoint_tie_break": "earliest_epoch",
        "resume_unit": "authenticated_completed_epoch",
        "corrupt_latest_checkpoint_policy": "hold_no_older_fallback",
        "incomplete_epoch_policy": "replay_from_latest_authenticated_completed_epoch",
        "test_access": "prohibited",
    }
    _require(recipe == required, "F2 artifact-finetune recipe differs from AM-89 exact contract")
    return recipe


def f2_recipe_sha256() -> str:
    return sha256_bytes(canonical_json(f2_recipe()))


def learning_rate_for_epoch(epoch: int) -> float:
    recipe = f2_recipe()
    total = int(recipe["epochs"])
    warmup = int(recipe["lr_warmup_epochs"])
    base = float(recipe["lr"])
    start = float(recipe["lr_warmup_start_factor"])
    minimum = float(recipe["lr_min"])
    _require(0 <= epoch < total, "F2 epoch is outside schedule")
    if epoch < warmup:
        return base * (start + (1 - start) * epoch / max(warmup - 1, 1))
    j = epoch - warmup
    duration = total - warmup
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * j / max(duration - 1, 1)))  # literal-ok: AM-78 cosine formula


class F2Trainer:
    """Epoch-atomic production trainer with fail-closed exact-next resume."""

    def __init__(
        self,
        *,
        authorization: Mapping[str, Any],
        authorization_sha256: str,
        runtime_root: Path,
        dataset: F2ArtifactDataset,
        device: str,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.authorization = dict(authorization)
        self.authorization_sha256 = authorization_sha256
        self.runtime_root = Path(runtime_root)
        self.dataset = dataset
        self.device = torch.device(device)
        self.recipe = f2_recipe()
        self.progress_callback = progress_callback
        model = load_frozen_reference_classifier(self.device, allow_download=False)
        model.requires_grad_(True)
        model.train()
        self.model = model
        self.optimizer = SGD(
            self.model.parameters(),
            lr=learning_rate_for_epoch(0),
            momentum=float(self.recipe["momentum"]),
            weight_decay=float(self.recipe["weight_decay"]),
            nesterov=bool(self.recipe["nesterov"]),
        )
        self.loss = nn.CrossEntropyLoss(label_smoothing=float(self.recipe["label_smoothing"]))
        self.completed_epoch = -1
        self.total_optimizer_steps = 0
        self.training_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.checkpoint_history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_validation_top1 = float("-inf")

    @property
    def checkpoints_dir(self) -> Path:
        return self.runtime_root / "checkpoints"

    @property
    def latest_path(self) -> Path:
        return self.runtime_root / "latest.json"

    def _lineage(self) -> dict[str, Any]:
        return {
            "scope": F2_SCOPE,
            "authorization_id": self.authorization["authorization_id"],
            "authorization_sha256": self.authorization_sha256,
            "source_commit": self.authorization["source_commit"],
            "execution_profile_id": self.authorization["execution_profile"]["execution_profile_id"],
            "device": self.authorization["execution_profile"]["device"],
            "f1_completion_id": F1_COMPLETION_ID,
            "f1_completion_sha256": F1_COMPLETION_SHA256,
            "f1_corpus_id": F1_CORPUS_ID,
            "f1_manifest_sha256": F1_MANIFEST_SHA256,
            "g1_parent_checkpoint_id": G1_CHECKPOINT_ID,
            "g1_parent_checkpoint_sha256": G1_CHECKPOINT_SHA256,
            "recipe_sha256": f2_recipe_sha256(),
            "train_seed": int(self.recipe["train_seed"]),
            "classifier_variant": F2_VARIANT,
        }

    def _emit_progress(self, value: Mapping[str, Any]) -> None:
        payload = {"schema_version": 1, "campaign": "G8_F/F2/BR-12", **dict(value)}
        atomic_bytes(self.runtime_root / "progress.json", canonical_json(payload))
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": F2_SCHEMA_VERSION,
            "artifact_role": "g8_f_f2_epoch_checkpoint",
            "lineage": self._lineage(),
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "total_optimizer_steps": self.total_optimizer_steps,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": {"completed_epoch": self.completed_epoch},
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "checkpoint_history": self.checkpoint_history,
            "best_epoch": self.best_epoch,
            "best_validation_top1": self.best_validation_top1,
            "expected_epochs": EXPECTED_EPOCHS,
            "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "protected_counters": {"f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0},
        }

    def save_checkpoint(self) -> dict[str, Any]:
        _require(self.completed_epoch >= 0, "cannot checkpoint before a completed F2 epoch")
        path = self.checkpoints_dir / f"epoch-{self.completed_epoch:02d}.pt"
        _require(not path.exists(), "F2 completed-epoch checkpoint already exists")
        checkpoint_id = _durable_torch_save(self._checkpoint_payload(), path)
        record = {"completed_epoch": self.completed_epoch, "path": str(path.relative_to(self.runtime_root)), "checkpoint_id": checkpoint_id, "bytes": path.stat().st_size}
        pointer = {"schema_version": 1, **record, "next_epoch": self.completed_epoch + 1, "authorization_id": self.authorization["authorization_id"], "authorization_sha256": self.authorization_sha256}
        atomic_bytes(self.latest_path, canonical_json(pointer))
        self.checkpoint_history.append(record)
        return record

    def resume(self) -> None:
        checkpoint_paths = sorted(self.checkpoints_dir.glob("epoch-*.pt")) if self.checkpoints_dir.is_dir() else []
        indices: list[int] = []
        for candidate in checkpoint_paths:
            match = re.fullmatch(r"epoch-(\d{2})\.pt", candidate.name)
            _require(match is not None and candidate.is_file() and not candidate.is_symlink(), "F2 checkpoint namespace contains a foreign/unsafe file")
            indices.append(int(match.group(1)))
        _require(indices and indices == list(range(indices[-1] + 1)), "F2 completed checkpoints are not an exact contiguous prefix")
        completed = indices[-1]
        _require(completed < EXPECTED_EPOCHS, "F2 latest completed epoch is invalid")
        expected_rel = f"checkpoints/epoch-{completed:02d}.pt"
        path = self.runtime_root / expected_rel
        raw_sha = sha256_bytes(path.read_bytes())
        pointer: Mapping[str, Any] | None = None
        if self.latest_path.exists():
            _require(self.latest_path.is_file() and not self.latest_path.is_symlink(), "F2 latest checkpoint pointer is unsafe")
            try:
                parsed = json.loads(self.latest_path.read_bytes())
            except (OSError, json.JSONDecodeError):
                raise F2Hold("F2 latest checkpoint pointer is corrupt") from None
            _require(isinstance(parsed, Mapping), "F2 latest checkpoint pointer is corrupt")
            pointer = parsed
            _require(pointer.get("authorization_id") == self.authorization["authorization_id"] and pointer.get("authorization_sha256") == self.authorization_sha256, "F2 latest checkpoint authorization differs")
            pointed = pointer.get("completed_epoch")
            _require(isinstance(pointed, int) and 0 <= pointed <= completed, "F2 latest pointer epoch is invalid")
            pointed_rel = f"checkpoints/epoch-{pointed:02d}.pt"
            pointed_path = self.runtime_root / pointed_rel
            _require(
                pointer.get("path") == pointed_rel
                and pointer.get("next_epoch") == pointed + 1
                and pointed_path.is_file()
                and not pointed_path.is_symlink()
                and pointer.get("bytes") == pointed_path.stat().st_size
                and pointer.get("checkpoint_id") == sha256_bytes(pointed_path.read_bytes()),
                "F2 latest pointer bytes/path differ",
            )
        # A checkpoint newer than the pointer is the one permitted interruption
        # window: durable epoch publication succeeded before pointer publication.
        # It is authenticated below and becomes the exact latest completion;
        # no optimizer step from that epoch is replayed.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise F2Hold("F2 latest checkpoint cannot be loaded") from None
        required_payload = {
            "schema_version", "artifact_role", "lineage", "completed_epoch", "next_epoch",
            "total_optimizer_steps", "model_state", "optimizer_state", "scheduler_state",
            "training_history", "validation_history", "checkpoint_history", "best_epoch",
            "best_validation_top1", "expected_epochs", "expected_optimizer_steps",
            "protected_counters",
        }
        _require(isinstance(payload, Mapping) and set(payload) == required_payload, "F2 checkpoint schema differs")
        _require(payload.get("schema_version") == F2_SCHEMA_VERSION and payload.get("artifact_role") == "g8_f_f2_epoch_checkpoint", "F2 checkpoint role/version differs")
        _require(payload.get("lineage") == self._lineage(), "F2 checkpoint lineage differs")
        _require(payload.get("scheduler_state") == {"completed_epoch": completed}, "F2 checkpoint scheduler state differs")
        _require(payload.get("expected_epochs") == EXPECTED_EPOCHS and payload.get("expected_optimizer_steps") == EXPECTED_OPTIMIZER_STEPS, "F2 checkpoint expected arithmetic differs")
        _require(payload.get("protected_counters") == {"f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0}, "F2 checkpoint protected boundary differs")
        _require(payload.get("completed_epoch") == completed and payload.get("next_epoch") == completed + 1, "F2 checkpoint epoch differs")
        training = payload.get("training_history")
        validation = payload.get("validation_history")
        prior = payload.get("checkpoint_history")
        _require(isinstance(training, list) and [entry.get("epoch") for entry in training] == list(range(completed + 1)), "F2 training history is not exact completed prefix")
        _require(isinstance(validation, list) and [entry.get("epoch") for entry in validation] == list(range(completed + 1)), "F2 validation history is not exact completed prefix")
        _require(isinstance(prior, list) and [entry.get("completed_epoch") for entry in prior] == list(range(completed)), "F2 prior checkpoint history is not exact prefix")
        for record in prior:
            prior_path = self.runtime_root / str(record["path"])
            _require(prior_path.is_file() and not prior_path.is_symlink(), "F2 prior checkpoint is missing or unsafe")
            _require(prior_path.stat().st_size == record["bytes"] and sha256_bytes(prior_path.read_bytes()) == record["checkpoint_id"], "F2 prior checkpoint bytes differ")
        expected_steps = (completed + 1) * EXPECTED_STEPS_PER_EPOCH
        _require(payload.get("total_optimizer_steps") == expected_steps, "F2 checkpoint optimizer-step count differs")
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except (KeyError, RuntimeError, ValueError):
            raise F2Hold("F2 checkpoint model/optimizer state is invalid") from None
        self.completed_epoch = completed
        self.total_optimizer_steps = expected_steps
        self.training_history = list(training)
        self.validation_history = list(validation)
        current_record = {
            "completed_epoch": completed,
            "path": expected_rel,
            "checkpoint_id": raw_sha,
            "bytes": path.stat().st_size,
        }
        self.checkpoint_history = list(prior) + [current_record]
        self.best_epoch = payload.get("best_epoch")
        self.best_validation_top1 = float(payload.get("best_validation_top1"))
        if pointer is None or pointer.get("completed_epoch") != completed:
            atomic_bytes(self.latest_path, canonical_json({
                "schema_version": 1,
                **current_record,
                "next_epoch": completed + 1,
                "authorization_id": self.authorization["authorization_id"],
                "authorization_sha256": self.authorization_sha256,
            }))

    def _set_lr(self, epoch: int) -> float:
        value = learning_rate_for_epoch(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def run(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        start_epoch = self.completed_epoch + 1
        _require(start_epoch < EXPECTED_EPOCHS, "F2 run is already complete")
        for epoch in range(start_epoch, EXPECTED_EPOCHS):
            epoch_started = time.monotonic()
            view = self.dataset.for_epoch(epoch)
            sampler = EpochPermutationSampler(len(view), int(self.recipe["train_seed"]), epoch)
            loader = DataLoader(
                view,
                batch_size=EXPECTED_BATCH_SIZE,
                sampler=sampler,
                num_workers=int(self.recipe["dataloader_workers"]),
                drop_last=False,
                pin_memory=bool(self.recipe["pin_memory"]),
            )
            _require(len(loader) == EXPECTED_STEPS_PER_EPOCH, "F2 optimizer-step arithmetic differs")
            self.model.train()
            lr = self._set_lr(epoch)
            total_loss = 0.0
            total_examples = 0
            for step, (inputs, labels) in enumerate(loader, start=1):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs.to(self.device, non_blocking=True))
                loss = self.loss(logits, labels.to(self.device, non_blocking=True))
                loss.backward()
                self.optimizer.step()
                count = int(labels.numel())
                total_loss += float(loss.detach().item()) * count
                total_examples += count
                self.total_optimizer_steps += 1
                if step == 1 or step % 10 == 0 or step == EXPECTED_STEPS_PER_EPOCH:  # literal-ok: operational progress cadence
                    self._emit_progress({
                        "status": "RUNNING",
                        "completed_epoch": self.completed_epoch,
                        "current_epoch": epoch,
                        "step_in_epoch": step,
                        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
                        "total_optimizer_steps": self.total_optimizer_steps,
                        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
                        "latest_train_loss": float(loss.detach().item()),
                        "best_validation_top1": None if self.best_epoch is None else self.best_validation_top1,
                        "best_epoch": self.best_epoch,
                        "elapsed_epoch_seconds": time.monotonic() - epoch_started,
                    })
            _require(total_examples == EXPECTED_MATERIALIZED and self.total_optimizer_steps == (epoch + 1) * EXPECTED_STEPS_PER_EPOCH, "F2 completed epoch arithmetic differs")
            training = {"epoch": epoch, "lr": lr, "loss": total_loss / total_examples, "examples": total_examples, "steps": EXPECTED_STEPS_PER_EPOCH, "duration_seconds": time.monotonic() - epoch_started, "sample_order_sha256": sha256_bytes(canonical_json(list(sampler)))}
            validation_dataset = ValidationClassifierDataset("imagenette160")
            result: ValidationResult = validate(self.model, validation_dataset, batch_size=EXPECTED_BATCH_SIZE, device=self.device, num_workers=int(self.recipe["dataloader_workers"]))
            validation_record = {"epoch": epoch, "n_correct": result.n_correct, "n_total": result.n_total, "top1_accuracy": result.top1_accuracy, "role": "f2_checkpoint_selection_validation_not_f3_cached_sweep"}
            self.training_history.append(training)
            self.validation_history.append(validation_record)
            if result.top1_accuracy > self.best_validation_top1:
                self.best_validation_top1 = result.top1_accuracy
                self.best_epoch = epoch
            self.completed_epoch = epoch
            checkpoint = self.save_checkpoint()
            self._emit_progress({
                "status": "COMPLETED" if epoch == EXPECTED_EPOCHS - 1 else "RUNNING",
                "completed_epoch": epoch,
                "current_epoch": None if epoch == EXPECTED_EPOCHS - 1 else epoch + 1,
                "step_in_epoch": 0,
                "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
                "total_optimizer_steps": self.total_optimizer_steps,
                "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
                "latest_train_loss": training["loss"],
                "latest_validation_top1": result.top1_accuracy,
                "best_validation_top1": self.best_validation_top1,
                "best_epoch": self.best_epoch,
                "latest_checkpoint": checkpoint,
                "protected_counters": {"f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0},
            })
