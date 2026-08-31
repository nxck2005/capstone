"""Standalone W8 final trainer with fresh-init and authenticated resume.

W8 is not a renamed W7 campaign.  This module has its own roles, lineage,
checkpoint schema, and policy boundary.  A core trainer can only construct a
fresh keyed model; restoration is permitted only from an authenticated
checkpoint produced by the same W8 run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from artifacts.rng import keyed_standard_normal
from config.params import get
from config.run_config import RunConfig, config_hash as run_config_hash
from data.classifier import EpochPermutationSampler
from data.djscc_training import TrainingDJSCCDataset
from models.djscc import DJSCC, build_djscc
from runtime.w8_lock import W8_GLOBAL_LOCK_PATH
from training.deterministic_core import (
    OptimizerUpdate,
    canonical_bytes,
    canonical_sha256,
    apply_optimizer_update,
    gradient_status,
    module_gradient_status,
    optimizer_parameters,
    state_tree_sha256,
)
from training.djscc_loss import DJSCCObjective
from training.w8_protocol import (
    W8_ACCUMULATION_FACTOR,
    W8_CHECKPOINT_ROLE,
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_COMPONENT_PATH,
    W8_CORE_ROLE,
    W8_DATASET,
    W8_EPOCHS,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_EXECUTION_IMAGE_FAMILY,
    W8_FINAL_PARTIAL_BATCH,
    W8_EXPECTED_MICROBATCHES,
    W8_PHYSICAL_BATCH_SIZE,
    W8_PROFILE_ID,
    W8_PROTOCOL_VERSION,
    W8_SELECTED_GPU_UUID,
    W8_SELECTED_ROLE,
    W8_SMOKE_ROLE,
    W8_TRAINING_EPOCH_ROLE,
    W8_TRAIN_SAMPLE_COUNT,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    eligibility_for_role,
    fresh_initialization_identity,
    load_w8_config,
    protocol_config_hash,
    validate_w8_config,
)


W8_CHECKPOINT_SCHEMA_VERSION = 1
W8_EPOCH_RECORD_SCHEMA_VERSION = 1
W8_SMOKE_CHECKPOINT_ROLE = "W8_NON_SCIENTIFIC_SMOKE_CHECKPOINT"
W8_SMOKE_EPOCH_ROLE = "W8_NON_SCIENTIFIC_SMOKE_EPOCH_RECORD"
W8_SMOKE_SIDECAR_ROLE = "W8_NON_SCIENTIFIC_SMOKE_CHECKPOINT_SIDECAR"
W8_RNG_STATE_POLICY = {
    "python_random": "not_consumed",
    "numpy": "keyed_stateless_only",
    "torch_cpu": "not_consumed_after_isolated_keyed_initialization",
    "torch_cuda": "not_consumed_by_training_stochasticity",
    "channel": "keyed_training_channel_noise_per_sample_epoch",
    "augmentation": "keyed_per_stable_sample_train_seed_epoch",
    "batch_order": "keyed_philox_per_train_seed_epoch",
    "validation_channel_noise": "keyed_per_image_snr_seed_ratio_not_lambda",
    "serialized_sequential_rng_states": [],
}
W8_CORE_PROTECTED_COUNTERS = {
    "w8_final_training_runs": 0,
    "w8_scientific_optimizer_steps": 0,
    "g10_adjudications": 0,
    "er2_randomized_training": 0,
    "papr_constrained_training": 0,
    "er9_training": 0,
    "learned_test_inference": 0,
    "test_model_facing_access": 0,
}
W8_SMOKE_PROTECTED_COUNTERS = {
    **W8_CORE_PROTECTED_COUNTERS,
    "w8_smoke_optimizer_steps": 0,
}


class W8Hold(RuntimeError):
    """A W8 protocol, lineage, checkpoint, resume, or scope violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8Hold(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _full_sha(value: object, width: int) -> bool:
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # literal-ok: bounded hashing block size
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_bytes(path: Path, raw: bytes) -> None:
    """Publish immutable bytes with same-directory fsync and no replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise W8Hold(f"refusing symlink at immutable W8 path {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise W8Hold(f"immutable W8 path already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_pointer(path: Path, raw: bytes) -> None:
    """Replace only the mutable latest pointer, never through a symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise W8Hold(f"W8 latest pointer is unsafe: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _portable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_portable(item) for item in value)
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _torch_save_bytes(value: Any) -> bytes:
    import io

    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def keyed_training_complex_noise(
    identities: Sequence[Mapping[str, Any]],
    k: int,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct stateless AM-91 complex noise for one train microbatch."""

    expected = list(get("artifacts.rng_identity_fields.training_channel_noise"))
    if not identities:
        raise ValueError("W8 training noise requires one or more identities")
    if not _is_int(k) or k <= 0:
        raise ValueError("W8 training noise k must be positive")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("W8 training noise dtype must be complex64 or complex128")
    rows: list[np.ndarray] = []
    for identity in identities:
        if set(identity) != set(expected):
            raise ValueError("W8 training noise identity differs from params")
        components = keyed_standard_normal(
            "training_channel_noise", identity, size=(2, k)
        )
        rows.append((components[0] + 1j * components[1]) / math.sqrt(2.0))  # literal-ok: unit complex Gaussian convention
    return torch.as_tensor(np.stack(rows), dtype=dtype, device=device)


def learned_recipe(config: RunConfig) -> dict[str, Any]:
    """Copy the complete result-affecting AM-91 recipe from generated params."""

    learned = config.parameters["learned_system"]
    fields = (
        "optimizer", "optimizer_implementation", "adam_beta1", "adam_beta2",
        "adam_epsilon", "adam_weight_decay", "adam_amsgrad", "adam_maximize",
        "adam_foreach", "adam_capturable", "adam_differentiable", "adam_fused",
        "lr", "lr_schedule", "lr_schedule_equation", "lr_min", "lr_warmup_epochs",
        "scheduler_step_unit", "scheduler_epoch_indexing", "scheduler_resume_state",
        "amp", "amp_device_type", "amp_dtype", "grad_scaler_enabled",
        "grad_scaler_init_scale", "grad_scaler_growth_factor",
        "grad_scaler_backoff_factor", "grad_scaler_growth_interval", "batch_order",
        "grad_accumulation_allowed", "drop_last", "dataloader_workers", "pin_memory", "batch_size_policy",
        "accumulation_gradient_rule", "final_partial_accumulation",
        "scheduler_steps_under_accumulation", "checkpoint_every_epochs",
        "checkpoint_timing", "checkpoint_resume_unit", "corrupt_latest_checkpoint_policy",
        "incomplete_epoch_policy", "checkpoint_schema_version", "w5_checkpoint_selection",
        "loss", "augmentation",
    )
    recipe = {field: learned[field] for field in fields}
    recipe["augmentation"] = list(recipe["augmentation"])
    expected = {
        "optimizer": "adam",
        "optimizer_implementation": "torch.optim.Adam",
        "lr_schedule": "cosine",
        "lr_warmup_epochs": 0,
        "scheduler_step_unit": "epoch_start",
        "scheduler_epoch_indexing": "zero_based",
        "scheduler_resume_state": "completed_epoch",
        "amp": True,
        "amp_device_type": "cuda",
        "amp_dtype": "float16",
        "grad_scaler_enabled": True,
        "batch_order": "keyed_philox_permutation_per_epoch",
        "grad_accumulation_allowed": True,
        "drop_last": False,
        "batch_size_policy": "effective_target_with_profile_bound_physical_microbatch",
        "accumulation_gradient_rule": "sample_weighted_mean_over_effective_batch",
        "final_partial_accumulation": "optimizer_step_over_all_remaining_samples_no_drop",
        "scheduler_steps_under_accumulation": "once_per_epoch_at_epoch_start_not_per_optimizer_step",
        "checkpoint_timing": "after_completed_epoch_and_before_next_epoch",
        "checkpoint_resume_unit": "authenticated_completed_epoch",
        "corrupt_latest_checkpoint_policy": "hold_no_older_fallback",
        "incomplete_epoch_policy": "replay_from_latest_authenticated_completed_epoch",
        "checkpoint_schema_version": W8_CHECKPOINT_SCHEMA_VERSION,
        "w5_checkpoint_selection": "prohibited_non_scientific_smoke_only",
        "loss": "CE + lambda * MSE",
        "augmentation": ["random_resized_crop", "horizontal_flip"],
    }
    for key, value in expected.items():
        _require(recipe[key] == value, f"W8 AM-91 recipe differs at {key}")
    return recipe


def learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(_is_int(epoch) and 0 <= epoch < total, "W8 epoch is outside the schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: exact cosine schedule coefficient
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


@dataclass(frozen=True)
class W8SourceLineage:
    source_commit: str
    source_manifest_id: str
    source_manifest_sha256: str
    execution_image: str = W8_EXECUTION_IMAGE_FAMILY

    def validate(self) -> None:
        _require(_full_sha(self.source_commit, 40), "W8 source commit must be full SHA-1")  # literal-ok: Git SHA-1 width
        _require(bool(self.source_manifest_id), "W8 source manifest ID is empty")
        _require(_full_sha(self.source_manifest_sha256, 64), "W8 source manifest SHA is invalid")  # literal-ok: SHA-256 width
        _require(bool(self.execution_image), "W8 execution image is empty")


@dataclass(frozen=True)
class W8TrainingPolicy:
    """Explicit W8 role policy; eligibility is never inferred from a filename."""

    role: str
    validation_enabled: bool
    scientific: bool
    protected_counters: Mapping[str, int]

    def validate(self, config: RunConfig) -> None:
        validate_w8_config(config)
        _require(config.resolved["artifact_role"] == self.role, "W8 policy/config role differs")
        expected = eligibility_for_role(self.role)
        _require(
            all(config.resolved.get(key) == value for key, value in expected.items()),
            "W8 policy eligibility is not exact",
        )
        if self.role == W8_CORE_ROLE:
            _require(self.validation_enabled and self.scientific, "W8 core policy is not scientific")
            _require(dict(self.protected_counters) == W8_CORE_PROTECTED_COUNTERS, "W8 core counters differ")
        elif self.role == W8_SMOKE_ROLE:
            _require(not self.validation_enabled and not self.scientific, "W8 smoke policy is scientific")
            _require(dict(self.protected_counters) == W8_SMOKE_PROTECTED_COUNTERS, "W8 smoke counters differ")
        else:
            raise W8Hold(f"unsupported W8 policy role {self.role!r}")


W8_CORE_TRAINING_POLICY = W8TrainingPolicy(
    role=W8_CORE_ROLE,
    validation_enabled=True,
    scientific=True,
    protected_counters=W8_CORE_PROTECTED_COUNTERS,
)
W8_SMOKE_POLICY = W8TrainingPolicy(
    role=W8_SMOKE_ROLE,
    validation_enabled=False,
    scientific=False,
    protected_counters=W8_SMOKE_PROTECTED_COUNTERS,
)


class _EpochScheduler:
    """Small explicit representation of the frozen epoch-start scheduler."""

    def __init__(self) -> None:
        self.completed_epoch = -1

    def state_dict(self) -> dict[str, int]:
        return {"completed_epoch": self.completed_epoch}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        _require(set(value) == {"completed_epoch"}, "W8 scheduler schema differs")
        _require(_is_int(value["completed_epoch"]) and value["completed_epoch"] >= -1, "W8 scheduler epoch is invalid")
        self.completed_epoch = int(value["completed_epoch"])


class W8Trainer:
    """One fresh W8 run; checkpoints cannot cross runs or W7 roles."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | str,
        runtime_root: Path,
        source_lineage: W8SourceLineage,
        profile_binding: Mapping[str, Any],
        campaign_id: str = "w8-test-campaign",
        run_id: str | None = None,
        policy: W8TrainingPolicy | None = None,
        model: DJSCC | None = None,
        num_workers: int | None = None,
        initial_checkpoint: Path | str | Mapping[str, Any] | None = None,
    ) -> None:
        if initial_checkpoint is not None:
            raise W8Hold(
                "W8 fresh initialization forbids an initial checkpoint; use resume() "
                "only after this run has published its own authenticated checkpoint"
            )
        _require(isinstance(config, RunConfig), "W8 requires a resolved RunConfig")
        validate_w8_config(config)
        self.policy = policy or W8_CORE_TRAINING_POLICY
        if model is not None and self.policy.scientific:
            raise W8Hold(
                "scientific W8 core runs must construct their own keyed fresh model; "
                "injected model state is not an initialization source"
            )
        self.config = config
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        self._validate_initial_runtime_namespace()
        self.campaign_id = str(campaign_id)
        self.run_id = str(run_id or self._default_run_id(config))
        _require(self.campaign_id != "" and self.run_id != "", "W8 campaign/run ID is empty")
        self.source_lineage = source_lineage
        self.source_lineage.validate()
        self.profile_binding = dict(profile_binding)
        self._validate_profile_binding()
        self.config_hash = run_config_hash(config)
        _require(
            self.profile_binding.get("config_hash") == self.config_hash,
            "W8 profile binding config hash differs",
        )
        _require(
            self.profile_binding.get("git_commit") == self.source_lineage.source_commit,
            "W8 profile binding source commit differs",
        )
        self.policy.validate(config)
        self.recipe = learned_recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.protocol_hash = protocol_config_hash(config)
        self.num_workers = int(
            self.recipe["dataloader_workers"] if num_workers is None else num_workers
        )
        _require(self.num_workers >= 0, "W8 dataloader worker count is invalid")
        # build_djscc uses the declared init identity and an isolated Torch RNG.
        # Supplying a model is retained only as a test seam; it never bypasses
        # the explicit no-initial-checkpoint rule above.
        self.model = model if model is not None else build_djscc(config, device=self.device)
        self.model.to(self.device)
        self.objective = DJSCCObjective.from_config(config)
        self.optimizer = self._new_optimizer(self.model)
        self.scheduler = _EpochScheduler()
        self.amp_enabled = bool(self.recipe["amp"] and self.device.type == "cuda")
        self.scaler = self._new_scaler()
        self.completed_epoch = -1
        self.global_optimizer_step = 0
        self.predecessor_checkpoint_id: str | None = None
        self.initialization = fresh_initialization_identity(int(config.resolved["train_seed"]))
        # This digest is captured before any optimizer step and is restored from
        # the same keyed construction on a fresh process.  It makes the
        # no-transfer boundary auditable rather than relying only on a label.
        self.initialization["initial_model_state_sha256"] = state_tree_sha256(
            self.model.state_dict()
        )

    @staticmethod
    def _default_run_id(config: RunConfig) -> str:
        resolved = config.resolved
        return f"w8-{resolved['bw_ratio']}-train{resolved['train_seed']}-channel{resolved['channel_seed']}"

    def _validate_initial_runtime_namespace(self) -> None:
        """Reject foreign files before a fresh W8 run can publish anything.

        A process can die between publishing an epoch-0 record/checkpoint and
        publishing ``latest.json``.  That is not an authenticated completed
        epoch and must be replayed from genesis, but only the narrowly
        recognised epoch-0 publication suffix may be discarded by the runner.
        Anything older, newer, or unknown remains a HOLD.
        """

        _require(not self.runtime_root.is_symlink(), "W8 runtime root is unsafe")
        if not self.runtime_root.exists():
            return
        _require(self.runtime_root.is_dir(), "W8 runtime root is unsafe")
        allowed_top = {
            "checkpoints", "epochs", "latest.json", "validation",
            "selected_checkpoint.json", "run_completion.json",
        }
        _require(
            all(path.name in allowed_top for path in self.runtime_root.iterdir()),
            "W8 runtime root contains foreign state",
        )
        for file_name in (
            "latest.json", "selected_checkpoint.json", "run_completion.json",
        ):
            path = self.runtime_root / file_name
            if path.exists() or path.is_symlink():
                _require(not path.is_symlink() and path.is_file(), f"W8 runtime file is unsafe: {path}")
        validation = self.runtime_root / "validation"
        if validation.exists() or validation.is_symlink():
            _require(not validation.is_symlink() and validation.is_dir(), "W8 validation directory is unsafe")
        latest = self.runtime_root / "latest.json"
        if latest.exists() or latest.is_symlink():
            _require(not latest.is_symlink() and latest.is_file(), "W8 latest pointer is unsafe")
            return
        for directory_name in ("checkpoints", "epochs"):
            directory = self.runtime_root / directory_name
            if not directory.exists():
                continue
            _require(not directory.is_symlink() and directory.is_dir(), "W8 runtime directory is unsafe")
            for path in directory.iterdir():
                _require(not path.is_symlink() and path.is_file(), "W8 runtime contains unsafe unpublished state")
                index = self._runtime_epoch_artifact_index(path.name, directory_name)
                _require(index == 0, "W8 runtime without latest contains non-genesis state")
        if validation.exists():
            _require(not any(validation.iterdir()), "W8 runtime has validation without an authenticated latest pointer")

    @staticmethod
    def _runtime_epoch_artifact_index(name: str, directory_name: str) -> int | None:
        """Return the epoch for one immutable artifact or its atomic temp."""

        if directory_name == "checkpoints":
            suffixes = (".sidecar.json", ".pt")
        elif directory_name == "epochs":
            suffixes = (".json",)
        else:
            return None
        for suffix in suffixes:
            index = W8Trainer._epoch_file_index(name, suffix)
            if index is not None:
                return index
        if not name.startswith(".") or not name.endswith(".tmp"):
            return None
        stem = name[1:-len(".tmp")]
        if "." not in stem:
            return None
        base, token = stem.rsplit(".", 1)
        if not token:
            return None
        for suffix in suffixes:
            index = W8Trainer._epoch_file_index(base, suffix)
            if index is not None:
                return index
        return None

    def discard_unpublished_genesis_suffix(self) -> None:
        """Discard only an unauthenticated epoch-0 publication suffix.

        This method is intentionally explicit: constructing a trainer never
        deletes state.  The campaign runner calls it while holding the
        campaign lock after it has established that no latest pointer exists.
        """

        pointer = self.runtime_root / "latest.json"
        _require(not pointer.exists() and not pointer.is_symlink(), "cannot discard W8 state with a latest pointer")
        for name in ("selected_checkpoint.json", "run_completion.json"):
            path = self.runtime_root / name
            _require(not path.exists() and not path.is_symlink(), f"W8 runtime has terminal state without latest: {name}")
        validation = self.runtime_root / "validation"
        if validation.exists() or validation.is_symlink():
            _require(not validation.is_symlink() and validation.is_dir(), "W8 validation directory is unsafe")
            _require(not any(validation.iterdir()), "W8 validation state cannot be replayed from genesis")
        directories: list[Path] = []
        for name in ("checkpoints", "epochs"):
            directory = self.runtime_root / name
            if not directory.exists():
                continue
            _require(not directory.is_symlink() and directory.is_dir(), "W8 runtime directory is unsafe")
            directories.append(directory)
        self._discard_incomplete_suffix(-1, directories)

    def _validate_profile_binding(self) -> None:
        required = {
            "authentication_status", "execution_profile_id", "gpu_uuid", "gpu_name",
            "gpu_compute_capability", "lock_file_sha256", "git_commit", "config_hash",
        }
        _require(required <= set(self.profile_binding), "W8 profile binding fields are incomplete")
        _require(self.profile_binding["authentication_status"] == "PASSED", "W8 profile was not authenticated")
        _require(self.profile_binding["execution_profile_id"] == W8_PROFILE_ID, "W8 profile differs")
        _require(self.profile_binding["gpu_uuid"] == W8_SELECTED_GPU_UUID, "W8 GPU is not the frozen GTX UUID")
        _require(self.profile_binding["gpu_name"] == "NVIDIA GeForce GTX 1080 Ti", "W8 GPU is not the frozen GTX 1080 Ti")
        _require(str(self.profile_binding["gpu_compute_capability"]) == "6.1", "W8 Pascal capability differs")  # literal-ok: Pascal compute capability
        _require(self.profile_binding["lock_file_sha256"] == "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82", "W8 Pascal lock differs")
        _require(_full_sha(self.profile_binding["git_commit"], 40), "W8 profile source commit is invalid")  # literal-ok: Git SHA-1 width
        if "binding_sha256" in self.profile_binding:
            binding = dict(self.profile_binding)
            digest = binding.pop("binding_sha256")
            _require(digest == canonical_sha256(binding), "W8 profile binding digest differs")

    def _new_optimizer(self, model: nn.Module) -> Adam:
        return Adam(
            model.parameters(),
            lr=float(self.recipe["lr"]),
            betas=(float(self.recipe["adam_beta1"]), float(self.recipe["adam_beta2"])),
            eps=float(self.recipe["adam_epsilon"]),
            weight_decay=float(self.recipe["adam_weight_decay"]),
            amsgrad=bool(self.recipe["adam_amsgrad"]),
            maximize=bool(self.recipe["adam_maximize"]),
            foreach=bool(self.recipe["adam_foreach"]),
            capturable=bool(self.recipe["adam_capturable"]),
            differentiable=bool(self.recipe["adam_differentiable"]),
            fused=bool(self.recipe["adam_fused"]),
        )

    def _new_scaler(self) -> torch.amp.GradScaler | None:
        if not self.amp_enabled:
            return None
        return torch.amp.GradScaler(
            "cuda",
            init_scale=float(self.recipe["grad_scaler_init_scale"]),
            growth_factor=float(self.recipe["grad_scaler_growth_factor"]),
            backoff_factor=float(self.recipe["grad_scaler_backoff_factor"]),
            growth_interval=int(self.recipe["grad_scaler_growth_interval"]),
            enabled=bool(self.recipe["grad_scaler_enabled"]),
        )

    def _set_lr(self, epoch: int) -> float:
        value = learning_rate_for_epoch(self.config, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def _training_noise_identities(self, stable_ids: Sequence[str], epoch: int) -> list[dict[str, Any]]:
        resolved = self.config.resolved
        return [
            {
                "dataset_version": resolved["dataset_version"],
                "split_manifest_hash": get(f"datasets.{resolved['dataset']}.manifest_sha256"),
                "stable_sample_id": stable_id,
                "train_seed": resolved["train_seed"],
                "channel_seed": resolved["channel_seed"],
                "epoch": epoch,
                "channel": resolved["channel"],
                "bw_ratio": resolved["bw_ratio"],
                "k": resolved["k"],
                "train_snr_db": resolved["train_snr_db"],
            }
            for stable_id in stable_ids
        ]

    def _expected_ids(self, dataset: Dataset[tuple[torch.Tensor, int, str]], epoch: int) -> list[str] | None:
        source_sample = getattr(dataset, "source_sample", None)
        if not callable(source_sample):
            source = getattr(dataset, "_source", None)
            source_sample = getattr(source, "source_sample", None)
        if not callable(source_sample):
            return None
        base = [str(source_sample(index).stable_sample_id) for index in range(len(dataset))]
        order = tuple(EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch))
        return [base[index] for index in order]

    @staticmethod
    def _id_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()

    @staticmethod
    def _id_set_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest()

    def train_epoch(
        self,
        epoch: int,
        dataset: Dataset[tuple[torch.Tensor, int, str]],
    ) -> dict[str, Any]:
        _require(epoch == self.completed_epoch + 1, "W8 epoch is not the exact next epoch")
        _require(len(dataset) > 0, "W8 training dataset is empty")
        if self.policy.scientific:
            _require(len(dataset) == W8_TRAIN_SAMPLE_COUNT, "W8 train denominator differs from 8469")
        expected_ids = self._expected_ids(dataset, epoch)
        if self.policy.scientific:
            _require(expected_ids is not None, "W8 scientific training dataset lacks manifest-bound stable IDs")
        target_batch = int(self.config.resolved["effective_batch_size"])
        physical_batch = int(self.config.resolved["physical_batch_size"])
        accumulation_factor = int(self.config.resolved["accumulation_factor"])
        _require(physical_batch * accumulation_factor == target_batch, "W8 batch arithmetic differs")
        _require(target_batch == W8_EFFECTIVE_BATCH_SIZE, "W8 effective batch is not 32")
        sampler = EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch)
        loader = DataLoader(
            dataset,
            batch_size=physical_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        self.model.train()
        lr = self._set_lr(epoch)
        started = time.monotonic()
        total_weighted = 0.0
        ce_weighted = 0.0
        mse_weighted = 0.0
        samples = 0
        microbatches = 0
        optimizer_steps = 0
        scaler_skips = 0
        opportunities = 0
        observed_ids: list[str] = []
        observed_noise_ids: list[str] = []
        all_finite = True
        all_named_finite = True
        named_status_last: dict[str, Any] = {}
        optimizer_status_last: dict[str, Any] = {}
        optimizer_parameter_count: int | None = None
        optimizer_gradient_count_min: int | None = None
        optimizer_gradient_count_max: int | None = None
        group_samples = 0
        group_microbatches = 0
        self.optimizer.zero_grad(set_to_none=True)

        def apply_group(last_named: Mapping[str, Any]) -> OptimizerUpdate:
            nonlocal group_samples, group_microbatches, optimizer_steps, scaler_skips
            nonlocal opportunities, all_finite, optimizer_status_last
            _require(group_samples > 0 and group_microbatches > 0, "W8 empty accumulation group")
            update = apply_optimizer_update(
                self.optimizer, self.scaler, denominator=group_samples
            )
            opportunities += 1
            optimizer_status_last = dict(update.optimizer_gradients)
            all_finite = all_finite and bool(update.optimizer_gradients["finite"])
            if update.applied:
                optimizer_steps += 1
                self.global_optimizer_step += 1
            else:
                scaler_skips += 1
            # A skipped update is still an opportunity; the exact same group is
            # never retried inside this completed epoch.
            if update.applied:
                _require(
                    all(
                        bool(status["finite"]) and bool(status["nonzero"])
                        for name, status in last_named.items()
                        if name != "reconstruction_head"
                    ),
                    "W8 encoder/task gradient coverage is invalid",
                )
                if float(self.config.resolved["lambda"]) > 0:
                    _require(
                        bool(last_named["reconstruction_head"]["finite"])
                        and bool(last_named["reconstruction_head"]["nonzero"]),
                        "W8 reconstruction gradient coverage is invalid",
                    )
            group_samples = 0
            group_microbatches = 0
            self.optimizer.zero_grad(set_to_none=True)
            return update

        for microbatch, (inputs, labels, stable_ids) in enumerate(loader):
            ids = [str(value) for value in stable_ids]
            _require(len(ids) == int(labels.numel()), "W8 batch identity/label count differs")
            observed_ids.extend(ids)
            identities = self._training_noise_identities(ids, epoch)
            observed_noise_ids.extend(canonical_sha256(identity) for identity in identities)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            unit_noise = keyed_training_complex_noise(
                identities, int(self.config.resolved["k"]), device=self.device
            )
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with context:
                output = self.model(
                    inputs, self.config.resolved["train_snr_db"], unit_noise=unit_noise
                )
                loss = self.objective(output, labels, inputs)
            values = {
                "total": float(loss.total.detach().float().item()),
                "cross_entropy": float(loss.cross_entropy.detach().float().item()),
                "reconstruction_mse": float(loss.reconstruction_mse.detach().float().item()),
            }
            _require(all(math.isfinite(value) for value in values.values()), "W8 loss is non-finite")
            count = int(labels.numel())
            weighted_loss = loss.total * count
            if self.scaler is None:
                weighted_loss.backward()
            else:
                self.scaler.scale(weighted_loss).backward()
            named_status = module_gradient_status(self.model)
            optimizer_status = gradient_status(optimizer_parameters(self.optimizer))
            named_status_last = named_status
            optimizer_status_last = optimizer_status
            optimizer_parameter_count = int(optimizer_status["parameter_count"])
            gradient_count = int(optimizer_status["gradient_count"])
            optimizer_gradient_count_min = gradient_count if optimizer_gradient_count_min is None else min(optimizer_gradient_count_min, gradient_count)
            optimizer_gradient_count_max = gradient_count if optimizer_gradient_count_max is None else max(optimizer_gradient_count_max, gradient_count)
            all_finite = all_finite and bool(optimizer_status["finite"])
            all_named_finite = all_named_finite and all(
                bool(status["finite"])
                for status in named_status.values()
                if bool(status["present"])
            )
            group_samples += count
            group_microbatches += 1
            samples += count
            microbatches += 1
            total_weighted += values["total"] * count
            ce_weighted += values["cross_entropy"] * count
            mse_weighted += values["reconstruction_mse"] * count
            group_complete = group_samples >= target_batch
            last_batch = microbatch + 1 == len(loader)
            if group_complete or last_batch:
                apply_group(named_status)

        _require(samples == len(dataset), "W8 epoch did not process the complete dataset")
        _require(len(observed_ids) == len(dataset), "W8 train denominator or final batch differs")
        _require(len(set(observed_ids)) == len(observed_ids), "W8 training stable ID duplicated")
        if expected_ids is not None:
            _require(observed_ids == expected_ids, "W8 keyed batch order or train IDs differ")
        expected_opportunities = math.ceil(samples / target_batch)
        _require(opportunities == expected_opportunities, "W8 optimizer opportunity arithmetic differs")
        _require(
            optimizer_steps + scaler_skips == opportunities,
            "W8 optimizer steps plus GradScaler skips do not cover opportunities",
        )
        if self.policy.scientific:
            _require(
                microbatches == W8_EXPECTED_MICROBATCHES,
                "W8 microbatch count differs from the frozen denominator",
            )
            _require(
                observed_ids
                and len(observed_ids)
                - (len(observed_ids) // physical_batch) * physical_batch
                == W8_FINAL_PARTIAL_BATCH,
                "W8 final partial batch differs from the frozen denominator",
            )
        self.completed_epoch = epoch
        self.scheduler.completed_epoch = epoch
        record = {
            "schema_version": W8_EPOCH_RECORD_SCHEMA_VERSION,
            "artifact_role": W8_TRAINING_EPOCH_ROLE if self.policy.scientific else W8_SMOKE_EPOCH_ROLE,
            "eligibility": eligibility_for_role(self.policy.role),
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "lineage": self._lineage(predecessor=self.predecessor_checkpoint_id),
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "samples": samples,
            "expected_samples": W8_TRAIN_SAMPLE_COUNT if self.policy.scientific else len(dataset),
            "stable_id_count": len(observed_ids),
            "stable_id_order": list(observed_ids),
            "stable_id_order_sha256": self._id_digest(observed_ids),
            "stable_id_set_sha256": self._id_set_digest(observed_ids),
            "training_noise_id_count": len(observed_noise_ids),
            "training_noise_id_sha256": self._id_digest(observed_noise_ids),
            "microbatches": microbatches,
            "expected_microbatches": W8_EXPECTED_MICROBATCHES if self.policy.scientific else math.ceil(len(dataset) / physical_batch),
            "final_physical_batch": W8_FINAL_PARTIAL_BATCH if self.policy.scientific else len(dataset) % physical_batch or physical_batch,
            "optimizer_step_opportunities": opportunities,
            "optimizer_steps": optimizer_steps,
            "grad_scaler_skips": scaler_skips,
            "global_optimizer_step": self.global_optimizer_step,
            "lr": lr,
            "total_loss": total_weighted / samples,
            "cross_entropy": ce_weighted / samples,
            "reconstruction_mse": mse_weighted / samples,
            "duration_seconds": time.monotonic() - started,
            "finite_loss": True,
            "gradient_checks": {
                "optimizer_parameter_count": optimizer_parameter_count,
                "optimizer_gradient_count_min": optimizer_gradient_count_min,
                "optimizer_gradient_count_max": optimizer_gradient_count_max,
                "all_optimizer_gradients_finite": all_finite,
                "all_named_present_gradients_finite": all_named_finite,
                "last_named": named_status_last,
                "last_optimizer": optimizer_status_last,
            },
            "checkpoint_written_after_epoch": True,
            "training_noise_identity_digest": canonical_sha256(observed_noise_ids),
            "validation_noise_identity_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
        }
        return record

    def _lineage(self, *, predecessor: str | None) -> dict[str, Any]:
        resolved = self.config.resolved
        return {
            "protocol_version": W8_PROTOCOL_VERSION,
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "execution_image": self.source_lineage.execution_image,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "protocol_config_hash": self.protocol_hash,
            "resolved_config": self.config.to_dict(),
            "execution_profile_id": resolved["execution_profile_id"],
            "gpu_uuid": self.profile_binding["gpu_uuid"],
            "dataset": resolved["dataset"],
            "dataset_version": resolved["dataset_version"],
            "split_manifest_hash": get(f"datasets.{resolved['dataset']}.manifest_sha256"),
            "architecture": resolved["architecture"],
            "bw_ratio": resolved["bw_ratio"],
            "k": resolved["k"],
            "train_seed": resolved["train_seed"],
            "channel_seed": resolved["channel_seed"],
            "train_snr_db": resolved["train_snr_db"],
            "checkpoint_selection_snr_db": resolved["checkpoint_selection_snr_db"],
            "checkpoint_selection_snr_parameter": resolved["checkpoint_selection_snr_parameter"],
            "lambda": resolved["lambda"],
            "recipe_sha256": self.recipe_sha256,
            "initialization": self.initialization,
            "predecessor_checkpoint_id": predecessor,
        }

    def _checkpoint_payload(self, record_id: str, record_sha256: str) -> dict[str, Any]:
        _require(self.completed_epoch >= 0, "cannot checkpoint before a completed W8 epoch")
        return {
            "schema_version": W8_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": W8_CHECKPOINT_ROLE if self.policy.scientific else W8_SMOKE_CHECKPOINT_ROLE,
            "eligibility": eligibility_for_role(self.policy.role),
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "lineage": self._lineage(predecessor=self.predecessor_checkpoint_id),
            "execution_profile": dict(self.profile_binding),
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "accumulation_position": 0,
            "model_state": _portable(self.model.state_dict()),
            "optimizer_state": _portable(self.optimizer.state_dict()),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": None if self.scaler is None else _portable(self.scaler.state_dict()),
            "rng_state_policy": dict(W8_RNG_STATE_POLICY),
            "initialization": dict(self.initialization),
            "epoch_manifest": {
                "path": f"epochs/epoch-{self.completed_epoch:04d}.json",
                "record_id": record_id,
                "record_sha256": record_sha256,
            },
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "protected_counters": dict(self.policy.protected_counters),
        }

    def save_checkpoint(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Publish an immutable epoch record/checkpoint/sidecar and latest."""

        _require(record.get("epoch") == self.completed_epoch, "W8 record epoch differs")
        _require(record.get("campaign_id") == self.campaign_id and record.get("run_id") == self.run_id, "W8 record run binding differs")
        record_value = dict(record)
        record_value.pop("checkpoint_written_after_epoch", None)
        record_id = canonical_sha256(record_value)
        record_path = self.runtime_root / f"epochs/epoch-{self.completed_epoch:04d}.json"
        record_raw = canonical_bytes({**record_value, "record_id": record_id})
        record_sha256 = hashlib.sha256(record_raw).hexdigest()
        _publish_new_bytes(record_path, record_raw)
        checkpoint_rel = f"checkpoints/epoch-{self.completed_epoch:04d}.pt"
        checkpoint_path = self.runtime_root / checkpoint_rel
        payload = self._checkpoint_payload(record_id, record_sha256)
        checkpoint_started = time.monotonic()
        _publish_new_bytes(checkpoint_path, _torch_save_bytes(payload))
        checkpoint_id = _sha256_file(checkpoint_path)
        sidecar = {
            "schema_version": W8_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": W8_CHECKPOINT_SIDECAR_ROLE if self.policy.scientific else W8_SMOKE_SIDECAR_ROLE,
            "eligibility": eligibility_for_role(self.policy.role),
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "checkpoint_path": checkpoint_rel,
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "accumulation_position": 0,
            "config_hash": self.config_hash,
            "protocol_config_hash": self.protocol_hash,
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "execution_image": self.source_lineage.execution_image,
            "execution_profile_id": self.config.resolved["execution_profile_id"],
            "gpu_uuid": self.profile_binding["gpu_uuid"],
            "dataset": self.config.resolved["dataset"],
            "ratio": self.config.resolved["bw_ratio"],
            "k": self.config.resolved["k"],
            "lambda": self.config.resolved["lambda"],
            "train_seed": self.config.resolved["train_seed"],
            "channel_seed": self.config.resolved["channel_seed"],
            "train_snr_db": self.config.resolved["train_snr_db"],
            "checkpoint_selection_snr_db": self.config.resolved["checkpoint_selection_snr_db"],
            "checkpoint_selection_channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "initialization": dict(self.initialization),
            "epoch_record_path": f"epochs/epoch-{self.completed_epoch:04d}.json",
            "epoch_record_id": record_id,
            "epoch_record_sha256": record_sha256,
            "checkpoint_write_seconds": time.monotonic() - checkpoint_started,
        }
        sidecar_path = checkpoint_path.with_suffix(".sidecar.json")
        _publish_new_bytes(sidecar_path, canonical_bytes(sidecar))
        _replace_pointer(self.runtime_root / "latest.json", canonical_bytes(sidecar))
        self.predecessor_checkpoint_id = checkpoint_id
        return sidecar

    def run_epochs(
        self,
        *,
        final_epoch: int,
        dataset_factory: Callable[[int], Dataset[tuple[torch.Tensor, int, str]]] | None = None,
        repo_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        total = int(get(f"learned_system.epochs.{self.config.resolved['dataset']}"))
        _require(_is_int(final_epoch) and self.completed_epoch < final_epoch < total, "W8 final epoch is outside schedule")
        records: list[dict[str, Any]] = []
        for epoch in range(self.completed_epoch + 1, final_epoch + 1):
            dataset = (
                dataset_factory(epoch)
                if dataset_factory is not None
                else TrainingDJSCCDataset(
                    str(self.config.resolved["dataset"]),
                    int(self.config.resolved["train_seed"]),
                    epoch,
                    repo_root=repo_root,
                )
            )
            record = self.train_epoch(epoch, dataset)
            self.save_checkpoint(record)
            records.append(record)
        return records

    def run_complete(
        self,
        *,
        dataset_factory: Callable[[int], Dataset[tuple[torch.Tensor, int, str]]] | None = None,
        repo_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        return self.run_epochs(
            final_epoch=int(get(f"learned_system.epochs.{self.config.resolved['dataset']}")) - 1,
            dataset_factory=dataset_factory,
            repo_root=repo_root,
        )

    def resume(self) -> dict[str, Any]:
        """Restore exactly this run's latest authenticated completed epoch."""

        pointer_path = self.runtime_root / "latest.json"
        _require(pointer_path.is_file() and not pointer_path.is_symlink(), "W8 latest pointer is missing or unsafe")
        try:
            sidecar = json.loads(pointer_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise W8Hold("W8 latest pointer is corrupt") from None
        self._validate_sidecar(sidecar)
        completed = int(sidecar["completed_epoch"])
        self._validate_runtime_prefix(completed, sidecar)
        payload = self._load_authenticated_payload(sidecar)
        self._restore_payload(payload, completed, checkpoint_id=str(sidecar["checkpoint_id"]))
        return dict(sidecar)

    def load_checkpoint_epoch(self, epoch: int) -> dict[str, Any]:
        """Restore one authenticated same-run epoch for validation selection."""

        _require(_is_int(epoch) and epoch >= 0, "W8 evaluation epoch is invalid")
        sidecars: list[dict[str, Any]] = []
        for index in range(epoch + 1):
            path = self.runtime_root / f"checkpoints/epoch-{index:04d}.sidecar.json"
            _require(path.is_file() and not path.is_symlink(), "W8 checkpoint chain has a gap")
            try:
                value = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise W8Hold("W8 evaluation sidecar is corrupt") from None
            self._validate_sidecar(value)
            _require(value["completed_epoch"] == index, "W8 checkpoint filename/epoch differs")
            expected_predecessor = None if index == 0 else sidecars[index - 1]["checkpoint_id"]
            _require(value["predecessor_checkpoint_id"] == expected_predecessor, "W8 checkpoint predecessor differs")
            sidecars.append(dict(value))
        sidecar = sidecars[-1]
        payload = self._load_authenticated_payload(sidecar)
        self._restore_payload(payload, int(sidecar["completed_epoch"]), checkpoint_id=str(sidecar["checkpoint_id"]))
        return sidecar

    def _load_authenticated_payload(self, sidecar: Mapping[str, Any]) -> Mapping[str, Any]:
        checkpoint_path = self.runtime_root / str(sidecar["checkpoint_path"])
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise W8Hold("W8 authenticated checkpoint cannot be loaded") from None
        self._validate_payload(payload, sidecar)
        _require(isinstance(payload, Mapping), "W8 checkpoint payload is not a mapping")
        return payload

    def _restore_payload(self, payload: Mapping[str, Any], completed: int, *, checkpoint_id: str) -> None:
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            if self.scaler is None:
                _require(payload["scaler_state"] is None, "W8 CPU resume unexpectedly carries scaler state")
            else:
                _require(isinstance(payload["scaler_state"], Mapping), "W8 CUDA resume lacks scaler state")
                self.scaler.load_state_dict(payload["scaler_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W8Hold(f"W8 checkpoint state is invalid: {exc}") from None
        _require(self.scheduler.completed_epoch == completed, "W8 scheduler epoch differs")
        _require(self.optimizer.param_groups[0]["lr"] == learning_rate_for_epoch(self.config, completed), "W8 optimizer LR state differs")
        _require(payload["initialization"] == self.initialization, "W8 resume initialization identity differs")
        self.completed_epoch = completed
        self.global_optimizer_step = int(payload["global_optimizer_step"])
        self.predecessor_checkpoint_id = checkpoint_id

    def _validate_epoch_record(
        self,
        record: Mapping[str, Any],
        *,
        record_id: str,
        expected_epoch: int,
        expected_predecessor: str | None,
    ) -> None:
        """Validate the complete authenticated training record, not just its hash."""

        required = {
            "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
            "lineage", "epoch", "next_epoch", "samples", "expected_samples",
            "stable_id_count", "stable_id_order", "stable_id_order_sha256",
            "stable_id_set_sha256", "training_noise_id_count",
            "training_noise_id_sha256", "microbatches", "expected_microbatches",
            "final_physical_batch", "optimizer_step_opportunities", "optimizer_steps",
            "grad_scaler_skips", "global_optimizer_step", "lr", "total_loss",
            "cross_entropy", "reconstruction_mse", "duration_seconds", "finite_loss",
            "gradient_checks", "training_noise_identity_digest",
            "validation_noise_identity_rule",
        }
        _require(set(record) == required, "W8 epoch record schema differs")
        expected_role = W8_TRAINING_EPOCH_ROLE if self.policy.scientific else W8_SMOKE_EPOCH_ROLE
        _require(record["schema_version"] == W8_EPOCH_RECORD_SCHEMA_VERSION, "W8 epoch record version differs")
        _require(record["artifact_role"] == expected_role, "W8 epoch record role differs")
        _require(record["eligibility"] == eligibility_for_role(self.policy.role), "W8 epoch record eligibility differs")
        _require(record["campaign_id"] == self.campaign_id and record["run_id"] == self.run_id, "W8 epoch record run differs")
        _require(record["epoch"] == expected_epoch and record["next_epoch"] == expected_epoch + 1, "W8 epoch record epoch differs")
        _require(record["lineage"] == self._lineage(predecessor=expected_predecessor), "W8 epoch record lineage differs")
        _require(_full_sha(record_id, 64), "W8 epoch record ID is invalid")  # literal-ok: SHA-256 width

        samples = record["samples"]
        _require(_is_int(samples) and samples > 0, "W8 epoch sample count is invalid")
        expected_samples = W8_TRAIN_SAMPLE_COUNT if self.policy.scientific else samples
        _require(record["expected_samples"] == expected_samples and record["stable_id_count"] == samples, "W8 epoch sample accounting differs")
        stable_ids = record["stable_id_order"]
        _require(isinstance(stable_ids, list) and len(stable_ids) == samples and all(isinstance(item, str) and item for item in stable_ids), "W8 epoch stable IDs are invalid")
        _require(len(set(stable_ids)) == samples, "W8 epoch stable IDs are duplicated")
        _require(record["stable_id_order_sha256"] == self._id_digest(stable_ids), "W8 epoch stable-ID order digest differs")
        _require(record["stable_id_set_sha256"] == self._id_set_digest(stable_ids), "W8 epoch stable-ID set digest differs")
        for field in ("stable_id_order_sha256", "stable_id_set_sha256", "training_noise_id_sha256", "training_noise_identity_digest"):
            _require(_full_sha(record[field], 64), f"W8 epoch {field} is invalid")  # literal-ok: SHA-256 width
        _require(record["training_noise_id_count"] == samples, "W8 training-noise identity count differs")
        _require(_full_sha(record["training_noise_id_sha256"], 64) and _full_sha(record["training_noise_identity_digest"], 64), "W8 training-noise digest fields are invalid")  # literal-ok: SHA-256 width for both authenticated digests
        expected_noise_ids = [
            canonical_sha256(identity)
            for identity in self._training_noise_identities(stable_ids, expected_epoch)
        ]
        _require(record["training_noise_id_sha256"] == self._id_digest(expected_noise_ids), "W8 training-noise ID digest differs")
        _require(record["training_noise_identity_digest"] == canonical_sha256(expected_noise_ids), "W8 training-noise identity digest differs")

        physical = int(self.config.resolved["physical_batch_size"])
        target = int(self.config.resolved["effective_batch_size"])
        expected_microbatches = math.ceil(samples / physical)
        _require(record["microbatches"] == expected_microbatches and record["expected_microbatches"] == expected_microbatches, "W8 epoch microbatch accounting differs")
        _require(record["final_physical_batch"] == (samples % physical or physical), "W8 epoch final batch differs")
        opportunities = math.ceil(samples / target)
        _require(record["optimizer_step_opportunities"] == opportunities, "W8 epoch optimizer opportunity count differs")
        for field in ("optimizer_steps", "grad_scaler_skips", "global_optimizer_step"):
            _require(_is_int(record[field]) and record[field] >= 0, f"W8 epoch {field} is invalid")
        _require(record["optimizer_steps"] + record["grad_scaler_skips"] == opportunities, "W8 epoch optimizer/skips accounting differs")
        _require(record["global_optimizer_step"] >= record["optimizer_steps"], "W8 epoch global optimizer step is invalid")
        _require(record["lr"] == learning_rate_for_epoch(self.config, expected_epoch), "W8 epoch learning rate differs")
        for field in ("total_loss", "cross_entropy", "reconstruction_mse", "duration_seconds"):
            _require(isinstance(record[field], int | float) and not isinstance(record[field], bool) and math.isfinite(float(record[field])), f"W8 epoch {field} is invalid")
        _require(record["duration_seconds"] >= 0 and record["finite_loss"] is True, "W8 epoch finite/duration status differs")
        _require(record["validation_noise_identity_rule"] == W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE, "W8 epoch validation-noise rule differs")

        checks = record["gradient_checks"]
        _require(isinstance(checks, Mapping) and set(checks) == {
            "optimizer_parameter_count", "optimizer_gradient_count_min",
            "optimizer_gradient_count_max", "all_optimizer_gradients_finite",
            "all_named_present_gradients_finite", "last_named", "last_optimizer",
        }, "W8 epoch gradient-check schema differs")
        for field in ("optimizer_parameter_count", "optimizer_gradient_count_min", "optimizer_gradient_count_max"):
            _require(checks[field] is None or (_is_int(checks[field]) and checks[field] >= 0), f"W8 epoch gradient count {field} is invalid")
        _require(isinstance(checks["all_optimizer_gradients_finite"], bool) and isinstance(checks["all_named_present_gradients_finite"], bool), "W8 epoch gradient finiteness is invalid")
        for label, status in [("last_optimizer", checks["last_optimizer"]), *[(str(name), item) for name, item in (checks["last_named"] if isinstance(checks["last_named"], Mapping) else {}).items()]]:
            _require(isinstance(status, Mapping), f"W8 {label} gradient status is invalid")
            _require(set(status) == {"parameter_count", "gradient_count", "present", "finite", "nonzero"}, f"W8 {label} gradient status schema differs")
            _require(_is_int(status["parameter_count"]) and status["parameter_count"] >= 0 and _is_int(status["gradient_count"]) and 0 <= status["gradient_count"] <= status["parameter_count"], f"W8 {label} gradient counts are invalid")
            _require(all(isinstance(status[field], bool) for field in ("present", "finite", "nonzero")), f"W8 {label} gradient flags are invalid")

    def _validate_sidecar(self, sidecar: object) -> None:
        required = {
            "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
            "checkpoint_path", "checkpoint_id", "checkpoint_bytes", "completed_epoch",
            "next_epoch", "global_optimizer_step", "accumulation_position", "config_hash",
            "protocol_config_hash", "source_commit", "source_manifest_id",
            "source_manifest_sha256", "execution_image", "execution_profile_id", "gpu_uuid",
            "dataset", "ratio", "k", "lambda", "train_seed", "channel_seed",
            "train_snr_db", "checkpoint_selection_snr_db",
            "checkpoint_selection_channel_seed_rule", "predecessor_checkpoint_id",
            "initialization", "epoch_record_path", "epoch_record_id", "epoch_record_sha256",
            "checkpoint_write_seconds",
        }
        _require(isinstance(sidecar, Mapping) and set(sidecar) == required, "W8 checkpoint sidecar schema differs")
        value = dict(sidecar)
        expected_role = W8_CHECKPOINT_SIDECAR_ROLE if self.policy.scientific else W8_SMOKE_SIDECAR_ROLE
        _require(value["schema_version"] == W8_CHECKPOINT_SCHEMA_VERSION and value["artifact_role"] == expected_role, "W8 sidecar role/version differs")
        _require(value["eligibility"] == eligibility_for_role(self.policy.role), "W8 sidecar eligibility differs")
        _require(value["campaign_id"] == self.campaign_id and value["run_id"] == self.run_id, "W8 sidecar run differs")
        _require(_full_sha(value["checkpoint_id"], 64), "W8 checkpoint ID is invalid")  # literal-ok: SHA-256 width
        _require(_full_sha(value["source_commit"], 40), "W8 sidecar source commit is invalid")  # literal-ok: Git SHA-1 width
        _require(_full_sha(value["source_manifest_sha256"], 64), "W8 sidecar source manifest SHA is invalid")  # literal-ok: SHA-256 width
        _require(_full_sha(value["epoch_record_id"], 64), "W8 epoch record ID is invalid")  # literal-ok: SHA-256 width
        _require(_full_sha(value["epoch_record_sha256"], 64), "W8 epoch record SHA is invalid")  # literal-ok: SHA-256 width
        _require(_is_int(value["checkpoint_bytes"]) and value["checkpoint_bytes"] > 0, "W8 checkpoint byte count is invalid")
        completed = value["completed_epoch"]
        _require(_is_int(completed) and completed >= 0 and value["next_epoch"] == completed + 1, "W8 sidecar epoch differs")
        _require(_is_int(value["global_optimizer_step"]) and value["global_optimizer_step"] >= 0, "W8 sidecar optimizer step is invalid")
        _require(value["accumulation_position"] == 0, "W8 sidecar accumulation position differs")
        _require(value["checkpoint_path"] == f"checkpoints/epoch-{completed:04d}.pt", "W8 sidecar path differs")
        _require(value["epoch_record_path"] == f"epochs/epoch-{completed:04d}.json", "W8 epoch record path differs")
        for key, expected in {
            "config_hash": self.config_hash,
            "protocol_config_hash": self.protocol_hash,
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "execution_image": self.source_lineage.execution_image,
            "execution_profile_id": self.config.resolved["execution_profile_id"],
            "gpu_uuid": self.profile_binding["gpu_uuid"],
            "dataset": self.config.resolved["dataset"],
            "ratio": self.config.resolved["bw_ratio"],
            "k": self.config.resolved["k"],
            "lambda": self.config.resolved["lambda"],
            "train_seed": self.config.resolved["train_seed"],
            "channel_seed": self.config.resolved["channel_seed"],
            "train_snr_db": self.config.resolved["train_snr_db"],
            "checkpoint_selection_snr_db": self.config.resolved["checkpoint_selection_snr_db"],
            "checkpoint_selection_channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
            "initialization": self.initialization,
        }.items():
            _require(value[key] == expected, f"W8 sidecar {key} differs")
        _require(isinstance(value["checkpoint_write_seconds"], int | float) and not isinstance(value["checkpoint_write_seconds"], bool) and math.isfinite(float(value["checkpoint_write_seconds"])) and float(value["checkpoint_write_seconds"]) >= 0, "W8 checkpoint write duration is invalid")
        checkpoint = self.runtime_root / value["checkpoint_path"]
        _require(checkpoint.is_file() and not checkpoint.is_symlink(), "W8 checkpoint is missing or unsafe")
        _require(checkpoint.stat().st_size == value["checkpoint_bytes"], "W8 checkpoint byte length differs")
        _require(_sha256_file(checkpoint) == value["checkpoint_id"], "W8 checkpoint SHA-256 differs")
        record = self.runtime_root / value["epoch_record_path"]
        _require(record.is_file() and not record.is_symlink(), "W8 epoch record is missing or unsafe")
        raw = record.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == value["epoch_record_sha256"], "W8 epoch record SHA differs")
        try:
            record_value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise W8Hold("W8 epoch record is corrupt") from None
        _require(isinstance(record_value, dict), "W8 epoch record is not a mapping")
        record_id = record_value.pop("record_id", None)
        _require(record_id == value["epoch_record_id"], "W8 epoch record ID differs")
        _require(canonical_sha256(record_value) == record_id, "W8 epoch record content digest differs")
        _require(record_value["global_optimizer_step"] == value["global_optimizer_step"], "W8 sidecar/epoch optimizer step differs")
        self._validate_epoch_record(
            record_value,
            record_id=record_id,
            expected_epoch=completed,
            expected_predecessor=value["predecessor_checkpoint_id"],
        )

    @staticmethod
    def _epoch_file_index(name: str, suffix: str) -> int | None:
        prefix = "epoch-"
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        text = name[len(prefix) : -len(suffix)] if suffix else name[len(prefix) :]
        if len(text) != 4 or not text.isdigit():  # literal-ok: fixed four-digit epoch filenames
            return None
        return int(text)

    def _discard_incomplete_suffix(self, completed: int, directories: Sequence[Path]) -> None:
        """Remove only a crash-left next-epoch publication suffix.

        The latest pointer is authenticated before this method is reached.  A
        record/checkpoint/sidecar left after a crash before latest-pointer
        publication is therefore incomplete, not an older checkpoint to fall
        back to.  Unknown files and gaps still HOLD.
        """

        suffix_paths: list[Path] = []
        for directory in directories:
            for path in directory.iterdir():
                _require(not path.is_symlink() and path.is_file(), "W8 runtime prefix contains unsafe state")
                index = self._runtime_epoch_artifact_index(path.name, directory.name)
                _require(index is not None, "W8 runtime contains an unknown epoch artifact")
                _require(index <= completed + 1, "W8 runtime contains state beyond the next replayable epoch")
                if path.name.startswith(".") or index == completed + 1:
                    suffix_paths.append(path)
        if suffix_paths:
            for path in suffix_paths:
                path.unlink()
            for directory in directories:
                _fsync_directory(directory)

    def _validate_runtime_prefix(self, completed: int, latest: Mapping[str, Any]) -> None:
        checkpoint_dir = self.runtime_root / "checkpoints"
        epoch_dir = self.runtime_root / "epochs"
        _require(checkpoint_dir.is_dir() and epoch_dir.is_dir(), "W8 runtime directories are missing")
        self._discard_incomplete_suffix(completed, (checkpoint_dir, epoch_dir))
        checkpoints = [checkpoint_dir / f"epoch-{epoch:04d}.pt" for epoch in range(completed + 1)]
        sidecars = [checkpoint_dir / f"epoch-{epoch:04d}.sidecar.json" for epoch in range(completed + 1)]
        epochs = [epoch_dir / f"epoch-{epoch:04d}.json" for epoch in range(completed + 1)]
        _require(all(path.is_file() and not path.is_symlink() for path in [*checkpoints, *sidecars, *epochs]), "W8 checkpoint prefix is not exact")
        expected_predecessor: str | None = None
        previous_global_step = 0
        for index, path in enumerate(sidecars):
            try:
                value = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise W8Hold("W8 checkpoint sidecar is corrupt") from None
            self._validate_sidecar(value)
            _require(value["completed_epoch"] == index, "W8 checkpoint filename/epoch differs")
            _require(value["predecessor_checkpoint_id"] == expected_predecessor, "W8 checkpoint predecessor chain differs")
            record_path = self.runtime_root / value["epoch_record_path"]
            try:
                record_value = json.loads(record_path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise W8Hold("W8 epoch record is corrupt") from None
            _require(isinstance(record_value, Mapping), "W8 epoch record is not a mapping")
            record_value = dict(record_value)
            record_value.pop("record_id", None)
            _require(value["global_optimizer_step"] == previous_global_step + record_value["optimizer_steps"], "W8 checkpoint global optimizer chain differs")
            previous_global_step = int(value["global_optimizer_step"])
            expected_predecessor = value["checkpoint_id"]
        _require(sidecars[-1].read_bytes() == (self.runtime_root / "latest.json").read_bytes(), "W8 latest pointer is not the newest authenticated sidecar")
        _require(latest["completed_epoch"] == completed, "W8 latest epoch differs")

    def _validate_payload(self, payload: object, sidecar: Mapping[str, Any]) -> None:
        required = {
            "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
            "lineage", "execution_profile", "completed_epoch", "next_epoch",
            "global_optimizer_step", "accumulation_position", "model_state",
            "optimizer_state", "scheduler_state", "scaler_state", "rng_state_policy",
            "initialization", "epoch_manifest", "predecessor_checkpoint_id",
            "protected_counters",
        }
        _require(isinstance(payload, Mapping) and set(payload) == required, "W8 checkpoint payload schema differs")
        expected_role = W8_CHECKPOINT_ROLE if self.policy.scientific else W8_SMOKE_CHECKPOINT_ROLE
        _require(payload["schema_version"] == W8_CHECKPOINT_SCHEMA_VERSION and payload["artifact_role"] == expected_role, "W8 checkpoint role/version differs")
        _require(payload["eligibility"] == eligibility_for_role(self.policy.role), "W8 checkpoint eligibility differs")
        _require(payload["campaign_id"] == self.campaign_id and payload["run_id"] == self.run_id, "W8 checkpoint run differs")
        _require(payload["execution_profile"] == self.profile_binding, "W8 checkpoint execution binding differs")
        _require(payload["rng_state_policy"] == W8_RNG_STATE_POLICY, "W8 checkpoint RNG policy differs")
        _require(payload["protected_counters"] == dict(self.policy.protected_counters), "W8 checkpoint protected counters differ")
        _require(payload["initialization"] == self.initialization, "W8 checkpoint initialization differs")
        completed = int(sidecar["completed_epoch"])
        _require(payload["completed_epoch"] == completed and payload["next_epoch"] == completed + 1, "W8 checkpoint epoch differs")
        _require(
            _is_int(payload["global_optimizer_step"])
            and payload["global_optimizer_step"] >= 0
            and payload["global_optimizer_step"] == sidecar["global_optimizer_step"]
            and payload["accumulation_position"] == 0,
            "W8 checkpoint optimizer state differs",
        )
        _require(payload["predecessor_checkpoint_id"] == sidecar["predecessor_checkpoint_id"], "W8 checkpoint predecessor differs")
        _require(payload["lineage"] == self._lineage(predecessor=sidecar["predecessor_checkpoint_id"]), "W8 checkpoint lineage differs")
        manifest = payload["epoch_manifest"]
        _require(isinstance(manifest, Mapping) and dict(manifest) == {
            "path": sidecar["epoch_record_path"],
            "record_id": sidecar["epoch_record_id"],
            "record_sha256": sidecar["epoch_record_sha256"],
        }, "W8 epoch manifest binding differs")
        expected_model = build_djscc(self.config, device="cpu")
        expected_optimizer = self._new_optimizer(expected_model)
        expected_group = {key: value for key, value in expected_optimizer.param_groups[0].items() if key != "params"}
        expected_group["lr"] = learning_rate_for_epoch(self.config, completed)
        candidate_optimizer = self._new_optimizer(expected_model)
        try:
            candidate_optimizer.load_state_dict(payload["optimizer_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W8Hold(f"W8 optimizer state is invalid: {exc}") from None
        _require(len(candidate_optimizer.param_groups) == 1, "W8 optimizer group count differs")
        candidate_group = {key: value for key, value in candidate_optimizer.param_groups[0].items() if key != "params"}
        _require(candidate_group == expected_group, "W8 optimizer recipe/state differs")
        temporary_scheduler = _EpochScheduler()
        temporary_scheduler.load_state_dict(payload["scheduler_state"])
        _require(temporary_scheduler.completed_epoch == completed, "W8 scheduler state differs")
        if self.scaler is None:
            _require(payload["scaler_state"] is None, "W8 CPU checkpoint unexpectedly carries scaler")
        else:
            _require(
                isinstance(payload["scaler_state"], Mapping),
                "W8 CUDA checkpoint scaler state is invalid",
            )


def checkpoint_state_digest(trainer: W8Trainer) -> dict[str, str]:
    """Small resume-test digest; it contains no scientific metric."""

    return {
        "model_state_sha256": state_tree_sha256(trainer.model.state_dict()),
        "optimizer_state_sha256": state_tree_sha256(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": state_tree_sha256(trainer.scheduler.state_dict()),
        "scaler_state_sha256": state_tree_sha256(None if trainer.scaler is None else trainer.scaler.state_dict()),
    }


def load_w8_smoke_config(ratio: str = "r_1_6", *, train_seed: int = 0, channel_seed: int = 0) -> RunConfig:
    """Resolve the explicitly non-scientific smoke role; never a core run."""

    return load_w8_config(ratio, train_seed, channel_seed, role=W8_SMOKE_ROLE)
