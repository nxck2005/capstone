"""PAPR-constrained lifecycle contracts: selection, completion and terminal verification.

The PAPR authority/selection/completion artifacts are explicit prospective
identities; none of them impersonates a historical W8 role.  The terminal
verifier authenticates the frozen authority, the exact execution commit's
protected source closure, the complete 100-epoch chain, its optimizer and
GradScaler accounting, every validation record, the frozen max/earliest
checkpoint-selection rule, and the selected checkpoint bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.params import get
from training.deterministic_core import canonical_sha256
from evaluation.w8_validation import (
    ValidationNamespace,
    W8_FORBIDDEN_SELECTION_INPUTS,
    W8_VALIDATION_NOISE_POLICY,
    W8_VALIDATION_ORDER,
    W8_VALIDATION_SAMPLE_COUNT,
    select_checkpoint_epoch,
)
from runtime.source_epochs import load_w10_manifest, source_record
from runtime.source_guard import committed_source_differences, git_tree_hashes
from training.papr_constrained import (
    PAPR_AUTHORITY_PATH,
    PAPR_CAMPAIGN_ID,
    PAPR_SCHEMA_VERSION_AUTHORITY,
    PAPR_CHECKPOINT_ROLE,
    PAPR_COMPLETION_PATH,
    PAPR_COMPLETION_PREFIX,
    PAPR_COMPLETION_ROLE,
    PAPR_CORE_ROLE,
    PAPR_ELIGIBILITY,
    PAPR_EPOCH_ROLE,
    PAPR_EPOCHS,
    PAPR_K,
    PAPR_RATIO,
    PAPR_RUN_ID,
    PAPR_SELECTED_CHECKPOINT_PATH,
    PAPR_SELECTED_ELIGIBILITY,
    PAPR_SELECTED_PREFIX,
    PAPR_SELECTED_ROLE,
    PAPR_SIDECAR_ROLE,
    PAPR_TRAIN_SEED,
    PAPR_CHANNEL_SEED,
    PAPR_VALIDATION_ROLE,
    active_protected_counters,
    papr_cap_db,
    papr_protocol_config_hash,
    papr_protocol_version,
    pre_execution_protected_counters,
)
from training.w8_final import (
    W8_CHECKPOINT_SCHEMA_VERSION,
    W8_EPOCH_RECORD_SCHEMA_VERSION,
    W8_PAPR_BOUND_TOLERANCE_DB,
    W8_PAPR_DOMAIN,
    publish_immutable_json,
)
from training.w8_protocol import (
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
    W8_COMPONENT_PATH,
    W8_DATASET,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_EXPECTED_MICROBATCHES,
    W8_FINAL_PARTIAL_BATCH,
    W8_PROFILE_ID,
    W8_TRAIN_SAMPLE_COUNT,
    W8_VALIDATION_BATCH_SIZE,
    checkpoint_selection_snr_db,
    fresh_initialization_identity,
)

PAPR_AUTHORITY_KIND = "W10_PAPR_CONSTRAINED_TRAINING_AUTHORITY"
PAPR_AUTHORITY_STATUS = "FROZEN_PRE_EXECUTION"
PAPR_AUTHORITY_SCOPE = "PAPR_CONSTRAINED_TRAINING_ONLY"
PAPR_SELECTED_STATUS = "SELECTED_VALIDATION_ONLY"
PAPR_COMPLETION_STATUS = "COMPLETE_VALIDATION_ONLY"
PAPR_SCHEMA_VERSION = PAPR_SCHEMA_VERSION_AUTHORITY
PAPR_SELECTED_SCHEMA_VERSION = 1
PAPR_COMPLETION_SCHEMA_VERSION = 1

PAPR_VALIDATION_NAMESPACE = ValidationNamespace(
    summary_role=PAPR_VALIDATION_ROLE,
    eligibility=PAPR_ELIGIBILITY,
    expected_k={PAPR_RATIO: PAPR_K},
    selected_role=PAPR_SELECTED_ROLE,
    selected_eligibility=PAPR_SELECTED_ELIGIBILITY,
    train_seeds=(PAPR_TRAIN_SEED,),
    summary_schema_version=2,
    metric="validation_top1_accuracy",
    mode="max",
    tie_break="earliest_epoch",
    validation_sample_count=W8_VALIDATION_SAMPLE_COUNT,
    order=W8_VALIDATION_ORDER,
    noise_policy=W8_VALIDATION_NOISE_POLICY,
    forbidden_selection_inputs=tuple(W8_FORBIDDEN_SELECTION_INPUTS),
    measure_papr=True,
    papr_cap_db=papr_cap_db(),
)


class PaprLifecycleHold(RuntimeError):
    """The PAPR authority, chain, selection or completion differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaprLifecycleHold(message)


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaprLifecycleHold(f"{label} is corrupt: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if check:
        _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout.strip() if result.returncode == 0 else ""


def authority_relative_path(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def execution_commit_for(root: Path, authority_path: Path) -> str:
    """The exact commit that carries the frozen authority bytes."""

    relative = authority_relative_path(authority_path, root)
    commit = _git(root, "log", "-1", "--format=%H", "--", relative)
    _require(_full_sha(commit, 40), "PAPR authority has no frozen Git commit")  # literal-ok: Git SHA-1 width
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    _require(committed.returncode == 0, "PAPR authority bytes are absent from its commit")
    _require(
        hashlib.sha256(committed.stdout).hexdigest() == _sha256_file(authority_path),
        "PAPR authority bytes differ from its frozen commit",
    )
    return commit


def assert_authority_source_closure(
    root: Path, authority: Mapping[str, Any], execution_commit: str
) -> dict[str, Any]:
    """The execution commit's protected source must equal the frozen epoch."""

    manifest = load_w10_manifest(root, live=True)
    _require(
        authority.get("source_manifest") == source_record(root, manifest),
        "PAPR authority source manifest binding differs",
    )
    _require(authority.get("source_commit") == manifest["source_commit"], "PAPR authority source commit differs")
    source_commit = str(manifest["source_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, execution_commit],
        cwd=root,
        check=False,
    )
    _require(ancestor.returncode == 0, "PAPR execution commit does not descend from the frozen epoch")
    _require(
        not committed_source_differences(root, source_commit, head=execution_commit),
        "PAPR execution commit changed protected source since the frozen epoch",
    )
    _require(
        git_tree_hashes(root, source_commit) == dict(manifest["tree_hashes"]),
        "PAPR frozen-epoch Git tree closure differs",
    )
    return manifest


def verify_papr_authority(
    root: Path,
    authority_path: Path | None = None,
    *,
    require_live_source: bool = True,
) -> dict[str, Any]:
    """Authenticate the frozen one-run PAPR authority (no runtime required)."""

    root = Path(root).resolve()
    path = Path(authority_path) if authority_path is not None else root / PAPR_AUTHORITY_PATH
    value = _read_json(path, "PAPR training authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    _require(
        identifier == "paprtrainingauth-" + canonical_sha256(body),
        "PAPR authority ID differs",
    )
    _require(value.get("schema_version") == PAPR_SCHEMA_VERSION and value.get("authority_kind") == PAPR_AUTHORITY_KIND, "PAPR authority role differs")
    _require(value.get("status") == PAPR_AUTHORITY_STATUS and value.get("authorization_scope") == PAPR_AUTHORITY_SCOPE, "PAPR authority scope/status differs")
    protocol = value.get("protocol")
    _require(isinstance(protocol, Mapping), "PAPR authority protocol is missing")
    _require(protocol.get("papr_cap_db") == papr_cap_db(), "PAPR authority cap differs")
    _require(protocol.get("papr_domain") == W8_PAPR_DOMAIN, "PAPR authority domain differs")
    _require(protocol.get("training_runs") == 1 and value.get("training_count") == 1, "PAPR authority run count differs")  # literal-ok: AM-98 one run
    _require(protocol.get("train_seed") == PAPR_TRAIN_SEED and protocol.get("channel_seed") == PAPR_CHANNEL_SEED, "PAPR authority seed cell differs")
    _require(protocol.get("fresh_initialization") is True and protocol.get("transfer_initialization_permitted") is False, "PAPR authority fresh-init boundary differs")
    _require(protocol.get("initialization_rule") == "fresh_keyed_init_same_component_as_w8_headline", "PAPR authority initialization rule differs")
    _require(protocol.get("component_path") == W8_COMPONENT_PATH, "PAPR authority component path differs")
    _require(protocol.get("additional_seeds_permitted") is False and protocol.get("outcome_conditioned_tuning_permitted") is False, "PAPR authority seed/tuning boundary differs")
    _require(protocol.get("epochs") == PAPR_EPOCHS, "PAPR authority epoch count differs")
    _require(protocol.get("bw_ratio") == PAPR_RATIO and protocol.get("k") == PAPR_K, "PAPR authority ratio/budget differs")
    _require(protocol.get("recipe") == "frozen_w8_r1_6_recipe" and protocol.get("lambda_core") == 3.0, "PAPR authority recipe/lambda differs")  # literal-ok: G-4 selected lambda
    _require(protocol.get("train_snr_db") == int(get("channel.train_snr_db_fixed")), "PAPR authority training SNR differs")
    _require(protocol.get("protocol_version") == papr_protocol_version(), "PAPR authority protocol version differs")
    _require(protocol.get("protected_counters") == pre_execution_protected_counters(), "PAPR authority pre-execution counters differ")
    execution = protocol.get("execution")
    _require(
        isinstance(execution, Mapping)
        and execution.get("execution_commit_rule") == "commit_that_adds_this_authority_file"
        and execution.get("source_epoch_rule") == "protected_source_at_execution_commit_equals_frozen_epoch"
        and execution.get("sole_writer") is True,
        "PAPR authority execution custody differs",
    )
    _require(value.get("w10_authorized") is False and value.get("test_authorized") is False, "PAPR authority authorizes forbidden work")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "PAPR authority test boundary differs")
    _require(value.get("host") == "confessor" and value.get("device") == "cuda:0", "PAPR authority host/device differs")
    _require(value.get("gpu_uuid") == "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b" and value.get("gpu_name") == "NVIDIA GeForce GTX 1080 Ti", "PAPR authority GPU differs")  # literal-ok: frozen profile identity
    _require(value.get("cuda_visible_devices") == value.get("gpu_uuid"), "PAPR authority CUDA binding differs")
    _require(value.get("config_path") == "configs/learned-papr-constrained-r1-6.yaml", "PAPR authority config path differs")
    if require_live_source:
        execution_commit = execution_commit_for(root, path)
        assert_authority_source_closure(root, value, execution_commit)
    return value


# ---------------------------------------------------------------------------
# Chain, validation, selection and completion
# ---------------------------------------------------------------------------


def epoch_chain_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "chain": [
                {
                    "epoch": int(entry["epoch"]),
                    "epoch_record_sha256": str(entry["epoch_record_sha256"]),
                    "checkpoint_id": str(entry["checkpoint_id"]),
                }
                for entry in entries
            ]
        }
    )


def validation_trajectory_digest(summaries: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {"summary_ids": [str(summary["summary_id"]) for summary in summaries]}
    )


def _expected_record_keys() -> set[str]:
    return {
        "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
        "lineage", "epoch", "next_epoch", "samples", "expected_samples",
        "stable_id_count", "stable_id_order", "stable_id_order_sha256",
        "stable_id_set_sha256", "training_noise_id_count", "training_noise_id_sha256",
        "microbatches", "expected_microbatches", "final_physical_batch",
        "optimizer_step_opportunities", "optimizer_steps", "grad_scaler_skips",
        "global_optimizer_step", "lr", "total_loss", "cross_entropy",
        "reconstruction_mse", "duration_seconds", "finite_loss", "gradient_checks",
        "training_noise_identity_digest", "validation_noise_identity_rule",
        "papr_checks",
    }


def _expected_sidecar_keys() -> set[str]:
    return {
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


def _expected_payload_keys() -> set[str]:
    return {
        "schema_version", "artifact_role", "eligibility", "campaign_id", "run_id",
        "lineage", "execution_profile", "completed_epoch", "next_epoch",
        "global_optimizer_step", "accumulation_position", "model_state",
        "optimizer_state", "scheduler_state", "scaler_state", "rng_state_policy",
        "initialization", "epoch_manifest", "predecessor_checkpoint_id",
        "protected_counters",
    }


def recompute_initial_state_sha256() -> str:
    """Rebuild the keyed fresh PAPR model and hash its exact initial state."""

    from training.deterministic_core import state_tree_sha256
    from training.papr_constrained import build_papr_model, load_papr_config

    model = build_papr_model(load_papr_config(), "cpu")
    return state_tree_sha256(model.state_dict())


def _expected_initialization(*, initial_state_sha256: str) -> dict[str, Any]:
    value = fresh_initialization_identity(PAPR_TRAIN_SEED)
    return {**value, "initial_model_state_sha256": initial_state_sha256}


def _validate_lineage(
    lineage: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    execution_commit: str,
    expected_predecessor: str | None,
    expected_initialization: Mapping[str, Any],
) -> None:
    require = _require
    require(isinstance(lineage, Mapping), "PAPR lineage is not a mapping")
    protocol = authority["protocol"]
    expected = {
        "protocol_version": papr_protocol_version(),
        "source_commit": execution_commit,
        "source_manifest_id": authority["source_manifest"]["manifest_id"],
        "source_manifest_sha256": authority["source_manifest"]["sha256"],
        "campaign_id": PAPR_CAMPAIGN_ID,
        "run_id": PAPR_RUN_ID,
        "config_hash": authority["config_hash"],
        "protocol_config_hash": authority["protocol_config_hash"],
        "execution_profile_id": W8_PROFILE_ID,
        "gpu_uuid": authority["gpu_uuid"],
        "dataset": W8_DATASET,
        "architecture": protocol["architecture"],
        "bw_ratio": PAPR_RATIO,
        "k": PAPR_K,
        "train_seed": PAPR_TRAIN_SEED,
        "channel_seed": PAPR_CHANNEL_SEED,
        "train_snr_db": protocol["train_snr_db"],
        "checkpoint_selection_snr_db": checkpoint_selection_snr_db(),
        "checkpoint_selection_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        "lambda": protocol["lambda_core"],
        "predecessor_checkpoint_id": expected_predecessor,
    }
    for key, value in expected.items():
        require(lineage.get(key) == value, f"PAPR lineage {key} differs")
    require(lineage.get("initialization") == dict(expected_initialization), "PAPR lineage initialization differs")
    require(lineage.get("execution_image") is not None, "PAPR lineage execution image is missing")
    resolved_config = lineage.get("resolved_config")
    require(isinstance(resolved_config, Mapping), "PAPR lineage resolved config is missing")
    resolved = resolved_config.get("resolved")
    require(isinstance(resolved, Mapping), "PAPR lineage resolved choices are missing")
    require(resolved.get("artifact_role") == PAPR_CORE_ROLE, "PAPR lineage artifact role differs")
    require(resolved.get("train_seed") == PAPR_TRAIN_SEED and resolved.get("channel_seed") == PAPR_CHANNEL_SEED, "PAPR lineage resolved seeds differ")
    require(_full_sha(lineage.get("recipe_sha256"), 64), "PAPR lineage recipe digest is invalid")  # literal-ok: SHA-256 width


def _load_payload(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        raise PaprLifecycleHold("PAPR checkpoint cannot be loaded") from None
    _require(isinstance(payload, Mapping), "PAPR checkpoint payload is not a mapping")
    return payload


def verify_epoch_chain(
    root: Path,
    authority: Mapping[str, Any],
    execution_commit: str,
    *,
    expected_initial_state_sha256: str | None,
) -> dict[str, Any]:
    """Authenticate the complete contiguous 100-epoch PAPR chain."""

    runtime = root / str(authority["runtime_root"])
    _require(runtime.is_dir() and not runtime.is_symlink(), "PAPR runtime root is missing or unsafe")
    checkpoint_dir = runtime / "checkpoints"
    epoch_dir = runtime / "epochs"
    _require(checkpoint_dir.is_dir() and epoch_dir.is_dir(), "PAPR runtime directories are missing")
    if expected_initial_state_sha256 is None:
        expected_initial_state_sha256 = recompute_initial_state_sha256()
    _require(_full_sha(expected_initial_state_sha256, 64), "PAPR expected initial-state digest is invalid")  # literal-ok: SHA-256 width
    expected_initialization = _expected_initialization(initial_state_sha256=expected_initial_state_sha256)
    entries: list[dict[str, Any]] = []
    previous_checkpoint: str | None = None
    previous_global_step = 0
    total_steps = 0
    total_skips = 0
    total_opportunities = 0
    max_observed_papr: float | None = None
    for epoch in range(PAPR_EPOCHS):
        record_path = epoch_dir / f"epoch-{epoch:04d}.json"
        checkpoint_path = checkpoint_dir / f"epoch-{epoch:04d}.pt"
        sidecar_path = checkpoint_dir / f"epoch-{epoch:04d}.sidecar.json"
        record = _read_json(record_path, f"PAPR epoch record {epoch}")
        sidecar = _read_json(sidecar_path, f"PAPR epoch sidecar {epoch}")
        _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "PAPR checkpoint is missing or unsafe")
        record_id = record.pop("record_id", None)
        _require(_full_sha(record_id, 64), "PAPR epoch record ID is invalid")  # literal-ok: SHA-256 width
        _require(set(record) == _expected_record_keys(), "PAPR epoch record schema differs")
        _require(canonical_sha256(record) == record_id, "PAPR epoch record content digest differs")
        record_sha256 = _sha256_file(record_path)
        _require(set(sidecar) == _expected_sidecar_keys(), "PAPR sidecar schema differs")
        _require(sidecar.get("schema_version") == W8_CHECKPOINT_SCHEMA_VERSION and sidecar.get("artifact_role") == PAPR_SIDECAR_ROLE, "PAPR sidecar role/version differs")
        _require(sidecar.get("eligibility") == dict(PAPR_ELIGIBILITY), "PAPR sidecar eligibility differs")
        _require(sidecar.get("campaign_id") == PAPR_CAMPAIGN_ID and sidecar.get("run_id") == PAPR_RUN_ID, "PAPR sidecar run differs")
        _require(sidecar.get("completed_epoch") == epoch and sidecar.get("next_epoch") == epoch + 1, "PAPR sidecar epoch differs")
        _require(sidecar.get("checkpoint_path") == f"checkpoints/epoch-{epoch:04d}.pt", "PAPR sidecar checkpoint path differs")
        _require(sidecar.get("epoch_record_path") == f"epochs/epoch-{epoch:04d}.json", "PAPR sidecar record path differs")
        _require(sidecar.get("epoch_record_id") == record_id and sidecar.get("epoch_record_sha256") == record_sha256, "PAPR sidecar record binding differs")
        _require(sidecar.get("source_commit") == execution_commit, "PAPR sidecar source commit differs")
        _require(sidecar.get("source_manifest_id") == authority["source_manifest"]["manifest_id"] and sidecar.get("source_manifest_sha256") == authority["source_manifest"]["sha256"], "PAPR sidecar source manifest differs")
        _require(sidecar.get("config_hash") == authority["config_hash"] and sidecar.get("protocol_config_hash") == authority["protocol_config_hash"], "PAPR sidecar config binding differs")
        _require(sidecar.get("execution_profile_id") == W8_PROFILE_ID and sidecar.get("gpu_uuid") == authority["gpu_uuid"], "PAPR sidecar execution profile differs")
        _require(sidecar.get("dataset") == W8_DATASET and sidecar.get("ratio") == PAPR_RATIO and sidecar.get("k") == PAPR_K, "PAPR sidecar identity differs")
        _require(sidecar.get("lambda") == 3.0 and sidecar.get("train_seed") == PAPR_TRAIN_SEED and sidecar.get("channel_seed") == PAPR_CHANNEL_SEED, "PAPR sidecar recipe/seed differs")  # literal-ok: G-4 selected lambda
        _require(sidecar.get("train_snr_db") == authority["protocol"]["train_snr_db"], "PAPR sidecar training SNR differs")
        _require(sidecar.get("checkpoint_selection_snr_db") == checkpoint_selection_snr_db(), "PAPR sidecar selection SNR differs")
        _require(sidecar.get("checkpoint_selection_channel_seed_rule") == W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE, "PAPR sidecar selection noise rule differs")
        _require(sidecar.get("initialization") == expected_initialization, "PAPR sidecar initialization differs")
        _require(sidecar.get("predecessor_checkpoint_id") == previous_checkpoint, "PAPR checkpoint predecessor chain differs")
        checkpoint_id = _sha256_file(checkpoint_path)
        _require(sidecar.get("checkpoint_id") == checkpoint_id, "PAPR checkpoint SHA-256 differs")
        _require(sidecar.get("checkpoint_bytes") == checkpoint_path.stat().st_size and checkpoint_path.stat().st_size > 0, "PAPR checkpoint byte count differs")
        _require(
            _is_int(sidecar.get("checkpoint_write_seconds")) or isinstance(sidecar.get("checkpoint_write_seconds"), float),
            "PAPR checkpoint write duration is invalid",
        )
        _require(math.isfinite(float(sidecar["checkpoint_write_seconds"])) and float(sidecar["checkpoint_write_seconds"]) >= 0, "PAPR checkpoint write duration is negative")

        _require(record.get("schema_version") == W8_EPOCH_RECORD_SCHEMA_VERSION and record.get("artifact_role") == PAPR_EPOCH_ROLE, "PAPR epoch record role/version differs")
        _require(record.get("eligibility") == dict(PAPR_ELIGIBILITY), "PAPR epoch record eligibility differs")
        _require(record.get("campaign_id") == PAPR_CAMPAIGN_ID and record.get("run_id") == PAPR_RUN_ID, "PAPR epoch record run differs")
        _require(record.get("epoch") == epoch and record.get("next_epoch") == epoch + 1, "PAPR epoch record epoch differs")
        _validate_lineage(
            record.get("lineage"),
            authority=authority,
            execution_commit=execution_commit,
            expected_predecessor=previous_checkpoint,
            expected_initialization=expected_initialization,
        )
        _require(record.get("samples") == W8_TRAIN_SAMPLE_COUNT and record.get("expected_samples") == W8_TRAIN_SAMPLE_COUNT, "PAPR epoch train denominator differs")
        _require(record.get("stable_id_count") == W8_TRAIN_SAMPLE_COUNT, "PAPR epoch stable ID count differs")
        stable_ids = record.get("stable_id_order")
        _require(isinstance(stable_ids, list) and len(stable_ids) == W8_TRAIN_SAMPLE_COUNT and len(set(stable_ids)) == W8_TRAIN_SAMPLE_COUNT, "PAPR epoch stable ID list differs")
        for field in ("stable_id_order_sha256", "stable_id_set_sha256", "training_noise_id_sha256", "training_noise_identity_digest"):
            _require(_full_sha(record.get(field), 64), f"PAPR epoch {field} is invalid")  # literal-ok: SHA-256 width
        _require(record.get("training_noise_id_count") == W8_TRAIN_SAMPLE_COUNT, "PAPR epoch noise identity count differs")
        _require(record.get("microbatches") == W8_EXPECTED_MICROBATCHES and record.get("expected_microbatches") == W8_EXPECTED_MICROBATCHES, "PAPR epoch microbatch count differs")
        _require(record.get("final_physical_batch") == W8_FINAL_PARTIAL_BATCH, "PAPR epoch final partial batch differs")
        opportunities = record.get("optimizer_step_opportunities")
        steps = record.get("optimizer_steps")
        skips = record.get("grad_scaler_skips")
        _require(_is_int(opportunities) and opportunities == math.ceil(W8_TRAIN_SAMPLE_COUNT / W8_EFFECTIVE_BATCH_SIZE), "PAPR epoch opportunity count differs")
        _require(_is_int(steps) and steps >= 0 and _is_int(skips) and skips >= 0, "PAPR epoch optimizer/skip counts are invalid")
        _require(steps + skips == opportunities, "PAPR epoch optimizer/skip accounting differs")
        _require(record.get("global_optimizer_step") == previous_global_step + steps, "PAPR global optimizer chain differs")
        previous_global_step = int(record["global_optimizer_step"])
        total_steps += int(steps)
        total_skips += int(skips)
        total_opportunities += int(opportunities)
        papr = record.get("papr_checks")
        _require(
            isinstance(papr, Mapping)
            and set(papr) == {
                "cap_db", "domain", "max_observed_db", "bound_tolerance_db",
                "constraint_installed", "compliant",
            },
            "PAPR epoch check schema differs",
        )
        _require(
            float(papr["cap_db"]) == papr_cap_db()
            and papr["domain"] == W8_PAPR_DOMAIN
            and float(papr["bound_tolerance_db"]) == W8_PAPR_BOUND_TOLERANCE_DB
            and papr["constraint_installed"] is True
            and papr["compliant"] is True,
            "PAPR epoch cap check differs",
        )
        _require(
            isinstance(papr["max_observed_db"], int | float)
            and not isinstance(papr["max_observed_db"], bool)
            and math.isfinite(float(papr["max_observed_db"]))
            and float(papr["max_observed_db"]) <= papr_cap_db() + W8_PAPR_BOUND_TOLERANCE_DB,
            "PAPR epoch observed maximum exceeds the frozen cap",
        )
        max_observed_papr = (
            float(papr["max_observed_db"])
            if max_observed_papr is None
            else max(max_observed_papr, float(papr["max_observed_db"]))
        )

        payload = _load_payload(checkpoint_path)
        _require(set(payload) == _expected_payload_keys(), "PAPR checkpoint payload schema differs")
        _require(payload.get("schema_version") == W8_CHECKPOINT_SCHEMA_VERSION and payload.get("artifact_role") == PAPR_CHECKPOINT_ROLE, "PAPR checkpoint role/version differs")
        _require(payload.get("eligibility") == dict(PAPR_ELIGIBILITY), "PAPR checkpoint eligibility differs")
        _require(payload.get("campaign_id") == PAPR_CAMPAIGN_ID and payload.get("run_id") == PAPR_RUN_ID, "PAPR checkpoint run differs")
        _require(payload.get("completed_epoch") == epoch and payload.get("next_epoch") == epoch + 1, "PAPR checkpoint epoch differs")
        _require(payload.get("global_optimizer_step") == previous_global_step and payload.get("accumulation_position") == 0, "PAPR checkpoint optimizer position differs")
        _require(payload.get("predecessor_checkpoint_id") == previous_checkpoint, "PAPR checkpoint predecessor differs")
        _require(payload.get("lineage") == record["lineage"], "PAPR checkpoint lineage differs")
        _require(payload.get("initialization") == expected_initialization, "PAPR checkpoint initialization differs")
        _require(payload.get("protected_counters") == active_protected_counters(), "PAPR checkpoint protected counters differ")
        manifest = payload.get("epoch_manifest")
        _require(
            isinstance(manifest, Mapping)
            and dict(manifest) == {
                "path": sidecar["epoch_record_path"],
                "record_id": record_id,
                "record_sha256": record_sha256,
            },
            "PAPR checkpoint epoch manifest differs",
        )
        profile = payload.get("execution_profile")
        _require(isinstance(profile, Mapping) and profile.get("gpu_uuid") == authority["gpu_uuid"] and profile.get("execution_profile_id") == W8_PROFILE_ID, "PAPR checkpoint execution profile differs")
        _require(payload.get("model_state") is not None and payload.get("optimizer_state") is not None, "PAPR checkpoint state is missing")
        del payload
        entries.append(
            {
                "epoch": epoch,
                "epoch_record_sha256": record_sha256,
                "checkpoint_id": checkpoint_id,
                "global_optimizer_step": previous_global_step,
            }
        )
        previous_checkpoint = checkpoint_id
    latest = _read_json(runtime / "latest.json", "PAPR latest pointer")
    _require(latest == _read_json(checkpoint_dir / f"epoch-{PAPR_EPOCHS - 1:04d}.sidecar.json", "PAPR final sidecar"), "PAPR latest pointer is not the final authenticated sidecar")
    _require(total_steps + total_skips == total_opportunities, "PAPR run optimizer/skip totals differ")
    _require(previous_global_step == total_steps, "PAPR global optimizer total differs")
    _require(max_observed_papr is not None, "PAPR run observed no symbol PAPR")
    return {
        "entries": entries,
        "epoch_chain_digest": epoch_chain_digest(entries),
        "optimizer_steps": total_steps,
        "grad_scaler_skips": total_skips,
        "optimizer_step_opportunities": total_opportunities,
        "global_optimizer_step": previous_global_step,
        "max_observed_papr_db": float(max_observed_papr),
        "initialization": expected_initialization,
    }


def load_validation_summaries(runtime_root: Path) -> list[dict[str, Any]]:
    """Load the exact ordered epoch-summary prefix from one PAPR runtime root."""

    directory = Path(runtime_root) / "validation"
    if not directory.exists():
        return []
    _require(directory.is_dir() and not directory.is_symlink(), "PAPR validation directory is unsafe")
    values: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        _require(path.is_file() and not path.is_symlink(), "PAPR validation artifact is unsafe")
        _require(path.name.startswith("epoch-") and path.name.endswith(".json"), "PAPR validation directory contains an unknown file")
        number = path.name.removeprefix("epoch-").removesuffix(".json")
        _require(len(number) == 4 and number.isdigit() and int(number) == len(values), "PAPR validation summaries are not an ordered prefix")  # literal-ok: fixed four-digit epoch filenames
        values.append(_read_json(path, "PAPR validation summary"))
    return values


def validate_summary_chain(
    root: Path,
    authority: Mapping[str, Any],
    chain: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Authenticate every summary against its authenticated checkpoint sidecar."""

    from evaluation.w8_validation import _validate_summary, evaluation_config_hash

    runtime = root / str(authority["runtime_root"])
    summaries = load_validation_summaries(runtime)
    _require(len(summaries) == PAPR_EPOCHS, "PAPR validation history is incomplete")
    expected_evaluation_hash = canonical_sha256(
        {
            "protocol_config_hash": authority["protocol_config_hash"],
            "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "snr_db": checkpoint_selection_snr_db(),
            "channel_seed": PAPR_CHANNEL_SEED,
            "batch_size": W8_VALIDATION_BATCH_SIZE,
            "order": W8_VALIDATION_ORDER,
        }
    )
    for epoch, summary in enumerate(summaries):
        _validate_summary(
            summary,
            expected_epoch=epoch,
            expected_evaluation_config_hash=expected_evaluation_hash,
            namespace=PAPR_VALIDATION_NAMESPACE,
        )
        _require(summary["campaign_id"] == PAPR_CAMPAIGN_ID and summary["run_id"] == PAPR_RUN_ID, "PAPR summary run differs")
        _require(summary["ratio"] == PAPR_RATIO and summary["k"] == PAPR_K, "PAPR summary ratio/budget differs")
        sidecar = _read_json(
            runtime / f"checkpoints/epoch-{epoch:04d}.sidecar.json",
            f"PAPR summary sidecar {epoch}",
        )
        _require(sidecar["checkpoint_id"] == summary["checkpoint_id"] == chain["entries"][epoch]["checkpoint_id"], "PAPR summary checkpoint binding differs")
    _require(
        all(
            summary["validation_noise_id_digest"] == summaries[0]["validation_noise_id_digest"]
            for summary in summaries[1:]
        ),
        "PAPR validation noise did not remain fixed across epochs",
    )
    return summaries


def build_papr_selection_record(
    *,
    authority: Mapping[str, Any],
    authority_relative: str,
    authority_sha256: str,
    execution_commit: str,
    chain: Mapping[str, Any],
    validation_digest: str,
    selection: Mapping[str, Any],
    selected_sidecar: Mapping[str, Any],
    checkpoint_relative: str,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": PAPR_SELECTED_SCHEMA_VERSION,
        "artifact_role": PAPR_SELECTED_ROLE,
        "status": PAPR_SELECTED_STATUS,
        "authority_id": authority["authority_id"],
        "authority_path": authority_relative,
        "authority_sha256": authority_sha256,
        "source_manifest": authority["source_manifest"],
        "scientific_source_commit": authority["source_commit"],
        "execution_commit": execution_commit,
        "config_hash": authority["config_hash"],
        "protocol_config_hash": authority["protocol_config_hash"],
        "papr_cap_db": papr_cap_db(),
        "papr_domain": W8_PAPR_DOMAIN,
        "train_seed": PAPR_TRAIN_SEED,
        "channel_seed": PAPR_CHANNEL_SEED,
        "epochs_completed": PAPR_EPOCHS,
        "epoch_chain_digest": chain["epoch_chain_digest"],
        "validation_trajectory_digest": validation_digest,
        "selection": dict(selection),
        "selected_epoch": int(selection["selected_epoch"]),
        "checkpoint_id": str(selection["selected_checkpoint_id"]),
        "checkpoint_path": checkpoint_relative,
        "checkpoint_bytes": int(checkpoint_bytes),
        "n_correct": int(selection["n_correct"]),
        "n_total": int(selection["n_total"]),
        "training_runs": 1,
        "protected_counters": active_protected_counters(),
        "test": "SEALED",
        "test_access": 0,
    }
    _require(selected_sidecar["checkpoint_id"] == body["checkpoint_id"], "PAPR selection sidecar binding differs")
    body["selection_id"] = PAPR_SELECTED_PREFIX + canonical_sha256(body)
    return body


def build_papr_completion(
    *,
    authority: Mapping[str, Any],
    authority_relative: str,
    authority_sha256: str,
    execution_commit: str,
    chain: Mapping[str, Any],
    validation_digest: str,
    selection_record: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": PAPR_COMPLETION_SCHEMA_VERSION,
        "artifact_role": PAPR_COMPLETION_ROLE,
        "status": PAPR_COMPLETION_STATUS,
        "authority_id": authority["authority_id"],
        "authority_path": authority_relative,
        "authority_sha256": authority_sha256,
        "source_manifest": authority["source_manifest"],
        "scientific_source_commit": authority["source_commit"],
        "execution_commit": execution_commit,
        "runtime_root": authority["runtime_root"],
        "campaign_id": PAPR_CAMPAIGN_ID,
        "run_id": PAPR_RUN_ID,
        "config_hash": authority["config_hash"],
        "protocol_config_hash": authority["protocol_config_hash"],
        "papr_cap_db": papr_cap_db(),
        "papr_domain": W8_PAPR_DOMAIN,
        "papr_max_observed_db": float(chain["max_observed_papr_db"]),
        "papr_cap_compliant": bool(
            float(chain["max_observed_papr_db"])
            <= papr_cap_db() + W8_PAPR_BOUND_TOLERANCE_DB
        ),
        "epochs": PAPR_EPOCHS,
        "epoch_chain_digest": chain["epoch_chain_digest"],
        "validation_trajectory_digest": validation_digest,
        "optimizer_step_opportunities": int(chain["optimizer_step_opportunities"]),
        "optimizer_steps": int(chain["optimizer_steps"]),
        "grad_scaler_skips": int(chain["grad_scaler_skips"]),
        "global_optimizer_step": int(chain["global_optimizer_step"]),
        "selection_id": str(selection_record["selection_id"]),
        "selected_epoch": int(selection_record["selected_epoch"]),
        "checkpoint_id": str(selection_record["checkpoint_id"]),
        "checkpoint_path": str(selection_record["checkpoint_path"]),
        "checkpoint_bytes": int(selection_record["checkpoint_bytes"]),
        "n_correct": int(selection_record["n_correct"]),
        "n_total": int(selection_record["n_total"]),
        "training_runs": 1,
        "train_seed": PAPR_TRAIN_SEED,
        "channel_seed": PAPR_CHANNEL_SEED,
        "fresh_initialization": True,
        "transfer_initialization": False,
        "initial_model_state_sha256": str(chain["initialization"]["initial_model_state_sha256"]),
        "protected_counters": active_protected_counters(),
        "test": "SEALED",
        "test_access": 0,
    }
    body["completion_id"] = PAPR_COMPLETION_PREFIX + canonical_sha256(body)
    return body


def verify_papr_terminal(
    root: Path,
    *,
    authority_path: Path | None = None,
    expected_initial_state_sha256: str | None = None,
) -> dict[str, Any]:
    """The complete PAPR lifecycle terminal proof (worker-side runtime required)."""

    root = Path(root).resolve()
    path = Path(authority_path) if authority_path is not None else root / PAPR_AUTHORITY_PATH
    authority = verify_papr_authority(root, path)
    execution_commit = execution_commit_for(root, path)
    assert_authority_source_closure(root, authority, execution_commit)
    chain = verify_epoch_chain(
        root,
        authority,
        execution_commit,
        expected_initial_state_sha256=expected_initial_state_sha256,
    )
    summaries = validate_summary_chain(root, authority, chain)
    validation_digest = validation_trajectory_digest(summaries)
    recomputed = select_checkpoint_epoch(
        summaries, expected_epochs=PAPR_EPOCHS, namespace=PAPR_VALIDATION_NAMESPACE
    )
    selected_path = root / PAPR_SELECTED_CHECKPOINT_PATH
    stored = _read_json(selected_path, "PAPR selected checkpoint")
    stored_body = dict(stored)
    stored_id = stored_body.pop("selection_id", None)
    _require(stored_id == PAPR_SELECTED_PREFIX + canonical_sha256(stored_body), "PAPR selection ID differs")
    selected_epoch = int(stored["selected_epoch"])
    sidecar = _read_json(
        root / str(authority["runtime_root"]) / f"checkpoints/epoch-{selected_epoch:04d}.sidecar.json",
        "PAPR selected sidecar",
    )
    checkpoint = root / str(stored["checkpoint_path"])
    _require(checkpoint.is_file() and not checkpoint.is_symlink(), "PAPR selected checkpoint is missing or unsafe")
    _require(stored["checkpoint_id"] == sidecar["checkpoint_id"] == _sha256_file(checkpoint), "PAPR selected checkpoint bytes differ")
    authority_relative = authority_relative_path(path, root)
    authority_sha256 = _sha256_file(path)
    expected_selection = build_papr_selection_record(
        authority=authority,
        authority_relative=authority_relative,
        authority_sha256=authority_sha256,
        execution_commit=execution_commit,
        chain=chain,
        validation_digest=validation_digest,
        selection=recomputed,
        selected_sidecar=sidecar,
        checkpoint_relative=str(stored["checkpoint_path"]),
        checkpoint_bytes=int(stored["checkpoint_bytes"]),
    )
    _require(stored == expected_selection, "PAPR selected checkpoint record differs from the recomputed lifecycle")
    completion = _read_json(root / PAPR_COMPLETION_PATH, "PAPR training completion")
    completion_body = dict(completion)
    completion_id = completion_body.pop("completion_id", None)
    _require(completion_id == PAPR_COMPLETION_PREFIX + canonical_sha256(completion_body), "PAPR completion ID differs")
    expected_completion = build_papr_completion(
        authority=authority,
        authority_relative=authority_relative,
        authority_sha256=authority_sha256,
        execution_commit=execution_commit,
        chain=chain,
        validation_digest=validation_digest,
        selection_record=stored,
    )
    _require(completion == expected_completion, "PAPR completion record differs from the recomputed lifecycle")
    _require(completion["protected_counters"] == active_protected_counters(), "PAPR completion counters differ")
    _require(completion["training_runs"] == 1, "PAPR completion run count differs")
    _require(completion["fresh_initialization"] is True and completion["transfer_initialization"] is False, "PAPR completion initialization boundary differs")
    _require(completion["test"] == "SEALED" and completion["test_access"] == 0, "PAPR completion crossed test boundary")
    return {
        "authority_id": authority["authority_id"],
        "execution_commit": execution_commit,
        "selection_id": stored["selection_id"],
        "completion_id": completion["completion_id"],
        "selected_epoch": selected_epoch,
        "checkpoint_id": stored["checkpoint_id"],
        "optimizer_steps": chain["optimizer_steps"],
        "grad_scaler_skips": chain["grad_scaler_skips"],
        "papr_max_observed_db": chain["max_observed_papr_db"],
        "test_access": 0,
    }


def publish_selected_and_completion(
    root: Path,
    *,
    authority: Mapping[str, Any],
    authority_path: Path,
    execution_commit: str,
    chain: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    selected_sidecar: Mapping[str, Any],
    checkpoint_relative: str,
    checkpoint_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the selected checkpoint and terminal completion (idempotent)."""

    root = Path(root).resolve()
    authority_path = Path(authority_path)
    validation_digest = validation_trajectory_digest(summaries)
    authority_relative = authority_relative_path(authority_path, root)
    authority_sha256 = _sha256_file(authority_path)
    selected = build_papr_selection_record(
        authority=authority,
        authority_relative=authority_relative,
        authority_sha256=authority_sha256,
        execution_commit=execution_commit,
        chain=chain,
        validation_digest=validation_digest,
        selection=selection,
        selected_sidecar=selected_sidecar,
        checkpoint_relative=checkpoint_relative,
        checkpoint_bytes=checkpoint_bytes,
    )
    completion = build_papr_completion(
        authority=authority,
        authority_relative=authority_relative,
        authority_sha256=authority_sha256,
        execution_commit=execution_commit,
        chain=chain,
        validation_digest=validation_digest,
        selection_record=selected,
    )
    publish_immutable_json(root / PAPR_SELECTED_CHECKPOINT_PATH, selected)
    publish_immutable_json(root / PAPR_COMPLETION_PATH, completion)
    runtime = root / str(authority["runtime_root"])
    publish_immutable_json(runtime / "selected_checkpoint.json", selected)
    publish_immutable_json(runtime / "run_completion.json", completion)
    return selected, completion


__all__ = [
    "PAPR_AUTHORITY_KIND",
    "PAPR_AUTHORITY_SCOPE",
    "PAPR_AUTHORITY_STATUS",
    "PAPR_COMPLETION_STATUS",
    "PAPR_SELECTED_STATUS",
    "PAPR_VALIDATION_NAMESPACE",
    "PaprLifecycleHold",
    "assert_authority_source_closure",
    "build_papr_completion",
    "build_papr_selection_record",
    "epoch_chain_digest",
    "execution_commit_for",
    "load_validation_summaries",
    "publish_selected_and_completion",
    "recompute_initial_state_sha256",
    "validate_summary_chain",
    "validation_trajectory_digest",
    "verify_epoch_chain",
    "verify_papr_authority",
    "verify_papr_terminal",
]
