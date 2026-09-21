"""Synthetic PAPR lifecycle integration: roles, counters, resume, terminal verifier.

These tests exercise the real PAPR protocol/config path with bounded fixtures.
No real Imagenette training, no GPU and no scientific execution occurs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluation.w8_validation as validation
import training.papr_constrained as papr
import training.papr_lifecycle as lifecycle
import training.w8_final as w8_training
from evaluation.w8_validation import _validate_summary, select_checkpoint_epoch
from runtime.source_epochs import source_record
from tests.w8_hardening_fixtures import (
    TinyDJSCC,
    TinyValidationDataset,
    TinyW8Dataset,
    lineage,
    profile_binding,
)
from training.deterministic_core import canonical_bytes, canonical_sha256
from training.papr_constrained import (
    PAPR_CAMPAIGN_ID,
    PAPR_CHANNEL_SEED,
    PAPR_CHECKPOINT_ROLE,
    PAPR_ELIGIBILITY,
    PAPR_EPOCH_ROLE,
    PAPR_RUN_ID,
    PAPR_SIDECAR_ROLE,
    PAPR_TRAIN_SEED,
    PaprConstrainedTrainer,
    PAPR_TRAINING_POLICY,
    active_protected_counters,
    load_papr_config,
)
from training.w8_final import W8Hold, fresh_initialization_identity, publish_immutable_json
from training.w8_protocol import W8_CHECKPOINT_SELECTION_SNR_PARAMETER, checkpoint_selection_snr_db


class _CompliantTinyDJSCC(TinyDJSCC):
    """The tiny fixture with symbol PAPR inside the frozen 3.0 dB cap."""

    def forward(self, inputs, snr_db, *, unit_noise=None):
        import dataclasses

        value = super().forward(inputs, snr_db, unit_noise=unit_noise)
        return dataclasses.replace(value, papr_db=torch.zeros_like(value.papr_db))


@pytest.fixture(autouse=True)
def _tiny_model(monkeypatch):
    monkeypatch.setattr(w8_training, "build_djscc", lambda *_args, **_kwargs: _CompliantTinyDJSCC())
    monkeypatch.setattr(papr, "build_papr_model", lambda *_args, **_kwargs: _CompliantTinyDJSCC())


@pytest.fixture
def _bounded_counts(monkeypatch):
    monkeypatch.setattr(w8_training, "W8_TRAIN_SAMPLE_COUNT", 5)
    monkeypatch.setattr(w8_training, "W8_EXPECTED_MICROBATCHES", 1)
    monkeypatch.setattr(w8_training, "W8_FINAL_PARTIAL_BATCH", 5)


def _trainer(root: Path) -> PaprConstrainedTrainer:
    config = load_papr_config()
    return PaprConstrainedTrainer(
        config,
        device="cpu",
        runtime_root=root,
        source_lineage=lineage(),
        profile_binding=profile_binding(config),
        campaign_id=PAPR_CAMPAIGN_ID,
        run_id=PAPR_RUN_ID,
        policy=PAPR_TRAINING_POLICY,
        num_workers=0,
    )


def test_papr_trainer_emits_papr_roles_counters_and_cap_checks(tmp_path: Path, _bounded_counts) -> None:
    trainer = _trainer(tmp_path / "runtime")
    record = trainer.train_epoch(0, TinyW8Dataset(0))
    sidecar = trainer.save_checkpoint(record)

    assert record["artifact_role"] == PAPR_EPOCH_ROLE
    assert record["eligibility"] == PAPR_ELIGIBILITY
    assert record["papr_checks"]["cap_db"] == papr.papr_cap_db()
    assert record["papr_checks"]["compliant"] is True
    assert record["optimizer_steps"] + record["grad_scaler_skips"] == record["optimizer_step_opportunities"]
    assert sidecar["artifact_role"] == PAPR_SIDECAR_ROLE
    assert sidecar["eligibility"] == PAPR_ELIGIBILITY
    payload = torch.load(trainer.runtime_root / sidecar["checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["artifact_role"] == PAPR_CHECKPOINT_ROLE
    assert payload["eligibility"] == PAPR_ELIGIBILITY
    assert payload["protected_counters"] == active_protected_counters()
    assert payload["protected_counters"]["papr_constrained_training"] == 1
    assert payload["initialization"]["predecessor_checkpoint_id"] is None
    assert payload["lineage"]["protocol_version"] == papr.papr_protocol_version()

    fresh = _trainer(tmp_path / "runtime")
    fresh.resume()
    assert fresh.completed_epoch == 0
    assert fresh.initialization == trainer.initialization


def test_papr_validation_namespace_is_resumable_and_foreign_logs_are_rejected(
    tmp_path: Path, monkeypatch, _bounded_counts
) -> None:
    monkeypatch.setattr(validation, "W8_VALIDATION_SAMPLE_COUNT", 5)
    monkeypatch.setattr(
        validation,
        "ValidationDJSCCDataset",
        lambda dataset, repo_root=None: TinyValidationDataset(dataset, repo_root=repo_root, count=5),
    )
    actual_get = validation.get

    def get_value(path: str):
        if path == "datasets.imagenette160.val_images":
            return 5
        return actual_get(path)

    monkeypatch.setattr(validation, "get", get_value)
    bounded_namespace = lifecycle.PAPR_VALIDATION_NAMESPACE.__class__(
        **{**lifecycle.PAPR_VALIDATION_NAMESPACE.__dict__, "validation_sample_count": 5}
    )

    runtime = tmp_path / "runtime"
    trainer = _trainer(runtime)
    record = trainer.train_epoch(0, TinyW8Dataset(0))
    sidecar = trainer.save_checkpoint(record)
    evaluation = validation.evaluate_validation(
        trainer, checkpoint_id=sidecar["checkpoint_id"], namespace=bounded_namespace
    )
    assert evaluation.summary["artifact_role"] == papr.PAPR_VALIDATION_ROLE
    publish_immutable_json(runtime / "validation/epoch-0000.json", evaluation.summary)

    # A fresh trainer on the same authenticated prefix accepts its own runtime.
    resumed = _trainer(runtime)
    resumed.resume()
    assert resumed.completed_epoch == 0
    assert len(lifecycle.load_validation_summaries(runtime)) == 1

    # Crash window: the authenticated checkpoint exists but its derived
    # validation record does not.  It is re-derivable, not a HOLD.
    (runtime / "validation/epoch-0000.json").unlink()
    restored = _trainer(runtime)
    restored.resume()
    sidecar_again = restored.load_checkpoint_epoch(0)
    rederived = validation.evaluate_validation(
        restored, checkpoint_id=sidecar_again["checkpoint_id"], namespace=bounded_namespace
    )
    assert rederived.summary["n_correct"] == evaluation.summary["n_correct"]
    publish_immutable_json(runtime / "validation/epoch-0000.json", rederived.summary)

    # A foreign W8-style side log is never accepted into the PAPR runtime.
    (runtime / "validation_summaries.jsonl").write_text("{}\n")
    with pytest.raises(W8Hold, match="foreign state"):
        _trainer(runtime)


def _bounded_lifecycle_constants(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "PAPR_TRAIN_SEED", PAPR_TRAIN_SEED)
    monkeypatch.setattr(lifecycle, "PAPR_CHANNEL_SEED", PAPR_CHANNEL_SEED)
    monkeypatch.setattr(lifecycle, "PAPR_K", papr.PAPR_K)
    monkeypatch.setattr(lifecycle, "W8_TRAIN_SAMPLE_COUNT", 5)
    monkeypatch.setattr(lifecycle, "W8_EXPECTED_MICROBATCHES", 1)
    monkeypatch.setattr(lifecycle, "W8_FINAL_PARTIAL_BATCH", 5)
    monkeypatch.setattr(lifecycle, "W8_EFFECTIVE_BATCH_SIZE", 5)
    monkeypatch.setattr(lifecycle, "W8_VALIDATION_BATCH_SIZE", 5)
    monkeypatch.setattr(lifecycle, "W8_PROFILE_ID", "confessor_pascal_cu126")
    monkeypatch.setattr(validation, "W8_VALIDATION_BATCH_SIZE", 5)


def _synthetic_manifest(root: Path) -> dict:
    body = {
        "schema_version": 2,
        "manifest_kind": "W10_PREPARATORY_SOURCE_SUCCESSOR_V2",
        "source_commit": "a" * 40,
        "source_commit_comparison": "exact_clean_HEAD_at_freeze",
        "tree_hashes": {key: "b" * 40 for key in ("repository", "src", "tools", "configs", "spec", "tests")},
        "requirements_pascal_lock_sha256": "c" * 64,
        "relevant_config_sha256": {"configs/learned-papr-constrained-r1-6.yaml": "d" * 64},
        "protected_source_prefixes": [],
        "allowed_evidence_runtime_prefixes": [],
        "working_tree_guard": {},
        "governs": ["papr_constrained_training", "w10_validation_rehearsal"],
        "historical_w9_downstream_manifest": {"path": "x", "source_commit": "e" * 40, "manifest_id": "f"},
        "pre_science_state": {
            "papr_constrained_training_runs": 0,
            "w10_authority_frozen": False,
            "w10_scientific_units": 0,
            "g12_freeze_manifest": False,
            "test": "SEALED",
            "test_access": 0,
        },
    }
    body["manifest_id"] = "w10downstreamsourcev2-" + canonical_sha256(body)
    path = root / "results/learned/w10/w10_downstream_source_manifest_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return body


def _synthetic_authority(root: Path, monkeypatch) -> dict:
    manifest = _synthetic_manifest(root)
    monkeypatch.setattr(lifecycle, "load_w10_manifest", lambda _root, live=True, epoch=None: manifest)
    monkeypatch.setattr(lifecycle, "source_record", lambda _root, _manifest, path=None: source_record(root, manifest))
    authority_path = root / "results/learned/w10/papr_training_authorization.json"
    config = load_papr_config()
    protocol = papr.papr_authority_protocol(config, source_record_value=source_record(root, manifest))
    body = {
        "schema_version": papr.PAPR_SCHEMA_VERSION_AUTHORITY,
        "authority_kind": "W10_PAPR_CONSTRAINED_TRAINING_AUTHORITY",
        "status": "FROZEN_PRE_EXECUTION",
        "authorization_scope": "PAPR_CONSTRAINED_TRAINING_ONLY",
        "source_manifest": source_record(root, manifest),
        "source_binding": manifest,
        "source_commit": manifest["source_commit"],
        "protocol": protocol,
        "protocol_config_hash": papr.papr_protocol_config_hash(config),
        "config_path": "configs/learned-papr-constrained-r1-6.yaml",
        "config_hash": papr.run_config_hash(config),
        "runtime_root": "checkpoints/synthetic_papr/train0_channel0",
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
        "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
        "compute_capability": "6.1",
        "device": "cuda:0",
        "cuda_visible_devices": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
        "cuda_mapping": {
            "cuda_visible_devices": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
            "logical_device": "cuda:0",
            "cuda0_gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
            "cuda0_gpu_name": "NVIDIA GeForce GTX 1080 Ti",
            "cuda0_compute_capability": "6.1",
            "device_count": 1,
        },
        "training_count": 1,
        "w10_authorized": False,
        "test_authorized": False,
        "test": "SEALED",
        "test_access": 0,
    }
    body["authority_id"] = papr.PAPR_AUTHORITY_PREFIX + canonical_sha256(body)
    publish_immutable_json(authority_path, body)
    monkeypatch.setattr(lifecycle, "execution_commit_for", lambda _root, _path: "a" * 40)
    monkeypatch.setattr(lifecycle, "assert_authority_source_closure", lambda *_a, **_k: manifest)
    # The live active-epoch closure needs a real Git history; the synthetic
    # lifecycle tests exercise the authority contract and the terminal chain,
    # while the real repository test in tests/test_w10_source_epochs.py proves
    # the historical-v4/active-successor closure end to end.
    monkeypatch.setattr(lifecycle, "assert_active_epoch_closure", lambda *_a, **_k: None)
    return body


def _synthetic_chain(root: Path, authority: dict, *, initial_sha: str) -> dict:
    epochs = lifecycle.PAPR_EPOCHS
    runtime = root / str(authority["runtime_root"])
    config = load_papr_config()
    protocol = authority["protocol"]
    expected_initialization = {
        **fresh_initialization_identity(PAPR_TRAIN_SEED),
        "initial_model_state_sha256": initial_sha,
    }
    entries = []
    summaries = []
    previous_checkpoint = None
    previous_global_step = 0
    for epoch in range(epochs):
        lineage_body = {
            "protocol_version": papr.papr_protocol_version(),
            "source_commit": "a" * 40,
            "source_manifest_id": authority["source_manifest"]["manifest_id"],
            "source_manifest_sha256": authority["source_manifest"]["sha256"],
            "execution_image": "pascal-cu126-requirements-pascal-lock-v1",
            "campaign_id": PAPR_CAMPAIGN_ID,
            "run_id": PAPR_RUN_ID,
            "config_hash": authority["config_hash"],
            "protocol_config_hash": authority["protocol_config_hash"],
            "resolved_config": config.to_dict(),
            "execution_profile_id": "confessor_pascal_cu126",
            "gpu_uuid": authority["gpu_uuid"],
            "dataset": "imagenette160",
            "dataset_version": str(config.resolved["dataset_version"]),
            "split_manifest_hash": "9" * 64,
            "architecture": "djscc_residual_v1",
            "bw_ratio": "r_1_6",
            "k": papr.PAPR_K,
            "train_seed": PAPR_TRAIN_SEED,
            "channel_seed": PAPR_CHANNEL_SEED,
            "train_snr_db": protocol["train_snr_db"],
            "checkpoint_selection_snr_db": checkpoint_selection_snr_db(),
            "checkpoint_selection_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "lambda": 3.0,
            "recipe_sha256": "7" * 64,
            "initialization": expected_initialization,
            "predecessor_checkpoint_id": previous_checkpoint,
        }
        stable_ids = [f"stable-{index:04d}" for index in range(5)]
        record = {
            "schema_version": w8_training.W8_EPOCH_RECORD_SCHEMA_VERSION,
            "artifact_role": PAPR_EPOCH_ROLE,
            "eligibility": PAPR_ELIGIBILITY,
            "campaign_id": PAPR_CAMPAIGN_ID,
            "run_id": PAPR_RUN_ID,
            "lineage": lineage_body,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "samples": 5,
            "expected_samples": 5,
            "stable_id_count": 5,
            "stable_id_order": stable_ids,
            "stable_id_order_sha256": hashlib.sha256("\n".join(stable_ids).encode()).hexdigest(),
            "stable_id_set_sha256": hashlib.sha256("\n".join(sorted(stable_ids)).encode()).hexdigest(),
            "training_noise_id_count": 5,
            "training_noise_id_sha256": hashlib.sha256("\n".join(["n"] * 5).encode()).hexdigest(),
            "microbatches": 1,
            "expected_microbatches": 1,
            "final_physical_batch": 5,
            "optimizer_step_opportunities": 1,
            "optimizer_steps": 1,
            "grad_scaler_skips": 0,
            "global_optimizer_step": previous_global_step + 1,
            "lr": 1e-4,
            "total_loss": 1.0,
            "cross_entropy": 0.5,
            "reconstruction_mse": 0.1,
            "duration_seconds": 0.1,
            "finite_loss": True,
            "gradient_checks": {
                "optimizer_parameter_count": 1,
                "optimizer_gradient_count_min": 1,
                "optimizer_gradient_count_max": 1,
                "all_optimizer_gradients_finite": True,
                "all_named_present_gradients_finite": True,
                "last_named": {},
                "last_optimizer": {"parameter_count": 1, "gradient_count": 1, "present": True, "finite": True, "nonzero": True},
            },
            "training_noise_identity_digest": hashlib.sha256(b"noise").hexdigest(),
            "validation_noise_identity_rule": "run_channel_seed",
            "papr_checks": {
                "cap_db": papr.papr_cap_db(),
                "domain": w8_training.W8_PAPR_DOMAIN,
                "max_observed_db": 2.0,
                "bound_tolerance_db": w8_training.W8_PAPR_BOUND_TOLERANCE_DB,
                "constraint_installed": True,
                "compliant": True,
            },
        }
        record_id = canonical_sha256(record)
        stored_record = {**record, "record_id": record_id}
        publish_immutable_json(runtime / f"epochs/epoch-{epoch:04d}.json", stored_record)
        payload = {
            "schema_version": w8_training.W8_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": PAPR_CHECKPOINT_ROLE,
            "eligibility": PAPR_ELIGIBILITY,
            "campaign_id": PAPR_CAMPAIGN_ID,
            "run_id": PAPR_RUN_ID,
            "lineage": lineage_body,
            "execution_profile": {"gpu_uuid": authority["gpu_uuid"], "execution_profile_id": "confessor_pascal_cu126"},
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "global_optimizer_step": previous_global_step + 1,
            "accumulation_position": 0,
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "scaler_state": None,
            "rng_state_policy": {},
            "initialization": expected_initialization,
            "epoch_manifest": {
                "path": f"epochs/epoch-{epoch:04d}.json",
                "record_id": record_id,
                "record_sha256": hashlib.sha256(canonical_bytes(stored_record)).hexdigest(),
            },
            "predecessor_checkpoint_id": previous_checkpoint,
            "protected_counters": active_protected_counters(),
        }
        import io

        stream = io.BytesIO()
        torch.save(payload, stream)
        checkpoint_bytes = stream.getvalue()
        checkpoint_path = runtime / f"checkpoints/epoch-{epoch:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(checkpoint_bytes)
        checkpoint_id = hashlib.sha256(checkpoint_bytes).hexdigest()
        record_sha256 = hashlib.sha256(canonical_bytes(stored_record)).hexdigest()
        sidecar = {
            "schema_version": w8_training.W8_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": PAPR_SIDECAR_ROLE,
            "eligibility": PAPR_ELIGIBILITY,
            "campaign_id": PAPR_CAMPAIGN_ID,
            "run_id": PAPR_RUN_ID,
            "checkpoint_path": f"checkpoints/epoch-{epoch:04d}.pt",
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": len(checkpoint_bytes),
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "global_optimizer_step": previous_global_step + 1,
            "accumulation_position": 0,
            "config_hash": authority["config_hash"],
            "protocol_config_hash": authority["protocol_config_hash"],
            "source_commit": "a" * 40,
            "source_manifest_id": authority["source_manifest"]["manifest_id"],
            "source_manifest_sha256": authority["source_manifest"]["sha256"],
            "execution_image": "pascal-cu126-requirements-pascal-lock-v1",
            "execution_profile_id": "confessor_pascal_cu126",
            "gpu_uuid": authority["gpu_uuid"],
            "dataset": "imagenette160",
            "ratio": "r_1_6",
            "k": papr.PAPR_K,
            "lambda": 3.0,
            "train_seed": PAPR_TRAIN_SEED,
            "channel_seed": PAPR_CHANNEL_SEED,
            "train_snr_db": protocol["train_snr_db"],
            "checkpoint_selection_snr_db": checkpoint_selection_snr_db(),
            "checkpoint_selection_channel_seed_rule": "run_channel_seed",
            "predecessor_checkpoint_id": previous_checkpoint,
            "initialization": expected_initialization,
            "epoch_record_path": f"epochs/epoch-{epoch:04d}.json",
            "epoch_record_id": record_id,
            "epoch_record_sha256": record_sha256,
            "checkpoint_write_seconds": 0.1,
        }
        publish_immutable_json(runtime / f"checkpoints/epoch-{epoch:04d}.sidecar.json", sidecar)
        (runtime / "latest.json").unlink(missing_ok=True)
        publish_immutable_json(runtime / "latest.json", sidecar)
        summary = {
            "schema_version": 2,
            "artifact_role": papr.PAPR_VALIDATION_ROLE,
            "eligibility": PAPR_ELIGIBILITY,
            "campaign_id": PAPR_CAMPAIGN_ID,
            "run_id": PAPR_RUN_ID,
            "ratio": "r_1_6",
            "k": papr.PAPR_K,
            "train_seed": PAPR_TRAIN_SEED,
            "channel_seed": PAPR_CHANNEL_SEED,
            "checkpoint_id": checkpoint_id,
            "epoch": epoch,
            "validation_split": "val",
            "validation_order": "stable_manifest_order",
            "validation_augmentation": False,
            "validation_batch_size": 5,
            "validation_snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "validation_snr_resolution": "params.channel.train_snr_db_fixed",
            "validation_snr_db": checkpoint_selection_snr_db(),
            "validation_channel_seed_rule": "run_channel_seed",
            "validation_channel_seed": PAPR_CHANNEL_SEED,
            "validation_noise_policy": "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs",
            "validation_noise_id_digest": hashlib.sha256(b"fixed-noise").hexdigest(),
            "validation_noise_id_count": 1000,
            "n_correct": 400 + (100 if epoch == 3 else 0),
            "n_total": 1000,
            "top1_accuracy": (400 + (100 if epoch == 3 else 0)) / 1000,
            "prediction_digest": hashlib.sha256(f"pred-{epoch}".encode()).hexdigest(),
            "row_digest": hashlib.sha256(f"row-{epoch}".encode()).hexdigest(),
            "evaluation_config_hash": "0" * 64,
            "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"],
            "test_model_facing_access": 0,
            "papr_checks": {
                "cap_db": papr.papr_cap_db(),
                "domain": w8_training.W8_PAPR_DOMAIN,
                "max_observed_db": 1.5,
                "bound_tolerance_db": w8_training.W8_PAPR_BOUND_TOLERANCE_DB,
                "constraint_installed": True,
                "compliant": True,
            },
        }
        summary["summary_id"] = canonical_sha256(summary)
        # The terminal verifier recomputes evaluation_config_hash from the
        # authority protocol hash; build the exact expected value.
        expected_eval_hash = canonical_sha256(
            {
                "protocol_config_hash": authority["protocol_config_hash"],
                "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
                "snr_db": checkpoint_selection_snr_db(),
                "channel_seed": PAPR_CHANNEL_SEED,
                "batch_size": 5,
                "order": "stable_manifest_order",
            }
        )
        summary["evaluation_config_hash"] = expected_eval_hash
        summary["summary_id"] = canonical_sha256({key: value for key, value in summary.items() if key != "summary_id"})
        publish_immutable_json(runtime / f"validation/epoch-{epoch:04d}.json", summary)
        summaries.append(summary)
        entries.append(
            {
                "epoch": epoch,
                "epoch_record_sha256": record_sha256,
                "checkpoint_id": checkpoint_id,
                "global_optimizer_step": previous_global_step + 1,
            }
        )
        previous_checkpoint = checkpoint_id
        previous_global_step += 1
    selection = select_checkpoint_epoch(summaries, expected_epochs=epochs, namespace=lifecycle.PAPR_VALIDATION_NAMESPACE)
    chain = {
        "entries": entries,
        "epoch_chain_digest": lifecycle.epoch_chain_digest(entries),
        "optimizer_steps": previous_global_step,
        "grad_scaler_skips": 0,
        "optimizer_step_opportunities": previous_global_step,
        "global_optimizer_step": previous_global_step,
        "max_observed_papr_db": 2.0,
        "initialization": expected_initialization,
    }
    selected_epoch = int(selection["selected_epoch"])
    sidecar = json.loads((runtime / f"checkpoints/epoch-{selected_epoch:04d}.sidecar.json").read_bytes())
    selected, completion = lifecycle.publish_selected_and_completion(
        root,
        authority=authority,
        authority_path=root / "results/learned/w10/papr_training_authorization.json",
        execution_commit="a" * 40,
        chain=chain,
        summaries=summaries,
        selection=selection,
        selected_sidecar=sidecar,
        checkpoint_relative=f"{authority['runtime_root']}/checkpoints/epoch-{selected_epoch:04d}.pt",
        checkpoint_bytes=(runtime / f"checkpoints/epoch-{selected_epoch:04d}.pt").stat().st_size,
    )
    return {
        "selection": selected,
        "completion": completion,
        "summaries": summaries,
        "chain": chain,
        "selection_input": selection,
        "selected_sidecar": sidecar,
        "checkpoint_relative": f"{authority['runtime_root']}/checkpoints/epoch-{selected_epoch:04d}.pt",
        "checkpoint_bytes": (runtime / f"checkpoints/epoch-{selected_epoch:04d}.pt").stat().st_size,
        "execution_commit": "a" * 40,
        "authority_path": root / "results/learned/w10/papr_training_authorization.json",
    }


def _republish_synthetic_terminal(root: Path, authority: dict, result: dict) -> tuple[dict, dict]:
    return lifecycle.publish_selected_and_completion(
        root,
        authority=authority,
        authority_path=result["authority_path"],
        execution_commit=result["execution_commit"],
        chain=result["chain"],
        summaries=result["summaries"],
        selection=result["selection_input"],
        selected_sidecar=result["selected_sidecar"],
        checkpoint_relative=result["checkpoint_relative"],
        checkpoint_bytes=result["checkpoint_bytes"],
    )


def test_papr_selected_to_completion_crash_window_is_recoverable_and_idempotent(
    tmp_path: Path, monkeypatch, _bounded_counts
) -> None:
    _bounded_lifecycle_constants(monkeypatch)
    authority = _synthetic_authority(tmp_path, monkeypatch)
    result = _synthetic_chain(tmp_path, authority, initial_sha="d" * 64)
    completion_path = tmp_path / lifecycle.PAPR_COMPLETION_PATH
    completion_path.unlink()

    selected, completion = _republish_synthetic_terminal(tmp_path, authority, result)
    assert selected == result["selection"]
    assert completion == result["completion"]
    assert completion_path.is_file()
    assert _republish_synthetic_terminal(tmp_path, authority, result) == (selected, completion)
    verified = lifecycle.verify_papr_terminal(
        tmp_path,
        authority_path=result["authority_path"],
        expected_initial_state_sha256="d" * 64,
    )
    assert verified["completion_id"] == completion["completion_id"]


def test_papr_terminal_publication_holds_on_completion_without_selected_or_mismatch(
    tmp_path: Path, monkeypatch, _bounded_counts
) -> None:
    _bounded_lifecycle_constants(monkeypatch)
    authority = _synthetic_authority(tmp_path, monkeypatch)
    result = _synthetic_chain(tmp_path, authority, initial_sha="d" * 64)
    selected_path = tmp_path / lifecycle.PAPR_SELECTED_CHECKPOINT_PATH
    completion_path = tmp_path / lifecycle.PAPR_COMPLETION_PATH

    selected_path.unlink()
    with pytest.raises(lifecycle.PaprLifecycleHold, match="completion exists without"):
        _republish_synthetic_terminal(tmp_path, authority, result)

    # Recreate the valid selected record for the independent mismatch check.
    publish_immutable_json(selected_path, result["selection"])
    mutated = dict(result["selection"])
    mutated["selected_epoch"] = int(mutated["selected_epoch"]) + 1
    mutated["selection_id"] = lifecycle.PAPR_SELECTED_PREFIX + canonical_sha256(
        {key: value for key, value in mutated.items() if key != "selection_id"}
    )
    selected_path.write_bytes(canonical_bytes(mutated))
    with pytest.raises(lifecycle.PaprLifecycleHold, match="selected checkpoint differs"):
        _republish_synthetic_terminal(tmp_path, authority, result)


@pytest.mark.parametrize(
    "mutation",
    (
        "config_hash",
        "protocol_config_hash",
        "protocol_cap",
        "protocol_recipe",
        "source_binding",
        "cuda_mapping_missing",
        "cuda_mapping_uuid",
        "cuda_mapping_device",
        "cuda_mapping_name",
        "cuda_mapping_compute",
        "cuda_mapping_count",
    ),
)
def test_papr_authority_recomputes_projected_config_protocol_and_source(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    authority = _synthetic_authority(tmp_path, monkeypatch)
    path = tmp_path / lifecycle.PAPR_AUTHORITY_PATH
    value = json.loads(path.read_bytes())
    if mutation == "config_hash":
        value["config_hash"] = "0" * 64
    elif mutation == "protocol_config_hash":
        value["protocol_config_hash"] = "0" * 64
    elif mutation == "protocol_cap":
        value["protocol"]["papr_cap_db"] = 99.0
    elif mutation == "protocol_recipe":
        value["protocol"]["recipe"] = "unauthorized_recipe"
    elif mutation == "source_binding":
        value["source_binding"] = {**value["source_binding"], "source_commit": "0" * 40}
    elif mutation == "cuda_mapping_missing":
        value.pop("cuda_mapping")
    else:
        mapping = dict(value["cuda_mapping"])
        field = {
            "cuda_mapping_uuid": "cuda0_gpu_uuid",
            "cuda_mapping_device": "logical_device",
            "cuda_mapping_name": "cuda0_gpu_name",
            "cuda_mapping_compute": "cuda0_compute_capability",
            "cuda_mapping_count": "device_count",
        }[mutation]
        mapping[field] = {
            "cuda0_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "logical_device": "cuda:1",
            "cuda0_gpu_name": "NVIDIA TITAN Xp",
            "cuda0_compute_capability": "7.5",
            "device_count": 2,
        }[field]
        value["cuda_mapping"] = mapping
    body = dict(value)
    body.pop("authority_id", None)
    value["authority_id"] = papr.PAPR_AUTHORITY_PREFIX + canonical_sha256(body)
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(lifecycle.PaprLifecycleHold):
        lifecycle.verify_papr_authority(tmp_path, authority_path=path)


def test_papr_terminal_verifier_recomputes_and_rejects_mutations(
    tmp_path: Path, monkeypatch, _bounded_counts
) -> None:
    _bounded_lifecycle_constants(monkeypatch)
    authority = _synthetic_authority(tmp_path, monkeypatch)
    result = _synthetic_chain(tmp_path, authority, initial_sha="d" * 64)

    verified = lifecycle.verify_papr_terminal(
        tmp_path,
        authority_path=tmp_path / "results/learned/w10/papr_training_authorization.json",
        expected_initial_state_sha256="d" * 64,
    )
    assert verified["selected_epoch"] == result["selection"]["selected_epoch"]
    assert verified["completion_id"] == result["completion"]["completion_id"]
    assert verified["test_access"] == 0

    # Mutations must fail closed.
    completion_path = tmp_path / lifecycle.PAPR_COMPLETION_PATH
    original = json.loads(completion_path.read_bytes())
    mutated = dict(original)
    mutated["training_runs"] = 2
    mutated["completion_id"] = lifecycle.PAPR_COMPLETION_PREFIX + canonical_sha256(
        {key: value for key, value in mutated.items() if key != "completion_id"}
    )
    completion_path.write_bytes(json.dumps(mutated, sort_keys=True).encode())
    with pytest.raises(lifecycle.PaprLifecycleHold, match="completion record differs"):
        lifecycle.verify_papr_terminal(
            tmp_path,
            authority_path=tmp_path / "results/learned/w10/papr_training_authorization.json",
            expected_initial_state_sha256="d" * 64,
        )
    completion_path.write_bytes(canonical_bytes(original))

    record_path = tmp_path / authority["runtime_root"] / f"epochs/epoch-{lifecycle.PAPR_EPOCHS - 1:04d}.json"
    record = json.loads(record_path.read_bytes())
    record["optimizer_steps"] = 0
    record["record_id"] = canonical_sha256({key: value for key, value in record.items() if key != "record_id"})
    record_path.write_bytes(json.dumps(record, sort_keys=True).encode())
    with pytest.raises(lifecycle.PaprLifecycleHold):
        lifecycle.verify_papr_terminal(
            tmp_path,
            authority_path=tmp_path / "results/learned/w10/papr_training_authorization.json",
            expected_initial_state_sha256="d" * 64,
        )


def test_papr_execution_commit_closure_requires_protected_source_equality(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    for name in ("src", "tools", "configs", "spec", "tests"):
        (repo / name).mkdir()
        (repo / name / "placeholder.txt").write_text(f"{name}\n")
    (repo / "src/example.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repo, check=True)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest = _synthetic_manifest(repo)
    from runtime.source_guard import git_tree_hashes

    manifest["source_commit"] = source_commit
    manifest["tree_hashes"] = git_tree_hashes(repo, source_commit)
    monkeypatch.setattr(lifecycle, "load_w10_manifest", lambda _root, live=True, epoch=None: manifest)

    authority_path = repo / "evidence/authority.json"
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text("{}\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "evidence"], cwd=repo, check=True)
    execution_commit = lifecycle.execution_commit_for(repo, authority_path)
    authority = {
        "source_binding": manifest,
        "source_manifest": source_record(repo, manifest),
        "source_commit": source_commit,
    }
    assert lifecycle.assert_authority_source_closure(repo, authority, execution_commit) == manifest

    # A protected change carried by the authority's own execution commit is refused.
    (repo / "src/example.py").write_text("value = 2\n")
    authority_path.write_text('{"changed": true}\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "protected change"], cwd=repo, check=True)
    new_execution = lifecycle.execution_commit_for(repo, authority_path)
    assert new_execution != execution_commit
    with pytest.raises(lifecycle.PaprLifecycleHold, match="protected source"):
        lifecycle.assert_authority_source_closure(repo, authority, new_execution)


def test_w10_papr_binding_and_dispatch_agree(tmp_path: Path, monkeypatch, _bounded_counts) -> None:
    _bounded_lifecycle_constants(monkeypatch)
    authority = _synthetic_authority(tmp_path, monkeypatch)
    _synthetic_chain(tmp_path, authority, initial_sha="d" * 64)
    # The binding resolver runs the full terminal verifier; the synthetic chain
    # already exercised it, so only the binding schema agreement remains.
    terminal = lifecycle.verify_papr_terminal(
        tmp_path,
        authority_path=tmp_path / "results/learned/w10/papr_training_authorization.json",
        expected_initial_state_sha256="d" * 64,
    )
    monkeypatch.setattr(lifecycle, "verify_papr_terminal", lambda _root, authority_path=None: terminal)

    from evaluation import w10_bindings

    record = w10_bindings._papr_checkpoint(tmp_path)
    assert record["state"] == "FROZEN"
    assert record["checkpoint_id"] == terminal["checkpoint_id"]
    assert record["checkpoint_path"].endswith(".pt")
    assert record["selection_id"] == terminal["selection_id"]
    # The learned dispatch route consumes exactly these keys.
    selected = json.loads((tmp_path / w10_bindings.PAPR_SELECTED_CHECKPOINT).read_bytes())
    assert selected["checkpoint_id"] == record["checkpoint_id"]
    assert selected["checkpoint_path"] == record["checkpoint_path"]
