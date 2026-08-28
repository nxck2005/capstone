"""Detached W7 campaign/recovery state-machine regressions without science."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluation.w7_validation as validation
import run_w7_campaign as campaign
import training.w7_g4 as w7_training
from adjudication.w7_g4 import CANDIDATE_ELIGIBILITY
from config.run_config import config_hash as run_config_hash
from training.deterministic_core import canonical_sha256
from training.w7_protocol import (
    W7_EXECUTION_IMAGE_FAMILY,
    W7_LAMBDA_GRID,
    W7_SELECTED_GPU_UUID,
)
from tests.w7_hardening_fixtures import (
    TinyDJSCC,
    TinyW7Dataset,
    fake_validation_summary,
    lineage,
    profile_binding,
    tiny_config,
    validation_get,
)


@pytest.fixture
def campaign_fixture(monkeypatch):
    actual_trainer = w7_training.W7Trainer
    actual_get = campaign.get
    monkeypatch.setattr(campaign, "get", validation_get(actual_get, 5))
    monkeypatch.setattr(validation, "get", validation_get(validation.get, 5))
    monkeypatch.setattr(w7_training, "build_djscc", lambda *_args, **_kwargs: TinyDJSCC())

    def trainer_factory(config, *, runtime_root, source_lineage, profile_binding, policy, **_kwargs):
        return actual_trainer(
            config,
            device="cpu",
            runtime_root=runtime_root,
            source_lineage=source_lineage,
            profile_binding=profile_binding,
            policy=policy,
            model=TinyDJSCC(),
            num_workers=0,
        )

    monkeypatch.setattr(campaign, "W7Trainer", trainer_factory)
    monkeypatch.setattr(campaign, "validate_candidate", lambda value: dict(value))
    import data.djscc_training as training_data

    monkeypatch.setattr(
        training_data,
        "TrainingDJSCCDataset",
        lambda _dataset, _seed, epoch, repo_root=None: TinyW7Dataset(epoch),
    )

    def evaluate(trainer, *, checkpoint_id, **_kwargs):
        return SimpleNamespace(summary=fake_validation_summary(trainer, checkpoint_id), rows=())

    monkeypatch.setattr(campaign, "evaluate_validation", evaluate)

    def select(summaries, *, expected_epochs):
        return validation.select_checkpoint_epoch(summaries, expected_epochs=expected_epochs)

    monkeypatch.setattr(campaign, "select_checkpoint_epoch", select)

    def selected(trainer, *, selection, **_kwargs):
        sidecar = trainer.load_checkpoint_epoch(selection["selected_epoch"])
        summaries = campaign._load_epoch_summaries(trainer.runtime_root)
        body = {
            "schema_version": 1,
            "artifact_role": "W7_SELECTED_CHECKPOINT_VALIDATION_RESULT",
            "checkpoint_id": sidecar["checkpoint_id"],
            "checkpoint_epoch": selection["selected_epoch"],
            "selection": dict(selection),
            "calibration_validation": summaries[selection["selected_epoch"]],
            "calibration_rows": [],
            "psnr_evaluation": {
                "checkpoint_id": sidecar["checkpoint_id"],
                "snr_db": 15,
                "denominator": 5,
                "data_range": 1.0,
                "psnr_definition": "fixture",
                "psnr_db": 21.0,
                "papr_definition": "fixture",
                "papr_db": 0.0,
                "per_image_digest": "9" * 64,
                "per_image": [],
            },
            "protected_counters": {
                "w7_candidate_results": 0,
                "learned_test_inference": 0,
                "test_model_facing_access": 0,
            },
        }
        body["result_digest"] = canonical_sha256(body)
        return body

    monkeypatch.setattr(campaign, "selected_checkpoint_result", selected)
    return SimpleNamespace(trainer_factory=trainer_factory)


def _candidate_args(root: Path, config):
    return {
        "config": config,
        "root": root,
        "source_lineage": lineage(),
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "profile_binding": profile_binding(config),
        "repo_root": None,
        "heartbeat": root.parent / "heartbeat.json",
        "campaign_id": "w7-test-campaign",
    }


def test_lambda_sequence_is_exact_and_candidates_are_w8_ineligible():
    assert list(W7_LAMBDA_GRID) == [0.0, 0.1, 0.3, 1.0, 3.0]
    assert CANDIDATE_ELIGIBILITY["w8_eligibility"] == "NOT_ELIGIBLE_FOR_W8_INITIALIZATION"
    config = tiny_config(epochs=1)
    assert config.resolved["w8_eligibility"] == "NOT_ELIGIBLE_FOR_W8_INITIALIZATION"


def test_completed_candidate_is_authenticated_and_skipped_not_rerun(
    tmp_path: Path, monkeypatch, campaign_fixture
):
    config = tiny_config(epochs=2)
    root = tmp_path / "lambda"
    first = campaign._candidate(**_candidate_args(root, config))
    assert first["status"] == "COMPLETE"
    assert first["eligibility"]["w8_eligibility"] == "NOT_ELIGIBLE_FOR_W8_INITIALIZATION"

    import data.djscc_training as training_data

    monkeypatch.setattr(
        training_data,
        "TrainingDJSCCDataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("completed candidate reran training")),
    )
    second = campaign._candidate(**_candidate_args(root, config))
    assert second == first

    checkpoint = torch.load(root / "checkpoints/epoch-0001.pt", map_location="cpu", weights_only=False)
    assert checkpoint["eligibility"]["w8_eligibility"] == "NOT_ELIGIBLE_FOR_W8_INITIALIZATION"
    assert checkpoint["artifact_role"] == "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT"


def test_incomplete_candidate_resumes_exact_latest_checkpoint(
    tmp_path: Path, monkeypatch, campaign_fixture
):
    config = tiny_config(epochs=2)
    root = tmp_path / "resume"
    trainer = campaign_fixture.trainer_factory(
        config,
        runtime_root=root,
        source_lineage=lineage(),
        profile_binding=profile_binding(config),
        policy=campaign.W7_G4_PILOT_POLICY,
    )
    record = trainer.train_epoch(0, TinyW7Dataset(0))
    first_sidecar = trainer.save_checkpoint(record)
    campaign._publish_validation(
        root,
        SimpleNamespace(summary=fake_validation_summary(trainer, first_sidecar["checkpoint_id"])),
    )

    trained_epochs: list[int] = []
    import data.djscc_training as training_data

    def dataset(_dataset, _seed, epoch, **_kwargs):
        trained_epochs.append(epoch)
        return TinyW7Dataset(epoch)

    monkeypatch.setattr(training_data, "TrainingDJSCCDataset", dataset)
    candidate = campaign._candidate(**_candidate_args(root, config))
    assert candidate["status"] == "COMPLETE"
    assert trained_epochs == [1]
    latest = json.loads((root / "latest.json").read_bytes())
    assert latest["predecessor_checkpoint_id"] == first_sidecar["checkpoint_id"]


def test_checkpoint_without_validation_replays_only_validation(
    tmp_path: Path, monkeypatch, campaign_fixture
):
    config = tiny_config(epochs=1)
    root = tmp_path / "validation-only"
    trainer = campaign_fixture.trainer_factory(
        config,
        runtime_root=root,
        source_lineage=lineage(),
        profile_binding=profile_binding(config),
        policy=campaign.W7_G4_PILOT_POLICY,
    )
    record = trainer.train_epoch(0, TinyW7Dataset(0))
    trainer.save_checkpoint(record)

    import data.djscc_training as training_data

    monkeypatch.setattr(
        training_data,
        "TrainingDJSCCDataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("authenticated epoch was rerun")),
    )
    candidate = campaign._candidate(**_candidate_args(root, config))
    assert candidate["status"] == "COMPLETE"
    assert (root / "validation/epoch-0000.json").is_file()


def test_corrupt_latest_and_foreign_nonempty_state_hold(
    tmp_path: Path, campaign_fixture
):
    config = tiny_config(epochs=2)
    root = tmp_path / "corrupt"
    trainer = campaign_fixture.trainer_factory(
        config,
        runtime_root=root,
        source_lineage=lineage(),
        profile_binding=profile_binding(config),
        policy=campaign.W7_G4_PILOT_POLICY,
    )
    record = trainer.train_epoch(0, TinyW7Dataset(0))
    sidecar = trainer.save_checkpoint(record)
    (root / sidecar["checkpoint_path"]).write_bytes(b"truncated")
    with pytest.raises(w7_training.W7Hold, match="byte length|SHA-256"):
        campaign._candidate(**_candidate_args(root, config))

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "foreign.json").write_text("{}", encoding="ascii")
    with pytest.raises(RuntimeError, match="no authenticated latest"):
        campaign._candidate(**_candidate_args(foreign, config))


def test_completed_candidate_resigned_source_gpu_lambda_seed_mismatches_hold(
    tmp_path: Path, campaign_fixture
):
    config = tiny_config(epochs=1)
    root = tmp_path / "mutations"
    campaign._candidate(**_candidate_args(root, config))
    original = json.loads((root / "candidate_completion.json").read_bytes())
    mutations = {
        "source": lambda value: value["lineage"].__setitem__("source_commit", "d" * 40),
        "gpu": lambda value: value["lineage"].__setitem__("gpu_uuid", "GPU-foreign"),
        "lambda": lambda value: value.__setitem__("lambda", 1.0),
        "seed": lambda value: value["lineage"].__setitem__("train_seed", 3),
    }
    for label, mutate in mutations.items():
        value = json.loads(json.dumps(original))
        mutate(value)
        (root / "candidate_completion.json").write_bytes(campaign.canonical_bytes(value))
        with pytest.raises(RuntimeError, match="lambda|lineage"):
            campaign._candidate(**_candidate_args(root, config))
    (root / "candidate_completion.json").write_bytes(campaign.canonical_bytes(original))


def test_execution_authorization_role_mutation_holds_after_resigning(tmp_path: Path):
    completion = json.loads(campaign.W7_A_COMPLETION_PATH.read_bytes())
    body = {
        "schema_version": 1,
        "artifact_role": "FOREIGN_AUTHORIZATION_ROLE",
        "status": "AUTHORIZED",
        "campaign_id": "fixture",
        "source_commit": "a" * 40,
        "source_manifest_id": "w7source-fixture",
        "source_manifest_sha256": "b" * 64,
        "profile_freeze_id": "freeze-fixture",
        "profile_freeze_sha256": "c" * 64,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "lambda_grid": list(W7_LAMBDA_GRID),
        "lambda_order": "exact_configured_lambda_grid_order",
        "w7_a_completion_id": completion["completion_id"],
        "w7_a_completion_sha256": hashlib.sha256(campaign.W7_A_COMPLETION_PATH.read_bytes()).hexdigest(),
        "scientific_execution_authorization": "PRESENT",
        "test_access": "SEALED",
    }
    value = dict(body)
    value["authorization_id"] = "w7auth-" + canonical_sha256(body)
    path = tmp_path / "authorization.json"
    path.write_bytes(campaign.canonical_bytes(value))
    with pytest.raises(RuntimeError, match="not active"):
        campaign._load_authorization(path)


def test_orchestrator_is_sequential_and_completes_not_adjudicated(
    tmp_path: Path, monkeypatch
):
    source_commit = "a" * 40
    source = {
        "manifest_id": "w7source-fixture",
        "source_commit": source_commit,
    }
    source_path = tmp_path / "source.json"
    source_path.write_bytes(campaign.canonical_bytes(source))
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    freeze = {
        "status": "FROZEN",
        "profile_freeze_id": "freeze-fixture",
        "execution_profile_id": "confessor_pascal_cu126",
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "physical_batch_size": 32,
        "accumulation_factor": 1,
        "validation_batch_size": 32,
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_bytes(campaign.canonical_bytes(freeze))
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    authorization = {
        "authorization_id": "auth-fixture",
        "campaign_id": "campaign-fixture",
        "source_commit": source_commit,
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": source_sha,
        "profile_freeze_id": freeze["profile_freeze_id"],
        "profile_freeze_sha256": freeze_sha,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "w7_a_completion_id": "completion-fixture",
    }
    monkeypatch.setattr(campaign, "_load_authorization", lambda _path: authorization)
    monkeypatch.setattr(campaign, "verify_source_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(campaign, "verify_profile_freeze", lambda value: value)
    monkeypatch.setattr(campaign, "_verify_upstream", lambda: None)
    monkeypatch.setattr(campaign, "authenticate_w7_gpu", lambda **kwargs: {"config_hash": kwargs["config_hash"]})
    monkeypatch.setattr(campaign, "verify_frozen_gpu_binding", lambda *_args, **_kwargs: None)

    class _RunResult:
        def __init__(self, stdout=""):
            self.stdout = stdout

    calls = iter([_RunResult(source_commit + "\n"), _RunResult("")])
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_args, **_kwargs: next(calls))

    class _Lock:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(campaign, "W7CampaignLock", _Lock)
    active = 0
    maximum_active = 0
    observed: list[float] = []

    def candidate(*, config, **_kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        observed.append(float(config.resolved["lambda"]))
        active -= 1
        return {"lambda": float(config.resolved["lambda"])}

    monkeypatch.setattr(campaign, "_candidate", candidate)
    root = tmp_path / "campaign"
    args = argparse.Namespace(
        authorization=tmp_path / "auth.json",
        source_manifest=source_path,
        profile_freeze=freeze_path,
        campaign_id="campaign-fixture",
        campaign_root=root,
        heartbeat=tmp_path / "heartbeat.json",
        gpu_uuid=W7_SELECTED_GPU_UUID,
        repo_root=None,
        execution_image=W7_EXECUTION_IMAGE_FAMILY,
    )
    assert campaign.run(args) == 0
    assert observed == list(W7_LAMBDA_GRID)
    assert maximum_active == 1
    completion = json.loads((root / "campaign_completion.json").read_bytes())
    assert completion["status"] == "COMPLETE_NOT_ADJUDICATED"
    assert completion["candidate_lambdas"] == list(W7_LAMBDA_GRID)
    assert completion["g4_adjudication_run"] == 0
    assert completion["lambda_core_updated"] is False
