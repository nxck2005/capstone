"""Focused integration contracts for the actual prospective W9 v4 path."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

import config.execution_profiles as execution_profiles
import runtime.transactional_epochs as transactional_epochs
from config.run_config import config_hash, load_experiment
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates
from runtime.source_guard import (
    ALLOWED_EVIDENCE_PREFIXES,
    PROTECTED_PREFIXES,
    SourceGuardHold,
    V4_RELEVANT_CONFIG_PATHS,
    V4_WORKING_TREE_GUARD,
    assert_v4_manifest_contract,
    build_manifest,
)
from runtime.transactional_epochs import TransactionalEpochStore
from training.er9_v4 import ER9V4CandidateTrainer


REPO = Path(__file__).resolve().parents[1]


def _publish(store: TransactionalEpochStore, epoch: int) -> None:
    store.publish_epoch(
        epoch,
        f"checkpoint-{epoch}".encode("ascii"),
        {
            "optimizer_opportunities": 1,
            "applied_optimizer_steps": 1,
            "grad_scaler_skips": 0,
            "validation_n_correct": epoch,
        },
    )


def test_active_stage1_entrypoint_is_v4_only() -> None:
    text = (REPO / "tools/run_er9_campaign.py").read_text(encoding="utf-8")
    assert "ER9V4CandidateTrainer" in text
    assert "ER2V4RandomizedTrainer" in text
    assert "authenticate_live_w9_pascal" in text
    assert "resolve_runtime_root(REPO, authority)" in text
    assert "ER9V4" in text
    for forbidden in (
        "local_4060_cu130",
        "checkpoints/er9_successor_v2",
        "er_execution_source_manifest_v3.json",
        "er9_stage1_execution_authorization_v3.json",
        "configs/er9-digital.yaml",
        "ER9Trainer",
        "run_completion.json",
        "selected_checkpoint.json",
        "checkpoint.sidecar",
    ):
        assert forbidden not in text


def test_active_randomized_er2_entrypoint_is_authority_bound_v4_only() -> None:
    text = (REPO / "tools/run_er9_campaign.py").read_text(encoding="utf-8")
    assert 'ER2_AUTHORITY = REPO / "results/learned/er2_randomized/er2_execution_authorization_v4.json"' in text
    assert '"checkpoints/er2_randomized_pascal_v4"' in text
    assert "ER2V4RandomizedTrainer" in text
    for forbidden in ("ER2RandomizedTrainer", "latest.json", "checkpoint.sidecar", "selected_checkpoint.json"):
        assert forbidden not in text


def test_transactional_authentication_is_linear_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"run_id": "linear", "source_binding": "fixture", "config_hash": "a" * 64}
    store = TransactionalEpochStore(tmp_path / "runtime", identity=identity, total_epochs=13, role="W9_LINEAR")
    for epoch in range(12):
        _publish(store, epoch)

    restarted = TransactionalEpochStore(tmp_path / "runtime", identity=identity, total_epochs=13, role="W9_LINEAR")
    checkpoint_reads = 0
    original_read_bytes = transactional_epochs.Path.read_bytes

    def counted_read(path: Path) -> bytes:
        nonlocal checkpoint_reads
        if path.name == "checkpoint.pt":
            checkpoint_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(transactional_epochs.Path, "read_bytes", counted_read)
    assert len(restarted.inspect()) == 12
    assert restarted.last_inspect_authenticated_count == 12
    assert checkpoint_reads == 12

    checkpoint_reads = 0
    _publish(restarted, 12)
    assert checkpoint_reads == 1
    assert restarted.last_inspect_authenticated_count == 1

    checkpoint_reads = 0
    (tmp_path / "runtime" / "epochs" / "epoch-0000" / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(transactional_epochs.TransactionalRuntimeHold, match="checkpoint hash"):
        restarted.inspect()
    assert checkpoint_reads <= 13


def test_cuda_visible_uuid_contract_maps_frozen_uuid_to_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    uuid = "GPU-fixture"

    class Properties:
        pass

    properties = Properties()
    properties.uuid = uuid
    properties.name = "NVIDIA TITAN Xp"
    properties.major = 6
    properties.minor = 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    mapping = execution_profiles.authenticate_cuda_visible_mapping(
        expected_gpu_uuid=uuid,
        expected_gpu_name="NVIDIA TITAN Xp",
        expected_compute_capability="6.1",
    )
    assert mapping["logical_device"] == "cuda:0"
    assert mapping["cuda0_gpu_uuid"] == uuid
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(execution_profiles.ProfileAuthenticationError, match="CUDA_VISIBLE_DEVICES"):
        execution_profiles.authenticate_cuda_visible_mapping(expected_gpu_uuid=uuid)


def _manifest_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "manifest-fixture"
    for directory in ("src", "tools", "configs", "spec", "tests", "results/learned/er9"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/protected.py").write_text("value = 1\n", encoding="utf-8")
    (root / "tools/runner.py").write_text("value = 2\n", encoding="utf-8")
    (root / "configs/pascal.yaml").write_text("choice: pascal\n", encoding="utf-8")
    (root / "spec/frozen.md").write_text("frozen\n", encoding="utf-8")
    (root / "tests/contract.py").write_text("value = 3\n", encoding="utf-8")
    (root / "requirements-pascal.lock").write_text("fixture-lock\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "W9 fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return root, commit


def test_source_manifest_rejects_hybrid_or_dirty_source_images(tmp_path: Path) -> None:
    root, commit = _manifest_fixture(tmp_path)
    manifest = build_manifest(root, source_commit=commit, relevant_config_paths=("configs/pascal.yaml",))
    assert manifest["source_commit"] == commit
    (root / "configs/pascal.yaml").write_text("choice: changed\n", encoding="utf-8")
    with pytest.raises(SourceGuardHold, match="clean protected worktree"):
        build_manifest(root, source_commit=commit, relevant_config_paths=("configs/pascal.yaml",))
    subprocess.run(["git", "add", "configs/pascal.yaml"], cwd=root, check=True)
    with pytest.raises(SourceGuardHold, match="clean protected worktree"):
        build_manifest(root, source_commit=commit, relevant_config_paths=("configs/pascal.yaml",))
    subprocess.run(["git", "restore", "--staged", "configs/pascal.yaml"], cwd=root, check=True)
    (root / "src/untracked.py").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(SourceGuardHold, match="clean protected worktree"):
        build_manifest(root, source_commit=commit, relevant_config_paths=("configs/pascal.yaml",))


def test_v4_manifest_contract_has_one_exact_protected_source_image() -> None:
    manifest = {
        "schema_version": 2,
        "manifest_kind": "W9_V4_FULL_SCIENTIFIC_SOURCE_CLOSURE",
        "source_commit": "a" * 40,
        "source_commit_comparison": "exact_clean_HEAD_at_freeze",
        "tree_hashes": {name: "b" * 40 for name in ("repository", "src", "tools", "configs", "spec", "tests")},
        "requirements_pascal_lock_sha256": "c" * 64,
        "relevant_config_sha256": {path: "d" * 64 for path in V4_RELEVANT_CONFIG_PATHS},
        "protected_source_prefixes": list(PROTECTED_PREFIXES),
        "allowed_evidence_runtime_prefixes": list(ALLOWED_EVIDENCE_PREFIXES),
        "working_tree_guard": V4_WORKING_TREE_GUARD,
    }
    assert_v4_manifest_contract(manifest)
    manifest["relevant_config_sha256"].pop(V4_RELEVANT_CONFIG_PATHS[-1])
    with pytest.raises(SourceGuardHold, match="relevant config closure"):
        assert_v4_manifest_contract(manifest)


def test_stage1_arithmetic_is_solver_derived_exactly() -> None:
    config = load_experiment("configs/er9-digital-pascal-v4.yaml", train_seed=0, channel_seed=0)
    floor = packetisation_floor(int(config.resolved["k"]))
    assert floor.payload_bits == 4248
    assert len(all_configured_pairs()) == 32
    assert len(feasible_pairs(floor.payload_bits)) == 19
    assert [item.as_dict() for item in stage1_candidates(floor.payload_bits)] == [
        {"transmit_dim": 64, "quantiser_bits": 2},
        {"transmit_dim": 128, "quantiser_bits": 2},
        {"transmit_dim": 256, "quantiser_bits": 2},
        {"transmit_dim": 512, "quantiser_bits": 2},
        {"transmit_dim": 1024, "quantiser_bits": 2},
        {"transmit_dim": 2048, "quantiser_bits": 2},
    ]


def test_smoke_script_has_only_synthetic_model_path() -> None:
    text = (REPO / "tools/run_w9_pascal_lifecycle_smoke.py").read_text(encoding="utf-8")
    assert "ER9V4CandidateTrainer" in text
    assert "synthetic_provider" in text
    assert "W9V4SyntheticFixtureTrainer" not in text
    assert "authenticate_live_w9_pascal" in text
    assert "checkpoints/smoke/" in text
    assert "NON_SCIENTIFIC" in text and "TEST_NOT_ACCESSED" in text
    assert "DataLoader" not in text
    assert "TrainingDJSCCDataset" not in text
    assert "ValidationDJSCCDataset" not in text


def test_prospective_er9_trainer_accepts_only_injected_synthetic_fixture(tmp_path: Path) -> None:
    config = load_experiment("configs/er9-digital-pascal-v4.yaml", train_seed=0, channel_seed=0)

    def provider(epoch: int, _device: torch.device):
        generator = torch.Generator(device="cpu").manual_seed(9400 + epoch)
        return (
            torch.rand((2, 3, 160, 160), generator=generator),
            torch.tensor([epoch % 10, (epoch + 1) % 10], dtype=torch.long),
            ("fixture-0", "fixture-1"),
        )

    kwargs = {
        "config": config,
        "transmit_dim": 64,
        "quantiser_bits": 2,
        "device": "cpu",
        "runtime_root": tmp_path / "runtime",
        "source_binding": {"source_commit": "a" * 40},
        "campaign_id": "synthetic-er9-v4",
        "run_id": "synthetic-er9-v4",
        "live_authentication": {
            "authority": {
                "source_binding": {"source_commit": "a" * 40},
                "config_hash": "placeholder",
                "execution_profile_id": "confessor_pascal_cu126",
                "host": "confessor",
                "device": "cuda:0",
            },
            "environment": {"git_dirty": False},
        },
        "synthetic_provider": provider,
        "identity_overrides": {
            "fixture_id": "synthetic-er9-v4",
            "eligibility": {"SYNTHETIC_ONLY": True},
        },
        "total_epochs": 2,
        "runtime_role": "W9_SYNTHETIC_ONLY_ER9",
    }
    kwargs["live_authentication"]["authority"]["config_hash"] = config_hash(config)
    first = ER9V4CandidateTrainer(**kwargs, resume=False)
    assert first.run(max_epochs=1) is None
    second = ER9V4CandidateTrainer(**kwargs, resume=True)
    terminal = second.run(max_epochs=1)
    assert terminal is not None
    assert terminal["applied_optimizer_steps"] == 2
    assert [item.epoch for item in second.runtime.inspect()] == [0, 1]
