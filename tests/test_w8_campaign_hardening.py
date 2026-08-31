"""W8 campaign lock, heartbeat and detached-launch boundary tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_w8_campaign as campaign
from config.run_config import config_hash as run_config_hash
from config.w8_execution import W8ExecutionHold, verify_frozen_w8_gpu_binding
from runtime.w8_lock import W8CampaignLock, W8LockBusy
from training.deterministic_core import canonical_bytes, canonical_sha256
from tests.w8_hardening_fixtures import profile_binding, tiny_config


def _child_code(lock_path: Path) -> str:
    return f'''
from runtime.w8_lock import W8CampaignLock
lock = W8CampaignLock(campaign_id="fixture", source_commit="a" * 40,
    execution_image="fixture", gpu_uuid="GPU-fixture", lock_path={str(lock_path)!r})
lock.acquire()
print("ACQUIRED", flush=True)
try:
    input()
finally:
    lock.release()
'''


def test_second_w8_campaign_writer_is_kernel_blocked_then_death_releases(tmp_path: Path):
    lock_path = tmp_path / "w8-global.lock"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    first = subprocess.Popen(
        [sys.executable, "-c", _child_code(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert first.stdout is not None
        assert first.stdout.readline().strip() == "ACQUIRED"
        second_code = f'''
from runtime.w8_lock import W8CampaignLock, W8LockBusy
lock = W8CampaignLock(campaign_id="second", source_commit="b" * 40,
    execution_image="fixture", gpu_uuid="GPU-fixture", lock_path={str(lock_path)!r})
try:
    lock.acquire()
except W8LockBusy:
    print("BUSY")
else:
    print("BAD")
    lock.release()
'''
        blocked = subprocess.run(
            [sys.executable, "-c", second_code],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        assert blocked.stdout.strip() == "BUSY"
        first.kill()
        first.wait(timeout=5)
        released = subprocess.run(
            [sys.executable, "-c", second_code],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        assert released.stdout.strip() == "BAD"
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()


def test_heartbeat_is_atomic_operational_state_only(tmp_path: Path):
    path = tmp_path / "heartbeat.json"
    campaign.write_heartbeat(
        path,
        campaign_id="w8-test",
        current_run_index=2,
        ratio="r_1_24",
        train_seed=1,
        channel_seed=1,
        current_epoch=17,
        process_state="VALIDATING",
        latest_checkpoint_id="a" * 64,
        completed_runs=1,
        completed_epoch_cycles=18,
    )
    value = json.loads(path.read_bytes())
    assert set(value) == {
        "schema_version", "artifact_role", "campaign_id", "current_run_index",
        "total_runs", "ratio", "train_seed", "channel_seed", "current_epoch",
        "process_state", "latest_checkpoint_id", "completed_runs",
        "completed_epoch_cycles", "updated_at_utc",
    }
    assert value["total_runs"] == 6
    assert all(word not in value for word in ("accuracy", "loss", "psnr", "papr"))
    path2 = tmp_path / "heartbeat-symlink.json"
    path2.symlink_to(path)
    with pytest.raises(campaign.W8CampaignHold, match="unsafe"):
        campaign.write_heartbeat(
            path2,
            campaign_id="w8-test",
            current_run_index=None,
            ratio=None,
            train_seed=None,
            channel_seed=None,
            current_epoch=None,
            process_state="IDLE",
            latest_checkpoint_id=None,
            completed_runs=0,
            completed_epoch_cycles=0,
        )


def _synthetic_launch(tmp_path: Path) -> tuple[dict, dict, Path]:
    w8_path = tmp_path / "w8-a.json"
    w8 = {
        "authorization_id": "w8auth-fixture",
        "scientific_source": {"source_manifest_file_sha256": "b" * 64},
    }
    w8_path.write_bytes(canonical_bytes(w8))
    source = {"source_commit": "a" * 40, "manifest_id": "w8source-fixture"}
    body = {
        "schema_version": 1,
        "artifact_role": campaign.LAUNCH_AUTHORIZATION_ROLE,
        "status": "AUTHORIZED",
        "authorization_scope": "W8_SIX_CORE_RUNS_ONLY",
        "issued_at_utc": "2026-08-31T00:00:00+00:00",
        "w8_a_authorization_id": w8["authorization_id"],
        "w8_a_authorization_sha256": campaign._sha(w8_path),
        "source_commit": source["source_commit"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": "b" * 64,
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_root": campaign.CAMPAIGN_ROOT,
        "profile": {
            "execution_profile_id": "confessor_pascal_cu126",
            "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
            "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
            "device": "cuda:0",
            "requirements_lock": "requirements-pascal.lock",
            "requirements_lock_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
            "physical_batch_size": 32,
            "accumulation_factor": 1,
            "effective_batch_size": 32,
            "validation_batch_size": 32,
        },
        "scope": {"core_runs": 6, "er2_randomized_training": False, "papr_constrained_training": False, "er9_training": False, "g10": False},
        "test": {"status": "SEALED", "model_facing_access": 0, "learned_inference": 0},
        "owner_authorization": True,
    }
    value = dict(body)
    value["authorization_id"] = "w8blaunch-" + canonical_sha256(body)
    launch_path = tmp_path / "launch.json"
    launch_path.write_bytes(canonical_bytes(value))
    return w8, source, launch_path


def test_detached_launch_requires_exact_owner_scope_and_rejects_mutations(tmp_path: Path):
    w8, source, launch_path = _synthetic_launch(tmp_path)
    w8_path = tmp_path / "w8-a.json"
    accepted = campaign.verify_launch_authorization(
        launch_path,
        w8_authorization=w8,
        w8_authorization_path=w8_path,
        source_manifest=source,
    )
    assert accepted["owner_authorization"] is True
    original = json.loads(launch_path.read_bytes())
    for mutation in (
        lambda item: item["scope"].__setitem__("g10", True),
        lambda item: item["test"].__setitem__("model_facing_access", 1),
        lambda item: item.__setitem__("campaign_id", "foreign"),
    ):
        changed = json.loads(json.dumps(original))
        mutation(changed)
        body = dict(changed)
        body.pop("authorization_id")
        changed["authorization_id"] = "w8blaunch-" + canonical_sha256(body)
        launch_path.write_bytes(canonical_bytes(changed))
        with pytest.raises(campaign.W8CampaignHold):
            campaign.verify_launch_authorization(
                launch_path,
                w8_authorization=w8,
                w8_authorization_path=w8_path,
                source_manifest=source,
            )
    launch_path.write_bytes(canonical_bytes(original))


def test_root_preflight_does_not_create_or_accept_nonempty_root(tmp_path: Path):
    absent = tmp_path / "absent"
    campaign._safe_root(absent, allow_existing=False)
    assert not absent.exists()
    present = tmp_path / "present"
    present.mkdir()
    (present / "foreign").write_text("x", encoding="ascii")
    with pytest.raises(campaign.W8CampaignHold, match="not empty"):
        campaign._safe_root(present, allow_existing=False)
    with pytest.raises(campaign.W8CampaignHold, match="foreign state"):
        campaign._safe_root(present, allow_existing=True)


def test_w8_test_boundary_is_structurally_sealed():
    campaign._verify_test_boundary(campaign.REPO)
    from data.registry import DatasetRegistryError, load_dataset

    with pytest.raises(DatasetRegistryError, match="test"):
        load_dataset("imagenette160", "test")


def test_frozen_gpu_binding_rejects_outer_and_nested_profile_mutations():
    config = tiny_config()
    expected_hash = run_config_hash(config)
    original = profile_binding(config)
    verify_frozen_w8_gpu_binding(
        original, config_hash=expected_hash, source_commit="a" * 40
    )
    for location, key, replacement in (
        ("outer", "gpu_uuid", "GPU-titan-xp"),
        ("outer", "lock_file_sha256", "c" * 64),
        ("nested", "gpu_name", "NVIDIA GeForce RTX 4060 Laptop GPU"),
        ("nested", "config_hash", "d" * 64),
    ):
        candidate = json.loads(json.dumps(original))
        target = candidate if location == "outer" else candidate["profile_environment"]
        target[key] = replacement
        body = dict(candidate)
        body.pop("binding_sha256")
        candidate["binding_sha256"] = canonical_sha256(body)
        with pytest.raises(W8ExecutionHold):
            verify_frozen_w8_gpu_binding(
                candidate, config_hash=expected_hash, source_commit="a" * 40
            )
