from __future__ import annotations

import copy
import hashlib
import json

import pytest

from config.execution_profiles import (
    ProfileAuthenticationError,
    bind_execution_profile,
    canonical_json_bytes,
    selection_record,
    verify_historical_generated_params_bytes,
    verify_historical_local_compatibility,
    verify_selection_record,
)
from config.params import get
from config.run_config import FrozenMap, RunConfig, config_hash


def test_profile_selection_is_frozen_and_fingerprinted(run_config_factory):
    run_config = run_config_factory()
    local = bind_execution_profile(run_config, "local_4060_cu130")
    assert local.resolved["execution_profile_id"] == "local_4060_cu130"
    assert config_hash(local) != config_hash(run_config)
    with pytest.raises(ValueError, match="already frozen"):
        bind_execution_profile(local, "confessor_pascal_cu126")


def test_qualified_pascal_profile_can_open_science(run_config_factory):
    run_config = run_config_factory()
    pascal = bind_execution_profile(run_config, "confessor_pascal_cu126")
    assert pascal.resolved["execution_profile_id"] == "confessor_pascal_cu126"


def test_pending_role_mutation_cannot_open_science(run_config_factory, monkeypatch):
    run_config = run_config_factory()
    from config import execution_profiles

    original = execution_profiles.profile_definition

    def pending(profile_id):
        profile = dict(original(profile_id))
        profile["role"] = "eligible_production_execution_profile_pending_qualification"
        return profile

    monkeypatch.setattr(execution_profiles, "profile_definition", pending)
    with pytest.raises(ValueError, match="not production eligible"):
        bind_execution_profile(run_config, "confessor_pascal_cu126")


def test_selection_record_mutations_fail():
    record = selection_record(
        scope_id="campaign-a",
        scope_kind="characterization_campaign",
        profile_id="local_4060_cu130",
        git_commit="a" * 40,
        config_hash="b" * 64,
    )
    verify_selection_record(record, expected_scope_id="campaign-a")
    for key in ("execution_profile_id", "lock_file_sha256", "git_commit", "config_hash"):
        changed = dict(record)
        changed[key] = "x"
        with pytest.raises((ValueError, KeyError)):
            verify_selection_record(changed, expected_scope_id="campaign-a")


def _historical(config: RunConfig) -> RunConfig:
    body = config.to_dict()
    body["fingerprint_schema_version"] = 1
    body["parameters"]["compute"].pop("primary_device_scope")
    body["parameters"]["compute"].pop("execution_profile_policy")
    for key in (
        "execution_profile_registry_schema_version",
        "execution_profile_id_required_for_new_science",
        "execution_profile_record_fields",
        "execution_profiles",
        "qualification",
        "historical_profile_compatibility",
        "scientific_writer_authentication",
    ):
        body["parameters"]["environment"].pop(key)
    return RunConfig.from_dict(body)


def test_historical_compatibility_is_narrow(run_config_factory):
    run_config = run_config_factory()
    historical = _historical(run_config)
    verify_historical_local_compatibility(historical)
    body = historical.to_dict()
    body["parameters"]["channel"]["train_snr_db_fixed"] += 1
    with pytest.raises(ValueError, match="unrelated drift"):
        verify_historical_local_compatibility(RunConfig.from_dict(body))


def test_historical_compatibility_rejects_profile_reinterpretation(run_config_factory, monkeypatch):
    run_config = run_config_factory()
    historical = _historical(run_config)
    from config import execution_profiles

    original = execution_profiles.get
    monkeypatch.setattr(
        execution_profiles,
        "get",
        lambda path: "confessor_pascal_cu126"
        if path == "environment.historical_profile_compatibility.archived_profile"
        else original(path),
    )
    with pytest.raises(ValueError, match="reinterpretation"):
        verify_historical_local_compatibility(historical)


def test_generated_params_compatibility_is_additive_only():
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "76e789c9f3d036427d5c1fe83bd95a61d655c5f0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    archived = subprocess.run(
        ["git", "show", f"{commit}:spec/params.generated.yaml"],
        capture_output=True,
        check=True,
    ).stdout
    verify_historical_generated_params_bytes(archived)
    changed = archived.replace(b"train_snr_db_fixed: 7", b"train_snr_db_fixed: 8", 1)
    with pytest.raises(ValueError, match="unrelated drift"):
        verify_historical_generated_params_bytes(changed)


def test_authentication_rejects_generic_cuda_before_runtime_probe(monkeypatch):
    from config import execution_profiles

    with pytest.raises(ProfileAuthenticationError, match="explicit cuda:N"):
        execution_profiles.authenticate_execution_profile(
            "local_4060_cu130", device="cuda", config_hash="0" * 64
        )
