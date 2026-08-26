from __future__ import annotations

import os

import run_quality_gate as gate


def _joined(profile: str) -> str:
    return "\n".join(" ".join(command) for command in gate.profile_commands(profile))


def test_evidence_profile_has_no_execution_entrypoints():
    commands = _joined("evidence")
    assert "run_one_unit" not in commands
    assert "run_g8_bler_characterization" not in commands
    assert "verify_g8_evidence_readonly.py" in commands
    assert "tests/test_g8_bler_characterization_v2.py" in commands


def test_software_profile_uses_offline_runner_verification():
    commands = _joined("static")
    assert "verify_g8_bler_runner_contract_offline.py" in commands
    assert "verify_g8_bler_characterization_manifest_v2.py" in commands
    assert "verify_w5_training_system.py" in commands
    assert "verify_w5_training_system.py --pre-source" not in commands
    assert "gen_g8_bler_runner_contract.py" not in commands


def test_cpu_profile_excludes_only_audited_nonportable_categories(monkeypatch):
    monkeypatch.delenv("CAPSTONE_INCLUDE_EXTERNAL_LDPC_FIXTURE", raising=False)
    commands = _joined("ci-cpu")
    assert "not primary_runtime and not external_ldpc_fixture" in commands
    assert "not external_dataset and not frozen_checkpoint" in commands
    assert "not external_codec_runtime and not historical_profile_artifact" in commands


def test_weekly_cpu_profile_adds_only_the_external_ldpc_fixture(monkeypatch):
    monkeypatch.setenv("CAPSTONE_INCLUDE_EXTERNAL_LDPC_FIXTURE", "1")
    commands = _joined("ci-cpu")
    assert "not primary_runtime and not external_dataset" in commands
    assert "not external_codec_runtime and not historical_profile_artifact" in commands
    assert "external_ldpc_fixture" not in commands
