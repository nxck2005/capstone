from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from baseline import g8_campaign
from config import execution_profiles


def _manifest() -> dict[str, object]:
    return json.loads((g8_campaign.REPO_ROOT / "results/baseline/g8/campaign_manifest.json").read_bytes())


def test_historical_g8_artifacts_accept_the_exact_am83_to_am86_addition() -> None:
    manifest = _manifest()
    g8_campaign.verify_historical_normative_sources(manifest["normative_sources"])
    g8_campaign.verify_historical_contract_sources(manifest["contract_sources"])


def test_historical_compatibility_rejects_unrelated_spec_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = tmp_path / "spec"
    spec.mkdir()
    (spec / "SPEC.md").write_bytes((g8_campaign.REPO_ROOT / "spec/SPEC.md").read_bytes() + b"\nUnrelated drift.\n")
    monkeypatch.setattr(g8_campaign, "REPO_ROOT", tmp_path)
    with pytest.raises(g8_campaign.G8ContractError, match="exact post-AM-86"):
        g8_campaign._verify_historical_profile_spec(b"archived-spec")


def test_historical_compatibility_rejects_unrelated_g8_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "tools"
    source.mkdir()
    relative = "tools/verify_g8_preflight.py"
    current = (g8_campaign.REPO_ROOT / relative).read_bytes()
    (source / "verify_g8_preflight.py").write_bytes(current + b"\n# unrelated drift\n")
    monkeypatch.setattr(g8_campaign, "REPO_ROOT", tmp_path)
    with pytest.raises(g8_campaign.G8ContractError, match="exact AM-83"):
        g8_campaign._verify_historical_profile_source(relative, b"archived-source")


def test_historical_compatibility_rejects_unrelated_g8_instruction_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "instructions"
    destination.mkdir()
    relative = "instructions/G8.txt"
    current = (g8_campaign.REPO_ROOT / relative).read_bytes()
    (destination / "G8.txt").write_bytes(current + b"\nUnrelated drift.\n")
    monkeypatch.setattr(g8_campaign, "REPO_ROOT", tmp_path)
    with pytest.raises(g8_campaign.G8ContractError, match="exact AM-83"):
        g8_campaign._verify_historical_profile_source(relative, b"archived-source")


def test_historical_compatibility_rejects_structure_drift_in_campaign_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "src/baseline"
    destination.mkdir(parents=True)
    relative = "src/baseline/g8_campaign.py"
    current = (g8_campaign.REPO_ROOT / relative).read_bytes()
    (destination / "g8_campaign.py").write_bytes(current + b"\n# unrelated drift\n")
    monkeypatch.setattr(g8_campaign, "REPO_ROOT", tmp_path)
    with pytest.raises(g8_campaign.G8ContractError, match="exact AM-83"):
        g8_campaign._verify_historical_profile_source(relative, b"archived-source")


def test_historical_compatibility_rejects_substituted_campaign_archive() -> None:
    with pytest.raises(g8_campaign.G8ContractError, match="bound pre-AM-83"):
        g8_campaign._verify_historical_profile_source(
            "src/baseline/g8_campaign.py", b"substituted-archived-source"
        )


def test_generated_params_compatibility_rejects_unrelated_nested_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    archived = subprocess.run(
        ["git", "show", "76e789c9f3d036427d5c1fe83bd95a61d655c5f0:spec/params.generated.yaml"],
        cwd=execution_profiles.REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    params = tmp_path / "spec"
    params.mkdir()
    current = (execution_profiles.REPO_ROOT / "spec/params.generated.yaml").read_bytes()
    mutated = current.replace(b"train_snr_db_fixed: 7", b"train_snr_db_fixed: 8", 1)
    (params / "params.generated.yaml").write_bytes(mutated)
    monkeypatch.setattr(execution_profiles, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="unrelated drift"):
        execution_profiles.verify_historical_generated_params_bytes(archived)
