"""Mutation coverage for the fail-closed offline G-7 verifier."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from config.params import REPO_ROOT
from profile_djscc_g7 import (
    DEFAULT_CONFIG,
    DEFAULT_REPORT,
    PROFILE_TOOL_RELATIVE_PATH,
    ProfileError,
    _bound_worker_command,
    _source_record,
    profile,
)
import verify_g7_profile as verifier
from verify_g7_profile import VerificationError, verify


def _historical_report_repo() -> Path:
    report = REPO_ROOT / "results/profiling/g7_djscc_profile.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    profile_tool = Path(
        payload["execution_sources"]["profile_tool_source"]["resolved_runtime_path"]
    )
    return profile_tool.parents[1]


@pytest.fixture(autouse=True)
def bind_archived_report_to_its_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute provenance paths belong to the checkout that produced G-7."""

    historical_repo = _historical_report_repo()
    monkeypatch.setattr(verifier, "REPO", historical_repo)
    monkeypatch.setattr(
        verifier,
        "REPORT_PATH",
        historical_repo / "results/profiling/g7_djscc_profile.json",
    )


@pytest.fixture
def profile_report(tmp_path: Path) -> Path:
    source = _historical_report_repo() / "results/profiling/g7_djscc_profile.json"
    destination = tmp_path / source.name
    destination.write_bytes(source.read_bytes())
    return destination


def _mutate(path: Path, mutation) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_committed_g7_report_is_canonical_and_verifies():
    path = _historical_report_repo() / "results/profiling/g7_djscc_profile.json"
    raw = path.read_text(encoding="utf-8")

    assert raw.endswith("\n")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
    assert verify(path)["verdict"] == "PASS"


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_verifier_rejects_missing_or_unexpected_required_field(
    profile_report: Path,
    mutation: str,
):
    if mutation == "missing":
        _mutate(profile_report, lambda value: value.pop("gate"))
    else:
        _mutate(profile_report, lambda value: value.update(unknown_field=True))

    with pytest.raises(VerificationError, match="report fields differ"):
        verify(profile_report)


def test_verifier_rejects_dirty_implementation(profile_report: Path):
    _mutate(profile_report, lambda value: value.update(git_dirty=True))

    with pytest.raises(VerificationError, match="implementation state was dirty"):
        verify(profile_report)


def test_verifier_rejects_wrong_implementation_commit(profile_report: Path):
    _mutate(profile_report, lambda value: value.update(implementation_commit="0" * 40))

    with pytest.raises(VerificationError, match="wrong implementation commit"):
        verify(profile_report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["dataset"].update(name="stl10"), "wrong dataset"),
        (lambda value: value["dataset"].update(split="test"), "wrong split"),
        (lambda value: value["model"].update(bw_ratio="r_1_3"), "wrong ratio"),
        (
            lambda value: value["model"].update(
                architecture="width_halved_djscc_residual_v1"
            ),
            "wrong architecture",
        ),
        (lambda value: value["model"].update(k=1), "wrong complex-symbol budget"),
    ],
)
def test_verifier_rejects_wrong_profile_selector(
    profile_report: Path,
    mutation,
    message: str,
):
    _mutate(profile_report, mutation)

    with pytest.raises(VerificationError, match=message):
        verify(profile_report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["model"].update(
                parameter_count=value["model"]["parameter_count"] + 1
            ),
            "incorrect model parameter count",
        ),
        (
            lambda value: value["model"].update(absolute_parameter_cap=1),
            "incorrect absolute parameter cap",
        ),
        (
            lambda value: value["model"].update(
                reference_classifier_parameter_count=1
            ),
            "incorrect reference parameter cap",
        ),
    ],
)
def test_verifier_rejects_incorrect_parameter_caps(
    profile_report: Path,
    mutation,
    message: str,
):
    _mutate(profile_report, mutation)

    with pytest.raises(VerificationError, match=message):
        verify(profile_report)


def test_verifier_rejects_batch_below_32(profile_report: Path):
    _mutate(profile_report, lambda value: value["training"].update(batch_size=31))

    with pytest.raises(VerificationError, match="batch size below configured 32"):
        verify(profile_report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["training"].update(epochs_completed=0),
        lambda value: value["training"].update(num_examples=8468),
        lambda value: value["training"].update(num_batches=264),
    ],
)
def test_verifier_rejects_incomplete_epoch(profile_report: Path, mutation):
    _mutate(profile_report, mutation)

    with pytest.raises(VerificationError, match="incomplete epoch"):
        verify(profile_report)


def test_verifier_rejects_missing_cuda_environment_data(profile_report: Path):
    _mutate(
        profile_report,
        lambda value: value["environment"]["run_metadata"].pop("cuda_version"),
    )

    with pytest.raises(VerificationError, match="environment.run_metadata fields differ"):
        verify(profile_report)


def test_verifier_rejects_cpu_projection(profile_report: Path):
    _mutate(
        profile_report,
        lambda value: value["environment"].update(real_cuda=False),
    )

    with pytest.raises(VerificationError, match="CPU projection"):
        verify(profile_report)


def test_verifier_rejects_peak_vram_above_limit(profile_report: Path):
    gib = 1024**3

    def mutation(value):
        value["memory"]["peak_reserved_gb"] = 8.0
        value["memory"]["peak_reserved_bytes"] = 8 * gib

    _mutate(profile_report, mutation)

    with pytest.raises(VerificationError, match="peak VRAM above configured limit"):
        verify(profile_report)


def test_verifier_rejects_projected_runtime_above_limit(profile_report: Path):
    def mutation(value):
        value["training"]["epoch_time_s"] = 180.0
        value["training"]["images_per_second"] = (
            value["training"]["num_examples"] / 180.0
        )
        value["training"]["projected_training_hours"] = 5.0

    _mutate(profile_report, mutation)

    with pytest.raises(VerificationError, match="projected runtime above configured limit"):
        verify(profile_report)


def test_verifier_rejects_inconsistent_pass_component(profile_report: Path):
    _mutate(
        profile_report,
        lambda value: value["conditions"].update(peak_reserved_vram="FAIL"),
    )

    with pytest.raises(VerificationError, match="component verdict is inconsistent"):
        verify(profile_report)


def test_verifier_rejects_overall_pass_inconsistency(profile_report: Path):
    _mutate(profile_report, lambda value: value.update(verdict="HOLD"))

    with pytest.raises(VerificationError, match="overall G-7 PASS verdict is inconsistent"):
        verify(profile_report)


def test_verifier_rejects_any_test_split_claim(profile_report: Path):
    _mutate(
        profile_report,
        lambda value: value["data_isolation"].update(test_split_accessed=True),
    )

    with pytest.raises(VerificationError, match="test-split claim"):
        verify(profile_report)


def test_verifier_rejects_config_hash_disagreement(profile_report: Path):
    _mutate(profile_report, lambda value: value.update(config_hash="0" * 64))

    with pytest.raises(VerificationError, match="config hash disagrees"):
        verify(profile_report)


def test_verifier_rejects_manifest_disagreement(profile_report: Path):
    _mutate(
        profile_report,
        lambda value: value["dataset"].update(manifest_sha256="0" * 64),
    )

    with pytest.raises(VerificationError, match="manifest identity disagrees"):
        verify(profile_report)


def test_different_clean_checkout_cannot_silently_execute_current_code():
    with pytest.raises(ProfileError, match="profile target HEAD"):
        profile(
            config_path=DEFAULT_CONFIG,
            report_path=DEFAULT_REPORT,
            git_repo=REPO_ROOT,
            data_repo=REPO_ROOT,
        )


def test_bound_worker_command_uses_implementation_script_and_separate_data_root(
    tmp_path: Path,
):
    implementation = tmp_path / "implementation"
    data_root = tmp_path / "verified-data"
    command = _bound_worker_command(
        profile_script=implementation / PROFILE_TOOL_RELATIVE_PATH,
        audit_path=tmp_path / "audit.json",
        implementation_config=implementation / "configs/learned-g7-profile.yaml",
        raw_report_path=tmp_path / "raw.json",
        git_repo=implementation,
        data_repo=data_root,
    )

    assert command[3] == str(implementation / PROFILE_TOOL_RELATIVE_PATH)
    assert command[command.index("--git-repo") + 1] == str(implementation)
    assert command[command.index("--data-repo") + 1] == str(data_root)
    assert str(REPO_ROOT / PROFILE_TOOL_RELATIVE_PATH) not in command


def _make_source_repository(tmp_path: Path) -> tuple[Path, str, Path]:
    repository = tmp_path / "implementation"
    source = repository / "src/env.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "G-7 Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "src/env.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return repository, commit, source


def test_modified_implementation_source_is_rejected(tmp_path: Path):
    repository, commit, source = _make_source_repository(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="executed bytes differ"):
        _source_record(
            git_repo=repository,
            implementation_commit=commit,
            repository_relative_path="src/env.py",
            runtime_path=source,
        )


def test_critical_module_imported_outside_worktree_is_rejected(tmp_path: Path):
    repository, commit, _ = _make_source_repository(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="outside implementation worktree"):
        _source_record(
            git_repo=repository,
            implementation_commit=commit,
            repository_relative_path="src/env.py",
            runtime_path=outside,
        )


@pytest.mark.parametrize("field", ["sha256", "git_blob_sha"])
def test_verifier_rejects_wrong_execution_source_identity(
    profile_report: Path, field: str
):
    def mutation(value):
        source = value["execution_sources"]["critical_files"]["src/env.py"]
        source[field] = "0" * len(source[field])

    _mutate(profile_report, mutation)
    with pytest.raises(VerificationError, match="implementation commit"):
        verify(profile_report)


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_verifier_rejects_missing_or_unexpected_execution_source(
    profile_report: Path, mutation: str
):
    def change(value):
        sources = value["execution_sources"]["critical_files"]
        if mutation == "missing":
            sources.pop("src/env.py")
        else:
            sources["src/unexpected.py"] = dict(sources["src/env.py"])

    _mutate(profile_report, change)
    with pytest.raises(VerificationError, match="critical source entries differ"):
        verify(profile_report)
