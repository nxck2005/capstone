"""Mutation coverage for the fail-closed offline G-7 verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.params import REPO_ROOT
from verify_g7_profile import VerificationError, verify


@pytest.fixture
def profile_report(tmp_path: Path) -> Path:
    source = REPO_ROOT / "results/profiling/g7_djscc_profile.json"
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
    path = REPO_ROOT / "results/profiling/g7_djscc_profile.json"
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
