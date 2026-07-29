"""Network-free regression coverage for the frozen G-1 adjudication."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from config.params import REPO_ROOT
from verify_g1_adjudication import VerificationError, verify


ARTIFACT_NAMES = (
    "best_checkpoint.json",
    "epochs.jsonl",
    "g1_adjudication.json",
    "resolved_config.json",
    "validation_summary.json",
)


@pytest.fixture
def adjudication_repo(tmp_path: Path) -> Path:
    source = REPO_ROOT / "results/reference_classifier"
    destination = tmp_path / "results/reference_classifier"
    destination.mkdir(parents=True)
    for name in ARTIFACT_NAMES:
        shutil.copyfile(source / name, destination / name)
    shutil.copyfile(REPO_ROOT / "requirements.lock", tmp_path / "requirements.lock")
    return tmp_path


def _json_path(repo: Path, name: str) -> Path:
    return repo / "results/reference_classifier" / name


def _mutate_json(repo: Path, name: str, mutation) -> None:
    path = _json_path(repo, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_committed_adjudication_is_sorted_json_with_trailing_newline():
    path = REPO_ROOT / "results/reference_classifier/g1_adjudication.json"
    raw = path.read_text(encoding="utf-8")

    assert raw.endswith("\n")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"


def test_verifier_accepts_committed_evidence_without_local_checkpoint(adjudication_repo: Path):
    result = verify(adjudication_repo)

    assert result["verdict"] == "PASS"
    assert result["epochs"] == 100
    assert result["local_checkpoint_verified"] is False


def test_verifier_rejects_missing_epoch(adjudication_repo: Path):
    path = _json_path(adjudication_repo, "epochs.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="exactly 100"):
        verify(adjudication_repo)


@pytest.mark.parametrize("mutation", ["duplicate", "out_of_order"])
def test_verifier_rejects_duplicate_or_out_of_order_epoch(
    adjudication_repo: Path,
    mutation: str,
):
    path = _json_path(adjudication_repo, "epochs.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "duplicate":
        lines[1] = lines[0]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="contiguous, ordered, and unique"):
        verify(adjudication_repo)


def test_verifier_rejects_inconsistent_integral_accuracy(adjudication_repo: Path):
    path = _json_path(adjudication_repo, "epochs.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["validation"]["top1_accuracy"] = 0.999
    lines[0] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="inconsistent with integral counts"):
        verify(adjudication_repo)


def test_verifier_rejects_incorrect_best_epoch(adjudication_repo: Path):
    _mutate_json(adjudication_repo, "g1_adjudication.json", lambda value: value.update(best_epoch=98))

    with pytest.raises(VerificationError, match="best_epoch"):
        verify(adjudication_repo)


def test_verifier_rejects_result_that_does_not_clear_floor(adjudication_repo: Path):
    _mutate_json(
        adjudication_repo,
        "resolved_config.json",
        lambda value: value["parameters"]["datasets"]["imagenette160"].update(clean_acc_floor=0.9),
    )
    _mutate_json(adjudication_repo, "g1_adjudication.json", lambda value: value.update(floor=0.9))

    with pytest.raises(VerificationError, match="does not clear"):
        verify(adjudication_repo)


def test_verifier_rejects_absolute_checkpoint_path(adjudication_repo: Path):
    _mutate_json(
        adjudication_repo,
        "g1_adjudication.json",
        lambda value: value.update(checkpoint_repository_path="/tmp/epoch-99.pt"),
    )

    with pytest.raises(VerificationError, match="relative, not absolute"):
        verify(adjudication_repo)


def test_verifier_rejects_hash_disagreement(adjudication_repo: Path):
    _mutate_json(
        adjudication_repo,
        "g1_adjudication.json",
        lambda value: value.update(checkpoint_sha256="0" * 64),
    )

    with pytest.raises(VerificationError, match="checkpoint_sha256"):
        verify(adjudication_repo)


def test_verifier_rejects_missing_external_artifact_identity(adjudication_repo: Path):
    _mutate_json(
        adjudication_repo,
        "g1_adjudication.json",
        lambda value: value["checkpoint_external_artifact"].pop("asset_name"),
    )

    with pytest.raises(VerificationError, match="external artifact identity missing"):
        verify(adjudication_repo)
