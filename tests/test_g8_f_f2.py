from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from baseline.g8_f_closeout import INFEASIBLE, MANIFEST_FIELDS, MATERIALIZED, OMISSION
from data.classifier import epoch_permutation
from training.g8_f_f2 import (
    F2ArtifactDataset,
    F2Hold,
    F2Trainer,
    RECONSTRUCTION_BYTES,
    learning_rate_for_epoch,
)


def _fixture(tmp_path: Path):
    runtime = tmp_path / "runtime"
    raw = bytes([37]) * RECONSTRUCTION_BYTES
    digest = hashlib.sha256(raw).hexdigest()
    object_path = runtime / "objects/reconstruction" / f"{digest}.rgb"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(raw)

    def row(ordinal: int, assignment: str, stable: str, quality: str, outcome: str):
        value = {field: "" for field in MANIFEST_FIELDS}
        value.update({
            "ordinal": str(ordinal),
            "assignment_id": assignment,
            "stable_sample_id": stable,
            "class_label": "2",
            "quality_id": quality,
            "payload_budget_bytes": "100",
            "encode_axis_px": "160",
            "outcome": outcome,
            "request_id": f"request-{ordinal}",
            "result_id": f"result-{ordinal}",
            "request_record_sha256": f"{ordinal + 1:064x}",
            "result_record_sha256": f"{ordinal + 11:064x}",
        })
        if outcome == MATERIALIZED:
            value.update({
                "codestream_sha256": f"{ordinal + 21:064x}",
                "codestream_bytes": "50",
                "codestream_path": f"objects/codestream/{ordinal + 21:064x}.j2k",
                "reconstruction_sha256": digest,
                "reconstruction_bytes": str(RECONSTRUCTION_BYTES),
                "reconstruction_path": f"objects/reconstruction/{digest}.rgb",
            })
        else:
            value["omission_state"] = OMISSION
        return value

    rows = [
        row(0, "assignment-0", "stable-a", "quality-a", MATERIALIZED),
        row(1, "assignment-1", "stable-a", "quality-b", MATERIALIZED),
        row(2, "assignment-2", "stable-b", "quality-c", INFEASIBLE),
    ]
    authority = [
        {"assignment_id": row["assignment_id"], "stable_sample_id": row["stable_sample_id"], "class_label": 2, "quality_id": row["quality_id"], "split": "train"}
        for row in rows
    ]
    return runtime, rows, authority, digest


def _dataset(tmp_path: Path, **changes):
    runtime, rows, authority, digest = _fixture(tmp_path)
    arguments = {
        "runtime_root": runtime,
        "assignment_authority": authority,
        "epoch": 0,
        "train_seed": 0,
        "corpus_id": "corpus",
        "corpus_sha256": "a" * 64,
        "expected_corpus_id": "corpus",
        "expected_corpus_sha256": "a" * 64,
        "expected_assignment_rows": 3,
        "expected_materialized_rows": 2,
        "expected_omitted_rows": 1,
    }
    arguments.update(changes)
    return F2ArtifactDataset(rows, **arguments), rows, authority, runtime, digest


def test_assignment_multiplicity_is_not_reconstruction_deduplication(tmp_path: Path):
    dataset, _rows, _authority, _runtime, digest = _dataset(tmp_path)
    assert len(dataset) == 2
    assert dataset.summary.distinct_materialized_assignments == 2
    assert dataset.summary.unique_reconstruction_sha256 == 1
    assert dataset.trace(0).reconstruction_sha256 == dataset.trace(1).reconstruction_sha256 == digest
    assert dataset.summary.omitted_rows == 1
    assert dataset.summary.validation_ids == dataset.summary.test_ids == 0


@pytest.mark.parametrize("field,bad", [("corpus_id", "foreign"), ("corpus_sha256", "b" * 64)])
def test_wrong_f1_corpus_binding_rejected(tmp_path: Path, field: str, bad: str):
    with pytest.raises(F2Hold):
        _dataset(tmp_path, **{field: bad})


def test_foreign_duplicate_removed_and_wrong_class_rows_rejected(tmp_path: Path):
    runtime, rows, authority, _digest = _fixture(tmp_path)
    common = dict(runtime_root=runtime, epoch=0, train_seed=0, corpus_id="c", corpus_sha256="d", expected_assignment_rows=3, expected_materialized_rows=2, expected_omitted_rows=1)
    for mutate in ("foreign", "duplicate", "removed", "class"):
        changed = [dict(row) for row in rows]
        auth = [dict(item) for item in authority]
        expected_rows = 3
        if mutate == "foreign":
            changed[0]["assignment_id"] = "foreign"
        elif mutate == "duplicate":
            changed[1]["assignment_id"] = changed[0]["assignment_id"]
            auth[1]["assignment_id"] = auth[0]["assignment_id"]
        elif mutate == "removed":
            changed.pop()
            auth.pop()
        else:
            changed[0]["class_label"] = "3"
        with pytest.raises(F2Hold):
            F2ArtifactDataset(changed, assignment_authority=auth, **common)


def test_omission_cannot_be_materialized_or_substituted(tmp_path: Path):
    runtime, rows, authority, digest = _fixture(tmp_path)
    changed = [dict(row) for row in rows]
    changed[2]["reconstruction_sha256"] = digest
    changed[2]["reconstruction_bytes"] = str(RECONSTRUCTION_BYTES)
    changed[2]["reconstruction_path"] = f"objects/reconstruction/{digest}.rgb"
    with pytest.raises(F2Hold):
        F2ArtifactDataset(changed, runtime_root=runtime, assignment_authority=authority, epoch=0, train_seed=0, corpus_id="c", corpus_sha256="d", expected_assignment_rows=3, expected_materialized_rows=2, expected_omitted_rows=1)


def test_missing_corrupt_substituted_and_symlink_reconstruction_rejected(tmp_path: Path):
    for mutation in ("missing", "corrupt", "substitution", "symlink"):
        base = tmp_path / mutation
        runtime, rows, authority, digest = _fixture(base)
        path = runtime / "objects/reconstruction" / f"{digest}.rgb"
        if mutation == "missing":
            path.unlink()
        elif mutation == "corrupt":
            path.write_bytes(b"x" * RECONSTRUCTION_BYTES)
        elif mutation == "substitution":
            alternate = runtime / "clean.rgb"
            alternate.write_bytes(path.read_bytes())
            rows[0]["reconstruction_path"] = "clean.rgb"
        else:
            target = base / "target.rgb"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        with pytest.raises(F2Hold):
            F2ArtifactDataset(rows, runtime_root=runtime, assignment_authority=authority, epoch=0, train_seed=0, corpus_id="c", corpus_sha256="d", expected_assignment_rows=3, expected_materialized_rows=2, expected_omitted_rows=1)


@pytest.mark.parametrize("split", ["val", "test"])
def test_nontraining_authority_rejected(tmp_path: Path, split: str):
    runtime, rows, authority, _digest = _fixture(tmp_path)
    authority[0]["split"] = split
    with pytest.raises(F2Hold):
        F2ArtifactDataset(rows, runtime_root=runtime, assignment_authority=authority, epoch=0, train_seed=0, corpus_id="c", corpus_sha256="d", expected_assignment_rows=3, expected_materialized_rows=2, expected_omitted_rows=1)


def test_exact_schedule_and_order_are_direct_epoch_functions():
    assert learning_rate_for_epoch(0) == pytest.approx(0.001)
    assert learning_rate_for_epoch(4) == pytest.approx(0.01)
    assert learning_rate_for_epoch(19) == pytest.approx(0.0, abs=1e-15)
    assert epoch_permutation(17, 0, 3) == epoch_permutation(17, 0, 3)
    assert epoch_permutation(17, 0, 3) != epoch_permutation(17, 0, 4)


class _TinyParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([7.0]))

    def forward(self, inputs):
        return inputs * self.weight


def _authorization(identifier="auth"):
    return {
        "authorization_id": identifier,
        "source_commit": "1" * 40,
        "execution_profile": {"execution_profile_id": "confessor_pascal_cu126", "device": "cuda:0"},
    }


def test_trainer_initializes_from_exact_parent_not_random(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("training.g8_f_f2.load_frozen_reference_classifier", lambda device, allow_download: _TinyParent())
    trainer = F2Trainer(authorization=_authorization(), authorization_sha256="a" * 64, runtime_root=tmp_path, dataset=object(), device="cpu")
    assert trainer.model.weight.item() == 7.0
    assert trainer.model.weight.requires_grad


def test_resume_authenticates_exact_latest_epoch_and_never_replays_it(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("training.g8_f_f2.load_frozen_reference_classifier", lambda device, allow_download: _TinyParent())
    first = F2Trainer(authorization=_authorization(), authorization_sha256="a" * 64, runtime_root=tmp_path, dataset=object(), device="cpu")
    first.completed_epoch = 0
    first.total_optimizer_steps = 345
    first.training_history = [{"epoch": 0}]
    first.validation_history = [{"epoch": 0}]
    first.best_epoch = 0
    first.best_validation_top1 = 0.5
    first.save_checkpoint()
    resumed = F2Trainer(authorization=_authorization(), authorization_sha256="a" * 64, runtime_root=tmp_path, dataset=object(), device="cpu")
    resumed.resume()
    assert resumed.completed_epoch == 0
    assert resumed.total_optimizer_steps == 345
    assert resumed.completed_epoch + 1 == 1
    (tmp_path / "checkpoints/epoch-00.pt").write_bytes(b"corrupt")
    with pytest.raises(F2Hold):
        F2Trainer(authorization=_authorization(), authorization_sha256="a" * 64, runtime_root=tmp_path, dataset=object(), device="cpu").resume()


def test_resume_rejects_foreign_authorization_lineage(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("training.g8_f_f2.load_frozen_reference_classifier", lambda device, allow_download: _TinyParent())
    first = F2Trainer(authorization=_authorization(), authorization_sha256="a" * 64, runtime_root=tmp_path, dataset=object(), device="cpu")
    first.completed_epoch = 0
    first.total_optimizer_steps = 345
    first.training_history = [{"epoch": 0}]
    first.validation_history = [{"epoch": 0}]
    first.best_epoch = 0
    first.best_validation_top1 = 0.5
    first.save_checkpoint()
    foreign = F2Trainer(authorization=_authorization("foreign"), authorization_sha256="b" * 64, runtime_root=tmp_path, dataset=object(), device="cpu")
    with pytest.raises(F2Hold):
        foreign.resume()
