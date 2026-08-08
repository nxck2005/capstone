"""Inference-loader coverage for the immutable G-1 classifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from config.params import REPO_ROOT
import models.frozen_reference_classifier as frozen
from models.frozen_reference_classifier import (
    FrozenClassifierError,
    _validate_payload,
    load_frozen_reference_classifier,
)


@pytest.fixture(scope="module")
def checkpoint_payload():
    return torch.load(
        REPO_ROOT / "checkpoints/reference_classifier/epoch-99.pt",
        map_location="cpu",
        weights_only=False,
    )


@pytest.fixture(scope="module")
def adjudication():
    return json.loads(
        (REPO_ROOT / "results/reference_classifier/g1_adjudication.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def committed_config():
    return json.loads(
        (REPO_ROOT / "results/reference_classifier/resolved_config.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.frozen_checkpoint
def test_exact_checkpoint_is_accepted_and_frozen(monkeypatch: pytest.MonkeyPatch):
    def forbidden_download(**kwargs):
        raise AssertionError("valid local checkpoint must not trigger a download")

    monkeypatch.setattr(frozen, "_download_checkpoint", forbidden_download)
    model = load_frozen_reference_classifier("cpu", allow_download=True)

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.total_parameter_count == 11_181_642


@pytest.mark.frozen_checkpoint
def test_wrong_checkpoint_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"not the frozen checkpoint")
    monkeypatch.setattr(frozen, "_verify_g1_adjudication", lambda repo_root: None)

    with pytest.raises(FrozenClassifierError, match="byte length|SHA-256"):
        load_frozen_reference_classifier(
            "cpu",
            checkpoint_path=checkpoint,
            allow_download=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_hash", "0" * 64, "config hash"),
        ("split_manifest_hash", "0" * 64, "split_manifest_hash"),
    ],
)
@pytest.mark.frozen_checkpoint
def test_wrong_config_or_manifest_identity_is_rejected(
    checkpoint_payload,
    adjudication,
    committed_config,
    field: str,
    value: str,
    message: str,
):
    payload = dict(checkpoint_payload)
    payload[field] = value

    with pytest.raises(FrozenClassifierError, match=message):
        _validate_payload(
            payload,
            adjudication=adjudication,
            committed_config=committed_config,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
@pytest.mark.frozen_checkpoint
def test_missing_or_extra_state_dict_key_is_rejected(
    checkpoint_payload,
    adjudication,
    committed_config,
    mutation: str,
):
    payload = dict(checkpoint_payload)
    state = dict(checkpoint_payload["model_state"])
    if mutation == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.weight"] = torch.zeros(1)
    payload["model_state"] = state

    with pytest.raises(FrozenClassifierError, match="model state differs"):
        _validate_payload(
            payload,
            adjudication=adjudication,
            committed_config=committed_config,
        )


@pytest.mark.frozen_checkpoint
def test_smoke_or_non_g1_checkpoint_is_rejected(
    checkpoint_payload,
    adjudication,
    committed_config,
):
    payload = dict(checkpoint_payload)
    payload.update(
        execution_mode="smoke",
        full_run_requested=False,
        run_complete=False,
        g1_eligible=False,
        lineage_g1_eligible=False,
        smoke_steps=1,
        smoke_val_batches=1,
    )

    with pytest.raises(FrozenClassifierError, match="execution_mode"):
        _validate_payload(
            payload,
            adjudication=adjudication,
            committed_config=committed_config,
        )


@pytest.mark.frozen_checkpoint
def test_incomplete_full_run_lineage_is_rejected(
    checkpoint_payload,
    adjudication,
    committed_config,
):
    payload = copy.copy(checkpoint_payload)
    payload["training_history"] = checkpoint_payload["training_history"][:-1]

    with pytest.raises(FrozenClassifierError, match="incomplete full-run"):
        _validate_payload(
            payload,
            adjudication=adjudication,
            committed_config=committed_config,
        )
