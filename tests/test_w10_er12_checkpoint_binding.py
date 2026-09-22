"""ER-12 must consume the ER-9 checkpoint identity through the frozen schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_ID = "a" * 64


@pytest.fixture(scope="module")
def er12_generator():
    path = REPO / "tools/gen_w10_er12_validation_selection.py"
    spec = importlib.util.spec_from_file_location(
        "gen_w10_er12_validation_selection_binding_regression", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_er12_checkpoint_identity_matches_real_binding_shape(er12_generator) -> None:
    binding = {
        "checkpoint": {
            "selected_checkpoint_sha256": CHECKPOINT_ID,
            "runtime_root": "checkpoints/er9_production_pascal_v4",
        }
    }
    assets = {"checkpoint_id": CHECKPOINT_ID, "model": object()}

    assert "checkpoint_id" not in binding["checkpoint"]
    assert er12_generator._er12_checkpoint_id(binding, assets) == CHECKPOINT_ID


def test_er12_checkpoint_identity_rejects_missing_frozen_binding_id(er12_generator) -> None:
    binding = {"checkpoint": {"runtime_root": "checkpoints/er9_production_pascal_v4"}}

    with pytest.raises(RuntimeError, match="selected_checkpoint_sha256"):
        er12_generator._er12_checkpoint_id(binding, {"checkpoint_id": CHECKPOINT_ID})


def test_er12_checkpoint_identity_rejects_missing_loaded_id(er12_generator) -> None:
    binding = {"checkpoint": {"selected_checkpoint_sha256": CHECKPOINT_ID}}

    with pytest.raises(RuntimeError, match="checkpoint_id"):
        er12_generator._er12_checkpoint_id(binding, {"model": object()})


def test_er12_checkpoint_identity_rejects_mismatch(er12_generator) -> None:
    binding = {"checkpoint": {"selected_checkpoint_sha256": CHECKPOINT_ID}}

    with pytest.raises(RuntimeError, match="differs from the frozen binding"):
        er12_generator._er12_checkpoint_id(binding, {"checkpoint_id": "b" * 64})
