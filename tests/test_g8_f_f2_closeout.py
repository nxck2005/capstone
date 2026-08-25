from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training.g8_f_f2_closeout import (
    BEST_EPOCH,
    BEST_RELEASE,
    BEST_TOP1,
    COMPLETION_PREFIX,
    F2CloseoutHold,
    PROTECTED,
    _verify_id,
    identified,
    select_best,
    verify_compact,
)

ROOT = Path(__file__).resolve().parents[1]


def _validation(values: list[float]) -> list[dict[str, float | int]]:
    return [
        {"epoch": epoch, "n_correct": round(value * 1000), "n_total": 1000, "top1_accuracy": value}
        for epoch, value in enumerate(values)
    ]


def test_preregistered_selection_is_max_with_earliest_tie_break() -> None:
    values = [0.80] * 20
    values[2] = 0.89
    values[17] = 0.89
    assert select_best(_validation(values)) == (2, 0.89)


def test_observed_selected_checkpoint_constants_are_frozen() -> None:
    assert BEST_EPOCH == 17
    assert BEST_TOP1 == 0.89
    assert BEST_RELEASE == {
        "provider": "github_release",
        "repository": "nxck2005/capstone",
        "release_tag": "g8-f-f2-artifact-classifier-2026-08-25",
        "asset_name": "artifact-finetuned-imagenette160-epoch17-468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1.pt",
        "bytes": 89555403,
        "sha256": "468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1",
    }


def test_content_identity_rejects_mutation() -> None:
    value = identified({"schema_version": 1, "status": "complete"}, field="completion_id", prefix=COMPLETION_PREFIX)
    _verify_id(value, field="completion_id", prefix=COMPLETION_PREFIX)
    value["status"] = "changed"
    with pytest.raises(F2CloseoutHold, match="content identity"):
        _verify_id(value, field="completion_id", prefix=COMPLETION_PREFIX)


def test_closeout_protected_boundary_is_exactly_zero() -> None:
    assert PROTECTED == {
        "f3_cached_sweep_rescoring": 0,
        "pass_two": 0,
        "pass_three": 0,
        "fallback": 0,
        "ratio_adjudication": 0,
        "learned_training": 0,
        "test_access": 0,
    }


def test_committed_closeout_freeze_and_monitor_authenticate() -> None:
    completion, freeze = verify_compact()
    assert completion["completion_id"].startswith(COMPLETION_PREFIX)
    assert freeze["checkpoint_id"] == BEST_RELEASE["sha256"]
    assert completion["protected_state"] == freeze["protected_state"] == PROTECTED


def test_closeout_has_no_training_or_later_phase_entry_point() -> None:
    tree = ast.parse((ROOT / "src/training/g8_f_f2_closeout.py").read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "step" not in calls
    assert "backward" not in calls
    assert "train" not in calls
    text = (ROOT / "tools/closeout_g8_f_f2.py").read_text(encoding="utf-8")
    assert "--start" not in text
    assert "--resume" not in text
    assert "optimizer.step" not in text


def test_selection_requires_all_twenty_epochs() -> None:
    with pytest.raises(F2CloseoutHold, match="length"):
        select_best(_validation([0.8] * 19))
