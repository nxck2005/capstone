"""ER-9 pre-science outage and SR-18 compatibility regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.classical.outage import load_outage_policy
from baseline.classical.records import per_image_schema
from config.params import REPO_ROOT
from evaluation.er9_campaign import OUTAGE_POLICY_PATH, score_er9_outage


def test_authenticated_br13_outage_policy_and_class_match() -> None:
    policy = load_outage_policy(OUTAGE_POLICY_PATH, expected_dataset="imagenette160")
    matching = score_er9_outage(policy, policy.selected_class, failure_reason="decode_failure")
    mismatching = score_er9_outage(policy, (policy.selected_class + 1) % policy.class_count, failure_reason="decode_failure")
    assert matching == {"prediction": policy.selected_class, "correct": True, "delivered": False, "failure_reason": "decode_failure", "outage_reason": "decode_failure"}
    assert mismatching["correct"] is False and mismatching["delivered"] is False


def test_malformed_outage_artifact_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "outage.json"
    value = json.loads(OUTAGE_POLICY_PATH.read_bytes())
    value["selected_class"] = 99
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_outage_policy(path, expected_dataset="imagenette160")


def test_er9_rows_use_exact_sr18_schema() -> None:
    assert tuple(per_image_schema()) == (
        "run_id", "pair_id", "noise_id", "analysis_cell_id", "dataset", "dataset_version", "split",
        "stable_sample_id", "bw_ratio", "test_snr_db", "true_label", "pred_label", "correct",
        "outage", "outage_reason", "source_bytes",
    )
    assert REPO_ROOT.name == "capstone"
