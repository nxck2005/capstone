"""Successor-v6 custody and historical-lineage gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import runtime.source_epochs as epochs
from evaluation.w10_jpeg_carrier import (
    JPEG_CARRIER_DESCRIPTOR_PATH,
    JPEG_CARRIER_PATH,
    JPEG_SELECTION_CONTRACT_SHA256,
    JPEG_SELECTION_ID,
    JPEG_SELECTION_RAW_BYTES,
    JPEG_SELECTION_RAW_SHA256,
    JPEG_SELECTION_SOURCE_RECORD,
)
from evaluation.w10_selections import er12_contract_sha256, er12_selection_contract
from runtime.source_epochs import (
    W10_V4_MANIFEST_ID,
    W10_V5_MANIFEST_ID,
    W10_V6_MANIFEST_KIND,
    W10_V6_SOURCE_PATH,
    SourceEpochHold,
    active_manifest_path,
    assert_clean_active_source_closure,
    assert_successor_lineage,
    assert_w10_manifest_contract,
    load_w10_manifest,
    source_record,
    successor_path,
)

REPO = Path(__file__).resolve().parents[1]
V6_PATH = REPO / W10_V6_SOURCE_PATH


pytestmark = pytest.mark.skipif(
    not V6_PATH.is_file(),
    reason="successor-v6 is added only after the carrier publication commit",
)


def _v6() -> dict:
    return load_w10_manifest(REPO, live=False, epoch="v6")


def test_v6_records_one_closed_jpeg_and_keeps_future_work_unopened() -> None:
    value = _v6()
    assert value["manifest_kind"] == W10_V6_MANIFEST_KIND
    assert active_manifest_path(REPO) == V6_PATH
    assert value["pre_future_work_state"] == epochs.W10_V6_PRE_FUTURE_WORK_STATE
    transition = value["transition_from_v5"]
    assert transition["jpeg_validation_selection_count"] == 1
    assert transition["jpeg_validation_selection_closed"] is True
    assert transition["er12_validation_selection_count"] == 0
    assert transition["w10_authority_frozen"] is False
    assert transition["w10_scientific_units"] == 0
    assert transition["g12_freeze_manifest"] is False
    assert transition["test_access"] == 0
    assert transition["jpeg_validation_selection_id"] == JPEG_SELECTION_ID
    assert transition["jpeg_validation_selection_contract_sha256"] == JPEG_SELECTION_CONTRACT_SHA256
    assert transition["jpeg_validation_selection_raw_sha256"] == JPEG_SELECTION_RAW_SHA256
    assert transition["jpeg_validation_selection_raw_bytes"] == JPEG_SELECTION_RAW_BYTES
    assert transition["jpeg_validation_selection_source_epoch"] == JPEG_SELECTION_SOURCE_RECORD
    assert transition["jpeg_validation_selection_carrier_path"] == JPEG_CARRIER_PATH
    assert transition["jpeg_validation_selection_carrier_descriptor_path"] == JPEG_CARRIER_DESCRIPTOR_PATH
    assert transition["jpeg_validation_selection_carrier_sha256"]
    assert transition["jpeg_validation_selection_carrier_descriptor_id"].startswith("w10jpegcarrier-")
    assert transition["jpeg_validation_selection_carrier_descriptor_sha256"]


def test_v6_rejects_a_zero_jpeg_count() -> None:
    value = _v6()
    mutated = dict(value)
    mutated["transition_from_v5"] = {**value["transition_from_v5"], "jpeg_validation_selection_count": 0}
    with pytest.raises(SourceEpochHold, match="JPEG selection count"):
        assert_w10_manifest_contract(mutated)


def test_v6_rejects_changed_v5_bytes() -> None:
    value = _v6()
    mutated = dict(value["transition_from_v5"])
    mutated["sha256"] = "0" * 64
    with pytest.raises(SourceEpochHold, match="predecessor bytes"):
        epochs._require_w10_v6_transition(REPO, {**value, "transition_from_v5": mutated})


@pytest.mark.parametrize(
    "relative",
    (
        epochs.W10_ER12_SELECTION_SOURCE_PATH,
        epochs.W10_AUTHORITY_SOURCE_PATH,
        epochs.W10_CLOSEOUT_SOURCE_PATH,
        epochs.W10_RUNTIME_SOURCE_PATH,
        epochs.G12_FREEZE_MANIFEST_SOURCE_PATH,
    ),
)
def test_v6_freeze_rejects_forbidden_future_artifacts(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    if relative == epochs.W10_RUNTIME_SOURCE_PATH:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    with pytest.raises(SourceEpochHold, match="freeze boundary"):
        epochs.assert_v6_freeze_boundary(tmp_path)


def test_v6_freeze_rejects_nonzero_test_access(tmp_path: Path) -> None:
    path = tmp_path / epochs.PAPR_COMPLETION_SOURCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"test": "SEALED", "test_access": 1}).encode())
    with pytest.raises(SourceEpochHold, match="test access"):
        epochs.assert_v6_freeze_boundary(tmp_path)


def test_lineage_keeps_papr_at_v4_and_jpeg_at_v5() -> None:
    active = _v6()
    v4 = load_w10_manifest(REPO, live=False, epoch="v4")
    v5 = load_w10_manifest(REPO, live=False, epoch="v5")
    assert v4["manifest_id"] == W10_V4_MANIFEST_ID
    assert v5["manifest_id"] == W10_V5_MANIFEST_ID
    assert_successor_lineage(REPO, v4, successor=active)
    assert_successor_lineage(REPO, v5, successor=active)
    assert active["transition_from_v5"]["manifest_id"] == W10_V5_MANIFEST_ID
    assert source_record(REPO, v5, path=successor_path(REPO, epoch="v5")) == JPEG_SELECTION_SOURCE_RECORD


def test_future_er12_contract_uses_active_v6() -> None:
    active = _v6()
    expected = source_record(REPO, active, path=V6_PATH)
    identity = er12_selection_contract().identity()
    assert identity["source_epoch"] == expected
    assert er12_contract_sha256() != "3e395dc6ead072ec067372952acc149da36e3bebd3a651f18f43ea15e67bf824"


def test_protected_source_drift_after_v6_fails_closed(monkeypatch) -> None:
    active = _v6()
    monkeypatch.setattr(
        epochs,
        "working_tree_source_differences",
        lambda root: {"unstaged": ["src/protected.py"], "staged": [], "untracked": []},
    )
    with pytest.raises(epochs.SourceEpochHold, match="protected working-tree source drift"):
        assert_clean_active_source_closure(REPO, active)
