"""Successor-v8 custody after the failed pre-candidate ER-12 startup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import runtime.source_epochs as epochs
from runtime.source_epochs import (
    W10_V4_MANIFEST_ID,
    W10_V4_SOURCE_PATH,
    W10_V5_MANIFEST_ID,
    W10_V5_SOURCE_PATH,
    W10_V6_MANIFEST_ID,
    W10_V6_SOURCE_PATH,
    W10_V7_MANIFEST_ID,
    W10_V7_MANIFEST_KIND,
    W10_V7_MANIFEST_SHA256,
    W10_V7_SOURCE_COMMIT,
    W10_V7_SOURCE_PATH,
    W10_V8_MANIFEST_KIND,
    W10_V8_REPAIR_REASON,
    W10_V8_SOURCE_PATH,
    W10_V8_TRANSITION_KIND,
    SourceEpochHold,
    assert_successor_lineage,
    assert_w10_manifest_contract,
    build_w10_manifest_v8,
    load_w10_manifest,
    predecessor_binding,
    successor_path,
)

REPO = Path(__file__).resolve().parents[1]
V8_PATH = REPO / W10_V8_SOURCE_PATH


def _build_candidate_v8() -> dict:
    if V8_PATH.is_file():
        return load_w10_manifest(REPO, live=False, epoch="v8")

    predecessor = load_w10_manifest(REPO, live=False, epoch="v7")
    original = epochs.build_manifest

    def fake_build_manifest(root, *, source_commit, relevant_config_paths):
        del root, relevant_config_paths
        result = dict(predecessor)
        result.pop("manifest_id")
        result.pop("transition_from_v6")
        result.pop("pre_future_work_state")
        result["source_commit"] = source_commit
        result["manifest_id"] = "mock-source-manifest"
        return result

    epochs.build_manifest = fake_build_manifest
    try:
        return build_w10_manifest_v8(REPO, source_commit="a" * 40)
    finally:
        epochs.build_manifest = original


@pytest.fixture(scope="module")
def v8_manifest() -> dict:
    return _build_candidate_v8()


def test_v8_binds_the_exact_immutable_v7_after_startup_failure(v8_manifest: dict) -> None:
    assert v8_manifest["manifest_kind"] == W10_V8_MANIFEST_KIND
    assert_w10_manifest_contract(v8_manifest)
    transition = v8_manifest["transition_from_v7"]
    assert transition["path"] == W10_V7_SOURCE_PATH
    assert transition["manifest_kind"] == W10_V7_MANIFEST_KIND
    assert transition["manifest_id"] == W10_V7_MANIFEST_ID
    assert transition["sha256"] == W10_V7_MANIFEST_SHA256
    assert transition["source_commit"] == W10_V7_SOURCE_COMMIT
    assert transition["transition_kind"] == W10_V8_TRANSITION_KIND
    assert transition["repair_reason"] == W10_V8_REPAIR_REASON
    assert hashlib.sha256((REPO / W10_V7_SOURCE_PATH).read_bytes()).hexdigest() == W10_V7_MANIFEST_SHA256
    assert successor_path(REPO, epoch="v8") == V8_PATH
    assert epochs.epoch_for_kind(W10_V8_MANIFEST_KIND) == "v8"
    assert epochs.manifest_prefix(W10_V8_MANIFEST_KIND) == "w10downstreamsourcev8-"
    assert predecessor_binding(v8_manifest) == transition


def test_v8_preserves_truthful_pre_future_work_state(v8_manifest: dict) -> None:
    assert "pre_science_state" not in v8_manifest
    assert "superseded_successor" not in v8_manifest
    assert v8_manifest["pre_future_work_state"] == epochs.W10_V8_PRE_FUTURE_WORK_STATE
    state = v8_manifest["pre_future_work_state"]
    assert state["papr_constrained_training_runs"] == 1
    assert state["papr_lifecycle_closed"] is True
    assert state["jpeg_validation_selection_count"] == 1
    assert state["jpeg_validation_selection_closed"] is True
    assert state["er12_validation_selection_count"] == 0
    assert state["w10_authority_frozen"] is False
    assert state["w10_scientific_units"] == 0
    assert state["g12_freeze_manifest"] is False
    assert state["test"] == "SEALED"
    assert state["test_access"] == 0


def test_v8_contract_rejects_changed_transition_or_frontier(v8_manifest: dict) -> None:
    bad_transition = dict(v8_manifest)
    bad_transition["transition_from_v7"] = {
        **v8_manifest["transition_from_v7"],
        "er12_validation_selection_count": 1,
    }
    with pytest.raises(SourceEpochHold, match="post-failed-ER-12-startup/pre-candidate-selection"):
        assert_w10_manifest_contract(bad_transition)

    bad_state = dict(v8_manifest)
    bad_state["pre_future_work_state"] = {
        **v8_manifest["pre_future_work_state"],
        "test_access": 1,
    }
    with pytest.raises(SourceEpochHold, match="v8 pre-future-work state differs"):
        assert_w10_manifest_contract(bad_state)


@pytest.mark.parametrize(
    "relative",
    (
        epochs.W10_ER12_SELECTION_SOURCE_PATH,
        epochs.W10_AUTHORITY_SOURCE_PATH,
        epochs.W10_RUNTIME_SOURCE_PATH,
        epochs.W10_CLOSEOUT_SOURCE_PATH,
        epochs.G12_FREEZE_MANIFEST_SOURCE_PATH,
    ),
)
def test_v8_freeze_rejects_future_artifacts(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    if relative == epochs.W10_RUNTIME_SOURCE_PATH:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    with pytest.raises(SourceEpochHold, match="freeze boundary"):
        epochs.assert_v8_freeze_boundary(tmp_path)


def test_v8_freeze_rejects_test_access(tmp_path: Path) -> None:
    path = tmp_path / epochs.PAPR_COMPLETION_SOURCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"test":"SEALED","test_access":1}\n')
    with pytest.raises(SourceEpochHold, match="test access"):
        epochs.assert_v8_freeze_boundary(tmp_path)


def test_v8_lineage_walks_through_v7_v6_v5_to_historical_v4(v8_manifest: dict) -> None:
    for epoch, expected_id in (
        ("v7", W10_V7_MANIFEST_ID),
        ("v6", W10_V6_MANIFEST_ID),
        ("v5", W10_V5_MANIFEST_ID),
        ("v4", W10_V4_MANIFEST_ID),
    ):
        historical = load_w10_manifest(REPO, live=False, epoch=epoch)
        assert historical["manifest_id"] == expected_id
        assert_successor_lineage(REPO, historical, successor=v8_manifest)
    assert v8_manifest["transition_from_v7"]["manifest_id"] == W10_V7_MANIFEST_ID
    assert load_w10_manifest(REPO, live=False, epoch="v7")["transition_from_v6"]["path"] == W10_V6_SOURCE_PATH
    assert load_w10_manifest(REPO, live=False, epoch="v6")["transition_from_v5"]["path"] == W10_V5_SOURCE_PATH
    assert load_w10_manifest(REPO, live=False, epoch="v5")["transition_from_v4"]["manifest_id"] == W10_V4_MANIFEST_ID


def test_active_manifest_prefers_v8(tmp_path: Path) -> None:
    path = successor_path(tmp_path, epoch="v8")
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    assert epochs.active_manifest_path(tmp_path) == path
