"""Successor-v7 static-policy repair and post-JPEG frontier gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import runtime.source_epochs as epochs
from evaluation.w10_jpeg_carrier import (
    JPEG_CARRIER_PATH,
    JPEG_SELECTION_ID,
    JPEG_SELECTION_RAW_BYTES,
    JPEG_SELECTION_RAW_SHA256,
    JPEG_SELECTION_SOURCE_RECORD,
    check_jpeg_carrier,
)
from evaluation.w10_selections import er12_contract_sha256, er12_selection_contract
from runtime.source_epochs import (
    PAPR_COMPLETION_SOURCE_PATH,
    W10_V4_MANIFEST_ID,
    W10_V4_SOURCE_COMMIT,
    W10_V4_SOURCE_PATH,
    W10_V5_MANIFEST_ID,
    W10_V5_SOURCE_PATH,
    W10_V6_MANIFEST_ID,
    W10_V6_MANIFEST_KIND,
    W10_V6_MANIFEST_SHA256,
    W10_V6_SOURCE_COMMIT,
    W10_V6_SOURCE_PATH,
    W10_V7_MANIFEST_KIND,
    W10_V7_REPAIR_REASON,
    W10_V7_SOURCE_PATH,
    W10_V7_TRANSITION_KIND,
    SourceEpochHold,
    assert_clean_active_source_closure,
    assert_successor_lineage,
    assert_w10_manifest_contract,
    load_w10_manifest,
    source_record,
    successor_path,
)

REPO = Path(__file__).resolve().parents[1]
V7_PATH = REPO / W10_V7_SOURCE_PATH


def _build_candidate_v7() -> dict:
    """Exercise the v7 builder before freeze without changing the clean-source rule."""

    if V7_PATH.exists():
        return load_w10_manifest(REPO, live=False, epoch="v7")
    v6 = load_w10_manifest(REPO, live=False, epoch="v6")
    base = dict(v6)
    base.pop("manifest_id")
    original = epochs.build_manifest

    def fake_build_manifest(root, *, source_commit, relevant_config_paths):
        del root, relevant_config_paths
        result = dict(base)
        result["source_commit"] = source_commit
        result["manifest_id"] = "mock-source-manifest"
        return result

    epochs.build_manifest = fake_build_manifest
    try:
        return epochs.build_w10_manifest_v7(REPO, source_commit=epochs.W10_V6_SOURCE_COMMIT)
    finally:
        epochs.build_manifest = original


@pytest.fixture(scope="module")
def v7_manifest() -> dict:
    return _build_candidate_v7()


def test_v7_binds_the_exact_v6_manifest(v7_manifest: dict) -> None:
    assert v7_manifest["manifest_kind"] == W10_V7_MANIFEST_KIND
    assert_w10_manifest_contract(v7_manifest)
    transition = v7_manifest["transition_from_v6"]
    assert transition["path"] == W10_V6_SOURCE_PATH
    assert transition["manifest_kind"] == W10_V6_MANIFEST_KIND
    assert transition["manifest_id"] == W10_V6_MANIFEST_ID
    assert transition["sha256"] == W10_V6_MANIFEST_SHA256
    assert transition["source_commit"] == W10_V6_SOURCE_COMMIT
    assert transition["transition_kind"] == W10_V7_TRANSITION_KIND
    assert transition["repair_reason"] == W10_V7_REPAIR_REASON
    assert hashlib.sha256((REPO / W10_V6_SOURCE_PATH).read_bytes()).hexdigest() == W10_V6_MANIFEST_SHA256
    assert epochs.successor_path(REPO, epoch="v7") == V7_PATH
    assert epochs.epoch_for_kind(W10_V7_MANIFEST_KIND) == "v7"
    assert epochs.manifest_prefix(W10_V7_MANIFEST_KIND) == "w10downstreamsourcev7-"
    assert epochs.predecessor_binding(v7_manifest) == transition


def test_v7_records_post_jpeg_state_without_zero_science_claim(v7_manifest: dict) -> None:
    assert "pre_science_state" not in v7_manifest
    assert "superseded_successor" not in v7_manifest
    assert v7_manifest["pre_future_work_state"] == epochs.W10_V7_PRE_FUTURE_WORK_STATE
    state = v7_manifest["pre_future_work_state"]
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


def test_v7_contract_rejects_changed_jpeg_or_er12_counts(v7_manifest: dict) -> None:
    for field, bad_value in (
        ("jpeg_validation_selection_count", 0),
        ("jpeg_validation_selection_closed", False),
        ("er12_validation_selection_count", 1),
    ):
        mutated = dict(v7_manifest)
        mutated["transition_from_v6"] = {**v7_manifest["transition_from_v6"], field: bad_value}
        with pytest.raises(SourceEpochHold, match="post-JPEG/pre-ER-12"):
            assert_w10_manifest_contract(mutated)


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
def test_v7_freeze_rejects_existing_future_artifacts(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    if relative == epochs.W10_RUNTIME_SOURCE_PATH:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    with pytest.raises(SourceEpochHold, match="freeze boundary"):
        epochs.assert_v7_freeze_boundary(tmp_path)


def test_v7_freeze_rejects_raw_jpeg_json_and_test_access(tmp_path: Path) -> None:
    from evaluation.w10_jpeg_carrier import JPEG_SELECTION_PATH

    raw_path = tmp_path / JPEG_SELECTION_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"{}\n")
    with pytest.raises(SourceEpochHold, match="carrier-only boundary"):
        epochs.assert_v7_freeze_boundary(tmp_path)

    raw_path.unlink()
    papr_path = tmp_path / PAPR_COMPLETION_SOURCE_PATH
    papr_path.parent.mkdir(parents=True, exist_ok=True)
    papr_path.write_text(json.dumps({"test": "SEALED", "test_access": 1}))
    with pytest.raises(SourceEpochHold, match="test access"):
        epochs.assert_v7_freeze_boundary(tmp_path)


def test_v7_lineage_reaches_v6_v5_and_v4(v7_manifest: dict) -> None:
    v6 = load_w10_manifest(REPO, live=False, epoch="v6")
    v5 = load_w10_manifest(REPO, live=False, epoch="v5")
    v4 = load_w10_manifest(REPO, live=False, epoch="v4")
    assert v6["manifest_id"] == W10_V6_MANIFEST_ID
    assert v5["manifest_id"] == W10_V5_MANIFEST_ID
    assert v4["manifest_id"] == W10_V4_MANIFEST_ID
    assert_successor_lineage(REPO, v6, successor=v7_manifest)
    assert_successor_lineage(REPO, v5, successor=v7_manifest)
    assert_successor_lineage(REPO, v4, successor=v7_manifest)
    assert v6["transition_from_v5"]["manifest_id"] == W10_V5_MANIFEST_ID
    assert v5["transition_from_v4"]["manifest_id"] == W10_V4_MANIFEST_ID


def test_papr_and_jpeg_sources_remain_historical(v7_manifest: dict) -> None:
    transition = v7_manifest["transition_from_v6"]
    assert transition["jpeg_validation_selection_id"] == JPEG_SELECTION_ID
    assert transition["jpeg_validation_selection_raw_sha256"] == JPEG_SELECTION_RAW_SHA256
    assert transition["jpeg_validation_selection_raw_bytes"] == JPEG_SELECTION_RAW_BYTES
    assert transition["jpeg_validation_selection_source_epoch"] == JPEG_SELECTION_SOURCE_RECORD
    assert JPEG_SELECTION_SOURCE_RECORD["path"] == W10_V5_SOURCE_PATH
    assert JPEG_SELECTION_SOURCE_RECORD["manifest_id"] == W10_V5_MANIFEST_ID
    assert W10_V4_SOURCE_COMMIT == "982688ab119959d3b861af302f896c8e952a4c60"


@pytest.mark.skipif(not V7_PATH.is_file(), reason="active-v7 PAPR authority check runs after the immutable v7 freeze")
def test_papr_authority_remains_bound_to_v4() -> None:
    from training.papr_lifecycle import verify_papr_authority

    authority = verify_papr_authority(REPO)
    assert authority["source_manifest"] == {
        "path": W10_V4_SOURCE_PATH,
        "manifest_id": W10_V4_MANIFEST_ID,
        "sha256": epochs.W10_V4_MANIFEST_SHA256,
    }
    assert authority["source_commit"] == W10_V4_SOURCE_COMMIT


def test_carrier_identity_is_unchanged_and_raw_json_absent(v7_manifest: dict) -> None:
    loaded = check_jpeg_carrier(REPO)
    assert loaded.value["selection_id"] == JPEG_SELECTION_ID
    assert loaded.value["source_epoch"] == JPEG_SELECTION_SOURCE_RECORD
    assert loaded.provenance["raw_sha256"] == JPEG_SELECTION_RAW_SHA256
    assert loaded.provenance["raw_bytes"] == 133540835
    assert loaded.provenance["carrier_path"] == JPEG_CARRIER_PATH
    assert loaded.provenance["carrier_sha256"] == "460443de59e88de488ffa345976d04cf3c442c4b8256a92c05869da490c0ae1e"
    assert loaded.provenance["carrier_bytes"] == 1032328
    assert loaded.provenance["carrier_descriptor_id"] == "w10jpegcarrier-0eaa11ceb83e4fd2d737d7561127ca788d50b692105d8bddbfb2fac9a46b44f9"
    assert not (REPO / "results/learned/w10/jpeg_validation_selection.json").exists()


@pytest.mark.skipif(not V7_PATH.is_file(), reason="active ER-12 source binding is v7 after the immutable freeze")
def test_active_er12_contract_uses_v7(v7_manifest: dict) -> None:
    active = load_w10_manifest(REPO, live=True)
    expected = source_record(REPO, active, path=V7_PATH)
    identity = er12_selection_contract().identity()
    assert active["manifest_kind"] == W10_V7_MANIFEST_KIND
    assert identity["source_epoch"] == expected
    assert er12_contract_sha256() != "39584f9a9dee921b7c06d80ea62f6a584df9ebb1eb8c635f9e10f770483cc841"
    assert v7_manifest["manifest_id"] == active["manifest_id"]


def test_protected_source_drift_after_v7_fails_closed(v7_manifest: dict, monkeypatch) -> None:
    monkeypatch.setattr(epochs, "committed_source_differences", lambda root, source_commit, head=None: [])
    monkeypatch.setattr(
        epochs,
        "working_tree_source_differences",
        lambda root: {"unstaged": ["src/protected.py"], "staged": [], "untracked": []},
    )
    with pytest.raises(SourceEpochHold, match="protected working-tree source drift"):
        assert_clean_active_source_closure(REPO, v7_manifest)
