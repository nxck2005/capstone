"""W10 successor source-epoch custody: zero-science history and the post-PAPR v5 repair."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from evaluation.downstream_v4 import immutable_write
from runtime.source_epochs import (
    G12_FREEZE_MANIFEST_SOURCE_PATH,
    PAPR_COMPLETION_ID,
    PAPR_COMPLETION_SHA256,
    PAPR_COMPLETION_SOURCE_PATH,
    W10_AUTHORITY_SOURCE_PATH,
    W10_CLOSEOUT_SOURCE_PATH,
    W10_ER12_SELECTION_SOURCE_PATH,
    W10_JPEG_SELECTION_SOURCE_PATH,
    W10_RELEVANT_CONFIG_PATHS,
    W10_RUNTIME_SOURCE_PATH,
    W10_V2_MANIFEST_KIND,
    W10_V2_SOURCE_PATH,
    W10_V3_MANIFEST_KIND,
    W10_V3_SOURCE_PATH,
    W10_V4_MANIFEST_ID,
    W10_V4_MANIFEST_KIND,
    W10_V4_MANIFEST_SHA256,
    W10_V4_SOURCE_COMMIT,
    W10_V4_SOURCE_PATH,
    W10_V5_MANIFEST_KIND,
    W10_V5_PRE_FUTURE_WORK_STATE,
    W10_V5_REPAIR_REASON,
    W10_V5_SOURCE_PATH,
    W10_V5_TRANSITION_KIND,
    W10_V6_MANIFEST_KIND,
    W10_V6_SOURCE_PATH,
    W10_V7_MANIFEST_KIND,
    W10_V7_SOURCE_PATH,
    W10_V8_MANIFEST_KIND,
    W10_V8_SOURCE_PATH,
    W10_V9_MANIFEST_KIND,
    W10_V9_SOURCE_PATH,
    SourceEpochHold,
    active_manifest_path,
    assert_v5_freeze_boundary,
    assert_w10_manifest_contract,
    build_w10_manifest_v5,
    load_w10_manifest,
)
from training.deterministic_core import canonical_bytes

REPO = Path(__file__).resolve().parents[1]
V1_PATH = REPO / "results/learned/w10/w10_downstream_source_manifest.json"
V2_PATH = REPO / W10_V2_SOURCE_PATH
V3_PATH = REPO / W10_V3_SOURCE_PATH
V4_PATH = REPO / W10_V4_SOURCE_PATH
V5_PATH = REPO / W10_V5_SOURCE_PATH


def test_successor_v1_v2_v3_and_v4_bytes_remain_immutable() -> None:
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == (
        "dedc2c894dad47b24ead8cdbc22513d2dbe54945e41415e4e9d3692421eab3ea"
    )
    assert hashlib.sha256(V2_PATH.read_bytes()).hexdigest() == (
        "d2ce8f2ee88d97a11acdae5bc60e1dcdb35e5181c2a017e0a68a66e292f975e3"
    )
    assert hashlib.sha256(V3_PATH.read_bytes()).hexdigest() == (
        "0ab6be914f5a02e772b10827e61785d6699c6dd4c3b916a6c2caca484d3e40f8"
    )
    assert hashlib.sha256(V4_PATH.read_bytes()).hexdigest() == W10_V4_MANIFEST_SHA256


def test_successor_v4_remains_exact_zero_science_history_until_v5() -> None:
    value = load_w10_manifest(REPO, live=False, epoch="v4")
    assert value["manifest_kind"] == W10_V4_MANIFEST_KIND
    assert value["manifest_id"] == W10_V4_MANIFEST_ID
    assert value["source_commit"] == W10_V4_SOURCE_COMMIT
    predecessor = value["superseded_successor"]
    v3 = json.loads(V3_PATH.read_bytes())
    assert predecessor == {
        "path": W10_V3_SOURCE_PATH,
        "manifest_kind": W10_V3_MANIFEST_KIND,
        "manifest_id": v3["manifest_id"],
        "sha256": hashlib.sha256(V3_PATH.read_bytes()).hexdigest(),
        "source_commit": v3["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }
    assert value["pre_science_state"] == {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    active = active_manifest_path(REPO)
    v9_path = REPO / W10_V9_SOURCE_PATH
    v8_path = REPO / W10_V8_SOURCE_PATH
    v7_path = REPO / W10_V7_SOURCE_PATH
    v6_path = REPO / W10_V6_SOURCE_PATH
    if v9_path.is_file():
        assert active == v9_path
        assert load_w10_manifest(REPO, live=True)["manifest_kind"] == W10_V9_MANIFEST_KIND
    elif v8_path.is_file():
        assert active == v8_path
        assert load_w10_manifest(REPO, live=True)["manifest_kind"] == W10_V8_MANIFEST_KIND
    elif v7_path.is_file():
        assert active == v7_path
        assert load_w10_manifest(REPO, live=True)["manifest_kind"] == W10_V7_MANIFEST_KIND
    elif v6_path.is_file():
        assert active == v6_path
    elif V5_PATH.is_file():
        assert active == V5_PATH
        assert load_w10_manifest(REPO, live=True)["manifest_kind"] == W10_V5_MANIFEST_KIND
    else:
        assert active == V4_PATH


def test_successor_v3_remains_exact_superseded_history() -> None:
    value = load_w10_manifest(REPO, live=False, epoch="v3")
    predecessor = value["superseded_successor"]
    v2 = json.loads(V2_PATH.read_bytes())
    assert predecessor == {
        "path": W10_V2_SOURCE_PATH,
        "manifest_kind": W10_V2_MANIFEST_KIND,
        "manifest_id": v2["manifest_id"],
        "sha256": hashlib.sha256(V2_PATH.read_bytes()).hexdigest(),
        "source_commit": v2["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }
    assert value["pre_science_state"] == {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }


def _post_papr_repo(root: Path) -> str:
    """A minimal clean repository carrying the exact immutable v4/PAPR bytes."""

    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    for name in ("src", "tools", "configs", "spec", "tests"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "placeholder.txt").write_text(f"{name}\n")
    for relative in W10_RELEVANT_CONFIG_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")
    (root / "requirements-pascal.lock").write_text("lock\n")
    subprocess.run(["git", "add", "--", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=root, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    for source, relative in (
        (V4_PATH, W10_V4_SOURCE_PATH),
        (REPO / PAPR_COMPLETION_SOURCE_PATH, PAPR_COMPLETION_SOURCE_PATH),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return head


def _build_v5(root: Path) -> dict:
    head = _post_papr_repo(root)
    value = build_w10_manifest_v5(root, source_commit=head)
    immutable_write(root / W10_V5_SOURCE_PATH, value)
    return value


def test_successor_v5_records_the_honest_post_papr_transition(tmp_path: Path) -> None:
    value = _build_v5(tmp_path)
    assert value["manifest_kind"] == W10_V5_MANIFEST_KIND
    assert value["manifest_id"].startswith("w10downstreamsourcev5-")
    assert "pre_science_state" not in value
    assert "superseded_successor" not in value
    assert value["pre_future_work_state"] == W10_V5_PRE_FUTURE_WORK_STATE
    transition = value["transition_from_v4"]
    assert transition == {
        "path": W10_V4_SOURCE_PATH,
        "manifest_kind": W10_V4_MANIFEST_KIND,
        "manifest_id": W10_V4_MANIFEST_ID,
        "sha256": W10_V4_MANIFEST_SHA256,
        "source_commit": W10_V4_SOURCE_COMMIT,
        "transition_kind": W10_V5_TRANSITION_KIND,
        "predecessor_papr_constrained_training_runs": 1,
        "predecessor_papr_completion_path": PAPR_COMPLETION_SOURCE_PATH,
        "predecessor_papr_completion_id": PAPR_COMPLETION_ID,
        "predecessor_papr_completion_sha256": PAPR_COMPLETION_SHA256,
        "jpeg_validation_selection_count": 0,
        "er12_validation_selection_count": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
        "repair_reason": W10_V5_REPAIR_REASON,
    }
    loaded = load_w10_manifest(tmp_path, live=False, epoch="v5")
    assert loaded["manifest_id"] == value["manifest_id"]
    assert active_manifest_path(tmp_path) == tmp_path / W10_V5_SOURCE_PATH
    assert load_w10_manifest(tmp_path, live=True)["manifest_id"] == value["manifest_id"]
    assert assert_v5_freeze_boundary(tmp_path) is None


def test_successor_v5_rejects_a_zero_papr_run_claim(tmp_path: Path) -> None:
    value = _build_v5(tmp_path / "state")
    value["pre_future_work_state"] = {
        **value["pre_future_work_state"],
        "papr_constrained_training_runs": 0,
    }
    with pytest.raises(SourceEpochHold, match="pre-future-work state"):
        assert_w10_manifest_contract(value)

    value = _build_v5(tmp_path / "transition")
    value["transition_from_v4"] = {
        **value["transition_from_v4"],
        "predecessor_papr_constrained_training_runs": 0,
    }
    with pytest.raises(SourceEpochHold, match="exactly one closed PAPR"):
        assert_w10_manifest_contract(value)


def test_successor_v5_rejects_before_science_supersession_semantics(tmp_path: Path) -> None:
    value = _build_v5(tmp_path / "supersession")
    value["superseded_successor"] = {
        **value["transition_from_v4"],
        "superseded_before_science": True,
    }
    with pytest.raises(SourceEpochHold, match="zero-science supersession"):
        assert_w10_manifest_contract(value)

    value = _build_v5(tmp_path / "missing_transition")
    value.pop("transition_from_v4")
    with pytest.raises(SourceEpochHold, match="transition record is missing"):
        assert_w10_manifest_contract(value)


def test_successor_v5_binds_exact_immutable_v4_bytes(tmp_path: Path) -> None:
    _build_v5(tmp_path)
    predecessor = json.loads((tmp_path / W10_V4_SOURCE_PATH).read_bytes())
    predecessor["source_commit"] = "0" * 40
    (tmp_path / W10_V4_SOURCE_PATH).write_bytes(canonical_bytes(predecessor))
    with pytest.raises(SourceEpochHold, match="predecessor bytes differ"):
        load_w10_manifest(tmp_path, live=False, epoch="v5")


def test_successor_v5_binds_the_terminal_papr_completion(tmp_path: Path) -> None:
    _build_v5(tmp_path)
    completion_path = tmp_path / PAPR_COMPLETION_SOURCE_PATH
    completion = json.loads(completion_path.read_bytes())
    completion["training_runs"] = 2
    body = dict(completion)
    body.pop("completion_id")
    completion["completion_id"] = "paprcompletion-" + hashlib.sha256(canonical_bytes(body)).hexdigest()
    completion_path.write_bytes(canonical_bytes(completion))
    with pytest.raises(SourceEpochHold, match="PAPR completion"):
        load_w10_manifest(tmp_path, live=False, epoch="v5")


@pytest.mark.parametrize(
    ("relative", "is_directory"),
    (
        (W10_JPEG_SELECTION_SOURCE_PATH, False),
        (W10_ER12_SELECTION_SOURCE_PATH, False),
        (W10_AUTHORITY_SOURCE_PATH, False),
        (W10_CLOSEOUT_SOURCE_PATH, False),
        (G12_FREEZE_MANIFEST_SOURCE_PATH, False),
        (W10_RUNTIME_SOURCE_PATH, True),
    ),
)
def test_successor_v5_freeze_boundary_rejects_existing_post_papr_work(
    tmp_path: Path, relative: str, is_directory: bool
) -> None:
    path = tmp_path / relative
    if is_directory:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    with pytest.raises(SourceEpochHold, match="freeze boundary"):
        assert_v5_freeze_boundary(tmp_path)


def test_successor_v5_builder_rejects_a_selection_that_already_exists(tmp_path: Path) -> None:
    head = _post_papr_repo(tmp_path)
    selection = tmp_path / W10_JPEG_SELECTION_SOURCE_PATH
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("{}\n")
    with pytest.raises(SourceEpochHold, match="freeze boundary"):
        build_w10_manifest_v5(tmp_path, source_commit=head)


def test_successor_v5_live_closure_fails_closed_on_protected_source_drift(tmp_path: Path) -> None:
    _build_v5(tmp_path)
    assert load_w10_manifest(tmp_path, live=True)["manifest_kind"] == W10_V5_MANIFEST_KIND
    (tmp_path / "src/placeholder.txt").write_text("changed\n")
    with pytest.raises(SourceEpochHold, match="source drift"):
        load_w10_manifest(tmp_path, live=True)


@pytest.mark.skipif(
    not (REPO / W10_V7_SOURCE_PATH).is_file(),
    reason="closed PAPR live-source closure is checked after the v7 static repair is frozen",
)
def test_closed_papr_authority_stays_bound_to_historical_v4() -> None:
    """PAPR executed under v4; a later successor must not rewrite that history."""

    from training.papr_lifecycle import verify_papr_authority

    authority = verify_papr_authority(REPO)
    assert authority["source_manifest"] == {
        "path": W10_V4_SOURCE_PATH,
        "manifest_id": W10_V4_MANIFEST_ID,
        "sha256": W10_V4_MANIFEST_SHA256,
    }
    assert authority["source_commit"] == W10_V4_SOURCE_COMMIT
    assert hashlib.sha256(V4_PATH.read_bytes()).hexdigest() == W10_V4_MANIFEST_SHA256
    active = load_w10_manifest(REPO, live=True)
    v8_path = REPO / W10_V8_SOURCE_PATH
    v7_path = REPO / W10_V7_SOURCE_PATH
    v6_path = REPO / W10_V6_SOURCE_PATH
    if v8_path.is_file():
        assert active["manifest_kind"] == W10_V8_MANIFEST_KIND
        assert active["transition_from_v7"]["path"] == W10_V7_SOURCE_PATH
        assert active["transition_from_v7"]["manifest_id"] == json.loads(v7_path.read_bytes())["manifest_id"]
    elif v7_path.is_file():
        assert active["manifest_kind"] == W10_V7_MANIFEST_KIND
        assert active["transition_from_v6"]["manifest_id"] == json.loads(v6_path.read_bytes())["manifest_id"]
    elif v6_path.is_file():
        assert active["manifest_kind"] == W10_V6_MANIFEST_KIND
        transition = active["transition_from_v5"]
        assert transition["manifest_id"] == json.loads(V5_PATH.read_bytes())["manifest_id"]
    elif V5_PATH.is_file():
        assert active["manifest_kind"] == W10_V5_MANIFEST_KIND
        assert active["manifest_id"] != W10_V4_MANIFEST_ID
        transition = active["transition_from_v4"]
        assert transition["manifest_id"] == W10_V4_MANIFEST_ID
        assert transition["sha256"] == W10_V4_MANIFEST_SHA256
        assert transition["source_commit"] == W10_V4_SOURCE_COMMIT
        assert transition["predecessor_papr_constrained_training_runs"] == 1
    else:
        assert active["manifest_id"] == W10_V4_MANIFEST_ID
