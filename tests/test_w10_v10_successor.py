"""Offline gates for the failed-v9 prefix and the ordinal-231 successor."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import evaluation.w10_backends as backends
import evaluation.w10_successor_v10 as v10
import run_w10_v10 as runner
import runtime.source_epochs as epochs
from evaluation.w10_evidence import per_image_relative_path, unit_relative_path
from evaluation.w10_rehearsal import execute, unit_evidence_requirement
from evaluation.w10_scope import work_units
from training.deterministic_core import canonical_bytes, canonical_sha256

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((REPO / "results/learned/w10/w10_rehearsal_authorization.json").read_bytes())
V9 = json.loads((REPO / "results/learned/w10/w10_continuation_authorization_v9.json").read_bytes())


def _authority() -> dict:
    return {
        **V9,
        "authority_kind": v10.V10_AUTHORITY_KIND,
        "authority_id": "w10continuationauthv10-synthetic",
        "historical_completed_ordinals": list(range(v10.V10_START)),
        "new_authorized_ordinals": list(range(v10.V10_START, v10.V10_STOP)),
        "v9_authority": {
            "authority_id": V9["authority_id"],
            "source_manifest_id": V9["source_manifest"]["manifest_id"],
        },
        "v9_custody": {"custody_id": "w10v9prefix-synthetic"},
    }


def test_reconstruction_aggregate_names_its_clean_classifier(monkeypatch) -> None:
    rows = [{"correct": False, "outage": False, "outage_reason": None} for _ in range(1000)]
    modes = []

    def fake_rows(*_args, **kwargs):
        modes.append(kwargs["mode"])
        return rows, [1.0] * len(rows)

    monkeypatch.setattr(backends, "_learned_rows", fake_rows)
    value = backends.recon_ablation_unit(
        object(), work_units()[231], model=object(), config=object(), checkpoint_id="frozen"
    )
    assert modes == ["recon_ablation"]
    assert value["primary_classifier_variant"] == "clean"
    assert value["binding"]["scorer"] == "frozen_g1_reference_classifier"
    assert ORIGINAL["scope"]["entries"][-1]["classifier_variants"] == ["clean"]


def test_v10_manifest_declares_three_epochs_and_rejects_wrong_frontier() -> None:
    value = dict(epochs.load_w10_manifest(REPO, live=False, epoch="v9"))
    value.pop("transition_from_v8")
    value["manifest_kind"] = epochs.W10_V10_MANIFEST_KIND
    value["pre_future_work_state"] = dict(epochs.W10_V10_PRE_FUTURE_WORK_STATE)
    value["transition_from_v9"] = {
        "transition_kind": "failed_v9_partial_science_to_v10_suffix",
        "v8_completed_ordinals": [0, 125],
        "v9_completed_ordinals": [126, 230],
        "failed_ordinal": 231,
        "path": epochs.W10_V9_SOURCE_PATH,
        "manifest_kind": epochs.W10_V9_MANIFEST_KIND,
    }
    value["manifest_id"] = epochs.W10_V10_MANIFEST_PREFIX + canonical_sha256({
        k: v for k, v in value.items() if k != "manifest_id"
    })
    epochs.assert_w10_manifest_contract(value)
    assert epochs.predecessor_binding(value) == value["transition_from_v9"]
    bad = copy.deepcopy(value)
    bad["transition_from_v9"]["v9_completed_ordinals"] = [126, 231]
    with pytest.raises(epochs.SourceEpochHold, match="historical frontier"):
        epochs.assert_w10_manifest_contract(bad)


def test_v10_manifest_builder_binds_the_failed_v9_custody(monkeypatch, tmp_path: Path) -> None:
    predecessor = epochs.load_w10_manifest(REPO, live=False, epoch="v9")
    transition = {
        "path": epochs.W10_V9_SOURCE_PATH,
        "manifest_kind": epochs.W10_V9_MANIFEST_KIND,
        "transition_kind": "failed_v9_partial_science_to_v10_suffix",
        "v8_completed_ordinals": [0, 125],
        "v9_completed_ordinals": [126, 230],
        "failed_ordinal": 231,
        "v9_custody_id": "w10v9prefix-frozen",
    }

    def fake_build(_root, *, source_commit, relevant_config_paths):
        assert tuple(relevant_config_paths) == epochs.W10_RELEVANT_CONFIG_PATHS
        value = dict(predecessor)
        value.pop("transition_from_v8")
        value.pop("pre_future_work_state")
        value["source_commit"] = source_commit
        return value

    monkeypatch.setattr(epochs, "build_manifest", fake_build)
    monkeypatch.setattr(epochs, "_v10_transition", lambda _root: transition)
    value = epochs.build_w10_manifest_v10(tmp_path, source_commit="a" * 40)
    epochs.assert_w10_manifest_contract(value)
    assert value["transition_from_v9"] == transition
    assert value["pre_future_work_state"] == epochs.W10_V10_PRE_FUTURE_WORK_STATE
    assert value["manifest_kind"] == epochs.W10_V10_MANIFEST_KIND
    assert not (tmp_path / epochs.W10_V10_SOURCE_PATH).exists()


def test_plan_and_launch_are_bound_to_231_251_and_seal() -> None:
    authority = _authority()
    plan = v10.build_v10_plan(authority)
    launch = v10.build_v10_launch(authority, plan)
    assert plan["v8_ordinals"] == [0, 125]
    assert plan["v9_ordinals"] == [126, 230]
    assert plan["v10_ordinals"] == [231, 251]
    assert launch["authorized_ordinals"] == list(range(231, 252))
    assert launch["test"] == "SEALED" and launch["test_access"] == 0
    assert work_units()[231]["system"] == "semantic_recon_ablation"
    assert work_units()[231]["bw_ratio"] == "r_1_6"
    assert work_units()[231]["snr_db"] == -8
    replay = copy.deepcopy(authority)
    replay["new_authorized_ordinals"] = list(range(230, 252))
    with pytest.raises(Exception, match="frontier"):
        v10.build_v10_plan(replay)
    changed = copy.deepcopy(plan)
    changed["test_access"] = 1
    with pytest.raises(Exception, match="plan"):
        v10.build_v10_launch(authority, changed)


def test_v10_authority_projects_v9_science_without_reassigning_history(tmp_path: Path, monkeypatch) -> None:
    source = {"manifest_kind": epochs.W10_V10_MANIFEST_KIND, "source_commit": "a" * 40}
    original = tmp_path / "results/learned/w10/w10_continuation_authorization_v9.json"
    original.parent.mkdir(parents=True)
    original.write_bytes(canonical_bytes(V9))
    custody_path = tmp_path / v10.V9_CUSTODY_PATH
    custody_path.write_bytes(b"custody\n")
    monkeypatch.setattr(v10, "verify_v9_authority_for_succession", lambda _root: ({}, V9))
    monkeypatch.setattr(v10, "load_v9_custody", lambda *_args, **_kwargs: {"custody_id": "w10v9prefix-frozen"})
    monkeypatch.setattr(v10, "source_record", lambda _root, _source: {"manifest_id": "w10v10-frozen"})
    body = v10.build_v10_authority(tmp_path, source)
    assert body["scope"] == V9["scope"] and body["bindings"] == V9["bindings"]
    assert body["v9_authority"]["authority_id"] == V9["authority_id"]
    assert body["historical_completed_ordinals"] == list(range(231))
    assert body["new_authorized_ordinals"] == list(range(231, 252))
    assert body["source_manifest"] == {"manifest_id": "w10v10-frozen"}


def test_authority_preflight_authenticates_gpu_without_writing(tmp_path: Path, monkeypatch) -> None:
    calls = []
    body = {"scope": {"frozen": True}, "cuda_mapping": {"logical_device": "cuda:0"}}
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "load_w10_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "build_v10_authority", lambda *_args: body)
    monkeypatch.setattr(runner, "authenticate_live_w9_pascal", lambda *_args, **_kwargs: calls.append("gpu") or {"environment": {"cuda_mapping": body["cuda_mapping"]}})
    monkeypatch.setattr(runner, "immutable_write", lambda *_args: pytest.fail("preflight wrote evidence"))
    assert runner.main(["authority", "--preflight"]) == 0
    assert calls == ["gpu"]


def test_execute_refuses_to_recreate_missing_historical_unit(tmp_path: Path) -> None:
    calls = []
    authority = _authority()
    with pytest.raises(Exception, match="historical W10 unit is missing"):
        execute(tmp_path, authority=authority, backend=lambda unit: calls.append(unit), expected_stable_ids=None)
    assert calls == []


def _fake_historical_layout(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    runtime = tmp_path / "runtime"
    (runtime / "units").mkdir(parents=True)
    (runtime / "per_image").mkdir()
    for unit in work_units()[:231]:
        (runtime / unit_relative_path(unit, "json")).touch()
    authority = _authority()
    authority["runtime_root"] = "runtime"
    monkeypatch.setattr(v10, "verify_historical_prefix", lambda *_args, **_kwargs: {"custody_id": "w10v9prefix-synthetic", "streams": []})
    monkeypatch.setattr(v10, "v8_continuation_custody", lambda *_args: {"streams": []})
    return runtime, authority


def test_evidence_set_rejects_missing_duplicate_and_wrong_authority(tmp_path: Path, monkeypatch) -> None:
    runtime, authority = _fake_historical_layout(tmp_path, monkeypatch)
    assert v10.verify_v10_evidence_set(tmp_path, authority=authority, stable_ids=[], phase="plan") == 0
    freeze = tmp_path / "results/freeze_manifest.json"
    freeze.parent.mkdir()
    freeze.touch()
    with pytest.raises(Exception, match="G12 is already open"):
        v10.verify_v10_evidence_set(tmp_path, authority=authority, stable_ids=[])
    freeze.unlink()
    missing = runtime / unit_relative_path(work_units()[230], "json")
    missing.unlink()
    with pytest.raises(Exception, match="gap or unexpected"):
        v10.verify_v10_evidence_set(tmp_path, authority=authority, stable_ids=[])
    missing.touch()
    duplicate = runtime / "units" / "230-extra.json"
    duplicate.touch()
    with pytest.raises(Exception, match="gap or unexpected"):
        v10.verify_v10_evidence_set(tmp_path, authority=authority, stable_ids=[])
    duplicate.unlink()
    next_path = runtime / unit_relative_path(work_units()[231], "json")
    next_path.write_bytes(canonical_bytes({"binding": {"execution_authority_id": "unauthorized"}}))
    monkeypatch.setattr(v10, "validate_unit", lambda *_args: None)
    with pytest.raises(Exception, match="wrong authority"):
        v10.verify_v10_evidence_set(tmp_path, authority=authority, stable_ids=[])


def test_v10_closeout_preserves_each_epoch_provenance(monkeypatch, tmp_path: Path) -> None:
    authority = _authority()
    values = [
        {"unit_id": f"unit-{i}", "binding": (
            {} if i < 126 else {"execution_authority_id": V9["authority_id"] if i < 231 else authority["authority_id"]}
        )}
        for i in range(252)
    ]
    monkeypatch.setattr(v10, "validate_unit", lambda *_args: None)
    monkeypatch.setattr(v10, "unit_manifest", lambda *_args: {"ordered_unit_ids_digest": "units"})
    monkeypatch.setattr(v10, "per_image_manifest", lambda *_args: {"stream_count": 357, "ordered_per_image_digest": "streams"})
    closeout, _, _ = v10.build_v10_closeout(tmp_path, values, authority)
    rows = closeout["ordinal_provenance"]
    assert rows[125]["source_manifest_id"] == epochs.W10_V8_MANIFEST_ID
    assert rows[126]["source_manifest_id"] == V9["source_manifest"]["manifest_id"]
    assert rows[230]["authority_id"] == V9["authority_id"]
    assert rows[231]["authority_id"] == authority["authority_id"]
    assert closeout["test"] == "SEALED" and closeout["test_access"] == 0
    values[230]["binding"]["execution_authority_id"] = authority["authority_id"]
    with pytest.raises(Exception, match="unit authority"):
        v10.build_v10_closeout(tmp_path, values, authority)


def test_custody_loader_rejects_unpinned_source_and_duplicate_rows(tmp_path: Path, monkeypatch) -> None:
    source = {"manifest_id": "source"}
    authority = {"authority_id": "authority"}
    auth_path = tmp_path / "results/learned/w10/w10_continuation_authorization_v9.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_bytes(b"authority\n")
    monkeypatch.setattr(v10, "source_record", lambda _root, item: {"manifest_id": item["manifest_id"]})
    units = [{"ordinal": i, "path": f"units/{i}.json"} for i in range(126, 231)]
    streams = [{"ordinal": 126 + i // 2, "path": f"per_image/{i}.json"} for i in range(168)]
    body = {
        "schema_version": 1, "artifact_role": "W10_V9_FAILED_PREFIX_CUSTODY",
        "status": "FAILED_PARTIAL_NOT_CLOSEOUT",
        "source": {"manifest_id": "source"},
        "authority": {"path": "results/learned/w10/w10_continuation_authorization_v9.json", "authority_id": "authority", "sha256": v10._sha(auth_path)},
        "v8_custody_id": epochs.W10_V8_CUSTODY_ID,
        "v8_custody_sha256": epochs.W10_V8_CUSTODY_SHA256,
        "completed_ordinals": [126, 230], "failed_ordinal": 231,
        "unit_count": 105, "stream_count": 168, "units": units, "streams": streams,
        "complete_prefix_digest": canonical_sha256({"v8_prefix_digest": epochs.W10_V8_PREFIX_DIGEST, "units": units, "streams": streams}),
        "validation_only": True, "test": "SEALED", "test_access": 0, "g12_unopened": True,
    }
    body["custody_id"] = "w10v9prefix-" + canonical_sha256(body)
    path = tmp_path / v10.V9_CUSTODY_PATH
    path.write_bytes(canonical_bytes(body))
    assert v10.load_v9_custody(tmp_path, source=source, authority=authority) == body
    with pytest.raises(Exception, match="source differs"):
        v10.load_v9_custody(tmp_path, source={"manifest_id": "other"}, authority=authority)
    duplicate = copy.deepcopy(body)
    duplicate["units"][1]["path"] = duplicate["units"][0]["path"]
    duplicate["complete_prefix_digest"] = canonical_sha256({
        "v8_prefix_digest": epochs.W10_V8_PREFIX_DIGEST, "units": duplicate["units"], "streams": duplicate["streams"],
    })
    duplicate["custody_id"] = "w10v9prefix-" + canonical_sha256({k: v for k, v in duplicate.items() if k != "custody_id"})
    path.write_bytes(canonical_bytes(duplicate))
    with pytest.raises(Exception, match="paths repeat"):
        v10.load_v9_custody(tmp_path, source=source, authority=authority)


def test_v9_authority_is_authenticated_at_its_recorded_epoch(monkeypatch) -> None:
    source, authority = v10.verify_v9_authority_for_succession(REPO)
    assert source["manifest_id"] == V9["source_manifest"]["manifest_id"]
    assert authority["authority_id"] == V9["authority_id"]
    original_read = v10.read_json

    def forged_read(path, label):
        value = original_read(path, label)
        if label == "W10 v9 authority":
            value["source_binding"]["source_commit"] = "0" * 40
            body = dict(value)
            body.pop("authority_id")
            value["authority_id"] = "w10continuationauth-" + canonical_sha256(body)
        return value

    monkeypatch.setattr(v10, "read_json", forged_read)
    with pytest.raises(Exception, match="frozen v8 predecessor"):
        v10.verify_v9_authority_for_succession(REPO)


def test_v10_transition_rejects_changed_source_binding(monkeypatch) -> None:
    expected = {"path": epochs.W10_V9_SOURCE_PATH, "source_commit": "a" * 40}
    monkeypatch.setattr(epochs, "_v10_transition", lambda _root: expected)
    epochs._require_w10_v10_transition(REPO, {"transition_from_v9": expected})
    with pytest.raises(epochs.SourceEpochHold, match="transition differs"):
        epochs._require_w10_v10_transition(REPO, {
            "transition_from_v9": {**expected, "source_commit": "b" * 40},
        })


def test_published_verifier_keeps_v8_and_v9_custody_distinct(tmp_path: Path, monkeypatch) -> None:
    authority = _authority()
    monkeypatch.setattr(v10, "validate_unit", lambda *_args: None)
    values = []
    manifest_rows = []
    streams = []
    v8_units, v9_units, v8_streams, v9_streams = [], [], [], []
    for index, unit in enumerate(work_units()):
        variants = []
        for variant_index, classifier in enumerate(unit_evidence_requirement(unit)["classifier_variants"]):
            relative = per_image_relative_path(unit, variant_index)
            variant = {"classifier_variant": classifier, "path": relative, "sha256": f"stream-{index}-{variant_index}", "n_correct": 0, "n_total": 1000}
            variants.append(variant)
            stream = {"ordinal": index, "classifier_variant": classifier, "per_image_path": relative, "per_image_sha256": variant["sha256"], "n_correct": 0, "n_total": 1000}
            streams.append(stream)
            frozen = {"ordinal": index, "path": relative, "canonical_sha256": variant["sha256"]}
            if index < 126:
                v8_streams.append(frozen)
            elif index < 231:
                v9_streams.append(frozen)
        value = {
            "unit_id": f"unit-{index}", "per_image_path": variants[0]["path"],
            "per_image_sha256": variants[0]["sha256"], "n_correct": 0, "n_total": 1000,
            "scorer_variants": variants,
            "binding": {} if index < 126 else {"execution_authority_id": V9["authority_id"] if index < 231 else authority["authority_id"]},
        }
        values.append(value)
        digest = v10.hashlib.sha256(canonical_bytes(value)).hexdigest()
        row = {
            "ordinal": index, "unit_id": value["unit_id"], "unit_path": unit_relative_path(unit, "json"),
            "unit_sha256": digest, "per_image_path": value["per_image_path"],
            "per_image_sha256": value["per_image_sha256"], "n_correct": 0, "n_total": 1000,
            "scorer_variants": variants,
        }
        manifest_rows.append(row)
        if index < 126:
            v8_units.append({"unit_id": value["unit_id"], "sha256": digest})
        elif index < 231:
            v9_units.append({"unit_id": value["unit_id"], "sha256": digest})
    units = {"schema_version": 1, "artifact_role": "W10_VALIDATION_REHEARSAL_UNIT_MANIFEST", "unit_count": 252, "ordered_unit_ids_digest": canonical_sha256({"unit_ids": [value["unit_id"] for value in values]}), "units": manifest_rows}
    units["unit_manifest_id"] = "w10unitmanifest-" + canonical_sha256(units)
    images = {"schema_version": 1, "artifact_role": "W10_VALIDATION_REHEARSAL_PER_IMAGE_MANIFEST", "stream_count": len(streams), "ordered_per_image_digest": canonical_sha256({"streams": streams}), "streams": streams}
    images["per_image_manifest_id"] = "w10imagemanifest-" + canonical_sha256(images)
    closeout = v10._closeout_from_manifests(values, authority, units, images)
    published = v10.published_v10_units(values, authority)
    root = tmp_path / "results/learned/w10"
    root.mkdir(parents=True)
    paths = {
        "units": root / "w10_continuation_units_v10.json",
        "unit_manifest": root / "w10_continuation_unit_manifest_v10.json",
        "images": root / "w10_continuation_per_image_manifest_v10.json",
        "closeout": root / "w10_continuation_closeout_v10.json",
    }
    for name, value in (("units", published), ("unit_manifest", units), ("images", images), ("closeout", closeout)):
        paths[name].write_bytes(canonical_bytes(value))
    monkeypatch.setattr(v10, "verify_v10_authority", lambda *_args: authority)
    monkeypatch.setattr(v10, "verify_v10_launch", lambda *_args: None)
    monkeypatch.setattr(v10, "v8_continuation_custody", lambda *_args: {"units": v8_units, "streams": v8_streams})
    monkeypatch.setattr(v10, "verify_v9_authority_for_succession", lambda *_args: ({}, {}))
    monkeypatch.setattr(v10, "load_v9_custody", lambda *_args, **_kwargs: {"units": v9_units, "streams": v9_streams})
    assert v10.verify_v10_published(tmp_path)["stream_count"] == len(streams)
    tampered = copy.deepcopy(published)
    tampered["units"][230]["unit_id"] = "relabelled"
    body = dict(tampered)
    body.pop("published_units_id")
    tampered["published_units_id"] = "w10continuationunitsv10-" + canonical_sha256(body)
    paths["units"].write_bytes(canonical_bytes(tampered))
    with pytest.raises(Exception, match="unit manifest row 230"):
        v10.verify_v10_published(tmp_path)
