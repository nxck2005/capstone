"""Cheap pre-freeze checks of the v9 bridge and BR-16 resolution."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import evaluation.w10_classical as classical
import evaluation.am98_spec_compatibility as am98
import evaluation.am99_spec_compatibility as am99
import runtime.source_epochs as epochs
from evaluation.w10_continuation import (
    CONTINUATION_AUTHORITY_PATH,
    CONTINUATION_START,
    build_continuation_authority,
    build_continuation_plan,
    verify_suffix_evidence_set,
)
from evaluation.w10_evidence import per_image_relative_path, run_identity, scheduled_noise_id, sr18_row, validate_per_image
from evaluation.w10_rehearsal import unit_body, unit_evidence_requirement
from evaluation.w10_scope import SCOPE, work_units
from training.deterministic_core import canonical_bytes, canonical_sha256

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((REPO / "results/learned/w10/w10_rehearsal_authorization.json").read_bytes())
CUSTODY = json.loads((REPO / epochs.W10_V8_CUSTODY_PATH).read_bytes())


def _binding(role: str) -> dict:
    return next(item for item in ORIGINAL["bindings"] if item["scope"]["role"] == role)


def test_historical_custody_is_self_consistent_and_failed_partial() -> None:
    assert epochs.v8_continuation_custody(REPO) == CUSTODY
    assert CUSTODY["status"] == "FAILED_PARTIAL_NOT_CLOSEOUT"
    assert CUSTODY["completed_ordinals"] == [0, 125]
    assert CUSTODY["failed_ordinal"] == 126
    assert [row["ordinal"] for row in CUSTODY["units"]] == list(range(126))
    assert len(CUSTODY["streams"]) == 168


def test_completed_learned_science_and_common_evidence_code_are_byte_unchanged() -> None:
    for relative in (
        "src/evaluation/w10_dispatch.py",
        "src/evaluation/w10_backends.py",
        "src/evaluation/w10_evidence.py",
        "src/evaluation/w10_scope.py",
        "src/evaluation/w10_bindings.py",
        "src/evaluation/w10_selections.py",
        "src/data/djscc_validation.py",
    ):
        original = subprocess.run(
            ["git", "show", f"{epochs.W10_V8_EXECUTION_COMMIT}:{relative}"],
            cwd=REPO, check=True, capture_output=True,
        ).stdout
        assert (REPO / relative).read_bytes() == original, relative


def test_adaptive_candidate_resolution_matches_original_for_both_completed_arms() -> None:
    candidates = classical._candidate_table(REPO)
    for role in ("er1_headline_classical", "er11_efficiency_classical"):
        binding = _binding(role)["selection"]
        for point in classical._selection_map(binding).values():
            original = candidates[str(point["authority_candidate_id"])]
            expected = (
                str(original["modulation"]),
                str(original["ldpc_rate"]),
                int(original["encode_axis_px"]),
                canonical_sha256(original),
            )
            assert classical.resolve_classical_configuration(
                binding=binding, point=point, candidates=candidates, codec_kind="jpeg2000", quality=None
            ) == expected


def test_br16_original_keyerror_is_repaired_without_candidate_identity() -> None:
    binding = _binding("br16_fixed_mcs")["selection"]
    assert len(classical._selection_map(binding)) == 21
    for point in classical._selection_map(binding).values():
        with pytest.raises(KeyError, match="authority_candidate_id"):
            point["authority_candidate_id"]  # the exact v8 line-177 failure
        resolved = classical.resolve_classical_configuration(
            binding=binding, point=point, candidates={}, codec_kind="jpeg2000", quality=None
        )
        assert resolved == ("qam16", "1/2", 160, canonical_sha256(binding["fixed_configuration"]))
    malformed = copy.deepcopy(binding)
    del malformed["fixed_configuration"]["ldpc_rate"]
    with pytest.raises(KeyError, match="ldpc_rate"):
        classical._selection_map(malformed)


def test_jpeg_resolution_is_byte_identical_to_previous_behavior() -> None:
    binding = _binding("dec9_jpeg_secondary")["selection"]
    for point in classical._selection_map(binding).values():
        expected = (
            str(point["modulation"]),
            str(point["ldpc_rate"]),
            int(point["encode_axis_px"]),
            canonical_sha256(point),
        )
        assert classical.resolve_classical_configuration(
            binding=binding, point=point, candidates={}, codec_kind="jpeg", quality=int(point["quality"])
        ) == expected
        with pytest.raises(RuntimeError, match="quality"):
            classical.resolve_classical_configuration(
                binding=binding, point=point, candidates={}, codec_kind="jpeg", quality=int(point["quality"]) + 1
            )


def test_v9_source_transition_records_executed_v8_without_freezing(monkeypatch, tmp_path: Path) -> None:
    predecessor = epochs.load_w10_manifest(REPO, live=False, epoch="v8")
    frozen_v9_path = REPO / epochs.W10_V9_SOURCE_PATH
    frozen_v9_before = frozen_v9_path.read_bytes() if frozen_v9_path.is_file() else None
    real_successor_path = epochs.successor_path

    def fake_successor_path(root, *, epoch="v2"):
        if epoch == "v9":
            return tmp_path / epochs.W10_V9_SOURCE_PATH.rsplit("/", 1)[-1]
        return real_successor_path(root, epoch=epoch)

    def fake_build(_root, *, source_commit, relevant_config_paths):
        assert tuple(relevant_config_paths) == epochs.W10_RELEVANT_CONFIG_PATHS
        base = dict(predecessor)
        base.pop("transition_from_v7")
        base.pop("pre_future_work_state")
        base["source_commit"] = source_commit
        return base

    monkeypatch.setattr(epochs, "build_manifest", fake_build)
    monkeypatch.setattr(epochs, "successor_path", fake_successor_path)
    value = epochs.build_w10_manifest_v9(REPO, source_commit="a" * 40)
    epochs.assert_w10_manifest_contract(value)
    assert value["manifest_kind"] == epochs.W10_V9_MANIFEST_KIND
    assert value["transition_from_v8"]["v8_scientific_work_executed"] is True
    assert value["transition_from_v8"]["historical_custody_id"] == CUSTODY["custody_id"]
    assert "superseded_successor" not in value
    assert epochs.predecessor_binding(value) == value["transition_from_v8"]
    assert (frozen_v9_path.read_bytes() if frozen_v9_path.is_file() else None) == frozen_v9_before


def test_am99_accepts_v9_only_with_authenticated_v8_lineage(monkeypatch) -> None:
    historical = epochs.load_w10_manifest(REPO, live=False, epoch="v8")
    historical_path = REPO / epochs.W10_V8_SOURCE_PATH
    manifest_path = epochs.successor_path(REPO, epoch="v9")
    if not manifest_path.is_file():
        manifest_path = historical_path
    source = {
        "manifest_kind": epochs.W10_V9_MANIFEST_KIND,
        "manifest_id": epochs.W10_V9_MANIFEST_PREFIX + "synthetic",
        "source_commit": "a" * 40,
        "governs": ["papr_constrained_training", "w10_validation_rehearsal"],
        "transition_from_v8": {
            "path": epochs.W10_V8_SOURCE_PATH,
            "manifest_kind": historical["manifest_kind"],
            "manifest_id": historical["manifest_id"],
            "sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
            "source_commit": historical["source_commit"],
        },
    }
    current = [source]

    def fake_load(_root, *, live=True, epoch=None):
        del live
        return historical if epoch == "v8" else current[0]

    monkeypatch.setattr(am99, "load_w10_manifest", fake_load)
    monkeypatch.setattr(am98, "load_w10_manifest", fake_load)
    monkeypatch.setattr(am99, "active_manifest_path", lambda _root: manifest_path)
    monkeypatch.setattr(am98, "active_manifest_path", lambda _root: manifest_path)
    monkeypatch.setattr(am98.am97, "load", lambda *_args, **_kwargs: {})

    result = am99.load(REPO)
    assert result["successor_manifest"]["kind"] == epochs.W10_V9_MANIFEST_KIND
    assert result["successor_manifest"]["manifest_id"] == source["manifest_id"]

    current[0] = {
        **source,
        "transition_from_v8": {**source["transition_from_v8"], "sha256": "0" * 64},
    }
    with pytest.raises(epochs.SourceEpochHold, match="historical successor manifest bytes differ"):
        am99.load(REPO)


def test_continuation_authority_projects_original_science_without_freeze(monkeypatch) -> None:
    import evaluation.w10_continuation as continuation

    source = copy.deepcopy(epochs.load_w10_manifest(REPO, live=False, epoch="v8"))
    source["manifest_kind"] = epochs.W10_V9_MANIFEST_KIND
    source["manifest_id"] = "w10downstreamsourcev9-synthetic"
    monkeypatch.setattr(continuation, "verify_historical_authority", lambda _root, *, live: ORIGINAL)
    monkeypatch.setattr(continuation, "source_record", lambda _root, _source: {"path": epochs.W10_V9_SOURCE_PATH, "manifest_id": _source["manifest_id"], "sha256": "s" * 64})
    body = build_continuation_authority(REPO, source)
    assert body["bindings"] == ORIGINAL["bindings"]
    assert body["scope_sha256"] == ORIGINAL["scope_sha256"]
    assert body["historical_completed_ordinals"] == list(range(126))
    assert body["new_authorized_ordinals"] == list(range(126, 252))
    assert body["gpu_uuid"] == ORIGINAL["gpu_uuid"]
    authority = {**body, "authority_id": "w10continuationauth-synthetic"}
    plan = build_continuation_plan(authority)
    assert plan["work_units"] == list(work_units())
    assert plan["historical_ordinals"] == [0, 125] and plan["new_ordinals"] == [126, 251]
    assert CONTINUATION_START == 126


def test_suffix_file_set_rejects_missing_prefix_and_unexpected_files(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "units").mkdir(parents=True)
    (runtime / "per_image").mkdir()
    authority = {"authority_id": "w10continuationauth-synthetic"}
    with pytest.raises(Exception, match="historical evidence disappeared"):
        verify_suffix_evidence_set(runtime, CUSTODY, authority)


def test_all_12_arms_and_21_snr_points_have_frozen_cheap_preflight() -> None:
    units = work_units()
    assert len(SCOPE) == 12 and len(units) == 252
    assert [unit["ordinal"] for unit in units] == list(range(252))
    by_role = {_binding(entry.role)["scope"]["role"]: _binding(entry.role) for entry in SCOPE}
    assert len(by_role) == 12
    for entry in SCOPE:
        subset = [unit for unit in units if unit["role"] == entry.role]
        assert len(subset) == 21
        binding = by_role[entry.role]
        assert binding["scope"] == entry.identity()
        assert binding["checkpoint"]["state"] == "FROZEN"
        assert binding["selection"]["state"] == "FROZEN"
        for unit in subset:
            assert unit_evidence_requirement(unit)["classifier_variants"] == list(entry.classifier_variants)
            assert unit["split"] == "val" and unit["channel_seed"] == 0
            if entry.backend in ("classical", "jpeg_secondary"):
                selection = classical._selection_map(binding["selection"])
                assert len(selection) == 21 and int(unit["snr_db"]) in selection
                point = selection[int(unit["snr_db"])]
                classical.resolve_classical_configuration(
                    binding=binding["selection"],
                    point=point,
                    candidates=classical._candidate_table(REPO) if entry.backend == "classical" else {},
                    codec_kind="jpeg" if entry.backend == "jpeg_secondary" else "jpeg2000",
                    quality=int(point["quality"]) if entry.backend == "jpeg_secondary" else None,
                )
            elif entry.backend == "er9_digital":
                assert any(float(point["snr_db"]) == float(unit["snr_db"]) for point in binding["checkpoint"]["per_snr_phy"])
            elif entry.backend == "label_bound":
                assert any(float(point["snr_db"]) == float(unit["snr_db"]) for point in binding["selection"]["selections"])
            else:
                assert binding["checkpoint"].get("checkpoint_id") or binding["checkpoint"].get("checkpoint_sha256")
            # The one scheduled identity is independent of system at the same ratio/SNR.
            noise = scheduled_noise_id(stable_sample_id="synthetic-stable-id", bw_ratio=unit["bw_ratio"], test_snr_db=unit["snr_db"], k=12800 if unit["bw_ratio"] == "r_1_6" else 3200)
            assert len(noise) == 64


def test_sr18_synthetic_schema_rejects_missing_and_unexpected_fields() -> None:
    unit = work_units()[0]
    identity = run_identity(
        system=unit["system"], bw_ratio=unit["bw_ratio"], snr_db=unit["snr_db"],
        config_hash="a" * 64, checkpoint_id="b" * 64, classifier_variant="own_task_head",
        ldpc_rate="not_applicable", modulation="not_applicable", quantiser_bits=None,
        transmit_dim=None, reconstruction_weight=None,
    )
    ids = [f"synthetic-{index:04d}" for index in range(1000)]
    rows = [sr18_row(identity=identity, stable_sample_id=sid, true_label=0, pred_label=0, correct=True, outage=False, outage_reason=None, source_bytes=None) for sid in ids]
    validate_per_image(rows, expected_stable_ids=ids, system=unit["system"], bw_ratio=unit["bw_ratio"], snr_db=unit["snr_db"])
    bad = copy.deepcopy(rows)
    bad[0]["unexpected"] = 1
    with pytest.raises(ValueError, match="schema-exact"):
        validate_per_image(bad, expected_stable_ids=ids, system=unit["system"], bw_ratio=unit["bw_ratio"], snr_db=unit["snr_db"])
    bad = copy.deepcopy(rows)
    del bad[0]["noise_id"]
    with pytest.raises(ValueError, match="schema-exact"):
        validate_per_image(bad, expected_stable_ids=ids, system=unit["system"], bw_ratio=unit["bw_ratio"], snr_db=unit["snr_db"])


def test_cross_authority_suffix_substitution_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "units").mkdir(parents=True)
    (runtime / "per_image").mkdir()
    for row in CUSTODY["units"]:
        (runtime / row["path"]).write_text("{}")
    for row in CUSTODY["streams"]:
        (runtime / row["path"]).write_text("{}")
    unit = work_units()[126]
    requirement = unit_evidence_requirement(unit)
    streams = [
        {"classifier_variant": variant, "path": per_image_relative_path(unit, index), "sha256": "a" * 64, "n_correct": 0, "n_total": 1000}
        for index, variant in enumerate(requirement["classifier_variants"])
    ]
    value = unit_body(
        expected=unit,
        evidence={"n_correct": 0, "n_total": 1000, "per_image": [], "binding": {"execution_authority_id": "other-authority"}, "papr_denominator": 1000},
        per_image={"_sha256": "a" * 64}, per_image_path=streams[0]["path"], scorer_variants=streams,
    )
    (runtime / f"units/126-{unit['system']}-{unit['bw_ratio']}-snr{int(unit['snr_db']):+03d}.json").write_text(json.dumps(value))
    with pytest.raises(Exception, match="cross-authority"):
        verify_suffix_evidence_set(runtime, CUSTODY, {"authority_id": "w10continuationauth-synthetic"})


def test_historical_custody_and_authority_byte_tamper_are_rejected(tmp_path: Path, monkeypatch) -> None:
    original_source = epochs.load_w10_manifest(REPO, live=False, epoch="v8")
    monkeypatch.setattr(epochs, "load_w10_manifest", lambda _root, *, live, epoch: original_source)
    for relative in (epochs.W10_V8_SOURCE_PATH, epochs.W10_AUTHORITY_SOURCE_PATH, epochs.W10_V8_CUSTODY_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    assert epochs.v8_continuation_custody(tmp_path)["custody_id"] == CUSTODY["custody_id"]
    custody_path = tmp_path / epochs.W10_V8_CUSTODY_PATH
    custody_path.write_bytes(custody_path.read_bytes() + b" ")
    with pytest.raises(Exception, match="custody bytes differ"):
        epochs.v8_continuation_custody(tmp_path)
    custody_path.write_bytes((REPO / epochs.W10_V8_CUSTODY_PATH).read_bytes())
    authority_path = tmp_path / epochs.W10_AUTHORITY_SOURCE_PATH
    authority_path.write_bytes(authority_path.read_bytes() + b" ")
    with pytest.raises(Exception, match="authority bytes differ"):
        epochs.v8_continuation_custody(tmp_path)


def test_new_authority_tamper_is_rejected_before_execution(tmp_path: Path, monkeypatch) -> None:
    import evaluation.w10_continuation as continuation

    body = {"authority_kind": continuation.CONTINUATION_AUTHORITY_KIND, "new_authorized_ordinals": list(range(126, 252))}
    identifier = "w10continuationauth-" + canonical_sha256(body)
    path = tmp_path / CONTINUATION_AUTHORITY_PATH
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(continuation, "load_w10_manifest", lambda _root, *, live, epoch: {"manifest_kind": epochs.W10_V9_MANIFEST_KIND})
    monkeypatch.setattr(continuation, "build_continuation_authority", lambda _root, _source: body)
    path.write_text(json.dumps({**body, "authority_id": identifier, "test_access": 1}))
    with pytest.raises(Exception, match="authority ID differs"):
        continuation.verify_continuation_authority(tmp_path, live=False)
    mutated = {**body, "test_access": 1}
    path.write_text(json.dumps({**mutated, "authority_id": "w10continuationauth-" + canonical_sha256(mutated)}))
    with pytest.raises(Exception, match="authority fields differ"):
        continuation.verify_continuation_authority(tmp_path, live=False)


def test_published_contract_authenticates_mixed_provenance_without_worker_bytes(tmp_path: Path, monkeypatch) -> None:
    import tools.verify_w10_continuation as verifier
    from evaluation.w10_continuation import published_continuation_units

    authority = {"authority_id": "w10continuationauth-synthetic", "source_manifest": {"manifest_id": "w10downstreamsourcev9-synthetic"}, "scope_sha256": ORIGINAL["scope_sha256"]}
    bodies = []
    unit_rows = []
    streams = []
    frozen_prefix_units = []
    frozen_prefix_streams = []
    for unit in work_units():
        ordinal = unit["ordinal"]
        variants = [
            {"classifier_variant": name, "path": per_image_relative_path(unit, index), "sha256": "a" * 64, "n_correct": 0, "n_total": 1000}
            for index, name in enumerate(unit_evidence_requirement(unit)["classifier_variants"])
        ]
        binding = {"kind": "synthetic"}
        if ordinal >= 126:
            binding["execution_authority_id"] = authority["authority_id"]
        value = unit_body(
            expected=unit,
            evidence={"n_correct": 0, "n_total": 1000, "per_image": [], "binding": binding, "papr_denominator": 1000},
            per_image={"_sha256": "a" * 64}, per_image_path=variants[0]["path"], scorer_variants=variants,
        )
        bodies.append(value)
        row = {
            "ordinal": ordinal, "unit_id": value["unit_id"],
            "unit_path": f"units/{ordinal:03d}-{unit['system']}-{unit['bw_ratio']}-snr{int(unit['snr_db']):+03d}.json",
            "unit_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
            "per_image_path": value["per_image_path"], "per_image_sha256": value["per_image_sha256"],
            "n_correct": 0, "n_total": 1000, "scorer_variants": variants,
        }
        unit_rows.append(row)
        if ordinal < 126:
            frozen_prefix_units.append({"ordinal": ordinal, "path": row["unit_path"], "unit_id": value["unit_id"], "sha256": row["unit_sha256"]})
        for variant in variants:
            streams.append({"ordinal": ordinal, "classifier_variant": variant["classifier_variant"], "per_image_path": variant["path"], "per_image_sha256": variant["sha256"], "n_correct": 0, "n_total": 1000})
            if ordinal < 126:
                frozen_prefix_streams.append({"ordinal": ordinal, "path": variant["path"], "canonical_sha256": variant["sha256"]})
    custody = {"custody_id": "w10v8prefix-synthetic", "units": frozen_prefix_units, "streams": frozen_prefix_streams, "stream_count": len(frozen_prefix_streams)}
    published = published_continuation_units(bodies, {**authority, "authority_id": authority["authority_id"]})
    published["historical_custody_id"] = custody["custody_id"]
    published["published_units_id"] = "w10continuationunits-" + canonical_sha256({k: v for k, v in published.items() if k != "published_units_id"})
    unit_manifest = {"schema_version": 1, "artifact_role": "W10_VALIDATION_REHEARSAL_UNIT_MANIFEST", "unit_count": 252, "ordered_unit_ids_digest": canonical_sha256({"unit_ids": [x["unit_id"] for x in bodies]}), "units": unit_rows}
    unit_manifest["unit_manifest_id"] = "w10unitmanifest-" + canonical_sha256(unit_manifest)
    images = {"schema_version": 1, "artifact_role": "W10_VALIDATION_REHEARSAL_PER_IMAGE_MANIFEST", "stream_count": len(streams), "ordered_per_image_digest": canonical_sha256({"streams": streams}), "streams": streams}
    images["per_image_manifest_id"] = "w10imagemanifest-" + canonical_sha256(images)
    provenance = [
        {"ordinal": index, "source_manifest_id": epochs.W10_V8_MANIFEST_ID if index < 126 else authority["source_manifest"]["manifest_id"], "authority_id": epochs.W10_V8_AUTHORITY_ID if index < 126 else authority["authority_id"], "unit_id": value["unit_id"]}
        for index, value in enumerate(bodies)
    ]
    closeout = {
        "schema_version": 1, "artifact_role": "W10_VALIDATION_CONTINUATION_CLOSEOUT_V9", "status": "COMPLETE",
        "original_authority_id": epochs.W10_V8_AUTHORITY_ID, "continuation_authority_id": authority["authority_id"],
        "historical_custody_id": custody["custody_id"], "historical_custody_sha256": "c" * 64,
        "scope_sha256": authority["scope_sha256"], "unit_count": 252, "stream_count": len(streams),
        "ordered_unit_ids_digest": unit_manifest["ordered_unit_ids_digest"], "ordered_per_image_digest": images["ordered_per_image_digest"],
        "unit_manifest_sha256": canonical_sha256(unit_manifest), "per_image_manifest_sha256": canonical_sha256(images),
        "ordinal_provenance": provenance, "validation_only": True, "test": "SEALED", "test_access": 0,
    }
    closeout["closeout_id"] = "w10continuationcloseout-" + canonical_sha256(closeout)
    paths = {key: tmp_path / f"{key}.json" for key in ("units", "unit_manifest", "images", "closeout")}
    for key, value in (("units", published), ("unit_manifest", unit_manifest), ("images", images), ("closeout", closeout)):
        paths[key].write_bytes(canonical_bytes(value))
    monkeypatch.setattr(verifier, "PUBLISHED_PATHS", paths)
    monkeypatch.setattr(verifier, "verify_continuation_authority", lambda _root: {**authority, "historical_custody_sha256": "c" * 64})
    monkeypatch.setattr(verifier, "verify_launch_authorization", lambda _root, _authority, _plan: {"status": "OWNER_AUTHORIZED_SUFFIX_ONLY"})
    monkeypatch.setattr(verifier, "v8_continuation_custody", lambda _root: custody)
    assert verifier.verify_published()["unit_count"] == 252
    closeout["ordinal_provenance"][0]["authority_id"] = authority["authority_id"]
    closeout["closeout_id"] = "w10continuationcloseout-" + canonical_sha256({k: v for k, v in closeout.items() if k != "closeout_id"})
    paths["closeout"].write_bytes(canonical_bytes(closeout))
    with pytest.raises(Exception, match="closeout provenance differs"):
        verifier.verify_published()


def test_suffix_requires_a_separate_content_bound_launch_grant(tmp_path: Path) -> None:
    import evaluation.w10_continuation as continuation

    authority = {"authority_id": "w10continuationauth-synthetic", "source_manifest": {"manifest_id": "w10downstreamsourcev9-synthetic"}, "scope_sha256": ORIGINAL["scope_sha256"]}
    plan = build_continuation_plan(authority)
    with pytest.raises(Exception, match="lacks separate owner authorization"):
        continuation.verify_launch_authorization(tmp_path, authority, plan)
    launch = continuation.build_launch_authorization(authority, plan)
    path = tmp_path / continuation.CONTINUATION_LAUNCH_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_bytes(launch))
    assert continuation.verify_launch_authorization(tmp_path, authority, plan) == launch
    launch["authorized_ordinals"][0] = 125
    path.write_bytes(canonical_bytes(launch))
    with pytest.raises(Exception, match="launch authorization differs"):
        continuation.verify_launch_authorization(tmp_path, authority, plan)
