"""Synthetic crash recovery for the frozen W10 v9 suffix, without evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import evaluation.w10_rehearsal as rehearsal
from evaluation.downstream_v4 import DownstreamHold
from evaluation.w10_continuation import verify_runtime_root_namespace, verify_suffix_evidence_set
from evaluation.w10_evidence import W10_PER_IMAGE_ROLE, per_image_relative_path, run_identity, sr18_row, unit_relative_path
from evaluation.w10_scope import W10_CHANNEL_SEED, W10_TRAIN_SEED, work_units
from training.deterministic_core import canonical_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CUSTODY = json.loads((ROOT / "results/learned/w10/w10_v8_failed_prefix_custody.json").read_bytes())
IDS = [f"synthetic-val-{index:04d}" for index in range(1000)]
AUTHORITY = {
    "authority_kind": "W10_VALIDATION_CONTINUATION_AUTHORITY_V9",
    "authority_id": "w10continuationauth-synthetic",
    "historical_completed_ordinals": list(range(126)),
    "new_authorized_ordinals": list(range(126, 252)),
    "cell": {"train_seed": W10_TRAIN_SEED, "channel_seed": W10_CHANNEL_SEED},
    "split": "val",
    "systems": list(rehearsal.W10_SYSTEMS),
    "validation_only": True,
    "test_authorized": False,
    "test_access": 0,
}


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    (runtime / "units").mkdir(parents=True)
    (runtime / "per_image").mkdir()
    (runtime / "j2k_cache").mkdir()
    (runtime / "plan.json").write_text("{}")
    (runtime / "continuation_plan_v9.json").write_text("{}")
    for row in CUSTODY["units"] + CUSTODY["streams"]:
        (runtime / row["path"]).write_bytes(b"historical-placeholder")
    return runtime


def _rows(unit: dict, variant: str) -> list[dict]:
    identity = run_identity(
        system=unit["system"], bw_ratio=unit["bw_ratio"], snr_db=unit["snr_db"],
        config_hash="a" * 64, checkpoint_id="b" * 64, classifier_variant=variant,
        ldpc_rate="1/2", modulation="qam16", quantiser_bits=None,
        transmit_dim=None, reconstruction_weight=None,
    )
    return [
        sr18_row(
            identity=identity, stable_sample_id=stable_id, true_label=0,
            pred_label=0, correct=True, outage=False, outage_reason=None, source_bytes=None,
        )
        for stable_id in IDS
    ]


def _backend(unit: dict) -> dict:
    variants = rehearsal.unit_evidence_requirement(unit)["classifier_variants"]
    return {
        "n_correct": 1000,
        "n_total": 1000,
        "per_image": _rows(unit, variants[0]),
        "primary_classifier_variant": variants[0],
        "secondary_streams": [
            {"classifier_variant": variant, "per_image": _rows(unit, variant)}
            for variant in variants[1:]
        ],
        "binding": {"selection_kind": "br16_fixed_mcs"},
        "papr_denominator": 1000,
    }


def _executor(monkeypatch: pytest.MonkeyPatch, units: tuple[dict, ...]) -> dict:
    monkeypatch.setattr(rehearsal, "work_units", lambda: units)
    monkeypatch.setattr(rehearsal, "unit_count", lambda: len(units))
    monkeypatch.setattr(rehearsal, "scope_sha256", lambda: "synthetic-scope")
    return {**AUTHORITY, "unit_count": len(units), "scope_sha256": "synthetic-scope"}


def _preflight(runtime: Path) -> None:
    verify_suffix_evidence_set(runtime, CUSTODY, AUTHORITY, expected_stable_ids=IDS)


def _historical_bytes(runtime: Path) -> dict[Path, bytes]:
    return {runtime / row["path"]: (runtime / row["path"]).read_bytes() for row in CUSTODY["units"] + CUSTODY["streams"]}


@pytest.mark.parametrize("crash_after", ["primary", "secondary"])
def test_exact_stream_replay_commits_pending_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_after: str) -> None:
    runtime = _runtime(tmp_path)
    historical = _historical_bytes(runtime)
    unit = work_units()[126]
    authority = _executor(monkeypatch, (unit,))
    writer = rehearsal._immutable_w10_write

    def crash(path: Path, value: dict) -> None:
        if (crash_after == "primary" and path.name.endswith(".1.json")) or (
            crash_after == "secondary" and path.parent.name == "units"
        ):
            raise RuntimeError("synthetic process stop")
        writer(path, value)

    monkeypatch.setattr(rehearsal, "_immutable_w10_write", crash)
    with pytest.raises(RuntimeError, match="synthetic process stop"):
        rehearsal.execute(runtime, authority=authority, backend=_backend, expected_stable_ids=IDS)
    pending_paths = [runtime / per_image_relative_path(unit, index) for index in range(2)]
    assert pending_paths[0].is_file()
    assert pending_paths[1].is_file() == (crash_after == "secondary")
    assert not (runtime / unit_relative_path(unit, "json")).exists()
    frozen = {path: path.read_bytes() for path in pending_paths if path.exists()}
    _preflight(runtime)

    monkeypatch.setattr(rehearsal, "_immutable_w10_write", writer)
    results = rehearsal.execute(runtime, authority=authority, backend=_backend, expected_stable_ids=IDS)
    assert len(results) == 1
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
    assert all(path.read_bytes() == raw for path, raw in historical.items())
    _preflight(runtime)

    def no_backend(_unit: dict) -> dict:
        raise AssertionError("a committed unit was re-evaluated")

    assert rehearsal.execute(runtime, authority=authority, backend=no_backend, expected_stable_ids=IDS) == results


def test_multiple_committed_suffix_units_resume_without_reexecution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path)
    historical = _historical_bytes(runtime)
    units = (work_units()[126], work_units()[127])
    authority = _executor(monkeypatch, units)
    results = rehearsal.execute(runtime, authority=authority, backend=_backend, expected_stable_ids=IDS)
    _preflight(runtime)
    frozen = {path: path.read_bytes() for unit in units for path in (
        runtime / unit_relative_path(unit, "json"),
        *(runtime / per_image_relative_path(unit, index) for index in range(2)),
    )}
    assert rehearsal.execute(runtime, authority=authority, backend=lambda _: (_ for _ in ()).throw(AssertionError("re-evaluation")), expected_stable_ids=IDS) == results
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
    assert all(path.read_bytes() == raw for path, raw in historical.items())


def test_later_or_out_of_order_orphan_stream_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first, later = work_units()[126:128]
    for unit, index in ((later, 0), (first, 1)):
        path = runtime / per_image_relative_path(unit, index)
        path.write_bytes(canonical_bytes(rehearsal.build_per_image_record(
            ordinal=unit["ordinal"], unit_key=str(path.relative_to(runtime)),
            rows=_rows(unit, rehearsal.unit_evidence_requirement(unit)["classifier_variants"][index]),
            classifier_variant=rehearsal.unit_evidence_requirement(unit)["classifier_variants"][index],
        )))
        with pytest.raises(DownstreamHold):
            _preflight(runtime)
        path.unlink()


@pytest.mark.parametrize("mutation", ["truncated", "identity", "variant", "row", "noncanonical"])
def test_bad_pending_stream_holds_without_replacement(tmp_path: Path, mutation: str) -> None:
    runtime = _runtime(tmp_path)
    unit = work_units()[126]
    path = runtime / per_image_relative_path(unit, 0)
    variant = rehearsal.unit_evidence_requirement(unit)["classifier_variants"][0]
    record = rehearsal.build_per_image_record(
        ordinal=126, unit_key=str(path.relative_to(runtime)), rows=_rows(unit, variant),
        classifier_variant=variant,
    )
    if mutation == "truncated":
        raw = b'{"artifact_role":'
    else:
        record = copy.deepcopy(record)
        if mutation == "identity":
            record["per_image_id"] = "wrong"
        elif mutation == "variant":
            record["classifier_variant"] = "unexpected"
        elif mutation == "row":
            record["rows"][0]["noise_id"] = "wrong"
        if mutation in {"variant", "row"}:
            body = dict(record)
            body.pop("per_image_id")
            record["per_image_id"] = W10_PER_IMAGE_ROLE.lower() + "-" + canonical_sha256(body)
        raw = canonical_bytes(record) if mutation != "noncanonical" else json.dumps(record).encode()
    path.write_bytes(raw)
    with pytest.raises((DownstreamHold, ValueError)):
        _preflight(runtime)
    assert path.read_bytes() == raw
    if mutation == "truncated":
        with pytest.raises(DownstreamHold, match="immutable artifact differs"):
            rehearsal._immutable_w10_write(path, record)
        assert path.read_bytes() == raw


def test_runtime_root_namespace_is_phase_aware_and_exact(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _preflight(runtime)
    for name, directory in (("unexpected.json", False), ("unexpected_dir", True)):
        path = runtime / name
        path.mkdir() if directory else path.write_text("{}")
        with pytest.raises(DownstreamHold, match="unexpected W10 runtime-root"):
            _preflight(runtime)
        path.rmdir() if directory else path.unlink()
    link = runtime / "unsafe"
    link.symlink_to(runtime / "plan.json")
    with pytest.raises(DownstreamHold):
        _preflight(runtime)
    link.unlink()
    for name in ("continuation_unit_manifest_v9.json", "continuation_per_image_manifest_v9.json", "continuation_closeout_v9.json"):
        (runtime / name).write_text("{}")
    verify_runtime_root_namespace(runtime, phase="closeout")
    with pytest.raises(DownstreamHold):
        verify_runtime_root_namespace(runtime, phase="execute")


def test_expected_root_symlink_and_missing_plan_are_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    plan = runtime / "continuation_plan_v9.json"
    plan.unlink()
    verify_runtime_root_namespace(runtime, phase="plan")
    with pytest.raises(DownstreamHold, match="continuation plan is missing"):
        verify_runtime_root_namespace(runtime, phase="execute")
    plan.symlink_to(runtime / "plan.json")
    with pytest.raises(DownstreamHold, match="unsafe"):
        verify_runtime_root_namespace(runtime, phase="plan")


def test_pending_symlink_and_extra_stream_are_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    unit = work_units()[126]
    primary = runtime / per_image_relative_path(unit, 0)
    primary.symlink_to(runtime / "plan.json")
    with pytest.raises(DownstreamHold, match="missing or unsafe"):
        _preflight(runtime)
    primary.unlink()
    extra = runtime / per_image_relative_path(unit, 2)
    extra.write_text("{}")
    with pytest.raises(DownstreamHold, match="unexpected W10 scorer evidence"):
        _preflight(runtime)


def test_valid_orphan_that_disagrees_with_regenerated_output_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path)
    unit = work_units()[126]
    authority = _executor(monkeypatch, (unit,))
    path = runtime / per_image_relative_path(unit, 0)
    variant = rehearsal.unit_evidence_requirement(unit)["classifier_variants"][0]
    rows = _rows(unit, variant)
    rows[0]["pred_label"] = 1
    rows[0]["correct"] = False
    record = rehearsal.build_per_image_record(
        ordinal=126, unit_key=str(path.relative_to(runtime)), rows=rows,
        classifier_variant=variant,
    )
    path.write_bytes(canonical_bytes(record))
    _preflight(runtime)  # valid evidence, but it is still an uncommitted attempt
    original = path.read_bytes()
    with pytest.raises(DownstreamHold, match="immutable artifact differs"):
        rehearsal.execute(runtime, authority=authority, backend=_backend, expected_stable_ids=IDS)
    assert path.read_bytes() == original
    assert not (runtime / unit_relative_path(unit, "json")).exists()


def test_malformed_secondary_orphan_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    unit = work_units()[126]
    for index, variant in enumerate(rehearsal.unit_evidence_requirement(unit)["classifier_variants"]):
        path = runtime / per_image_relative_path(unit, index)
        record = rehearsal.build_per_image_record(
            ordinal=126, unit_key=str(path.relative_to(runtime)), rows=_rows(unit, variant),
            classifier_variant=variant,
        )
        if index == 1:
            record["rows"][0]["pair_id"] = "wrong"
            body = dict(record)
            body.pop("per_image_id")
            record["per_image_id"] = W10_PER_IMAGE_ROLE.lower() + "-" + canonical_sha256(body)
        path.write_bytes(canonical_bytes(record))
    with pytest.raises(DownstreamHold, match="pair identity differs"):
        _preflight(runtime)


def test_execute_tool_checks_launch_before_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.run_w10_continuation as runner

    runtime = _runtime(tmp_path)
    authority = {**AUTHORITY, "runtime_root": "runtime"}
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "verify_continuation_authority", lambda _root: authority)
    monkeypatch.setattr(runner, "ValidationView", lambda: type("View", (), {"stable_ids": IDS})())
    monkeypatch.setattr(runner, "verify_worker_prefix", lambda *_args, **_kwargs: CUSTODY)
    monkeypatch.setattr(runner, "verify_suffix_evidence_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "verify_continuation_plan", lambda *_args: {"plan_id": "synthetic"})
    monkeypatch.setattr(runner, "verify_launch_authorization", lambda *_args: (_ for _ in ()).throw(DownstreamHold("launch missing")))
    monkeypatch.setattr(runner, "dispatch", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("backend started")))
    with pytest.raises(DownstreamHold, match="launch missing"):
        runner.main(["execute"])
    assert not any((runtime / "units").glob("126-*.json"))


def test_failed_atomic_publication_never_exposes_partial_final_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = work_units()[126]
    path = tmp_path / per_image_relative_path(unit, 0)
    real_link = rehearsal.os.link
    monkeypatch.setattr(rehearsal.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("synthetic stop")))
    with pytest.raises(OSError, match="synthetic stop"):
        rehearsal._immutable_w10_write(path, {"sample": 1})
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    monkeypatch.setattr(rehearsal.os, "link", real_link)
