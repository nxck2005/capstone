"""Adversarial, bounded proof for the corrected pre-data G8_E epoch.

Every fixture in this file is explicitly non-scientific and non-selection. It
never opens a project dataset payload, loads the G-1 checkpoint, constructs an
owner authorization, or writes the corrected production runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline import g8_e_corrected as corrected
from baseline.classical.composition import G8Authorization, SweepBudgetError
from data.identity import stable_sample_id


def _codestream() -> bytes:
    return (
        b"\xff\x4f"
        + b"\xff\x90\x00\x0a\x00\x00"
        + (17).to_bytes(4, "big")
        + b"\x00\x00"
        + b"\xff\x93abc\xff\xd9"
    )


def _fixture_authority(bundle: dict[str, dict]) -> dict[str, object]:
    real = bundle["authority"]
    groups = corrected._physical_equivalence_plan(real)["groups"]
    equal_group = next(group for group in groups if len(group["structural_identity_ids"]) >= 2)
    selected_ids = list(equal_group["structural_identity_ids"][:2])
    selected_ids.append(next(
        group["structural_identity_ids"][0]
        for group in groups
        if group["payload_budget_bytes"] != equal_group["payload_budget_bytes"]
    ))
    selected_ids.append(next(
        group["structural_identity_ids"][0]
        for group in groups
        if group["payload_budget_bytes"] not in {equal_group["payload_budget_bytes"], selected_ids[-1]}
    ))
    selected = [row for row in real["structural_identities"] if row["structural_identity_id"] in selected_ids]
    selected.sort(key=lambda row: row["structural_identity_id"])
    logical = {}
    for structural in selected:
        candidate_id = next(
            candidate_id
            for candidate_id, structural_id in real["logical_candidate_to_structural_id"].items()
            if structural_id == structural["structural_identity_id"]
        )
        logical[candidate_id] = structural["structural_identity_id"]
    return {
        **real,
        "authority_id": "g8e-test-authority",
        "structural_identities": selected,
        "logical_candidate_to_structural_id": logical,
        "counts": {**real["counts"], "structural_initial": len(selected), "logical_initial_snr_cells": len(selected)},
    }


class _Backend:
    def __init__(self, contract: dict[str, object], failure_budget: int) -> None:
        self.snapshot = contract["codec"]["snapshot"]
        self.configuration_hash = contract["codec"]["configuration_hash"]
        self.failure_budget = failure_budget
        self.calls = 0

    def encode_to_budget(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
        del image
        self.calls += 1
        budget = int(kwargs["budget_bytes"])
        if budget == self.failure_budget:
            return SimpleNamespace(feasible=False, codestream=None)
        return SimpleNamespace(feasible=True, codestream=_codestream())


class _Decoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, codestream: bytes) -> np.ndarray:
        del codestream
        self.calls += 1
        if self.calls == 2:
            raise ValueError("synthetic reconstruction failure")
        return np.full((160, 160, 3), 7, dtype=np.uint8)


class _Classifier:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, pixels: np.ndarray) -> int:
        assert pixels.shape == (160, 160, 3)
        self.calls += 1
        return 0


@pytest.fixture()
def synthetic_stack(tmp_path: Path) -> dict[str, object]:
    bundle = corrected.verify_corrected_bundle()
    authority = _fixture_authority(bundle)
    sample_bytes = b"NON-SCIENTIFIC-SYNTHETIC-SOURCE"
    sample_id = stable_sample_id(sample_bytes)
    sample = corrected.SyntheticSample(
        stable_sample_id=sample_id,
        label=0,
        source_bytes=sample_bytes,
        canonical_pixels=np.full((160, 160, 3), 9, dtype=np.uint8),
    )
    work_units = corrected.expected_work_units(authority, (sample_id,))
    failure_budget = next(
        int(row["payload_budget_bytes"])
        for row in authority["structural_identities"]
        if int(row["payload_budget_bytes"]) not in {
            int(authority["structural_identities"][0]["payload_budget_bytes"]),
            int(authority["structural_identities"][1]["payload_budget_bytes"]),
        }
    )
    backend = _Backend(bundle["contract"], failure_budget)
    decoder = _Decoder()
    classifier = _Classifier()
    executor = corrected.MeasurementExecutor(
        bundle={"authority": authority, "contract": bundle["contract"]},
        runtime_root=tmp_path / "runtime",
        backend=backend,
        decoder=decoder,
        classifier=classifier,
        non_scientific_fixture=True,
    )
    samples = {sample_id: sample}
    campaign = corrected.AtomicE2Campaign(
        runtime_root=tmp_path / "runtime",
        contract=bundle["contract"],
        authority=authority,
        work_units=work_units,
        executor=executor,
        sample_provider=samples.__getitem__,
    )
    return {
        "bundle": bundle,
        "authority": authority,
        "sample": sample,
        "sample_ids": (sample_id,),
        "work_units": work_units,
        "backend": backend,
        "decoder": decoder,
        "classifier": classifier,
        "executor": executor,
        "campaign": campaign,
        "runtime": tmp_path / "runtime",
    }


def test_old_superseded_campaign_cannot_execute() -> None:
    with pytest.raises(corrected.CorrectedG8EError, match="superseded-before-data"):
        corrected.reject_old_campaign(corrected.OLD_CAMPAIGN_ID)


def test_old_refusal_stub_is_historical_not_current() -> None:
    old = (corrected.REPO_ROOT / "tools/run_g8_e.py").read_text()
    assert "refuse_e2_execution" in old
    manifest = json.loads(corrected.CORRECTED_SOURCE_MANIFEST_PATH.read_text())
    assert all(entry["path"] != "tools/run_g8_e.py" for entry in manifest["source_entries"])
    assert any(entry["path"] == "tools/run_g8_e_corrected.py" for entry in manifest["source_entries"])


def test_current_runner_refuses_before_validation_decode(tmp_path: Path) -> None:
    contract = json.loads(corrected.CORRECTED_CONTRACT_PATH.read_text())
    result = subprocess.run(
        [
            sys.executable,
            str(corrected.REPO_ROOT / "tools/run_g8_e_corrected.py"),
            "--start",
            "--campaign-id",
            contract["campaign_id"],
            "--runtime-root",
            str(tmp_path / "runtime"),
        ],
        cwd=corrected.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "before validation payload decode" in result.stderr
    assert not (tmp_path / "runtime").exists()


def test_source_change_after_corrected_freeze_rejects() -> None:
    source = json.loads(corrected.CORRECTED_SOURCE_MANIFEST_PATH.read_text())
    source["source_entries"][0]["sha256"] = "0" * 64
    with pytest.raises(corrected.CorrectedG8EError, match="source manifest ID|source drift"):
        corrected.validate_source_manifest(source)


def test_logical_authority_retains_every_snr_cell_and_mapping_is_total() -> None:
    result = corrected.verify_corrected_bundle()
    authority = result["authority"]
    mapping = result["mapping"]
    assert authority["counts"]["logical_initial_snr_cells"] == 6048
    assert authority["counts"]["logical_all_roles_snr_cells"] == 12096
    assert mapping["mapping_count"] == authority["counts"]["logical_all_roles_snr_cells"]
    assert len({row["logical_candidate_id"] for row in mapping["mapping_rows"]}) == mapping["mapping_count"]
    assert all(row["measurement_identity_id"] for row in mapping["mapping_rows"])


def test_measurement_authority_excludes_snr_exactly() -> None:
    result = corrected.verify_corrected_bundle()
    rows = result["mapping"]["mapping_rows"]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row["measurement_identity_id"], []).append(row)
    assert all(len(items) == 21 for items in grouped.values())
    assert all(len({item["snr_db"] for item in items}) == 21 for items in grouped.values())
    assert result["authority"]["identity_semantics"]["snr_excluded_from_structural_identity"] is True


def test_modulation_and_rate_changes_cannot_alias_structural_identity() -> None:
    rows = corrected.verify_corrected_bundle()["authority"]["structural_identities"]
    same_cell = [row for row in rows if row["dataset"] == "imagenette160" and row["ratio"] == "r_1_6" and row["encode_axis_px"] == 160]
    assert len({row["structural_identity_id"] for row in same_cell}) == len(same_cell)
    assert len({row["modulation"] for row in same_cell}) > 1
    assert len({row["ldpc_rate"] for row in same_cell}) > 1


def test_same_ratio_axis_different_budget_changes_physical_cache_key() -> None:
    pixels = np.zeros((160, 160, 3), dtype=np.uint8)
    common = {
        "source_bytes": b"x",
        "canonical_pixels": pixels,
        "encode_axis_px": 160,
        "codec_configuration_hash": "a" * 64,
        "codec_runtime_identity": "openjpeg-test",
    }
    first = corrected.physical_cache_key(payload_budget_bytes=100, **common)
    second = corrected.physical_cache_key(payload_budget_bytes=101, **common)
    assert first.key_id != second.key_id


def test_exact_equal_full_cache_keys_are_reusable(tmp_path: Path) -> None:
    bundle = corrected.verify_corrected_bundle()
    backend = _Backend(bundle["contract"], failure_budget=10_000_000)
    cache = corrected.PhysicalCodecCache(tmp_path, backend)
    pixels = np.zeros((160, 160, 3), dtype=np.uint8)
    key = corrected.physical_cache_key(
        source_bytes=b"same",
        canonical_pixels=pixels,
        payload_budget_bytes=100,
        encode_axis_px=160,
        codec_configuration_hash=bundle["contract"]["codec"]["configuration_hash"],
        codec_runtime_identity=str(bundle["contract"]["codec"]["snapshot"]["environment"]),
    )
    first = cache.get_or_create(key, pixels)
    second = cache.get_or_create(key, pixels)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert backend.calls == 1


def test_cache_reuse_does_not_merge_logical_candidates(synthetic_stack: dict[str, object]) -> None:
    records = []
    for unit in synthetic_stack["work_units"]:
        records.append(synthetic_stack["executor"](unit, synthetic_stack["sample"]))
    assert len({record.value["work_unit_id"] for record in records}) == len(records)
    assert len({record.value["measurement_identity_id"] for record in records}) == len(records)
    assert len({record.value["physical_cache_key"]["payload_budget_bytes"] for record in records}) > 1


def test_wrong_packet_payload_budget_rejected(synthetic_stack: dict[str, object]) -> None:
    unit = synthetic_stack["work_units"][0]
    record = synthetic_stack["executor"](unit, synthetic_stack["sample"]).as_dict()
    record["packet_budget"]["payload_budget_bytes"] += 1
    record["record_id"] = corrected._id(corrected.RECORD_PREFIX, {key: value for key, value in record.items() if key != "record_id"})
    with pytest.raises(corrected.CorrectedG8EError, match="payload budget"):
        corrected.MeasurementRecord.from_mapping(record)


def test_same_count_wrong_stable_id_rejected(synthetic_stack: dict[str, object]) -> None:
    unit = synthetic_stack["work_units"][0]
    record = synthetic_stack["executor"](unit, synthetic_stack["sample"]).as_dict()
    record["stable_sample_id"] = "0" * len(record["stable_sample_id"])
    record["record_id"] = corrected._id(corrected.RECORD_PREFIX, {key: value for key, value in record.items() if key != "record_id"})
    with pytest.raises(corrected.CorrectedG8EError, match="stable sample ID"):
        corrected.MeasurementRecord.from_mapping(record)


def test_old_pb3_authorization_rejects_corrected_selection() -> None:
    old = G8Authorization("G-8", "old-test", "old smoke scope", 64, 25)
    with pytest.raises(SweepBudgetError):
        corrected.authorization_scope_accepts(candidates=1008, samples=1000, authorization=old)
    with pytest.raises(SweepBudgetError):
        corrected.authorization_scope_accepts(candidates=1008, samples=1000, authorization=None)


def test_corrected_exact_authorization_and_plus_one_refusals() -> None:
    auth = G8Authorization("G-8", "owner-test-only", "bounded authorization fixture", 1008, 1000)
    budget = corrected.authorization_scope_accepts(candidates=1008, samples=1000, authorization=auth)
    assert budget.max_candidates == 1008
    assert budget.max_samples == 1000
    assert budget.max_workload is None
    with pytest.raises(SweepBudgetError):
        corrected.authorization_scope_accepts(candidates=1009, samples=1000, authorization=auth)
    with pytest.raises(SweepBudgetError):
        corrected.authorization_scope_accepts(candidates=1008, samples=1001, authorization=auth)


def test_selection_call_plan_is_derived_and_typed_workload_is_none() -> None:
    plan = corrected.verify_corrected_bundle()["contract"]["selection_authorization"]
    assert plan["max_candidates"] == 1008
    assert plan["max_samples"] == 1000
    assert plan["typed_max_workload"] is None
    assert plan["artifact_only_workload_bound"] == 1008000
    assert plan["call_count"] == 18


def test_clean_failure_denominator_and_outage_separation(synthetic_stack: dict[str, object]) -> None:
    campaign = synthetic_stack["campaign"]
    with pytest.raises(RuntimeError, match="synthetic crash"):
        campaign.run_next(crash_after="record")
    campaign.run_all()
    records = [json.loads(path.read_text()) for path in sorted((synthetic_stack["runtime"] / "records").glob("*.json"))]
    e4 = corrected.aggregate_e4_counts(
        authority=synthetic_stack["authority"],
        sample_ids=synthetic_stack["sample_ids"],
        record_values=records,
        production=False,
    )
    assert e4["outage_term_included"] is False
    assert all("accuracy" not in obj for obj in e4["objects"])
    assert any(obj["status"] == "eligible" and obj["total_count"] == 1 for obj in e4["objects"])
    assert any(obj["status"] == "ineligible" for obj in e4["objects"])
    assert all(record["outage_applied"] is False for record in records)


def test_missing_and_duplicate_work_units_rejected(synthetic_stack: dict[str, object]) -> None:
    records = [synthetic_stack["executor"](unit, synthetic_stack["sample"]) for unit in synthetic_stack["work_units"]]
    with pytest.raises(corrected.CorrectedG8EError, match="exact-set mismatch"):
        corrected.merge_e3_records(
            authority=synthetic_stack["authority"],
            sample_ids=synthetic_stack["sample_ids"],
            record_values=records[:-1],
            production=False,
        )
    with pytest.raises(corrected.CorrectedG8EError, match="duplicate"):
        corrected.merge_e3_records(
            authority=synthetic_stack["authority"],
            sample_ids=synthetic_stack["sample_ids"],
            record_values=records + [records[0]],
            production=False,
        )


def test_e3_exact_merge_and_e4_counts(synthetic_stack: dict[str, object]) -> None:
    records = [synthetic_stack["executor"](unit, synthetic_stack["sample"]) for unit in synthetic_stack["work_units"]]
    merged = corrected.merge_e3_records(
        authority=synthetic_stack["authority"],
        sample_ids=synthetic_stack["sample_ids"],
        record_values=records,
        production=False,
    )
    assert merged["work_unit_count"] == len(records)
    result = corrected.aggregate_e4_counts(
        authority=synthetic_stack["authority"],
        sample_ids=synthetic_stack["sample_ids"],
        record_values=records,
        production=False,
    )
    assert result["object_count"] == len(synthetic_stack["work_units"])
    assert result["scientific_evidence"] is False
    assert result["merge_eligible"] is False


def test_test_split_and_training_are_rejected(synthetic_stack: dict[str, object]) -> None:
    sample = synthetic_stack["sample"]
    bad = corrected.SyntheticSample(
        stable_sample_id=sample.stable_sample_id,
        label=sample.label,
        source_bytes=sample.source_bytes,
        canonical_pixels=sample.canonical_pixels,
        split="test",
    )
    with pytest.raises(corrected.CorrectedG8EError, match="validation"):
        synthetic_stack["executor"](synthetic_stack["work_units"][0], bad)
    record = synthetic_stack["executor"](synthetic_stack["work_units"][0], sample).as_dict()
    record["training"] = 1
    with pytest.raises(corrected.CorrectedG8EError, match="record ID differs|safety"):
        corrected.MeasurementRecord.from_mapping(record)


def test_acc_clean_never_accepts_outage_value(synthetic_stack: dict[str, object]) -> None:
    record = synthetic_stack["executor"](synthetic_stack["work_units"][0], synthetic_stack["sample"]).as_dict()
    record["outage_applied"] = True
    with pytest.raises(corrected.CorrectedG8EError):
        corrected.MeasurementRecord.from_mapping(record)
    assert "accuracy" not in record


def test_no_full_validation_decoding_or_production_runtime_in_corrective_tests() -> None:
    assert not corrected.CORRECTED_RUNTIME_ROOT.exists()
    assert not (corrected.CORRECTED_ROOT / "e2_execution_authorization.json").exists()
    assert corrected.verify_corrected_bundle()["contract"]["safety"]["validation_decoding"] == 0
