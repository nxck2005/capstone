"""Pre-data regression coverage for the G8_E corrected-v2 contract.

Every fixture here is NON-SCIENTIFIC, NON-SELECTION, NOT PRODUCTION E2
EVIDENCE, and MERGE-INELIGIBLE FOR PRODUCTION.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline import g8_e_corrected_v2 as v2


def _require_v2_bundle():
    if not v2.V2_CONTRACT_PATH.is_file():
        pytest.skip("v2 evidence bundle is generated after the source-freeze commit")


class _SyntheticBR11:
    def as_dict(self):
        return {"synthetic": True, "record_labels": ["NON-SCIENTIFIC"]}


@pytest.fixture
def fake_br11(monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", lambda *args, **kwargs: _SyntheticBR11())


def _context(n: int = 1000):
    labels = [0] * 37 + [1] * 113 + [2] * 211 + [3] * 17 + [4] * 89 + [5] * 71 + [6] * 143 + [7] * 97 + [8] * 151 + [9] * 71
    assert len(labels) == 1000
    labels = labels[:n]
    structural = {
        "structural_identity_id": "g8e-synthetic-structural-" + "a" * 64,
        "dataset": v2.INITIAL_DATASET,
        "dataset_role": "headline",
        "source_codec": "jpeg2000",
        "ratio": "r_1_2",
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "encode_axis_px": 2,
        "packet_config_id": "synthetic-packet",
        "payload_budget_bytes": 10,
        "packet_accounting": {"payload_bytes": 10},
    }
    authority = {
        "authority_id": "g8e-synthetic-authority-" + "b" * 64,
        "structural_identities": [structural],
        "logical_candidate_to_structural_id": {},
    }
    direct = {"synthetic": True, "labels": ["NON-SCIENTIFIC"]}
    contract = {
        "campaign_id": "g8e-synthetic-campaign-" + "c" * 64,
        "contract_id": "g8e-synthetic-contract-" + "d" * 64,
        "execution_profile": {"profile_id": "synthetic-profile"},
        "source_manifest": {"source_commit": "synthetic", "id": "synthetic-source"},
        "direct_upstream_bindings": direct,
        "outage_policy": {
            "selected_class": 2,
            "numerator": 211,
            "denominator": 1000,
            "selection_is_count_derived": True,
            "path": "synthetic/outage_policy.json",
            "sha256": "e" * 64,
        },
        "codec": {"configuration_hash": "f" * 64, "runtime_identity": "synthetic-codec"},
        "authorization": {
            "schema_scope_frozen": {
                "validation_decode": True, "test_access": False, "training": False,
                "fallback": False, "ratio_adjudication": False, "pass_one": False,
                "pass_two": False,
            }
        },
    }
    ids = tuple(f"synthetic-{index:04d}" for index in range(n))
    samples = {
        sample_id: v2.SyntheticSample(
            sample_id, labels[index], b"source-" + sample_id.encode(),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
        for index, sample_id in enumerate(ids)
    }
    units = tuple({
        "work_unit_id": v2._work_unit_id(structural["structural_identity_id"], sample_id),
        "ordinal": index,
        "measurement_identity_id": structural["structural_identity_id"],
        "logical_candidate_ids": [],
        "stable_sample_id": sample_id,
        "dataset": v2.INITIAL_DATASET,
        "split": v2.VALIDATION_SPLIT,
    } for index, sample_id in enumerate(ids))
    return {
        "labels": labels, "structural": structural, "authority": authority,
        "contract": contract, "ids": ids, "samples": samples, "units": units,
    }


def _records(ctx, *, infeasible: set[int] = frozenset(), wrong: set[int] = frozenset(), structural_invalid: bool = False):
    structural = dict(ctx["structural"])
    if structural_invalid:
        structural["structurally_legal"] = False
        ctx["authority"]["structural_identities"] = [structural]
    result = []
    for index, sample_id in enumerate(ctx["ids"]):
        sample = ctx["samples"][sample_id]
        unit = ctx["units"][index]
        key = v2.make_physical_cache_key(
            source_bytes=sample.source_bytes,
            canonical_pixels=sample.canonical_pixels,
            payload_budget_bytes=10,
            encode_axis_px=2,
            codec_configuration_hash="1" * 64,
            codec_runtime_identity="synthetic-codec",
        )
        if structural_invalid:
            codec = None
            record_key = None
            reconstruction = observation = None
        elif index in infeasible:
            codec = v2.CodecArtifactV2(
                key, v2.OUTCOME_CODEC_INFEASIBILITY, "synthetic budget",
                None, None, f"codec-{index}", False,
            )
            record_key = key
            reconstruction = observation = None
        else:
            codec = v2.CodecArtifactV2(key, "feasible", None, b"abc", 3, f"codec-{index}", False)
            record_key = key
            reconstruction = v2.ReconstructionArtifactV2(
                f"recon-{index}", "delivered", None, sample.canonical_pixels, False,
            )
            prediction = (sample.label + 1) % 10 if index in wrong else sample.label
            observation = v2.ClassifierObservationV2(
                f"observation-{index}", prediction, False,
            )
        result.append(v2.MeasurementRecordV2.build(
            campaign_id=ctx["contract"]["campaign_id"],
            contract_id=ctx["contract"]["contract_id"],
            authority=ctx["authority"],
            work_unit=unit,
            structural=structural,
            sample=sample,
            physical_key=record_key,
            codec=codec,
            reconstruction=reconstruction,
            observation=observation,
            outage_policy=ctx["contract"]["outage_policy"],
            profile_id="synthetic-profile",
            source_commit="synthetic",
            g8_c_linkage_digest=v2.sha256_bytes(
                v2.canonical_json(ctx["contract"]["direct_upstream_bindings"])
            ),
            record_labels=(),
        ))
    return result


def test_uneven_outage_fixture_and_all_rows(fake_br11):
    ctx = _context()
    records = _records(ctx, wrong=set(range(400)))
    e4 = v2.aggregate_e4_counts_v2(
        authority=ctx["authority"], sample_ids=ctx["ids"],
        record_values=records, contract=ctx["contract"],
    )
    obj = e4["objects"][0]
    assert obj["status"] == "eligible"
    assert obj["delivered_count"] == 1000
    assert obj["correct_count"] == 600
    assert obj["total_count"] == 1000
    assert e4["outage_accuracy"] == {
        "selected_class": 2, "numerator": 211, "denominator": 1000,
        "policy_path": "synthetic/outage_policy.json",
        "policy_sha256": "e" * 64, "selection_is_count_derived": True,
    }


def test_seventy_codec_infeasible_rows_stay_in_denominator(fake_br11):
    ctx = _context()
    outage = set(range(930, 1000))
    e4 = v2.aggregate_e4_counts_v2(
        authority=ctx["authority"], sample_ids=ctx["ids"],
        record_values=_records(ctx, infeasible=outage), contract=ctx["contract"],
    )
    obj = e4["objects"][0]
    assert obj["status"] == "eligible"
    assert obj["delivered_count"] == 930
    assert obj["codec_infeasibility_count"] == 70
    assert obj["total_count"] == 1000
    assert obj["correct_count"] == 930 + sum(ctx["labels"][index] == 2 for index in outage)
    assert obj["infeasible_rate"] == pytest.approx(0.07)


def test_codec_infeasibility_concentrated_in_or_out_of_outage_class(fake_br11):
    ctx = _context()
    in_class = {index for index, label in enumerate(ctx["labels"]) if label == 2} 
    in_class = set(sorted(in_class)[:70])
    out_class = {index for index, label in enumerate(ctx["labels"]) if label != 2}
    out_class = set(sorted(out_class)[:70])
    for selected, expected in ((in_class, 1000), (out_class, 930)):
        e4 = v2.aggregate_e4_counts_v2(
            authority=ctx["authority"], sample_ids=ctx["ids"],
            record_values=_records(ctx, infeasible=selected), contract=ctx["contract"],
        )
        assert e4["objects"][0]["correct_count"] == expected
        assert e4["objects"][0]["total_count"] == 1000


def test_br4_p_one_p_zero_and_source_outage_is_not_omitted(fake_br11):
    ctx = _context()
    obj = v2.aggregate_e4_counts_v2(
        authority=ctx["authority"], sample_ids=ctx["ids"],
        record_values=_records(ctx, wrong=set(range(400))), contract=ctx["contract"],
    )["objects"][0]
    assert v2.compose_expected_accuracy(
        p_success=1, acc_clean_correct=obj["correct_count"],
        acc_clean_total=obj["total_count"],
        acc_outage_numerator=211, acc_outage_denominator=1000,
    ) == pytest.approx(0.6)
    assert v2.compose_expected_accuracy(
        p_success=0, acc_clean_correct=obj["correct_count"],
        acc_clean_total=obj["total_count"],
        acc_outage_numerator=211, acc_outage_denominator=1000,
    ) == pytest.approx(0.211)
    source_outage = v2.score_outage(2, 2)
    for probability in (0, 0.37, 1):
        assert probability * source_outage[0] + (1 - probability) * source_outage[0] == 1
    assert "outage_accuracy" in v2.aggregate_e4_counts_v2(
        authority=ctx["authority"], sample_ids=ctx["ids"],
        record_values=_records(ctx), contract=ctx["contract"],
    )


def test_structural_infeasibility_preserves_candidate_level_policy(fake_br11):
    ctx = _context(3)
    e4 = v2.aggregate_e4_counts_v2(
        authority=ctx["authority"], sample_ids=ctx["ids"],
        record_values=_records(ctx, structural_invalid=True), contract=ctx["contract"],
    )
    assert e4["objects"][0]["status"] == "ineligible"
    assert e4["objects"][0]["total_count"] is None


@pytest.mark.parametrize("failure", ["codec", "decoder", "classifier", "malformed", "budget", "shape"])
def test_unexpected_runtime_failures_never_become_scientific_rows(tmp_path, fake_br11, failure):
    ctx = _context(1)
    if failure == "codec":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend")))
        decoder = lambda stream: np.zeros((2, 2, 3), dtype=np.uint8)
        classifier = SimpleNamespace(predict=lambda pixels: 0)
    elif failure == "decoder":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: SimpleNamespace(feasible=True, codestream=b"abc", emitted_byte_count=3))
        decoder = lambda stream: (_ for _ in ()).throw(RuntimeError("decoder"))
        classifier = SimpleNamespace(predict=lambda pixels: 0)
    elif failure == "classifier":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: SimpleNamespace(feasible=True, codestream=b"abc", emitted_byte_count=3))
        decoder = lambda stream: np.zeros((2, 2, 3), dtype=np.uint8)
        classifier = SimpleNamespace(predict=lambda pixels: (_ for _ in ()).throw(RuntimeError("classifier")))
    elif failure == "malformed":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: SimpleNamespace(feasible="yes", codestream=b"abc", emitted_byte_count=3))
        decoder = lambda stream: np.zeros((2, 2, 3), dtype=np.uint8)
        classifier = SimpleNamespace(predict=lambda pixels: 0)
    elif failure == "budget":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: SimpleNamespace(feasible=True, codestream=b"01234567890", emitted_byte_count=11))
        decoder = lambda stream: np.zeros((2, 2, 3), dtype=np.uint8)
        classifier = SimpleNamespace(predict=lambda pixels: 0)
    else:
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: SimpleNamespace(feasible=True, codestream=b"abc", emitted_byte_count=3))
        decoder = lambda stream: np.zeros((5, 5, 3), dtype=np.uint8)
        classifier = SimpleNamespace(predict=lambda pixels: 0)
    engine = v2.MeasurementExecutorV2(
        contract=ctx["contract"], authority=ctx["authority"], runtime_root=tmp_path,
        backend=backend, decoder=decoder, classifier=classifier,
        non_scientific_fixture=True,
    )
    with pytest.raises(v2.FatalExecutionError):
        engine.execute(ctx["units"][0], ctx["samples"][ctx["ids"][0]])
    assert not (tmp_path / "records").exists()


def test_cache_corruption_is_fatal_and_full_key_controls_reuse(tmp_path):
    calls = {"n": 0}

    class Backend:
        def encode_to_budget(self, image, **kwargs):
            calls["n"] += 1
            return SimpleNamespace(feasible=False, codestream=None, emitted_byte_count=None, reason="synthetic")

    key = v2.PhysicalCacheKey("1" * 64, "2" * 64, (2, 2, 3), 10, 2, "3" * 64, "runtime")
    cache = v2.PhysicalCodecCacheV2(tmp_path, Backend())
    first = cache.get_or_create(key, np.zeros((2, 2, 3), dtype=np.uint8))
    second = cache.get_or_create(key, np.zeros((2, 2, 3), dtype=np.uint8))
    assert first.cache_hit is False and second.cache_hit is True and calls["n"] == 1
    different_budget = v2.PhysicalCacheKey("1" * 64, "2" * 64, (2, 2, 3), 11, 2, "3" * 64, "runtime")
    cache.get_or_create(different_budget, np.zeros((2, 2, 3), dtype=np.uint8))
    assert calls["n"] == 2
    cache._path(key).write_text("{}")
    with pytest.raises(v2.FatalExecutionError):
        cache.get_or_create(key, np.zeros((2, 2, 3), dtype=np.uint8))


def test_classifier_observation_cache_is_content_and_checkpoint_bound(tmp_path):
    calls = {"n": 0}

    class Classifier:
        def predict(self, pixels):
            calls["n"] += 1
            return 4

    cache = v2.ClassifierObservationCacheV2(
        tmp_path, Classifier(), checkpoint_sha256="1" * 64,
        config_identity="cfg", runtime_identity="rt",
    )
    one = v2.ReconstructionArtifactV2("recon-a", "delivered", None, np.zeros((2, 2, 3), dtype=np.uint8), False)
    two = v2.ReconstructionArtifactV2("recon-b", "delivered", None, np.ones((2, 2, 3), dtype=np.uint8), False)
    assert cache.get_or_create(one).predicted_label == 4
    assert cache.get_or_create(one).cache_hit is True
    assert cache.get_or_create(two).cache_hit is False
    assert calls["n"] == 2
    identity = cache._identity(one, one.pixels)
    path = tmp_path / "observation" / f"{v2._id(v2.V2_OBSERVATION_PREFIX, identity)}.json"
    value = json.loads(path.read_text())
    value["predicted_label"] = 3
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(v2.FatalExecutionError):
        cache.get_or_create(one)


def test_transaction_is_compact_and_resume_reconciles_once(tmp_path, fake_br11):
    ctx = _context(8)
    records = _records(ctx)
    by_unit = dict(zip((unit["work_unit_id"] for unit in ctx["units"]), records))
    executor = lambda unit, sample: by_unit[unit["work_unit_id"]]
    campaign = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=executor,
        sample_provider=lambda sid: ctx["samples"][sid], mode="start",
    )
    assert campaign.reconciliation_record_visits == 0
    with pytest.raises(RuntimeError):
        campaign.run_next(crash_after="record")
    assert campaign.state()["completed_prefix_count"] == 0
    resumed = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=executor,
        sample_provider=lambda sid: ctx["samples"][sid], mode="resume",
    )
    assert resumed.reconciliation_record_visits == 1
    resumed.run_all()
    assert resumed.state()["completed_prefix_count"] == 8
    assert resumed.state()["status"] == v2.COMPLETE_STATUS
    assert "completed_work_unit_ids" not in resumed.state()
    assert not (tmp_path / "aggregates").exists()
    assert resumed.reconciliation_record_visits == 1


def test_transaction_chain_mutation_and_prefix_loss_are_rejected(tmp_path, fake_br11):
    ctx = _context(3)
    records = _records(ctx)
    by_unit = dict(zip((unit["work_unit_id"] for unit in ctx["units"]), records))
    executor = lambda unit, sample: by_unit[unit["work_unit_id"]]
    campaign = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=executor,
        sample_provider=lambda sid: ctx["samples"][sid], mode="start",
    )
    campaign.run_all()
    state = json.loads((tmp_path / "campaign_state.json").read_text())
    state["rolling_prefix_digest"] = "0" * 64
    state["state_sha256"] = v2.sha256_bytes(
        v2.canonical_json({key: value for key, value in state.items() if key != "state_sha256"})
    )
    (tmp_path / "campaign_state.json").write_bytes(v2.rendered_json(state))
    with pytest.raises(v2.G8EV2Error):
        v2.AtomicE2CampaignV2(
            runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
            work_units=ctx["units"], executor=executor,
            sample_provider=lambda sid: ctx["samples"][sid], mode="resume",
        )


def test_e3_exact_set_rejects_missing_extra_reordered_and_substituted(fake_br11):
    ctx = _context(4)
    records = _records(ctx)
    kwargs = {"authority": ctx["authority"], "sample_ids": ctx["ids"], "contract": ctx["contract"]}
    with pytest.raises(v2.G8EV2Error):
        v2.merge_e3_records_v2(record_values=records[:-1], **kwargs)
    with pytest.raises(v2.G8EV2Error):
        v2.merge_e3_records_v2(record_values=records + [records[0]], **kwargs)
    with pytest.raises(v2.G8EV2Error):
        v2.merge_e3_records_v2(record_values=list(reversed(records)), **kwargs)
    substituted = list(records)
    substituted[0] = records[1]
    with pytest.raises(v2.G8EV2Error):
        v2.merge_e3_records_v2(record_values=substituted, **kwargs)
    merged = v2.merge_e3_records_v2(record_values=records, **kwargs)
    assert merged["work_unit_count"] == 4


def test_start_resume_matrix_and_authorization_before_payload(tmp_path, fake_br11):
    assert v2.check_runtime_mode("start", tmp_path) is None
    (tmp_path / "existing").write_text("runtime")
    with pytest.raises(v2.G8EV2Error, match="resume"):
        v2.check_runtime_mode("start", tmp_path)
    with pytest.raises(v2.G8EV2Error, match="start"):
        v2.check_runtime_mode("resume", tmp_path / "missing")
    ctx = _context(1)
    (tmp_path / "existing").unlink()
    campaign = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=lambda unit, sample: _records(ctx)[0],
        sample_provider=lambda sid: ctx["samples"][sid], mode="start",
    )
    assert campaign.state()["completed_prefix_count"] == 0
    with pytest.raises(v2.G8EV2Error, match="foreign"):
        foreign = dict(ctx["contract"])
        foreign["campaign_id"] = "foreign"
        v2.AtomicE2CampaignV2(
            runtime_root=tmp_path, contract=foreign, authority=ctx["authority"],
            work_units=ctx["units"], executor=lambda unit, sample: _records(ctx)[0],
            sample_provider=lambda sid: ctx["samples"][sid], mode="resume",
        )


def test_runner_refuses_without_owner_artifact_before_validation():
    _require_v2_bundle()
    contract_path = Path("results/baseline/g8_e/e1_corrected_v2/measurement_contract.json")
    campaign_id = json.loads(contract_path.read_text())["campaign_id"]
    result = subprocess.run(
        [sys.executable, "tools/run_g8_e_corrected_v2.py", "--start", "--campaign-id", campaign_id],
        cwd=Path.cwd(), env={"PYTHONPATH": "src"}, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "before validation payload decode" in result.stderr
    assert not Path("results/baseline/g8_e/e1_corrected_v2/runtime").exists()


def test_provenance_and_safety_boundaries():
    _require_v2_bundle()
    contract = v2.verify_bundle(verify_live_sources=True)["contract"]
    with pytest.raises(v2.G8EV2Error):
        v2.reject_superseded_campaign(v2.FIRST_CORRECTED_CAMPAIGN_ID)
    assert contract["declarations"]["test_split_sealed"] is True
    assert contract["safety"]["training"] == 0
    assert contract["safety"]["fallback_invoked"] is False
    assert contract["safety"]["pass_one_started"] is False
    with pytest.raises(v2.G8EV2Error):
        v2.SyntheticSample("test", 0, b"x", np.zeros((2, 2, 3), dtype=np.uint8), split="test")
    drifted = json.loads(Path("results/baseline/g8_e/e1_corrected_v2/execution_source_manifest.json").read_text())
    drifted["source_entries"][0]["sha256"] = "0" * 64
    with pytest.raises(v2.G8EV2Error):
        v2.validate_source_manifest(drifted)


def test_owner_authorization_schema_is_frozen_but_not_issued(tmp_path):
    _require_v2_bundle()
    bundle = v2.verify_bundle()["contract"]
    scope = bundle["authorization"]["schema_scope_frozen"]
    value = {
        "schema_version": 2,
        "artifact_role": "g8_e_v2_owner_e2_authorization",
        "status": "AUTHORIZED",
        "authorized_by": "synthetic-owner",
        "reason": "NON-SCIENTIFIC fixture only",
        "campaign_id": bundle["campaign_id"],
        "contract_id": bundle["contract_id"],
        "source_manifest_id": bundle["source_manifest"]["id"],
        "profile_id": bundle["execution_profile"]["profile_id"],
        "scope": scope,
        "issued_sha256": None,
    }
    value["issued_sha256"] = v2.sha256_bytes(
        v2.canonical_json({key: child for key, child in value.items() if key != "issued_sha256"})
    )
    path = tmp_path / "synthetic-authorization.json"
    path.write_bytes(v2.rendered_json(value))
    assert v2.authenticate_owner_authorization_v2(path, bundle)["status"] == "AUTHORIZED"
    assert bundle["authorization"]["issued"] is False


def test_typed_scientific_decode_failure_is_a_counted_outage(fake_br11, tmp_path):
    ctx = _context(1)
    backend = SimpleNamespace(
        encode_to_budget=lambda *args, **kwargs: SimpleNamespace(
            feasible=True, codestream=b"abc", emitted_byte_count=3,
        )
    )
    engine = v2.MeasurementExecutorV2(
        contract=ctx["contract"], authority=ctx["authority"], runtime_root=tmp_path,
        backend=backend, decoder=lambda stream: v2.ScientificDecodeFailure("explicit protocol decode result"),
        classifier=SimpleNamespace(predict=lambda pixels: 0), non_scientific_fixture=True,
    )
    record = engine.execute(ctx["units"][0], ctx["samples"][ctx["ids"][0]])
    assert record.value["outcome"] == v2.OUTCOME_DECODE_FAILURE
    assert record.value["outage_applied"] is True
    assert record.value["total_count"] == 1
    assert record.value["inference"] == 0


def test_campaign_failure_enters_hold_and_resume_retries_same_unit(tmp_path, fake_br11):
    ctx = _context(1)
    bad = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=lambda unit, sample: (_ for _ in ()).throw(RuntimeError("unexpected")),
        sample_provider=lambda sid: ctx["samples"][sid], mode="start",
    )
    with pytest.raises(v2.CampaignHoldError):
        bad.run_next()
    state = bad.state()
    assert state["status"] == v2.HOLD_STATUS
    assert state["completed_prefix_count"] == 0
    assert state["counters"]["validation_decoding"] == 0
    assert list((tmp_path / "records").glob("*.json")) == []
    assert list((tmp_path / "diagnostics").glob("*.json"))
    good_record = _records(ctx)[0]
    resumed = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=lambda unit, sample: good_record,
        sample_provider=lambda sid: ctx["samples"][sid], mode="resume",
    )
    resumed.run_next()
    assert resumed.state()["completed_prefix_count"] == 1
    assert resumed.state()["status"] == v2.COMPLETE_STATUS


def test_record_publication_failure_does_not_advance_prefix(tmp_path, fake_br11, monkeypatch):
    ctx = _context(1)
    record = _records(ctx)[0]
    campaign = v2.AtomicE2CampaignV2(
        runtime_root=tmp_path, contract=ctx["contract"], authority=ctx["authority"],
        work_units=ctx["units"], executor=lambda unit, sample: record,
        sample_provider=lambda sid: ctx["samples"][sid], mode="start",
    )
    real_publish = v2._atomic_publish

    def fail_records(path, payload):
        if Path(path).parent.name == "records":
            raise v2.FatalExecutionError("synthetic publication failure")
        return real_publish(path, payload)

    monkeypatch.setattr(v2, "_atomic_publish", fail_records)
    with pytest.raises(v2.CampaignHoldError):
        campaign.run_next()
    assert campaign.state()["completed_prefix_count"] == 0
    assert campaign.state()["status"] == v2.HOLD_STATUS
    assert list((tmp_path / "records").glob("*.json")) == []


def test_production_e3_rejects_marked_synthetic_records(fake_br11):
    ctx = _context(2)
    records = _records(ctx)
    records[0].value["record_labels"].append("NON-SCIENTIFIC")
    with pytest.raises(v2.G8EV2Error):
        v2.merge_e3_records_v2(
            authority=ctx["authority"], sample_ids=ctx["ids"],
            record_values=records, contract=ctx["contract"], production=True,
        )


def test_two_scientific_structural_identities_do_not_merge_by_cache_key(fake_br11):
    ctx = _context(2)
    second = dict(ctx["structural"])
    second["structural_identity_id"] = "g8e-synthetic-structural-" + "9" * 64
    second["modulation"] = "bpsk"
    ctx["authority"]["structural_identities"] = [ctx["structural"], second]
    ctx["authority"]["logical_candidate_to_structural_id"] = {}
    records = []
    for structural_index, structural in enumerate(ctx["authority"]["structural_identities"]):
        for sample_index, sample_id in enumerate(ctx["ids"]):
            sample = ctx["samples"][sample_id]
            unit = {
                "work_unit_id": v2._work_unit_id(structural["structural_identity_id"], sample_id),
                "ordinal": structural_index * 2 + sample_index,
                "measurement_identity_id": structural["structural_identity_id"],
                "logical_candidate_ids": [], "stable_sample_id": sample_id,
                "dataset": v2.INITIAL_DATASET, "split": v2.VALIDATION_SPLIT,
            }
            key = v2.make_physical_cache_key(
                source_bytes=sample.source_bytes, canonical_pixels=sample.canonical_pixels,
                payload_budget_bytes=10, encode_axis_px=2,
                codec_configuration_hash="1" * 64, codec_runtime_identity="synthetic-codec",
            )
            records.append(v2.MeasurementRecordV2.build(
                campaign_id=ctx["contract"]["campaign_id"], contract_id=ctx["contract"]["contract_id"],
                authority=ctx["authority"], work_unit=unit, structural=structural, sample=sample,
                physical_key=key,
                codec=v2.CodecArtifactV2(key, v2.OUTCOME_CODEC_INFEASIBILITY, "synthetic", None, None, f"codec-{structural_index}-{sample_index}", False),
                reconstruction=None, observation=None, outage_policy=ctx["contract"]["outage_policy"],
                profile_id="synthetic-profile", source_commit="synthetic",
                g8_c_linkage_digest=v2.sha256_bytes(v2.canonical_json(ctx["contract"]["direct_upstream_bindings"])),
            ))
    # The cache key can be identical while scientific structural identities remain distinct.
    assert records[0].value["physical_cache_key"] == records[2].value["physical_cache_key"]
    assert records[0].value["measurement_identity_id"] != records[2].value["measurement_identity_id"]
