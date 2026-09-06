from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baseline import g8_d


def _campaign_fixture(tmp_path: Path, count: int = 3):
    contract = g8_d.build_g8_d_contract()
    split = g8_d.ValidationSplitIdentity.from_mapping(
        next(item for item in contract["validation_split_bindings"] if item["dataset"] == "imagenette160")
    )
    classifier = g8_d.ClassifierIdentity.from_mapping(contract["classifier_binding"])
    table = g8_d.G8CTableIdentity.from_mapping(contract["g8_c_binding"])
    codec = g8_d.CodecConfigurationIdentity.from_mapping(contract["codec_binding"])
    work_units: list[g8_d.WorkUnitIdentity] = []
    records: dict[str, g8_d.CleanClassifierMeasurementRecord] = {}
    for ordinal in range(count):
        source = f"d5-source-{ordinal}".encode()
        image = g8_d.ImageIdentity.from_pixels(
            split_identity=split,
            stable_sample_id=f"d5-image-{ordinal}",
            source_bytes=source,
            canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        budget = g8_d.BudgetIdentity(
            bw_ratio=f"fixture-{ordinal}",
            bytes_sent=80,
            payload_bytes=80,
            packet_accounting={"payload_bytes": 80, "channel_bits": 640},
        )
        candidate = g8_d.CandidateIdentity(
            image_identity_id=image.identity_id,
            budget_identity_id=budget.identity_id,
            codec_configuration_id=codec.identity_id,
            g8_c_table_identity_id=table.identity_id,
            bler_identity={
                "k_and_n": [128, 256],
                "base_graph": 2,
                "lifting_size": 22,
                "modulation": "qpsk",
                "decoder_algorithm": "offset_min_sum",
                "decoder_offset": 0.5,
                "iterations": 50,
                "snr_convention": "es_n0_per_symbol",
                "rate": "1/2",
            },
            snr_db=float(ordinal),
            encode_axis_px=8,
        )
        work_unit = g8_d.WorkUnitIdentity(contract["campaign_id"], ordinal, candidate.identity_id)
        reconstruction = g8_d.ReconstructionIdentity(
            image.identity_id,
            "g8demitted-" + f"{ordinal:01x}" * 64,
            codec.identity_id,
            (8, 8, 3),
            "bicubic",
            True,
        )
        record = g8_d.CleanClassifierMeasurementRecord.from_outcomes(
            work_unit=work_unit,
            candidate=candidate,
            image=image,
            validation_split=split,
            classifier=classifier,
            g8_c_table=table,
            reconstruction=reconstruction,
            reconstruction_cache_object_id="g8dreconobj-" + f"{ordinal + 1:01x}" * 64,
            outcomes=[True, ordinal % 2 == 0, False],
            source="d5-synthetic-fixture",
        )
        work_units.append(work_unit)
        records[work_unit.identity_id] = record
    calls = {"count": 0}

    def factory(work_unit: g8_d.WorkUnitIdentity) -> g8_d.CleanClassifierMeasurementRecord:
        calls["count"] += 1
        return records[work_unit.identity_id]

    def make_campaign(hook=None) -> g8_d.AtomicMeasurementCampaign:
        return g8_d.AtomicMeasurementCampaign(
            tmp_path,
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=work_units,
            record_factory=factory,
            hook=hook,
        )

    return contract, work_units, records, calls, make_campaign


class _CrashOnce:
    def __init__(self, event: str) -> None:
        self.event = event
        self.fired = False

    def __call__(self, event: str, work_unit: g8_d.WorkUnitIdentity) -> None:
        if event == self.event and not self.fired:
            self.fired = True
            raise RuntimeError(f"injected crash at {event} for {work_unit.identity_id}")


def test_atomic_campaign_commits_exact_prefix_and_reuses_complete_output(tmp_path, post_g10_am94) -> None:
    del post_g10_am94
    _contract, work_units, _records, calls, make_campaign = _campaign_fixture(tmp_path)
    campaign = make_campaign()
    campaign.initialize()
    results = campaign.run_all()
    assert len(results) == len(work_units)
    state = campaign.read_state()
    assert state["completed_work_unit_ids"] == [work_unit.identity_id for work_unit in work_units]
    assert state["in_progress_work_unit_id"] is None
    assert len(state["record_refs"]) == len(work_units)
    assert len(state["cache_refs"]) == len(work_units)
    assert state["aggregate_ref"]["record_count"] == len(work_units)
    assert calls["count"] == len(work_units)

    assert campaign.run_all() == ()
    reused = campaign.run_next()
    assert reused.reused is True
    assert reused.completed_count == len(work_units)
    assert calls["count"] == len(work_units)


@pytest.mark.parametrize(
    "event",
    [
        "before_cache_publication",
        "after_cache_publication",
        "after_record_publication",
        "after_aggregate_publication",
        "before_state_publication",
    ],
)
def test_atomic_campaign_recovers_each_publication_boundary(tmp_path, event, post_g10_am94) -> None:
    del post_g10_am94
    _contract, work_units, _records, _calls, make_campaign = _campaign_fixture(tmp_path)
    crashing = make_campaign(_CrashOnce(event))
    crashing.initialize()
    with pytest.raises(RuntimeError, match=event):
        crashing.run_next()

    resumed = make_campaign()
    results = resumed.run_all()
    assert len(results) == len(work_units)
    assert resumed.read_state()["completed_work_unit_ids"] == [work_unit.identity_id for work_unit in work_units]


def test_changed_contract_and_semantically_stale_state_fail_closed(tmp_path, post_g10_am94) -> None:
    del post_g10_am94
    contract, _work_units, _records, _calls, make_campaign = _campaign_fixture(tmp_path, count=1)
    campaign = make_campaign()
    campaign.initialize()
    stale = json.loads(campaign.state_path.read_bytes())
    stale["contract_id"] = "g8dcontract-" + "f" * 64
    campaign.state_path.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
    with pytest.raises(g8_d.G8DContractError, match="semantic digest|contract"):
        campaign.read_state()

    with pytest.raises(g8_d.G8DContractError, match="current authenticated contract"):
        g8_d.AtomicMeasurementCampaign(
            tmp_path / "changed",
            contract_id="g8dcontract-" + "f" * 64,
            campaign_id=contract["campaign_id"],
            work_units=_work_units,
            record_factory=lambda work_unit: _records[work_unit.identity_id],
        )


def test_duplicate_or_missing_order_is_rejected_before_any_work(tmp_path, post_g10_am94) -> None:
    del post_g10_am94
    contract, work_units, records, _calls, _make_campaign = _campaign_fixture(tmp_path)
    with pytest.raises(g8_d.G8DContractError, match="duplicate"):
        g8_d.AtomicMeasurementCampaign(
            tmp_path / "duplicate",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[work_units[0], g8_d.WorkUnitIdentity(contract["campaign_id"], 1, work_units[0].candidate_identity_id)],
            record_factory=lambda work_unit: records[work_unit.identity_id],
        )
    with pytest.raises(g8_d.G8DContractError, match="contiguous"):
        g8_d.AtomicMeasurementCampaign(
            tmp_path / "missing",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[work_units[0], g8_d.WorkUnitIdentity(contract["campaign_id"], 2, work_units[1].candidate_identity_id)],
            record_factory=lambda work_unit: records[work_unit.identity_id],
        )


def test_corrupted_record_or_cache_is_not_reused(tmp_path, post_g10_am94) -> None:
    del post_g10_am94
    _contract, work_units, _records, _calls, make_campaign = _campaign_fixture(tmp_path, count=1)
    campaign = make_campaign()
    campaign.run_next()
    record_path = campaign.records_dir / f"{work_units[0].identity_id}.json"
    record_path.write_bytes(record_path.read_bytes().replace(b"d5-synthetic-fixture", b"mutated-fixture"))
    with pytest.raises(g8_d.G8DContractError):
        campaign.read_state()


def test_stale_aggregate_ahead_of_durable_evidence_is_rejected(tmp_path, post_g10_am94) -> None:
    del post_g10_am94
    _contract, work_units, records, _calls, make_campaign = _campaign_fixture(tmp_path, count=2)
    campaign = make_campaign()
    campaign.run_next()
    state = campaign.read_state()
    first_ref = state["record_refs"][0]
    second = records[work_units[1].identity_id]
    second_raw = g8_d.rendered_json(second.as_dict())
    second_ref = campaign._record_ref(work_units[1], second, second_raw)
    ahead = campaign._aggregate_payload(
        work_units,
        [records[work_units[0].identity_id], second],
        [first_ref, second_ref],
    )
    (campaign.aggregates_dir / f"{ahead['aggregate_id']}.json").write_bytes(g8_d.rendered_json(ahead))
    with pytest.raises(g8_d.G8DContractError, match="stale aggregate"):
        campaign.read_state()
