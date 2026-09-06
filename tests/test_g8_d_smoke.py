from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from baseline import g8_d
from run_g8_d_smoke import MUTATION_CASES, run_smoke
from verify_g8_d_smoke import verify


def _fixture(tmp_path: Path) -> dict[str, object]:
    contract = g8_d.build_g8_d_contract()
    context = __import__("run_g8_d_smoke", fromlist=["_synthetic_context"])._synthetic_context(contract)
    engine = g8_d.CodecSearchEngine(
        tmp_path / "codec",
        backend=context["backend"],
        codec_identity=context["codec"],
    )
    image = context["image"]
    budget = context["budget"]
    search = engine.search(
        image_identity=image,
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=budget,
        encode_axis_px=8,
    )
    assert search.emitted_codestream is not None and search.emitted_identity is not None
    decoder = __import__("run_g8_d_smoke", fromlist=["_SmokeDecoder"])._SmokeDecoder()
    reconstruction_cache = g8_d.ReconstructionCache(
        tmp_path / "reconstruction",
        context["codec"],
        decoder=decoder,
    )
    reconstruction = reconstruction_cache.get_or_create(
        image_identity=image,
        emitted_file_identity=search.emitted_identity,
        codestream=search.emitted_codestream,
        output_shape=image.canonical_shape,
    )
    work_unit = g8_d.WorkUnitIdentity(contract["campaign_id"], 0, context["candidate"].identity_id)
    record = g8_d.CleanClassifierMeasurementRecord.from_outcomes(
        work_unit=work_unit,
        candidate=context["candidate"],
        image=image,
        validation_split=context["split"],
        classifier=context["classifier"],
        g8_c_table=context["table"],
        reconstruction=reconstruction.identity,
        reconstruction_cache_object_id=reconstruction.cache_object_id,
        outcomes=[True, False, True],
        source="d6-mutation-fixture",
    )
    return {
        "contract": contract,
        "context": context,
        "engine": engine,
        "search": search,
        "reconstruction_cache": reconstruction_cache,
        "reconstruction": reconstruction,
        "work_unit": work_unit,
        "record": record,
        "decoder": decoder,
    }


class _CrashBeforeRecord:
    def __call__(self, event: str, work_unit: g8_d.WorkUnitIdentity) -> None:
        del work_unit
        if event == "before_record_publication":
            raise RuntimeError("synthetic interrupted publication")


@pytest.mark.parametrize("case", MUTATION_CASES, ids=MUTATION_CASES)
def test_d6_required_mutation_matrix_rejects_or_preserves_boundary(
    tmp_path: Path, case: str, post_g10_am94
) -> None:
    del post_g10_am94
    fixture = _fixture(tmp_path)
    contract = fixture["contract"]
    context = fixture["context"]
    record = fixture["record"]

    if case == "wrong_g8_c_table_binding":
        payload = record.as_dict()
        payload["g8_c_table"]["table_sha256"] = "c" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "predecessor_table_instead_of_pascal_successor":
        payload = record.as_dict()
        payload["candidate"]["g8_c_table_identity_id"] = "g8dtable-" + "d" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "wrong_validation_manifest":
        payload = record.as_dict()
        payload["image"]["manifest_sha256"] = "e" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "test_split_requested":
        payload = record.as_dict()
        payload["validation_split"]["split"] = "test"
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "changed_classifier_checkpoint":
        payload = record.as_dict()
        payload["classifier"]["checkpoint_sha256"] = "f" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "codec_configuration_mutation":
        payload = record.as_dict()
        payload["candidate"]["codec_configuration_id"] = "g8dcodec-" + "a" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "image_content_mutation":
        payload = record.as_dict()
        payload["image"]["source_bytes_sha256"] = "b" * 64
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "cache_key_alias_attempt":
        image = context["image"]
        changed_image = g8_d.ImageIdentity.from_pixels(
            split_identity=context["split"],
            stable_sample_id="d6-alias-image",
            source_bytes=b"changed synthetic source",
            canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        first_path = tmp_path / "codec" / "codec_search" / f"{fixture['search'].search_key.identity_id}.json"
        changed_key = g8_d.CodecSearchKey(changed_image.identity_id, context["budget"].identity_id, context["codec"].identity_id, 8)
        changed_path = tmp_path / "codec" / "codec_search" / f"{changed_key.identity_id}.json"
        changed_path.write_bytes(first_path.read_bytes())
        with pytest.raises(g8_d.G8DContractError):
            fixture["engine"].search(
                image_identity=changed_image,
                encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
                budget=context["budget"],
                encode_axis_px=8,
            )
        del image
    elif case == "emitted_bytes_exceed_budget":
        from run_g8_d_smoke import _SmokeCodec

        backend = _SmokeCodec(codestream=b"x" * 31)
        engine = g8_d.CodecSearchEngine(tmp_path / "over", backend=backend, codec_identity=context["codec"])
        with pytest.raises(g8_d.G8DContractError, match="exceeds payload budget"):
            engine.search(
                image_identity=context["image"],
                encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
                budget=context["budget"],
                encode_axis_px=8,
            )
    elif case == "structural_infeasibility":
        result = fixture["engine"].search(
            image_identity=context["image"],
            encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
            budget=context["budget"],
            encode_axis_px=8,
            structurally_feasible=False,
            structural_reason="mutation structural boundary",
        )
        assert result.status == g8_d.STRUCTURAL_INFEASIBILITY
        assert fixture["context"]["backend"].calls == 1
    elif case == "codec_infeasibility":
        from run_g8_d_smoke import _SmokeCodec

        backend = _SmokeCodec(infeasible=True)
        engine = g8_d.CodecSearchEngine(tmp_path / "infeasible", backend=backend, codec_identity=context["codec"])
        result = engine.search(
            image_identity=context["image"],
            encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
            budget=context["budget"],
            encode_axis_px=8,
        )
        assert result.status == g8_d.CODEC_INFEASIBILITY
        assert result.emitted_codestream is None and result.emitted_identity is None
    elif case == "reconstruction_cache_corruption":
        reconstruction = fixture["reconstruction"]
        path = tmp_path / "reconstruction" / "reconstruction" / f"{reconstruction.identity.identity_id}.json"
        data = json.loads(path.read_bytes())
        data["decoded_pixels_sha256"] = "0" * 64
        path.write_bytes(g8_d.rendered_json(data))
        with pytest.raises(g8_d.G8DContractError):
            fixture["reconstruction_cache"].get_or_create(
                image_identity=context["image"],
                emitted_file_identity=fixture["search"].emitted_identity,
                codestream=fixture["search"].emitted_codestream,
                output_shape=context["image"].canonical_shape,
            )
    elif case == "br11_accounting_mutation":
        row = g8_d.account_br11(
            fixture["search"].emitted_codestream,
            emitted_file_identity=fixture["search"].emitted_identity,
            bytes_sent=context["budget"].bytes_sent,
            verdict="delivered",
        )
        payload = row.as_dict()
        payload["payload_bytes"] += 1
        with pytest.raises(g8_d.G8DContractError):
            g8_d.BR11Accounting.from_mapping(payload)
    elif case == "incorrect_accuracy_counts":
        payload = record.as_dict()
        payload["correct_count"] = 1
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping(payload)
    elif case == "assumed_bare_accuracy":
        with pytest.raises(g8_d.G8DContractError):
            g8_d.CleanClassifierMeasurementRecord.from_mapping({"accuracy": 0.5})
    elif case == "duplicate_work_unit":
        duplicate = g8_d.WorkUnitIdentity(contract["campaign_id"], 1, context["candidate"].identity_id)
        with pytest.raises(g8_d.G8DContractError, match="duplicate"):
            g8_d.AtomicMeasurementCampaign(
                tmp_path / "duplicate",
                contract_id=contract["contract_id"],
                campaign_id=contract["campaign_id"],
                work_units=[fixture["work_unit"], duplicate],
                record_factory=lambda _work_unit: record,
            )
    elif case == "missing_work_unit":
        missing = g8_d.WorkUnitIdentity(contract["campaign_id"], 2, "g8dcandidate-" + "d" * 64)
        with pytest.raises(g8_d.G8DContractError, match="contiguous"):
            g8_d.AtomicMeasurementCampaign(
                tmp_path / "missing",
                contract_id=contract["contract_id"],
                campaign_id=contract["campaign_id"],
                work_units=[fixture["work_unit"], missing],
                record_factory=lambda _work_unit: record,
            )
    elif case == "stale_aggregate":
        candidate2 = replace(context["candidate"], snr_db=1.0)
        work2 = g8_d.WorkUnitIdentity(contract["campaign_id"], 1, candidate2.identity_id)
        record2 = g8_d.CleanClassifierMeasurementRecord.from_outcomes(
            work_unit=work2,
            candidate=candidate2,
            image=context["image"],
            validation_split=context["split"],
            classifier=context["classifier"],
            g8_c_table=context["table"],
            reconstruction=fixture["reconstruction"].identity,
            reconstruction_cache_object_id="g8dreconobj-" + "c" * 64,
            outcomes=[True],
            source="d6-stale-aggregate",
        )
        campaign = g8_d.AtomicMeasurementCampaign(
            tmp_path / "stale",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[fixture["work_unit"], work2],
            record_factory=lambda work_unit: record if work_unit.ordinal == 0 else record2,
        )
        campaign.run_next()
        state = campaign.read_state()
        second_raw = g8_d.rendered_json(record2.as_dict())
        ahead = campaign._aggregate_payload(
            [fixture["work_unit"], work2],
            [record, record2],
            [state["record_refs"][0], campaign._record_ref(work2, record2, second_raw)],
        )
        aggregate_path = campaign.aggregates_dir / f"{ahead['aggregate_id']}.json"
        aggregate_path.write_bytes(g8_d.rendered_json(ahead))
        with pytest.raises(g8_d.G8DContractError, match="stale aggregate"):
            campaign.read_state()
    elif case == "interrupted_publication":
        campaign = g8_d.AtomicMeasurementCampaign(
            tmp_path / "interrupted",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[fixture["work_unit"]],
            record_factory=lambda _work_unit: record,
            hook=_CrashBeforeRecord(),
        )
        campaign.initialize()
        with pytest.raises(RuntimeError, match="interrupted"):
            campaign.run_next()
        resumed = g8_d.AtomicMeasurementCampaign(
            tmp_path / "interrupted",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[fixture["work_unit"]],
            record_factory=lambda _work_unit: record,
        )
        resumed.run_all()
        assert resumed.read_state()["completed_work_unit_ids"] == [fixture["work_unit"].identity_id]
    elif case == "changed_source_or_config_on_resume":
        with pytest.raises(g8_d.G8DContractError, match="current authenticated contract"):
            g8_d.AtomicMeasurementCampaign(
                tmp_path / "changed",
                contract_id="g8dcontract-" + "f" * 64,
                campaign_id=contract["campaign_id"],
                work_units=[fixture["work_unit"]],
                record_factory=lambda _work_unit: record,
            )
    else:
        raise AssertionError(f"unhandled mutation case {case}")


def test_bounded_smoke_artifact_and_independent_verifier_pass(
    tmp_path: Path, post_g10_am94
) -> None:
    del post_g10_am94
    output = tmp_path / "bounded_smoke.json"
    artifact = run_smoke(output)
    verified = verify(output)
    assert artifact["artifact_id"] == verified["artifact_id"]
    assert verified["mutation_case_names"] == list(MUTATION_CASES)
