from __future__ import annotations

import numpy as np
import pytest

from baseline import g8_d


def _fixture() -> dict[str, object]:
    contract = g8_d.build_g8_d_contract()
    split = g8_d.ValidationSplitIdentity.from_mapping(
        next(item for item in contract["validation_split_bindings"] if item["dataset"] == "imagenette160")
    )
    classifier = g8_d.ClassifierIdentity.from_mapping(contract["classifier_binding"])
    table = g8_d.G8CTableIdentity.from_mapping(contract["g8_c_binding"])
    codec = g8_d.CodecConfigurationIdentity.from_mapping(contract["codec_binding"])
    image = g8_d.ImageIdentity.from_pixels(
        split_identity=split,
        stable_sample_id="fixture-image",
        source_bytes=b"fixture-source-bytes",
        canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    budget = g8_d.BudgetIdentity(
        bw_ratio="fixture",
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
        snr_db=0.0,
        encode_axis_px=8,
    )
    work_unit = g8_d.WorkUnitIdentity(contract["campaign_id"], 0, candidate.identity_id)
    reconstruction = g8_d.ReconstructionIdentity(
        image.identity_id,
        "g8demitted-" + "a" * 64,
        codec.identity_id,
        (8, 8, 3),
        "bicubic",
        True,
    )
    return {
        "work_unit": work_unit,
        "candidate": candidate,
        "image": image,
        "validation_split": split,
        "classifier": classifier,
        "g8_c_table": table,
        "reconstruction": reconstruction,
        "reconstruction_cache_object_id": "g8dreconobj-" + "b" * 64,
    }


def _record() -> g8_d.CleanClassifierMeasurementRecord:
    return g8_d.CleanClassifierMeasurementRecord.from_outcomes(
        **_fixture(),
        outcomes=[True, False, True, False, True],
        source="d4-bounded-fixture",
    )


def test_clean_classifier_record_derives_accuracy_from_counts() -> None:
    record = _record()
    assert (record.correct_count, record.total_count) == (3, 5)
    assert record.accuracy == 3 / 5
    measured = record.measured_accuracy()
    assert isinstance(measured, g8_d.composition.MeasuredCodecAccuracy)
    assert (measured.correct, measured.total, measured.split) == (3, 5, "val")
    assert record.as_dict()["accuracy_derivation"] == "correct_count / total_count"
    assert record.as_dict()["validation_only"] is True
    assert record.as_dict()["test_access"] == 0
    assert record.as_dict()["training"] is False
    assert record.as_dict()["merge_eligible"] is False


def test_clean_classifier_record_round_trips_and_rejects_accuracy_mutations() -> None:
    record = _record()
    restored = g8_d.CleanClassifierMeasurementRecord.from_mapping(record.as_dict())
    assert restored.record_id == record.record_id
    assert restored.measured_accuracy().value == 3 / 5

    wrong_accuracy = record.as_dict()
    wrong_accuracy["accuracy"] = 0.99
    with pytest.raises(g8_d.G8DContractError, match="accuracy differs"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_accuracy)

    wrong_count = record.as_dict()
    wrong_count["correct_count"] = 4
    with pytest.raises(g8_d.G8DContractError, match="record ID differs"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_count)

    with pytest.raises(TypeError):
        g8_d.CleanClassifierMeasurementRecord(**_fixture(), outcomes=0.6, source="bad")  # type: ignore[call-arg]


def test_clean_classifier_record_rejects_non_validation_or_changed_provenance() -> None:
    record = _record()
    wrong_split = record.as_dict()
    wrong_split["validation_split"]["split"] = "test"
    with pytest.raises(g8_d.G8DContractError):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_split)

    wrong_classifier = record.as_dict()
    wrong_classifier["classifier"]["checkpoint_sha256"] = "c" * 64
    with pytest.raises(g8_d.G8DContractError, match="classifier"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_classifier)

    wrong_table = record.as_dict()
    wrong_table["g8_c_table"]["table_id"] = "predecessor"
    with pytest.raises(g8_d.G8DContractError, match="table"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_table)

    wrong_cache = record.as_dict()
    wrong_cache["reconstruction_cache_object_id"] = "g8dreconobj-" + "c" * 64
    with pytest.raises(g8_d.G8DContractError, match="record ID differs"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_cache)


def test_clean_classifier_record_rejects_identity_aliases_and_schema_mutation() -> None:
    record = _record()
    wrong_image = record.as_dict()
    wrong_image["image"]["source_bytes_sha256"] = "d" * 64
    with pytest.raises(g8_d.G8DContractError, match="image"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_image)

    wrong_reconstruction = record.as_dict()
    wrong_reconstruction["reconstruction"]["output_shape"] = [4, 4, 3]
    with pytest.raises(g8_d.G8DContractError, match="reconstruction"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(wrong_reconstruction)

    extra = record.as_dict()
    extra["unexpected"] = True
    with pytest.raises(g8_d.G8DContractError, match="schema differs"):
        g8_d.CleanClassifierMeasurementRecord.from_mapping(extra)


def test_outcomes_must_be_nonempty_bools_and_counts_must_be_valid() -> None:
    fixture = _fixture()
    with pytest.raises(g8_d.G8DContractError, match="empty"):
        g8_d.CleanClassifierMeasurementRecord.from_outcomes(**fixture, outcomes=[], source="fixture")
    with pytest.raises(g8_d.G8DContractError, match="booleans"):
        g8_d.CleanClassifierMeasurementRecord.from_outcomes(**fixture, outcomes=[True, 1], source="fixture")

    with pytest.raises(g8_d.G8DContractError, match="exceeds"):
        g8_d.CleanClassifierMeasurementRecord(
            **fixture,
            correct_count=2,
            total_count=1,
            source="fixture",
        )
