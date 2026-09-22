"""Focused JPEG persistence, carrier and representation-boundary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import evaluation.w10_jpeg_carrier as carrier
import evaluation.w10_selections as selections
from runtime.source_epochs import load_w10_manifest, source_record, successor_path
from training.deterministic_core import canonical_bytes, canonical_sha256

REPO = Path(__file__).resolve().parents[1]


def _synthetic_selection() -> bytes:
    body = {
        "schema_version": 2,
        "artifact_role": "W10_JPEG_SECONDARY_VALIDATION_SELECTION",
        "status": "FROZEN_VALIDATION_ONLY_SELECTION",
        "contract_sha256": "c" * 64,
        "source_epoch": {"path": "v5.json", "manifest_id": "v5", "sha256": "d" * 64},
        "test": "SEALED",
        "test_access": 0,
        "selections": [{"snr_db": index} for index in range(21)],
        "candidate_scores": [
            {"snr_db": index, "candidates": [{"candidate": candidate} for candidate in range(912)]}
            for index in range(21)
        ],
    }
    value = dict(body)
    value["selection_id"] = carrier.JPEG_SELECTION_PREFIX + canonical_sha256(body)
    return canonical_bytes(value)


def _patch_synthetic_constants(monkeypatch, raw: bytes) -> None:
    value = json.loads(raw)
    monkeypatch.setattr(carrier, "JPEG_SELECTION_RAW_BYTES", len(raw))
    monkeypatch.setattr(carrier, "JPEG_SELECTION_RAW_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(carrier, "JPEG_SELECTION_ID", value["selection_id"])
    monkeypatch.setattr(carrier, "JPEG_SELECTION_CONTRACT_SHA256", value["contract_sha256"])
    monkeypatch.setattr(carrier, "JPEG_SELECTION_SOURCE_RECORD", value["source_epoch"])


def test_carrier_round_trip_supports_a_clean_clone_without_raw(tmp_path: Path, monkeypatch) -> None:
    raw = _synthetic_selection()
    _patch_synthetic_constants(monkeypatch, raw)
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    built = carrier.build_jpeg_carrier(tmp_path, raw_path=source)
    assert built.raw_bytes == raw
    assert built.provenance["raw_bytes"] == len(raw)
    assert built.provenance["raw_sha256"] == hashlib.sha256(raw).hexdigest()

    (tmp_path / carrier.JPEG_SELECTION_PATH).unlink(missing_ok=True)
    loaded = carrier.check_jpeg_carrier(tmp_path)
    assert loaded.raw_bytes == raw
    assert loaded.value["selection_id"] == json.loads(raw)["selection_id"]


def test_corrupted_carrier_is_rejected(tmp_path: Path, monkeypatch) -> None:
    raw = _synthetic_selection()
    _patch_synthetic_constants(monkeypatch, raw)
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    carrier.build_jpeg_carrier(tmp_path, raw_path=source)
    path = tmp_path / carrier.JPEG_CARRIER_PATH
    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(carrier.JpegCarrierHold):
        carrier.check_jpeg_carrier(tmp_path)


def test_corrupted_descriptor_is_rejected(tmp_path: Path, monkeypatch) -> None:
    raw = _synthetic_selection()
    _patch_synthetic_constants(monkeypatch, raw)
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    carrier.build_jpeg_carrier(tmp_path, raw_path=source)
    path = tmp_path / carrier.JPEG_CARRIER_DESCRIPTOR_PATH
    descriptor = json.loads(path.read_bytes())
    descriptor["carrier_bytes"] += 1
    path.write_bytes(canonical_bytes(descriptor))
    with pytest.raises(carrier.JpegCarrierHold):
        carrier.check_jpeg_carrier(tmp_path)


def test_raw_and_carrier_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    raw = _synthetic_selection()
    _patch_synthetic_constants(monkeypatch, raw)
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    carrier.build_jpeg_carrier(tmp_path, raw_path=source)
    logical = tmp_path / carrier.JPEG_SELECTION_PATH
    logical.parent.mkdir(parents=True, exist_ok=True)
    changed = bytearray(raw)
    changed[-2] ^= 1
    logical.write_bytes(changed)
    with pytest.raises(carrier.JpegCarrierHold):
        carrier.check_jpeg_carrier(tmp_path)


def test_persisted_packet_metadata_normalizes_tuple_and_rejects_changed_er(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    historical = load_w10_manifest(REPO, live=False, epoch="v5")
    historical_path = successor_path(REPO, epoch="v5")
    historical_record = source_record(REPO, historical, path=historical_path)
    # The live source closure intentionally holds until successor-v6 is frozen;
    # this test targets the real packet/evidence persistence boundary itself.
    monkeypatch.setattr(selections, "load_w10_manifest", lambda root, live=True, epoch=None: historical)
    monkeypatch.setattr(selections, "source_record", lambda root, manifest, path=None: historical_record)
    entry = {
        "snr_db": -8,
        "modulation": "bpsk",
        "ldpc_rate": "1/3",
        "encode_axis_px": 64,
        "quality": 5,
        "n_correct": 678,
        "n_delivered": 1000,
        "n_decode_failure": 0,
        "n_infeasible": 0,
        "n_total": 1000,
        "per_image_correct_digest": selections.score_vector_digest([True] * 678 + [False] * 322),
        "status": "eligible",
    }
    clean = {
        "correct": 678,
        "total": 1000,
        "feasible_count": 1000,
        "decode_failure_count": 0,
        "infeasible_count": 0,
        "outage_correct_count": 0,
        "correct_digest": entry["per_image_correct_digest"],
        "emitted_bytes_digest": "a" * 64,
        "source": "worker_clean_jpeg_validation_codec_measurement",
        "split": "val",
        "scorer_binding": selections.scorer_binding(REPO),
    }
    analytic = selections.jpeg_analytic_evidence(REPO, entry, clean=clean)
    assert analytic["packet"]["packet_metadata"]["e_r"] == [12800]
    assert json.loads(json.dumps(analytic, sort_keys=True)) == analytic
    entry["analytic_evidence"] = analytic
    entry["expected_accuracy"] = analytic["composition"]["expected_accuracy"]
    entry["success_probability"] = analytic["composition"]["success_probability"]
    entry["candidate_evidence"] = selections.bind_candidate_evidence(
        {
            "schema_version": 1,
            "method": "jpeg_clean_codec_measurement",
            "snr_db": entry["snr_db"],
            "modulation": entry["modulation"],
            "ldpc_rate": entry["ldpc_rate"],
            "encode_axis_px": entry["encode_axis_px"],
            "quality": entry["quality"],
            "clean": clean,
            "selection_contract_sha256": selections._contract_sha256(
                "jpeg_secondary",
                source_manifest=historical,
                source_root=REPO,
            ),
            "scorer_binding": clean["scorer_binding"],
            "source_epoch": historical_record,
        }
    )
    persisted = json.loads(json.dumps(entry, sort_keys=True))
    assert persisted["analytic_evidence"]["packet"]["packet_metadata"]["e_r"] == [12800]
    selections._verify_jpeg_analytic_candidate(REPO, persisted)

    changed = json.loads(json.dumps(persisted, sort_keys=True))
    changed["analytic_evidence"]["packet"]["packet_metadata"]["e_r"] = [12801]
    with pytest.raises(selections.W10SelectionHold):
        selections._verify_jpeg_analytic_candidate(REPO, changed)


@pytest.mark.skipif(
    not (REPO / carrier.JPEG_CARRIER_PATH).is_file(),
    reason="tracked carrier is added in the publication commit",
)
def test_published_repository_carrier_keeps_historical_jpeg_identity() -> None:
    loaded = carrier.check_jpeg_carrier(REPO)
    assert loaded.provenance["raw_bytes"] == carrier.JPEG_SELECTION_RAW_BYTES
    assert loaded.provenance["raw_sha256"] == carrier.JPEG_SELECTION_RAW_SHA256
    assert loaded.provenance["selection_id"] == carrier.JPEG_SELECTION_ID
    assert loaded.value["source_epoch"] == carrier.JPEG_SELECTION_SOURCE_RECORD
