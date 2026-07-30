"""Mutation coverage for every fail-closed G-2 adjudication class."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from config.params import REPO_ROOT
from verify_g2_adjudication import (
    REQUIRED_FILES,
    SOURCE_MANIFEST,
    VerificationError,
    verify,
)

SOURCE = REPO_ROOT / "results/baseline/g2"
pytestmark = pytest.mark.skipif(
    not (SOURCE / "g2_adjudication.json").is_file(),
    reason="G-2 evidence has not been generated",
)


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    target = tmp_path / "g2"
    target.mkdir()
    for name in REQUIRED_FILES | {SOURCE_MANIFEST}:
        shutil.copyfile(SOURCE / name, target / name)
    return target


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _refresh(evidence: Path, name: str) -> None:
    adjudication = _json(evidence / "g2_adjudication.json")
    adjudication["evidence_files"][name] = hashlib.sha256(
        (evidence / name).read_bytes()
    ).hexdigest()
    _write(evidence / "g2_adjudication.json", adjudication)


def _mutate_json(evidence: Path, name: str, mutation, *, refresh: bool = True) -> None:
    value = _json(evidence / name)
    mutation(value)
    _write(evidence / name, value)
    if refresh and name != "g2_adjudication.json":
        _refresh(evidence, name)


def _mutate_csv(evidence: Path, mutation) -> None:
    path = evidence / "bler_results.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    mutation(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh(evidence, path.name)


def test_committed_g2_evidence_verifies():
    assert verify()["verdict"] == "PASS"


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("golden_vector_provenance.json", lambda v: v.update(asset_sha256="0" * 64)),
        ("golden_vector_provenance.json", lambda v: v.update(source_rung=3)),
        ("golden_vector_provenance.json", lambda v: v.update(alignment="drop_2Z_twice")),
        ("resolved_config.json", lambda v: v.update(sionna_version="0.0.0")),
        ("resolved_config.json", lambda v: v.update(standard_version="wrong")),
        ("known_answer_summary.json", lambda v: v["modulation"]["qam16"].update(labels_recovered=False)),
        ("known_answer_summary.json", lambda v: v["modulation"]["qam16"].update(disabled_interleaver_detected=False)),
        ("known_answer_summary.json", lambda v: v["modulation"]["qpsk"].update(sign_flip_detected=False)),
        ("known_answer_summary.json", lambda v: v["crc"]["crc24b"].update({"pass": False})),
        ("bler_reference.json", lambda v: v["settings"].update(base_graph=1)),
        ("bler_reference.json", lambda v: v["settings"].update(lifting_size=24)),
        ("bler_reference.json", lambda v: v.update(decoder_offset=0.0)),
        ("bler_reference.json", lambda v: v.update(iterations=49)),
        ("packetisation_runtime_check.json", lambda v: v["mismatches"].append("x")),
        (
            "packetisation_runtime_check.json",
            lambda v: v["expected_structural_infeasibility"].append(
                {"tag": "unexpected", "reason": "coerced"}
            ),
        ),
        ("resolved_config.json", lambda v: v["test_split_access"].update(inference_calls=1)),
    ],
)
def test_json_failure_classes(evidence: Path, name: str, mutation):
    _mutate_json(evidence, name, mutation)
    with pytest.raises(VerificationError):
        verify(evidence, require_evidence_commit=False)


def test_unreachable_measurement_commit_fails(evidence: Path):
    _mutate_json(
        evidence, "g2_adjudication.json",
        lambda value: value.update(measurement_commit="f" * 40),
        refresh=False,
    )
    with pytest.raises(VerificationError, match="git object"):
        verify(evidence, require_evidence_commit=False)


def test_dirty_measurement_commit_fails(evidence: Path):
    _mutate_json(
        evidence, "g2_adjudication.json",
        lambda value: value.update(measurement_dirty=True),
        refresh=False,
    )
    with pytest.raises(VerificationError, match="dirty"):
        verify(evidence, require_evidence_commit=False)


def test_wrong_snr_conversion_fails(evidence: Path):
    _mutate_csv(evidence, lambda rows: rows[0].update(esn0_db="123"))
    with pytest.raises(VerificationError, match="SNR conversion"):
        verify(evidence, require_evidence_commit=False)


def test_missing_modulation_and_insufficient_cells_fail(evidence: Path):
    _mutate_csv(evidence, lambda rows: rows.__setitem__(slice(None), [
        row for row in rows if row["modulation"] != "qam16"
    ]))
    with pytest.raises(VerificationError, match="missing modulation"):
        verify(evidence, require_evidence_commit=False)


def test_insufficient_blocks_fail(evidence: Path):
    _mutate_csv(evidence, lambda rows: rows[0].update(blocks="1"))
    with pytest.raises(VerificationError, match="insufficient simulation blocks"):
        verify(evidence, require_evidence_commit=False)


def test_displacement_above_tolerance_fails(evidence: Path):
    _mutate_json(
        evidence, "g2_adjudication.json",
        lambda value: value["waterfalls"]["bpsk"].update(displacement_db=1.0),
        refresh=False,
    )
    with pytest.raises(VerificationError, match="displacement"):
        verify(evidence, require_evidence_commit=False)
