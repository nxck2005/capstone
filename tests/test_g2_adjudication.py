"""Mutation coverage for every fail-closed G-2 adjudication class."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import gen_g2_source_manifest as gen
import verify_g2_adjudication as verifier
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


# --- execution-source binding ------------------------------------------------------
#
# Binding the evidence files, the measurement commit and the ancestry
# measurement -> evidence says nothing about the bytes of the implementation that
# produced the measurement. These cases are the ones that make
# execution_source_manifest.json load-bearing rather than decorative: each mutation
# below leaves the whole recorded campaign internally consistent and still must fail.
#
# `evidence_commit=False` in most cases because the evidence commit is resolved
# through Git path history against the real repository, which a tmp_path copy has
# none of. The two cases that DO exercise that resolution patch `git` directly.

RUNTIME_FILE = "src/baseline/ldpc/adapter.py"
# Read, not restated, so a re-adjudication cannot leave these tests asserting
# against a commit the evidence no longer names.
MEASUREMENT_COMMIT = _json(SOURCE / "g2_adjudication.json")["measurement_commit"]
# A real commit that is NOT a descendant of the measurement commit: the
# transparency-probe implementation, which predates W3. Using a real commit rather
# than a synthetic one means the ancestry check is what fails, not object lookup.
NON_DESCENDANT = "90007f165f8f669a54127bdd6539472cb2d3f534"


def _mutate_manifest(evidence: Path, mutation) -> None:
    _mutate_json(evidence, SOURCE_MANIFEST, mutation, refresh=False)


def _entry(manifest: dict, path: str) -> dict:
    return next(item for item in manifest["sources"] if item["path"] == path)


def test_manifest_binding_a_different_measurement_commit_fails(evidence: Path):
    """The manifest must bind the commit the adjudication names, not another one."""
    _mutate_manifest(evidence, lambda v: v.update(measurement_commit=NON_DESCENDANT))
    with pytest.raises(VerificationError, match="binds a different measurement commit"):
        verify(evidence, require_evidence_commit=False)


def test_unreachable_measurement_commit_in_manifest_fails(evidence: Path):
    """An unreachable commit must fail even when every file agrees on it."""
    _mutate_json(evidence, "g2_adjudication.json",
                 lambda v: v.update(measurement_commit="f" * 40), refresh=False)
    _mutate_json(evidence, "resolved_config.json",
                 lambda v: v.update(measurement_commit="f" * 40))
    _mutate_manifest(evidence, lambda v: v.update(measurement_commit="f" * 40))
    with pytest.raises(VerificationError, match="git object"):
        verify(evidence, require_evidence_commit=False)


def test_unreachable_evidence_commit_fails(monkeypatch: pytest.MonkeyPatch):
    """Git path history returning nothing means the evidence is unpublished.

    This is the failure mode the resolution policy has to handle explicitly: there
    is no recorded `evidence_commit` to fall back on, by design, so an empty
    resolution must fail closed rather than skip the ancestry check.
    """
    real = verifier.git

    def resolved(*args: str) -> str:
        return "" if args[:2] == ("log", "-1") else real(*args)

    monkeypatch.setattr(verifier, "git", resolved)
    with pytest.raises(VerificationError, match="evidence commit is unreachable"):
        verify()


def test_wrong_commit_ancestry_fails(monkeypatch: pytest.MonkeyPatch):
    """Evidence that does not descend from the measurement proves nothing about it."""
    real = verifier.git

    def resolved(*args: str) -> str:
        return NON_DESCENDANT if args[:2] == ("log", "-1") else real(*args)

    monkeypatch.setattr(verifier, "git", resolved)
    with pytest.raises(VerificationError, match="ancestry"):
        verify()


def test_missing_source_fails(evidence: Path):
    _mutate_manifest(evidence, lambda v: v.__setitem__("sources", [
        item for item in v["sources"] if item["path"] != RUNTIME_FILE
    ]))
    with pytest.raises(VerificationError, match="missing sources"):
        verify(evidence, require_evidence_commit=False)


def test_unexpected_source_fails(evidence: Path):
    """A manifest may not quietly widen its own scope.

    Without this, a re-generated manifest could add an unrelated file and claim
    broader provenance than the expected set actually asserts.
    """
    def add(value: dict) -> None:
        extra = dict(_entry(value, RUNTIME_FILE))
        extra["path"] = "src/baseline/ldpc/not_adjudicated.py"
        value["sources"].append(extra)

    _mutate_manifest(evidence, add)
    with pytest.raises(VerificationError, match="unexpected sources"):
        verify(evidence, require_evidence_commit=False)


def test_wrong_blob_sha_fails(evidence: Path):
    _mutate_manifest(
        evidence,
        lambda v: _entry(v, RUNTIME_FILE).update(measurement_blob="0" * 40),
    )
    with pytest.raises(VerificationError, match="Git blob does not match"):
        verify(evidence, require_evidence_commit=False)


@pytest.mark.parametrize(
    "path", [RUNTIME_FILE, "spec/evidence/check_packetisation.py"],
    ids=["runtime", "record"],
)
def test_wrong_byte_sha_fails(evidence: Path, path: str):
    """Both the asserted-current runtime and the history-only roles are bound.

    The byte hash is checked for every role, not just `runtime`. A
    `measurement_runner` or `record` entry that is free to say anything would make
    the historical half of the manifest unfalsifiable.
    """
    _mutate_manifest(evidence, lambda v: _entry(v, path).update(
        measurement_sha256="0" * 64))
    with pytest.raises(VerificationError, match="byte SHA-256 does not match"):
        verify(evidence, require_evidence_commit=False)


def test_wrong_role_fails(evidence: Path):
    """Relabelling `runtime` as history would silently drop the current-bytes rule."""
    _mutate_manifest(evidence, lambda v: _entry(v, RUNTIME_FILE).update(
        role="configuration"))
    with pytest.raises(VerificationError, match="is not 'runtime'"):
        verify(evidence, require_evidence_commit=False)


def test_modified_current_ldpc_runtime_fails(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
):
    """The gap this whole manifest exists to close.

    Every recorded number still verifies; only the implementation changed. Before
    the manifest, this passed.
    """
    real = verifier.current_bytes

    def drifted(source_path: str) -> bytes:
        value = real(source_path)
        return value + b"\n" if source_path == RUNTIME_FILE else value

    monkeypatch.setattr(verifier, "current_bytes", drifted)
    with pytest.raises(VerificationError, match="HOLD — G-2 runtime differs"):
        verify(evidence, require_evidence_commit=False)


def _drift(monkeypatch: pytest.MonkeyPatch, path: str = RUNTIME_FILE):
    """Make one runtime file read as changed, and return its drifted bytes."""

    real = verifier.current_bytes
    drifted_bytes = real(path) + b"\n"

    def drifted(source_path: str) -> bytes:
        return drifted_bytes if source_path == path else real(source_path)

    monkeypatch.setattr(verifier, "current_bytes", drifted)
    return drifted_bytes


def _readjudication(manifest: dict, path: str, current: bytes, **overrides) -> dict:
    entry = {
        "path": path,
        "kind": "off_measurement_path",
        "readjudicated_at": "synthetic, test only",
        "measurement_sha256": _entry(manifest, path)["measurement_sha256"],
        "current_sha256": hashlib.sha256(current).hexdigest(),
        "justification": "synthetic re-adjudication, test only",
        "evidence": ["synthetic evidence, test only"],
    }
    entry.update(overrides)
    return {key: value for key, value in entry.items() if value is not _ABSENT}


_ABSENT = object()


def test_recorded_readjudication_permits_runtime_drift(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
):
    """The escape hatch works, and only for the path and bytes it names.

    Without this case the rule above could be satisfied by a verifier that ignores
    `readjudications` entirely, and a legitimate re-adjudication would have no way
    to land.
    """
    drifted = _drift(monkeypatch)
    _mutate_manifest(evidence, lambda v: v["readjudications"].append(
        _readjudication(v, RUNTIME_FILE, drifted)))
    assert verify(evidence, require_evidence_commit=False)["verdict"] == "PASS"


def test_readjudication_of_another_path_does_not_excuse_drift(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
):
    _drift(monkeypatch)
    other = "src/baseline/ldpc/crc.py"
    _mutate_manifest(evidence, lambda v: v["readjudications"].append(
        _readjudication(v, other, verifier.current_bytes(other))))
    with pytest.raises(VerificationError, match="HOLD — G-2 runtime differs"):
        verify(evidence, require_evidence_commit=False)


def test_readjudication_stops_covering_a_file_that_changed_again(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
):
    """The escape hatch is pinned to bytes, so it cannot be inherited.

    A re-adjudication says "these current bytes are justified". If the file then
    changes once more, the justification no longer describes it and the HOLD must
    come back rather than being carried by the stale entry.
    """
    drifted = _drift(monkeypatch)
    _mutate_manifest(evidence, lambda v: v["readjudications"].append(
        _readjudication(v, RUNTIME_FILE, drifted + b"# changed again\n")))
    with pytest.raises(VerificationError, match="does not cover the current bytes"):
        verify(evidence, require_evidence_commit=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": _ABSENT}, "readjudication kind must be one of"),
        ({"kind": "because I said so"}, "readjudication kind must be one of"),
        ({"justification": _ABSENT}, "readjudication has no justification"),
        ({"justification": "   "}, "readjudication has no justification"),
        ({"readjudicated_at": _ABSENT}, "readjudication has no readjudicated_at"),
        ({"evidence": []}, "readjudication records no evidence"),
        ({"evidence": _ABSENT}, "readjudication records no evidence"),
        ({"measurement_sha256": "0" * 64}, "not the adjudicated measurement bytes"),
    ],
)
def test_an_unjustified_readjudication_is_rejected(
    evidence: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict, message: str
):
    drifted = _drift(monkeypatch)
    _mutate_manifest(evidence, lambda v: v["readjudications"].append(
        _readjudication(v, RUNTIME_FILE, drifted, **overrides)))
    with pytest.raises(VerificationError, match=message):
        verify(evidence, require_evidence_commit=False)


def test_a_readjudication_may_not_name_a_history_only_source(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only `runtime` is asserted at HEAD, so nothing else can be re-adjudicated."""

    drifted = _drift(monkeypatch)
    _mutate_manifest(evidence, lambda v: v["readjudications"].extend([
        _readjudication(v, RUNTIME_FILE, drifted),
        _readjudication(v, "spec/params.generated.yaml", b"", current_sha256="0" * 64),
    ]))
    with pytest.raises(VerificationError, match="names a non-runtime source"):
        verify(evidence, require_evidence_commit=False)


def test_the_committed_transport_readjudication_is_recorded_and_reported():
    """The real B1.1 re-adjudication, not a synthetic one."""

    result = verify()
    assert result["verdict"] == "PASS"
    assert result["runtime_readjudicated"] == ["src/baseline/ldpc/transport.py"]
    entry, = _json(SOURCE / SOURCE_MANIFEST)["readjudications"]
    assert entry["kind"] == "off_measurement_path"
    assert entry["current_sha256"] == hashlib.sha256(
        (REPO_ROOT / entry["path"]).read_bytes()
    ).hexdigest()
    assert len(entry["evidence"]) >= 3
    assert "build_packet_plan" in " ".join(entry["evidence"])


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("resolved_config.json", "config_sha256"),
        ("resolved_config.json", "params_sha256"),
        ("packetisation_runtime_check.json", "solver_record_sha256"),
    ],
)
def test_measurement_time_cross_check_disagreement_fails(
    evidence: Path, name: str, field: str
):
    """The campaign's own hashes must agree with the manifest.

    These three hashes were written by the campaign before the manifest existed, so
    they are what distinguishes "these were the bytes at some commit" from "these
    were the bytes the measurement loaded". Mutating the campaign side rather than
    the manifest side is deliberate: it isolates the corroboration check from the
    per-file hash checks above, which would otherwise fire first.
    """
    _mutate_json(evidence, name, lambda v: v.update({field: "0" * 64}))
    with pytest.raises(VerificationError, match=f"disagree with the {field}"):
        verify(evidence, require_evidence_commit=False)


def test_recorded_evidence_commit_is_rejected(evidence: Path):
    """The resolution policy is asserted, not merely documented.

    `g2_adjudication.json` used to carry `"evidence_commit": null`, which reads as a
    missing value rather than a deliberate one. Any recorded value here is either
    self-referential or wrong, so the field must not come back.
    """
    _mutate_json(evidence, "g2_adjudication.json",
                 lambda v: v.update(evidence_commit=MEASUREMENT_COMMIT), refresh=False)
    with pytest.raises(VerificationError, match="records an evidence_commit"):
        verify(evidence, require_evidence_commit=False)


def test_source_manifest_is_regenerable():
    """The committed manifest is exactly what the generator produces from Git."""
    manifest = gen.build(MEASUREMENT_COMMIT)
    committed = json.loads((SOURCE / SOURCE_MANIFEST).read_text())
    assert manifest == committed
