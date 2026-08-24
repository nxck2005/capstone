"""F0/F1 readiness tests using synthetic bytes and a non-scientific backend only."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from baseline.g8_f_materializer import (
    CODEC_CONFIGURATION_HASH,
    F1Assignment,
    F1Materializer,
    G8FMaterializationHold,
    canonical_json,
    load_frozen_assignments,
    rendered_json,
    sha256_bytes,
    validate_exact_result_prefix,
)
from data.adapters import SourceSample
from data.identity import stable_sample_id


@dataclass
class _Result:
    feasible: bool
    codestream: bytes | None
    emitted_byte_count: int | None
    codec_configuration_hash: str
    decoded_image: np.ndarray | None
    decode_success: bool
    codestream_sha256: str | None
    cache_key: str = "synthetic-cache-key"


class _Backend:
    configuration_hash = CODEC_CONFIGURATION_HASH

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.calls = 0

    def encode_to_budget(self, image: np.ndarray, **_kwargs: object) -> _Result:
        self.calls += 1
        if self.outcome == "raise":
            raise RuntimeError("synthetic unexpected failure")
        if self.outcome == "infeasible":
            return _Result(False, None, None, CODEC_CONFIGURATION_HASH, None, False, None)
        codestream = b"synthetic-not-real-j2k"
        return _Result(
            True,
            codestream,
            len(codestream),
            CODEC_CONFIGURATION_HASH,
            image.copy(),
            True,
            hashlib.sha256(codestream).hexdigest(),
        )


def _fixture() -> tuple[F1Assignment, SourceSample]:
    image = Image.fromarray(np.full((160, 160, 3), 73, dtype=np.uint8), mode="RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    stable_id = stable_sample_id(raw)
    return (
        F1Assignment(0, stable_id, 3, "g8fquality-synthetic", 128, 160, "g8dcodec-39f14b7eaba4f727c70759eb1c5250e8e13f7d5e871c0831aa6b602aef706858"),
        SourceSample("imagenette160", stable_id, 3, raw),
    )


def _materialized(tmp_path: Path, *, scientific: bool = True) -> tuple[F1Assignment, SourceSample, F1Materializer]:
    assignment, source = _fixture()
    materializer = F1Materializer(tmp_path, _Backend("success"), scientific=scientific)
    materializer.materialize(assignment, source, split="train")
    return assignment, source, materializer


def _rewrite(path: Path, mutate: object, *, identity: str | None = None) -> dict:
    value = json.loads(path.read_bytes())
    mutate(value)  # type: ignore[operator]
    if identity == "request":
        body = dict(value)
        body.pop("request_id", None)
        value["request_id"] = "g8frequest-" + sha256_bytes(canonical_json(body))
    elif identity == "result":
        body = dict(value)
        body.pop("result_id", None)
        value["result_id"] = "g8fresult-" + sha256_bytes(canonical_json(body))
    path.write_bytes(rendered_json(value))
    return value


def _prefix_holds(tmp_path: Path, assignment: F1Assignment, match: str | None = None) -> None:
    with pytest.raises(G8FMaterializationHold, match=match):
        validate_exact_result_prefix(tmp_path, (assignment,), expected_scientific=True)


def test_production_assignment_loader_is_exact_am88_not_am87_cartesian() -> None:
    assignments = load_frozen_assignments()
    assert len(assignments) == 50_814  # literal-ok: frozen AM-88 count
    assert {assignment.ordinal for assignment in assignments} == set(range(50_814))  # literal-ok: exact prefix
    assert len({(assignment.stable_sample_id, assignment.quality_id) for assignment in assignments}) == 50_814
    assert len(assignments) != 1_016_280  # literal-ok: forbidden AM-87 Cartesian multiplicity


def test_synthetic_success_is_deterministic_and_resume_reuses_exact_record(tmp_path: Path) -> None:
    assignment, source = _fixture()
    backend = _Backend("success")
    materializer = F1Materializer(tmp_path, backend, scientific=False)
    first = materializer.materialize(assignment, source, split="train")
    second = materializer.materialize(assignment, source, split="train")
    assert first == second
    assert backend.calls == 1
    assert first["scientific"] is False
    assert first["outcome"] == "materialized_verified_artifact"
    assert first["resampled"] is False and first["replacement_assignment"] is None
    assert validate_exact_result_prefix(tmp_path, (assignment,), expected_scientific=False) == 1


def test_typed_synthetic_infeasibility_records_no_objects_and_never_resamples(tmp_path: Path) -> None:
    assignment, source = _fixture()
    value = F1Materializer(tmp_path, _Backend("infeasible"), scientific=False).materialize(
        assignment, source, split="train"
    )
    assert value["outcome"] == "typed_image_codec_infeasibility"
    assert value["codestream"] is None and value["reconstruction"] is None
    assert value["replacement_assignment"] is None and value["resampled"] is False
    assert not (tmp_path / "objects").exists()


def test_unexpected_synthetic_failure_is_hold_not_omission(tmp_path: Path) -> None:
    assignment, source = _fixture()
    with pytest.raises(G8FMaterializationHold, match="unexpected codec/runtime failure"):
        F1Materializer(tmp_path, _Backend("raise"), scientific=False).materialize(
            assignment, source, split="train"
        )
    assert not (tmp_path / "results/00000.json").exists()
    assert (tmp_path / "requests/00000.json").exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_validation_and_test_are_refused_before_codec(tmp_path: Path, split: str) -> None:
    assignment, source = _fixture()
    backend = _Backend("success")
    with pytest.raises(G8FMaterializationHold, match="train split"):
        F1Materializer(tmp_path, backend, scientific=False).materialize(assignment, source, split=split)
    assert backend.calls == 0


def test_prefix_hole_is_hold(tmp_path: Path) -> None:
    assignment, source = _fixture()
    materializer = F1Materializer(tmp_path, _Backend("success"), scientific=False)
    materializer.materialize(assignment, source, split="train")
    path = tmp_path / "results/00000.json"
    path.rename(tmp_path / "results/00001.json")
    with pytest.raises(G8FMaterializationHold, match="exact ordinal prefix"):
        validate_exact_result_prefix(tmp_path, (assignment,), expected_scientific=False)


def test_legal_orphan_request_is_reused_deterministically_after_interruption(tmp_path: Path) -> None:
    assignment, source = _fixture()
    failing = F1Materializer(tmp_path, _Backend("raise"), scientific=True)
    with pytest.raises(G8FMaterializationHold, match="unexpected codec/runtime failure"):
        failing.materialize(assignment, source, split="train")
    request_before = (tmp_path / "requests/00000.json").read_bytes()
    assert validate_exact_result_prefix(tmp_path, (assignment,)) == 0
    succeeding_backend = _Backend("success")
    succeeding = F1Materializer(tmp_path, succeeding_backend, scientific=True)
    succeeding.materialize(assignment, source, split="train")
    assert (tmp_path / "requests/00000.json").read_bytes() == request_before
    assert succeeding_backend.calls == 1
    assert validate_exact_result_prefix(tmp_path, (assignment,)) == 1


def test_changed_result_body_with_correct_assignment_id_holds(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "results/00000.json", lambda value: value.__setitem__("quality_id", "foreign"), identity="result")
    _prefix_holds(tmp_path, assignment, "scientific body")


def test_wrong_result_id_holds(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "results/00000.json", lambda value: value.__setitem__("result_id", "g8fresult-wrong"))
    _prefix_holds(tmp_path, assignment, "result identity")


def test_changed_result_request_id_holds(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "results/00000.json", lambda value: value.__setitem__("request_id", "g8frequest-foreign"), identity="result")
    _prefix_holds(tmp_path, assignment, "another request")


def test_changed_request_body_holds_even_with_recomputed_request_id(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "requests/00000.json", lambda value: value["assignment"].__setitem__("label", 9), identity="request")
    _prefix_holds(tmp_path, assignment, "assignment body")


def test_wrong_request_id_holds(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "requests/00000.json", lambda value: value.__setitem__("request_id", "g8frequest-wrong"))
    _prefix_holds(tmp_path, assignment, "request identity")


def test_production_prefix_rejects_scientific_false(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path, scientific=False)
    _prefix_holds(tmp_path, assignment, "scientific flag")


@pytest.mark.parametrize("field,value", [("resampled", True), ("replacement_assignment", "g8fassign-foreign")])
def test_resampling_or_replacement_holds(tmp_path: Path, field: str, value: object) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "results/00000.json", lambda result: result.__setitem__(field, value), identity="result")
    _prefix_holds(tmp_path, assignment, "resampled or replaced")


@pytest.mark.parametrize(
    "object_name,operation,match",
    [
        ("codestream", "delete", "codestream"),
        ("codestream", "corrupt", "codestream object SHA-256"),
        ("reconstruction", "delete", "reconstruction"),
        ("reconstruction", "corrupt", "reconstruction object SHA-256"),
    ],
)
def test_missing_or_corrupt_referenced_object_holds(
    tmp_path: Path, object_name: str, operation: str, match: str
) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    result = json.loads((tmp_path / "results/00000.json").read_bytes())
    path = tmp_path / result[object_name]["path"]
    if operation == "delete":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    _prefix_holds(tmp_path, assignment, match)


@pytest.mark.parametrize("mutation", ["length", "hash", "path"])
def test_wrong_object_length_hash_or_path_holds(tmp_path: Path, mutation: str) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    def mutate(result: dict) -> None:
        if mutation == "length":
            result["codestream"]["bytes"] += 1
        elif mutation == "hash":
            result["codestream"]["sha256"] = "0" * 64
        else:
            result["codestream"]["path"] = "../foreign.j2k"
    _rewrite(tmp_path / "results/00000.json", mutate, identity="result")
    _prefix_holds(tmp_path, assignment)


def test_symlinked_object_is_never_accepted_as_completed(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    result = json.loads((tmp_path / "results/00000.json").read_bytes())
    path = tmp_path / result["codestream"]["path"]
    target = tmp_path / "foreign-object.j2k"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    _prefix_holds(tmp_path, assignment, "symlink")


def test_typed_infeasibility_carrying_an_object_holds(tmp_path: Path) -> None:
    assignment, source = _fixture()
    F1Materializer(tmp_path, _Backend("infeasible"), scientific=True).materialize(assignment, source, split="train")
    _rewrite(
        tmp_path / "results/00000.json",
        lambda result: result.__setitem__("codestream", {"sha256": "0" * 64}),
        identity="result",
    )
    _prefix_holds(tmp_path, assignment, "carries an object")


def test_foreign_assignment_with_valid_json_and_result_id_holds(tmp_path: Path) -> None:
    assignment, _source, _materializer = _materialized(tmp_path)
    _rewrite(tmp_path / "results/00000.json", lambda result: result.__setitem__("assignment_id", "g8fassign-foreign"), identity="result")
    _prefix_holds(tmp_path, assignment, "another assignment")


def test_direct_existing_result_reuse_authenticates_object_bytes(tmp_path: Path) -> None:
    assignment, source, materializer = _materialized(tmp_path)
    result = json.loads((tmp_path / "results/00000.json").read_bytes())
    (tmp_path / result["codestream"]["path"]).unlink()
    with pytest.raises(G8FMaterializationHold, match="codestream"):
        materializer.materialize(assignment, source, split="train")


def test_f0_cli_cannot_start_without_separate_f1_owner_artifact() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_g8_f_f1.py",
            "--start",
            "--f0-authorization",
            "results/baseline/g8_f/f0_v2_execution_authorization.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "separate owner-issued F1 launch authorization" in result.stderr
