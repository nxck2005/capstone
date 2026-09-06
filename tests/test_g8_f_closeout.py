"""Synthetic mutation coverage for the compact G8_F/F1 closeout path."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from baseline.g8_f_closeout import (
    COMPLETION_PATH,
    MANIFEST_PATH,
    COMPLETION_PREFIX,
    G8FF1CloseoutHold,
    collect_authenticated_runtime,
    verify_closeout,
    verify_monitor_closeout,
)
from baseline.g8_f_materializer import (
    CODEC_CONFIGURATION_HASH,
    CODEC_CONFIGURATION_ID,
    F1Assignment,
    F1Materializer,
    G8FMaterializationHold,
    canonical_json,
    rendered_json,
    sha256_bytes,
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
    cache_key: str = "synthetic-closeout-cache"


class _Backend:
    configuration_hash = CODEC_CONFIGURATION_HASH

    def __init__(self, outcome: str = "materialized") -> None:
        self.outcome = outcome

    def encode_to_budget(self, image: np.ndarray, **_kwargs: object) -> _Result:
        if self.outcome == "raise":
            raise RuntimeError("synthetic interruption")
        if self.outcome == "infeasible":
            return _Result(False, None, None, CODEC_CONFIGURATION_HASH, None, False, None)
        codestream = b"synthetic-closeout-codestream" + bytes([int(image[0, 0, 0])])
        return _Result(
            True,
            codestream,
            len(codestream),
            CODEC_CONFIGURATION_HASH,
            image.copy(),
            True,
            hashlib.sha256(codestream).hexdigest(),
        )


def _fixture(index: int) -> tuple[F1Assignment, SourceSample]:
    pixels = np.full((160, 160, 3), 40 + index, dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    raw = stream.getvalue()
    stable_id = stable_sample_id(raw)
    assignment = F1Assignment(
        index,
        stable_id,
        index,
        f"g8fquality-synthetic-{index}",
        256,
        160,
        CODEC_CONFIGURATION_ID,
    )
    return assignment, SourceSample("imagenette160", stable_id, index, raw)


def _materialize(root: Path, index: int, outcome: str = "materialized") -> F1Assignment:
    assignment, source = _fixture(index)
    F1Materializer(root, _Backend(outcome), scientific=True).materialize(assignment, source, split="train")
    return assignment


def _rewrite(path: Path, mutate: object, identity: str) -> None:
    value = json.loads(path.read_bytes())
    mutate(value)  # type: ignore[operator]
    body = dict(value)
    body.pop(f"{identity}_id", None)
    value[f"{identity}_id"] = f"g8f{identity}-" + sha256_bytes(canonical_json(body))
    path.write_bytes(rendered_json(value))


def test_synthetic_complete_closeout_authenticates_every_record_and_object(tmp_path: Path) -> None:
    assignment = _materialize(tmp_path, 0)
    value = collect_authenticated_runtime(tmp_path, (assignment,), expected_total=1)
    assert value["authenticated_prefix"] == 1
    assert value["outcomes"]["materialized_verified_artifact"] == 1
    assert value["outcomes"]["unexpected_or_other"] == 0
    assert value["objects"]["unique_codestream_objects"] == 1
    assert value["objects"]["unique_reconstruction_objects"] == 1


def test_incomplete_prefix_and_orphan_request_hold_at_closeout(tmp_path: Path) -> None:
    first = _materialize(tmp_path, 0)
    second, source = _fixture(1)
    with pytest.raises(G8FMaterializationHold, match="unexpected codec/runtime failure"):
        F1Materializer(tmp_path, _Backend("raise"), scientific=True).materialize(second, source, split="train")
    with pytest.raises(G8FF1CloseoutHold, match="incomplete"):
        collect_authenticated_runtime(tmp_path, (first, second), expected_total=2)


@pytest.mark.parametrize("kind", ["request", "result"])
def test_corrupt_request_or_result_holds_at_closeout(tmp_path: Path, kind: str) -> None:
    assignment = _materialize(tmp_path, 0)
    path = tmp_path / kind / "00000.json" if kind == "result" else tmp_path / "requests/00000.json"
    if kind == "result":
        path = tmp_path / "results/00000.json"
    raw = path.read_bytes()
    path.write_bytes(raw[:-2] + b"x\n")
    with pytest.raises((G8FMaterializationHold, json.JSONDecodeError)):
        collect_authenticated_runtime(tmp_path, (assignment,), expected_total=1)


@pytest.mark.parametrize("operation", ["missing", "corrupt"])
def test_missing_or_corrupt_object_holds_at_closeout(tmp_path: Path, operation: str) -> None:
    assignment = _materialize(tmp_path, 0)
    result = json.loads((tmp_path / "results/00000.json").read_bytes())
    path = tmp_path / result["reconstruction"]["path"]
    if operation == "missing":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(G8FMaterializationHold, match="reconstruction"):
        collect_authenticated_runtime(tmp_path, (assignment,), expected_total=1)


def test_foreign_assignment_and_duplicate_ordinal_content_hold(tmp_path: Path) -> None:
    first = _materialize(tmp_path, 0)
    second = _materialize(tmp_path, 1)
    (tmp_path / "results/00001.json").write_bytes((tmp_path / "results/00000.json").read_bytes())
    with pytest.raises(G8FMaterializationHold, match="another assignment|another request"):
        collect_authenticated_runtime(tmp_path, (first, second), expected_total=2)


def test_unexpected_outcome_is_not_downgraded_to_infeasibility(tmp_path: Path) -> None:
    assignment = _materialize(tmp_path, 0)
    _rewrite(
        tmp_path / "results/00000.json",
        lambda value: value.__setitem__("outcome", "decode_failure"),
        "result",
    )
    with pytest.raises(G8FMaterializationHold, match="outcome is not permitted"):
        collect_authenticated_runtime(tmp_path, (assignment,), expected_total=1)


def test_validation_and_test_membership_never_enter_synthetic_closeout(tmp_path: Path) -> None:
    assignment, source = _fixture(0)
    backend = _Backend()
    for split in ("val", "test"):
        with pytest.raises(G8FMaterializationHold, match="train split"):
            F1Materializer(tmp_path, backend, scientific=True).materialize(assignment, source, split=split)
    assert not (tmp_path / "requests").exists()


def _mutated_completion(tmp_path: Path, mutate: object) -> Path:
    value = copy.deepcopy(json.loads(COMPLETION_PATH.read_bytes()))
    mutate(value)  # type: ignore[operator]
    body = dict(value)
    body.pop("completion_id", None)
    value["completion_id"] = COMPLETION_PREFIX + sha256_bytes(canonical_json(body))
    path = tmp_path / "completion.json"
    path.write_bytes(rendered_json(value))
    return path


def test_committed_f1_closeout_verifies_offline_without_worker_corpus(post_g10_am94) -> None:
    del post_g10_am94
    value = verify_closeout()
    assert value["coverage"]["authenticated_prefix"] == 50_814  # literal-ok: frozen AM-88 assignment count
    assert value["outcomes"]["unexpected_or_other"] == 0
    assert value["f2_readiness"]["f2_launched"] is False
    monitor = verify_monitor_closeout(completion=value)
    assert monitor["delivery"]["http_status"] == 204  # literal-ok: Discord successful no-content response
    assert monitor["transition"]["active_f1_polling"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["lineage"].__setitem__("ordered_pair_sha256", "0" * 64), "AM-88 digest"),
        (lambda value: value["lineage"]["f0"].__setitem__("authorization_id", "g8ff0v3auth-foreign"), "F0 lineage"),
        (lambda value: value["protected_counters"].__setitem__("artifact_classifier_optimizer_steps", 1), "counter"),
        (lambda value: value["data_membership"].__setitem__("validation_ids", 1), "validation/test"),
        (lambda value: value["f2_readiness"].__setitem__("f2_launched", True), "F2 was opened"),
        (lambda value: value["digests"].__setitem__("ordered_result_record_sha256", "0" * 64), "ordered F1 result-record digest"),
        (lambda value: value["outcomes"].__setitem__("unexpected_or_other", 1), "outcome counts|unexpected outcomes"),
    ],
)
def test_offline_closeout_mutations_fail_closed(
    tmp_path: Path, mutation: object, match: str, post_g10_am94
) -> None:
    del post_g10_am94
    path = _mutated_completion(tmp_path, mutation)
    with pytest.raises(G8FF1CloseoutHold, match=match):
        verify_closeout(path, MANIFEST_PATH)
