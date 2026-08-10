"""Epoch-2 orchestration, history, and provenance regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline import g8_bler_characterization_v2 as characterization
from baseline import g8_bler_resume as resume
from baseline.g8_campaign import sha256_file
import merge_g8_bler_characterization_v2 as merge
import verify_g8_bler_characterization_manifest_v2 as provenance


REPO = Path(__file__).resolve().parents[1]


def test_registered_epoch2_manifest_has_exact_activation_boundary() -> None:
    payload = json.loads(characterization.SOURCE_MANIFEST_PATH.read_bytes())
    validated = characterization.validate_source_manifest(payload, require_registered=True)
    assert validated["epoch"] == 2
    assert validated["activation_boundary"] == provenance.EXPECTED_ACTIVATION
    assert validated["source_epochs"][0]["accepted_result_ordinals"] == [0, 178]
    assert validated["source_epochs"][1]["accepted_result_ordinals"] == [179, 3212]


def test_post_data_independent_verifier_tracks_registered_progress() -> None:
    result = provenance.verify()
    completed = result["completed_count"]
    assert provenance.EXPECTED_ACTIVATION["epoch_1_accepted_result_count"] <= completed <= characterization.REQUIRED_COUNT
    assert result["remaining_count"] == characterization.REQUIRED_COUNT - completed
    assert result["test_split_access"] == 0


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["predecessor"].update({"sha256": "0" * 64}),
        lambda value: value["source_epochs"].__setitem__(1, {**value["source_epochs"][1], "accepted_result_ordinals": [180, 3212]}),
        lambda value: value["activation_boundary"].update({"first_legal_epoch_2_attempt": 4}),
        lambda value: value["sources"][1].update({"sha256": "0" * 64}),
        lambda value: value.update({"full_strength_trials": 4999}),
    ),
)
def test_epoch2_manifest_mutations_fail_closed(mutator) -> None:
    payload = json.loads(characterization.SOURCE_MANIFEST_PATH.read_bytes())
    mutator(payload)
    with pytest.raises(characterization.CharacterizationError):
        characterization.validate_source_manifest(payload)


def test_epoch1_bytes_and_registration_remain_immutable() -> None:
    expected = {
        "results/baseline/g8/bler_characterization_source_manifest.json": "a917f839f945232e85852d6d27f02de4b5dc272adc72b1966a95e9b5e62a014e",
        "src/baseline/g8_bler_characterization.py": "a79e9e0f968fcf1733f1a6ee7ac39eb3660c721e83c110e786638bbb217f1cd0",
        "tools/run_g8_bler_characterization.py": "367e4a67906feb8c47f825ca3ca5aaebd5b95642f98bf36c3b035629c4b295a3",
        "tools/gen_g8_bler_characterization_manifest.py": "2f536a3ebc0d9462a88acd9b1905a0f5184d42607af1e82e627fe125e27a87ba",
        "tools/verify_g8_bler_characterization_manifest.py": "01f0700ef20144c2ae7bc46f5ca27429fd8d2329f307ec699f270aced3d8bf05",
        "tools/merge_g8_bler_characterization.py": "636308cc4de28ea1d7fdb64afed71dbc072ecf6cfdcb06a4899d3c970dd50186",
        "tools/verify_g8_bler_table.py": "bc0208d5a0e8131e934aeb04cdcc0b1157d0fc92ae0995f90c40eed442c85965",
    }
    for relative, digest in expected.items():
        assert sha256_file(REPO / relative) == digest, relative


def test_request_only_and_failed_history_are_distinguished(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def request(_context, work_unit_id, attempt, **_kwargs):
        calls.append(("request", attempt))
        return {"request": {"work_unit_id": work_unit_id}, "request_sha256": "a" * 64}

    def result(_context, work_unit_id, attempt, **_kwargs):
        calls.append(("result", attempt))
        return {"status": "failed" if attempt == 2 else "complete"}

    monkeypatch.setattr(resume, "validate_request_file", request)
    monkeypatch.setattr(resume, "validate_result_file", result)
    counts = merge._historical_attempts(
        object(),
        Path("/tmp/unused-g8c-history-root"),
        "unit",
        3,
        [1, 2, 3],
        [2, 3],
    )
    assert counts == (3, 1, 1)
    assert calls == [("request", 1), ("request", 2), ("result", 2), ("request", 3), ("result", 3)]


def test_request_only_history_is_not_rewritten_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resume,
        "validate_request_file",
        lambda *_args, **_kwargs: {"request": {"work_unit_id": "unit"}, "request_sha256": "a" * 64},
    )
    observed: list[int] = []

    def result(*_args, **kwargs):
        observed.append(kwargs.get("attempt", _args[2] if len(_args) > 2 else -1))
        return {"status": "complete"}

    monkeypatch.setattr(resume, "validate_result_file", result)
    assert merge._historical_attempts(object(), Path("/tmp/unused-g8c-history-root"), "unit", 3, [1, 2, 3], [3]) == (3, 2, 0)
    assert observed == [3]


def test_v2_merge_and_post_data_verifier_do_not_import_v1_verifier() -> None:
    source = (REPO / "tools/verify_g8_bler_characterization_manifest_v2.py").read_text()
    assert "import verify_g8_bler_characterization_manifest as" not in source
    assert "source_epoch" in (REPO / "tools/merge_g8_bler_characterization_v2.py").read_text()
