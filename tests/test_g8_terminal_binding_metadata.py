"""Bounded legacy/current G8 terminal-binding identity regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline import g8_g_closeout as closeout
from baseline.g8_f_f3 import identified, rendered_json


def _refresh_correction(value: dict) -> dict:
    body = {
        key: child
        for key, child in value.items()
        if key not in {"correction_id", "artifact_content_sha256"}
    }
    return identified(
        body,
        field="correction_id",
        prefix=closeout.TERMINAL_BINDING_CORRECTION_PREFIX,
    )


def _mutated_correction(tmp_path: Path, mutate) -> Path:
    value = json.loads(closeout.TERMINAL_BINDING_CORRECTION_PATH.read_bytes())
    mutate(value)
    path = tmp_path / "correction.json"
    path.write_bytes(rendered_json(_refresh_correction(value)))
    return path


def test_exact_historical_closeout_and_current_typed_bindings_verify() -> None:
    historical = closeout.verify_historical_closeout()
    correction = closeout.verify_terminal_binding_correction()
    typed = closeout.current_typed_bindings()
    assert historical["closeout_id"] == correction["historical_closeout_id"]
    assert historical["bindings"]["adjudication_input"]["id"].startswith("g8fpass2compare-")
    assert typed["adjudication_input"]["id"].startswith("g8ginput-")
    assert typed["g8_d_handoff"]["id"].startswith("g8dhandoff-")
    assert typed["closeout_repair_provenance"]["id"].startswith("g8closeoutrepair-")


@pytest.mark.parametrize("replacement", ["wrong-own-id", "g8fpass2compare-upstream-substitution"])
def test_correction_rejects_wrong_or_upstream_own_id(tmp_path: Path, replacement: str) -> None:
    path = _mutated_correction(
        tmp_path,
        lambda value: value["affected_bindings"][0].__setitem__(
            "corrected_own_artifact_id", replacement
        ),
    )
    with pytest.raises(closeout.G8CloseoutHold, match="does not reproduce"):
        closeout.verify_terminal_binding_correction(path)


@pytest.mark.parametrize("field", ["path", "file_sha256"])
def test_correction_rejects_path_or_file_hash_drift(tmp_path: Path, field: str) -> None:
    path = _mutated_correction(
        tmp_path,
        lambda value: value["affected_bindings"][1].__setitem__(field, "drift"),
    )
    with pytest.raises(closeout.G8CloseoutHold, match="does not reproduce"):
        closeout.verify_terminal_binding_correction(path)


def test_typed_binding_rejects_unknown_type_and_schema(tmp_path: Path) -> None:
    with pytest.raises(closeout.G8CloseoutHold, match="unknown G8 terminal artifact type"):
        closeout.typed_artifact_binding("not_registered")
    value = json.loads(closeout.INPUT_PATH.read_bytes())
    value["schema_version"] += 1
    path = tmp_path / "future.json"
    path.write_bytes(rendered_json(value))
    with pytest.raises(closeout.G8CloseoutHold, match="unknown schema"):
        closeout.typed_artifact_binding("adjudication_input", path=path)


def test_typed_binding_rejects_missing_own_identity(tmp_path: Path) -> None:
    value = json.loads(closeout.INPUT_PATH.read_bytes())
    value.pop("input_id")
    path = tmp_path / "missing.json"
    path.write_bytes(rendered_json(value))
    with pytest.raises(closeout.G8CloseoutHold, match="missing own identity"):
        closeout.typed_artifact_binding("adjudication_input", path=path)


def test_typed_binding_rejects_duplicate_own_identity(tmp_path: Path) -> None:
    raw = closeout.INPUT_PATH.read_text(encoding="ascii")
    needle = '  "input_id": '
    line = next(line for line in raw.splitlines() if line.startswith(needle))
    duplicate = raw.replace(line, line + "\n" + line, 1)
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="ascii")
    with pytest.raises(closeout.G8CloseoutHold, match="duplicate JSON key 'input_id'"):
        closeout.typed_artifact_binding("adjudication_input", path=path)


def test_modified_historical_closeout_bytes_are_rejected(tmp_path: Path) -> None:
    raw = bytearray(closeout.CLOSEOUT_PATH.read_bytes())
    raw[-2] = ord(" ")
    path = tmp_path / "changed-closeout.json"
    path.write_bytes(raw)
    with pytest.raises(closeout.G8CloseoutHold, match="historical G8 closeout bytes differ"):
        closeout.verify_historical_closeout(path)


def test_legacy_acceptance_cannot_be_broadened_to_corrected_id(tmp_path: Path) -> None:
    value = json.loads(closeout.CLOSEOUT_PATH.read_bytes())
    value["bindings"]["adjudication_input"]["id"] = closeout.typed_artifact_binding(
        "adjudication_input"
    )["id"]
    path = tmp_path / "broadened.json"
    path.write_bytes(rendered_json(value))
    with pytest.raises(closeout.G8CloseoutHold, match="historical G8 closeout bytes differ"):
        closeout.verify_historical_closeout(path)


def test_scientific_boundary_is_exactly_zero() -> None:
    value = closeout.verify_terminal_binding_correction()
    assert value["scientific_boundary"] == closeout._SCIENTIFIC_BOUNDARY_ZERO
