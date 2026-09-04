"""Focused fail-closed mutation tests for the W8-C terminal carrier."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import verify_w8_c as verifier  # noqa: E402


RECONCILIATION = Path(__file__).resolve().parents[1] / "results/learned/w8/w8_c_reconciliation.json"


def _value() -> dict:
    return json.loads(RECONCILIATION.read_bytes())


def _reidentify(value: dict) -> dict:
    body = dict(value)
    body.pop("reconciliation_id")
    value["reconciliation_id"] = "w8creconcile-" + verifier.canonical_sha256(body)
    return value


def _expect_hold(mutator) -> None:
    value = copy.deepcopy(_value())
    mutator(value)
    value = _reidentify(value)
    with pytest.raises(verifier.W8CHold):
        verifier.verify_reconciliation(value)


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(lambda value: value["transaction_accounting"]["per_run"][0]["transactions"].pop(), id="missing-epoch"),
        pytest.param(lambda value: value["transaction_accounting"]["per_run"][0]["transactions"][0].pop("scientific_sidecar"), id="missing-sidecar"),
        pytest.param(lambda value: value["transaction_accounting"]["per_run"][0]["transactions"][0]["checkpoint_payload"].update(file_sha256="0" * 64), id="altered-checkpoint-sha"),
        pytest.param(lambda value: value["transaction_accounting"]["per_run"][0]["transactions"][1].update(predecessor_checkpoint_id="0" * 64), id="broken-predecessor"),
        pytest.param(lambda value: value["validation_noise"].update(denominator=999), id="wrong-validation-denominator"),  # literal-ok: mutation fixture
        pytest.param(lambda value: value["validation_noise"]["per_run"][0].update(digest="0" * 64), id="changed-validation-noise"),
        pytest.param(lambda value: value["selection"].update(metric="reconstruction_loss"), id="wrong-selection-metric"),
        pytest.param(lambda value: value["selection"].update(tie_break="latest_epoch"), id="wrong-tie-break"),
        pytest.param(lambda value: value["selection"]["per_run"][0]["runner_published"].update(checkpoint_id="0" * 64), id="selected-checkpoint-mismatch"),
        pytest.param(lambda value: value["predecessor_exclusion"].update(old_partial_checkpoint_in_successor=True), id="old-c5a8-reuse"),
        pytest.param(lambda value: value["source"].update(source_commit="0" * 40), id="wrong-source"),
        pytest.param(lambda value: value["authorities"]["execution_authorization"].update(id="w8auth-" + "0" * 64), id="wrong-authority"),
        pytest.param(lambda value: value["run_identities"].pop(), id="missing-run"),
        pytest.param(lambda value: value["run_identities"].append({"run_id": "w8-foreign-run"}), id="foreign-seventh-run"),
        pytest.param(lambda value: value["protected_boundaries"].update(g10=1), id="nonzero-g10"),  # literal-ok: mutation fixture
        pytest.param(lambda value: value["protected_boundaries"].update(test_model_facing_access=1), id="nonzero-test-access"),  # literal-ok: mutation fixture
    ],
)
def test_w8_c_compact_carrier_rejects_mutation(mutator) -> None:
    _expect_hold(mutator)


def test_w8_c_compact_carrier_accepts_frozen_bytes() -> None:
    verifier.verify_reconciliation(_value())


def _completion_value() -> dict:
    path = Path(__file__).resolve().parents[1] / "results/learned/w8/w8_completion.json"
    return json.loads(path.read_bytes())


def _reidentify_completion(value: dict) -> dict:
    body = dict(value)
    body.pop("completion_id")
    value["completion_id"] = "w8completion-" + verifier.canonical_sha256(body)
    return value


def test_w8_terminal_completion_binds_reconciliation_sha_and_selection() -> None:
    reconciliation = _value()
    completion = _completion_value()
    actual_sha = verifier._sha_bytes(RECONCILIATION.read_bytes())
    verifier.verify_terminal_completion(completion, reconciliation, actual_sha)

    completion = copy.deepcopy(completion)
    completion["reconciliation"]["sha256"] = "0" * 64
    _reidentify_completion(completion)
    with pytest.raises(verifier.W8CHold):
        verifier.verify_terminal_completion(completion, reconciliation, actual_sha)

    completion = _completion_value()
    completion["selected_checkpoints"][0]["checkpoint_id"] = "0" * 64
    _reidentify_completion(completion)
    with pytest.raises(verifier.W8CHold):
        verifier.verify_terminal_completion(completion, reconciliation, actual_sha)
