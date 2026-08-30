"""W7-C terminal authentication and consequential mutation regressions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import verify_w7_g4 as verifier

REPO = Path(__file__).resolve().parents[1]
G4_PATH = REPO / "results/learned/w7/w7_g4_result.json"
TERMINAL_PATH = REPO / "results/learned/w7/w7_completion.json"

Mutation = Callable[[dict[str, Any]], None]


@pytest.fixture(scope="module")
def _authenticated_b2r():
    # Authenticate the real compact evidence once; mutation cases exercise the
    # terminal boundary without repeating the full read-only B2R traversal.
    return verifier._verify_b2r_and_authority()


@pytest.fixture(autouse=True)
def _cache_b2r(monkeypatch: pytest.MonkeyPatch, _authenticated_b2r):
    monkeypatch.setattr(verifier, "_verify_b2r_and_authority", lambda: _authenticated_b2r)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _resign(value: dict[str, Any], key: str, prefix: str) -> None:
    value[key] = prefix + verifier.canonical_sha256(
        {name: item for name, item in value.items() if name != key}
    )


def _fails_g4(tmp_path: Path, mutate: Mutation) -> None:
    value = _load(G4_PATH)
    mutate(value)
    _resign(value, "adjudication_id", "w7g4adjudication-")
    path = tmp_path / "w7_g4_result.json"
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_adjudication(path)


def _fails_terminal(tmp_path: Path, mutate: Mutation) -> None:
    value = _load(TERMINAL_PATH)
    mutate(value)
    _resign(value, "completion_id", "w7completion-")
    path = tmp_path / "w7_completion.json"
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_terminal_completion(path)


def test_authenticated_g4_and_terminal_closeout_pass():
    g4 = verifier.verify_adjudication(G4_PATH)
    terminal = verifier.verify_terminal_completion(TERMINAL_PATH, g4=g4)
    assert g4["candidate_lambdas"] == [0.0, 0.1, 0.3, 1.0, 3.0]
    assert g4["primary_qualifying_lambdas"] == [3.0]
    assert g4["relaxed_qualifying_lambdas"] == [0.1, 0.3, 1.0, 3.0]
    assert g4["selection_tier"] == "PRIMARY"
    assert g4["selected_lambda"] == 3.0
    assert terminal["normative_lambda"] == {
        "source_of_truth": "spec/SPEC.md",
        "lambda_core": 3.0,
        "lambda_status": "selected_at_G-4",
        "provisional_g4_status_cleared": True,
        "spec_views": terminal["normative_lambda"]["spec_views"],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda value: value.__setitem__("A0", 0.826), id="altered-A0"),
        pytest.param(lambda value: value.__setitem__("accuracy_tolerance_pp", 0.1), id="altered-tolerance"),
        pytest.param(lambda value: value.__setitem__("primary_psnr_floor_db", 19.0), id="altered-primary-floor"),
        pytest.param(lambda value: value.__setitem__("relaxed_psnr_floor_db", 15.0), id="altered-relaxed-floor"),
        pytest.param(lambda value: value["candidates"][1]["validation"].__setitem__("top1_accuracy", 0.9), id="altered-candidate-metric"),
        pytest.param(lambda value: value.__setitem__("primary_qualifying_lambdas", [1.0, 3.0]), id="altered-qualifying-set"),
        pytest.param(lambda value: value.__setitem__("selected_lambda", 1.0), id="altered-selected-lambda"),
        pytest.param(lambda value: value["adjudicator_output"].__setitem__("selected_lambda", 1.0), id="altered-inner-adjudicator-output"),
        pytest.param(lambda value: value.__setitem__("selection_tier", "RELAXED"), id="primary-relaxed-substitution"),
        pytest.param(lambda value: value["adjudication_boundary"].__setitem__("g4_adjudication_run", 2), id="second-adjudication"),
        pytest.param(lambda value: value["adjudication_boundary"].__setitem__("w8_final_training_runs", 1), id="opened-W8"),
        pytest.param(lambda value: value["adjudication_boundary"].__setitem__("test_model_facing_access", 1), id="test-access"),
        pytest.param(lambda value: value["w7_pilot_weights"].__setitem__("w8_initialization_eligibility", "ELIGIBLE"), id="pilot-W8-eligibility"),
    ],
)
def test_g4_mutations_fail_closed(tmp_path: Path, mutate: Mutation):
    _fails_g4(tmp_path, mutate)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda value: value["normative_lambda"].__setitem__("lambda_core", 1.0), id="normative-lambda"),
        pytest.param(lambda value: value["normative_lambda"].__setitem__("lambda_status", "provisional_until_G-4"), id="provisional-status"),
        pytest.param(lambda value: value["g4_adjudication"].__setitem__("accuracy_baseline_A0", 0.826), id="terminal-A0"),
        pytest.param(lambda value: value["g4_adjudication"].__setitem__("accuracy_tolerance_pp", 0.1), id="terminal-tolerance"),
        pytest.param(lambda value: value["g4_adjudication"].__setitem__("primary_qualifying_lambdas", [1.0, 3.0]), id="terminal-qualifying-set"),
        pytest.param(lambda value: value["g4_adjudication"].__setitem__("selected_lambda", 1.0), id="terminal-selected-lambda"),
        pytest.param(lambda value: value["g4_adjudication"].__setitem__("selection_tier", "RELAXED"), id="terminal-tier"),
        pytest.param(lambda value: value["protected_counters"].__setitem__("g4_adjudications", 2), id="terminal-second-adjudication"),
        pytest.param(lambda value: value["protected_counters"].__setitem__("w8_final_training_runs", 1), id="terminal-W8"),
        pytest.param(lambda value: value["protected_counters"].__setitem__("test_model_facing_access", 1), id="terminal-test-access"),
        pytest.param(lambda value: value["w7_pilot_weights"].__setitem__("status", "ELIGIBLE_FOR_W8_INITIALIZATION"), id="terminal-pilot-eligibility"),
        pytest.param(lambda value: value["future_boundary"].__setitem__("w8_initialization_from_w7_pilot", True), id="terminal-pilot-initialization"),
    ],
)
def test_terminal_mutations_fail_closed(tmp_path: Path, mutate: Mutation):
    _fails_terminal(tmp_path, mutate)


def test_independent_reconstruction_selects_numeric_minimum_primary():
    g4 = verifier.verify_adjudication(G4_PATH)
    reconstructed = verifier.reconstruct_frozen_decision(g4["candidates"])
    assert reconstructed["selection_tier"] == "PRIMARY"
    assert reconstructed["selected_lambda"] == min(reconstructed["primary_qualifying_lambdas"]) == 3.0
    assert reconstructed == g4["adjudicator_output"]
