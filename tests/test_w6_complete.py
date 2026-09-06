"""Terminal W6-B publication binding and hostile mutation coverage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import verify_w6_complete as terminal


COMPLETION = terminal.COMPLETION_PATH


def _load_completion() -> dict[str, Any]:
    return json.loads(COMPLETION.read_text(encoding="utf-8"))


def _write_resigned(path: Path, value: dict[str, Any]) -> None:
    body = copy.deepcopy(value)
    body.pop("completion_id", None)
    body.pop("artifact_content_sha256", None)
    path.write_bytes(terminal.canonical(terminal._identified(body)))


def _mutated_completion(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    value = _load_completion()
    mutate(value)
    path = tmp_path / "w6_completion.json"
    _write_resigned(path, value)
    return path


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("source-commit", lambda x: x["authority"].update(source_commit="0" * 40)),
        ("source-manifest-id", lambda x: x["authority"]["source_manifest"].update(id="w6asource-" + "0" * 64)),
        ("source-manifest-sha", lambda x: x["authority"]["source_manifest"].update(file_sha256="0" * 64)),
        ("index-id", lambda x: x["index_matrix"]["index"].update(id="w6aindex-" + "0" * 64)),
        ("index-sha", lambda x: x["index_matrix"]["index"].update(file_sha256="0" * 64)),
        ("matrix-id", lambda x: x["index_matrix"]["matrix"].update(id="w6amatrix-" + "0" * 64)),
        ("matrix-sha", lambda x: x["index_matrix"]["matrix"].update(file_sha256="0" * 64)),
        ("requirement-count", lambda x: x["index_matrix"]["requirement_counts"].update(W6_REQUIRED_AND_SATISFIED=20)),
        ("g1-readiness", lambda x: x["readiness"]["g1"]["readiness"].update(verdict="HOLD")),
        ("g2-readiness", lambda x: x["readiness"]["g2"]["readiness"].update(measurement_commit="0" * 40)),
        ("w4-readiness", lambda x: x["readiness"]["w4"]["readiness"].update(g8_status="complete")),
        ("f1-corpus-count", lambda x: x["foundations"]["f1"]["outcomes"].update(materialized_verified_artifact=44040)),
        ("f1-corpus-id", lambda x: x["foundations"]["f1"].update(completion_id="g8ff1completion-" + "0" * 64)),
        ("f2-scorer-checkpoint", lambda x: x["foundations"]["f2"].update(selected_checkpoint_id="0" * 64)),
        ("f3-scorer", lambda x: x["foundations"]["f3"]["scorer"].update(scorer_identity="clean_reference_classifier")),
        ("pass-two-count", lambda x: x["terminal_scientific_values"]["pass_two_scope"].update(calls=19)),
        ("pass-three", lambda x: x["terminal_scientific_values"]["g8_passes"].update(pass_three=1)),
        ("ratios", lambda x: x["terminal_scientific_values"]["operating_points"].update(headline_ratio="r_1_24")),
        ("er1-strength", lambda x: x["terminal_scientific_values"]["er1_strength"].update(decision="full_strength_both_ratios")),
        ("nondegeneracy", lambda x: x["terminal_scientific_values"]["classical_nondegeneracy"][0].update(feasible_below_half_overhead_ldpc_rate_count=1)),
        ("br16", lambda x: x["terminal_scientific_values"]["br16"].update(modulation="qpsk")),
        ("h2", lambda x: x["terminal_scientific_values"]["h2"].update(low_snr_db=2.0)),
        ("repaired-w5", lambda x: x["foundations"]["w5"].update(repair_id="w5completion-" + "0" * 64)),
        ("learned-counter", lambda x: x["protected_counters"].update(scientific_learned_training_runs=1)),
        ("test-counter", lambda x: x["protected_counters"].update(test_model_facing_access=1)),
        ("future-boundary", lambda x: x["future_boundary"].update(g12_outputs_exist=True)),
    ],
)
def test_resigned_terminal_completion_mutations_fail(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    post_g10_am94,
) -> None:
    del post_g10_am94
    path = _mutated_completion(tmp_path, mutate)
    with pytest.raises(terminal.W6CompleteHold):
        terminal.verify_completion(path, reauthenticate=False)


def test_current_terminal_completion_reproduces_without_upstream_rerun(
    post_g10_am94,
) -> None:
    del post_g10_am94
    value = terminal.verify_completion(COMPLETION, reauthenticate=False)
    assert value["status"] == "W6_GREEN_CLOSED_CLASSICAL_PRE_TEST_IMPLEMENTATION_AND_EVIDENCE_BOUNDARY_AUTHENTICATED"
    assert value["completion_id"].startswith("w6completion-")


def test_frozen_consumer_is_terminal_default_and_has_no_nonterminal_production_callers() -> None:
    value = terminal._verify_frozen_consumer()
    assert value["default_requires_terminal_bytes"] is True
    assert value["nonterminal_opt_out_production_callers"] == []
    assert value["selection_performed"] is False
    assert value["test_loading"] is False


def test_terminal_completion_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "w6_completion.json"
    path.write_bytes(COMPLETION.read_bytes())
    with pytest.raises(terminal.W6CompleteHold, match="already exists"):
        terminal._write_immutable(path, COMPLETION.read_bytes())


def test_terminal_identity_rejects_noncanonical_or_reidentified_record(tmp_path: Path) -> None:
    value = _load_completion()
    value["artifact_content_sha256"] = "0" * 64
    path = tmp_path / "w6_completion.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(terminal.W6CompleteHold):
        terminal.verify_completion(path, reauthenticate=False)


def test_future_boundary_does_not_claim_g12_or_downstream_outputs() -> None:
    value = _load_completion()
    future = value["future_boundary"]
    assert future["status"] == "FUTURE_NOT_COMPLETE"
    assert future["g12_opened"] is False
    assert future["g12_outputs_exist"] is False
    assert set(future["items"]) == set(terminal.FUTURE_ITEMS)
    assert future["requirements_not_claimed_complete"] == terminal.FUTURE_ITEMS
    assert future["w7_g4"] == 0
    assert future["w8"] == 0
    assert future["test_model_facing_access"] == 0
