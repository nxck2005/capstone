from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline import w6_evidence as w6
from baseline.classical.frozen_selection import FrozenSelectionError, load_frozen_selection


def _write(path: Path, value: dict) -> None:
    path.write_bytes(w6.canonical(value))


def _resign(value: dict, field: str, prefix: str) -> dict:
    body = copy.deepcopy(value); body.pop(field, None)
    return w6._identified(body, field=field, prefix=prefix)


def _mutated_index(tmp_path: Path, mutate) -> Path:
    value = w6.build_index(); mutate(value)
    value = _resign(value, "index_id", w6.INDEX_PREFIX)
    path = tmp_path / "index.json"; _write(path, value); return path


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("wrong-path", lambda x: x["bindings"][0].update(path="spec/NOT-SPEC.md")),
        ("wrong-file-sha", lambda x: x["bindings"][0].update(file_sha256="0" * 64)),
        ("wrong-own-id", lambda x: next(r for r in x["bindings"] if r["logical_name"] == "g8_closeout").update(own_artifact_id="g8closeout-" + "0" * 64)),
        ("nested-id-as-own", lambda x: next(r for r in x["bindings"] if r["logical_name"] == "g8_adjudication_input").update(own_artifact_id="g8fpass2compare-ac713b219348383a27152d4a3ba746f695e5899d8c585fea0d663f2f6a228c5f")),
        ("missing-artifact", lambda x: x["bindings"].pop()),
        ("wrong-ratio", lambda x: x["claims"]["operating_points"].update(headline_ratio="r_1_24")),
        ("pass-two-count", lambda x: x["claims"]["pass_two"]["counters"].update(pass_two=2)),
        ("pass-three", lambda x: x["claims"]["pass_two"]["counters"].update(pass_three=1)),
        ("bler", lambda x: x["claims"]["pass_two"]["inputs"].update(bler_table_id="g8pblertable-" + "0" * 64)),
        ("scorer", lambda x: x["claims"]["f3"]["scorer"].update(scorer_identity="clean_reference_classifier")),
        ("br16", lambda x: x["claims"]["br16_h2_validation_freeze"]["fixed_configuration"].update(modulation="qpsk")),
        ("h2-window", lambda x: x["claims"]["br16_h2_validation_freeze"].update(low_snr_db=2.0)),
        ("er1-strength", lambda x: x["claims"]["er1_strength"].update(decision="full_strength_both_ratios")),
        ("corpus-count", lambda x: x["claims"]["f1_corpus"]["outcomes"].update(materialized_verified_artifact=44040)),
        ("test-counter", lambda x: x["claims"]["pass_two"]["counters"].update(test_access=1)),
        ("learned-training", lambda x: x["claims"]["w5_terminal_authority"]["protected_counters"].update(scientific_learned_training_runs=1)),
        ("historical-w5-authority", lambda x: x["claims"]["w5_terminal_authority"].update(repair_id=x["claims"]["w5_terminal_authority"]["supersedes"]["completion_id"])),
        ("schema-drift", lambda x: x.update(schema_version=2)),
        ("unknown-future-schema", lambda x: x.update(schema_version=999)),
    ],
)
def test_resigned_malicious_index_fails_inner_or_reproduction(tmp_path: Path, name: str, mutate) -> None:
    path = _mutated_index(tmp_path, mutate)
    with pytest.raises(w6.W6Hold):
        w6.verify_index(path, invoke_upstream=False)


def test_false_future_classification_is_rejected_even_when_resigned(tmp_path: Path) -> None:
    index = w6.build_index(); value = w6.build_matrix(index)
    row = next(r for r in value["entries"] if r["requirement"] == "BR-4" and r["obligation"].startswith("validation tuning"))
    row["status"] = "FUTURE_G12_TEST_EXECUTION"
    value["counts"]["W6_REQUIRED_AND_SATISFIED"] -= 1
    value["counts"]["FUTURE_G12_TEST_EXECUTION"] += 1
    value = _resign(value, "matrix_id", w6.MATRIX_PREFIX)
    path = tmp_path / "matrix.json"; _write(path, value)
    with pytest.raises(w6.W6Hold, match="classification"):
        w6.verify_matrix(path, index=index)


def test_current_index_and_matrix_are_deterministic() -> None:
    index, matrix = w6.verify_all(invoke_upstream=False)
    assert index == w6.build_index()
    assert matrix == w6.build_matrix(index)
    assert matrix["counts"]["W6_REQUIRED_AND_MISSING"] == 0
    assert index["terminal_w6_completion_published"] is False


def test_frozen_selected_config_loader_consumes_without_selection() -> None:
    value = load_frozen_selection("r_1_6", "classical_adaptive", 7.0)
    assert value.candidate_id == "cand-15e6711e9b406157262234a8"
    assert value.candidate["modulation"] == "qam16"
    assert value.candidate["ldpc_rate"] == "1/2"
    source = Path(__import__("baseline.classical.frozen_selection", fromlist=["x"]).__file__).read_text()
    assert "select_operating_points" not in source
    assert "from data.test_access" not in source


def test_frozen_selected_config_loader_rejects_unknown_cell() -> None:
    with pytest.raises(FrozenSelectionError, match="expected one frozen"):
        load_frozen_selection("r_1_6", "classical_adaptive", 8.0)
