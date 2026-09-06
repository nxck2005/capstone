"""Source-only W9-A/G-10 authority checks; no model or dataset inference."""

from __future__ import annotations

from evaluation import g10_protocol as protocol


def test_corrected_grid_is_normative_and_exact() -> None:
    assert protocol.expected_grid() == protocol.EXPECTED_GRID
    assert protocol.protocol_identity()["snr_grid_db"] == list(protocol.EXPECTED_GRID)
    assert protocol.expected_cell_count() == 63


def test_am94_boundary_remains_pre_science() -> None:
    value = protocol.verify_am94_boundary()
    assert value["scientific_boundary"]["g10_model_facing_evaluations"] == 0
    assert value["scientific_boundary"]["test_split"] == "SEALED"


def test_classical_extract_is_exactly_frozen_adaptive_r1_6() -> None:
    value = protocol.build_classical_extract()
    protocol.verify_classical_extract(value)
    assert value["snr_grid_db"] == list(protocol.EXPECTED_GRID)
    assert value["point_count"] == 21
    assert all(row["ratio"] == "r_1_6" for row in value["points"])
    assert all(row["clean_denominator"] == 1000 for row in value["points"])
    assert all(row["stored_expected_accuracy_is_predicate_input"] is False for row in value["points"])


def test_w8_mapping_is_frozen_without_cross_seed_selection() -> None:
    rows = protocol._w8_selected_checkpoints()
    assert [(row["train_seed"], row["channel_seed"], row["epoch"]) for row in rows] == [
        (0, 0, 92),
        (1, 1, 77),
        (2, 2, 78),
    ]
    assert all(row["ratio"] == "r_1_6" for row in rows)
    assert [row["checkpoint_id"] for row in rows] == [
        "b0f72a3e16c537984b6afd3dc93bdf3ea87a0cae8a5b49f3565c803750a8826a",
        "5d7e692c723ee9657be9fc45bfdae5dc54adf137ee9e92a04de2b2e5a7bbdcda",
        "d5595f0931b010805f59ded64b3cda88b35730d200c5578aecf5962c8d058f41",
    ]


def test_no_outcome_files_exist_before_authority() -> None:
    actual = {
        path.relative_to(protocol.REPO_ROOT).as_posix()
        for path in (protocol.REPO_ROOT / "results/learned/w9").glob("**/*")
        if path.is_file()
    }
    assert actual == {"results/learned/w9/am94_pre_science_freeze.json"}
