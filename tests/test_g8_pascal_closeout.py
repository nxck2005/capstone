from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline.g8_pascal_merge import (
    MERGE_REPORT_PATH,
    TABLE_PATH,
    SuccessorMergeError,
    build_successor_bler_table,
    load_successor_bler_table,
)
from baseline.g8_pascal_successor import SUCCESSOR_ROOT
import verify_g8_pascal_closeout as independent


RUNTIME = SUCCESSOR_ROOT / "runtime"
PROVENANCE_PATH = SUCCESSOR_ROOT / "successor_closeout_provenance.json"


def _payload(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


@pytest.fixture(scope="module")
def artifacts():
    merge, merge_raw = _payload(MERGE_REPORT_PATH)
    table, table_raw = _payload(TABLE_PATH)
    provenance, _provenance_raw = _payload(PROVENANCE_PATH)
    return merge, table, provenance, merge_raw, table_raw


def _reidentify(payload: dict, field: str, prefix: str) -> None:
    payload[field] = independent._self_id(payload, field, prefix)


def _assert_mutation_rejected(artifacts, mutant_merge=None, mutant_table=None) -> None:
    merge, table, provenance, merge_raw, table_raw = artifacts
    with pytest.raises(independent.CloseoutVerificationError):
        independent.validate_payloads(
            merge if mutant_merge is None else mutant_merge,
            table if mutant_table is None else mutant_table,
            provenance,
            merge,
            table,
        )


def test_full_successor_closeout_payload_passes(artifacts) -> None:
    merge, table, provenance, merge_raw, table_raw = artifacts
    independent.validate_payloads(
        merge,
        table,
        provenance,
        merge,
        table,
        merge_raw=merge_raw,
        table_raw=table_raw,
        runtime_root=RUNTIME,
    )
    assert merge["accepted_count"] == merge["required_identity_count"] == 3213
    assert table["measured_point_count"] == 3213


def test_missing_authority_ordinal_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"].pop(0)
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_duplicate_authority_ordinal_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][1] = copy.deepcopy(mutant["units"][0])
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_foreign_campaign_result_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][0]["campaign_id"] = "g8-predecessor"
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_wrong_pascal_execution_profile_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["execution_profile_id"] = "local_4060_cu130"
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_altered_production_contract_binding_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["production_contract_sha256"] = "0" * 64
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_wrong_trials_completed_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][0]["trials_completed"] = 4999
    mutant["units"][0]["raw_measurement"]["trials_completed"] = 4999
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


@pytest.mark.parametrize("field", ["terminal_invalid_count", "unresolved_count", "old_result_ingest"])
def test_invalid_terminal_or_unresolved_state_fails(field: str, artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant[field] = 1 if field != "old_result_ingest" else True
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_nonzero_test_access_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["test_access"] = 1
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_nonzero_protected_counter_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["protected_counters"]["inference"] = 1
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


@pytest.mark.parametrize("field", ["request_sha256", "result_sha256", "state_sha256"])
def test_altered_request_result_state_hash_fails(field: str, artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][0][field] = "0" * 64
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_retry_history_counts_failed_attempt_zero_and_accepts_attempt_two_once(artifacts) -> None:
    merge = artifacts[0]
    for ordinal in (0, 1):
        row = merge["units"][ordinal]
        assert [item["attempt"] for item in row["historical_attempts"]] == [1, 2]
        assert row["historical_attempts"][0]["result_status"] == "failed"
        assert row["historical_attempts"][0]["required_coverage_contribution"] == 0
        assert row["historical_attempts"][1]["result_status"] == "complete"
        assert row["historical_attempts"][1]["required_coverage_contribution"] == 1
    assert merge["retry_history_ordinals"] == [0, 1]
    assert merge["failed_result_attempt_count"] == 2

    mutant = copy.deepcopy(merge)
    mutant["units"][0]["historical_attempts"][0]["required_coverage_contribution"] = 1
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_extra_unknown_identity_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"].append(copy.deepcopy(mutant["units"][-1]))
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_same_count_wrong_identity_or_snr_tuple_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][0]["snr_db"] = mutant["units"][0]["snr_db"] + 1
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_curve_canonicalization_is_input_order_independent(artifacts) -> None:
    from baseline.g8_pascal_merge import _table_curves

    units = artifacts[0]["units"]
    assert _table_curves(units) == _table_curves(list(reversed(units)))


@pytest.mark.parametrize("field", ["interpolation_used", "imputation_used", "extrapolation_used"])
def test_invention_markers_are_rejected(field: str, artifacts) -> None:
    mutant_merge = copy.deepcopy(artifacts[0])
    mutant_merge[field] = True
    _reidentify(mutant_merge, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant_merge)

    mutant_table = copy.deepcopy(artifacts[1])
    mutant_table[field] = True
    _reidentify(mutant_table, "table_id", independent.TABLE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_table=mutant_table)


def test_measured_zero_bler_survives_exactly_and_is_not_interpolated(artifacts) -> None:
    table = load_successor_bler_table(verify_runtime=False)
    zero_point = next(
        point
        for curve in artifacts[1]["curves"]
        for point in curve["points"]
        if point["bler"] == 0.0
    )
    identity = next(curve["identity"] for curve in artifacts[1]["curves"] if any(point["work_unit_id"] == zero_point["work_unit_id"] for point in curve["points"]))
    lookup = table.lookup(identity, zero_point["snr_db"])
    assert lookup.bler == 0.0
    assert lookup.interpolated is False

    mutant = copy.deepcopy(artifacts[1])
    mutant["curves"][0]["points"][0]["bler"] = 1e-12
    _reidentify(mutant, "table_id", independent.TABLE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_table=mutant)


def test_merge_raw_count_mutation_fails(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[0])
    mutant["units"][0]["bit_errors"] += 1
    mutant["units"][0]["raw_measurement"]["bit_errors"] += 1
    _reidentify(mutant, "report_id", independent.MERGE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_merge=mutant)


def test_table_hash_and_curve_mutations_fail(artifacts) -> None:
    mutant = copy.deepcopy(artifacts[1])
    mutant["curves"][0]["points"][0]["block_errors"] += 1
    _reidentify(mutant, "table_id", independent.TABLE_ID_PREFIX)
    _assert_mutation_rejected(artifacts, mutant_table=mutant)


def test_predecessor_table_cannot_be_loaded_as_successor() -> None:
    predecessor = Path("results/baseline/g8/bler_table_v2.json").resolve()
    with pytest.raises(SuccessorMergeError, match="non-successor table path"):
        load_successor_bler_table(predecessor, verify_runtime=False)


def test_partial_successor_merge_cannot_create_table(artifacts) -> None:
    partial = copy.deepcopy(artifacts[0])
    partial["units"].pop()
    with pytest.raises(SuccessorMergeError):
        build_successor_bler_table(partial)
