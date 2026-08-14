from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from verify_execution_profile_parity import ParityVerificationError, compare, verify_report


def _report(profile: str):
    indicators = [0, 1, 0, 1]
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_paired_numerical_parity",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile,
        "device": "cuda:0",
        "environment": {},
        "campaign_id": "g8-old",
        "parity_plan_sha256": "a" * 64,
        "required_identity_count": 3213,
        "selected_cell_count": 1,
        "paired_trial_count_per_cell": 4,
        "diagnostic_only": True,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
        "criterion": {
            "per_cell_disagreement_rate_max": 0.02,
            "aggregate_disagreement_rate_max": 0.01,
            "waterfall_displacement_db_max": 0.5,
        },
        "cells": [
            {
                "ordinal": 1,
                "work_unit_id": "bler-x",
                "identity": {"base_graph": 1, "k_and_n": [2, 4], "lifting_size": 22, "modulation": "bpsk", "rate": "1/2"},
                "snr_db": 2,
                "trials": 4,
                "information_bits": 8,
                "bit_errors": 2,
                "block_errors": 2,
                "bler": 0.5,
                "ber": 0.25,
                "block_error_indicators": indicators,
                "stream_seeds": {"information_bits": 1, "awgn_real": 2, "awgn_imag": 3},
            }
        ],
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def test_parity_equal_reports_pass():
    left = _report("local_4060_cu130")
    right = _report("confessor_pascal_cu126")
    assert compare(left, right)["criterion_pass"]


@pytest.mark.parametrize("mutation", ["indicator", "seed", "coverage"])
def test_parity_mutations_fail(mutation):
    left = _report("local_4060_cu130")
    right = _report("confessor_pascal_cu126")
    if mutation == "indicator":
        right["cells"][0]["block_error_indicators"][0] = 1
    elif mutation == "seed":
        right["cells"][0]["stream_seeds"]["awgn_real"] = 99
    else:
        right["g8_coverage"] = 1
    right["report_sha256"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in right.items() if k != "report_sha256"})).hexdigest()
    with pytest.raises(ParityVerificationError):
        compare(left, right)
