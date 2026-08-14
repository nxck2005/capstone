from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from verify_execution_profile_rng import RngAuditVerificationError, compare_reports, verify_report


def _report(profile: str = "local_4060_cu130"):
    stream = {
        "seed_uint64": 1,
        "count": 4,
        "shape": [1, 4],
        "dtype": "uint8",
        "sha256": "a" * 64,
    }
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_rng_audit",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile,
        "python_version": "3.14.6",
        "numpy_version": "2.5.1",
        "campaign_id": "g8-old",
        "required_identity_sha256": "b" * 64,
        "parity_plan_sha256": "c" * 64,
        "trials_per_selected_identity": 1,
        "stream_contract": "contract",
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
        "cells": [
            {
                "ordinal": 1,
                "work_unit_id": "bler-x",
                "identity": {},
                "snr_db": 0,
                "stratum": "near_waterfall",
                "streams": {
                    "information_bits": stream,
                    "awgn_real": {**stream, "dtype": "float64"},
                    "awgn_imag": {**stream, "dtype": "float64"},
                },
            }
        ],
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def test_rng_report_and_equal_comparison():
    left = _report()
    right = copy.deepcopy(left)
    right["execution_profile_id"] = "confessor_pascal_cu126"
    right["report_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in right.items() if key != "report_sha256"})
    ).hexdigest()
    verify_report(left)
    verify_report(right)
    result = compare_reports(left, right)
    assert result["status"] == "PASS"
    assert result["information_bits_equal"]


@pytest.mark.parametrize("mutation", ["hash", "coverage", "purpose"])
def test_rng_mutations_fail(mutation):
    report = _report()
    if mutation == "hash":
        report["cells"][0]["streams"]["information_bits"]["sha256"] = "d" * 64
    elif mutation == "coverage":
        report["g8_coverage"] = 1
    else:
        report["cells"][0]["streams"]["extra"] = report["cells"][0]["streams"].pop("awgn_imag")
    report["report_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in report.items() if key != "report_sha256"})
    ).hexdigest()
    if mutation == "hash":
        assert compare_reports(_report(), report)["status"] == "MISMATCH"
    else:
        with pytest.raises(RngAuditVerificationError):
            verify_report(report)
