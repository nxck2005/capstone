from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from verify_openjpeg_profile_parity import OpenJpegParityError, compare, verify


def _report(profile: str):
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_openjpeg_parity",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile,
        "python_version": "3.14.6",
        "numpy_version": "2.5.1",
        "glymur_version": "0.14.3",
        "openjpeg_version": "2.5.4",
        "fixture_shape": [64, 64, 3],
        "fixture_pixels_sha256": "a" * 64,
        "codec_configuration": {"progression_order": "LRCP"},
        "cells": [{
            "compression_ratio": 8.0,
            "requested_payload_budget_bytes": 10,
            "emitted_codestream_bytes": 10,
            "codestream_sha256": "b" * 64,
            "decoded_shape": [64, 64, 3],
            "decoded_dtype": "uint8",
            "decoded_pixels_sha256": "c" * 64,
        }],
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def test_openjpeg_equal_reports_pass():
    left = _report("local_4060_cu130")
    right = _report("confessor_pascal_cu126")
    assert compare(left, right)["codestream_and_decoded_pixels_equal"]


def test_openjpeg_hash_mutation_fails():
    left = _report("local_4060_cu130")
    right = copy.deepcopy(_report("confessor_pascal_cu126"))
    right["cells"][0]["codestream_sha256"] = "d" * 64
    right["report_sha256"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in right.items() if k != "report_sha256"})).hexdigest()
    with pytest.raises(AssertionError):
        assert compare(left, right)["codestream_and_decoded_pixels_equal"]
    verify(right)
