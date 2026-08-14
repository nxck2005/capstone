from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from verify_execution_profile_performance import PerformanceVerificationError, verify


def _report():
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_synthetic_performance",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": "confessor_pascal_cu126",
        "device": "cuda:0",
        "synthetic_data_only": True,
        "training_campaign": False,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "inference": 0,
        "measurements": [{"batch_size": 2, "status": "PASS", "peak_gpu_memory_bytes": 1, "throughput_images_per_s": 1.0, "all_finite": True}],
        "safe_batch_size": 2,
        "peak_gpu_memory_bytes_at_safe_batch": 1,
        "projected_epoch_time_s_at_safe_batch": 1.0,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def test_performance_fixture_passes():
    verify(_report(), expected_profile="confessor_pascal_cu126")


def test_performance_mutation_fails():
    report = _report()
    report["training_campaign"] = True
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in report.items() if k != "report_sha256"})).hexdigest()
    with pytest.raises(PerformanceVerificationError):
        verify(report, expected_profile="confessor_pascal_cu126")
