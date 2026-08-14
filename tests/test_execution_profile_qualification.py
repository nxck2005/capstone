from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from config.params import get
from verify_execution_profile_qualification import (
    QualificationVerificationError,
    verify_report,
)


def _report():
    profile_id = "local_4060_cu130"
    profile = get(f"environment.execution_profiles.{profile_id}")
    environment = {
        field: "fixture" for field in get("environment.execution_profile_record_fields")
    }
    environment.update(
        execution_profile_id=profile_id,
        lock_file=profile["lock_file"],
        lock_file_sha256=profile["lock_file_sha256"],
        gpu_uuid=profile["allowed_gpu_uuids"][0],
        gpu_index=0,
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_qualification",
        "scientific_status": "NON-SCIENTIFIC",
        "g8_coverage": 0,
        "test_access_count": 0,
        "validation_count": 0,
        "selection_count": 0,
        "training_campaign_count": 0,
        "synthetic_data_only": True,
        "execution_profile_id": profile_id,
        "device": "cuda:0",
        "environment": environment,
        "checks": {
            "real_cuda_tensor_operations": True,
            "ldpc": {"encode_pass": True, "decode_pass": True},
            "project_modulation_demodulation": {"bpsk": True, "qpsk": True, "qam16": True},
            "djscc": {
                "forward_pass": True,
                "backward_pass": True,
                "finite_loss": True,
                "finite_gradients": True,
                "optimizer_step": True,
                "checkpoint_save": True,
                "checkpoint_reload": True,
                "same_seed_repeatability": True,
                "safe_batch_size": 8,
                "peak_gpu_memory_bytes": 1,
                "throughput_images_per_s": 1.0,
            },
            "crash_kill_restart": {
                "child_reached_cuda": True,
                "terminated_child": True,
                "fresh_cuda_after_kill": True,
            },
        },
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def test_qualification_fixture_passes():
    verify_report(_report(), expected_profile="local_4060_cu130")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("g8_coverage",), 1),
        (("scientific_status",), "SCIENTIFIC"),
        (("checks", "ldpc", "decode_pass"), False),
        (("checks", "djscc", "finite_gradients"), False),
        (("environment", "lock_file_sha256"), "0" * 64),
        (("environment", "gpu_uuid"), "GPU-wrong"),
    ],
)
def test_qualification_mutations_fail(path, value):
    report = copy.deepcopy(_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report["report_sha256"] = hashlib.sha256(
        canonical_json_bytes({k: v for k, v in report.items() if k != "report_sha256"})
    ).hexdigest()
    with pytest.raises(QualificationVerificationError):
        verify_report(report, expected_profile="local_4060_cu130")
