"""Offline authentication of the bounded W7-B1 CUDA resume evidence."""

from __future__ import annotations

import json

import verify_w7_b1 as b1


def test_cuda_resume_smoke_is_authenticated_and_non_scientific():
    value = b1.verify_smoke(json.loads(b1.SMOKE_PATH.read_bytes()))
    assert value["status"] == "PASSED"
    assert value["scientific_status"] == "NON_SCIENTIFIC_ZERO_G4_COVERAGE"
    assert value["non_scientific_w7_b1_resume_smoke_optimizer_steps"] > 0
    assert value["scientific_boundary"]["w7_scientific_optimizer_steps"] == 0
    assert value["validation"] == {"performed": False, "model_facing": False}


def test_cuda_resume_smoke_seam_and_latest_chain_are_exact():
    value = b1.verify_smoke(json.loads(b1.SMOKE_PATH.read_bytes()))
    assert value["resume_seam"] == {
        "model_state_equal": True,
        "optimizer_state_equal": True,
        "scheduler_state_equal": True,
        "scaler_state_equal": True,
        "completed_epoch_equal": True,
        "global_optimizer_step_equal": True,
        "scaler_state_before_restore_sha256": value["process_a"]["scaler_state_sha256"],
        "scaler_state_after_restore_sha256": value["process_a"]["scaler_state_sha256"],
        "restored_checkpoint_id": value["process_a"]["checkpoint_id"],
    }
    assert value["checkpoint_chain"]["latest_only"] is True
    assert value["checkpoint_chain"]["older_fallback"] is False
    assert value["checkpoint_chain"]["successor_predecessor_checkpoint_id"] == value["process_a"]["checkpoint_id"]
