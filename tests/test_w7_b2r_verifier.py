"""Offline W7-B2R reconciliation authentication and fail-closed mutations."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import verify_w7_b2r as verifier
from training.deterministic_core import canonical_sha256


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "results/learned/w7"
ARTIFACTS = (
    "w7_b2_reconciliation_index.json",
    "w7_b2_checkpoint_custody.json",
    "w7_b2_common_noise_audit.json",
    "w7_b2_reconciliation.json",
    "w7_b2_completion.json",
)
ARTIFACT_IDS = {
    "w7_b2_reconciliation_index.json": ("index_id", "w7b2rindex-"),
    "w7_b2_checkpoint_custody.json": ("custody_id", "w7b2rcustody-"),
    "w7_b2_common_noise_audit.json": ("audit_id", "w7b2rnoise-"),
    "w7_b2_reconciliation.json": ("reconciliation_id", "w7b2rreconciliation-"),
    "w7_b2_completion.json": ("completion_id", "w7b2rcompletion-"),
}


Mutation = Callable[[dict[str, dict[str, Any]]], None]


def _load_bundle(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads((directory / name).read_bytes())
        for name in ARTIFACTS
    }


def _write_bundle(directory: Path, bundle: dict[str, dict[str, Any]]) -> None:
    directory.mkdir()
    for name, value in bundle.items():
        (directory / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )


def _resign(bundle: dict[str, dict[str, Any]], name: str) -> None:
    value = bundle[name]
    id_key, prefix = ARTIFACT_IDS[name]
    value[id_key] = prefix + canonical_sha256(
        {key: item for key, item in value.items() if key != id_key}
    )


def _resign_selected_result(bundle: dict[str, dict[str, Any]], candidate: int = 0) -> None:
    index = bundle["w7_b2_reconciliation_index.json"]
    result = index["candidates"][candidate]["selected"]["selected_checkpoint_result"]["value"]
    result["result_digest"] = canonical_sha256(
        {key: item for key, item in result.items() if key != "result_digest"}
    )
    _resign(bundle, "w7_b2_reconciliation_index.json")


def test_committed_reconciliation_verifies_without_worker_access():
    result = verifier.verify_artifacts(ARTIFACT_DIR, reauthenticate_upstream=False)
    assert result["status"] == "PASS"
    assert result["candidate_count"] == 5
    assert result["completed_epoch_cycles"] == 500
    assert result["checkpoint_count"] == 500
    assert result["g4_adjudication_run"] == 0
    assert result["lambda_decision"] == "NOT_PERFORMED"
    assert result["lambda_core_updated"] is False
    assert result["w8_final_training_runs"] == 0
    assert result["test_model_facing_access"] == 0


def _mutate_source(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["source"]["source_commit"] = "0" * 40
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_campaign_id(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["campaign_id"] = "foreign-campaign"
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_candidate_lambda(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["lambda"] = 0.1
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_candidate_count(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_completion.json"]["candidate_count"] = 4
    _resign(bundle, "w7_b2_completion.json")


def _mutate_selected_epoch(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["selected"]["selected_epoch"] = 85
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_selected_checkpoint(bundle: dict[str, dict[str, Any]]) -> None:
    result = bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["selected"]["selected_checkpoint_result"]["value"]
    result["checkpoint_id"] = "0" * 64
    _resign_selected_result(bundle)


def _mutate_top1_count(bundle: dict[str, dict[str, Any]]) -> None:
    result = bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["selected"]["selected_checkpoint_result"]["value"]
    result["calibration_validation"]["n_correct"] += 1
    _resign_selected_result(bundle)


def _mutate_psnr(bundle: dict[str, dict[str, Any]]) -> None:
    result = bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["selected"]["selected_checkpoint_result"]["value"]
    result["psnr_evaluation"]["psnr_db"] += 1.0
    _resign_selected_result(bundle)


def _mutate_gpu_uuid(bundle: dict[str, dict[str, Any]]) -> None:
    profile = bundle["w7_b2_reconciliation_index.json"]["candidates"][0]["profile_binding"]
    profile["gpu_uuid"] = "GPU-foreign"
    profile["binding_sha256"] = canonical_sha256(
        {key: item for key, item in profile.items() if key != "binding_sha256"}
    )
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_seed(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["protocol"]["train_seed"] = 1
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_validation_denominator(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_reconciliation_index.json"]["protocol"]["validation_denominator"] = 999
    _resign(bundle, "w7_b2_reconciliation_index.json")


def _mutate_noise_digest(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_common_noise_audit.json"]["noise_id_digest"] = "0" * 64
    _resign(bundle, "w7_b2_common_noise_audit.json")


def _mutate_campaign_role(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_completion.json"]["artifact_role"] = "W7_B2R_G4_COMPLETE"
    _resign(bundle, "w7_b2_completion.json")


def _mutate_g4_counter(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_completion.json"]["g4_adjudication_run"] = 1
    _resign(bundle, "w7_b2_completion.json")


def _mutate_w8_counter(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_completion.json"]["w8_final_training_runs"] = 1
    _resign(bundle, "w7_b2_completion.json")


def _mutate_test_seal(bundle: dict[str, dict[str, Any]]) -> None:
    bundle["w7_b2_completion.json"]["test_state"] = "OPEN"
    _resign(bundle, "w7_b2_completion.json")


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("source commit", _mutate_source),
        ("campaign ID", _mutate_campaign_id),
        ("candidate lambda", _mutate_candidate_lambda),
        ("candidate count", _mutate_candidate_count),
        ("selected epoch", _mutate_selected_epoch),
        ("selected checkpoint", _mutate_selected_checkpoint),
        ("top1 count", _mutate_top1_count),
        ("PSNR", _mutate_psnr),
        ("GPU UUID", _mutate_gpu_uuid),
        ("seed", _mutate_seed),
        ("validation denominator", _mutate_validation_denominator),
        ("noise digest", _mutate_noise_digest),
        ("campaign role", _mutate_campaign_role),
        ("G4 counter", _mutate_g4_counter),
        ("W8 counter", _mutate_w8_counter),
        ("test seal", _mutate_test_seal),
    ],
)
def test_consequential_inner_mutation_fails_closed(
    tmp_path: Path, label: str, mutate: Mutation
):
    del label
    bundle = _load_bundle(ARTIFACT_DIR)
    mutate(bundle)
    artifact_dir = tmp_path / "w7"
    _write_bundle(artifact_dir, bundle)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_artifacts(artifact_dir, reauthenticate_upstream=False)
