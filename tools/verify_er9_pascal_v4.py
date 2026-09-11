#!/usr/bin/env python3
"""Verify W9 v4 ER-9 source, authority and transactional terminal custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates  # noqa: E402
from runtime.source_guard import SourceGuardHold, assert_clean_source_closure, assert_v4_manifest_contract  # noqa: E402
from runtime.transactional_epochs import TransactionalEpochStore, TransactionalRuntimeHold  # noqa: E402
from runtime.w9_authority import W9AuthorityHold, load_authority, resolve_runtime_root  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402


SOURCE = REPO / "results/learned/er9/er_execution_source_manifest_v4.json"
AUTHORITY = REPO / "results/learned/er9/er9_stage1_execution_authorization_v4.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W9AuthorityHold(message)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"v4 artifact is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W9AuthorityHold(f"v4 artifact is corrupt: {path}: {exc}") from None
    _require(isinstance(value, dict), f"v4 artifact is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_manifest(path: Path = SOURCE) -> dict[str, Any]:
    value = _read(path)
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    _require(identifier == "er9sourcev4-" + canonical_sha256(body), "v4 source manifest ID differs")
    _require(value.get("schema_version") == 2, "v4 source manifest schema differs")
    _require(value.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze", "v4 source freeze rule differs")
    try:
        assert_v4_manifest_contract(value)
    except SourceGuardHold as exc:
        raise W9AuthorityHold(f"v4 source manifest contract differs: {exc}") from None
    try:
        assert_clean_source_closure(REPO, value)
    except SourceGuardHold as exc:
        raise W9AuthorityHold(f"v4 source closure differs: {exc}") from None
    return value


def _solver(config: Any) -> dict[str, Any]:
    floor = packetisation_floor(int(config.resolved["k"]))
    configured = all_configured_pairs()
    admissible = feasible_pairs(floor.payload_bits)
    candidates = stage1_candidates(floor.payload_bits)
    return {
        "k_symbols": int(config.resolved["k"]),
        "metadata_bits": 1,
        "packet_floor": floor.as_dict(),
        "configured_pairs": [item.as_dict() for item in configured],
        "admissible_pairs": [item.as_dict() for item in admissible],
        "rejected_pairs": [
            {**item.as_dict(), "reason": "raw_bound_exceeds_A_floor"}
            for item in configured
            if item not in admissible
        ],
        "stage1_candidates": [item.as_dict() for item in candidates],
    }


def assert_synthetic_smoke_ineligible(value: Mapping[str, Any]) -> None:
    """Prove that a smoke record cannot be admitted as Stage-1 evidence."""

    _require(value.get("artifact_role") == "W9_PASCAL_V4_SYNTHETIC_LIFECYCLE_SMOKE", "record is not the W9 v4 synthetic smoke")
    _require(value.get("status") == "NON_SCIENTIFIC", "synthetic smoke is not marked non-scientific")
    _require(isinstance(value.get("runtime_root"), str) and value["runtime_root"].startswith("checkpoints/smoke/"), "synthetic smoke runtime is not isolated")
    eligibility = value.get("eligibility")
    _require(isinstance(eligibility, Mapping), "synthetic smoke eligibility is missing")
    for marker in (
        "SYNTHETIC_ONLY",
        "INELIGIBLE_FOR_SELECTION",
        "INELIGIBLE_FOR_ER9_SEARCH",
        "INELIGIBLE_FOR_ER2_RESULT",
        "TEST_NOT_ACCESSED",
    ):
        _require(eligibility.get(marker) is True, f"synthetic smoke eligibility marker differs: {marker}")
    _require(value.get("scientific_data_accesses") == 0 and value.get("test_access") == 0 and value.get("test") == "SEALED", "synthetic smoke accessed scientific/test data")


def verify_stage1_authority(
    authority_path: Path = AUTHORITY,
    manifest_path: Path = SOURCE,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    manifest = verify_source_manifest(manifest_path)
    authority = load_authority(authority_path, kind="W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4")
    body = dict(authority)
    identifier = body.pop("authority_id", None)
    _require(identifier == "w9er9stage1v4auth-" + canonical_sha256(body), "v4 Stage-1 authority ID differs")
    source = authority.get("source_manifest")
    _require(isinstance(source, dict), "v4 Stage-1 source record is missing")
    _require(source.get("path") == str(manifest_path.relative_to(REPO)), "v4 Stage-1 source path differs")
    _require(source.get("manifest_id") == manifest["manifest_id"], "v4 Stage-1 manifest ID differs")
    _require(source.get("sha256") == _sha(manifest_path), "v4 Stage-1 manifest SHA differs")
    _require(authority.get("source_commit") == manifest["source_commit"], "v4 Stage-1 source commit differs")
    _require(authority.get("source_binding") == manifest, "v4 Stage-1 full source binding differs")
    _require(authority.get("schema_version") == 1, "v4 Stage-1 authority schema differs")
    _require(authority.get("status") == "FROZEN_STAGE1_ONLY_PRE_SCIENCE", "v4 Stage-1 authority status differs")
    _require(authority.get("authorization_scope") == "W9_ER9_STAGE1_ONLY", "v4 Stage-1 authority scope differs")
    _require(authority.get("config_path") == "configs/er9-digital-pascal-v4.yaml", "v4 Stage-1 config path differs")
    config = load_experiment(authority["config_path"], train_seed=0, channel_seed=0)
    _require(authority.get("config_hash") == config_hash(config), "v4 Stage-1 config hash differs")
    _require(authority.get("config_source_blob_sha256") == manifest["relevant_config_sha256"][authority["config_path"]], "v4 Stage-1 config source blob differs")
    _require(authority.get("runtime_root") == "checkpoints/er9_pascal_v4", "v4 Stage-1 runtime root differs")
    _require(authority.get("execution_profile_id") == "confessor_pascal_cu126", "v4 Stage-1 profile differs")
    _require(authority.get("host") == "confessor", "v4 Stage-1 host differs")
    _require(authority.get("device") == "cuda:0", "v4 Stage-1 logical device differs")
    gpu_uuid = authority.get("gpu_uuid")
    _require(isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"), "v4 Stage-1 GPU UUID differs")
    _require(authority.get("cuda_visible_devices") == gpu_uuid, "v4 Stage-1 visible UUID differs")
    mapping = authority.get("cuda_mapping")
    _require(
        mapping == {
            "cuda_visible_devices": gpu_uuid,
            "logical_device": "cuda:0",
            "cuda0_gpu_uuid": gpu_uuid,
            "cuda0_gpu_name": authority.get("gpu_name"),
            "cuda0_compute_capability": authority.get("compute_capability"),
            "device_count": 1,
        },
        "v4 Stage-1 CUDA mapping differs",
    )
    _require(authority.get("sole_writer") is True and authority.get("live_authentication_required_immediately_before_model_and_data") is True, "v4 Stage-1 writer/authentication policy differs")
    _require(authority.get("source_working_tree_guard_required") is True, "v4 Stage-1 source guard is not required")
    _require(authority.get("fresh_initialization_required") is True, "v4 Stage-1 fresh initialization is not required")
    _require(authority.get("stage2_authorized") is False and authority.get("production_authorized") is False and authority.get("randomized_er2_authorized") is False, "v4 Stage-1 scope is wider than Stage-1")
    _require(authority.get("test") == "SEALED" and authority.get("test_access") == 0, "v4 Stage-1 test boundary differs")
    counters = authority.get("pre_execution_counters")
    _require(
        counters == {
            "new_er9_v4_training": 0,
            "randomized_er2_scientific_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "v4 Stage-1 pre-execution counters differ",
    )
    solver = _solver(config)
    proof = authority.get("packet_budget_admissibility_proof")
    _require(isinstance(proof, dict), "v4 Stage-1 solver proof is missing")
    for key in ("k_symbols", "metadata_bits", "packet_floor", "configured_pairs", "admissible_pairs", "rejected_pairs", "stage1_candidates"):
        _require(proof.get(key) == solver[key], f"v4 Stage-1 derived arithmetic differs at {key}")
    _require(proof.get("configured_pair_count") == len(solver["configured_pairs"]), "v4 configured pair count differs")
    _require(proof.get("admissible_pair_count") == len(solver["admissible_pairs"]), "v4 admissible pair count differs")
    _require(authority.get("stage1_candidates") == solver["stage1_candidates"], "v4 Stage-1 candidate set differs")
    _require(authority.get("stage1_candidate_count") == len(solver["stage1_candidates"]), "v4 Stage-1 candidate count differs")
    _require(authority.get("stage1_training_count") == len(solver["stage1_candidates"]), "v4 Stage-1 training count differs")
    _require(authority.get("k_symbols") == solver["k_symbols"] and authority.get("metadata_bits") == solver["metadata_bits"], "v4 top-level packet arithmetic differs")
    _require(authority.get("packet_floor") == solver["packet_floor"], "v4 top-level packet floor differs")
    _require(authority.get("configured_pair_count") == len(solver["configured_pairs"]) and authority.get("admissible_pair_count") == len(solver["admissible_pairs"]), "v4 top-level pair counts differ")
    _require(authority.get("admissible_pairs") == solver["admissible_pairs"] and authority.get("rejected_pairs") == solver["rejected_pairs"], "v4 top-level pair lists differ")
    _require(
        authority.get("stage1_rule") == {
            "candidate_order": "ascending_numeric_transmit_dim",
            "cross_product": False,
            "quantiser_bits": 2,
            "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
            "tie_break": "smallest_transmit_dim",
        },
        "v4 Stage-1 candidate rule differs",
    )
    return manifest, authority, config, solver


def _verify_terminal(
    *,
    runtime: Path,
    candidate: dict[str, int],
    manifest: dict[str, Any],
    authority: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    from training.er9_v4 import expected_er9_v4_identity  # noqa: PLC0415
    from training.er9_v4 import ER9V4CandidateTrainer  # noqa: PLC0415

    source_binding = dict(manifest)
    identity = expected_er9_v4_identity(
        config=config,
        transmit_dim=candidate["transmit_dim"],
        quantiser_bits=candidate["quantiser_bits"],
        source_binding=source_binding,
        campaign_id="er9_stage1_v4",
        run_id=f"er9-stage1-v4-D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}",
    )
    total_epochs = int(config.parameters["learned_system"]["epochs"][config.resolved["dataset"]])
    store = TransactionalEpochStore(runtime, identity=identity, total_epochs=total_epochs, role=ER9V4CandidateTrainer.ROLE)
    try:
        committed = store.inspect()
    except TransactionalRuntimeHold as exc:
        raise W9AuthorityHold(f"v4 candidate transaction is not authentic: {runtime}: {exc}") from None
    _require(len(committed) == total_epochs, f"v4 candidate epoch count differs: {runtime}")
    terminal = _read(runtime / "run_terminal.json")
    terminal_body = dict(terminal)
    terminal_id = terminal_body.pop("terminal_id", None)
    _require(terminal_id == "w9terminal-" + canonical_sha256(terminal_body), f"v4 terminal ID differs: {runtime}")
    _require(terminal.get("run_identity") == identity, f"v4 terminal identity differs: {runtime}")
    _require(terminal.get("total_epochs") == total_epochs and terminal.get("test") == "SEALED" and terminal.get("test_access") == 0, f"v4 terminal scope differs: {runtime}")
    selected_epoch = terminal.get("selected_epoch")
    _require(isinstance(selected_epoch, int) and 0 <= selected_epoch < total_epochs, f"v4 selected epoch differs: {runtime}")
    selected = committed[selected_epoch]
    _require(terminal.get("selected_checkpoint_path") == str(selected.checkpoint_path.relative_to(runtime)), f"v4 selected checkpoint path differs: {runtime}")
    _require(terminal.get("selected_checkpoint_sha256") == selected.checkpoint_sha256, f"v4 selected checkpoint hash differs: {runtime}")
    _require(terminal.get("committed_epoch_chain_sha256") == committed[-1].chain_sha256, f"v4 terminal chain differs: {runtime}")
    _require(terminal.get("selection_metric") == {"metric": "validation_n_correct", "mode": "max", "value": selected.record["validation_n_correct"]}, f"v4 terminal selection metric differs: {runtime}")
    _require(terminal.get("tie_break") == "earliest_epoch", f"v4 terminal tie-break differs: {runtime}")
    expected_opportunities = sum(int(item.record.get("optimizer_opportunities", 0)) for item in committed)
    expected_applied = sum(int(item.record.get("applied_optimizer_steps", 0)) for item in committed)
    expected_skips = sum(int(item.record.get("grad_scaler_skips", 0)) for item in committed)
    _require(terminal.get("optimizer_opportunities") == expected_opportunities and terminal.get("applied_optimizer_steps") == expected_applied and terminal.get("grad_scaler_skips") == expected_skips, f"v4 terminal optimizer counters differ: {runtime}")
    expected_terminal, _ = store.terminalize(selected_epoch=selected_epoch, selection_metric=terminal["selection_metric"], tie_break=terminal["tie_break"], test=terminal["test"])
    _require(expected_terminal == terminal, f"v4 terminal does not reproduce canonically: {runtime}")
    return {
        "candidate": candidate,
        "runtime_root": str(runtime.relative_to(REPO)),
        "epochs": len(committed),
        "terminal_sha256": _sha(runtime / "run_terminal.json"),
        "selected_epoch": selected_epoch,
        "selected_checkpoint_sha256": selected.checkpoint_sha256,
        "chain_sha256": committed[-1].chain_sha256,
    }


def verify_stage1_terminals(
    manifest: dict[str, Any],
    authority: dict[str, Any],
    config: Any,
    *,
    require_terminals: bool = False,
) -> list[dict[str, Any]]:
    root = resolve_runtime_root(REPO, authority)
    if not root.exists() and not root.is_symlink():
        if require_terminals:
            raise W9AuthorityHold("v4 Stage-1 runtime is absent")
        return []
    _require(not root.is_symlink() and root.is_dir(), "v4 Stage-1 runtime root is unsafe")
    stage1_root = root / "stage1"
    _require(stage1_root.is_dir() and not stage1_root.is_symlink(), "v4 Stage-1 candidate root is unsafe")
    expected_names = {
        f"D{int(candidate['transmit_dim'])}_b{int(candidate['quantiser_bits'])}"
        for candidate in authority["stage1_candidates"]
    }
    actual_names = {child.name for child in stage1_root.iterdir()}
    _require(actual_names == expected_names, "v4 Stage-1 runtime contains an unexpected candidate or missing candidate")
    reports = []
    for candidate in authority["stage1_candidates"]:
        candidate = {"transmit_dim": int(candidate["transmit_dim"]), "quantiser_bits": int(candidate["quantiser_bits"])}
        runtime = stage1_root / f"D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}"
        reports.append(_verify_terminal(runtime=runtime, candidate=candidate, manifest=manifest, authority=authority, config=config))
    if require_terminals and len(reports) != int(authority["stage1_training_count"]):
        raise W9AuthorityHold("v4 Stage-1 terminal count differs")
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-terminals", action="store_true")
    args = parser.parse_args(argv)
    manifest, authority, config, _solver_result = verify_stage1_authority()
    reports = verify_stage1_terminals(manifest, authority, config, require_terminals=args.require_terminals)
    print(f"W9 v4 Stage-1 verifier PASS: terminals={len(reports)} source={manifest['source_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
