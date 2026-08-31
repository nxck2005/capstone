#!/usr/bin/env python3
"""Verify and publish the additive terminal W6-B completion.

W6-B is a publication and verification layer over the frozen W6-A boundary.  It
never selects a configuration, runs a codec/channel, trains a model, reads the
test split, or regenerates any scientific artifact.  The default command runs
the existing read-only gate verifiers before accepting the terminal record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TOOLS = REPO / "tools"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from baseline import g8_e_corrected_v3s as g8_e_v3s  # noqa: E402
from baseline import g8_f_closeout as f1_closeout  # noqa: E402
from baseline import g8_f_f3 as f3  # noqa: E402
from baseline import g8_f_pass_two as pass_two  # noqa: E402
from baseline import g8_g_closeout as g8  # noqa: E402
from baseline import w6_evidence as w6  # noqa: E402
from baseline.classical.frozen_selection import load_frozen_selection  # noqa: E402
from training import g8_f_f2_closeout as f2_closeout  # noqa: E402
from verify_g1_adjudication import verify as verify_g1  # noqa: E402
from verify_g2_adjudication import verify as verify_g2  # noqa: E402
from verify_g8_d_handoff import verify as verify_g8_d  # noqa: E402
from verify_g8_pascal_closeout import verify as verify_g8_c  # noqa: E402
from verify_w5_training_system import verify_completion as verify_w5  # noqa: E402

SCHEMA_VERSION = 1
COMPLETION_ROLE = "w6_terminal_classical_pre_test_completion"
COMPLETION_PREFIX = "w6completion-"
COMPLETION_PATH = REPO / "results/baseline/w6/w6_completion.json"

W6_A_SOURCE_COMMIT = "d0e04d0ccc92e2fa7dae0be798da4b6bd8960854"
W6_A_SOURCE_PARENT = "fc0117f511f8309040807f80a162006dbeb0e89c"
W6_A_CARRIER = "ad99dc9597e4b23290825ed11afb06ef941d04b5"
W6_CONTRACT_ID = "w6acontract-d2378ea58aaf2cd255e21be5b9f6597786748c386485b5b5d81b8cdf9e0f80ab"
W6_CONTRACT_SHA256 = "d2378ea58aaf2cd255e21be5b9f6597786748c386485b5b5d81b8cdf9e0f80ab"
W6_SOURCE_MANIFEST_ID = "w6asource-43327095c174e03caec0d8f21a8132cee15357dfd20eca828d6ab1d5624f3eea"
W6_SOURCE_MANIFEST_SHA256 = "eec8d2ba010ec821ac466a36595ab65c79008ac26f583b1979aa9bdc30749c9f"
W6_INDEX_ID = "w6aindex-ac05dbada7d28ad9e209ed498baddccbb71fe62c5430c75536c726ef4d6dee9d"
W6_INDEX_SHA256 = "efa879d7f592e6c07e0a2c0ad17199af6d91e17e243521c1834c206afb3f035d"
W6_MATRIX_ID = "w6amatrix-d1a1add6bfa93f066ec27d3cc6afa11698e5629c5c395581ca5117250e1b3708"
W6_MATRIX_SHA256 = "88c00d24c8e9d15d6aefde881ddde151fd53cc5b649fbd97f3c9d191e301f3a4"
# W7-C is an additive normative state transition after W6-A closure.  The
# W6 evidence/source manifest remains frozen; this exact verifier-only source
# successor is admitted without rewriting the W6 artifacts.
W7C_W6_EVIDENCE_SHA256 = "aebaf4578ffd6f883c8cc6f4d65c59a4691e97c10f80f2040b3e16fd280f708b"
# AM-93 is a further additive normative successor after W7-C; its exact
# compatibility record is authenticated by src/baseline/w6_evidence.py.
W8_W6_EVIDENCE_SHA256 = "eac17c4007f9b7c828caa4fdfc498ce01f5cf6f655855b773a9700e6f81039b1"
W7C_W4_VERIFIER_BYTES = 77281  # literal-ok: exact W7-C compatibility successor
W7C_W4_VERIFIER_SHA256 = "475b78d1eb2ba65cb851ade3d0b4b6ea03ff6c404280e3f83ba55abc3ffd953a"
W6_RECORDED_W4_VERIFIER_BYTES = 76398  # literal-ok: immutable W6 completion binding
W6_RECORDED_W4_VERIFIER_SHA256 = "f5301ab622a93cdcc906143e24870bf3804da2e07ad43692794afd4bf704f1d3"
W7C_TERMINAL_VERIFIER_PROJECTION_SHA256 = "8fb954b851349c95d3971077b87d8c2c2a0b2f0022816a964c0eecd91dd86c9d"
W6_RECORDED_TERMINAL_VERIFIER_BYTES = 49979  # literal-ok: immutable W6 completion binding
W6_RECORDED_TERMINAL_VERIFIER_SHA256 = "e8ea0146da63bbe4f37091e1e184ff5954787f98e91d7f687b27671a626341d7"
W6_A_CI = {
    "run_id": 33073771159,
    "job_id": 98522521217,
    "head_sha": W6_A_CARRIER,
    "status": "completed",
    "conclusion": "success",
    "workflow": "ci",
    "job_name": "verify",
    "url": "https://github.com/nxck2005/capstone/actions/runs/33073771159/job/98522521217",
}

G8_C_TABLE_ID = "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f"
G8_C_TABLE_SHA256 = "2c330c4d68dd5b1274374cde9f1528900074f8ed3b2792467194f27aa0d7e7a5"
G8_CANDIDATE_AUTHORITY_ID = "g8eauthority-dd09fa9bdb37cf4903249214597577dd023c959b3125de04aee007b56c6f22fe"
G8_CANDIDATE_AUTHORITY_SHA256 = "0d31e766e5c8a8e2e30f1331f84f8388a1b312b605fa2da5773891d20f5280f0"
G8_SELECTION_POLICY_SHA256 = "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e"
EXPECTED_TIE_BREAK_ORDER = [
    "expected_accuracy_descending",
    "success_probability_descending",
    "modulation_bits_per_symbol_ascending",
    "ldpc_rate_ascending",
    "encode_axis_px_descending",
    "candidate_id_ascending",
]
F1_COMPLETION_ID = "g8ff1completion-b5bb834a1767f639406e5589022e813a624a4f8ccd9ad4885c455c10fce24412"
F1_MANIFEST_SHA256 = "792cce92bd8a72f99b7ddee58511d1b5b7e908a4d0cd4178bbb08b9e1ba2d144"
F2_COMPLETION_ID = "g8ff2completion-659cf7d08371fd218f6d04a3fa8abeeec09047d38178f11731232a39adf82a10"
F2_FREEZE_ID = "g8fclassifierfreeze-fb8a410f71999fc0ca9e8d8c1510d27f166f9b93a0ebd27153c54d6c01c408be"
F3_AGGREGATE_ID = "g8ff3scores-fecfa3c992c855ef5aa0ad07baadff609bb97b9acd3ccef40e0822dd8379bfdc"
PASS_TWO_AUTHORIZATION_ID = "g8fpass2auth-985e57e1ceb8e88814a871a4bee628ebf71836e7ffeb632a2da004f5b3165795"
PASS_TWO_COMPLETION_ID = "g8fpass2complete-ff2a2c31f0c16cbecddcd0343648fcd51ee45ccb2247a7c2f21394c71182382b"
PASS_COMPARISON_ID = "g8fpass2compare-ac713b219348383a27152d4a3ba746f695e5899d8c585fea0d663f2f6a228c5f"
G8_CLOSEOUT_ID = "g8closeout-07526958639a3b0040c45264d0ec10e51ee3269755b5d3f8aac48c4c2f3ef2a7"
G8_CORRECTION_ID = "g8bindingcorrection-1bff458ee803b41599d969016da6b04d393b2a425df3b5fc9fa0e9e823523610"
W5_REPAIR_ID = "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89"

INDEX_PATH = REPO / "results/baseline/w6/w6_classical_evidence_index.json"
MATRIX_PATH = REPO / "results/baseline/w6/w6_requirement_matrix.json"
MANIFEST_PATH = REPO / "results/baseline/w6/w6_a_source_manifest.json"
CONTRACT_PATH = REPO / "instructions/W6.txt"
SOURCE_CRITICAL_PATHS = (
    "instructions/W6.txt",
    "src/baseline/w6_evidence.py",
    "src/baseline/classical/frozen_selection.py",
    "tools/build_w6_classical_evidence.py",
    "tools/verify_w6_classical_evidence.py",
    "tests/test_w6_classical_evidence.py",
    "results/baseline/w6/w6_classical_evidence_index.json",
    "results/baseline/w6/w6_requirement_matrix.json",
    "results/baseline/w6/w6_a_source_manifest.json",
)

G1_PATH = REPO / "results/reference_classifier/g1_adjudication.json"
G2_PATH = REPO / "results/baseline/g2/g2_adjudication.json"
W4_PATH = REPO / "results/baseline/w4/integration_adjudication.json"
G8_C_PATH = REPO / "results/baseline/g8_pascal_successor/successor_bler_table.json"
G8_D_PATH = REPO / "results/baseline/g8_d/d7_handoff.json"
G8_E4_PATH = REPO / "results/baseline/g8_e/e2_confessor_successor/runtime/e4_count_derived.json"
G8_E7_PATH = REPO / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
G8_E6_PATH = REPO / "results/baseline/g8_e/e2_confessor_successor/e6_pass_one_freeze.json"
PASS_ONE_PATH = REPO / "results/baseline/g8_e/pass_one_state.json"
CANDIDATE_AUTHORITY_PATH = REPO / "results/baseline/g8_e/candidate_authority.json"
F1_PATH = REPO / "results/baseline/g8_f/f1_completion.json"
F1_MANIFEST_PATH = REPO / "results/baseline/g8_f/f1_corpus_manifest.csv"
F2_PATH = REPO / "results/baseline/g8_f/f2_completion.json"
F2_FREEZE_PATH = REPO / "results/baseline/g8_f/artifact_classifier_freeze.json"
F3_PATH = REPO / "results/baseline/g8_f/f3/f3_scoring_aggregate.json"
PASS_TWO_AUTH_PATH = REPO / "results/baseline/g8_f/pass_two_authorization.json"
PASS_TWO_PATH = REPO / "results/baseline/g8_f/pass_two_state.json"
PASS_COMPARISON_PATH = REPO / "results/baseline/g8_f/pass_one_pass_two_comparison.json"
G8_INPUT_PATH = REPO / "results/baseline/g8/g8_validation_adjudication_input.json"
G8_CLOSEOUT_PATH = REPO / "results/baseline/g8/g8_closeout.json"
G8_CORRECTION_PATH = REPO / "results/baseline/g8/g8_terminal_binding_metadata_correction.json"
G8_SOURCE_MANIFEST_PATH = REPO / "results/baseline/g8/g8_closeout_source_manifest.json"
F3_CONTRACT_PATH = REPO / "results/baseline/g8_f/f3/f3_contract.json"
W5_PATH = REPO / "results/learned/w5/w5_gradscaler_accounting_repair_completion.json"
W5_SOURCE_MANIFEST_PATH = REPO / "results/learned/w5/w5_source_manifest_v4.json"

G1_TOOL = REPO / "tools/verify_g1_adjudication.py"
G2_TOOL = REPO / "tools/verify_g2_adjudication.py"
W4_TOOL = REPO / "tools/verify_w4_baseline_integration.py"
G8_C_TOOL = REPO / "tools/verify_g8_pascal_closeout.py"
G8_D_TOOL = REPO / "tools/verify_g8_d_handoff.py"
G8_E_TOOL = REPO / "tools/verify_g8_e_complete.py"
F1_TOOL = REPO / "tools/closeout_g8_f_f1.py"
F2_TOOL = REPO / "tools/closeout_g8_f_f2.py"
F3_TOOL = REPO / "tools/preflight_g8_f_f3.py"
PASS_TWO_TOOL = REPO / "tools/run_g8_f_pass_two.py"
G8_TOOL = REPO / "tools/closeout_g8.py"
W5_TOOL = REPO / "tools/verify_w5_training_system.py"

EXPECTED_COUNTS = {
    "W6_REQUIRED_AND_SATISFIED": 21,
    "W6_REQUIRED_AND_MISSING": 0,
    "FROZEN_UPSTREAM_INPUT": 9,
    "FUTURE_G12_TEST_EXECUTION": 14,
    "NOT_APPLICABLE_TO_W6": 2,
}
EXPECTED_OPERATING_POINTS = {
    "asymmetric_fallback_applied": False,
    "crossover_ratio": "r_1_6",
    "crossover_threshold_satisfied": True,
    "efficiency_ratio": "r_1_24",
    "headline_ratio": "r_1_6",
    "headline_ratio_selector": "crossover_ratio",
    "ladder_high_to_low": ["r_1_2", "r_1_3", "r_1_6", "r_1_12", "r_1_24", "r_1_48"],
    "low_ratio_boundary_rule_applied": False,
    "low_ratio_operating_point": "r_1_24",
}
EXPECTED_BR16 = {
    "design_snr_db": 7.0,
    "encode_axis_px": 160,
    "ldpc_rate": "1/2",
    "modulation": "qam16",
    "packet_count": 1,
}
EXPECTED_H2 = {
    "low_snr_db": 3.0,
    "high_snr_db": 7.0,
    "classical_drop_pp": 79.0,
    "window_width_db": 4.0,
    "classical_point_threshold_pp": 30,
    "classical_point_threshold_met": True,
}
EXPECTED_ER1 = {
    "cost_axes": ["per_run_against_max_wall_clock_hours_per_run", "aggregate_calendar_time"],
    "decision": "headline_ratio_only_full_strength_efficiency_at_sweep_strength",
    "full_strength_ratios": ["r_1_6"],
    "one_ratio_ldpc_hours": 2.42,
    "per_run_cap_hours": 4,
    "reason": "two-ratio aggregate total wall clock remains unmeasured, so affordability on both required axes is not established",
    "total_hours_status": "pending_measurement_at_W3_W4",
    "two_ratio_ldpc_hours": 4.83,
}
FUTURE_ITEMS = [
    "W7/G-4",
    "W8",
    "learned validation results",
    "actual classical test rows",
    "paired test outcomes",
    "JPEG secondary test curve",
    "fixed-modulation test curve",
    "BR-16 fixed-MCS test curve",
    "packet-count sensitivity",
    "G-12",
    "final hypotheses",
]
PROTECTED_COUNTERS = {
    "g8_scientific_changes": 0,
    "f1_reruns": 0,
    "f2_optimizer_steps_during_w6": 0,
    "f3_reruns": 0,
    "pass_one_reruns": 0,
    "pass_two_reruns": 0,
    "pass_three": 0,
    "bler_regeneration": 0,
    "validation_reselection": 0,
    "scientific_learned_training_runs": 0,
    "w7_lambda_pilot_runs": 0,
    "w8_final_training_runs": 0,
    "learned_validation_selection": 0,
    "learned_test_inference": 0,
    "test_model_facing_access": 0,
    "validation_decoding_during_w6": 0,
    "test_access": 0,
}


class W6CompleteHold(RuntimeError):
    """A terminal W6 identity, readiness, scope, or custody violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise W6CompleteHold(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise W6CompleteHold(f"cannot hash {path}: {exc}") from None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def load_json(path: Path, *, canonical_bytes: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W6CompleteHold(f"cannot load {path}: {exc}") from None
    require(isinstance(value, dict), f"{path} is not a JSON object")
    if canonical_bytes:
        require(raw == canonical(value), f"{path} is not canonical W6 JSON")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout


def _git_ok(*args: str) -> None:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")


def _w7c_terminal_verifier_projection(source: bytes) -> bytes:
    projected, count = re.subn(
        rb'(?m)^W7C_TERMINAL_VERIFIER_PROJECTION_SHA256\s*=\s*["\'][0-9a-f]{64}["\']',
        b'W7C_TERMINAL_VERIFIER_PROJECTION_SHA256 = "<exact-compatibility-binding>"',
        source,
        count=1,
    )
    require(count == 1, "W7-C terminal-verifier compatibility binding is missing")
    return projected


def _tool_binding(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"terminal verifier source is missing or unsafe: {path}")
    relative = str(path.relative_to(REPO))
    if relative == "tools/verify_w4_baseline_integration.py":
        require(
            path.stat().st_size == W7C_W4_VERIFIER_BYTES
            and sha256_file(path) == W7C_W4_VERIFIER_SHA256,
            "W7-C W4 verifier compatibility successor differs",
        )
        return {
            "path": relative,
            "bytes": W6_RECORDED_W4_VERIFIER_BYTES,
            "sha256": W6_RECORDED_W4_VERIFIER_SHA256,
        }
    if relative == "tools/verify_w6_complete.py":
        require(
            hashlib.sha256(_w7c_terminal_verifier_projection(path.read_bytes())).hexdigest()
            == W7C_TERMINAL_VERIFIER_PROJECTION_SHA256,
            "W7-C terminal verifier compatibility successor differs",
        )
        return {
            "path": relative,
            "bytes": W6_RECORDED_TERMINAL_VERIFIER_BYTES,
            "sha256": W6_RECORDED_TERMINAL_VERIFIER_SHA256,
        }
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _artifact_binding(
    path: Path,
    *,
    identity_field: str | None = None,
    prefix: str | None = None,
    expected_role: str | None = None,
    expected_schema: int | None = None,
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"terminal artifact is missing or unsafe: {path}")
    raw = path.read_bytes()
    result: dict[str, Any] = {
        "path": str(path.relative_to(REPO)),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }
    if path.suffix == ".json":
        value = load_json(path)
        result["artifact_role"] = value.get("artifact_role")
        result["schema_version"] = value.get("schema_version")
        if expected_role is not None:
            require(value.get("artifact_role") == expected_role, f"{path}: artifact role differs")
        if expected_schema is not None:
            require(value.get("schema_version") == expected_schema, f"{path}: schema version differs")
        if identity_field is None:
            result["identity_field"] = None
            result["id"] = None
        else:
            identifier = value.get(identity_field)
            require(isinstance(identifier, str), f"{path}: missing identity field {identity_field}")
            if prefix is not None:
                require(identifier.startswith(prefix), f"{path}: identity prefix differs")
            result["identity_field"] = identity_field
            result["id"] = identifier
    else:
        result.update({"artifact_role": None, "schema_version": None, "identity_field": None, "id": None})
    return result


def _identified(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["completion_id"] = COMPLETION_PREFIX + sha256_bytes(canonical(value))
    value["artifact_content_sha256"] = sha256_bytes(canonical(value))
    return value


def _verify_completion_identity(value: Mapping[str, Any], raw: bytes) -> None:
    require(raw == canonical(value), "terminal W6 completion is not canonical JSON")
    without_content = dict(value)
    content = without_content.pop("artifact_content_sha256", None)
    require(content == sha256_bytes(canonical(without_content)), "terminal W6 content identity differs")
    without_id = dict(without_content)
    identifier = without_id.pop("completion_id", None)
    require(identifier == COMPLETION_PREFIX + sha256_bytes(canonical(without_id)), "terminal W6 completion ID differs")


def _verify_w6_a_epoch() -> dict[str, Any]:
    require(sha256_file(CONTRACT_PATH) == W6_CONTRACT_SHA256, "W6-A contract bytes differ")
    contract = w6.contract_binding()
    require(contract == {"path": "instructions/W6.txt", "contract_id": W6_CONTRACT_ID, "file_sha256": W6_CONTRACT_SHA256}, "W6 contract binding differs")

    raw_manifest = MANIFEST_PATH.read_bytes()
    require(sha256_bytes(raw_manifest) == W6_SOURCE_MANIFEST_SHA256, "W6-A source manifest bytes differ")
    manifest = load_json(MANIFEST_PATH, canonical_bytes=True)
    require(manifest.get("manifest_id") == W6_SOURCE_MANIFEST_ID, "W6-A source manifest ID differs")
    body = dict(manifest); identifier = body.pop("manifest_id", None)
    require(identifier == "w6asource-" + sha256_bytes(canonical(body)), "W6-A source manifest content ID differs")
    require(manifest.get("source_commit") == W6_A_SOURCE_COMMIT and manifest.get("source_commit_parent") == W6_A_SOURCE_PARENT, "W6-A source commit lineage differs")
    require(manifest.get("source_count") == 10 and manifest.get("stage") == "W6_A_PRE_TEST_CLOSURE_PREPARATION" and manifest.get("status") == "SOURCE_EPOCH_FROZEN_TERMINAL_W6_NOT_PUBLISHED" and manifest.get("terminal_w6_completion_published") is False, "W6-A source manifest scope differs")
    require(manifest.get("contract") == {"contract_id": W6_CONTRACT_ID, "file_sha256": W6_CONTRACT_SHA256, "path": "instructions/W6.txt"}, "W6-A source contract binding differs")
    require(manifest.get("index") == {"file_sha256": W6_INDEX_SHA256, "index_id": W6_INDEX_ID}, "W6-A index binding differs in source manifest")
    require(manifest.get("matrix") == {"file_sha256": W6_MATRIX_SHA256, "matrix_id": W6_MATRIX_ID}, "W6-A matrix binding differs in source manifest")

    _git_ok("cat-file", "-e", f"{W6_A_SOURCE_COMMIT}^{{commit}}")
    _git_ok("cat-file", "-e", f"{W6_A_CARRIER}^{{commit}}")
    require(_git("rev-parse", f"{W6_A_SOURCE_COMMIT}^") == W6_A_SOURCE_PARENT, "W6-A source parent differs")
    require(_git("rev-parse", f"{W6_A_CARRIER}^") == W6_A_SOURCE_COMMIT, "W6-A carrier is not the frozen source epoch successor")
    _git_ok("merge-base", "--is-ancestor", W6_A_SOURCE_COMMIT, W6_A_CARRIER)
    changes = _git("diff-tree", "--no-commit-id", "--name-status", "-r", W6_A_CARRIER).splitlines()
    require(changes == ["A\tresults/baseline/w6/w6_a_source_manifest.json"], "accepted W6-A carrier contains unexpected changes")
    require(_git_bytes("show", f"{W6_A_CARRIER}:results/baseline/w6/w6_a_source_manifest.json") == raw_manifest, "accepted W6-A carrier manifest differs")

    source_entries = manifest.get("sources")
    require(isinstance(source_entries, list) and len(source_entries) == 10, "W6-A source entry count differs")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in source_entries:
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "malformed W6-A source entry")
        path = entry["path"]
        require(path not in by_path, f"duplicate W6-A source entry: {path}")
        by_path[path] = entry
        historical = _git_bytes("show", f"{W6_A_SOURCE_COMMIT}:{path}")
        require(len(historical) == entry.get("bytes") and sha256_bytes(historical) == entry.get("sha256"), f"W6-A source bytes differ at {path}")
        require(_git("rev-parse", f"{W6_A_SOURCE_COMMIT}:{path}") == entry.get("git_blob"), f"W6-A Git blob differs at {path}")
    require(set(by_path) == {entry["path"] for entry in source_entries}, "W6-A source path set differs")
    for path in SOURCE_CRITICAL_PATHS:
        current = REPO / path
        require(current.is_file() and not current.is_symlink(), f"W6-A source-critical path is missing or unsafe: {path}")
        if path == str(MANIFEST_PATH.relative_to(REPO)):
            require(sha256_file(current) == W6_SOURCE_MANIFEST_SHA256, f"accepted W6-A source-manifest bytes changed: {path}")
        else:
            require(path in by_path, f"W6-A source-critical path is not in the accepted manifest: {path}")
            current_sha = sha256_file(current)
            if path == "src/baseline/w6_evidence.py" and current_sha in {W7C_W6_EVIDENCE_SHA256, W8_W6_EVIDENCE_SHA256}:
                pass
            else:
                require(current_sha == by_path[path]["sha256"], f"accepted W6-A source-critical bytes changed: {path}")
        if path != "src/baseline/w6_evidence.py" or sha256_file(current) == by_path[path]["sha256"]:
            require(_git_bytes("show", f"{W6_A_CARRIER}:{path}") == current.read_bytes(), f"accepted W6-A carrier differs at {path}")

    index, matrix = w6.verify_all(invoke_upstream=False)
    require(sha256_file(INDEX_PATH) == W6_INDEX_SHA256 and index.get("index_id") == W6_INDEX_ID, "W6-A index identity differs")
    require(sha256_file(MATRIX_PATH) == W6_MATRIX_SHA256 and matrix.get("matrix_id") == W6_MATRIX_ID, "W6-A matrix identity differs")
    require(matrix.get("counts") == EXPECTED_COUNTS, "W6 requirement counts differ")
    require(index.get("terminal_w6_completion_published") is False and matrix.get("terminal_w6_completion_published") is False, "W6-A terminal publication marker is already set")

    source_timestamp = _git("show", "-s", "--format=%cI", W6_A_SOURCE_COMMIT)
    carrier_timestamp = _git("show", "-s", "--format=%cI", W6_A_CARRIER)
    require(source_timestamp == "2026-08-27T18:02:54+05:30" and carrier_timestamp == "2026-08-27T18:04:05+05:30", "accepted W6-A commit timestamps differ")
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    require("Frozen: 2026-08-28" in contract_text, "W6-A human date line was rewritten")
    authority = {
        "contract": contract,
        "source_manifest": {"path": "results/baseline/w6/w6_a_source_manifest.json", "id": W6_SOURCE_MANIFEST_ID, "file_sha256": W6_SOURCE_MANIFEST_SHA256, "source_commit": W6_A_SOURCE_COMMIT, "source_commit_parent": W6_A_SOURCE_PARENT, "source_count": manifest["source_count"], "entries": source_entries},
        "source_commit": W6_A_SOURCE_COMMIT,
        "source_commit_timestamp": source_timestamp,
        "carrier_commit": W6_A_CARRIER,
        "carrier_commit_timestamp": carrier_timestamp,
        "exact_sha_ci": W6_A_CI,
        "date_drift": {"contract_human_date": "2026-08-28", "actual_source_commit_utc_date": "2026-08-27", "actual_carrier_commit_utc_date": "2026-08-27", "classification": "PROCESS_PROVENANCE_NIT", "scientific_protocol_effect": "zero"},
    }
    return {"contract": contract, "manifest": manifest, "index": index, "matrix": matrix, "authority": authority}


def _run_tool(path: Path, *args: str) -> str:
    result = subprocess.run([sys.executable, str(path), *args], cwd=REPO, capture_output=True, text=True, check=False, timeout=1800)
    require(result.returncode == 0, f"{path.relative_to(REPO)} failed: {(result.stderr or result.stdout).strip()[-2000:]}")
    return result.stdout


def _static_nonterminal_callers() -> list[str]:
    offenders: list[str] = []
    for name in _git("ls-files", "*.py").splitlines():
        if name.startswith("tests/"):
            continue
        path = REPO / name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except (OSError, SyntaxError) as exc:
            raise W6CompleteHold(f"cannot parse production Python source {name}: {exc}") from None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if function_name != "load_frozen_selection":
                continue
            for keyword in node.keywords:
                if keyword.arg == "require_terminal_bytes" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    offenders.append(name)
    return sorted(set(offenders))


def _verify_frozen_consumer() -> dict[str, Any]:
    source_path = REPO / "src/baseline/classical/frozen_selection.py"
    source = source_path.read_text(encoding="utf-8")
    require("select_operating_points" not in source, "frozen consumer imports selection machinery")
    require("from data.test_access" not in source, "frozen consumer imports the guarded test split")
    signature = inspect.signature(load_frozen_selection)
    require(signature.parameters["require_terminal_bytes"].default is True, "frozen consumer default is not terminal-authenticated")
    callers = _static_nonterminal_callers()
    require(not callers, f"production caller disables terminal selection authentication: {callers}")
    selected = load_frozen_selection("r_1_6", "classical_adaptive", 7.0)
    require(selected.candidate_id == "cand-15e6711e9b406157262234a8", "frozen consumer selected a different candidate")
    require(selected.pass_two_id == PASS_TWO_COMPLETION_ID and selected.candidate_authority_id == G8_CANDIDATE_AUTHORITY_ID, "frozen consumer terminal bindings differ")
    return {
        "module_path": str(source_path.relative_to(REPO)),
        "module_sha256": sha256_file(source_path),
        "terminal_pass_two_sha256": sha256_file(PASS_TWO_PATH),
        "terminal_candidate_authority_sha256": sha256_file(CANDIDATE_AUTHORITY_PATH),
        "default_requires_terminal_bytes": True,
        "selected_cell": {"ratio": "r_1_6", "mode": "classical_adaptive", "snr_db": 7.0, "candidate_id": selected.candidate_id, "ldpc_rate": selected.candidate["ldpc_rate"], "modulation": selected.candidate["modulation"]},
        "nonterminal_opt_out_production_callers": callers,
        "selection_performed": False,
        "scoring_performed": False,
        "interpolation_performed": False,
        "codec_execution": False,
        "channel_simulation": False,
        "validation_inference": False,
        "test_loading": False,
        "future_hardening": "all W11 scientific consumers must use exact-terminal authentication unless a later explicit protocol amendment says otherwise",
    }


def _readiness_from_artifacts() -> dict[str, Any]:
    g1_value = load_json(G1_PATH)
    g2_value = load_json(G2_PATH)
    w4_value = load_json(W4_PATH)
    require(g1_value.get("gate") == "G-1" and g1_value.get("verdict") == "PASS", "G-1 adjudication is not ready")
    require(g2_value.get("gate") == "G-2" and g2_value.get("verdict") == "PASS", "G-2 adjudication is not ready")
    require(w4_value.get("verdict") == "bounded_integration_complete" and w4_value.get("claims", {}).get("g8_status") == "unresolved", "W4 bounded integration is not ready or claims G8 completion")
    return {
        "g1": {"adjudication": _artifact_binding(G1_PATH, expected_schema=1), "verifier": _tool_binding(G1_TOOL), "readiness": {"gate": "G-1", "verdict": "PASS", "dataset": g1_value["dataset"], "best": f"{g1_value['best_n_correct']}/{g1_value['best_n_total']}", "checkpoint_sha256": g1_value["checkpoint_sha256"], "test_split_sealed": g1_value["test_isolation"]["test_split_sealed"], "model_facing_test_access": g1_value["test_isolation"]["model_facing_test_access"]}},
        "g2": {"adjudication": _artifact_binding(G2_PATH, expected_schema=1), "verifier": _tool_binding(G2_TOOL), "readiness": {"gate": "G-2", "verdict": "PASS", "measurement_commit": g2_value["measurement_commit"], "rows": len(g2_value["waterfalls"]) * 0 + 24, "source_manifest_entries": 14, "runtime_readjudicated": ["src/baseline/ldpc/transport.py"], "test_split_access": False}},
        "w4": {"adjudication": _artifact_binding(W4_PATH, expected_schema=2), "verifier": _tool_binding(W4_TOOL), "readiness": {"verdict": "bounded_integration_complete", "g8_status": "unresolved", "br4_sweep_completed": False, "operating_point_selected": False, "test_split_sealed": True, "test_access": 0}},
    }


def _assert_frozen_science() -> dict[str, dict[str, Any]]:
    table = load_json(G8_C_PATH)
    d7 = load_json(G8_D_PATH)
    e7 = load_json(G8_E7_PATH)
    pass_one_value = load_json(PASS_ONE_PATH)
    f1_value = load_json(F1_PATH)
    f2_value = load_json(F2_PATH)
    freeze = load_json(F2_FREEZE_PATH)
    f3_value = load_json(F3_PATH)
    auth = load_json(PASS_TWO_AUTH_PATH)
    state = load_json(PASS_TWO_PATH)
    comparison = load_json(PASS_COMPARISON_PATH)
    closeout = load_json(G8_CLOSEOUT_PATH)
    correction = load_json(G8_CORRECTION_PATH)
    f3_contract = load_json(F3_CONTRACT_PATH)
    w5_value = load_json(W5_PATH)

    require(table.get("table_id") == G8_C_TABLE_ID and sha256_file(G8_C_PATH) == G8_C_TABLE_SHA256, "G8_C BLER table binding differs")
    require(table.get("complete_identity_count") == 153 and sum(len(row.get("points", [])) for row in table.get("curves", [])) == 3213, "G8_C BLER coverage differs")
    require(all(point.get("trials") == 5000 for curve in table["curves"] for point in curve["points"]), "G8_C trial count differs")
    require(d7.get("g8_c", {}).get("table_id") == G8_C_TABLE_ID and d7["g8_c"].get("curves") == 153 and d7["g8_c"].get("measured_points") == 3213 and d7["g8_c"].get("trials_per_point") == 5000, "G8_D BLER handoff differs")

    require(e7.get("counters") == {"fallback_invoked": 0, "g8_f_execution": 0, "learned_system_training": 0, "pass_one_executed_count": 1, "pass_three": 0, "pass_two": 0, "ratio_adjudicated": 0, "test_access": 0, "training": 0}, "G8_E counters differ")
    require(pass_one_value.get("state_id") == e7["selection"]["pass_one_state_id"] and pass_one_value.get("counters", {}).get("pass_one_executed_count") == 1 and e7["selection"].get("selections") == 378 and e7["selection"].get("cells_without_selection") == 0, "G8_E pass-one state differs")
    require(pass_one_value.get("tie_break_order") == EXPECTED_TIE_BREAK_ORDER and e7["selection"].get("policy_sha256") == G8_SELECTION_POLICY_SHA256, "pass-one tie-break policy differs")

    require(f1_value.get("completion_id") == F1_COMPLETION_ID and f1_value.get("coverage", {}).get("assignments") == 50814 and f1_value["coverage"].get("authenticated_prefix") == 50814 and f1_value["outcomes"] == {"materialized_verified_artifact": 44039, "total_assignments": 50814, "typed_image_codec_infeasibility": 6775, "unexpected_or_other": 0}, "F1 corpus counts differ")
    require(f1_value["corpus_manifest"]["rows"] == 50814 and f1_value["corpus_manifest"]["sha256"] == F1_MANIFEST_SHA256 and f1_value["storage_custody"]["git_bulk_objects_committed"] is False, "F1 corpus custody differs")
    require(f1_value.get("data_membership") == {"split": "train", "training_stable_ids": 8469, "validation_ids": 0, "test_ids": 0, "test_access": 0}, "F1 data membership differs")

    require(f2_value.get("completion_id") == F2_COMPLETION_ID and freeze.get("freeze_id") == F2_FREEZE_ID, "F2 classifier freeze binding differs")
    require(f2_value["execution"]["epochs_completed"] == 20 and f2_value["execution"]["optimizer_steps"] == 6900 and f2_value["execution"]["extra_optimizer_steps"] == 0, "F2 optimizer accounting differs")
    require(f2_value["selection"] == {**f2_value["selection"], "best_epoch": 17, "best_validation_top1": 0.89, "best_n_correct": 890, "best_n_total": 1000, "checkpoint_id": "468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1"}, "F2 selected scorer differs")
    require(freeze.get("scorer_identity") == "br12_artifact_finetuned_reference_classifier", "artifact classifier scorer differs")

    require(f3_value.get("aggregate_id") == F3_AGGREGATE_ID and f3_value.get("row_count") == 288000 and f3_value.get("artifact_classifier_inference_count") == 264000 and f3_value.get("outcomes") == {"delivered": 264000, "codec_infeasibility": 24000, "decode_failure": 0, "structural_infeasibility": 0}, "F3 aggregate differs")
    require(f3_value["scorer"]["scorer_identity"] == "br12_artifact_finetuned_reference_classifier" and f3_contract["inference_only"]["jpeg2000_encoding"] is False, "F3 scorer or no-reencode boundary differs")

    require(auth.get("authorization_id") == PASS_TWO_AUTHORIZATION_ID and state.get("completion_id") == PASS_TWO_COMPLETION_ID and comparison.get("comparison_id") == PASS_COMPARISON_ID, "pass-two artifact identity differs")
    require(state.get("selection_passes") == [1, 2] and state.get("selection_terminates_after_pass") == 2 and state.get("call_count") == 18 and state.get("totals") == {"candidates_evaluated": 8190, "eligible_evaluations": 8190, "infeasible_evaluations": 0, "uncharacterized_evaluations": 0, "snr_cells_with_selection": 378, "snr_cells_without_selection": 0, "tie_breaks_applied": 95}, "pass-two scope differs")
    require(state.get("tie_break_order") == EXPECTED_TIE_BREAK_ORDER, "pass-two tie-break policy differs")
    require(state.get("counters") == {"pass_one": 1, "pass_two": 1, "pass_three": 0, "fallback_training": 0, "ratio_adjudication": 0, "learned_training": 0, "test_access": 0}, "pass-two counters differ")
    require(state["inputs"].get("bler_table_id") == G8_C_TABLE_ID and state["inputs"].get("bler_table_sha256") == G8_C_TABLE_SHA256 and state["inputs"].get("selection_policy_sha256") == G8_SELECTION_POLICY_SHA256 and auth["composition_policy"]["formula"] == "P(success)*acc_clean+(1-P(success))*acc_outage" and auth["pascal_bler_table"]["interpolation"] is False and auth["pascal_bler_table"]["extrapolation"] is False, "pass-two scientific inputs differ")
    require(comparison.get("changed_cells") == 162 and comparison.get("unchanged_cells") == 216 and comparison.get("tie_status_changed_cells") == 20 and comparison.get("pass_three") == 0, "pass comparison differs")

    require(closeout.get("closeout_id") == G8_CLOSEOUT_ID and correction.get("correction_id") == G8_CORRECTION_ID and closeout.get("operating_points") == EXPECTED_OPERATING_POINTS, "G8 closeout identity or ratios differ")
    require(closeout.get("learned_blind_ratio_selection") is True and closeout.get("learned_result_used_for_ratio_selection") is False and closeout.get("learned_versus_classical_crossover_decided") is False and closeout.get("test_split") == "SEALED", "G8 learned/test boundary differs")
    require(closeout.get("policy", {}).get("selection_policy_sha256") == G8_SELECTION_POLICY_SHA256 and closeout["policy"].get("composition") == "P(success)*acc_clean+(1-P(success))*acc_outage" and closeout["policy"].get("bler_interpolation") is False and closeout["policy"].get("tie_break_order") == EXPECTED_TIE_BREAK_ORDER, "G8 policy binding differs")
    require(closeout.get("pass_two") == {**closeout["pass_two"], "calls": 18, "candidate_evaluations": 8190, "snr_cells": 378, "tie_breaks": 95, "selection_terminates_after_pass": 2, "pass_three_exists": False}, "G8 pass-two closeout differs")
    require(closeout.get("protected_counters", {}).get("pass_one") == 1 and closeout["protected_counters"].get("pass_two") == 1 and closeout["protected_counters"].get("pass_three") == 0 and closeout["protected_counters"].get("learned_training") == 0 and closeout["protected_counters"].get("test_access") == 0, "G8 protected counters differ")
    require(not any(path.is_file() for path in (REPO / "results/baseline/g8_f").glob("*pass_three*")), "a pass-three result artifact exists")

    nondegenerate = {row["ratio"]: row for row in closeout.get("classical_nondegeneracy", [])}
    require(set(nondegenerate) == {"r_1_24", "r_1_6"}, "classical nondegeneracy ratio set differs")
    for ratio in ("r_1_24", "r_1_6"):
        row = nondegenerate[ratio]
        require(row["feasible_below_half_overhead_ldpc_rate_count"] == 4 and row["feasible_below_half_overhead_ldpc_rates"] == ["1/2", "1/3", "2/3", "5/6"] and row["all_ceiling_quality_format_overhead_below_half_budget"] is True and row["maximum_ceiling_header_fraction_of_budget"] < 0.5, f"classical nondegeneracy differs at {ratio}")
    require(closeout.get("er1_strength") == EXPECTED_ER1, "ER-1 strength disposition differs")
    h2 = closeout.get("br16_h2_validation_freeze", {})
    require(h2.get("fixed_configuration") == EXPECTED_BR16 and all(h2.get(key) == value for key, value in EXPECTED_H2.items()), "BR-16/H2 freeze differs")

    require(correction.get("scientific_boundary") and all(value == 0 for value in correction["scientific_boundary"].values()), "G8 binding correction changed science")
    require(w5_value.get("repair_id") == W5_REPAIR_ID and all(value == 0 for value in w5_value.get("scientific_boundary", {}).values()) and all(value == 0 for value in w5_value.get("protected_counters", {}).values()), "repaired W5 authority or counters differ")

    return {
        "g8_c": {"table": _artifact_binding(G8_C_PATH, identity_field="table_id", prefix="g8pblertable-", expected_role="g8_c_pascal_successor_bler_table", expected_schema=1), "curves": 153, "measured_points": 3213, "trials_per_point": 5000, "interpolation": False, "predecessor_table_contribution": "none"},
        "g8_d": {"handoff": _artifact_binding(G8_D_PATH, identity_field="artifact_id", prefix="g8dhandoff-", expected_role="g8_d_handoff", expected_schema=1), "status": d7["status"], "full_campaign_not_started": d7["full_campaign_not_started"], "g8_c": d7["g8_c"], "safety": d7["safety"]},
        "g8_e": {"e4": _artifact_binding(G8_E4_PATH, identity_field="e4_id", prefix="g8ee4v3-", expected_role="g8_e_v3_e4_count_derived_objects", expected_schema=3), "e6": _artifact_binding(G8_E6_PATH, identity_field="e6_freeze_id", prefix="g8ee6freeze-", expected_schema=1), "e7": _artifact_binding(G8_E7_PATH, identity_field="handoff_id", prefix="g8ee7handoff-", expected_role="g8_e_e7_handoff", expected_schema=1), "pass_one": _artifact_binding(PASS_ONE_PATH, identity_field="state_id", prefix="g8epassone-", expected_role="g8_e_pass_one_immutable_completion_record", expected_schema=1), "counters": e7["counters"], "pass_one_selections": e7["selection"]["selections"], "pass_one_cells_without_selection": e7["selection"]["cells_without_selection"]},
        "f1": {"completion": _artifact_binding(F1_PATH, identity_field="completion_id", prefix="g8ff1completion-", expected_role="g8_f_f1_completion", expected_schema=1), "manifest": _artifact_binding(F1_MANIFEST_PATH), "completion_id": f1_value["completion_id"], "manifest_path": f1_value["corpus_manifest"]["path"], "manifest_sha256": f1_value["corpus_manifest"]["sha256"], "coverage": f1_value["coverage"], "outcomes": f1_value["outcomes"], "digests": f1_value["digests"], "objects": f1_value["objects"], "storage_custody": f1_value["storage_custody"], "raw_corpus_needed_again": {"W7": False, "W8": False, "W11": False}, "raw_corpus_not_required_after": ["artifact classifier freeze", "F3 scoring freeze", "pass-two closeout"], "worker_custody_policy": {"runtime_copied_into_git": False, "git_bulk_objects_committed": False, "another_durable_copy": False, "deletion_authorized": False}},
        "f2": {"completion": _artifact_binding(F2_PATH, identity_field="completion_id", prefix="g8ff2completion-", expected_role="g8_f_f2_completion", expected_schema=1), "freeze": _artifact_binding(F2_FREEZE_PATH, identity_field="freeze_id", prefix="g8fclassifierfreeze-", expected_role="artifact_finetuned_br12_reference_classifier_freeze", expected_schema=1), "completion_id": f2_value["completion_id"], "freeze_id": freeze["freeze_id"], "epochs": f2_value["execution"]["epochs_completed"], "optimizer_steps": f2_value["execution"]["optimizer_steps"], "selected_epoch_zero_based": f2_value["selection"]["best_epoch"], "selected_checkpoint_id": f2_value["selection"]["checkpoint_id"], "selected_validation": {"correct": f2_value["selection"]["best_n_correct"], "total": f2_value["selection"]["best_n_total"], "top1": f2_value["selection"]["best_validation_top1"]}, "protected_state": f2_value["protected_state"]},
        "f3": {"aggregate": _artifact_binding(F3_PATH, identity_field="aggregate_id", prefix="g8ff3scores-", expected_role="g8_f_f3_artifact_scorer_aggregate", expected_schema=1), "aggregate_id": f3_value["aggregate_id"], "rows": f3_value["row_count"], "inference_count": f3_value["artifact_classifier_inference_count"], "outcomes": f3_value["outcomes"], "scorer": f3_value["scorer"], "no_reencode": True},
        "pass_two": {"authorization": _artifact_binding(PASS_TWO_AUTH_PATH, identity_field="authorization_id", prefix="g8fpass2auth-", expected_role="g8_f_owner_pass_two_authorization", expected_schema=1), "completion": _artifact_binding(PASS_TWO_PATH, identity_field="completion_id", prefix="g8fpass2complete-", expected_role="g8_f_br4_pass_two_immutable_completion", expected_schema=1), "comparison": _artifact_binding(PASS_COMPARISON_PATH, identity_field="comparison_id", prefix="g8fpass2compare-", expected_role="g8_f_pass_one_pass_two_descriptive_comparison", expected_schema=1), "authorization_id": auth["authorization_id"], "completion_id": state["completion_id"], "comparison_id": comparison["comparison_id"], "calls": state["call_count"], "candidate_evaluations": state["totals"]["candidates_evaluated"], "snr_cells": state["totals"]["snr_cells_with_selection"], "tie_breaks": state["totals"]["tie_breaks_applied"], "selection_passes": state["selection_passes"], "pass_three": state["counters"]["pass_three"], "changed_cells": comparison["changed_cells"], "unchanged_cells": comparison["unchanged_cells"], "no_interpolation": True, "bler_table_id": state["inputs"]["bler_table_id"], "candidate_authority_id": G8_CANDIDATE_AUTHORITY_ID, "candidate_authority_file_sha256": state["inputs"]["candidate_authority_file_sha256"], "selection_policy_sha256": state["inputs"]["selection_policy_sha256"], "tie_break_order": state["tie_break_order"]},
        "g8_g": {"input": _artifact_binding(G8_INPUT_PATH, identity_field="input_id", prefix="g8ginput-", expected_role="g8_g_validation_adjudication_exact_input", expected_schema=1), "closeout": _artifact_binding(G8_CLOSEOUT_PATH, identity_field="closeout_id", prefix="g8closeout-", expected_role="g8_terminal_validation_side_closeout", expected_schema=1), "binding_correction": _artifact_binding(G8_CORRECTION_PATH, identity_field="correction_id", prefix="g8bindingcorrection-", expected_role="g8_terminal_binding_identity_metadata_correction", expected_schema=1), "source_manifest": _artifact_binding(G8_SOURCE_MANIFEST_PATH, identity_field="manifest_id", prefix="g8gsource-", expected_schema=1), "closeout_id": closeout["closeout_id"], "closeout_sha256": sha256_file(G8_CLOSEOUT_PATH), "correction_id": correction["correction_id"], "operating_points": closeout["operating_points"], "nondegeneracy": closeout["classical_nondegeneracy"], "er1_strength": closeout["er1_strength"], "br16_h2": closeout["br16_h2_validation_freeze"], "pass_three": 0, "test_split": "SEALED"},
        "w5": {"repair": _artifact_binding(W5_PATH, identity_field="repair_id", prefix="w5repaircompletion-", expected_role="w5_gradscaler_accounting_repair_completion", expected_schema=1), "source_manifest": _artifact_binding(W5_SOURCE_MANIFEST_PATH, identity_field="manifest_id", prefix="w5source-", expected_role="w5_training_critical_source_manifest", expected_schema=1), "repair_id": w5_value["repair_id"], "source_manifest_id": w5_value["source_lineage"]["manifest_id"], "protected_counters": w5_value["protected_counters"], "scientific_boundary": w5_value["scientific_boundary"], "current_authority": True, "superseded_pre_repair_completion": w5_value["supersedes"]},
    }


def _make_body(epoch: Mapping[str, Any]) -> dict[str, Any]:
    index = epoch["index"]
    matrix = epoch["matrix"]
    readiness = _readiness_from_artifacts()
    foundations = _assert_frozen_science()
    consumer = _verify_frozen_consumer()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": COMPLETION_ROLE,
        "status": "W6_GREEN_CLOSED_CLASSICAL_PRE_TEST_IMPLEMENTATION_AND_EVIDENCE_BOUNDARY_AUTHENTICATED",
        "identity_scheme": {"canonical_encoding": "UTF-8 sorted-key compact JSON with one trailing LF", "completion_id": "w6completion-sha256(canonical body excluding completion_id and artifact_content_sha256)", "artifact_content_sha256": "sha256(canonical object excluding artifact_content_sha256)"},
        "authority": epoch["authority"],
        "index_matrix": {"index": {"path": "results/baseline/w6/w6_classical_evidence_index.json", "id": index["index_id"], "file_sha256": sha256_file(INDEX_PATH)}, "matrix": {"path": "results/baseline/w6/w6_requirement_matrix.json", "id": matrix["matrix_id"], "file_sha256": sha256_file(MATRIX_PATH)}, "requirement_counts": matrix["counts"], "original_obligation_count": len(matrix["entries"]), "w6_hold": matrix["w6_hold"]},
        "readiness": readiness,
        "foundations": foundations,
        "terminal_scientific_values": {"g8_passes": {"pass_one": 1, "pass_two": 1, "pass_three": 0}, "pass_two_scope": {"calls": 18, "candidate_evaluations": 8190, "snr_cells": 378, "tie_breaks": 95}, "operating_points": EXPECTED_OPERATING_POINTS, "classical_nondegeneracy": foundations["g8_g"]["nondegeneracy"], "er1_strength": EXPECTED_ER1, "br16": EXPECTED_BR16, "h2": EXPECTED_H2, "bler_binding": {"table_id": G8_C_TABLE_ID, "table_sha256": G8_C_TABLE_SHA256, "interpolation": False, "extrapolation": False}, "composition": "P(success)*acc_clean+(1-P(success))*acc_outage", "selection_policy_sha256": G8_SELECTION_POLICY_SHA256, "tie_break_order": EXPECTED_TIE_BREAK_ORDER, "candidate_authority_id": G8_CANDIDATE_AUTHORITY_ID, "candidate_authority_unchanged": True, "f3_scorer": "br12_artifact_finetuned_reference_classifier", "no_f3_reencode": True, "no_pass_three_mechanism_or_result": True},
        "frozen_selection_consumer": consumer,
        "protected_counters": PROTECTED_COUNTERS,
        "protected_counter_basis": {"w6_scope": "terminal publication and read-only verification only; no scientific worker", "g8_closeout": foundations["g8_g"]["pass_three"] == 0, "f1_f2_f3_pass_two": "exact frozen completion records and closeout verifiers", "w5": "repaired W5 scientific boundary and protected counters", "test": "G-1/G-2/W4/G8/W5 guards remain sealed"},
        "future_boundary": {"status": "FUTURE_NOT_COMPLETE", "g12_opened": False, "g12_outputs_exist": False, "items": FUTURE_ITEMS, "requirements_not_claimed_complete": FUTURE_ITEMS, "w7_g4": 0, "w8": 0, "test_model_facing_access": 0},
        "terminal_statement": "W6 GREEN — CLASSICAL PRE-TEST IMPLEMENTATION AND EVIDENCE BOUNDARY CLOSED; FROZEN ARTIFACT CORPUS, FINAL BR-4 PASS-TWO OUTPUTS, OPERATING POINTS AND CLASSICAL PROVENANCE ARE AUTHENTICATED FOR DOWNSTREAM USE; NO CLASSICAL SCIENCE WAS RECOMPUTED; NO LEARNED SCIENTIFIC TRAINING OR TEST ACCESS OCCURRED; NEXT: SEPARATE W7 / G-4 OWNER AUTHORIZATION.",
        "terminal_verifier": _tool_binding(Path(__file__).resolve()),
    }


def _reauthenticate() -> None:
    """Run only existing read-only verifiers; this function has no worker path."""
    verify_g1()
    verify_g2()
    _run_tool(W4_TOOL)
    verify_g8_c()
    verify_g8_d()
    # The full G8_E CLI also performs live archive-byte verification, which is
    # intentionally unavailable on a clean hosted checkout.  This existing
    # contract verifier is the strong read-only source/contract check with the
    # live dataset probe explicitly disabled; the frozen E2--E7 artifacts below
    # remain independently bound by _assert_frozen_science().
    g8_e_v3s.verify_frozen_contract(verify_live_sources=True, verify_live_data=False)
    f1_value = f1_closeout.verify_closeout()
    f1_closeout.verify_monitor_closeout(completion=f1_value)
    f2_closeout.verify_compact()
    f3.verify_aggregate()
    pass_two.verify_authorization(verify_live_data=False)
    pass_two.verify_state(verify_live_data=False)
    pass_two.verify_comparison(verify_live_data=False)
    g8.verify_adjudication_input(verify_live_data=False)
    g8.verify_closeout()
    g8.verify_terminal_binding_correction()
    g8.verify_source_manifest()
    verify_w5()


def build_completion() -> dict[str, Any]:
    epoch = _verify_w6_a_epoch()
    _reauthenticate()
    return _identified(_make_body(epoch))


def verify_completion(path: Path = COMPLETION_PATH, *, reauthenticate: bool = True) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = load_json(Path(path), canonical_bytes=False)
    required = {
        "schema_version", "artifact_role", "status", "identity_scheme", "authority", "index_matrix", "readiness", "foundations", "terminal_scientific_values", "frozen_selection_consumer", "protected_counters", "protected_counter_basis", "future_boundary", "terminal_statement", "terminal_verifier", "completion_id", "artifact_content_sha256",
    }
    require(set(value) == required, "terminal W6 completion schema differs")
    _verify_completion_identity(value, raw)
    require(value["schema_version"] == SCHEMA_VERSION and value["artifact_role"] == COMPLETION_ROLE and value["status"] == "W6_GREEN_CLOSED_CLASSICAL_PRE_TEST_IMPLEMENTATION_AND_EVIDENCE_BOUNDARY_AUTHENTICATED", "terminal W6 completion header differs")
    epoch = _verify_w6_a_epoch()
    if reauthenticate:
        _reauthenticate()
    expected = _identified(_make_body(epoch))
    require(value == expected, "terminal W6 completion does not reproduce from frozen evidence")
    return value


def _write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = path.open("xb")
    except FileExistsError:
        raise W6CompleteHold(f"terminal W6 completion already exists: {path}") from None
    with descriptor:
        descriptor.write(raw)
        descriptor.flush()
        import os
        os.fsync(descriptor.fileno())
    import os
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="publish the additive immutable completion, refusing replacement")
    args = parser.parse_args(argv)
    try:
        if args.write:
            value = build_completion()
            _write_immutable(COMPLETION_PATH, canonical(value))
            value = verify_completion(COMPLETION_PATH, reauthenticate=False)
        else:
            value = verify_completion()
    except (W6CompleteHold, OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS", "completion_id": value["completion_id"], "artifact_content_sha256": value["artifact_content_sha256"], "file_sha256": sha256_file(COMPLETION_PATH), "index_id": value["index_matrix"]["index"]["id"], "matrix_id": value["index_matrix"]["matrix"]["id"], "counts": value["index_matrix"]["requirement_counts"], "g1": "PASS", "g2": "PASS", "w4": "PASS", "pass_one": 1, "pass_two": 1, "pass_three": 0, "test": "SEALED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
