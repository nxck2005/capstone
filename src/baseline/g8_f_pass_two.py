"""Owner-gated BR-4 selection pass two over frozen F3 scores.

Only the measured codec-accuracy objects are replaced relative to pass one.
Candidate authority, mapping, BLER table, packetisation, composition, outage,
call plan, modes, and tie-break are loaded through the pass-one frozen chain.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_pass_one as pass_one
from baseline import g8_f_f3 as f3
from baseline.classical import composition
from config.params import REPO_ROOT

SCHEMA_VERSION = 1
SCOPE = "G8_BR4_SELECTION_PASS_TWO_ONLY"
ROOT = REPO_ROOT / "results/baseline/g8_f"
AUTHORIZATION_PATH = ROOT / "pass_two_authorization.json"
STATE_PATH = ROOT / "pass_two_state.json"
COMPARISON_PATH = ROOT / "pass_one_pass_two_comparison.json"
AUTH_PREFIX = "g8fpass2auth-"
STATE_PREFIX = "g8fpass2complete-"
COMPARISON_PREFIX = "g8fpass2compare-"
SCORER = "br12_artifact_finetuned_reference_classifier"
SOURCE_PATHS = (
    "src/baseline/g8_f_pass_two.py",
    "src/baseline/g8_g_closeout.py",
    "src/baseline/g8_e_pass_one.py",
    "src/baseline/g8_f_f3.py",
    "src/baseline/classical/composition.py",
    "tools/run_g8_f_pass_two.py",
    "tools/closeout_g8.py",
    "tests/g8_e5_gate.py",
)


class PassTwoHold(RuntimeError):
    """A pass-two precondition, authorization, or exact-once violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PassTwoHold(message)


def _identified(body: Mapping[str, Any], *, field: str, prefix: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = prefix + f3.sha256_bytes(f3.canonical_json(value))
    value["artifact_content_sha256"] = f3.sha256_bytes(f3.canonical_json(value))
    return value


def _verify_identified(value: Mapping[str, Any], *, field: str, prefix: str) -> None:
    without_digest = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    require(value.get("artifact_content_sha256") == f3.sha256_bytes(f3.canonical_json(without_digest)), f"{field} content digest differs")
    without_id = {key: child for key, child in without_digest.items() if key != field}
    require(value.get(field) == prefix + f3.sha256_bytes(f3.canonical_json(without_id)), f"{field} differs")


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _source_manifest() -> list[dict[str, Any]]:
    return [{"path": path, "sha256": f3.sha256_file(REPO_ROOT / path), "bytes": (REPO_ROOT / path).stat().st_size} for path in SOURCE_PATHS]


def _f3_context() -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = f3.verify_aggregate()
    context = pass_one.authenticate_inputs()
    by_id = {obj["measurement_identity_id"]: obj for obj in aggregate["objects"]}
    require(len(by_id) == f3.EXPECTED_STRUCTURAL, "F3 structural score universe differs")
    replacement_objects = []
    for old in context["e4"]["objects"]:
        new = by_id.get(old["measurement_identity_id"])
        require(new is not None and new["total_count"] == old["total_count"] == f3.EXPECTED_PER_STRUCTURAL, "F3/E4 structural denominator mapping differs")
        replacement_objects.append({"measurement_identity_id": old["measurement_identity_id"], "status": "eligible", "correct_count": new["correct_count"], "total_count": new["total_count"]})
    updated = dict(context)
    updated["e4"] = {"e4_id": aggregate["aggregate_id"], "objects": replacement_objects}
    return updated, aggregate


def build_authorization(*, source_commit: str, github_actions: Mapping[str, Any]) -> dict[str, Any]:
    context, aggregate = _f3_context()
    pass_state = pass_one.verify_pass_one_state()
    require(source_commit == _head(), "pass-two authorization source commit is not HEAD")
    require(github_actions.get("sha") == source_commit and github_actions.get("conclusion") == "success" and isinstance(github_actions.get("run_id"), int), "pass-two source GitHub Actions is not green at the exact source SHA")
    policy, fields = pass_one.recompute_selection_policy()
    require(policy == context["chain"]["selection_policy_sha256"], "pass-two live policy fingerprint differs")
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "g8_f_owner_pass_two_authorization",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "authorized_by": "repository owner via explicit F3-F5 validation-only closure prompt",
        "scope": SCOPE,
        "source_commit": source_commit,
        "source_manifest": _source_manifest(),
        "github_actions": dict(github_actions),
        "artifact_classifier": aggregate["scorer"],
        "f3": {"aggregate_id": aggregate["aggregate_id"], "file_sha256": f3.sha256_file(f3.AGGREGATE_PATH), "ordered_scoring_sha256": aggregate["ordered_scoring_sha256"], "scoring_set_sha256": aggregate["scoring_set_sha256"]},
        "e4_candidate_authority": {"e4_id": context["chain"]["e4_id"], "e4_sha256": context["chain"]["e4_sha256"], "candidate_authority_file_sha256": context["chain"]["candidate_authority_file_sha256"], "selection_call_plan_sha256": context["chain"]["selection_call_plan_sha256"]},
        "pascal_bler_table": {"table_id": context["chain"]["bler_table_id"], "file_sha256": context["chain"]["bler_table_sha256"], "interpolation": False, "extrapolation": False},
        "composition_policy": {"module": pass_one.SCORER_MODULE, "formula": "P(success)*acc_clean+(1-P(success))*acc_outage", "success_probability": "product_r(1-BLER_r)", "fingerprint": policy},
        "tie_break": {"order": list(composition.TIE_BREAK_ORDER), "fingerprint": policy, "covered_fields": fields},
        "universe": {"call_count": context["plan"]["call_count"], "candidate_evaluations": 8190, "snr_cells": 378, "samples_per_cell": f3.EXPECTED_PER_STRUCTURAL},
        "pass_one": {"state_id": pass_state["state_id"], "file_sha256": pass_state["file_sha256"], "executed_count": 1, "immutable": True},
        "output_path": str(STATE_PATH.relative_to(REPO_ROOT)),
        "pre_execution_counters": {"pass_two": 0, "pass_three": 0, "fallback": 0, "test_access": 0},
        "non_scorer_inputs_unchanged": ["candidate_authority", "logical_structural_mapping", "codec_qualities", "ratios", "encode_axes", "snr_grid", "ldpc_rates", "modulations", "packet_mode", "pascal_bler_table", "no_interpolation", "composition_equation", "outage_probability", "eligibility", "tie_break"],
    }
    return _identified(body, field="authorization_id", prefix=AUTH_PREFIX)


def verify_authorization(path: Path = AUTHORIZATION_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "pass-two authorization is not canonical")
    _verify_identified(value, field="authorization_id", prefix=AUTH_PREFIX)
    require(value["status"] == "AUTHORIZED_NOT_CONSUMED" and value["scope"] == SCOPE, "pass-two authorization status/scope differs")
    require(value["pre_execution_counters"] == {"pass_two": 0, "pass_three": 0, "fallback": 0, "test_access": 0}, "pass-two pre-execution counters differ")
    for row in value["source_manifest"]:
        # Authenticate the exact launch-bearing Git image.  Closeout-only
        # verifier repairs after immutable pass-two publication do not rewrite
        # that history and need not remain byte-identical at HEAD.
        historical = subprocess.run(["git", "show", f"{value['source_commit']}:{row['path']}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        require(len(historical) == row["bytes"] and f3.sha256_bytes(historical) == row["sha256"], f"pass-two historical source bytes differ: {row['path']}")
    context, aggregate = _f3_context()
    require(value["f3"]["aggregate_id"] == aggregate["aggregate_id"] and value["f3"]["file_sha256"] == f3.sha256_file(f3.AGGREGATE_PATH), "pass-two F3 binding differs")
    require(value["pascal_bler_table"]["table_id"] == context["chain"]["bler_table_id"] and value["pascal_bler_table"]["file_sha256"] == context["chain"]["bler_table_sha256"], "pass-two BLER binding differs")
    policy, _ = pass_one.recompute_selection_policy()
    require(value["composition_policy"]["fingerprint"] == value["tie_break"]["fingerprint"] == policy, "pass-two policy fingerprint differs")
    return value


def _typed_authorization(owner: Mapping[str, Any], plan: Mapping[str, Any]) -> Any:
    import importlib.util
    gate_path = REPO_ROOT / "tests/g8_e5_gate.py"
    spec = importlib.util.spec_from_file_location("g8_e5_gate", gate_path)
    require(spec is not None and spec.loader is not None, "pass-two typed authorization gate is absent")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.issue(authorized_by=str(owner["authorized_by"]), reason=f"pass two {owner['authorization_id']}", max_candidates=int(plan["max_candidates"]), max_samples=int(plan["max_samples"]))


def run_pass_two(*, output_path: Path = STATE_PATH) -> dict[str, Any]:
    require(not output_path.exists(), "pass two already has an immutable completion record; rerun is forbidden")
    owner = verify_authorization(); context, aggregate = _f3_context(); plan = context["plan"]
    typed = _typed_authorization(owner, plan)
    calls: list[dict[str, Any]] = []
    totals = {"candidates_evaluated": 0, "eligible_evaluations": 0, "infeasible_evaluations": 0, "uncharacterized_evaluations": 0, "snr_cells_with_selection": 0, "snr_cells_without_selection": 0, "tie_breaks_applied": 0}
    for call in plan["calls"]:
        evaluations = pass_one.build_call_evaluations(context, call)
        curve = composition.select_operating_points(str(call["mode"]), evaluations, samples_per_cell=int(call["samples_per_cell"]), authorization=typed)
        calls.append(pass_one._serialize_curve(curve, call, context, totals))
    require(totals["candidates_evaluated"] == owner["universe"]["candidate_evaluations"] and totals["snr_cells_with_selection"] == owner["universe"]["snr_cells"] and totals["snr_cells_without_selection"] == 0 and totals["infeasible_evaluations"] == 0 and totals["uncharacterized_evaluations"] == 0, "pass-two evaluated universe/coverage differs")
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "g8_f_br4_pass_two_immutable_completion",
        "status": "PASS_TWO_COMPLETE_SELECTION_TERMINATED",
        "authorization_id": owner["authorization_id"],
        "authorization_file_sha256": f3.sha256_file(AUTHORIZATION_PATH),
        "source_commit": owner["source_commit"],
        "pass_one": owner["pass_one"],
        "f3": owner["f3"],
        "artifact_classifier": owner["artifact_classifier"],
        "inputs": {"candidate_authority_file_sha256": owner["e4_candidate_authority"]["candidate_authority_file_sha256"], "selection_call_plan_sha256": owner["e4_candidate_authority"]["selection_call_plan_sha256"], "bler_table_id": owner["pascal_bler_table"]["table_id"], "bler_table_sha256": owner["pascal_bler_table"]["file_sha256"], "selection_policy_sha256": owner["tie_break"]["fingerprint"]},
        "scorer_changed_only": True,
        "call_count": len(calls),
        "calls": calls,
        "totals": totals,
        "tie_break_order": list(composition.TIE_BREAK_ORDER),
        "selection_passes": list(composition.selection_passes()),
        "selection_terminates_after_pass": 2,
        "counters": {"pass_one": 1, "pass_two": 1, "pass_three": 0, "fallback_training": 0, "ratio_adjudication": 0, "learned_training": 0, "test_access": 0},
    }
    value = _identified(body, field="completion_id", prefix=STATE_PREFIX)
    pass_one._atomic_publish(output_path, f3.rendered_json(value))
    return value


def verify_state(path: Path = STATE_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "pass-two state is not canonical")
    _verify_identified(value, field="completion_id", prefix=STATE_PREFIX)
    owner = verify_authorization()
    require(value["status"] == "PASS_TWO_COMPLETE_SELECTION_TERMINATED" and value["authorization_id"] == owner["authorization_id"], "pass-two state header/authorization differs")
    require(value["counters"] == {"pass_one": 1, "pass_two": 1, "pass_three": 0, "fallback_training": 0, "ratio_adjudication": 0, "learned_training": 0, "test_access": 0}, "pass-two protected counters differ")
    require(value["selection_passes"] == [1, 2] and value["selection_terminates_after_pass"] == 2 and composition.selection_passes() == (1, 2), "pass-two termination/pass-three guard differs")
    require(value["call_count"] == owner["universe"]["call_count"] and value["totals"]["candidates_evaluated"] == owner["universe"]["candidate_evaluations"] and value["totals"]["snr_cells_with_selection"] == owner["universe"]["snr_cells"] and value["totals"]["snr_cells_without_selection"] == 0, "pass-two exact scope differs")
    # Recompute every selection from frozen inputs.  No publication occurs.
    context, _ = _f3_context(); typed = _typed_authorization(owner, context["plan"])
    calls: list[dict[str, Any]] = []
    totals = {"candidates_evaluated": 0, "eligible_evaluations": 0, "infeasible_evaluations": 0, "uncharacterized_evaluations": 0, "snr_cells_with_selection": 0, "snr_cells_without_selection": 0, "tie_breaks_applied": 0}
    for call in context["plan"]["calls"]:
        evaluations = pass_one.build_call_evaluations(context, call)
        curve = composition.select_operating_points(str(call["mode"]), evaluations, samples_per_cell=int(call["samples_per_cell"]), authorization=typed)
        calls.append(pass_one._serialize_curve(curve, call, context, totals))
    require(value["calls"] == calls and value["totals"] == totals, "pass-two selections do not reproduce")
    return value


def build_comparison() -> dict[str, Any]:
    first = json.loads(f3.PASS_ONE_PATH.read_bytes()); second = verify_state()
    context = pass_one.authenticate_inputs()
    candidates = {row["candidate_id"]: dict(row) for row in context["candidate_authority"]["candidates"]}
    structural = {row["structural_identity_id"]: row for row in context["measurement_authority"]["structural_identities"]}
    for mapping in context["mapping"]["mapping_rows"]:
        candidate = candidates[mapping["candidate_id"]]
        identity = structural[mapping["measurement_identity_id"]]
        candidate["payload_budget_bytes"] = identity["payload_budget_bytes"]
    require([(c["ratio"], c["mode"]) for c in first["calls"]] == [(c["ratio"], c["mode"]) for c in second["calls"]], "pass-one/pass-two call structure differs")
    changes: list[dict[str, Any]] = []; unchanged = 0; tie_changes = 0
    for old_call, new_call in zip(first["calls"], second["calls"], strict=True):
        for old, new in zip(old_call["per_snr"], new_call["per_snr"], strict=True):
            require(old["snr_db"] == new["snr_db"], "pass-one/pass-two SNR structure differs")
            old_id, new_id = old["authority_candidate_id"], new["authority_candidate_id"]
            if old["tie_break_applied"] != new["tie_break_applied"]: tie_changes += 1
            if old_id == new_id:
                unchanged += 1; continue
            before, after = candidates[old_id], candidates[new_id]
            old_comp, new_comp = old["selected_composition"], new["selected_composition"]
            changes.append({"ratio": old_call["ratio"], "mode": old_call["mode"], "snr_db": old["snr_db"], "before_candidate_id": old_id, "after_candidate_id": new_id, "codec_budget_bytes": [before["payload_budget_bytes"], after["payload_budget_bytes"]], "encode_axis_px": [before["encode_axis_px"], after["encode_axis_px"]], "modulation": [before["modulation"], after["modulation"]], "ldpc_rate": [before["ldpc_rate"], after["ldpc_rate"]], "expected_accuracy": [old_comp["expected_accuracy"], new_comp["expected_accuracy"]], "success_probability": [old_comp["success_probability"], new_comp["success_probability"]], "tie_break_applied": [old["tie_break_applied"], new["tie_break_applied"]]})
    body = {"schema_version": 1, "artifact_role": "g8_f_pass_one_pass_two_descriptive_comparison", "status": "COMPLETE_NO_FURTHER_ITERATION", "pass_one_id": first["state_id"], "pass_two_id": second["completion_id"], "total_cells": 378, "unchanged_cells": unchanged, "changed_cells": len(changes), "tie_status_changed_cells": tie_changes, "changes": changes, "adaptive_followup": False, "pass_three": 0}
    require(unchanged + len(changes) == 378, "pass comparison cell count differs")
    return _identified(body, field="comparison_id", prefix=COMPARISON_PREFIX)


def verify_comparison(path: Path = COMPARISON_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "pass comparison is not canonical")
    _verify_identified(value, field="comparison_id", prefix=COMPARISON_PREFIX)
    require(value == build_comparison(), "pass comparison does not reproduce")
    return value
