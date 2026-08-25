"""Deterministic validation-only G8_G adjudication and immutable closeout.

The module consumes only frozen F3 scoring units, historical G8_E records, and
pass-two selections.  It has no training or test-split entry point.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s
from baseline import g8_e_pass_one as pass_one
from baseline import g8_f_f3 as f3
from baseline import g8_f_pass_two as pass_two
from config.params import REPO_ROOT, get

SCHEMA_VERSION = 1
ROOT = REPO_ROOT / "results/baseline/g8"
INPUT_PATH = ROOT / "g8_validation_adjudication_input.json"
CLOSEOUT_PATH = ROOT / "g8_closeout.json"
SOURCE_MANIFEST_PATH = ROOT / "g8_closeout_source_manifest.json"
CORPUS_PLAN_PATH = REPO_ROOT / "results/baseline/g8_f/corpus_plan.json"
PROBE_SUMMARY_PATH = REPO_ROOT / "results/probes/transparency_bitrate/summary.json"
PROBE_ROWS_PATH = REPO_ROOT / "results/probes/transparency_bitrate/per_image.csv"
G8_C_TABLE_PATH = REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
G8_D_HANDOFF_PATH = REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"
G8_E_HANDOFF_PATH = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
F1_COMPLETION_PATH = REPO_ROOT / "results/baseline/g8_f/f1_completion.json"
F2_COMPLETION_PATH = REPO_ROOT / "results/baseline/g8_f/f2_completion.json"
INPUT_PREFIX = "g8ginput-"
CLOSEOUT_PREFIX = "g8closeout-"
MANIFEST_PREFIX = "g8gsource-"
BOOTSTRAP_SEED = 20260730  # literal-ok: frozen validation-only transparency-probe seed
VALIDATION_COUNT = f3.EXPECTED_PER_STRUCTURAL
STRUCTURAL_COUNT = f3.EXPECTED_STRUCTURAL
EXPECTED_QUALITY_COUNT = 120  # literal-ok: frozen AM-87 support-plan quality count
ONE_SIDED_CONFIDENCE = 0.95  # literal-ok: frozen ER-3 one-sided confidence
PERCENT_SCALE = 100  # literal-ok: exact proportion-to-percentage conversion
BOOTSTRAP_BATCH = 100  # literal-ok: memory-only batching, no scientific effect
SOURCE_PATHS = (
    "src/baseline/g8_g_closeout.py",
    "src/baseline/g8_f_pass_two.py",
    "tools/closeout_g8.py",
)


class G8CloseoutHold(RuntimeError):
    """A frozen-input, rule, non-degeneracy, or saturation violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G8CloseoutHold(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _identified(body: Mapping[str, Any], *, field: str, prefix: str) -> dict[str, Any]:
    return f3.identified(body, field=field, prefix=prefix)


def _verify_identified(value: Mapping[str, Any], *, field: str, prefix: str) -> None:
    f3._verify_identified(value, field=field, prefix=prefix)


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _clean_validation_vector() -> tuple[list[str], np.ndarray, dict[str, Any]]:
    summary = _json(PROBE_SUMMARY_PATH)
    require(f3.sha256_file(PROBE_ROWS_PATH) == summary["per_image_file_hash"], "clean-validation trajectory file differs")
    require(summary["clean_validation"] == {"n_correct": 898, "n_total": VALIDATION_COUNT, "top1_accuracy": 0.898}, "G1 clean-validation aggregate differs")
    require(summary["test_isolation_declaration"] == {"test_accessed": False, "test_accuracy_computed": False, "test_inference": False, "test_split_sealed": True}, "clean-validation source does not preserve test isolation")
    clean: dict[str, int] = {}
    with PROBE_ROWS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stable_id = row["stable_sample_id"]
            value = int(row["clean_correct"].lower() == "true")
            require(stable_id not in clean or clean[stable_id] == value, "clean-validation trajectory is inconsistent")
            clean[stable_id] = value
    stable_ids = sorted(clean)
    require(len(stable_ids) == VALIDATION_COUNT and sum(clean.values()) == 898, "clean-validation exact set/count differs")
    binding = {"path": str(PROBE_ROWS_PATH.relative_to(REPO_ROOT)), "file_sha256": summary["per_image_file_hash"], "classifier_checkpoint_sha256": summary["classifier_checkpoint_identity"], "manifest_sha256": summary["manifest_identity"], "correct_count": 898, "total_count": VALIDATION_COUNT}
    return stable_ids, np.asarray([clean[sample_id] for sample_id in stable_ids], dtype=np.int8), binding


def _quality_vectors(runtime_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    aggregate = f3.verify_aggregate()
    context = pass_one.authenticate_inputs()
    contract_id = aggregate["contract_id"]
    stable_ids, _, _ = _clean_validation_vector()
    vectors: dict[str, np.ndarray] = {}
    structural_meta: dict[str, dict[str, Any]] = {}
    score_dir = Path(runtime_root) / f3.REMOTE_SCORE_DIRNAME
    for structural in sorted((row for row in context["measurement_authority"]["structural_identities"] if row["dataset"] == "imagenette160"), key=lambda row: row["structural_identity_id"]):
        structural_id = structural["structural_identity_id"]
        unit = f3.verify_scoring_unit(score_dir / f"{structural_id}.json", contract_id=contract_id, structural_id=structural_id)
        by_id = {row["stable_sample_id"]: int(row["correct"]) for row in unit["rows"]}
        require(sorted(by_id) == stable_ids and len(by_id) == VALIDATION_COUNT, f"F3 stable-ID denominator differs: {structural_id}")
        vectors[structural_id] = np.asarray([by_id[sample_id] for sample_id in stable_ids], dtype=np.int8)
        structural_meta[structural_id] = dict(structural)
    require(len(vectors) == STRUCTURAL_COUNT, "F3 structural scoring-unit set differs")

    plan = _json(CORPUS_PLAN_PATH)
    quality_rows: list[dict[str, Any]] = []
    quality_vectors: dict[str, np.ndarray] = {}
    covered: set[str] = set()
    for quality in plan["artifact_quality_projection"]["qualities"]:
        source_ids = sorted(row["structural_identity_id"] for row in quality["source_structural_identities"])
        require(source_ids and all(source_id in vectors for source_id in source_ids), "AM-87 quality has a foreign structural identity")
        reference = vectors[source_ids[0]]
        require(all(np.array_equal(reference, vectors[source_id]) for source_id in source_ids[1:]), "PHY aliases disagree on artifact-scoring trajectory")
        quality_id = quality["quality_id"]
        quality_vectors[quality_id] = reference
        covered.update(source_ids)
        ratios = sorted({structural_meta[source_id]["ratio"] for source_id in source_ids})
        quality_rows.append({"quality_id": quality_id, "identity": quality["identity"], "source_structural_identity_count": len(source_ids), "source_structural_ids": source_ids, "ratios": ratios, "correct_count": int(reference.sum()), "total_count": len(reference), "accuracy": float(reference.mean()), "outcome_counts": quality["validation_feasibility_audit"]})
    require(len(quality_rows) == EXPECTED_QUALITY_COUNT and covered == set(vectors), "AM-87/F3 quality projection does not cover the frozen quality and structural universes")
    return quality_rows, quality_vectors, structural_meta


def _ratio_ceiling_bootstrap(quality_rows: list[dict[str, Any]], quality_vectors: Mapping[str, np.ndarray], clean: np.ndarray) -> list[dict[str, Any]]:
    ratios = list(get("bandwidth.ratios"))
    resamples = int(get("evaluation.bootstrap_resamples"))
    require(resamples == int(get("evaluation.bootstrap_resamples")), "configured bootstrap resample count differs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(clean), size=(resamples, len(clean)), dtype=np.int32)
    clean_accuracy = clean[draw].mean(axis=1)
    results: list[dict[str, Any]] = []
    for ratio in ratios:
        candidates = sorted((row for row in quality_rows if ratio in row["ratios"]), key=lambda row: row["quality_id"])
        require(candidates, f"ratio {ratio} has no exact quality universe")
        matrix = np.stack([quality_vectors[row["quality_id"]] for row in candidates])
        point_counts = matrix.sum(axis=1)
        selected_index = int(np.argmax(point_counts))
        # Re-select the codec ceiling inside each bootstrap trajectory, matching
        # the already-frozen validation-probe selection-aware method.
        selected_bootstrap = np.empty(resamples, dtype=np.float64)
        for start in range(0, resamples, BOOTSTRAP_BATCH):
            indices = draw[start:start + BOOTSTRAP_BATCH]
            selected_bootstrap[start:start + len(indices)] = matrix[:, indices].mean(axis=2).max(axis=0)
        differences = selected_bootstrap - clean_accuracy
        lower = float(np.quantile(differences, 1.0 - ONE_SIDED_CONFIDENCE, method="lower"))
        results.append({"ratio": ratio, "quality_count": len(candidates), "selected_quality_id": candidates[selected_index]["quality_id"], "selected_correct_count": int(point_counts[selected_index]), "total_count": len(clean), "ceiling_accuracy": float(point_counts[selected_index] / len(clean)), "clean_accuracy": float(clean.mean()), "point_difference": float(point_counts[selected_index] / len(clean) - clean.mean()), "one_sided_95_lower_bound": lower, "meets_efficiency_threshold": lower >= -float(get("bandwidth.efficiency_ratio_threshold_pp")) / PERCENT_SCALE, "meets_crossover_threshold": lower >= -float(get("bandwidth.crossover_ratio_threshold_pp")) / PERCENT_SCALE})
    return results


def _overhead_by_structural(runtime_root: Path, structural_meta: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    context = pass_one.authenticate_inputs()
    by_id = {obj["measurement_identity_id"]: obj for obj in context["e4"]["objects"] if obj["measurement_identity_id"] in structural_meta}
    output: list[dict[str, Any]] = []
    for structural_id in sorted(structural_meta):
        obj = by_id[structural_id]
        output.append({"measurement_identity_id": structural_id, "payload_budget_bytes": structural_meta[structural_id]["payload_budget_bytes"], "emitted_count": 0, "header_bytes_sum": 0, "emitted_codestream_bytes_sum": 0, "maximum_header_bytes": 0})
    # Filled by the authority-order traversal below, avoiding a directory scan.
    output_by_id = {row["measurement_identity_id"]: row for row in output}
    expected = v3.expected_work_units(context["measurement_authority"], context["sample_ids"])
    e4_sha = {record_id: digest for obj in by_id.values() for record_id, digest in zip(obj["source_record_ids"], obj["source_record_sha256s"], strict=True)}
    seen: set[str] = set()
    for unit in expected:
        structural_id = unit["measurement_identity_id"]
        if structural_id not in output_by_id:
            continue
        path = Path(runtime_root) / "records" / f"{unit['work_unit_id']}.json"
        raw = path.read_bytes(); record = json.loads(raw)
        require(f3.sha256_bytes(raw) == e4_sha.get(record["record_id"]), "historical BR-11 record differs from E4")
        seen.add(record["record_id"])
        if record["br11"] is None:
            require(record["outcome"] == "codec_infeasibility", "missing BR-11 object is not typed codec infeasibility")
            continue
        br11 = record["br11"]; row = output_by_id[structural_id]
        row["emitted_count"] += 1
        row["header_bytes_sum"] += int(br11["header_bytes"])
        row["emitted_codestream_bytes_sum"] += int(br11["emitted_codestream_bytes"])
        row["maximum_header_bytes"] = max(row["maximum_header_bytes"], int(br11["header_bytes"]))
    require(len(seen) == 288_000, "BR-11 exact historical record set differs")
    for row in output:
        budget = row["payload_budget_bytes"]
        row["mean_header_bytes"] = None if not row["emitted_count"] else row["header_bytes_sum"] / row["emitted_count"]
        row["maximum_header_fraction_of_budget"] = row["maximum_header_bytes"] / budget
        row["format_overhead_below_half_budget"] = bool(row["emitted_count"] and row["maximum_header_bytes"] < budget / 2)
    return output


def build_adjudication_input(*, runtime_root: Path = v3s.V3S_RUNTIME_ROOT) -> dict[str, Any]:
    pass_state = pass_two.verify_state(); comparison = pass_two.verify_comparison()
    stable_ids, clean, clean_binding = _clean_validation_vector()
    qualities, vectors, structural = _quality_vectors(Path(runtime_root))
    ceilings = _ratio_ceiling_bootstrap(qualities, vectors, clean)
    overhead = _overhead_by_structural(Path(runtime_root), structural)
    body = {"schema_version": SCHEMA_VERSION, "artifact_role": "g8_g_validation_adjudication_exact_input", "status": "COMPLETE_READ_ONLY_HISTORICAL_CACHE", "source_commit": pass_state["source_commit"], "pass_two_id": pass_state["completion_id"], "pass_two_file_sha256": f3.sha256_file(pass_two.STATE_PATH), "comparison_id": comparison["comparison_id"], "f3_id": f3.verify_aggregate()["aggregate_id"], "f3_file_sha256": f3.sha256_file(f3.AGGREGATE_PATH), "clean_validation": clean_binding, "validation_stable_id_count": len(stable_ids), "quality_count": len(qualities), "quality_summary": qualities, "ratio_ceiling_rule": {"method": "stable_id_trajectory_selection_aware_paired_bootstrap", "seed": BOOTSTRAP_SEED, "resamples": int(get("evaluation.bootstrap_resamples")), "one_sided_confidence": ONE_SIDED_CONFIDENCE, "quantile_method": "lower", "learned_blind": True}, "ratio_ceilings": ceilings, "overhead_structural_count": len(overhead), "overhead_by_structural": overhead, "protected_counters": {"pass_two": 1, "pass_three": 0, "f2_optimizer_steps_during_closure": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0}, "no_new_artifact": True, "no_reencode": True}
    return _identified(body, field="input_id", prefix=INPUT_PREFIX)


def verify_adjudication_input(path: Path = INPUT_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "G8 adjudication input is not canonical")
    _verify_identified(value, field="input_id", prefix=INPUT_PREFIX)
    state = pass_two.verify_state()
    require(value["status"] == "COMPLETE_READ_ONLY_HISTORICAL_CACHE" and value["pass_two_id"] == state["completion_id"], "G8 input status/pass-two binding differs")
    require(value["quality_count"] == EXPECTED_QUALITY_COUNT and value["overhead_structural_count"] == STRUCTURAL_COUNT and value["validation_stable_id_count"] == VALIDATION_COUNT, "G8 input exact counts differ")
    require(value["protected_counters"] == {"pass_two": 1, "pass_three": 0, "f2_optimizer_steps_during_closure": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0}, "G8 input protected counters differ")
    return value


def _selected_ratios(inputs: Mapping[str, Any]) -> dict[str, Any]:
    ladder = list(get("bandwidth.ratios"))
    by_ratio = {row["ratio"]: row for row in inputs["ratio_ceilings"]}
    efficiency_matches = [ratio for ratio in reversed(ladder) if by_ratio[ratio]["meets_efficiency_threshold"]]
    crossover_matches = [ratio for ratio in reversed(ladder) if by_ratio[ratio]["meets_crossover_threshold"]]
    require(efficiency_matches, "Imagenette satisfies no efficiency ratio; DEC-1 STL-10 fallback requires separate frozen sweep")
    efficiency = efficiency_matches[0]
    require(efficiency != ladder[-1], "efficiency selection saturated the smallest ladder rung; downward extension is required")
    if crossover_matches:
        crossover = crossover_matches[0]; headline_selector = "crossover_ratio"; asymmetric = False
    else:
        crossover = ladder[0]; headline_selector = str(get("bandwidth.crossover_ratio_unsatisfiable_fallback")); asymmetric = True
    headline = efficiency if headline_selector == "efficiency_ratio" else crossover
    headline_index = ladder.index(headline)
    if headline_index + 2 < len(ladder):
        low = ladder[headline_index + 2]; low_boundary = False
    else:
        require(headline_index + 1 < len(ladder), "no lower ratio exists below headline")
        low = ladder[-1]; low_boundary = True
    return {"ladder_high_to_low": ladder, "efficiency_ratio": efficiency, "crossover_ratio": crossover, "crossover_threshold_satisfied": not asymmetric, "asymmetric_fallback_applied": asymmetric, "headline_ratio_selector": headline_selector, "headline_ratio": headline, "low_ratio_operating_point": low, "low_ratio_boundary_rule_applied": low_boundary}


def _calls_by_key(state: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(call["ratio"], call["mode"]): call for call in state["calls"]}


def _selected_candidate_details(state: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates = {row["candidate_id"]: row for row in context["candidate_authority"]["candidates"]}
    mapping = {row["candidate_id"]: row["measurement_identity_id"] for row in context["mapping"]["mapping_rows"]}
    rows: list[dict[str, Any]] = []
    for call in state["calls"]:
        for point in call["per_snr"]:
            candidate = candidates[point["authority_candidate_id"]]
            rows.append({"ratio": call["ratio"], "mode": call["mode"], "snr_db": point["snr_db"], "candidate_id": candidate["candidate_id"], "measurement_identity_id": mapping[candidate["candidate_id"]], "modulation": candidate["modulation"], "ldpc_rate": candidate["ldpc_rate"], "encode_axis_px": candidate["encode_axis_px"], "expected_accuracy": point["selected_composition"]["expected_accuracy"], "success_probability": point["selected_composition"]["success_probability"], "tie_break_applied": point["tie_break_applied"]})
    require(len(rows) == 378, "final selected-cell count differs")
    return rows, mapping


def _h2_freeze(state: Mapping[str, Any], headline: str) -> dict[str, Any]:
    call = _calls_by_key(state)[(headline, "classical_fixed_mcs")]
    require(call["held_fixed"]["design_snr_db"] == get("baseline.fixed_mcs_design_snr_db"), "BR-16 design SNR differs")
    points = {float(row["snr_db"]): row for row in call["per_snr"]}
    width = float(get("evaluation.cliff_window_db"))
    windows: list[tuple[float, float, float]] = []
    for low in sorted(points):
        high = low + width
        if high in points:
            low_acc = float(points[low]["selected_composition"]["expected_accuracy"])
            high_acc = float(points[high]["selected_composition"]["expected_accuracy"])
            windows.append((high_acc - low_acc, low, high))
    require(windows, "no exact configured H2 window exists")
    drop, low, high = max(windows, key=lambda row: (row[0], -row[1]))
    return {"system": "classical_fixed_mcs", "ratio": headline, "fixed_configuration": call["held_fixed"], "window_selection": get("evaluation.cliff_window_selection"), "window_width_db": width, "low_snr_db": low, "high_snr_db": high, "low_expected_accuracy": points[low]["selected_composition"]["expected_accuracy"], "high_expected_accuracy": points[high]["selected_composition"]["expected_accuracy"], "classical_drop_pp": drop * PERCENT_SCALE, "classical_point_threshold_pp": get("evaluation.cliff_drop_pp"), "classical_point_threshold_met": drop * PERCENT_SCALE >= float(get("evaluation.cliff_drop_pp")), "learned_h2_result": "SEALED_UNTIL_LATER_LEARNED_AND_TEST_GATES"}


def build_closeout() -> dict[str, Any]:
    inputs = verify_adjudication_input(); state = pass_two.verify_state(); comparison = pass_two.verify_comparison(); context = pass_one.authenticate_inputs()
    ratios = _selected_ratios(inputs)
    selected, mapping = _selected_candidate_details(state, context)
    overhead = {row["measurement_identity_id"]: row for row in inputs["overhead_by_structural"]}
    nondegenerate: list[dict[str, Any]] = []
    for ratio in (ratios["efficiency_ratio"], ratios["crossover_ratio"]):
        selected_at_ratio = [row for row in selected if row["ratio"] == ratio and row["mode"] == "classical_adaptive"]
        rate_count = len({row["ldpc_rate"] for row in selected_at_ratio})
        bad = [row for row in selected_at_ratio if not overhead[row["measurement_identity_id"]]["format_overhead_below_half_budget"]]
        require(rate_count > 1 and not bad, f"classical non-degeneracy/format-overhead gate failed at {ratio}")
        nondegenerate.append({"ratio": ratio, "selected_ldpc_rate_count": rate_count, "selected_ldpc_rates": sorted({row["ldpc_rate"] for row in selected_at_ratio}), "selected_cell_count": len(selected_at_ratio), "all_selected_format_overhead_below_half_budget": True, "maximum_selected_header_fraction_of_budget": max(overhead[row["measurement_identity_id"]]["maximum_header_fraction_of_budget"] for row in selected_at_ratio)})
    adaptive = _calls_by_key(state)[(ratios["crossover_ratio"], "classical_adaptive")]
    points = adaptive["per_snr"]
    high = points[len(points) - 1]; previous = points[len(points) - 2]
    rising = float(high["selected_composition"]["expected_accuracy"]) > float(previous["selected_composition"]["expected_accuracy"])
    require(not rising, "adaptive validation baseline is still rising at 18 dB; upper-grid extension is required")
    schedule = {"decision": "headline_ratio_only_full_strength_efficiency_at_sweep_strength", "full_strength_ratios": [ratios["headline_ratio"]], "one_ratio_ldpc_hours": get("compute.er1_projected_ldpc_decode_hours_one_ratio"), "two_ratio_ldpc_hours": get("compute.er1_projected_ldpc_decode_hours_two_ratios"), "per_run_cap_hours": get("compute.max_wall_clock_hours_per_run"), "total_hours_status": get("compute.er1_projected_total_hours_status"), "cost_axes": get("compute.schedule_cost_compared_as"), "reason": "two-ratio aggregate total wall clock remains unmeasured, so affordability on both required axes is not established"}
    h2 = _h2_freeze(state, ratios["headline_ratio"])
    bindings = {}
    for name, path in {"g8_c_bler_table": G8_C_TABLE_PATH, "g8_d_handoff": G8_D_HANDOFF_PATH, "g8_e_e4": v3s.V3S_E4_PATH, "g8_e_e7": G8_E_HANDOFF_PATH, "pass_one": f3.PASS_ONE_PATH, "am87_corpus_plan": CORPUS_PLAN_PATH, "am88_sampler_plan": REPO_ROOT / "results/baseline/g8_f/am88_sampler_plan.json", "f1_completion": F1_COMPLETION_PATH, "f2_completion": F2_COMPLETION_PATH, "f2_classifier_freeze": f3.F2_FREEZE_PATH, "f3_scores": f3.AGGREGATE_PATH, "pass_two_authorization": pass_two.AUTHORIZATION_PATH, "pass_two_completion": pass_two.STATE_PATH, "pass_comparison": pass_two.COMPARISON_PATH, "adjudication_input": INPUT_PATH}.items():
        value = _json(path); bindings[name] = {"path": str(path.relative_to(REPO_ROOT)), "id": next((value[key] for key in ("table_id", "handoff_id", "e4_id", "state_id", "plan_id", "completion_id", "freeze_id", "aggregate_id", "authorization_id", "comparison_id", "input_id") if key in value), None), "file_sha256": f3.sha256_file(path)}
    body = {"schema_version": SCHEMA_VERSION, "artifact_role": "g8_terminal_validation_side_closeout", "status": "G8_GREEN_VALIDATION_SIDE_CLOSED", "terminal_verdict": "G8 GREEN — BR-12 ARTIFACT SCORER FROZEN; VALIDATION CACHE RE-SCORED; BR-4 PASS TWO EXECUTED EXACTLY ONCE AND FROZEN; G8 VALIDATION-SIDE OPERATING POINTS CLOSED; TEST AND LEARNED-SYSTEM TRAINING REMAIN SEALED.", "source_commit": state["source_commit"], "bindings": bindings, "policy": {"selection_policy_sha256": state["inputs"]["selection_policy_sha256"], "tie_break_order": state["tie_break_order"], "composition": "P(success)*acc_clean+(1-P(success))*acc_outage", "bler_interpolation": False}, "ratio_rule_evaluations": inputs["ratio_ceilings"], "operating_points": ratios, "upper_grid_saturation": {"ratio_checked": ratios["crossover_ratio"], "previous_snr_db": previous["snr_db"], "maximum_snr_db": high["snr_db"], "previous_expected_accuracy": previous["selected_composition"]["expected_accuracy"], "maximum_expected_accuracy": high["selected_composition"]["expected_accuracy"], "still_rising": False}, "classical_nondegeneracy": nondegenerate, "er1_strength": schedule, "br16_h2_validation_freeze": h2, "dataset_disposition": {"primary": "imagenette160", "efficiency_threshold_satisfied": True, "stl10_fallback_invoked": False}, "artifact_classifier_release": {"released_for_classical_scoring": True, "scorer": f3.verify_aggregate()["scorer"], "fallback_training": 0}, "pass_two": {"completion_id": state["completion_id"], "calls": state["call_count"], "candidate_evaluations": state["totals"]["candidates_evaluated"], "snr_cells": state["totals"]["snr_cells_with_selection"], "tie_breaks": state["totals"]["tie_breaks_applied"], "selection_terminates_after_pass": 2, "pass_three_exists": False}, "pass_one_pass_two_comparison": {"comparison_id": comparison["comparison_id"], "changed_cells": comparison["changed_cells"], "unchanged_cells": comparison["unchanged_cells"], "tie_status_changed_cells": comparison["tie_status_changed_cells"]}, "selected_operating_points": selected, "protected_counters": {"pass_one": 1, "pass_two": 1, "pass_three": 0, "f2_optimizer_steps_during_closure": 0, "fallback_training": 0, "ratio_adjudication": 1, "learned_training": 0, "test_access": 0}, "learned_blind_ratio_selection": True, "learned_result_used_for_ratio_selection": False, "learned_versus_classical_crossover_decided": False, "test_split": "SEALED", "next_gate_not_authorized": "learned-system training"}
    return _identified(body, field="closeout_id", prefix=CLOSEOUT_PREFIX)


def verify_closeout(path: Path = CLOSEOUT_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "G8 closeout is not canonical")
    _verify_identified(value, field="closeout_id", prefix=CLOSEOUT_PREFIX)
    require(value == build_closeout(), "G8 closeout does not reproduce from frozen inputs")
    require(value["protected_counters"] == {"pass_one": 1, "pass_two": 1, "pass_three": 0, "f2_optimizer_steps_during_closure": 0, "fallback_training": 0, "ratio_adjudication": 1, "learned_training": 0, "test_access": 0}, "G8 closeout counters differ")
    require(value["test_split"] == "SEALED" and not value["learned_versus_classical_crossover_decided"], "G8 scope boundary differs")
    return value


def build_source_manifest() -> dict[str, Any]:
    closeout = verify_closeout()
    rows = [{"path": path, "bytes": (REPO_ROOT / path).stat().st_size, "sha256": f3.sha256_file(REPO_ROOT / path)} for path in SOURCE_PATHS]
    body = {"schema_version": SCHEMA_VERSION, "artifact_role": "g8_g_closeout_source_manifest", "status": "FROZEN", "source_commit": closeout["source_commit"], "closeout_id": closeout["closeout_id"], "closeout_file_sha256": f3.sha256_file(CLOSEOUT_PATH), "sources": rows}
    return _identified(body, field="manifest_id", prefix=MANIFEST_PREFIX)


def verify_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == f3.rendered_json(value), "G8 source manifest is not canonical")
    _verify_identified(value, field="manifest_id", prefix=MANIFEST_PREFIX)
    require(value == build_source_manifest(), "G8 source manifest does not reproduce")
    return value
