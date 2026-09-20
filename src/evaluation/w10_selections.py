"""Prospective W10 validation-only selection contracts for JPEG-secondary and ER-12.

Both selections are frozen *before* their lifecycle runs.  The contract
identity enumerates every degree of freedom AM-98 leaves to a downstream
selector; the artifact records the complete candidate score table so the
chosen candidate at each of the 21 SNR points is recomputable and verifiable
without re-running the channel.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from baseline.ldpc.modulation import bits_per_symbol
from baseline.classical.composition import (
    MeasuredCodecAccuracy,
    compose,
    measured_outage_accuracy_from_record,
)
from baseline.classical.outage import load_outage_policy
from baseline.ldpc.transport import build_packet_plan
from baseline.g8_pascal_merge import MERGE_REPORT_PATH, TABLE_PATH, load_successor_bler_table
from config.params import REPO_ROOT, get
from evaluation.w10_scope import HEADLINE_RATIO, W10_DATASET, W10_VALIDATION_DENOMINATOR, snr_grid
from evaluation.er9_search import configured_phy_candidates
from runtime.source_epochs import load_w10_manifest, source_record
from training.deterministic_core import canonical_sha256

JPEG_SELECTION_PATH = "results/learned/w10/jpeg_validation_selection.json"
ER12_SELECTION_PATH = "results/learned/w10/er12_validation_selection.json"
JPEG_SELECTION_PREFIX = "w10jpegselection-"
ER12_SELECTION_PREFIX = "w10er12selection-"
JPEG_SELECTION_ROLE = "W10_JPEG_SECONDARY_VALIDATION_SELECTION"
ER12_SELECTION_ROLE = "W10_ER12_VALIDATION_SELECTION"
SELECTION_STATUS = "FROZEN_VALIDATION_ONLY_SELECTION"
SELECTION_SCHEMA_VERSION = 2
BR12_FREEZE_PATH = "results/baseline/g8_f/artifact_classifier_freeze.json"
JPEG_PHY_SOURCE = "configured_packet_feasible_phy_x_configured_encode_axes"
ER12_PHY_SOURCE = "er9_production_closeout:train0_channel0"
JPEG_SCORER = "br12_artifact_finetuned_reference_classifier"
ER12_SCORER = "transmitted_predicted_label_then_true_label_scoring_only"
OUTAGE_POLICY = "results/baseline/w4/outage_policy.json"
NOISE_CONVENTION = "w10_scheduled_noise_id_per_stable_image_snr_k_shared_across_candidates"
JPEG_SELECTION_NOISE_CONVENTION = "none_for_br4_analytic_selection_actual_channel_only_after_freeze"
JPEG_ANALYTIC_EVIDENCE_VERSION = 1
ER12_CANDIDATE_EVIDENCE_VERSION = 1
ER12_PROTOCOL_VERSION_PARAMETER = "params.evaluation.w10_er12_protocol_version"

JPEG_TIE_BREAK = (
    "expected_accuracy_descending",
    "success_probability_descending",
    "modulation_bits_per_symbol_ascending",
    "ldpc_rate_ascending",
    "encode_axis_px_descending",
    "jpeg_quality_descending",
    "candidate_id_ascending",
)
ER12_TIE_BREAK = (
    "n_correct_descending",
    "n_delivered_descending",
    "modulation_bits_per_symbol_ascending",
    "ldpc_rate_ascending",
    "candidate_id_ascending",
)


class W10SelectionHold(RuntimeError):
    """A frozen W10 selection contract, artifact or recomputation differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W10SelectionHold(message)


def _full_sha(value: object, width: int = 64) -> bool:  # literal-ok: SHA-256 width
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W10SelectionHold(f"{label} is corrupt: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _parameter(name: str) -> Any:
    return get(f"evaluation.{name}")


@dataclass(frozen=True)
class SelectionContract:
    """One prospective W10 selection contract."""

    kind: str
    role: str
    ratio: str
    phy_source: str
    scorer: str
    candidate_qualities: tuple[int, ...]
    candidate_modulations: tuple[str, ...]
    candidate_ldpc_rates: tuple[str, ...]
    candidate_encode_axes: tuple[int, ...]
    tie_break: tuple[str, ...]
    objective: str = "n_correct_maximisation_over_the_complete_validation_split"
    noise_convention: str = NOISE_CONVENTION
    candidate_space: str = "configured_candidate_grid"
    selection_method: str = "per_image_channel_simulation"
    packet_budget_rule: str = "configured_packet_plan_feasible_then_image_codec_feasibility_is_scored"
    outage_policy: str = OUTAGE_POLICY
    validation_split: str = "val"
    denominator: int = W10_VALIDATION_DENOMINATOR
    test: str = "SEALED"
    test_access: int = 0

    def identity(self) -> dict[str, Any]:
        manifest = load_w10_manifest(REPO_ROOT, live=True)
        return {
            "contract_version": SELECTION_SCHEMA_VERSION,
            "kind": self.kind,
            "role": self.role,
            "dataset": W10_DATASET,
            "validation_split": self.validation_split,
            "denominator": self.denominator,
            "ratio": self.ratio,
            "snr_grid_db": list(snr_grid()),
            "candidate_qualities": list(self.candidate_qualities),
            "candidate_modulations": list(self.candidate_modulations),
            "candidate_ldpc_rates": list(self.candidate_ldpc_rates),
            "candidate_encode_axes_px": list(self.candidate_encode_axes),
            "phy_source": self.phy_source,
            "scorer": self.scorer,
            "outage_policy": self.outage_policy,
            "objective": self.objective,
            "tie_break": list(self.tie_break),
            "noise_convention": self.noise_convention,
            "candidate_space": self.candidate_space,
            "selection_method": self.selection_method,
            "packet_budget_rule": self.packet_budget_rule,
            "er12_protocol_version": int(_parameter("w10_er12_protocol_version")),
            "source_epoch": source_record(REPO_ROOT, manifest),
            "test": self.test,
            "test_access": self.test_access,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.identity())


def _configured_axes() -> tuple[int, ...]:
    axes = tuple(int(value) for value in get("baseline.downsample_axis_px")[W10_DATASET])
    return tuple(sorted(axes, reverse=True))


def jpeg_selection_contract() -> SelectionContract:
    return SelectionContract(
        kind="jpeg_secondary",
        role=JPEG_SELECTION_ROLE,
        ratio=HEADLINE_RATIO,
        phy_source=JPEG_PHY_SOURCE,
        scorer=JPEG_SCORER,
        candidate_qualities=tuple(int(value) for value in get("baseline.jpeg_quality_grid")),
        candidate_modulations=tuple(str(value) for value in get("baseline.modulations")),
        candidate_ldpc_rates=tuple(str(value) for value in get("baseline.ldpc_rates")),
        candidate_encode_axes=_configured_axes(),
        tie_break=JPEG_TIE_BREAK,
        objective="br4_expected_validation_accuracy_maximisation",
        noise_convention=JPEG_SELECTION_NOISE_CONVENTION,
        candidate_space="jpeg_quality_x_configured_encode_axis_x_packet_feasible_modulation_x_ldpc_rate",
        selection_method="br4_analytic_composition_no_per_candidate_channel_simulation",
        packet_budget_rule="build_packet_plan_k_modulation_ldpc_rate_then_exact_quality_codec_budget",
    )


def er12_selection_contract() -> SelectionContract:
    from evaluation.er9_search import configured_phy_candidates

    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))
    pairs = tuple(
        (str(modulation), str(rate))
        for modulation, rate, _packet in configured_phy_candidates(k)
    )
    return SelectionContract(
        kind="er12_label_bound",
        role=ER12_SELECTION_ROLE,
        ratio=HEADLINE_RATIO,
        phy_source=ER12_PHY_SOURCE,
        scorer=ER12_SCORER,
        candidate_qualities=(),
        candidate_modulations=tuple(sorted({pair[0] for pair in pairs}, key=bits_per_symbol)),
        candidate_ldpc_rates=tuple(sorted({pair[1] for pair in pairs})),
        candidate_encode_axes=(),
        tie_break=ER12_TIE_BREAK,
        candidate_space="configured_packet_feasible_modulation_x_ldpc_rate",
        selection_method="per_image_channel_simulation_with_bound_label_frame",
        packet_budget_rule="configured_packet_plan_feasible",
    )


def jpeg_contract_sha256() -> str:
    return jpeg_selection_contract().sha256()


def er12_contract_sha256() -> str:
    return er12_selection_contract().sha256()


def _contract_identity(kind: str) -> dict[str, Any]:
    if kind == "jpeg_secondary":
        return jpeg_selection_contract().identity()
    if kind == "er12_label_bound":
        return er12_selection_contract().identity()
    raise W10SelectionHold(f"unknown W10 selection kind: {kind}")


def _contract_sha256(kind: str) -> str:
    return canonical_sha256(_contract_identity(kind))


def _candidate_id(entry: Mapping[str, Any]) -> str:
    fields = {
        key: entry[key]
        for key in (
            "snr_db", "modulation", "ldpc_rate", "encode_axis_px", "quality",
        )
        if key in entry
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _n_correct(entry: Mapping[str, Any]) -> int:
    return int(entry["n_correct"])


def jpeg_rank_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(entry["expected_accuracy"]),
        -float(entry["success_probability"]),
        bits_per_symbol(str(entry["modulation"])),
        Fraction(str(entry["ldpc_rate"])),
        -int(entry["encode_axis_px"]),
        -int(entry["quality"]),
        _candidate_id(entry),
    )


def er12_rank_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -_n_correct(entry),
        -int(entry["n_delivered"]),
        bits_per_symbol(str(entry["modulation"])),
        Fraction(str(entry["ldpc_rate"])),
        _candidate_id(entry),
    )


def rank_candidates(kind: str, scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The frozen objective plus its complete deterministic tie-break order."""

    if not scores:
        raise W10SelectionHold("W10 selection has no candidates at this SNR")
    key = jpeg_rank_key if kind == "jpeg_secondary" else er12_rank_key
    eligible = [entry for entry in scores if entry.get("status", "eligible") == "eligible"]
    if not eligible:
        raise W10SelectionHold("W10 selection has no eligible candidates at this SNR")
    ordered = sorted(eligible, key=key)
    winner = dict(ordered[0])
    tied = [entry for entry in ordered if key(entry) == key(ordered[0])]
    winner["tie_break_applied"] = len(tied) > 1
    winner["candidate_id"] = _candidate_id(winner)
    return winner


def selection_identity(kind: str, selections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "kind": kind,
        "selections": [
            {
                "snr_db": float(item["snr_db"]),
                "candidate_id": str(item["candidate_id"]),
                "tie_break_applied": bool(item["tie_break_applied"]),
            }
            for item in selections
        ],
    }


def build_selection_artifact(
    kind: str,
    *,
    selections: Sequence[Mapping[str, Any]],
    candidate_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _contract_identity(kind)
    role = JPEG_SELECTION_ROLE if kind == "jpeg_secondary" else ER12_SELECTION_ROLE
    prefix = JPEG_SELECTION_PREFIX if kind == "jpeg_secondary" else ER12_SELECTION_PREFIX
    body: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "artifact_role": role,
        "status": SELECTION_STATUS,
        "contract": contract,
        "contract_sha256": _contract_sha256(kind),
        "source_epoch": contract["source_epoch"],
        "denominator": W10_VALIDATION_DENOMINATOR,
        "selections": [dict(item) for item in selections],
        "candidate_scores": [dict(item) for item in candidate_scores],
        "selection_evidence_model": (
            "jpeg_compact_clean_codec_counts_plus_packet_bler_outage_composition_recomputed"
            if kind == "jpeg_secondary"
            else "er12_worker_generated_candidate_evidence_id_and_correctness_digest_recomputed"
        ),
        "selection_evidence_trust_boundary": (
            "host_recomputes_br4_objective_from_bound_clean_codec_inputs_bler_cells_and_outage;"
            " classifier_clean_accuracy_is_a_bound_worker_input"
            if kind == "jpeg_secondary"
            else "host_recomputes_candidate_identity_and_cross_checks_content_addressed_worker_evidence_and_correctness_digest;"
            " awgn_ldpc_runtime_is_worker_authenticated"
        ),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["selection_id"] = prefix + canonical_sha256(body)
    return body


def _validate_score_entry(entry: Mapping[str, Any], *, kind: str) -> None:
    required = {
        "snr_db", "modulation", "ldpc_rate", "n_correct", "n_delivered",
        "n_decode_failure", "n_infeasible", "n_total", "per_image_correct_digest",
        "status", "candidate_evidence",
    }
    if kind == "jpeg_secondary":
        required |= {"encode_axis_px", "quality", "expected_accuracy", "success_probability", "analytic_evidence"}
    _require(set(entry) == required, "W10 candidate score schema differs")
    _require(int(entry["n_total"]) == W10_VALIDATION_DENOMINATOR, "W10 candidate denominator differs")
    _require(_full_sha(entry["per_image_correct_digest"]), "W10 candidate per-image digest is invalid")
    for field in ("n_correct", "n_delivered", "n_decode_failure", "n_infeasible"):
        value = entry[field]
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"W10 candidate {field} is invalid")
    _require(
        int(entry["n_delivered"]) + int(entry["n_decode_failure"]) + int(entry["n_infeasible"])
        == W10_VALIDATION_DENOMINATOR,
        "W10 candidate verdict split does not cover the denominator",
    )
    _require(int(entry["n_correct"]) <= W10_VALIDATION_DENOMINATOR, "W10 candidate correctness exceeds the denominator")
    _require(entry["status"] in {"eligible", "uncharacterized"}, "W10 candidate status is invalid")
    evidence = entry["candidate_evidence"]
    _require(isinstance(evidence, Mapping), "W10 candidate evidence is missing")
    evidence_body = dict(evidence)
    evidence_id = evidence_body.pop("evidence_id", None)
    _require(evidence_id == "w10candidateevidence-" + canonical_sha256(evidence_body), "W10 candidate evidence ID differs")
    _require(evidence.get("schema_version") == (JPEG_ANALYTIC_EVIDENCE_VERSION if kind == "jpeg_secondary" else ER12_CANDIDATE_EVIDENCE_VERSION), "W10 candidate evidence schema differs")
    _require(evidence.get("snr_db") == entry.get("snr_db") and evidence.get("modulation") == entry.get("modulation") and evidence.get("ldpc_rate") == entry.get("ldpc_rate"), "W10 candidate evidence candidate identity differs")
    if kind == "jpeg_secondary":
        _require(evidence.get("method") == "jpeg_clean_codec_measurement", "W10 JPEG candidate evidence method differs")
        _require(evidence.get("encode_axis_px") == entry.get("encode_axis_px") and evidence.get("quality") == entry.get("quality"), "W10 JPEG candidate evidence codec identity differs")
        analytic = entry.get("analytic_evidence")
        _require(isinstance(analytic, Mapping), "W10 JPEG analytic evidence is missing")
        _require(analytic.get("schema_version") == JPEG_ANALYTIC_EVIDENCE_VERSION and analytic.get("method") == "br4_analytic_composition", "W10 JPEG analytic method differs")
        if entry.get("status") == "eligible":
            _require(entry.get("expected_accuracy") is not None and entry.get("success_probability") is not None, "W10 eligible JPEG candidate has no analytic objective")
        else:
            _require(entry.get("expected_accuracy") is None and entry.get("success_probability") is None, "W10 uncharacterized JPEG candidate carries an analytic objective")
    else:
        _require(entry.get("status") == "eligible", "W10 ER-12 candidate status is not eligible")
        _require(evidence.get("method") == "er12_per_image_channel_simulation", "W10 ER-12 candidate evidence method differs")


def verify_selection_artifact(
    root: Path,
    kind: str,
    value: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    """Authenticate one frozen W10 selection artifact and recompute every winner."""

    root = Path(root).resolve()
    prefix = JPEG_SELECTION_PREFIX if kind == "jpeg_secondary" else ER12_SELECTION_PREFIX
    role = JPEG_SELECTION_ROLE if kind == "jpeg_secondary" else ER12_SELECTION_ROLE
    body = dict(value)
    identifier = body.pop("selection_id", None)
    _require(identifier == prefix + canonical_sha256(body), "W10 selection ID differs")
    _require(value.get("schema_version") == SELECTION_SCHEMA_VERSION and value.get("artifact_role") == role, "W10 selection role differs")
    _require(value.get("status") == SELECTION_STATUS, "W10 selection status differs")
    live_contract = _contract_identity(kind)
    _require(value.get("contract") == live_contract, "W10 selection contract differs from the live prospective contract")
    _require(value.get("contract_sha256") == _contract_sha256(kind), "W10 selection contract digest differs")
    manifest = load_w10_manifest(root, live=True)
    _require(value.get("source_epoch") == source_record(root, manifest), "W10 selection source epoch differs")
    _require(value.get("validation_only") is True and value.get("test") == "SEALED" and value.get("test_access") == 0, "W10 selection crossed the test boundary")
    _require(int(value.get("denominator")) == W10_VALIDATION_DENOMINATOR, "W10 selection denominator differs")
    _require(
        value.get("selection_evidence_model") == (
            "jpeg_compact_clean_codec_counts_plus_packet_bler_outage_composition_recomputed"
            if kind == "jpeg_secondary"
            else "er12_worker_generated_candidate_evidence_id_and_correctness_digest_recomputed"
        ),
        "W10 selection evidence model differs",
    )
    _require(
        value.get("selection_evidence_trust_boundary") == (
            "host_recomputes_br4_objective_from_bound_clean_codec_inputs_bler_cells_and_outage;"
            " classifier_clean_accuracy_is_a_bound_worker_input"
            if kind == "jpeg_secondary"
            else "host_recomputes_candidate_identity_and_cross_checks_content_addressed_worker_evidence_and_correctness_digest;"
            " awgn_ldpc_runtime_is_worker_authenticated"
        ),
        "W10 selection evidence trust boundary differs",
    )
    selections = value.get("selections")
    grid = list(snr_grid())
    _require(isinstance(selections, list) and len(selections) == len(grid), "W10 selection does not cover the exact SNR grid")
    _require([float(item["snr_db"]) for item in selections] == [float(value) for value in grid], "W10 selection SNR order differs")
    score_table = value.get("candidate_scores")
    _require(isinstance(score_table, list) and len(score_table) == len(grid), "W10 selection candidate score table differs")
    expected_candidates = _expected_candidate_set(kind, root)
    for point, snr in zip(score_table, grid, strict=True):
        _require(set(point) == {"snr_db", "candidates"}, "W10 candidate score point schema differs")
        _require(float(point["snr_db"]) == float(snr), "W10 candidate score SNR order differs")
        candidates = point["candidates"]
        _require(isinstance(candidates, list) and candidates, "W10 candidate score point is empty")
        for entry in candidates:
            _validate_score_entry(entry, kind=kind)
            _require(float(entry["snr_db"]) == float(snr), "W10 candidate score SNR differs")
            _require(_candidate_id(entry) in expected_candidates[str(int(snr))], "W10 candidate is outside the frozen candidate set")
            if kind == "jpeg_secondary":
                _verify_jpeg_analytic_candidate(root, entry)
            else:
                _verify_er12_candidate_evidence(root, entry)
        _require(
            {_candidate_id(entry) for entry in candidates} == expected_candidates[str(int(snr))],
            "W10 candidate score point does not cover the exact frozen candidate set",
        )
        winner = rank_candidates(kind, candidates)
        stored = dict(selections[grid.index(snr)])
        _require(stored == winner, "W10 selection winner is not the recomputed frozen winner")
    return dict(value)


def jpeg_phy_shortlist(root: Path) -> dict[int, dict[str, Any]]:
    """Reject the superseded JPEG2000-winner shortcut explicitly.

    Keeping this named entry point makes accidental reuse fail closed and gives
    mutation tests a precise guard: JPEG-secondary selection has no authority
    to import a pass-two JPEG2000 winner.
    """

    del root
    raise W10SelectionHold(
        "JPEG secondary selection cannot use the JPEG2000 pass-two PHY shortlist"
    )


def jpeg_candidate_space(root: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    """Derive JPEG's exact Cartesian candidate space from frozen configuration.

    Structural packet feasibility is evaluated once per configured
    modulation/rate pair.  Image-level JPEG budget feasibility remains a
    measured candidate input and never removes the candidate from this space;
    this is the BR-13 semantics that preserves an all-infeasible cell for
    adjudication rather than silently replacing it.
    """

    del root
    pairs = configured_phy_candidates(
        int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))
    )
    qualities = tuple(int(value) for value in get("baseline.jpeg_quality_grid"))
    axes = _configured_axes()
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for snr in snr_grid():
        entries = []
        for modulation, rate, _packet in pairs:
            for axis in axes:
                for quality in qualities:
                    entries.append({
                        "snr_db": int(snr),
                        "modulation": str(modulation),
                        "ldpc_rate": str(rate),
                        "encode_axis_px": int(axis),
                        "quality": int(quality),
                    })
        result[str(int(snr))] = tuple(entries)
    return result


def _expected_candidate_set(kind: str, root: Path) -> dict[str, set[str]]:
    """The exact prospective candidate space at each SNR, from its frozen source."""

    grid = list(snr_grid())
    if kind == "jpeg_secondary":
        expected: dict[str, set[str]] = {}
        for snr in grid:
            expected[str(int(snr))] = {
                _candidate_id(entry) for entry in jpeg_candidate_space(root)[str(int(snr))]
            }
        return expected

    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))
    pairs = tuple((str(modulation), str(rate)) for modulation, rate, _packet in configured_phy_candidates(k))
    return {
        str(int(snr)): {
            _candidate_id({"snr_db": int(snr), "modulation": modulation, "ldpc_rate": rate})
            for modulation, rate in pairs
        }
        for snr in grid
    }


@cache
def _br4_dependencies(root_text: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load the authoritative BR-4 table/outage inputs once per repository."""

    root = Path(root_text)
    # The committed table and merge report are the authoritative BLER inputs.
    # Selection verification must remain clean-clone safe; the worker-local
    # Pascal runtime is not a prerequisite for published evidence.
    table = load_successor_bler_table(verify_runtime=False)
    table_value = _read_json(root / TABLE_PATH.relative_to(REPO_ROOT), "successor BLER table")
    merge_value = _read_json(root / MERGE_REPORT_PATH.relative_to(REPO_ROOT), "successor BLER merge report")
    _require(table_value.get("table_id") and table_value.get("artifact_role") == "g8_c_pascal_successor_bler_table", "JPEG BLER table role differs")
    _require(merge_value.get("report_id") and merge_value.get("artifact_role") == "g8_c_pascal_successor_bler_merge_report", "JPEG BLER merge role differs")
    table_binding = {
        "path": str(TABLE_PATH.relative_to(REPO_ROOT)),
        "table_id": str(table_value["table_id"]),
        "sha256": _sha256_file(root / TABLE_PATH.relative_to(REPO_ROOT)),
        "merge_report_path": str(MERGE_REPORT_PATH.relative_to(REPO_ROOT)),
        "merge_report_id": str(merge_value["report_id"]),
        "merge_report_sha256": _sha256_file(root / MERGE_REPORT_PATH.relative_to(REPO_ROOT)),
        "lookup_mode": "exact_identity_with_declared_in_span_interpolation",
        "snr_convention": "es_n0_per_symbol",
        "uncharacterized_is_ineligible": True,
    }
    outage_path = root / OUTAGE_POLICY
    outage_value = _read_json(outage_path, "BR-13 outage policy")
    policy = load_outage_policy(
        outage_path,
        expected_dataset=W10_DATASET,
        expected_manifest_sha256=str(get(f"datasets.{W10_DATASET}.manifest_sha256")),
    )
    outage_binding = {
        "path": OUTAGE_POLICY,
        "sha256": _sha256_file(outage_path),
        "selection_policy": str(outage_value["selection_policy"]),
        "selected_class": int(policy.selected_class),
        "numerator": int(outage_value["numerator"]),
        "denominator": int(outage_value["denominator"]),
        "class_counts": [int(value) for value in outage_value["class_counts"]],
        "source": "results/baseline/w4/outage_policy.json",
    }
    return table, table_binding, {"record": outage_value, "policy": policy, "binding": outage_binding}


def _br4_packet_record(*, modulation: str, ldpc_rate: str) -> tuple[Any, list[dict[str, Any]]]:
    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))
    packet = build_packet_plan(k, modulation, ldpc_rate)
    _require(packet.feasible and packet.segmentation is not None, "JPEG candidate packet is not structurally feasible")
    layout = packet.segmentation
    identities = [
        {
            "k_and_n": [int(layout.k_prime), int(code_word_length)],
            "base_graph": int(layout.base_graph),
            "lifting_size": int(layout.lifting_size),
            "modulation": str(modulation),
            "decoder_algorithm": str(get("baseline.ldpc_decoder")),
            "decoder_offset": float(get("baseline.ldpc_decoder_offset")),
            "iterations": int(get("baseline.ldpc_max_iters")),
            "snr_convention": "es_n0_per_symbol",
            "rate": str(ldpc_rate),
        }
        for code_word_length in packet.e_r
    ]
    return packet, identities


def jpeg_analytic_evidence(
    root: Path,
    entry: Mapping[str, Any],
    *,
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the BR-4 objective from compact deterministic inputs."""

    _require(entry.get("snr_db") is not None, "JPEG candidate has no SNR")
    table, table_binding, outage = _br4_dependencies(str(Path(root).resolve()))
    packet, identities = _br4_packet_record(
        modulation=str(entry["modulation"]),
        ldpc_rate=str(entry["ldpc_rate"]),
    )
    clean_accuracy = MeasuredCodecAccuracy(
        correct=int(clean["correct"]),
        total=int(clean["total"]),
        split="val",
        source=str(clean["source"]),
    )
    outage_accuracy = measured_outage_accuracy_from_record(outage["record"])
    lookups = [
        table.lookup(identity, float(entry["snr_db"])) for identity in identities
    ]
    if all(lookup.characterized for lookup in lookups):
        if int(clean.get("infeasible_count", 0)):
            # BR-4's measured clean accuracy is defined only for a candidate
            # whose exact codec emitted a codestream for every validation
            # image.  Keep the image-level outage accounting in the compact
            # evidence, but do not turn a partially infeasible codec into an
            # analytic score or silently apply transport BLER to a row that
            # never had a payload.
            composition = None
            composition_record = None
            eligibility = "uncharacterized"
        else:
            composition = compose(
                [lookup.require() for lookup in lookups],
                codec_accuracy=clean_accuracy,
                outage_accuracy=outage_accuracy,
            )
            composition_record = composition.as_record()
            eligibility = "eligible"
    else:
        # BR-4 carries a structurally valid but uncharacterized candidate in
        # the complete candidate table and excludes it from ranking.  It must
        # never be converted into a guessed low score or BLER zero.
        composition = None
        composition_record = None
        eligibility = "uncharacterized"
    return {
        "schema_version": JPEG_ANALYTIC_EVIDENCE_VERSION,
        "method": "br4_analytic_composition",
        "snr_db": int(entry["snr_db"]),
        "modulation": str(entry["modulation"]),
        "ldpc_rate": str(entry["ldpc_rate"]),
        "encode_axis_px": int(entry["encode_axis_px"]),
        "quality": int(entry["quality"]),
        "clean": dict(clean),
        "packet": {
            "k_symbols": int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}")),
            "modulation": str(entry["modulation"]),
            "ldpc_rate": str(entry["ldpc_rate"]),
            "packet_metadata": packet.metadata(),
            "block_identities": identities,
        },
        "bler": {
            **table_binding,
            "lookups": [lookup.as_record() for lookup in lookups],
        },
        "outage": dict(outage["binding"]),
        "composition": composition_record,
        "eligibility": eligibility,
    }


def bind_candidate_evidence(body: Mapping[str, Any], *, prefix: str = "w10candidateevidence-") -> dict[str, Any]:
    """Content-address one compact worker-generated candidate evidence body."""

    value = dict(body)
    value["evidence_id"] = prefix + canonical_sha256({key: item for key, item in value.items() if key != "evidence_id"})
    return value


def scorer_binding(root: Path) -> dict[str, Any]:
    """Bind the clean validation scorer to its frozen BR-12 checkpoint."""

    freeze_path = Path(root) / BR12_FREEZE_PATH
    freeze = _read_json(freeze_path, "BR-12 classifier freeze")
    return {
        "path": BR12_FREEZE_PATH,
        "sha256": _sha256_file(freeze_path),
        "freeze_id": str(freeze["freeze_id"]),
        "classifier_variant": str(freeze["classifier_variant"]),
        "scorer_identity": str(freeze["scorer_identity"]),
        "checkpoint_id": str(freeze["checkpoint_id"]),
        "checkpoint_sha256": str(freeze["checkpoint_file_sha256"]),
    }


def _verify_jpeg_analytic_candidate(root: Path, entry: Mapping[str, Any]) -> None:
    analytic = entry["analytic_evidence"]
    clean = analytic.get("clean")
    _require(isinstance(clean, Mapping), "W10 JPEG clean evidence is missing")
    _require(clean.get("correct") == entry.get("n_correct") and clean.get("total") == entry.get("n_total"), "W10 JPEG clean counts differ")
    _require(clean.get("feasible_count") == entry.get("n_delivered"), "W10 JPEG codec-feasible count differs")
    _require(clean.get("decode_failure_count") == entry.get("n_decode_failure") and clean.get("infeasible_count") == entry.get("n_infeasible"), "W10 JPEG clean verdict counts differ")
    _require(clean.get("correct_digest") == entry.get("per_image_correct_digest"), "W10 JPEG clean correctness digest differs")
    _require(
        isinstance(clean.get("outage_correct_count"), int)
        and not isinstance(clean["outage_correct_count"], bool)
        and 0 <= clean["outage_correct_count"] <= clean["infeasible_count"],
        "W10 JPEG outage accounting differs",
    )
    _require(isinstance(clean.get("emitted_bytes_digest"), str) and _full_sha(clean["emitted_bytes_digest"]), "W10 JPEG emitted-byte digest differs")
    evidence = entry["candidate_evidence"]
    _require(evidence.get("clean") == dict(clean), "W10 JPEG candidate evidence clean inputs differ")
    _require(evidence.get("selection_contract_sha256") == _contract_sha256("jpeg_secondary"), "W10 JPEG candidate evidence contract differs")
    _require(evidence.get("scorer_binding") == clean.get("scorer_binding"), "W10 JPEG candidate evidence scorer binding differs")
    _require(evidence.get("scorer_binding") == scorer_binding(root), "W10 JPEG scorer binding differs from BR-12 freeze")
    manifest = load_w10_manifest(root, live=True)
    _require(evidence.get("source_epoch") == source_record(root, manifest), "W10 JPEG candidate evidence source epoch differs")
    recomputed = jpeg_analytic_evidence(root, entry, clean=clean)
    _require(dict(analytic) == recomputed, "W10 JPEG analytic evidence does not independently recompute")
    if entry.get("status") == "eligible":
        _require(recomputed.get("eligibility") == "eligible", "W10 JPEG candidate eligibility differs")
        _require(float(entry["expected_accuracy"]) == float(recomputed["composition"]["expected_accuracy"]), "W10 JPEG expected objective differs")
        _require(float(entry["success_probability"]) == float(recomputed["composition"]["success_probability"]), "W10 JPEG transport success probability differs")
    else:
        _require(recomputed.get("eligibility") == "uncharacterized", "W10 JPEG uncharacterized status differs")
        _require(entry.get("expected_accuracy") is None and entry.get("success_probability") is None, "W10 JPEG uncharacterized score differs")


def _verify_er12_candidate_evidence(root: Path, entry: Mapping[str, Any]) -> None:
    evidence = entry["candidate_evidence"]
    _require(evidence.get("method") == "er12_per_image_channel_simulation", "W10 ER-12 evidence method differs")
    _require(evidence.get("protocol_version") == int(_parameter("w10_er12_protocol_version")), "W10 ER-12 evidence protocol version differs")
    _require(evidence.get("label_bits") == int(_parameter("w10_er12_label_bits")), "W10 ER-12 evidence label width differs")
    _require(evidence.get("payload_frame") == str(_parameter("w10_er12_payload_frame")), "W10 ER-12 evidence frame differs")
    _require(evidence.get("true_label_in_payload") is False, "W10 ER-12 evidence carries true labels")
    outcomes = evidence.get("outcome_counts")
    _require(
        isinstance(outcomes, Mapping)
        and outcomes.get("delivered") == int(entry["n_delivered"])
        and outcomes.get("decode_failure") == int(entry["n_decode_failure"])
        and outcomes.get("infeasible") == int(entry["n_infeasible"]),
        "W10 ER-12 outcome evidence differs",
    )
    _require(evidence.get("correct") == int(entry["n_correct"]) and evidence.get("total") == int(entry["n_total"]), "W10 ER-12 correctness counts differ")
    _require(evidence.get("correct_digest") == entry["per_image_correct_digest"], "W10 ER-12 correctness digest differs")
    _require(evidence.get("noise_convention") == NOISE_CONVENTION and isinstance(evidence.get("noise_ids_digest"), str) and _full_sha(evidence["noise_ids_digest"]), "W10 ER-12 noise evidence differs")
    worker = evidence.get("worker_evidence")
    _require(isinstance(worker, Mapping), "W10 ER-12 worker evidence binding is missing")
    _require(worker.get("artifact_role") == "W10_ER12_CANDIDATE_WORKER_EVIDENCE", "W10 ER-12 worker evidence role differs")
    _require(worker.get("execution_profile_id") == "confessor_pascal_cu126", "W10 ER-12 worker evidence profile differs")
    _require(worker.get("snr_db") == entry.get("snr_db") and worker.get("modulation") == entry.get("modulation") and worker.get("ldpc_rate") == entry.get("ldpc_rate"), "W10 ER-12 worker evidence candidate differs")
    _require(isinstance(worker.get("checkpoint_id"), str) and worker.get("checkpoint_id"), "W10 ER-12 worker evidence checkpoint is missing")
    _require(worker.get("test_access") == 0, "W10 ER-12 worker evidence crossed the test boundary")
    worker_body = dict(worker)
    worker_id = worker_body.pop("worker_evidence_id", None)
    _require(worker_id == "w10er12workerevidence-" + canonical_sha256(worker_body), "W10 ER-12 worker evidence ID differs")
    manifest = load_w10_manifest(root, live=True)
    _require(worker.get("source_epoch") == source_record(root, manifest), "W10 ER-12 worker source epoch differs")
    _require(worker.get("selection_contract_sha256") == _contract_sha256("er12_label_bound"), "W10 ER-12 worker contract differs")
    _require(
        worker.get("correct") == evidence.get("correct")
        and worker.get("total") == evidence.get("total")
        and worker.get("correct_digest") == evidence.get("correct_digest")
        and worker.get("outcome_counts") == evidence.get("outcome_counts")
        and worker.get("noise_ids_digest") == evidence.get("noise_ids_digest"),
        "W10 ER-12 worker evidence outcome differs",
    )


def validate_selected_point(value: Mapping[str, Any], *, snr_db: float) -> Mapping[str, Any]:
    for item in value["selections"]:
        if float(item["snr_db"]) == float(snr_db):
            return item
    raise W10SelectionHold(f"W10 selection has no point at {snr_db}")


def score_vector_digest(correct: Sequence[bool]) -> str:
    """The per-candidate 1000-outcome digest recorded instead of 1000 rows."""

    return canonical_sha256({"correct": [bool(item) for item in correct]})


__all__ = [
    "ER12_SELECTION_PATH",
    "ER12_SELECTION_PREFIX",
    "ER12_SELECTION_ROLE",
    "ER12_TIE_BREAK",
    "JPEG_SELECTION_PATH",
    "JPEG_SELECTION_PREFIX",
    "JPEG_SELECTION_ROLE",
    "JPEG_TIE_BREAK",
    "OUTAGE_POLICY",
    "SELECTION_STATUS",
    "SelectionContract",
    "W10SelectionHold",
    "build_selection_artifact",
    "bind_candidate_evidence",
    "er12_contract_sha256",
    "er12_rank_key",
    "er12_selection_contract",
    "jpeg_contract_sha256",
    "jpeg_phy_shortlist",
    "jpeg_candidate_space",
    "jpeg_analytic_evidence",
    "jpeg_rank_key",
    "jpeg_selection_contract",
    "rank_candidates",
    "score_vector_digest",
    "scorer_binding",
    "selection_identity",
    "validate_selected_point",
    "verify_selection_artifact",
]
