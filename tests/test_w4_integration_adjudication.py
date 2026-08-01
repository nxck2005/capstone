"""Mutation tests for the W4 closing adjudication and the selection verifier.

`tests/test_w4_verification.py` covers the bounded *evidence*; this module
covers what PB_3 added on top of it — the integration adjudication and the
live checks that the BR-4 selection machinery still fails closed at HEAD.

Every test here mutates exactly one property and asserts the verifier fails for
*that* property rather than for something incidental, because a check that
fires on everything proves nothing about any one thing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import verify_w4_baseline_integration as verifier
from baseline.classical import composition
from config.params import REPO_ROOT, get

EVIDENCE = REPO_ROOT / "results" / "baseline" / "w4"
ADJUDICATION = "integration_adjudication.json"


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    """A copy of the committed W4 evidence, safe to mutate."""

    directory = tmp_path / "w4"
    shutil.copytree(EVIDENCE, directory)
    return directory


def _mutate(evidence: Path, mutate) -> None:
    path = evidence / ADJUDICATION
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _fails(evidence: Path, match: str) -> None:
    with pytest.raises(verifier.VerificationError, match=match):
        verifier.check_integration_adjudication(evidence)


# ---------------------------------------------------------------------------
# The committed adjudication verifies
# ---------------------------------------------------------------------------


def test_the_committed_adjudication_verifies(evidence: Path) -> None:
    payload = verifier.check_integration_adjudication(evidence)
    assert payload["claims"]["g8_status"] == "unresolved"
    assert payload["selection_machinery"]["passes_executed"] == 0


def test_the_committed_selection_machinery_behaviour_verifies() -> None:
    verifier.check_selection_machinery_behaviour()


def test_the_adjudication_is_regenerable_from_the_repository() -> None:
    import gen_w4_integration_adjudication as generator

    committed = json.loads((EVIDENCE / ADJUDICATION).read_text())
    regenerated = json.loads(json.dumps(generator.build()))
    committed.pop("head_at_generation", None)
    regenerated.pop("head_at_generation", None)
    assert committed == regenerated


# ---------------------------------------------------------------------------
# Adjudication content
# ---------------------------------------------------------------------------


def test_a_missing_adjudication_is_caught(evidence: Path) -> None:
    (evidence / ADJUDICATION).unlink()
    _fails(evidence, "missing W4 evidence file")


def test_falsely_claiming_a_full_sweep_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["claims"].__setitem__("br4_full_validation_sweep", True),
    )
    _fails(evidence, "claims br4_full_validation_sweep")


def test_falsely_claiming_a_completed_sweep_in_prose_is_caught(
    evidence: Path,
) -> None:
    _mutate(
        evidence,
        lambda p: p.__setitem__(
            "note", "the full validation sweep completed over the whole grid"
        ),
    )
    _fails(evidence, "full validation sweep completed")


def test_falsely_claiming_g8_resolution_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["claims"].__setitem__("g8_status", "resolved"))
    _fails(evidence, "does not record G-8 as unresolved")


def test_falsely_claiming_g8_started_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["remaining"].__setitem__("g8_started", True))
    _fails(evidence, "claims the BR-4 sweep or G-8 has started")


def test_falsely_claiming_an_operating_point_selection_is_caught(
    evidence: Path,
) -> None:
    _mutate(
        evidence,
        lambda p: p["claims"].__setitem__("g8_operating_point_selection", True),
    )
    _fails(evidence, "claims g8_operating_point_selection")


def test_claiming_training_or_finetuning_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["claims"].__setitem__("training_performed", True))
    _fails(evidence, "claims training_performed")


def test_dropping_the_test_split_seal_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["claims"].__setitem__("test_split_sealed", False))
    _fails(evidence, "does not declare the test split sealed")


def test_wrong_evidence_labels_are_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p.__setitem__("evidence_labels", ["ok"]))
    _fails(evidence, "wrong evidence labels")


def test_a_weakened_prominent_declaration_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p.__setitem__(
            "prominent_declaration", "W4 classical baseline results."
        ),
    )
    _fails(evidence, "does not state")


def test_a_recorded_evidence_commit_is_rejected(evidence: Path) -> None:
    _mutate(evidence, lambda p: p.__setitem__("evidence_commit", "0" * 40))
    _fails(evidence, "records an evidence_commit")


def test_a_null_evidence_commit_is_rejected_too(evidence: Path) -> None:
    _mutate(evidence, lambda p: p.__setitem__("evidence_commit", None))
    _fails(evidence, "records an evidence_commit")


def test_removing_the_evidence_commit_resolution_policy_is_caught(
    evidence: Path,
) -> None:
    _mutate(evidence, lambda p: p.__setitem__("evidence_commit_resolution", "  "))
    _fails(evidence, "no evidence-commit resolution policy")


# ---------------------------------------------------------------------------
# Bound hashes and sources
# ---------------------------------------------------------------------------


def test_a_stale_evidence_file_hash_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["evidence_files"].__setitem__("aggregate.csv", "0" * 64),
    )
    _fails(evidence, "does not match the hash bound")


def test_an_edited_evidence_file_is_caught(evidence: Path) -> None:
    path = evidence / "accounting_examples.json"
    path.write_text(path.read_text() + "\n")
    _fails(evidence, "does not match the hash bound")


def test_an_omitted_bound_evidence_file_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["evidence_files"].pop("per_image.csv"))
    _fails(evidence, "bound evidence set differs")


def test_a_stale_selection_source_hash_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_sources"][0]["sha256"] = "0" * 64

    _mutate(evidence, mutate)
    _fails(evidence, "has drifted since the adjudication was written")


def test_a_stale_selection_source_length_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_sources"][0]["bytes"] += 1

    _mutate(evidence, mutate)
    _fails(evidence, "has drifted since the adjudication was written")


def test_an_omitted_selection_source_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_sources"] = payload["selection_sources"][:1]

    _mutate(evidence, mutate)
    _fails(evidence, "bound selection sources differ")


def test_the_selection_module_is_not_bound_into_the_execution_manifest() -> None:
    """It postdates the bounded measurement and must not claim to precede it."""

    assert "src/baseline/classical/composition.py" not in verifier.EXPECTED_SOURCES


def test_a_changed_bounded_execution_commit_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p.__setitem__("bounded_evidence_execution_commit", "0" * 40),
    )
    _fails(evidence, "different bounded-run execution commit")


# ---------------------------------------------------------------------------
# Recomputed content
# ---------------------------------------------------------------------------


def test_an_altered_worked_example_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["composition"]["worked_example"]["expected_accuracy"] = 0.99

    _mutate(evidence, mutate)
    _fails(evidence, "worked composition example does not reproduce")


def test_altered_composition_arithmetic_is_caught(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation class: the composition itself changes under the evidence."""

    real = composition.compose

    def biased(block_blers, **kwargs):  # type: ignore[no-untyped-def]
        result = real(block_blers, **kwargs)
        return composition.CompositionResult(
            success_probability=result.success_probability,
            expected_accuracy=result.expected_accuracy + 0.01,
            code_blocks=result.code_blocks,
            block_blers=result.block_blers,
            codec_accuracy=result.codec_accuracy,
            outage_accuracy=result.outage_accuracy,
        )

    monkeypatch.setattr(composition, "compose", biased)
    _fails(evidence, "worked composition example does not reproduce")


def test_an_altered_outage_measurement_is_caught(evidence: Path) -> None:
    _mutate(evidence, lambda p: p["outage"].__setitem__("numerator", 250))
    _fails(evidence, "outage numerator disagrees")


def test_an_outage_accuracy_that_is_not_its_own_ratio_is_caught(
    evidence: Path,
) -> None:
    _mutate(
        evidence,
        lambda p: p["outage"].__setitem__("measured_validation_accuracy", 0.5),
    )
    _fails(evidence, "not its own numerator/denominator")


def test_dropping_the_uniform_outage_rejection_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["outage"].__setitem__(
            "assumed_uniform_accuracy_rejected", False
        ),
    )
    _fails(evidence, "does not reject the assumed")


def test_an_altered_bler_characterization_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["bler_characterization"]["characterized"][0]["support_db"] = [
            -100.0,
            100.0,
        ]

    _mutate(evidence, mutate)
    _fails(evidence, "BLER characterisation does not match")


def test_claiming_extrapolation_is_permitted_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["bler_characterization"]["extrapolation_permitted"] = True

    _mutate(evidence, mutate)
    _fails(evidence, "BLER characterisation does not match")


def test_an_incomplete_recorded_bler_key_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["bler_characterization"]["required_identity_fields"] = ["modulation"]

    _mutate(evidence, mutate)
    _fails(evidence, "BLER characterisation does not match")


def test_an_altered_selection_machinery_description_is_caught(
    evidence: Path,
) -> None:
    """The blanket backstop, on a field no named check covers.

    PB_3C gave the frozen-policy fields their own named checks, which fire
    first and report *which* rule moved.  This test deliberately mutates a
    field outside that set, so it still exercises the whole-dict comparison
    rather than silently becoming a duplicate of a named check.
    """

    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_machinery"][
            "uncharacterized_candidates_are"
        ] = "scored at zero"

    _mutate(evidence, mutate)
    _fails(evidence, "selection-machinery description does not match")


def test_recording_executed_selection_passes_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_machinery"]["passes_executed"] = 1

    _mutate(evidence, mutate)
    _fails(evidence, "selection-machinery description does not match")


def test_an_inflated_sweep_guard_limit_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["sweep_guard"]["max_candidates"] = 1_000_000

    _mutate(evidence, mutate)
    _fails(evidence, "sweep-guard limits do not match")


def test_claiming_the_repository_is_authorized_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["sweep_guard"]["repository_default_authorized"] = True

    _mutate(evidence, mutate)
    _fails(evidence, "sweep-guard limits do not match")


# ---------------------------------------------------------------------------
# Provisional operating points and test-split counters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["efficiency_ratio", "crossover_ratio", "low_ratio_operating_point"]
)
def test_a_changed_provisional_operating_ratio_is_caught(
    evidence: Path, name: str
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["provisional_operating_points"][name]["value"] = "r_1_48"

    _mutate(evidence, mutate)
    _fails(evidence, f"provisional {name} has moved")


@pytest.mark.parametrize(
    "name", ["efficiency_ratio", "crossover_ratio", "low_ratio_operating_point"]
)
def test_declaring_a_provisional_ratio_settled_is_caught(
    evidence: Path, name: str
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["provisional_operating_points"][name]["status"] = "selected_at_G-8"

    _mutate(evidence, mutate)
    _fails(evidence, "no longer provisional_until_G-8")


def test_claiming_w4_selected_an_operating_point_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["provisional_operating_points"]["crossover_ratio"][
            "selected_by_w4"
        ] = True

    _mutate(evidence, mutate)
    _fails(evidence, "claims W4 selected")


def test_a_dropped_provisional_operating_point_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["provisional_operating_points"].pop("crossover_ratio")

    _mutate(evidence, mutate)
    _fails(evidence, "all three provisional operating")


@pytest.mark.parametrize(
    "counter",
    ["decoder_calls", "canonicalization_calls", "inference_calls", "accuracy_calls"],
)
def test_test_access_counter_drift_is_caught(evidence: Path, counter: str) -> None:
    _mutate(evidence, lambda p: p["test_split_access"].__setitem__(counter, 1))
    _fails(evidence, f"non-zero {counter}")


def test_a_wrong_test_access_gate_is_caught(evidence: Path) -> None:
    _mutate(
        evidence, lambda p: p["test_split_access"].__setitem__("release_gate", "G-10")
    )
    _fails(evidence, "wrong test-access gate")


def test_the_recorded_test_access_gate_is_the_configured_one(
    evidence: Path,
) -> None:
    payload = json.loads((evidence / ADJUDICATION).read_text())
    assert payload["test_split_access"]["release_gate"] == get(
        "evaluation.test_access_gate"
    )


# ---------------------------------------------------------------------------
# Live behavioural mutations
# ---------------------------------------------------------------------------


class _AlwaysCharacterized:
    """A BLER table that answers every lookup, which is the defect."""

    def __init__(self, real: Any, *, on_partial_key: bool = True) -> None:
        self._real = real
        self._on_partial_key = on_partial_key

    def lookup(self, key: Any, snr_db: float) -> composition.BlerLookup:
        identity = composition.BlerIdentity.from_mapping(key)
        return composition.BlerLookup(
            status=composition.CHARACTERIZED,
            identity=identity,
            snr_db=snr_db,
            bler=0.001,
        )


def test_an_uncharacterized_identity_reported_as_characterized_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = composition.g2_bler_table()
    monkeypatch.setattr(
        composition, "g2_bler_table", lambda: _AlwaysCharacterized(real)
    )
    with pytest.raises(
        verifier.VerificationError, match="uncharacterized BLER identity"
    ):
        verifier.check_selection_machinery_behaviour()


def test_silent_extrapolation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    real = composition.g2_bler_table()

    class _Extrapolating:
        def lookup(self, key: Any, snr_db: float) -> composition.BlerLookup:
            result = real.lookup(key, snr_db)
            if result.characterized or result.reason == "identity_not_characterized":
                return result
            # Outside support: quietly answer anyway.
            return composition.BlerLookup(
                status=composition.CHARACTERIZED,
                identity=result.identity,
                snr_db=snr_db,
                bler=0.0,
            )

    monkeypatch.setattr(composition, "g2_bler_table", lambda: _Extrapolating())
    with pytest.raises(verifier.VerificationError, match="extrapolated beyond"):
        verifier.check_selection_machinery_behaviour()


def test_an_accepted_incomplete_bler_key_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = composition.g2_bler_table()

    class _Lenient:
        def lookup(self, key: Any, snr_db: float) -> composition.BlerLookup:
            full = dict(key)
            full.setdefault("lifting_size", 22)
            return real.lookup(full, snr_db)

    monkeypatch.setattr(composition, "g2_bler_table", lambda: _Lenient())
    with pytest.raises(
        verifier.VerificationError, match="incomplete BLER lookup key was accepted"
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_feasibility_cache_collision_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition.Candidate,
        "feasibility_key",
        lambda self: (
            self.dataset,
            self.ratio,
            self.ldpc_rate,
            self.encode_axis_px,
        ),
    )
    with pytest.raises(verifier.VerificationError, match="collides on modulation"):
        verifier.check_selection_machinery_behaviour()


def test_a_nondeterministic_tie_result_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def first_wins(evaluations):  # type: ignore[no-untyped-def]
        ordered = tuple(evaluations)
        return composition.Selection(
            selected=ordered[0],
            tied=ordered,
            tie_break_applied=True,
            evaluations=ordered,
        )

    monkeypatch.setattr(composition, "select_best", first_wins)
    with pytest.raises(
        verifier.VerificationError, match="not order-independent"
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_permitted_third_pass_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Unlimited(composition.SelectionCampaign):
        def run_pass(self, pass_id: Any, selector: Any, *, scorer: str) -> Any:
            selector(composition.PassContext(pass_id, {}))
            return composition.PassResult(
                pass_id=pass_id, mode=self.mode, scorer=scorer, selections=()
            )

    monkeypatch.setattr(composition, "SelectionCampaign", _Unlimited)
    with pytest.raises(
        verifier.VerificationError, match="further selection pass 3 was permitted"
    ):
        verifier.check_selection_machinery_behaviour()


def test_leaked_pass_state_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Leaky(composition.PassContext):
        def result_of(self, pass_id: int) -> Any:
            return composition.PassResult(
                pass_id=pass_id, mode="classical_adaptive", scorer="x", selections=()
            )

    monkeypatch.setattr(composition, "PassContext", _Leaky)

    class _Campaign(composition.SelectionCampaign):
        def run_pass(self, pass_id: Any, selector: Any, *, scorer: str) -> Any:
            selector(_Leaky(pass_id, {}))
            return composition.PassResult(
                pass_id=pass_id, mode=self.mode, scorer=scorer, selections=()
            )

    monkeypatch.setattr(composition, "SelectionCampaign", _Campaign)
    with pytest.raises(
        verifier.VerificationError, match="read its own or a later pass"
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_bypassed_sweep_guard_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        composition,
        "check_sweep_budget",
        lambda **kwargs: composition.sweep_budget(None),
    )
    with pytest.raises(
        verifier.VerificationError, match="guard permitted an over-budget run"
    ):
        verifier.check_selection_machinery_behaviour()


def test_an_environment_variable_bypass_would_be_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Proved by pointing the scan at a module that does consult one."""

    original = (REPO_ROOT / "src/baseline/classical/composition.py").read_text()
    poisoned = tmp_path / "src" / "baseline" / "classical"
    poisoned.mkdir(parents=True)
    (poisoned / "composition.py").write_text(
        original + '\n_BYPASS = os.environ.get("ALLOW_G8_SWEEP")\n'
    )
    for name in ("results",):
        (tmp_path / name).symlink_to(REPO_ROOT / name)
    (tmp_path / "tests").symlink_to(REPO_ROOT / "tests")
    monkeypatch.setattr(verifier, "REPO", tmp_path)
    with pytest.raises(verifier.VerificationError, match="consults os.environ"):
        verifier.check_selection_machinery_behaviour()


# ---------------------------------------------------------------------------
# PB_3C — the corrected selection semantics and the frozen selection policy
# ---------------------------------------------------------------------------
#
# Two idioms, and the split is deliberate.  A *recorded claim* that the
# machinery behaves is mutated as JSON and must be caught by
# `check_integration_adjudication`.  A *behaviour* is mutated in the live module
# and must be caught by `check_selection_machinery_behaviour` — a defect in the
# code has to fail a probe, not merely disagree with prose about itself.


def test_a_changed_configured_fixed_modulation_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_machinery"]["fixed_modulation"][
            "configured_value"
        ] = "qam16"

    _mutate(evidence, mutate)
    _fails(evidence, "records fixed modulation")


def test_claiming_the_fixed_curve_searches_modulations_is_caught(
    evidence: Path,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_machinery"]["fixed_modulation"][
            "searches_modulations"
        ] = True

    _mutate(evidence, mutate)
    _fails(evidence, "searches\nmodulations|searches modulations")


def test_a_wrong_fixed_modulation_source_is_caught(evidence: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_machinery"]["fixed_modulation"]["source"] = (
            "params.baseline.modulations"
        )

    _mutate(evidence, mutate)
    _fails(evidence, "core_modulation")


def test_a_mutated_tie_break_order_is_caught(evidence: Path) -> None:
    """Swapping two keys still yields a total order — and is still caught."""

    def mutate(payload: dict[str, Any]) -> None:
        order = payload["selection_machinery"]["tie_break_order"]
        order[2], order[3] = order[3], order[2]

    _mutate(evidence, mutate)
    _fails(evidence, "tie-break order is not the implementation's")


def test_a_missing_pre_g8_freeze_declaration_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["selection_machinery"].__setitem__(
            "tie_break_frozen_before_g8", False
        ),
    )
    _fails(evidence, "frozen before")


def test_freezing_against_the_wrong_gate_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["selection_machinery"].__setitem__(
            "tie_break_frozen_against_gate", "G-10"
        ),
    )
    _fails(evidence, "wrong gate")


def test_introducing_a_tie_tolerance_is_caught(evidence: Path) -> None:
    """Exact equality is the tie definition; a tolerance makes it negotiable."""

    _mutate(
        evidence,
        lambda p: p["selection_machinery"].__setitem__(
            "tie_equality", "within 1e-9"
        ),
    )
    _fails(evidence, "exact equality")


def test_a_stale_selection_policy_fingerprint_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["selection_machinery"].__setitem__(
            "selection_policy_sha256", "0" * 64
        ),
    )
    _fails(evidence, "does not reproduce")


def test_narrowing_the_fingerprint_coverage_is_caught(evidence: Path) -> None:
    """Dropping a covered field would let that field move unnoticed."""

    def mutate(payload: dict[str, Any]) -> None:
        fields = payload["selection_machinery"]["selection_policy_fields"]
        fields.remove("tie_break_order")

    _mutate(evidence, mutate)
    _fails(evidence, "covers different\\s+fields")


def test_dropping_the_resumed_state_rule_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["selection_machinery"].__setitem__(
            "resumed_state_validation", "trusted"
        ),
    )
    _fails(evidence, "exact ordered prefix")


def test_claiming_g8_characterization_has_started_is_caught(evidence: Path) -> None:
    _mutate(
        evidence,
        lambda p: p["claims"].__setitem__("g8_characterization_started", True),
    )
    _fails(evidence, "claims g8_characterization_started")


def test_an_out_of_date_adjudication_schema_version_is_caught(
    evidence: Path,
) -> None:
    _mutate(evidence, lambda p: p.__setitem__("schema_version", 1))
    _fails(evidence, "unexpected schema_version")


# -- live behavioural mutations ---------------------------------------------


def test_a_fixed_mod_curve_selecting_a_non_core_modulation_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect PB_3C removed: search the grid instead of reading the config."""

    real = composition.resolve_curve

    def searching(mode: str, evaluations_by_snr: Any) -> Any:
        if mode != composition.CLASSICAL_FIXED_MOD:
            return real(mode, evaluations_by_snr)
        # Whichever modulation wins the whole grid, exactly as before.
        grid = tuple(sorted(evaluations_by_snr))
        best = max(
            {
                e.candidate.modulation
                for evaluations in evaluations_by_snr.values()
                for e in evaluations
            },
            key=lambda modulation: sum(
                composition.select_best(
                    [
                        e
                        for e in evaluations_by_snr[snr]
                        if e.candidate.modulation == modulation
                    ]
                ).selected.composition.expected_accuracy
                for snr in grid
            ),
        )
        return composition.CurveSelection(
            mode,
            tuple(
                (
                    snr,
                    composition.select_best(
                        [
                            e
                            for e in evaluations_by_snr[snr]
                            if e.candidate.modulation == best
                        ]
                    ),
                )
                for snr in grid
            ),
            held_fixed={"modulation": best, "source": "whole-grid search"},
        )

    monkeypatch.setattr(composition, "resolve_curve", searching)
    with pytest.raises(
        verifier.VerificationError, match="rather than the configured"
    ):
        verifier.check_selection_machinery_behaviour()


def _trusting_campaign() -> Any:
    """A SelectionCampaign that admits resumed state without validating it."""

    class _Trusting(composition.SelectionCampaign):
        def _admit_resumed_sequence(self, completed: Any) -> None:
            for result in completed:
                self._completed[result.pass_id] = result

    return _Trusting


def test_resumed_pass_two_without_pass_one_being_admitted_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(composition, "SelectionCampaign", _trusting_campaign())
    with pytest.raises(
        verifier.VerificationError,
        match="admitted pass two with no pass one",
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_reversed_resumed_sequence_being_admitted_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sorting malformed state into validity must not pass the probe."""

    class _Sorting(composition.SelectionCampaign):
        def _admit_resumed_sequence(self, completed: Any) -> None:
            # Sorts first, then applies the prefix rule to the *sorted* list --
            # so it still refuses pass two alone, and the only thing it lets
            # through is precisely the reversal the probe is looking for.
            ordered = sorted(completed, key=lambda result: result.pass_id)
            seen: set[str] = set()
            for index, result in enumerate(ordered):
                if result.pass_id != self._allowed[index] or result.scorer in seen:
                    raise composition.SelectionPassError("refused")
                seen.add(result.scorer)
                self._completed[result.pass_id] = result

    monkeypatch.setattr(composition, "SelectionCampaign", _Sorting)
    with pytest.raises(
        verifier.VerificationError, match="admitted a reversed pass sequence"
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_duplicate_resumed_scorer_being_admitted_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoScorerRule(composition.SelectionCampaign):
        def _require_scorer(self, scorer: Any, pass_id: int, what: str) -> str:
            return str(scorer)

    monkeypatch.setattr(composition, "SelectionCampaign", _NoScorerRule)
    with pytest.raises(
        verifier.VerificationError,
        match="the same scorer reused across both passes",
    ):
        verifier.check_selection_machinery_behaviour()


def test_a_malformed_stored_selection_being_admitted_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoSelectionRule(composition.SelectionCampaign):
        def _require_selections(
            self, selections: Any, pass_id: int, verb: str
        ) -> Any:
            return tuple(selections)

    monkeypatch.setattr(composition, "SelectionCampaign", _NoSelectionRule)
    with pytest.raises(
        verifier.VerificationError,
        match="a stored pass holding a malformed selection",
    ):
        verifier.check_selection_machinery_behaviour()
