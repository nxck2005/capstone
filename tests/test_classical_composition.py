"""BR-4 analytic composition — arithmetic, worked examples and mutations.

The composition is the single funnel every BR-4 selection flows through
(AM-51), so the tests here are hand-checked worked examples rather than
round-trips through the implementation: a round-trip cannot tell a correct
mixture from a plausible one.

Two properties get their own mutation tests because they fail *silently*:
substituting ``1 / n_classes`` for the measured outage accuracy (the two are
numerically equal on the committed stratified manifest), and reconstructing an
accuracy from an assumed value rather than passing the measured one through.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest

from baseline.classical.composition import (
    BLER_IDENTITY_FIELDS,
    BLER_REQUIRED_FIELDS,
    CLASSICAL_ADAPTIVE,
    CLASSICAL_FIXED_MCS,
    CLASSICAL_FIXED_MOD,
    ELIGIBLE,
    FEASIBILITY_KEY_EXCLUSIONS,
    FEASIBILITY_KEY_FIELDS,
    INFEASIBLE,
    PASS_ONE,
    PASS_TWO,
    SYSTEM_MODES,
    TIE_BREAK_ORDER,
    UNCHARACTERIZED,
    BlerLookupError,
    Candidate,
    CandidateEvaluation,
    CompositionError,
    Feasibility,
    FeasibilityCache,
    PassContext,
    PassResult,
    SelectionCampaign,
    SelectionPassError,
    MeasuredCodecAccuracy,
    MeasuredOutageAccuracy,
    compose,
    expected_accuracy,
    measured_outage_accuracy_from_record,
    UncharacterizedBlerError,
    g2_bler_table,
    mode_policy,
    resolve_curve,
    select_best,
    selection_passes,
    transport_block_success_probability,
)
from config.params import REPO_ROOT, get

OUTAGE_POLICY_PATH = REPO_ROOT / "results" / "baseline" / "w4" / "outage_policy.json"


def _codec(correct: int = 870, total: int = 1000) -> MeasuredCodecAccuracy:
    return MeasuredCodecAccuracy(
        correct=correct,
        total=total,
        split="val",
        source="test fixture: measured validation cell",
    )


def _outage(
    numerator: int = 100, denominator: int = 1000, selected_class: int = 0
) -> MeasuredOutageAccuracy:
    return MeasuredOutageAccuracy(
        selected_class=selected_class,
        numerator=numerator,
        denominator=denominator,
        source="test fixture: frozen constant-class measurement",
    )


# --------------------------------------------------------------------------
# P(TB success) = product over code blocks of (1 - BLER_r)
# --------------------------------------------------------------------------


def test_one_code_block_success_probability_is_one_minus_its_bler() -> None:
    assert transport_block_success_probability([0.25]) == pytest.approx(0.75)


def test_two_code_blocks_multiply() -> None:
    # Hand-checked: (1 - 0.1) * (1 - 0.2) = 0.9 * 0.8 = 0.72.
    assert transport_block_success_probability([0.1, 0.2]) == pytest.approx(0.72)


def test_many_code_blocks_multiply() -> None:
    blers = [0.05] * 11
    # Hand-checked: 0.95 ** 11 = 0.5688000922764596 — deliberately not written as
    # a loop over the same code the function uses.
    assert transport_block_success_probability(blers) == pytest.approx(
        0.95**11, rel=1e-12
    )
    assert transport_block_success_probability(blers) == pytest.approx(
        0.5688000922764596, abs=1e-15
    )


def test_mixed_bler_values_compose_multiplicatively() -> None:
    # (1-0.5)*(1-0.25)*(1-0.125) = 0.5 * 0.75 * 0.875 = 0.328125, exact in binary.
    assert transport_block_success_probability([0.5, 0.25, 0.125]) == 0.328125


def test_success_probability_is_exactly_one_when_every_bler_is_zero() -> None:
    assert transport_block_success_probability([0.0, 0.0, 0.0]) == 1.0


def test_success_probability_is_exactly_zero_when_any_bler_is_one() -> None:
    assert transport_block_success_probability([0.1, 1.0, 0.2]) == 0.0


def test_no_code_blocks_is_a_defect_not_a_vacuous_certain_success() -> None:
    with pytest.raises(CompositionError, match="at least one"):
        transport_block_success_probability([])


@pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan"), "0.1", None, True])
def test_invalid_probabilities_are_rejected(bad: object) -> None:
    with pytest.raises(CompositionError):
        transport_block_success_probability([0.1, bad])  # type: ignore[list-item]


def test_a_bare_float_is_not_a_sequence_of_code_block_blers() -> None:
    with pytest.raises(CompositionError, match="not a sequence"):
        transport_block_success_probability(0.1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# expected accuracy = P * acc_clean + (1 - P) * acc_outage
# --------------------------------------------------------------------------


def test_worked_example_expected_accuracy() -> None:
    # Hand-checked: P = (1-0.1)(1-0.2) = 0.72; acc_clean = 0.87; acc_outage = 0.1.
    # 0.72 * 0.87 + 0.28 * 0.1 = 0.6264 + 0.028 = 0.6544.
    result = compose([0.1, 0.2], codec_accuracy=_codec(), outage_accuracy=_outage())
    assert result.success_probability == pytest.approx(0.72)
    assert result.expected_accuracy == pytest.approx(0.6544)
    assert result.code_blocks == 2


def test_degenerate_p_equals_one_is_exactly_the_codec_accuracy() -> None:
    result = compose([0.0], codec_accuracy=_codec(), outage_accuracy=_outage())
    assert result.success_probability == 1.0
    assert result.expected_accuracy == pytest.approx(0.87)


def test_degenerate_p_equals_zero_is_exactly_the_outage_accuracy() -> None:
    result = compose([1.0], codec_accuracy=_codec(), outage_accuracy=_outage())
    assert result.success_probability == 0.0
    assert result.expected_accuracy == pytest.approx(0.1)


def test_expected_accuracy_lies_between_the_two_measured_terms() -> None:
    for p_bler in (0.0, 0.01, 0.3, 0.7, 0.99, 1.0):
        result = compose(
            [p_bler], codec_accuracy=_codec(), outage_accuracy=_outage()
        )
        assert 0.1 - 1e-12 <= result.expected_accuracy <= 0.87 + 1e-12


def test_expected_accuracy_rejects_an_out_of_range_success_probability() -> None:
    with pytest.raises(CompositionError, match=r"outside \[0, 1\]"):
        expected_accuracy(
            success_probability=1.5,
            codec_accuracy=_codec(),
            outage_accuracy=_outage(),
        )


# --------------------------------------------------------------------------
# Measured, not assumed
# --------------------------------------------------------------------------


def test_codec_accuracy_cannot_be_supplied_as_a_bare_float() -> None:
    with pytest.raises(CompositionError, match="MeasuredCodecAccuracy"):
        expected_accuracy(
            success_probability=0.5,
            codec_accuracy=0.87,  # type: ignore[arg-type]
            outage_accuracy=_outage(),
        )


def test_outage_accuracy_cannot_be_supplied_as_a_bare_float() -> None:
    with pytest.raises(CompositionError, match="MeasuredOutageAccuracy"):
        expected_accuracy(
            success_probability=0.5,
            codec_accuracy=_codec(),
            outage_accuracy=0.1,  # type: ignore[arg-type]
        )


def test_measured_inputs_are_passed_through_not_reconstructed() -> None:
    """A non-stratified outage measurement must survive into the result.

    On the committed manifest the measured outage accuracy is 100/1000, which
    equals ``1 / n_classes`` exactly.  Any implementation that quietly rebuilt
    the term from the class count would agree there and only there — so this
    test uses a measurement that is deliberately *not* ``1/10``.
    """

    outage = _outage(numerator=137, denominator=1000)
    assert outage.value == 0.137
    result = compose([0.5], codec_accuracy=_codec(), outage_accuracy=outage)
    # 0.5 * 0.87 + 0.5 * 0.137 = 0.435 + 0.0685 = 0.5035.
    assert result.expected_accuracy == pytest.approx(0.5035)
    assert result.as_record()["outage_accuracy"]["numerator"] == 137
    assert result.as_record()["outage_accuracy"]["measured"] is True


def test_mutation_a_wrong_outage_score_changes_the_composition() -> None:
    """The mutation this catches: substituting ``1 / n_classes`` for a real one."""

    honest = compose(
        [0.5], codec_accuracy=_codec(), outage_accuracy=_outage(137, 1000)
    )
    mutated = compose(
        [0.5], codec_accuracy=_codec(), outage_accuracy=_outage(100, 1000)
    )
    assert honest.expected_accuracy != mutated.expected_accuracy
    assert mutated.expected_accuracy == pytest.approx(0.485)
    assert honest.expected_accuracy - mutated.expected_accuracy == pytest.approx(
        0.5 * (0.137 - 0.1)
    )


def test_codec_accuracy_requires_counts_and_provenance() -> None:
    with pytest.raises(CompositionError, match="empty denominator"):
        MeasuredCodecAccuracy(correct=0, total=0, split="val", source="x")
    with pytest.raises(CompositionError, match="exceeds its denominator"):
        MeasuredCodecAccuracy(correct=11, total=10, split="val", source="x")
    with pytest.raises(CompositionError, match="no provenance"):
        MeasuredCodecAccuracy(correct=1, total=10, split="val", source="  ")
    with pytest.raises(CompositionError, match="not an integer count"):
        MeasuredCodecAccuracy(
            correct=0.87,  # type: ignore[arg-type]
            total=1,
            split="val",
            source="x",
        )


def test_codec_accuracy_refuses_any_split_but_validation() -> None:
    with pytest.raises(CompositionError, match="validation-split"):
        MeasuredCodecAccuracy(correct=1, total=10, split="test", source="x")
    with pytest.raises(CompositionError, match="validation-split"):
        MeasuredCodecAccuracy(correct=1, total=10, split="train", source="x")


def test_outage_accuracy_requires_counts_and_provenance() -> None:
    with pytest.raises(CompositionError, match="empty denominator"):
        MeasuredOutageAccuracy(
            selected_class=0, numerator=0, denominator=0, source="x"
        )
    with pytest.raises(CompositionError, match="exceeds its denominator"):
        MeasuredOutageAccuracy(
            selected_class=0, numerator=11, denominator=10, source="x"
        )
    with pytest.raises(CompositionError, match="no provenance"):
        MeasuredOutageAccuracy(
            selected_class=0, numerator=1, denominator=10, source=""
        )


# --------------------------------------------------------------------------
# The committed PB_2 outage artifact is the source of acc_outage
# --------------------------------------------------------------------------


def test_outage_accuracy_reads_the_committed_frozen_measurement() -> None:
    record = json.loads(OUTAGE_POLICY_PATH.read_text())
    accuracy = measured_outage_accuracy_from_record(record)
    assert accuracy.selected_class == 0
    assert (accuracy.numerator, accuracy.denominator) == (100, 1000)
    assert accuracy.value == 0.1
    assert accuracy.source.endswith("outage_policy.json")


def test_outage_record_missing_counts_is_refused_rather_than_assumed() -> None:
    record = json.loads(OUTAGE_POLICY_PATH.read_text())
    del record["numerator"]
    with pytest.raises(CompositionError, match="refusing to assume"):
        measured_outage_accuracy_from_record(record)


def test_outage_record_whose_float_contradicts_its_counts_is_refused() -> None:
    record = json.loads(OUTAGE_POLICY_PATH.read_text())
    record["measured_validation_accuracy"] = 0.25
    with pytest.raises(CompositionError, match="disagrees with its own counts"):
        measured_outage_accuracy_from_record(record)


def test_outage_record_under_another_policy_is_refused() -> None:
    record = json.loads(OUTAGE_POLICY_PATH.read_text())
    record["selection_policy"] = "uniform_random_label"
    with pytest.raises(CompositionError, match="params.baseline.outage_policy"):
        measured_outage_accuracy_from_record(record)


def test_composition_record_carries_both_measured_inputs_forward() -> None:
    record = json.loads(OUTAGE_POLICY_PATH.read_text())
    result = compose(
        [0.02, 0.02],
        codec_accuracy=_codec(),
        outage_accuracy=measured_outage_accuracy_from_record(record),
    )
    row = result.as_record()
    assert row["code_blocks"] == 2
    assert row["block_blers"] == [0.02, 0.02]
    assert row["codec_accuracy"]["correct"] == 870
    assert row["outage_accuracy"]["assumed_uniform_accuracy_rejected"] is True
    assert math.isclose(
        row["expected_accuracy"],
        (0.98**2) * 0.87 + (1 - 0.98**2) * 0.1,
        rel_tol=1e-12,
    )


# --------------------------------------------------------------------------
# BLER lookup — complete identity, no extrapolation, fails closed
# --------------------------------------------------------------------------

CHARACTERIZED_KEY = {
    "k_and_n": (128, 256),
    "base_graph": 2,
    "lifting_size": 22,
    "modulation": "qpsk",
    "decoder_algorithm": "offset_min_sum",
    "decoder_offset": 0.5,
    "iterations": 50,
    "snr_convention": "eb_n0_per_information_bit",
    "rate": "0.5",
}


def _key(**overrides: object) -> dict[str, object]:
    key = dict(CHARACTERIZED_KEY)
    key.update(overrides)
    return key


def test_the_required_identity_fields_are_the_spec_s_plus_the_code_rate() -> None:
    assert set(BLER_IDENTITY_FIELDS) == set(get("baseline.ldpc_bler_reference_must_match"))
    assert set(BLER_REQUIRED_FIELDS) == set(BLER_IDENTITY_FIELDS) | {"rate"}


def test_the_committed_evidence_characterizes_exactly_three_configurations() -> None:
    table = g2_bler_table()
    modulations = {identity.modulation for identity in table.identities}
    assert modulations == {"bpsk", "qpsk", "qam16"}
    # One curve per modulation per declared SNR convention, and nothing else.
    assert len(table.identities) == 6
    for identity in table.identities:
        assert identity.k_and_n == (128, 256)
        assert (identity.base_graph, identity.lifting_size) == (2, 22)
        assert identity.rate == "0.5"
        assert identity.iterations == 50
        assert identity.decoder_algorithm == "offset_min_sum"
        assert identity.decoder_offset == 0.5


def test_an_exact_measured_point_returns_the_committed_value() -> None:
    result = g2_bler_table().lookup(CHARACTERIZED_KEY, 2.5)
    assert result.characterized
    assert result.bler == 0.0112  # the committed QPSK row at Eb/N0 2.5 dB
    assert result.interpolated is False
    assert result.trials_per_point == 5000
    assert result.require() == 0.0112


def test_interpolation_inside_support_is_log_linear_in_bler() -> None:
    """At the midpoint the declared representation is the geometric mean.

    Written as ``sqrt(a*b)`` rather than as the implementation's weighted
    log-average, so the test is not the code restated.
    """

    result = g2_bler_table().lookup(CHARACTERIZED_KEY, 2.25)
    assert result.characterized
    assert result.interpolated is True
    assert result.bler == pytest.approx(math.sqrt(0.0708 * 0.0112), rel=1e-12)


@pytest.mark.parametrize("snr", [1.4999, 1.0, -5.0, 2.7501, 3.0, 18.0])
def test_no_silent_extrapolation_outside_characterized_support(snr: float) -> None:
    result = g2_bler_table().lookup(CHARACTERIZED_KEY, snr)
    assert result.status == UNCHARACTERIZED
    assert result.reason == "snr_outside_characterized_support"
    assert result.bler is None
    assert result.support_db == (1.5, 2.75)
    with pytest.raises(UncharacterizedBlerError):
        result.require()


def test_absent_evidence_is_never_reported_as_zero_bler() -> None:
    """High SNR is exactly where a silent extrapolation would read as BLER 0."""

    result = g2_bler_table().lookup(CHARACTERIZED_KEY, 18.0)
    assert result.bler is None
    assert result.bler != 0.0
    with pytest.raises(UncharacterizedBlerError):
        result.require()


@pytest.mark.parametrize("field", sorted(BLER_REQUIRED_FIELDS))
def test_a_partial_key_missing_any_required_field_raises(field: str) -> None:
    key = _key()
    del key[field]
    with pytest.raises(BlerLookupError, match="incomplete BLER lookup key"):
        g2_bler_table().lookup(key, 2.5)


def test_an_over_specified_key_raises_rather_than_being_trimmed() -> None:
    with pytest.raises(BlerLookupError, match="unrecognised"):
        g2_bler_table().lookup(_key(channel="awgn"), 2.5)


def test_a_malformed_k_and_n_raises() -> None:
    with pytest.raises(BlerLookupError, match=r"\(K, N\) pair"):
        g2_bler_table().lookup(_key(k_and_n=128), 2.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k_and_n", (256, 512)),
        ("k_and_n", (128, 384)),
        ("base_graph", 1),
        ("lifting_size", 24),
        ("modulation", "qam64"),
        ("modulation", "bpsk_pi_over_2"),
        ("decoder_algorithm", "belief_propagation"),
        ("decoder_offset", 0.75),
        ("iterations", 25),
        ("snr_convention", "es_n0_per_information_bit"),
        ("rate", "0.6666666666666666"),
    ],
)
def test_mutation_every_identity_component_fails_closed_on_its_own(
    field: str, value: object
) -> None:
    """One component wrong at a time, everything else exactly right.

    The failure mode this rules out is a lookup that matches on a *subset* of
    the identity and returns a curve measured under different physics.
    """

    result = g2_bler_table().lookup(_key(**{field: value}), 2.5)
    assert result.status == UNCHARACTERIZED, f"{field}={value!r} was not rejected"
    assert result.reason == "identity_not_characterized"
    assert result.bler is None
    with pytest.raises(UncharacterizedBlerError):
        result.require()


def test_a_wrong_modulation_does_not_silently_reuse_the_same_k_and_n() -> None:
    """16-QAM at QPSK's SNR points is measured, but not at these SNRs."""

    qpsk = g2_bler_table().lookup(CHARACTERIZED_KEY, 2.5)
    qam16 = g2_bler_table().lookup(_key(modulation="qam16"), 2.5)
    assert qpsk.characterized
    assert qam16.status == UNCHARACTERIZED
    assert qam16.reason == "snr_outside_characterized_support"
    assert qam16.support_db == (4.0, 5.25)


def test_lookup_rejects_a_non_numeric_snr() -> None:
    with pytest.raises(BlerLookupError, match="not a number"):
        g2_bler_table().lookup(CHARACTERIZED_KEY, "2.5")
    with pytest.raises(BlerLookupError, match="NaN"):
        g2_bler_table().lookup(CHARACTERIZED_KEY, float("nan"))


def test_both_declared_snr_conventions_are_characterized_and_distinct() -> None:
    table = g2_bler_table()
    ebn0 = table.lookup(CHARACTERIZED_KEY, 2.5)
    esn0 = table.lookup(
        _key(modulation="qam16", snr_convention="es_n0_per_symbol"), 8.010299956639813
    )
    assert ebn0.characterized and esn0.characterized
    assert esn0.bler == 0.0068
    # 16-QAM at Eb/N0 8.01 dB is far outside its measured Eb/N0 span: reading
    # an Es/N0 number under the Eb/N0 convention must not resolve.
    assert (
        table.lookup(_key(modulation="qam16"), 8.010299956639813).status
        == UNCHARACTERIZED
    )


def test_the_table_is_bound_to_the_adjudicated_g2_evidence_bytes(
    tmp_path, monkeypatch
) -> None:
    """A tampered curve file must fail closed rather than feed the selection."""

    from baseline.classical import composition

    tampered = tmp_path / "bler_results.csv"
    original = composition._BLER_RESULTS.read_text()
    tampered.write_text(original.replace("0.0112", "0.0001"))
    monkeypatch.setattr(composition, "_BLER_RESULTS", tampered)
    composition.g2_bler_table.cache_clear()
    try:
        with pytest.raises(CompositionError, match="g2_adjudication.json"):
            composition.g2_bler_table()
    finally:
        composition.g2_bler_table.cache_clear()


def test_lookup_results_are_machine_readable() -> None:
    record = g2_bler_table().lookup(CHARACTERIZED_KEY, 18.0).as_record()
    assert record["status"] == UNCHARACTERIZED
    assert record["bler"] is None
    assert record["identity"]["modulation"] == "qpsk"
    assert json.dumps(record)  # must survive serialization into evidence


# --------------------------------------------------------------------------
# Feasibility caching and deterministic tie-breaking
# --------------------------------------------------------------------------


def _candidate(**overrides: object) -> Candidate:
    base = {
        "dataset": "imagenette160",
        "ratio": "r_1_6",
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "encode_axis_px": 160,
        "snr_db": 12.0,
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


class _CountingProbe:
    """A stand-in for G-8's real packetisation/codec feasibility computation."""

    def __init__(self, feasible: bool = True) -> None:
        self.calls: list[Candidate] = []
        self._feasible = feasible

    def __call__(self, candidate: Candidate) -> Feasibility:
        self.calls.append(candidate)
        return Feasibility(
            feasible=self._feasible,
            reason=None if self._feasible else "structural_infeasibility",
            code_blocks=len(self.calls),  # deliberately call-order dependent
            payload_bytes=candidate.encode_axis_px,
        )


def test_every_candidate_field_is_classified_into_or_out_of_the_cache_key() -> None:
    declared = set(Candidate.__dataclass_fields__)
    assert declared == set(FEASIBILITY_KEY_FIELDS) | set(FEASIBILITY_KEY_EXCLUSIONS)
    for name, reason in FEASIBILITY_KEY_EXCLUSIONS.items():
        assert reason.strip(), f"{name} is excluded without a reason"


def test_the_cache_key_covers_every_configuration_field() -> None:
    """Changing any keyed field must change the key."""

    base = _candidate()
    for field_name, changed in [
        ("dataset", "stl10"),
        ("ratio", "r_1_12"),
        ("modulation", "qam16"),
        ("ldpc_rate", "2/3"),
        ("encode_axis_px", 128),
    ]:
        other = _candidate(**{field_name: changed})
        assert other.feasibility_key() != base.feasibility_key(), field_name


def test_the_snr_exclusion_is_real_and_not_merely_documented() -> None:
    """The one excluded field must genuinely not move the key."""

    assert _candidate(snr_db=-8.0).feasibility_key() == _candidate(
        snr_db=18.0
    ).feasibility_key()
    assert set(FEASIBILITY_KEY_EXCLUSIONS) == {"snr_db"}


def test_cached_and_uncached_paths_return_identical_results() -> None:
    cache = FeasibilityCache()
    probe = _CountingProbe()
    first = cache.feasibility(_candidate(), probe)
    second = cache.feasibility(_candidate(), probe)
    assert first == second
    assert len(probe.calls) == 1, "the second call must be served from the cache"
    assert (cache.misses, cache.hits) == (1, 1)


def test_the_cache_is_keyed_on_configuration_not_on_snr() -> None:
    cache = FeasibilityCache()
    probe = _CountingProbe()
    cache.feasibility(_candidate(snr_db=-8.0), probe)
    cache.feasibility(_candidate(snr_db=18.0), probe)
    assert len(probe.calls) == 1
    assert len(cache) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "stl10"),
        ("ratio", "r_1_12"),
        ("modulation", "qam16"),
        ("ldpc_rate", "2/3"),
        ("encode_axis_px", 128),
    ],
)
def test_no_cross_configuration_cache_collision(field: str, value: object) -> None:
    """A different configuration must never reuse another's cached verdict."""

    cache = FeasibilityCache()
    probe = _CountingProbe()
    first = cache.feasibility(_candidate(), probe)
    second = cache.feasibility(_candidate(**{field: value}), probe)
    assert len(probe.calls) == 2, f"{field} collided with the base configuration"
    assert len(cache) == 2
    assert first.code_blocks != second.code_blocks


def test_the_cache_refuses_a_probe_that_does_not_return_a_feasibility() -> None:
    cache = FeasibilityCache()
    with pytest.raises(CompositionError, match="must return a Feasibility"):
        cache.feasibility(_candidate(), lambda candidate: True)


def test_the_cache_refuses_anything_that_is_not_a_candidate() -> None:
    cache = FeasibilityCache()
    with pytest.raises(CompositionError, match="takes a Candidate"):
        cache.feasibility(("imagenette160", "r_1_6"), _CountingProbe())


def _evaluation(
    accuracy_correct: int, blers: list[float], **overrides: object
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=_candidate(**overrides),
        status=ELIGIBLE,
        composition=compose(
            blers,
            codec_accuracy=_codec(correct=accuracy_correct),
            outage_accuracy=_outage(),
        ),
    )


def test_selection_takes_the_highest_expected_accuracy() -> None:
    worse = _evaluation(800, [0.01], modulation="bpsk")
    better = _evaluation(900, [0.01], modulation="qam16")
    selection = select_best([worse, better])
    assert selection.selected is better
    assert selection.tie_break_applied is False
    assert selection.counts()[ELIGIBLE] == 2


def test_selection_is_independent_of_enumeration_order() -> None:
    evaluations = [
        _evaluation(870, [0.01], modulation="bpsk"),
        _evaluation(870, [0.01], modulation="qpsk"),
        _evaluation(870, [0.01], modulation="qam16"),
        _evaluation(870, [0.02], modulation="qpsk", encode_axis_px=128),
    ]
    chosen = {
        select_best(list(order)).selected.candidate  # type: ignore[union-attr]
        for order in itertools.permutations(evaluations)
    }
    assert len(chosen) == 1


def test_tie_breaking_follows_the_documented_order() -> None:
    """Exactly-equal expected accuracies resolve by the published key."""

    assert TIE_BREAK_ORDER[0] == "expected_accuracy_descending"
    # Identical composition, three modulations: lower Qm wins (more robust).
    tied = [
        _evaluation(870, [0.01], modulation="qam16"),
        _evaluation(870, [0.01], modulation="qpsk"),
        _evaluation(870, [0.01], modulation="bpsk"),
    ]
    selection = select_best(tied)
    assert selection.selected.candidate.modulation == "bpsk"  # type: ignore[union-attr]
    assert selection.tie_break_applied is True
    assert len(selection.tied) == 3


def test_tie_breaking_prefers_the_stronger_channel_code_then_the_larger_axis() -> None:
    tied_rate = [
        _evaluation(870, [0.01], ldpc_rate="5/6"),
        _evaluation(870, [0.01], ldpc_rate="1/3"),
    ]
    assert select_best(tied_rate).selected.candidate.ldpc_rate == "1/3"  # type: ignore[union-attr]

    tied_axis = [
        _evaluation(870, [0.01], encode_axis_px=96),
        _evaluation(870, [0.01], encode_axis_px=160),
    ]
    assert select_best(tied_axis).selected.candidate.encode_axis_px == 160  # type: ignore[union-attr]


def test_tie_breaking_is_total_and_therefore_deterministic() -> None:
    """Two candidates differing only in the last key still resolve."""

    tied = [
        _evaluation(870, [0.01], dataset="stl10"),
        _evaluation(870, [0.01], dataset="imagenette160"),
    ]
    first = select_best(tied).selected
    second = select_best(list(reversed(tied))).selected
    assert first is not None and second is not None
    assert first.candidate == second.candidate
    assert first.candidate.dataset == "imagenette160"  # canonical-id order


def test_tie_breaking_prefers_the_more_reliable_link_before_the_configuration() -> None:
    """Equal expected accuracy, different P(TB success): the reliable one wins."""

    # acc_clean = acc_outage makes expected accuracy independent of P.
    flat = MeasuredCodecAccuracy(
        correct=100, total=1000, split="val", source="flat fixture"
    )
    reliable = CandidateEvaluation(
        candidate=_candidate(modulation="qam16"),
        status=ELIGIBLE,
        composition=compose([0.0], codec_accuracy=flat, outage_accuracy=_outage()),
    )
    fragile = CandidateEvaluation(
        candidate=_candidate(modulation="bpsk"),
        status=ELIGIBLE,
        composition=compose([0.5], codec_accuracy=flat, outage_accuracy=_outage()),
    )
    selection = select_best([fragile, reliable])
    assert selection.selected is reliable
    assert selection.selected.composition.success_probability == 1.0  # type: ignore[union-attr]


def test_an_uncharacterized_candidate_is_ineligible_not_low_scoring() -> None:
    eligible = _evaluation(500, [0.5])
    unknown = CandidateEvaluation(
        candidate=_candidate(modulation="qam16"),
        status=UNCHARACTERIZED,
        reason="identity_not_characterized",
    )
    selection = select_best([unknown, eligible])
    assert selection.selected is eligible
    assert selection.counts()[UNCHARACTERIZED] == 1
    assert unknown not in selection.eligible
    assert unknown.expected_accuracy is None


def test_an_infeasible_candidate_is_ineligible_and_recorded() -> None:
    infeasible = CandidateEvaluation(
        candidate=_candidate(ratio="r_1_48"),
        status=INFEASIBLE,
        reason="structural_infeasibility",
    )
    selection = select_best([infeasible, _evaluation(870, [0.01])])
    assert selection.counts()[INFEASIBLE] == 1
    assert selection.selected is not None
    assert selection.selected.status == ELIGIBLE


def test_no_eligible_candidate_is_stated_explicitly_never_guessed() -> None:
    selection = select_best(
        [
            CandidateEvaluation(_candidate(), UNCHARACTERIZED, reason="x"),
            CandidateEvaluation(_candidate(ratio="r_1_48"), INFEASIBLE, reason="y"),
        ]
    )
    assert selection.selected is None
    assert selection.reason == "no_eligible_candidate"
    assert selection.tie_break_applied is False


def test_selection_records_are_machine_readable_and_carry_the_tie_rule() -> None:
    selection = select_best([_evaluation(870, [0.01])])
    record = selection.as_record()
    assert record["tie_break_order"] == list(TIE_BREAK_ORDER)
    assert record["counts"][ELIGIBLE] == 1
    assert json.dumps(record)


# --------------------------------------------------------------------------
# The three system modes
# --------------------------------------------------------------------------


def test_the_three_modes_are_declared_system_values() -> None:
    declared = get("artifacts.system_values")
    for mode in SYSTEM_MODES:
        assert mode in declared
    assert SYSTEM_MODES == (
        "classical_adaptive",
        "classical_fixed_mod",
        "classical_fixed_mcs",
    )


def test_the_three_modes_are_distinct_policies() -> None:
    policies = [mode_policy(mode) for mode in SYSTEM_MODES]
    signatures = {
        (p.adapts_modulation, p.adapts_ldpc_rate, p.adapts_encode_axis)
        for p in policies
    }
    assert len(signatures) == 3
    assert mode_policy(CLASSICAL_ADAPTIVE).adapts_modulation is True
    assert mode_policy(CLASSICAL_FIXED_MOD).adapts_modulation is False
    assert mode_policy(CLASSICAL_FIXED_MOD).adapts_ldpc_rate is True
    assert mode_policy(CLASSICAL_FIXED_MCS).adapts_ldpc_rate is False


def test_only_the_fixed_mcs_mode_has_a_design_snr() -> None:
    assert mode_policy(CLASSICAL_ADAPTIVE).design_snr_db is None
    assert mode_policy(CLASSICAL_FIXED_MOD).design_snr_db is None
    assert mode_policy(CLASSICAL_FIXED_MCS).design_snr_db == float(
        get("baseline.fixed_mcs_design_snr_db")
    )


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(CompositionError, match="unknown classical system mode"):
        mode_policy("classical_magic")
    with pytest.raises(CompositionError, match="unknown classical system mode"):
        mode_policy("learned")


def _grid() -> dict[float, list[CandidateEvaluation]]:
    """A three-point grid where each mode must land somewhere different.

    BPSK is best when the link is bad, 16-QAM when it is clean, and the design
    SNR sits in the middle — so an adaptive curve switches, a fixed-modulation
    curve cannot, and a fixed-MCS curve is pinned to the middle point's choice.
    """

    def evaluation(
        modulation: str, correct: int, bler: float, **rest: object
    ) -> CandidateEvaluation:
        return _evaluation(correct, [bler], modulation=modulation, **rest)

    design = float(get("baseline.fixed_mcs_design_snr_db"))
    return {
        -6.0: [
            evaluation("bpsk", 700, 0.01),
            evaluation("qam16", 950, 0.99),
        ],
        design: [
            evaluation("bpsk", 700, 0.01),
            evaluation("qam16", 950, 0.50),
        ],
        18.0: [
            evaluation("bpsk", 700, 0.01),
            evaluation("qam16", 950, 0.01),
        ],
    }


def test_adaptive_mode_reselects_the_modulation_per_snr() -> None:
    curve = resolve_curve(CLASSICAL_ADAPTIVE, _grid())
    chosen = [
        selection.selected.candidate.modulation  # type: ignore[union-attr]
        for _, selection in curve.per_snr
    ]
    assert chosen[0] == "bpsk"  # noisy end
    assert chosen[-1] == "qam16"  # clean end
    assert len(set(chosen)) > 1
    assert curve.held_fixed == {}


def test_fixed_modulation_mode_holds_one_modulation_across_the_grid() -> None:
    curve = resolve_curve(CLASSICAL_FIXED_MOD, _grid())
    chosen = {
        selection.selected.candidate.modulation  # type: ignore[union-attr]
        for _, selection in curve.per_snr
    }
    assert len(chosen) == 1
    assert curve.held_fixed["modulation"] in chosen


def test_fixed_mcs_mode_holds_the_whole_configuration_from_the_design_snr() -> None:
    curve = resolve_curve(CLASSICAL_FIXED_MCS, _grid())
    design = float(get("baseline.fixed_mcs_design_snr_db"))
    assert curve.held_fixed["design_snr_db"] == design
    assert curve.held_fixed["packet_count"] == get("baseline.fixed_mcs_packet_count")
    configurations = {
        selection.selected.candidate.feasibility_key()  # type: ignore[union-attr]
        for _, selection in curve.per_snr
    }
    assert len(configurations) == 1
    # And it is the design point's choice, not the noisy or clean end's.
    assert (
        curve.selection_at(design).selected.candidate.modulation  # type: ignore[union-attr]
        == curve.held_fixed["modulation"]
    )


def test_the_three_modes_produce_different_curves_on_the_same_candidates() -> None:
    grid = _grid()
    records = {
        mode: json.dumps(resolve_curve(mode, grid).as_record(), sort_keys=True)
        for mode in SYSTEM_MODES
    }
    assert len(set(records.values())) == 3


def test_fixed_mcs_refuses_to_snap_to_a_neighbouring_design_snr() -> None:
    grid = _grid()
    design = float(get("baseline.fixed_mcs_design_snr_db"))
    del grid[design]
    with pytest.raises(CompositionError, match="refusing to snap"):
        resolve_curve(CLASSICAL_FIXED_MCS, grid)


def test_resolving_an_empty_grid_is_refused() -> None:
    with pytest.raises(CompositionError, match="no SNR points"):
        resolve_curve(CLASSICAL_ADAPTIVE, {})


# --------------------------------------------------------------------------
# Two passes, then stop — structurally
# --------------------------------------------------------------------------


def _selector(context: PassContext):
    return select_best([_evaluation(870, [0.01])])


def test_the_permitted_passes_come_from_the_spec() -> None:
    assert selection_passes() == (PASS_ONE, PASS_TWO)
    assert get("reference_classifier.br4_selection_terminates_after_pass") == 2


def test_pass_one_then_pass_two_is_the_whole_permitted_sequence() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    assert campaign.exhausted is False
    campaign.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")
    assert campaign.exhausted is True
    assert campaign.completed_passes == (1, 2)


def test_a_third_pass_raises() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    campaign.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")
    with pytest.raises(SelectionPassError, match="unknown selection pass 3"):
        campaign.run_pass(3, _selector, scorer="third")


def test_pass_one_may_run_only_once() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    with pytest.raises(SelectionPassError, match="pass 1 has already run"):
        campaign.run_pass(PASS_ONE, _selector, scorer="clean_again")


def test_pass_two_may_run_only_once() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    campaign.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")
    with pytest.raises(SelectionPassError, match="pass 2 has already run"):
        campaign.run_pass(PASS_TWO, _selector, scorer="finetuned_again")


def test_pass_two_cannot_run_before_pass_one() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    with pytest.raises(SelectionPassError, match="cannot run before pass 1"):
        campaign.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")


@pytest.mark.parametrize("pass_id", [0, -1, 3, 99])
def test_unknown_pass_identifiers_raise(pass_id: int) -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    with pytest.raises(SelectionPassError, match="unknown selection pass"):
        campaign.run_pass(pass_id, _selector, scorer="whatever")


@pytest.mark.parametrize("pass_id", ["1", 1.0, None, True])
def test_non_integer_pass_identifiers_raise(pass_id: object) -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    with pytest.raises(SelectionPassError, match="not an integer"):
        campaign.run_pass(pass_id, _selector, scorer="whatever")


def test_the_two_passes_must_use_different_scorers() -> None:
    """The second pass exists because BR-12 replaced the scorer (AM-54)."""

    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    with pytest.raises(SelectionPassError, match="reuses the scorer"):
        campaign.run_pass(PASS_TWO, _selector, scorer="clean")


def test_a_pass_must_name_its_scorer() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    with pytest.raises(SelectionPassError, match="must name its scorer"):
        campaign.run_pass(PASS_ONE, _selector, scorer="   ")


def test_pass_two_state_cannot_alter_pass_one_results() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    first = campaign.run_pass(
        PASS_ONE, lambda ctx: select_best([_evaluation(800, [0.01])]), scorer="clean"
    )
    before = json.dumps(first.as_record(), sort_keys=True)
    campaign.run_pass(
        PASS_TWO,
        lambda ctx: select_best([_evaluation(950, [0.5], modulation="qam16")]),
        scorer="artifact_finetuned",
    )
    assert json.dumps(campaign.result_of(PASS_ONE).as_record(), sort_keys=True) == before
    assert campaign.result_of(PASS_ONE) is first
    with pytest.raises(Exception):
        first.selections = ()  # type: ignore[misc]  # frozen dataclass


def test_pass_one_cannot_read_pass_two_results() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    leaked: list[object] = []

    def peeking(context: PassContext):
        assert context.previous() is None
        with pytest.raises(SelectionPassError, match="may not read pass 2"):
            context.result_of(PASS_TWO)
        with pytest.raises(SelectionPassError, match="may not read pass 1"):
            context.result_of(PASS_ONE)  # not even itself
        leaked.append(context)
        return select_best([_evaluation(870, [0.01])])

    campaign.run_pass(PASS_ONE, peeking, scorer="clean")
    # A context retained past its pass still cannot look forwards.
    with pytest.raises(SelectionPassError, match="may not read pass 2"):
        leaked[0].result_of(PASS_TWO)  # type: ignore[attr-defined]


def test_pass_two_reads_exactly_pass_one() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    first = campaign.run_pass(PASS_ONE, _selector, scorer="clean")
    seen: list[PassResult] = []

    def reading(context: PassContext):
        seen.append(context.result_of(PASS_ONE))
        assert context.previous() is first
        return select_best([_evaluation(870, [0.01])])

    campaign.run_pass(PASS_TWO, reading, scorer="artifact_finetuned")
    assert seen == [first]


def test_a_resumed_campaign_counts_the_passes_it_inherited() -> None:
    original = SelectionCampaign(CLASSICAL_ADAPTIVE)
    original.run_pass(PASS_ONE, _selector, scorer="clean")
    original.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")

    resumed = SelectionCampaign(CLASSICAL_ADAPTIVE, completed=original.state())
    assert resumed.exhausted is True
    with pytest.raises(SelectionPassError, match="already run"):
        resumed.run_pass(PASS_ONE, _selector, scorer="fresh")
    with pytest.raises(SelectionPassError, match="unknown selection pass 3"):
        resumed.run_pass(3, _selector, scorer="third")


def test_a_resumed_campaign_after_one_pass_may_run_exactly_the_second() -> None:
    original = SelectionCampaign(CLASSICAL_ADAPTIVE)
    original.run_pass(PASS_ONE, _selector, scorer="clean")
    resumed = SelectionCampaign(CLASSICAL_ADAPTIVE, completed=original.state())
    assert resumed.completed_passes == (1,)
    resumed.run_pass(PASS_TWO, _selector, scorer="artifact_finetuned")
    assert resumed.exhausted is True


def test_resumed_state_is_validated_rather_than_trusted() -> None:
    good = SelectionCampaign(CLASSICAL_ADAPTIVE)
    good.run_pass(PASS_ONE, _selector, scorer="clean")
    [first] = good.state()

    with pytest.raises(SelectionPassError, match="repeats pass 1"):
        SelectionCampaign(CLASSICAL_ADAPTIVE, completed=(first, first))
    with pytest.raises(SelectionPassError, match="unknown pass"):
        SelectionCampaign(
            CLASSICAL_ADAPTIVE,
            completed=(
                PassResult(
                    pass_id=3, mode=CLASSICAL_ADAPTIVE, scorer="x", selections=()
                ),
            ),
        )
    with pytest.raises(SelectionPassError, match="ran under mode"):
        SelectionCampaign(CLASSICAL_FIXED_MCS, completed=(first,))
    with pytest.raises(SelectionPassError, match="must be PassResults"):
        SelectionCampaign(CLASSICAL_ADAPTIVE, completed=({"pass_id": 1},))


def test_a_pass_must_return_selections() -> None:
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    with pytest.raises(SelectionPassError, match="must return Selections"):
        campaign.run_pass(PASS_ONE, lambda ctx: ["not a selection"], scorer="clean")


def test_pb3_does_not_train_the_artifact_finetuned_classifier() -> None:
    """The gate for that classifier is G-8, which PB_3 does not open."""

    assert get("reference_classifier.artifact_finetune_gate") == "G-8"
    campaign = SelectionCampaign(CLASSICAL_ADAPTIVE)
    record = campaign.as_record()
    assert record["completed_passes"] == []
    assert record["exhausted"] is False
