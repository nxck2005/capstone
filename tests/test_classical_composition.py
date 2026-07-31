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

import json
import math
import pytest

from baseline.classical.composition import (
    CompositionError,
    MeasuredCodecAccuracy,
    MeasuredOutageAccuracy,
    compose,
    expected_accuracy,
    measured_outage_accuracy_from_record,
    transport_block_success_probability,
)
from config.params import REPO_ROOT

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
