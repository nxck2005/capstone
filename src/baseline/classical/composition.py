"""BR-4 analytic selection machinery — built here, executed at G-8.

BR-4 requires the classical baseline to be tuned in its own favour at every SNR,
and it requires that tuning to be computed as *a cached feasibility table plus
per-cell block-error characterisation composed analytically* rather than as a
full per-image channel simulation of every candidate.  AM-51 writes the
composition down in the specification rather than leaving it to the
implementer, because every selection in the project flows through it:

    P(TB success) = product over code blocks of (1 - BLER_r)

    expected accuracy = P(TB success)      * measured codec accuracy
                      + (1 - P(TB success)) * measured outage accuracy

This module is that arithmetic and the selection scaffolding around it.  **It
does not run the sweep.**  The sweep is G-8's, and
:func:`select_operating_points` refuses any workload above the bounded budget
unless an explicit :class:`G8Authorization` is presented — an authorization
this repository does not construct anywhere outside its own refusal tests.

Three properties are load-bearing and are enforced structurally rather than
documented:

* **both accuracy inputs are measured.**  ``acc_outage`` is PB_2's frozen
  constant-class measurement (class 0, 100/1000 on the committed Imagenette-160
  validation manifest), never ``1 / n_classes`` — AM-58 forbids the assumption
  outright, and the two happening to coincide under an exactly stratified split
  is precisely why substituting one for the other would go unnoticed.
  ``acc_clean`` is a measured validation accuracy at a cached codec
  configuration.  Neither may be handed in as a bare float: the constructors
  take counts and provenance, so an assumed number has nowhere to enter;
* **the BLER lookup fails closed.**  The committed G-2 evidence characterises
  one physical-layer configuration per modulation and four SNR points each.  A
  lookup whose identity does not match the committed reference in *every* field
  of ``params.baseline.ldpc_bler_reference_must_match``, or whose SNR lies
  outside the measured support, returns an explicit ``uncharacterized`` verdict.
  It never reuses a neighbouring curve, never extrapolates and never treats
  absent evidence as zero BLER;
* **selection runs two passes and stops** (AM-54,
  ``params.reference_classifier.br4_selection_terminates_after_pass``).  A third
  pass raises, and a resumed campaign counts the passes it inherited.

Nothing here trains anything, opens G-8, selects a bandwidth ratio, or touches
the test split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ``EVIDENCE_LABELS`` is repeated verbatim by every W4 evidence artifact, in a
# machine-readable field and in prose; it is imported rather than restated so
# there is exactly one copy.  ``SELECTION_SPLIT`` is the only split BR-4
# selection may read.
from baseline.classical.outage import EVIDENCE_LABELS, SELECTION_SPLIT
from config.params import get

__all__ = [
    "EVIDENCE_LABELS",
    "SELECTION_SPLIT",
    "CompositionError",
    "MeasuredCodecAccuracy",
    "MeasuredOutageAccuracy",
    "CompositionResult",
    "transport_block_success_probability",
    "expected_accuracy",
    "compose",
    "measured_outage_accuracy_from_record",
]


class CompositionError(RuntimeError):
    """A BR-4 composition contract violation, never a link outcome."""


def _require_probability(value: Any, what: str) -> float:
    """Accept only a real probability.  ``bool`` is not a probability here."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompositionError(f"{what} is not a number: {value!r}")
    probability = float(value)
    if probability != probability:  # NaN; `math.isnan` would read no better
        raise CompositionError(f"{what} is NaN")
    if not 0.0 <= probability <= 1.0:
        raise CompositionError(f"{what} outside [0, 1]: {probability!r}")
    return probability


def _require_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompositionError(f"{what} is not an integer count: {value!r}")
    if value < 0:
        raise CompositionError(f"{what} is negative: {value!r}")
    return value


@dataclass(frozen=True)
class MeasuredCodecAccuracy:
    """``acc_clean`` — a *measured* validation accuracy, with its provenance.

    BR-4 calls this "a required measured input ... which is a real artifact and
    MUST be produced rather than assumed".  The constructor therefore takes the
    counts it was measured from rather than a ratio, so the value cannot be
    supplied as a guess that merely looks like a measurement, and the
    provenance travels with it into the evidence record.

    ``split`` may only be the validation split: BR-4 selection is a
    validation-split activity and the test split is sealed until G-12 (SR-22).
    """

    correct: int
    total: int
    split: str
    source: str

    def __post_init__(self) -> None:
        correct = _require_count(self.correct, "measured codec accuracy numerator")
        total = _require_count(self.total, "measured codec accuracy denominator")
        if total == 0:
            raise CompositionError("measured codec accuracy has an empty denominator")
        if correct > total:
            raise CompositionError(
                f"measured codec accuracy numerator exceeds its denominator: "
                f"{correct}/{total}"
            )
        if self.split != SELECTION_SPLIT:
            raise CompositionError(
                f"BR-4 selection is a validation-split measurement; "
                f"refusing split {self.split!r}"
            )
        if not str(self.source).strip():
            raise CompositionError("measured codec accuracy carries no provenance")

    @property
    def value(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class MeasuredOutageAccuracy:
    """``acc_outage`` — PB_2's frozen constant-class measurement.

    AM-58 is explicit that this must be the *measured* validation accuracy of
    the frozen constant class and not ``1 / classes``: assuming the lower one
    understates the digital arms, which is the direction ER-8 forbids.  On the
    committed Imagenette-160 manifest the two coincide (100/1000 = 1/10) because
    ``data.manifests`` enforces an exactly stratified validation split — which
    is exactly why this type refuses a bare float and demands the numerator,
    denominator and the selected class the measurement came from.
    """

    selected_class: int
    numerator: int
    denominator: int
    source: str

    def __post_init__(self) -> None:
        _require_count(self.selected_class, "outage selected class")
        numerator = _require_count(self.numerator, "measured outage accuracy numerator")
        denominator = _require_count(
            self.denominator, "measured outage accuracy denominator"
        )
        if denominator == 0:
            raise CompositionError("measured outage accuracy has an empty denominator")
        if numerator > denominator:
            raise CompositionError(
                f"measured outage accuracy numerator exceeds its denominator: "
                f"{numerator}/{denominator}"
            )
        if not str(self.source).strip():
            raise CompositionError("measured outage accuracy carries no provenance")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator


def measured_outage_accuracy_from_record(
    record: Mapping[str, Any],
) -> MeasuredOutageAccuracy:
    """Build ``acc_outage`` from PB_2's committed frozen-outage artifact.

    The record is the one written by ``tools/gen_w4_outage_policy.py`` and
    verified on every run of ``tools/verify_w4_baseline_integration.py``.  Only
    the count fields are read — never ``measured_validation_accuracy`` alone —
    so a record whose float disagreed with its own counts could not be
    laundered into a selection.
    """

    required = ("selected_class", "numerator", "denominator", "selection_policy")
    missing = [key for key in required if key not in record]
    if missing:
        raise CompositionError(
            f"outage policy record is missing {missing}; refusing to assume a value"
        )
    policy = get("baseline.outage_policy")
    if record["selection_policy"] != policy:
        raise CompositionError(
            f"outage policy record declares {record['selection_policy']!r}, "
            f"but params.baseline.outage_policy is {policy!r}"
        )
    accuracy = MeasuredOutageAccuracy(
        selected_class=int(record["selected_class"]),
        numerator=int(record["numerator"]),
        denominator=int(record["denominator"]),
        source="results/baseline/w4/outage_policy.json",
    )
    recorded = record.get("measured_validation_accuracy")
    if recorded is not None and float(recorded) != accuracy.value:
        raise CompositionError(
            "outage policy record's accuracy disagrees with its own counts: "
            f"{recorded!r} != {accuracy.numerator}/{accuracy.denominator}"
        )
    return accuracy


def transport_block_success_probability(block_blers: Sequence[float]) -> float:
    """``P(TB success)`` = product over code blocks of ``1 - BLER_r``.

    Under a transport-block CRC one failed code block kills the whole transport
    block, so the blocks compose multiplicatively.  An empty sequence is a
    defect, not a vacuous product of 1.0: a transport block always has at least
    one code block, and returning 1.0 for "no blocks were characterised" is
    precisely the silent-success failure this phase exists to prevent.
    """

    if isinstance(block_blers, str | bytes) or not isinstance(block_blers, Sequence):
        raise CompositionError(f"block BLERs are not a sequence: {block_blers!r}")
    if len(block_blers) == 0:
        raise CompositionError(
            "no code blocks supplied; a transport block has at least one, and an "
            "empty product would report certain success"
        )
    probability = 1.0
    for index, bler in enumerate(block_blers):
        probability *= 1.0 - _require_probability(bler, f"BLER of code block {index}")
    return probability


def expected_accuracy(
    *,
    success_probability: float,
    codec_accuracy: MeasuredCodecAccuracy,
    outage_accuracy: MeasuredOutageAccuracy,
) -> float:
    """The AM-51 mixture, with both accuracy terms supplied as measurements.

    Keyword-only and typed: a caller cannot pass two bare floats in the wrong
    order, and cannot pass a float at all where a measurement is required.
    """

    probability = _require_probability(success_probability, "P(TB success)")
    if not isinstance(codec_accuracy, MeasuredCodecAccuracy):
        raise CompositionError(
            "codec accuracy must be a MeasuredCodecAccuracy carrying its counts "
            f"and provenance, not {type(codec_accuracy).__name__}"
        )
    if not isinstance(outage_accuracy, MeasuredOutageAccuracy):
        raise CompositionError(
            "outage accuracy must be a MeasuredOutageAccuracy carrying its counts "
            f"and provenance, not {type(outage_accuracy).__name__}"
        )
    return (
        probability * codec_accuracy.value
        + (1.0 - probability) * outage_accuracy.value
    )


@dataclass(frozen=True)
class CompositionResult:
    """One composed cell: the inputs it used and the number they produced."""

    success_probability: float
    expected_accuracy: float
    code_blocks: int
    block_blers: tuple[float, ...]
    codec_accuracy: MeasuredCodecAccuracy
    outage_accuracy: MeasuredOutageAccuracy

    def as_record(self) -> dict[str, Any]:
        """A machine-readable row that carries its measured inputs forward."""

        return {
            "success_probability": self.success_probability,
            "expected_accuracy": self.expected_accuracy,
            "code_blocks": self.code_blocks,
            "block_blers": list(self.block_blers),
            "codec_accuracy": {
                "value": self.codec_accuracy.value,
                "correct": self.codec_accuracy.correct,
                "total": self.codec_accuracy.total,
                "split": self.codec_accuracy.split,
                "source": self.codec_accuracy.source,
                "measured": True,
            },
            "outage_accuracy": {
                "value": self.outage_accuracy.value,
                "selected_class": self.outage_accuracy.selected_class,
                "numerator": self.outage_accuracy.numerator,
                "denominator": self.outage_accuracy.denominator,
                "source": self.outage_accuracy.source,
                "measured": True,
                "assumed_uniform_accuracy_rejected": True,
            },
        }


def compose(
    block_blers: Sequence[float],
    *,
    codec_accuracy: MeasuredCodecAccuracy,
    outage_accuracy: MeasuredOutageAccuracy,
) -> CompositionResult:
    """Compose one candidate cell into an expected validation accuracy."""

    blers = tuple(
        _require_probability(bler, f"BLER of code block {index}")
        for index, bler in enumerate(block_blers)
    )
    probability = transport_block_success_probability(blers)
    return CompositionResult(
        success_probability=probability,
        expected_accuracy=expected_accuracy(
            success_probability=probability,
            codec_accuracy=codec_accuracy,
            outage_accuracy=outage_accuracy,
        ),
        code_blocks=len(blers),
        block_blers=blers,
        codec_accuracy=codec_accuracy,
        outage_accuracy=outage_accuracy,
    )
