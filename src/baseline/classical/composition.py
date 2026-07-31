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

import csv
import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``EVIDENCE_LABELS`` is repeated verbatim by every W4 evidence artifact, in a
# machine-readable field and in prose; it is imported rather than restated so
# there is exactly one copy.  ``SELECTION_SPLIT`` is the only split BR-4
# selection may read.
from baseline.classical.outage import EVIDENCE_LABELS, SELECTION_SPLIT
from config.params import REPO_ROOT, get

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
    "BlerLookupError",
    "UncharacterizedBlerError",
    "BlerIdentity",
    "BlerLookup",
    "BlerTable",
    "BLER_IDENTITY_FIELDS",
    "CHARACTERIZED",
    "UNCHARACTERIZED",
    "g2_bler_table",
    "Candidate",
    "Feasibility",
    "FeasibilityCache",
    "CandidateEvaluation",
    "Selection",
    "select_best",
    "FEASIBILITY_KEY_FIELDS",
    "FEASIBILITY_KEY_EXCLUSIONS",
    "TIE_BREAK_ORDER",
    "ELIGIBLE",
    "INFEASIBLE",
    "CLASSICAL_ADAPTIVE",
    "CLASSICAL_FIXED_MOD",
    "CLASSICAL_FIXED_MCS",
    "SYSTEM_MODES",
    "SystemModePolicy",
    "mode_policy",
    "PASS_ONE",
    "PASS_TWO",
    "SelectionPassError",
    "SelectionCampaign",
    "PassResult",
    "PassContext",
    "selection_passes",
    "CurveSelection",
    "resolve_curve",
    "SweepBudgetError",
    "SweepBudget",
    "G8Authorization",
    "G8_GATE",
    "MAX_UNAUTHORIZED_CANDIDATES",
    "MAX_UNAUTHORIZED_SAMPLES",
    "MAX_UNAUTHORIZED_WORKLOAD",
    "sweep_budget",
    "check_sweep_budget",
    "select_operating_points",
    "evaluate_candidate",
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


# ==========================================================================
# BLER lookup — complete identity required, fails closed outside support
# ==========================================================================
#
# The committed G-2 evidence characterises *one* physical-layer configuration
# — K=128, N=256, BG2, Z=22, rate 1/2, flooding offset-min-sum with offset 0.5
# and 50 iterations — at four Eb/N0 points for each of BPSK, QPSK and 16-QAM.
# That is the whole of the measured support.  Everything below exists to stop
# that narrow evidence being quietly generalised to configurations nobody
# measured, which is the failure mode that would manufacture a BR-4 result
# rather than compute one.

#: ``params.baseline.ldpc_bler_reference_must_match``.  Read at runtime so a
#: spec change that adds a required field breaks this module rather than
#: silently leaving a hole in the key.
BLER_IDENTITY_FIELDS: tuple[str, ...] = tuple(
    get("baseline.ldpc_bler_reference_must_match")
)

#: The committed evidence also fixes the code rate, which the spec's must-match
#: list does not name.  A curve measured at rate 1/2 says nothing about rate
#: 5/6 at the same (K, N), so ``rate`` is required here too: this is strictly
#: narrower than the spec demands, which is the safe direction.
BLER_EXTRA_IDENTITY_FIELDS: tuple[str, ...] = ("rate",)

#: Every field a lookup key must carry.  A key missing any of them is a defect,
#: not a near miss.
BLER_REQUIRED_FIELDS: tuple[str, ...] = (
    *BLER_IDENTITY_FIELDS,
    *BLER_EXTRA_IDENTITY_FIELDS,
)

CHARACTERIZED = "characterized"
UNCHARACTERIZED = "uncharacterized"

#: The two SNR conventions the committed evidence carries.  ``bler_results.csv``
#: records Eb/N0 (the reference platform's own convention, declared as
#: ``source_snr_convention``) and the Es/N0 column derived from it by the
#: per-modulation conversion recorded in ``g2_adjudication.json``.  Anything
#: else is a convention nobody measured in.
_EBN0 = "eb_n0_per_information_bit"
_ESN0 = "es_n0_per_symbol"
_SNR_COLUMN = {_EBN0: "ebn0_db", _ESN0: "esn0_db"}

#: ``bler_reference.json`` declares how the committed points may be joined up.
#: Interpolation is permitted only in the representation the reference itself
#: names; a change there must break this module rather than be reinterpreted.
_INTERPOLATION = "linear_in_snr_vs_log10_bler"

#: The measurement arm whose curves the project's own selection uses.  The
#: ``reference`` rows are the independent cross-check that G-2 compared against,
#: not a second usable curve.
_MEASUREMENT_SYSTEM = "sionna"

_G2_DIR = REPO_ROOT / "results" / "baseline" / "g2"
_BLER_RESULTS = _G2_DIR / "bler_results.csv"
_BLER_REFERENCE = _G2_DIR / "bler_reference.json"
_G2_ADJUDICATION = _G2_DIR / "g2_adjudication.json"


class BlerLookupError(CompositionError):
    """A malformed or incomplete BLER lookup key."""


class UncharacterizedBlerError(CompositionError):
    """The requested identity or SNR was never measured.

    Distinct from :class:`BlerLookupError` on purpose: a partial key is a bug in
    the caller, whereas an uncharacterized cell is a legitimate answer that the
    selection must treat as *ineligible* — never as a low-scoring candidate and
    never as zero BLER.
    """


@dataclass(frozen=True)
class BlerIdentity:
    """The complete physical-layer identity of one characterised BLER curve.

    Frozen and hashable so it can be a cache key, and constructed only through
    :meth:`from_mapping`, which refuses a partial or over-specified key.
    """

    k_and_n: tuple[int, int]
    base_graph: int
    lifting_size: int
    modulation: str
    decoder_algorithm: str
    decoder_offset: float
    iterations: int
    snr_convention: str
    rate: str

    @classmethod
    def from_mapping(cls, key: Mapping[str, Any]) -> BlerIdentity:
        if not isinstance(key, Mapping):
            raise BlerLookupError(f"BLER lookup key is not a mapping: {key!r}")
        missing = [name for name in BLER_REQUIRED_FIELDS if name not in key]
        if missing:
            raise BlerLookupError(
                "incomplete BLER lookup key; the committed evidence is only "
                f"valid for a complete physical-layer identity, missing {missing}"
            )
        unexpected = [name for name in key if name not in BLER_REQUIRED_FIELDS]
        if unexpected:
            raise BlerLookupError(
                f"unrecognised BLER lookup key fields: {sorted(unexpected)}"
            )
        k_and_n = key["k_and_n"]
        if (
            isinstance(k_and_n, str)
            or not isinstance(k_and_n, Sequence)
            or len(k_and_n) != 2
        ):
            raise BlerLookupError(f"k_and_n is not a (K, N) pair: {k_and_n!r}")
        return cls(
            k_and_n=(int(k_and_n[0]), int(k_and_n[1])),
            base_graph=int(key["base_graph"]),
            lifting_size=int(key["lifting_size"]),
            modulation=str(key["modulation"]),
            decoder_algorithm=str(key["decoder_algorithm"]),
            decoder_offset=float(key["decoder_offset"]),
            iterations=int(key["iterations"]),
            snr_convention=str(key["snr_convention"]),
            rate=str(key["rate"]),
        )

    def as_key(self) -> dict[str, Any]:
        return {
            "k_and_n": list(self.k_and_n),
            "base_graph": self.base_graph,
            "lifting_size": self.lifting_size,
            "modulation": self.modulation,
            "decoder_algorithm": self.decoder_algorithm,
            "decoder_offset": self.decoder_offset,
            "iterations": self.iterations,
            "snr_convention": self.snr_convention,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class BlerLookup:
    """The result of one lookup: a number, or an explicit refusal to guess."""

    status: str
    identity: BlerIdentity
    snr_db: float
    bler: float | None = None
    interpolated: bool = False
    reason: str | None = None
    support_db: tuple[float, float] | None = None
    trials_per_point: int | None = None

    @property
    def characterized(self) -> bool:
        return self.status == CHARACTERIZED

    def require(self) -> float:
        """The BLER, or :class:`UncharacterizedBlerError` — never a guess."""

        if not self.characterized or self.bler is None:
            raise UncharacterizedBlerError(
                f"no committed BLER evidence for {self.identity.as_key()} at "
                f"{self.snr_db} dB: {self.reason}"
            )
        return self.bler

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "identity": self.identity.as_key(),
            "snr_db": self.snr_db,
            "bler": self.bler,
            "interpolated": self.interpolated,
            "reason": self.reason,
            "support_db": list(self.support_db) if self.support_db else None,
            "trials_per_point": self.trials_per_point,
        }


@dataclass(frozen=True)
class _Curve:
    """One characterised curve: its measured points, in SNR order."""

    snr_db: tuple[float, ...]
    bler: tuple[float, ...]
    trials: int

    @property
    def support(self) -> tuple[float, float]:
        return self.snr_db[0], self.snr_db[-1]


class BlerTable:
    """The committed BLER curves, indexed by complete physical-layer identity.

    Lookups fail closed in every direction that could invent evidence:

    * an identity that is not an exact match returns ``uncharacterized`` —
      never the nearest curve, never the same (K, N) under another modulation,
      never a partial-key match;
    * an SNR outside the measured span of that curve returns
      ``uncharacterized`` — no extrapolation in either direction, and in
      particular no "BLER is 0 well above the waterfall";
    * inside the span the value is interpolated only in the representation the
      committed reference declares (``linear_in_snr_vs_log10_bler``).
    """

    def __init__(self, curves: Mapping[BlerIdentity, _Curve], provenance: str):
        self._curves = dict(curves)
        self.provenance = provenance

    @property
    def identities(self) -> tuple[BlerIdentity, ...]:
        return tuple(
            sorted(self._curves, key=lambda identity: json.dumps(identity.as_key()))
        )

    def lookup(self, key: Mapping[str, Any] | BlerIdentity, snr_db: Any) -> BlerLookup:
        identity = (
            key if isinstance(key, BlerIdentity) else BlerIdentity.from_mapping(key)
        )
        if isinstance(snr_db, bool) or not isinstance(snr_db, int | float):
            raise BlerLookupError(f"SNR is not a number: {snr_db!r}")
        snr = float(snr_db)
        if snr != snr:
            raise BlerLookupError("SNR is NaN")
        curve = self._curves.get(identity)
        if curve is None:
            return BlerLookup(
                status=UNCHARACTERIZED,
                identity=identity,
                snr_db=snr,
                reason="identity_not_characterized",
            )
        low, high = curve.support
        if not low <= snr <= high:
            return BlerLookup(
                status=UNCHARACTERIZED,
                identity=identity,
                snr_db=snr,
                reason="snr_outside_characterized_support",
                support_db=(low, high),
                trials_per_point=curve.trials,
            )
        value, interpolated = _interpolate(curve, snr)
        return BlerLookup(
            status=CHARACTERIZED,
            identity=identity,
            snr_db=snr,
            bler=value,
            interpolated=interpolated,
            support_db=(low, high),
            trials_per_point=curve.trials,
        )

    def require(self, key: Mapping[str, Any] | BlerIdentity, snr_db: Any) -> float:
        return self.lookup(key, snr_db).require()


def _interpolate(curve: _Curve, snr: float) -> tuple[float, bool]:
    """Linear in SNR against log10(BLER), strictly inside the measured span."""

    if _INTERPOLATION != get_reference_settings()["waterfall_interpolation"]:
        raise NotImplementedError(
            "the committed BLER reference no longer declares "
            f"{_INTERPOLATION!r} as its interpolation representation"
        )
    for point_snr, point_bler in zip(curve.snr_db, curve.bler, strict=True):
        if point_snr == snr:
            return point_bler, False
    for index in range(len(curve.snr_db) - 1):
        low, high = curve.snr_db[index], curve.snr_db[index + 1]
        if low < snr < high:
            low_bler, high_bler = curve.bler[index], curve.bler[index + 1]
            if low_bler <= 0.0 or high_bler <= 0.0:
                raise UncharacterizedBlerError(
                    "cannot interpolate through a non-positive measured BLER at "
                    f"{low} dB / {high} dB; the declared representation is "
                    f"{_INTERPOLATION}"
                )
            weight = (snr - low) / (high - low)
            log_value = (1.0 - weight) * math.log10(low_bler) + weight * math.log10(
                high_bler
            )
            return math.pow(10.0, log_value), True  # literal-ok: base of log10
    raise UncharacterizedBlerError(  # pragma: no cover - guarded by the caller
        f"{snr} dB is not inside the characterized support {curve.support}"
    )


@functools.cache
def get_reference_settings() -> dict[str, Any]:
    """``bler_reference.json``'s declared simulation settings."""

    return dict(json.loads(_BLER_REFERENCE.read_text())["settings"])


def _bound_evidence_bytes(path: Path, name: str) -> str:
    """Read a G-2 evidence file and check it against the adjudicated hash.

    The G-2 adjudication records the SHA-256 of every evidence file it stands
    on.  Building a selection table from bytes that no longer match those
    hashes would mean composing against numbers the gate never adjudicated, so
    it fails closed here rather than being noticed later by the verifier.
    """

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = json.loads(_G2_ADJUDICATION.read_text())["evidence_files"][name]
    if digest != expected:
        raise CompositionError(
            f"{name} does not match the hash bound by g2_adjudication.json: "
            f"{digest} != {expected}"
        )
    return payload.decode()


@functools.cache
def g2_bler_table() -> BlerTable:
    """Build the lookup table from the committed, hash-bound G-2 evidence.

    Every row contributes to two identities — one per SNR convention — because
    the CSV carries both the reference platform's Eb/N0 column and the Es/N0
    column derived from it by the conversion ``g2_adjudication.json`` records.
    Both are measured points of the same curve expressed in a declared
    convention; no third convention exists.
    """

    text = _bound_evidence_bytes(_BLER_RESULTS, _BLER_RESULTS.name)
    settings = get_reference_settings()
    if settings["source_snr_convention"] != _EBN0:
        raise NotImplementedError(
            "the committed BLER reference no longer records "
            f"{_EBN0!r} as its source SNR convention"
        )
    points: dict[BlerIdentity, list[tuple[float, float, int]]] = {}
    for row in csv.DictReader(text.splitlines()):
        if row["system"] != _MEASUREMENT_SYSTEM:
            continue
        for convention, column in _SNR_COLUMN.items():
            identity = BlerIdentity(
                k_and_n=(int(row["k"]), int(row["n"])),
                base_graph=int(row["base_graph"]),
                lifting_size=int(row["lifting_size"]),
                modulation=row["modulation"],
                decoder_algorithm=row["decoder"],
                decoder_offset=float(row["offset"]),
                iterations=int(row["iterations"]),
                snr_convention=convention,
                rate=row["rate"],
            )
            points.setdefault(identity, []).append(
                (float(row[column]), float(row["bler"]), int(row["blocks"]))
            )
    curves: dict[BlerIdentity, _Curve] = {}
    for identity, measured in points.items():
        measured.sort()
        trials = {trial for _, _, trial in measured}
        if len(trials) != 1:
            raise CompositionError(
                f"inconsistent trial counts for {identity.as_key()}: {sorted(trials)}"
            )
        curves[identity] = _Curve(
            snr_db=tuple(snr for snr, _, _ in measured),
            bler=tuple(bler for _, bler, _ in measured),
            trials=trials.pop(),
        )
    if not curves:
        raise CompositionError(
            f"no {_MEASUREMENT_SYSTEM} rows in {_BLER_RESULTS.name}"
        )
    return BlerTable(curves, provenance=f"results/baseline/g2/{_BLER_RESULTS.name}")


# ==========================================================================
# Candidate feasibility caching and deterministic tie-breaking
# ==========================================================================
#
# BR-4 requires the sweep to be "a cached feasibility table plus per-(rate, SNR,
# blocklength) block-error characterisation composed analytically".  Structural
# feasibility — does a legal TS 38.212 packetisation exist, and can the codec
# emit inside the resulting payload budget — is expensive and deterministic, so
# it is computed once per configuration and reused.
#
# The whole risk of a cache is that two configurations share a key.  The key
# here is therefore the candidate's *complete* configuration identity minus one
# explicitly named, tested exclusion.

#: The candidate fields the feasibility result may depend on.  Anything added
#: to :class:`Candidate` must be classified into this tuple or into
#: :data:`FEASIBILITY_KEY_EXCLUSIONS`, or building a key raises.
FEASIBILITY_KEY_FIELDS: tuple[str, ...] = (
    "dataset",
    "ratio",
    "modulation",
    "ldpc_rate",
    "encode_axis_px",
)

#: Excluded, with a reason rather than by omission.  Structural feasibility is
#: a packetisation and codec-budget question: it reads the transport-block
#: geometry and the payload budget, neither of which is a function of the
#: channel SNR.  The SNR enters the *composition*, through the BLER lookup,
#: never through feasibility — and there is a test that asserts exactly this
#: rather than trusting the comment.
FEASIBILITY_KEY_EXCLUSIONS: dict[str, str] = {
    "snr_db": (
        "structural feasibility is SNR-independent: it depends only on the "
        "transport-block geometry and the codec payload budget"
    ),
}

ELIGIBLE = "eligible"
INFEASIBLE = "infeasible"

#: The documented tie-breaking order, applied left to right, only among
#: candidates whose expected accuracy is *exactly* equal.  A tolerance would be
#: an unpreregistered free parameter, so equality is exact.
#:
#: After expected accuracy the order prefers, in turn: the more reliable link
#: (higher ``P(TB success)``); then the more robust modulation (lower ``Qm``);
#: then the stronger channel code (lower LDPC rate); then the more source
#: information (larger encode axis); and finally the candidate's canonical
#: identity string, which makes the order total.  The last key is what
#: guarantees the result cannot depend on the order candidates were enumerated
#: in — every earlier key can tie, that one cannot.
TIE_BREAK_ORDER: tuple[str, ...] = (
    "expected_accuracy_descending",
    "success_probability_descending",
    "modulation_bits_per_symbol_ascending",
    "ldpc_rate_ascending",
    "encode_axis_px_descending",
    "candidate_id_ascending",
)


@dataclass(frozen=True)
class Candidate:
    """One BR-4 sweep cell: a codec/channel configuration at one SNR."""

    dataset: str
    ratio: str
    modulation: str
    ldpc_rate: str
    encode_axis_px: int
    snr_db: float

    def __post_init__(self) -> None:
        classified = set(FEASIBILITY_KEY_FIELDS) | set(FEASIBILITY_KEY_EXCLUSIONS)
        declared = set(self.__dataclass_fields__)
        if declared != classified:
            raise CompositionError(
                "every Candidate field must be either part of the feasibility "
                "cache key or explicitly excluded with a reason; unclassified: "
                f"{sorted(declared - classified)}"
            )

    def feasibility_key(self) -> tuple[Any, ...]:
        """The complete structural identity, in a fixed field order."""

        return tuple(getattr(self, name) for name in FEASIBILITY_KEY_FIELDS)

    @property
    def candidate_id(self) -> str:
        """A canonical, sortable string identity covering *every* field."""

        return json.dumps(
            {name: getattr(self, name) for name in sorted(self.__dataclass_fields__)},
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class Feasibility:
    """The cached structural verdict for one configuration."""

    feasible: bool
    reason: str | None = None
    code_blocks: int | None = None
    payload_bytes: int | None = None


class FeasibilityCache:
    """A deterministic memo over :meth:`Candidate.feasibility_key`.

    The computation is injected rather than imported so this stays unit-testable
    without running a codec: G-8 will pass the real packetisation-plus-codec
    probe, and the tests here pass a counting stub that proves the cache is
    consulted, that a hit returns exactly what a miss returned, and that no two
    distinct configurations can share an entry.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], Feasibility] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(self._entries, key=repr))

    def feasibility(self, candidate: Candidate, compute: Any) -> Feasibility:
        if not isinstance(candidate, Candidate):
            raise CompositionError(
                f"feasibility takes a Candidate, not {type(candidate).__name__}"
            )
        key = candidate.feasibility_key()
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = compute(candidate)
        if not isinstance(result, Feasibility):
            raise CompositionError(
                "the feasibility probe must return a Feasibility, not "
                f"{type(result).__name__}"
            )
        self._entries[key] = result
        return result


@dataclass(frozen=True)
class CandidateEvaluation:
    """A scored candidate, or an explicit statement of why it is not one.

    ``status`` is one of :data:`ELIGIBLE`, :data:`INFEASIBLE` or
    :data:`UNCHARACTERIZED`.  The third is deliberately *not* a low score:
    a candidate whose BLER was never measured is ineligible, because scoring it
    at all would mean inventing the evidence that would justify it.
    """

    candidate: Candidate
    status: str
    composition: CompositionResult | None = None
    reason: str | None = None

    @property
    def expected_accuracy(self) -> float | None:
        return None if self.composition is None else self.composition.expected_accuracy

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate": json.loads(self.candidate.candidate_id),
            "status": self.status,
            "reason": self.reason,
            "composition": (
                None if self.composition is None else self.composition.as_record()
            ),
        }


def _tie_break_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
    """The documented total order, as a sort key (all keys ascending)."""

    from baseline.ldpc.modulation import bits_per_symbol

    composition = evaluation.composition
    if composition is None:  # pragma: no cover - callers filter first
        raise CompositionError("cannot rank a candidate with no composition")
    candidate = evaluation.candidate
    return (
        -composition.expected_accuracy,
        -composition.success_probability,
        bits_per_symbol(candidate.modulation),
        _rate_value(candidate.ldpc_rate),
        -candidate.encode_axis_px,
        candidate.candidate_id,
    )


def _rate_value(rate: str) -> float:
    """``"1/3"`` and ``"0.3333..."`` are the same rate; order them numerically."""

    text = str(rate)
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        return float(numerator) / float(denominator)
    return float(text)


@dataclass(frozen=True)
class Selection:
    """The outcome of ranking one SNR's candidates."""

    selected: CandidateEvaluation | None
    tied: tuple[CandidateEvaluation, ...]
    tie_break_applied: bool
    evaluations: tuple[CandidateEvaluation, ...]
    reason: str | None = None

    @property
    def eligible(self) -> tuple[CandidateEvaluation, ...]:
        return tuple(e for e in self.evaluations if e.status == ELIGIBLE)

    def counts(self) -> dict[str, int]:
        counts = {ELIGIBLE: 0, INFEASIBLE: 0, UNCHARACTERIZED: 0}
        for evaluation in self.evaluations:
            counts[evaluation.status] = counts.get(evaluation.status, 0) + 1
        return counts

    def as_record(self) -> dict[str, Any]:
        return {
            "selected": None if self.selected is None else self.selected.as_record(),
            "tied_candidates": [
                json.loads(e.candidate.candidate_id) for e in self.tied
            ],
            "tie_break_applied": self.tie_break_applied,
            "tie_break_order": list(TIE_BREAK_ORDER),
            "counts": self.counts(),
            "reason": self.reason,
        }


def select_best(evaluations: Sequence[CandidateEvaluation]) -> Selection:
    """Rank candidates by :data:`TIE_BREAK_ORDER`.  Order-independent.

    Only :data:`ELIGIBLE` candidates compete.  Infeasible and uncharacterized
    candidates are carried into the record so the selection can be audited, and
    are never ranked — an uncharacterized cell has no score to rank *with*.
    """

    ordered = tuple(evaluations)
    eligible = [e for e in ordered if e.status == ELIGIBLE]
    if not eligible:
        return Selection(
            selected=None,
            tied=(),
            tie_break_applied=False,
            evaluations=ordered,
            reason="no_eligible_candidate",
        )
    ranked = sorted(eligible, key=_tie_break_key)
    best = ranked[0]
    top = best.composition.expected_accuracy  # type: ignore[union-attr]
    tied = tuple(
        e
        for e in ranked
        if e.composition is not None and e.composition.expected_accuracy == top
    )
    return Selection(
        selected=best,
        tied=tied,
        tie_break_applied=len(tied) > 1,
        evaluations=ordered,
    )


# ==========================================================================
# System modes
# ==========================================================================
#
# ``params.artifacts.system_values`` carries three classical selection systems,
# and they are genuinely different experiments rather than three labels for one
# curve.  DEC-16 makes modulation an adaptive axis; the two fixed modes exist so
# the adaptation itself can be reported as a controlled variable rather than
# assumed to be free.

#: Full per-SNR adaptation: modulation, LDPC rate and codec axis all re-selected
#: at every SNR, per ``params.baseline.modulation_tuning = adaptive_per_snr``.
CLASSICAL_ADAPTIVE = "classical_adaptive"

#: One modulation for the whole curve, chosen for the grid as a whole; the LDPC
#: rate and codec axis still adapt per SNR.
CLASSICAL_FIXED_MOD = "classical_fixed_mod"

#: One complete MCS — modulation *and* LDPC rate — chosen once at
#: ``params.baseline.fixed_mcs_design_snr_db`` and held at every SNR.  This is
#: the mode PB_2's bounded evidence ran under, because it fixed one
#: configuration and built no adaptation.
CLASSICAL_FIXED_MCS = "classical_fixed_mcs"

SYSTEM_MODES: tuple[str, ...] = (
    CLASSICAL_ADAPTIVE,
    CLASSICAL_FIXED_MOD,
    CLASSICAL_FIXED_MCS,
)


@dataclass(frozen=True)
class SystemModePolicy:
    """What a mode is allowed to re-select as the SNR changes."""

    mode: str
    adapts_modulation: bool
    adapts_ldpc_rate: bool
    adapts_encode_axis: bool

    @property
    def design_snr_db(self) -> float | None:
        """The SNR a fixed MCS is designed at, or ``None`` when it adapts."""

        if self.mode != CLASSICAL_FIXED_MCS:
            return None
        return float(get("baseline.fixed_mcs_design_snr_db"))


_MODE_POLICIES: dict[str, SystemModePolicy] = {
    CLASSICAL_ADAPTIVE: SystemModePolicy(
        CLASSICAL_ADAPTIVE,
        adapts_modulation=True,
        adapts_ldpc_rate=True,
        adapts_encode_axis=True,
    ),
    CLASSICAL_FIXED_MOD: SystemModePolicy(
        CLASSICAL_FIXED_MOD,
        adapts_modulation=False,
        adapts_ldpc_rate=True,
        adapts_encode_axis=True,
    ),
    CLASSICAL_FIXED_MCS: SystemModePolicy(
        CLASSICAL_FIXED_MCS,
        adapts_modulation=False,
        adapts_ldpc_rate=False,
        adapts_encode_axis=False,
    ),
}


def mode_policy(mode: str) -> SystemModePolicy:
    """The policy for a declared system value, or a refusal.

    The three modes are checked against ``params.artifacts.system_values`` at
    call time, so a spec change that renames or drops one breaks here rather
    than producing evidence under a system label the schema does not know.
    """

    declared = tuple(get("artifacts.system_values"))
    missing = [name for name in SYSTEM_MODES if name not in declared]
    if missing:
        raise NotImplementedError(
            f"params.artifacts.system_values no longer declares {missing}"
        )
    if get("baseline.modulation_tuning") != "adaptive_per_snr":
        raise NotImplementedError(
            "params.baseline.modulation_tuning is no longer adaptive_per_snr; "
            "the three classical modes are defined relative to it"
        )
    try:
        return _MODE_POLICIES[mode]
    except KeyError:
        raise CompositionError(
            f"unknown classical system mode {mode!r}; expected one of {SYSTEM_MODES}"
        ) from None


# ==========================================================================
# Two-pass selection — structurally, not by convention
# ==========================================================================
#
# AM-54: pass one selects under BR-8's clean-trained classifier, BR-12 then
# trains on the corpus those selections define and re-scores the *cached* sweep
# once, and iteration terminates there
# (``params.reference_classifier.br4_selection_terminates_after_pass``).  The
# rationale is preregistration: a loop that may be re-run cannot be run until it
# flatters a result.  So the limit is enforced by the object's own state machine
# and survives serialization — a resumed campaign counts the passes it
# inherited.
#
# PB_3 builds this machinery and does not train the artifact-finetuned
# classifier: ``params.reference_classifier.artifact_finetune_gate`` is G-8.

PASS_ONE = 1
PASS_TWO = 2


class SelectionPassError(CompositionError):
    """A violation of the two-pass selection contract."""


def selection_passes() -> tuple[int, ...]:
    """The permitted pass identifiers, read from the spec."""

    total = get("reference_classifier.br4_selection_passes")
    terminates = get("reference_classifier.br4_selection_terminates_after_pass")
    if total != terminates:
        raise NotImplementedError(
            "params.reference_classifier.br4_selection_passes and "
            "br4_selection_terminates_after_pass disagree: "
            f"{total} != {terminates}"
        )
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise NotImplementedError(
            f"params.reference_classifier.br4_selection_passes is not a count: {total!r}"
        )
    return tuple(range(1, total + 1))


@dataclass(frozen=True)
class PassResult:
    """One completed selection pass.  Frozen: a later pass cannot edit it."""

    pass_id: int
    mode: str
    scorer: str
    selections: tuple[Selection, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "mode": self.mode,
            "scorer": self.scorer,
            "selections": [selection.as_record() for selection in self.selections],
        }


class PassContext:
    """What a pass is allowed to see: strictly the passes before it.

    Pass one cannot read pass two because pass two does not exist yet *and*
    because the accessor refuses any pass identifier at or above the current
    one — so a context accidentally retained across passes still cannot leak
    forwards.
    """

    def __init__(self, pass_id: int, completed: Mapping[int, PassResult]):
        self._pass_id = pass_id
        self._completed = dict(completed)

    @property
    def pass_id(self) -> int:
        return self._pass_id

    def result_of(self, pass_id: int) -> PassResult:
        if pass_id >= self._pass_id:
            raise SelectionPassError(
                f"pass {self._pass_id} may not read pass {pass_id}: a selection "
                "pass sees only the passes that completed before it"
            )
        try:
            return self._completed[pass_id]
        except KeyError:
            raise SelectionPassError(f"pass {pass_id} has not run") from None

    def previous(self) -> PassResult | None:
        """Pass one has no predecessor; pass two has exactly one."""

        return self._completed.get(self._pass_id - 1)


class SelectionCampaign:
    """A BR-4 selection run: at most two passes, in order, once each.

    The state machine is the enforcement.  Every refusal below is a separate
    test, because "documented as two passes" is exactly the kind of constraint
    that survives right up until someone loops it.
    """

    def __init__(self, mode: str, *, completed: Sequence[PassResult] = ()):
        self.policy = mode_policy(mode)
        self.mode = self.policy.mode
        self._allowed = selection_passes()
        self._completed: dict[int, PassResult] = {}
        for result in completed:
            self._admit_resumed(result)

    def _admit_resumed(self, result: PassResult) -> None:
        if not isinstance(result, PassResult):
            raise SelectionPassError(
                f"resumed state must be PassResults, not {type(result).__name__}"
            )
        if result.pass_id not in self._allowed:
            raise SelectionPassError(
                f"resumed state carries unknown pass {result.pass_id!r}; "
                f"permitted passes are {self._allowed}"
            )
        if result.pass_id in self._completed:
            raise SelectionPassError(
                f"resumed state repeats pass {result.pass_id}"
            )
        if result.mode != self.mode:
            raise SelectionPassError(
                f"resumed pass {result.pass_id} ran under mode {result.mode!r}, "
                f"not {self.mode!r}"
            )
        self._completed[result.pass_id] = result

    @property
    def completed_passes(self) -> tuple[int, ...]:
        return tuple(sorted(self._completed))

    @property
    def exhausted(self) -> bool:
        return set(self._completed) == set(self._allowed)

    def result_of(self, pass_id: int) -> PassResult:
        try:
            return self._completed[pass_id]
        except KeyError:
            raise SelectionPassError(f"pass {pass_id} has not run") from None

    def run_pass(self, pass_id: Any, selector: Any, *, scorer: str) -> PassResult:
        """Run one pass.  Out of order, twice, or a third time: all refused."""

        if isinstance(pass_id, bool) or not isinstance(pass_id, int):
            raise SelectionPassError(f"pass identifier is not an integer: {pass_id!r}")
        if pass_id not in self._allowed:
            raise SelectionPassError(
                f"unknown selection pass {pass_id}; BR-4 selection runs passes "
                f"{self._allowed} and then stops (AM-54)"
            )
        if pass_id in self._completed:
            raise SelectionPassError(f"selection pass {pass_id} has already run")
        if self.exhausted:
            raise SelectionPassError(
                f"selection is exhausted after pass {max(self._allowed)}; "
                "iteration terminates there (AM-54)"
            )
        expected = len(self._completed) + 1
        if pass_id != expected:
            raise SelectionPassError(
                f"selection pass {pass_id} cannot run before pass {expected}"
            )
        if not str(scorer).strip():
            raise SelectionPassError("a selection pass must name its scorer")
        if any(done.scorer == scorer for done in self._completed.values()):
            raise SelectionPassError(
                f"pass {pass_id} reuses the scorer {scorer!r} of an earlier pass; "
                "the second pass exists precisely because the scorer changed"
            )
        context = PassContext(pass_id, dict(self._completed))
        selections = selector(context)
        if isinstance(selections, Selection):
            selections = (selections,)
        selections = tuple(selections)
        for selection in selections:
            if not isinstance(selection, Selection):
                raise SelectionPassError(
                    "a selection pass must return Selections, not "
                    f"{type(selection).__name__}"
                )
        result = PassResult(
            pass_id=pass_id,
            mode=self.mode,
            scorer=str(scorer),
            selections=selections,
        )
        self._completed[pass_id] = result
        return result

    def state(self) -> tuple[PassResult, ...]:
        """Serializable completed state, for a resumed campaign."""

        return tuple(self._completed[pass_id] for pass_id in self.completed_passes)

    def as_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "permitted_passes": list(self._allowed),
            "completed_passes": list(self.completed_passes),
            "exhausted": self.exhausted,
            "passes": [result.as_record() for result in self.state()],
        }


@dataclass(frozen=True)
class CurveSelection:
    """A whole SNR curve selected under one mode."""

    mode: str
    per_snr: tuple[tuple[float, Selection], ...]
    held_fixed: dict[str, Any]

    def selection_at(self, snr_db: float) -> Selection:
        for snr, selection in self.per_snr:
            if snr == snr_db:
                return selection
        raise CompositionError(f"no selection at {snr_db} dB")

    def as_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "held_fixed": dict(self.held_fixed),
            "per_snr": [
                {"snr_db": snr, "selection": selection.as_record()}
                for snr, selection in self.per_snr
            ],
        }


def resolve_curve(
    mode: str, evaluations_by_snr: Mapping[float, Sequence[CandidateEvaluation]]
) -> CurveSelection:
    """Select one configuration per SNR under the constraints of ``mode``.

    The three modes differ in *what may move between SNR points*, which is the
    only thing that distinguishes them:

    * ``classical_adaptive`` — everything re-selects, independently per SNR;
    * ``classical_fixed_mod`` — one modulation is chosen for the whole grid
      (the one whose per-SNR bests sum highest), and the LDPC rate and codec
      axis then adapt underneath it;
    * ``classical_fixed_mcs`` — the complete configuration is chosen once at
      ``params.baseline.fixed_mcs_design_snr_db`` and held everywhere.

    The design SNR must be a point on the supplied grid: silently snapping to a
    nearby point would make the fixed arm's design depend on grid spacing.
    """

    policy = mode_policy(mode)
    grid = tuple(sorted(evaluations_by_snr))
    if not grid:
        raise CompositionError("no SNR points supplied")

    if policy.mode == CLASSICAL_ADAPTIVE:
        per_snr = tuple(
            (snr, select_best(evaluations_by_snr[snr])) for snr in grid
        )
        return CurveSelection(policy.mode, per_snr, held_fixed={})

    if policy.mode == CLASSICAL_FIXED_MOD:
        from baseline.ldpc.modulation import bits_per_symbol

        totals: dict[str, tuple[float, int, str]] = {}
        for modulation in sorted(
            {
                evaluation.candidate.modulation
                for evaluations in evaluations_by_snr.values()
                for evaluation in evaluations
            }
        ):
            best_per_snr = [
                select_best(
                    [
                        evaluation
                        for evaluation in evaluations_by_snr[snr]
                        if evaluation.candidate.modulation == modulation
                    ]
                )
                for snr in grid
            ]
            if any(selection.selected is None for selection in best_per_snr):
                continue
            total = sum(
                selection.selected.composition.expected_accuracy  # type: ignore[union-attr]
                for selection in best_per_snr
            )
            totals[modulation] = (
                -total,
                bits_per_symbol(modulation),
                modulation,
            )
        if not totals:
            raise CompositionError(
                "no modulation is eligible at every SNR on the grid; a fixed "
                "modulation cannot be held across it"
            )
        chosen = min(totals, key=lambda name: totals[name])
        per_snr = tuple(
            (
                snr,
                select_best(
                    [
                        evaluation
                        for evaluation in evaluations_by_snr[snr]
                        if evaluation.candidate.modulation == chosen
                    ]
                ),
            )
            for snr in grid
        )
        return CurveSelection(
            policy.mode, per_snr, held_fixed={"modulation": chosen}
        )

    design_snr = policy.design_snr_db
    if design_snr not in evaluations_by_snr:
        raise CompositionError(
            f"the fixed-MCS design SNR {design_snr} dB is not on the supplied "
            f"grid {list(grid)}; refusing to snap to a neighbouring point"
        )
    design = select_best(evaluations_by_snr[design_snr])
    if design.selected is None:
        raise CompositionError(
            f"no eligible candidate at the fixed-MCS design SNR {design_snr} dB"
        )
    held = design.selected.candidate
    per_snr = tuple(
        (
            snr,
            select_best(
                [
                    evaluation
                    for evaluation in evaluations_by_snr[snr]
                    if evaluation.candidate.feasibility_key()
                    == held.feasibility_key()
                ]
            ),
        )
        for snr in grid
    )
    return CurveSelection(
        policy.mode,
        per_snr,
        held_fixed={
            "design_snr_db": design_snr,
            "modulation": held.modulation,
            "ldpc_rate": held.ldpc_rate,
            "encode_axis_px": held.encode_axis_px,
            "packet_count": get("baseline.fixed_mcs_packet_count"),
        },
    )


# ==========================================================================
# Full-sweep guard
# ==========================================================================
#
# This is the one mechanism standing between an ordinary PB_3 call and an
# accidental G-8 campaign.  The BR-4 validation sweep is a scientific event: it
# selects the operating points the whole experiment is then reported at, and it
# must happen once, deliberately, at its gate.  So the entry point refuses any
# workload above a bounded budget unless an explicit authorization object is
# handed to it.
#
# Deliberately absent, and each absence is tested:
#
#   * no environment variable is read here.  A variable exported once in a
#     shell profile is exactly how a guard gets disarmed and stays disarmed;
#   * no default-true flag and no CLI option.  The authorization is a typed
#     object a caller must construct on purpose;
#   * no authorization is constructed anywhere in this repository outside the
#     tests that prove the refusals fire.

#: The bounded PB_3 budget.  These are unit-test scale on purpose: they are
#: large enough for the fixtures this phase exercises and far too small for any
#: real validation campaign, so a confused successor session hits the refusal
#: rather than a long run.
MAX_UNAUTHORIZED_CANDIDATES = 64  # literal-ok: bounded PB_3 test budget, not a spec parameter
MAX_UNAUTHORIZED_SAMPLES = 25  # literal-ok: bounded PB_3 test budget, not a spec parameter
MAX_UNAUTHORIZED_WORKLOAD = 512  # literal-ok: bounded PB_3 test budget, not a spec parameter

#: The gate an authorization must name.  Anything else is malformed.
G8_GATE = "G-8"


class SweepBudgetError(CompositionError):
    """A workload above the bounded budget, with no valid G-8 authorization."""


@dataclass(frozen=True)
class G8Authorization:
    """Explicit, typed permission to run the BR-4 validation sweep.

    Constructing one is a deliberate act with a named authoriser and a reason,
    and it carries its own limits so an authorization for one campaign cannot
    silently permit a larger one.  **Nothing in this repository constructs one
    outside the tests that prove the refusals work.**
    """

    gate: str
    authorized_by: str
    reason: str
    max_candidates: int
    max_samples: int

    def __post_init__(self) -> None:
        _validate_authorization(self)


def _validate_authorization(authorization: G8Authorization) -> None:
    """Every field checked, on construction *and* on use.

    Checked twice on purpose: ``__post_init__`` can be skipped entirely by
    ``object.__new__`` or by unpickling, so the guard re-validates whatever it
    is handed rather than trusting that it was built through the constructor.
    """

    if authorization.gate != G8_GATE:
        raise SweepBudgetError(
            f"a sweep authorization must name gate {G8_GATE!r}, "
            f"not {authorization.gate!r}"
        )
    for name in ("authorized_by", "reason"):
        if not str(getattr(authorization, name)).strip():
            raise SweepBudgetError(f"sweep authorization has no {name}")
    for name in ("max_candidates", "max_samples"):
        value = getattr(authorization, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SweepBudgetError(
                f"sweep authorization {name} is not a positive count: {value!r}"
            )


@dataclass(frozen=True)
class SweepBudget:
    """The limits in force for one call, and where they came from."""

    max_candidates: int
    max_samples: int
    max_workload: int | None
    authorized: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_samples": self.max_samples,
            "max_workload": self.max_workload,
            "authorized": self.authorized,
            "gate": G8_GATE,
        }


def sweep_budget(authorization: G8Authorization | None) -> SweepBudget:
    """Resolve the limits.  A malformed authorization is a refusal, not a pass."""

    if authorization is None:
        return SweepBudget(
            max_candidates=MAX_UNAUTHORIZED_CANDIDATES,
            max_samples=MAX_UNAUTHORIZED_SAMPLES,
            max_workload=MAX_UNAUTHORIZED_WORKLOAD,
            authorized=False,
        )
    if not isinstance(authorization, G8Authorization):
        raise SweepBudgetError(
            "sweep authorization must be a G8Authorization, not "
            f"{type(authorization).__name__}"
        )
    _validate_authorization(authorization)
    return SweepBudget(
        max_candidates=authorization.max_candidates,
        max_samples=authorization.max_samples,
        max_workload=None,
        authorized=True,
    )


def check_sweep_budget(
    *,
    candidates: int,
    samples: int,
    authorization: G8Authorization | None = None,
) -> SweepBudget:
    """Refuse a workload the caller is not authorized to run.

    Three separate limits, because a sweep can be too large in three ways: too
    many configurations, too many images per configuration, or a product of two
    individually-modest numbers.  Each is checked on its own so the refusal
    message names the one that fired.
    """

    for name, value in (("candidates", candidates), ("samples", samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SweepBudgetError(f"{name} is not a count: {value!r}")
    budget = sweep_budget(authorization)
    if candidates > budget.max_candidates:
        raise SweepBudgetError(
            f"{candidates} candidates exceeds the bounded limit of "
            f"{budget.max_candidates}; the BR-4 validation sweep runs at G-8 "
            "under an explicit authorization, not here"
        )
    if samples > budget.max_samples:
        raise SweepBudgetError(
            f"{samples} samples per cell exceeds the bounded limit of "
            f"{budget.max_samples}; the BR-4 validation sweep runs at G-8 "
            "under an explicit authorization, not here"
        )
    workload = candidates * samples
    if budget.max_workload is not None and workload > budget.max_workload:
        raise SweepBudgetError(
            f"a combined workload of {workload} cells exceeds the bounded limit "
            f"of {budget.max_workload}; the BR-4 validation sweep runs at G-8 "
            "under an explicit authorization, not here"
        )
    return budget


def select_operating_points(
    mode: str,
    evaluations_by_snr: Mapping[float, Sequence[CandidateEvaluation]],
    *,
    samples_per_cell: int,
    authorization: G8Authorization | None = None,
) -> CurveSelection:
    """The ordinary BR-4 selection entry point, behind the sweep guard.

    The guard runs *before* any work, so an over-budget call costs nothing and
    fails immediately rather than part way through a campaign.
    """

    candidates = sum(len(tuple(group)) for group in evaluations_by_snr.values())
    check_sweep_budget(
        candidates=candidates,
        samples=samples_per_cell,
        authorization=authorization,
    )
    return resolve_curve(mode, evaluations_by_snr)


def evaluate_candidate(
    candidate: Candidate,
    *,
    feasibility: Feasibility,
    block_identities: Sequence[Mapping[str, Any] | BlerIdentity],
    bler_table: BlerTable,
    codec_accuracy: MeasuredCodecAccuracy,
    outage_accuracy: MeasuredOutageAccuracy,
) -> CandidateEvaluation:
    """Turn one candidate into a scored evaluation, or into a stated refusal.

    This is where the three fail-closed behaviours meet.  A structurally
    infeasible candidate never reaches the BLER table.  A candidate whose code
    blocks are not *all* characterised at this SNR is
    :data:`UNCHARACTERIZED` — not partially scored, not scored on the blocks
    that happened to resolve, and not scored with the missing blocks treated as
    error-free.  Only a candidate with a measured BLER for every block is
    composed and made eligible.
    """

    if not isinstance(feasibility, Feasibility):
        raise CompositionError(
            f"feasibility must be a Feasibility, not {type(feasibility).__name__}"
        )
    if not feasibility.feasible:
        return CandidateEvaluation(
            candidate=candidate,
            status=INFEASIBLE,
            reason=feasibility.reason or "infeasible",
        )
    identities = tuple(block_identities)
    if not identities:
        raise CompositionError(
            "a feasible candidate has at least one code block; refusing to "
            "compose an empty transport block"
        )
    if (
        feasibility.code_blocks is not None
        and feasibility.code_blocks != len(identities)
    ):
        raise CompositionError(
            f"{len(identities)} block identities supplied for a transport block "
            f"of {feasibility.code_blocks} code blocks"
        )
    blers: list[float] = []
    for index, identity in enumerate(identities):
        lookup = bler_table.lookup(identity, candidate.snr_db)
        if not lookup.characterized:
            return CandidateEvaluation(
                candidate=candidate,
                status=UNCHARACTERIZED,
                reason=f"code_block_{index}:{lookup.reason}",
            )
        blers.append(lookup.require())
    return CandidateEvaluation(
        candidate=candidate,
        status=ELIGIBLE,
        composition=compose(
            blers,
            codec_accuracy=codec_accuracy,
            outage_accuracy=outage_accuracy,
        ),
    )
