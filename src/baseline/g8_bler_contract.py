"""Frozen G8_B contract for the BLER characterization runner.

B1 freezes *what* the G8_C runner must obey — trial-count ownership, seed
derivation, random-stream separation, work-unit request and result schemas,
count-authoritative BLER semantics, full-strength versus bounded-smoke
separation, and a diagnostic-only confidence policy — before any measurement
exists.  Freezing these before data is what stops a later phase from tuning the
measurement until it flatters a result.

This module is deliberately a pure contract: it contains no LDPC simulation, no
channel execution, no dataset or classifier import, no training, no selection,
and no authorization construction.  The stream helpers here exist so the frozen
RNG semantics are *testable* on tiny synthetic arrays; they never touch a
required G8 work-unit state and never constitute scientific execution.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from types import MappingProxyType
from typing import Any

import numpy as np

from baseline.classical.composition import BlerIdentity
from baseline.g8_campaign import (
    CAMPAIGN_MANIFEST,
    REQUIRED_BLER_IDENTITIES,
    canonical_json,
    load_campaign_manifest,
    load_required_bler_identities,
    sha256_bytes,
    sha256_file,
)
from config.params import get

# --------------------------------------------------------------------------
# Schema versions
# --------------------------------------------------------------------------

BLER_TOOLING_CONTRACT_SCHEMA_VERSION = 1
BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION = 1
BLER_WORK_UNIT_RESULT_SCHEMA_VERSION = 1

REQUEST_ARTIFACT_ROLE = "g8_bler_work_unit_request"
RESULT_ARTIFACT_ROLE = "g8_bler_work_unit_result"
TOOLING_CONTRACT_ARTIFACT_ROLE = "g8_bler_tooling_contract"
CONTRACT_ID_PREFIX = "g8bler"

# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------

#: The exact string already recorded in the live campaign state.  It is
#: reproduced verbatim rather than reinterpreted.
SEED_DERIVATION_IDENTITY = "sha256(campaign_id,work_unit_id,purpose)-v1"
SEED_DOMAIN_SEPARATOR = "capstone:g8:bler-seed:v1"
SEED_INPUT_ENCODING = (
    'utf8 bytes of compact JSON array ["<domain>","<campaign_id>",'
    '"<work_unit_id>","<purpose>"] with separators (",",":") and ensure_ascii'
)
SEED_DIGEST = "sha256"
SEED_OUTPUT_RULE = (
    "first 8 digest bytes as an unsigned big-endian integer; 0 <= seed < 2**64; "
    "no modulo; zero is a valid seed"
)

PURPOSE_INFORMATION_BITS = "information_bits"
PURPOSE_AWGN_REAL = "awgn_real"
PURPOSE_AWGN_IMAG = "awgn_imag"
#: Closed and explicit.  Separate streams stop a change to information-bit
#: generation from silently changing the noise draws.
SEED_PURPOSES = (PURPOSE_INFORMATION_BITS, PURPOSE_AWGN_REAL, PURPOSE_AWGN_IMAG)

_SEED_BYTES = 8  # literal-ok: uint64 seed width in bytes, fixed by SEED_OUTPUT_RULE
SEED_WIDTH_BITS = 64  # literal-ok: uint64 seed width, fixed by SEED_OUTPUT_RULE
_SEED_MODULUS = 1 << SEED_WIDTH_BITS

SEED_FORBIDDEN_INPUTS = (
    "python_hash",
    "process_id",
    "worker_id",
    "shard_number",
    "enumeration_order",
    "dictionary_order",
    "wall_clock_time",
    "hostname",
    "device",
    "absolute_path",
    "commit_time",
    "batch_number",
    "retry_number",
)

# --------------------------------------------------------------------------
# RNG engine contract
# --------------------------------------------------------------------------

RNG_LIBRARY = "numpy"
RNG_LIBRARY_VERSION = "2.5.1"
RNG_BIT_GENERATOR = "Philox"
#: Philox4x64 emits four uint64 words per counter step, so ``advance(n)`` skips
#: ``4 * n`` words.  Verified against ``random_raw`` rather than assumed.
PHILOX_WORDS_PER_COUNTER_STEP = 4  # literal-ok: Philox4x64 block width in uint64 words
BITS_PER_WORD = 64  # literal-ok: uint64 word width, fixed by the bit-extraction rule

INFORMATION_BIT_API = "numpy.random.Philox(key=seed).random_raw(words)"
INFORMATION_BIT_EXTRACTION = "bit_i = (word[i // 64] >> (i % 64)) & 1"
INFORMATION_BIT_DTYPE = "uint8"
NORMAL_API = "numpy.random.Generator(numpy.random.Philox(key=seed)).standard_normal(count)"
NORMAL_DTYPE = "float64"
ARRAY_ORDER = "C"

#: The information-bit stream is randomly addressable *and* chunk-boundary
#: invariant.  The Gaussian streams are chunk-boundary invariant but consumed
#: sequentially from index zero, because the ziggurat consumes a variable
#: number of raw words per draw.  That is sufficient under
#: :data:`RESUME_GRANULARITY`.
INFORMATION_BIT_STREAM_ADDRESSABLE = True
NORMAL_STREAM_ADDRESSABLE = False
STREAMS_ARE_CHUNK_BOUNDARY_INVARIANT = True

# --------------------------------------------------------------------------
# Execution classes, trial-count ownership and resume granularity
# --------------------------------------------------------------------------

EXECUTION_CLASS_FULL_STRENGTH = "full_strength"
EXECUTION_CLASS_BOUNDED_SMOKE = "bounded_smoke"
EXECUTION_CLASSES = (EXECUTION_CLASS_FULL_STRENGTH, EXECUTION_CLASS_BOUNDED_SMOKE)

#: BR-4 owns the G-8 trial count.  ``baseline.ldpc_bler_reference.blocks_per_snr``
#: is the narrow G-2 reference experiment's own count; the two currently hold
#: the same value but have different semantic ownership and must never be
#: confused.
FULL_STRENGTH_TRIAL_COUNT_SOURCE = "params.baseline.bler_characterisation_trials"
FULL_STRENGTH_TRIAL_COUNT_PARAMETER = "baseline.bler_characterisation_trials"
G2_REFERENCE_TRIAL_COUNT_KEY_NOT_USED = "params.baseline.ldpc_bler_reference.blocks_per_snr"
BOUNDED_SMOKE_TRIAL_COUNT_SOURCE = "g8b_bounded_smoke_ceiling_not_a_scientific_parameter"

ADAPTIVE_STOPPING_PERMITTED = False
NO_EARLY_STOPPING_RULE = (
    "the full-strength trial count is fixed before execution and never depends "
    "on observed errors; no adaptive stopping and no observed-error threshold"
)
NO_INTERPOLATION_RULE = (
    "a work unit matches exactly one required identity entry; no generalization, "
    "nearest match, interpolation, extrapolation, SNR rounding or implicit default"
)

RESUME_GRANULARITY = "work_unit_atomic"
RESUME_POLICY = (
    "G-8 BLER recovery is work-unit granular. Only a complete, atomically "
    "committed work-unit result is resumable evidence. If execution stops "
    "partway through a work unit the incomplete temporary output is discarded "
    "and that work unit restarts from trial zero using the same seeds. There is "
    "no mid-work-unit trial cursor and no seek into a Gaussian stream. Shard "
    "assignment and work-unit execution order never affect outputs."
)
MID_WORK_UNIT_RESUME_PERMITTED = False

BOUNDED_SMOKE_MAX_WORK_UNITS = 3
BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT = 16  # literal-ok: G8_B-owned smoke ceiling, not a scientific parameter
BOUNDED_SMOKE_SELECTION_RULE = (
    "the first canonical required identity for each configured modulation, or an "
    "explicitly labelled synthetic fixture identity"
)
BOUNDED_SMOKE_LABEL = "NON-SCIENTIFIC BOUNDED SMOKE"

# --------------------------------------------------------------------------
# Count-authoritative measurement semantics
# --------------------------------------------------------------------------

COUNT_FIELDS_AUTHORITATIVE = (
    "trials_completed",
    "information_bits",
    "bit_errors",
    "block_errors",
)
BLER_POINT_ESTIMATE_RULE = "block_errors / trials_completed"
BER_POINT_ESTIMATE_RULE = "bit_errors / information_bits"
COUNTS_OVERRIDE_STORED_FLOATS = True

CONFIDENCE_INTERVAL_METHOD = "wilson_score"
CONFIDENCE_INTERVAL_PERCENT = 95  # literal-ok: G8-owned diagnostic level; deliberately not baseline.ldpc_bler_reference.confidence_percent
CONFIDENCE_INTERVAL_ROLE = (
    "diagnostic only; not used in BR-4 ranking or eligibility; not a stopping rule"
)
CONFIDENCE_PARAMETER_SOURCE = (
    "no general confidence parameter governs G-8; "
    "baseline.ldpc_bler_reference.confidence_percent is G-2-specific and is not read here"
)

STATUS_INCOMPLETE = "incomplete"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
RESULT_STATUSES = (STATUS_INCOMPLETE, STATUS_COMPLETE, STATUS_FAILED)

TEST_SPLIT_ACCESS = 0

REQUEST_FIELDS = (
    "schema_version",
    "artifact_role",
    "execution_class",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "work_unit_id",
    "bler_identity",
    "snr_db",
    "source_packet_config_ids",
    "trials_requested",
    "trial_count_source",
    "seed_derivation_identity",
    "seed_domain_separator",
    "stream_seeds",
    "scientific_evidence",
    "merge_eligible",
    "test_split_access",
    "label",
)

RESULT_FIELDS = ("schema_version", "artifact_role", "status", "identity", "measurement",
                 "execution_metadata", "disposition")
RESULT_IDENTITY_FIELDS = (
    "execution_class",
    "request_sha256",
    "campaign_id",
    "campaign_manifest_sha256",
    "required_bler_artifact_sha256",
    "selection_policy_sha256",
    "work_unit_id",
    "bler_identity",
    "snr_db",
    "source_packet_config_ids",
    "trials_requested",
    "trial_count_source",
    "seed_derivation_identity",
    "seed_domain_separator",
    "stream_seeds",
    "implementation",
)
RESULT_MEASUREMENT_FIELDS = (
    "trials_completed",
    "information_bits",
    "bit_errors",
    "block_errors",
    "ber",
    "bler",
    "confidence_interval_method",
    "confidence_interval_percent",
    "bler_confidence_low",
    "bler_confidence_high",
    "confidence_interval_role",
)
RESULT_EXECUTION_METADATA_FIELDS = (
    "wall_time_s",
    "hostname",
    "device",
    "shard_index",
    "shard_count",
    "attempt",
)
RESULT_DISPOSITION_FIELDS = (
    "scientific_evidence",
    "merge_eligible",
    "test_split_access",
    "required_coverage_contribution",
)
#: Provenance only.  These never enter the measurement identity digest.
NON_IDENTITY_EXECUTION_METADATA = RESULT_EXECUTION_METADATA_FIELDS
IMPLEMENTATION_FIELDS = (
    "rng_library",
    "rng_library_version",
    "rng_bit_generator",
    "request_schema_version",
    "result_schema_version",
)


class G8BlerContractError(RuntimeError):
    """A frozen G8_B BLER runner contract invariant was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8BlerContractError(message)


def _require_exact_int(value: Any, name: str) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int),
        f"{name} must be an integer, not a boolean or float",
    )
    return int(value)


def _require_nonnegative_int(value: Any, name: str) -> int:
    number = _require_exact_int(value, name)
    _require(number >= 0, f"{name} must be non-negative")
    return number


def _require_nonblank_str(value: Any, name: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{name} must be a non-blank string")
    return value


def _require_finite(value: Any, name: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be a real number",
    )
    _require(math.isfinite(float(value)), f"{name} must be finite; NaN and infinity are rejected")
    return float(value)


def _identical(left: Any, right: Any) -> bool:
    """Exact structural equality, so ``13`` never matches ``13.0``."""

    return canonical_json(left) == canonical_json(right)


# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------


def seed_material(campaign_id: str, work_unit_id: str, purpose: str) -> bytes:
    """Return the frozen seed pre-image bytes.

    The pre-image is an ordered JSON *array*, so there is no dictionary
    ordering to depend on, no whitespace to depend on, and no delimiter a
    caller could smuggle through an identifier: JSON escaping makes
    ``a","b`` distinct from a genuine field boundary.
    """

    _require_nonblank_str(campaign_id, "campaign_id")
    _require_nonblank_str(work_unit_id, "work_unit_id")
    _require(purpose in SEED_PURPOSES, f"unknown random purpose {purpose!r}")
    return json.dumps(
        [SEED_DOMAIN_SEPARATOR, campaign_id, work_unit_id, purpose],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def derive_seed(campaign_id: str, work_unit_id: str, purpose: str) -> int:
    """Derive the frozen uint64 stream seed for one work unit and purpose."""

    digest = hashlib.sha256(seed_material(campaign_id, work_unit_id, purpose)).digest()
    seed = int.from_bytes(digest[:_SEED_BYTES], "big")
    _require(0 <= seed < _SEED_MODULUS, "derived seed left the unsigned 64-bit range")
    return seed


def seed_record(campaign_id: str, work_unit_id: str, purpose: str) -> dict[str, Any]:
    """Return the structured provenance record for one derived seed."""

    material = seed_material(campaign_id, work_unit_id, purpose)
    return {
        "seed_derivation_identity": SEED_DERIVATION_IDENTITY,
        "seed_domain_separator": SEED_DOMAIN_SEPARATOR,
        "purpose": purpose,
        "material_sha256": sha256_bytes(material),
        "seed_uint64": int.from_bytes(hashlib.sha256(material).digest()[:_SEED_BYTES], "big"),
    }


def stream_seed_records(campaign_id: str, work_unit_id: str) -> dict[str, dict[str, Any]]:
    """Return one seed record per allowed purpose, keyed by purpose."""

    return {
        purpose: seed_record(campaign_id, work_unit_id, purpose) for purpose in SEED_PURPOSES
    }


# --------------------------------------------------------------------------
# Frozen random streams (pure; never scientific execution)
# --------------------------------------------------------------------------


def philox_words(seed: int, start: int, count: int) -> np.ndarray:
    """Return ``count`` uint64 words of the Philox stream starting at ``start``."""

    _require_nonnegative_int(seed, "seed")
    _require(seed < _SEED_MODULUS, "seed exceeds the unsigned 64-bit range")
    _require_nonnegative_int(start, "start")
    _require_nonnegative_int(count, "count")
    if count == 0:
        return np.zeros(0, dtype=np.uint64)
    step, offset = divmod(start, PHILOX_WORDS_PER_COUNTER_STEP)
    generator = np.random.Philox(key=seed)
    if step:
        generator.advance(step)
    return np.asarray(generator.random_raw(offset + count)[offset:], dtype=np.uint64)


def information_bit_stream(seed: int, start: int, count: int) -> np.ndarray:
    """Return ``count`` information bits from index ``start``, LSB-first.

    ``bit_i = (word[i // 64] >> (i % 64)) & 1``.  Bits left unused in the final
    word are discarded; they are never carried into another work unit.
    """

    _require_nonnegative_int(start, "start")
    _require_nonnegative_int(count, "count")
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    first_word, last_word = start // BITS_PER_WORD, (start + count - 1) // BITS_PER_WORD
    words = philox_words(seed, first_word, last_word - first_word + 1)
    shifts = np.arange(BITS_PER_WORD, dtype=np.uint64)
    bits = ((words[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
    flat = np.ascontiguousarray(bits.reshape(-1))
    head = start - first_word * BITS_PER_WORD
    return flat[head : head + count]


def normal_stream(seed: int, count: int) -> np.ndarray:
    """Return ``count`` standard normal draws from the start of the stream.

    Gaussian streams are chunk-boundary invariant but not randomly addressable;
    under :data:`RESUME_POLICY` a work unit always consumes them from index zero.
    """

    _require_nonnegative_int(seed, "seed")
    _require(seed < _SEED_MODULUS, "seed exceeds the unsigned 64-bit range")
    _require_nonnegative_int(count, "count")
    generator = np.random.Generator(np.random.Philox(key=seed))
    return np.ascontiguousarray(generator.standard_normal(count), dtype=np.float64)


def rng_contract() -> dict[str, Any]:
    """Return the frozen, machine-readable RNG engine contract."""

    return {
        "library": RNG_LIBRARY,
        "library_version": RNG_LIBRARY_VERSION,
        "bit_generator": RNG_BIT_GENERATOR,
        "seed_width_bits": SEED_WIDTH_BITS,
        "stream_construction": "one Philox(key=derived uint64 seed) per (work_unit_id, purpose)",
        "words_per_counter_step": PHILOX_WORDS_PER_COUNTER_STEP,
        "bits_per_word": BITS_PER_WORD,
        "information_bit_api": INFORMATION_BIT_API,
        "information_bit_extraction": INFORMATION_BIT_EXTRACTION,
        "information_bit_dtype": INFORMATION_BIT_DTYPE,
        "normal_api": NORMAL_API,
        "normal_dtype": NORMAL_DTYPE,
        "array_order": ARRAY_ORDER,
        "chunk_boundary_invariant": STREAMS_ARE_CHUNK_BOUNDARY_INVARIANT,
        "information_bit_stream_addressable": INFORMATION_BIT_STREAM_ADDRESSABLE,
        "normal_stream_addressable": NORMAL_STREAM_ADDRESSABLE,
        "purposes": list(SEED_PURPOSES),
        "trailing_bits_discarded_never_carried": True,
    }


def implementation_binding() -> dict[str, Any]:
    """Return the dependency binding recorded inside every result identity."""

    return {
        "rng_library": RNG_LIBRARY,
        "rng_library_version": RNG_LIBRARY_VERSION,
        "rng_bit_generator": RNG_BIT_GENERATOR,
        "request_schema_version": BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "result_schema_version": BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
    }


def installed_rng_version_matches() -> bool:
    return str(np.__version__) == RNG_LIBRARY_VERSION


# --------------------------------------------------------------------------
# Trial count and confidence
# --------------------------------------------------------------------------


def full_strength_trial_count() -> int:
    """Read the BR-4 trial count from its own parameter and nowhere else."""

    value = get(FULL_STRENGTH_TRIAL_COUNT_PARAMETER)
    count = _require_exact_int(value, FULL_STRENGTH_TRIAL_COUNT_SOURCE)
    _require(count > 0, f"{FULL_STRENGTH_TRIAL_COUNT_SOURCE} must be positive")
    return count


def wilson_interval(
    errors: int,
    trials: int,
    confidence_percent: float = CONFIDENCE_INTERVAL_PERCENT,
) -> tuple[float, float]:
    """Return the diagnostic Wilson score interval for a binomial proportion."""

    errors = _require_nonnegative_int(errors, "errors")
    trials = _require_nonnegative_int(trials, "trials")
    _require(trials > 0, "a confidence interval needs a positive trial count")
    _require(errors <= trials, "errors cannot exceed trials")
    percent = _require_finite(confidence_percent, "confidence_percent")
    _require(0.0 < percent < 100.0, "confidence percent bounds")  # literal-ok: percentage range, not a parameter
    z = NormalDist().inv_cdf(0.5 + percent / 200.0)  # literal-ok: two-sided percent-to-quantile conversion
    p = errors / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator  # literal-ok: Wilson closed form
    return max(0.0, centre - margin), min(1.0, centre + margin)


def confidence_policy() -> dict[str, Any]:
    return {
        "method": CONFIDENCE_INTERVAL_METHOD,
        "percent": CONFIDENCE_INTERVAL_PERCENT,
        "role": CONFIDENCE_INTERVAL_ROLE,
        "parameter_source": CONFIDENCE_PARAMETER_SOURCE,
        "adaptive_stopping_permitted": ADAPTIVE_STOPPING_PERMITTED,
        "no_early_stopping_rule": NO_EARLY_STOPPING_RULE,
    }


def recompute_measurements(
    *,
    trials_completed: int,
    information_bits: int,
    bit_errors: int,
    block_errors: int,
) -> dict[str, Any]:
    """Derive every reported float from the authoritative counts."""

    trials_completed = _require_nonnegative_int(trials_completed, "trials_completed")
    information_bits = _require_nonnegative_int(information_bits, "information_bits")
    bit_errors = _require_nonnegative_int(bit_errors, "bit_errors")
    block_errors = _require_nonnegative_int(block_errors, "block_errors")
    _require(block_errors <= trials_completed, "block_errors cannot exceed trials_completed")
    _require(bit_errors <= information_bits, "bit_errors cannot exceed information_bits")
    if trials_completed == 0:
        return {"ber": None, "bler": None, "bler_confidence_low": None, "bler_confidence_high": None}
    low, high = wilson_interval(block_errors, trials_completed)
    return {
        "ber": (bit_errors / information_bits) if information_bits else None,
        "bler": block_errors / trials_completed,
        "bler_confidence_low": low,
        "bler_confidence_high": high,
    }


# --------------------------------------------------------------------------
# Campaign and required-artifact bindings
# --------------------------------------------------------------------------


@functools.cache
def campaign_bindings() -> dict[str, str]:
    """Return the immutable G8_A bindings every request and result carries.

    Cached because the G8_A manifest is frozen for the whole campaign; a change
    to it invalidates the campaign rather than being picked up mid-run.
    """

    manifest = load_campaign_manifest(CAMPAIGN_MANIFEST)
    policy = manifest.get("selection_policy") or {}
    return {
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": sha256_file(CAMPAIGN_MANIFEST),
        "required_bler_artifact_sha256": sha256_file(REQUIRED_BLER_IDENTITIES),
        "selection_policy_sha256": policy["selection_policy_sha256"],
    }


@functools.cache
def required_work_unit_index() -> Mapping[str, Mapping[str, Any]]:
    """Return the exact required work units keyed by their frozen IDs.

    Cached and read-only: the required-identity artifact is a frozen G8_A
    output, and re-reading 8.6 MB per lookup would dominate the runner.
    """

    payload = load_required_bler_identities(REQUIRED_BLER_IDENTITIES)
    units = payload.get("required_bler_work_units")
    _require(isinstance(units, list) and units, "required-BLER artifact has no work units")
    index: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        unit_id = unit["work_unit_id"]
        _require(unit_id not in index, f"duplicate required work-unit ID {unit_id!r}")
        index[unit_id] = MappingProxyType(dict(unit))
    return MappingProxyType(index)


def required_work_unit(work_unit_id: str) -> Mapping[str, Any]:
    """Return exactly one required entry, or fail closed.

    There is deliberately no nearest match, no SNR rounding and no default.
    """

    index = required_work_unit_index()
    _require(
        work_unit_id in index,
        f"work unit {work_unit_id!r} is not an exact required BLER identity; "
        f"{NO_INTERPOLATION_RULE}",
    )
    return index[work_unit_id]


# --------------------------------------------------------------------------
# Work-unit request schema
# --------------------------------------------------------------------------


def _seed_block(campaign_id: str, work_unit_id: str) -> dict[str, Any]:
    return stream_seed_records(campaign_id, work_unit_id)


def build_full_strength_request(work_unit_id: str) -> dict[str, Any]:
    """Build the request for exactly one required BLER work unit."""

    unit = required_work_unit(work_unit_id)
    bindings = campaign_bindings()
    packet_ids = list(unit["source_packet_config_ids"])
    request = {
        "schema_version": BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "artifact_role": REQUEST_ARTIFACT_ROLE,
        "execution_class": EXECUTION_CLASS_FULL_STRENGTH,
        **bindings,
        "work_unit_id": work_unit_id,
        "bler_identity": dict(unit["identity"]),
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": packet_ids,
        "trials_requested": full_strength_trial_count(),
        "trial_count_source": FULL_STRENGTH_TRIAL_COUNT_SOURCE,
        "seed_derivation_identity": SEED_DERIVATION_IDENTITY,
        "seed_domain_separator": SEED_DOMAIN_SEPARATOR,
        "stream_seeds": _seed_block(bindings["campaign_id"], work_unit_id),
        "scientific_evidence": True,
        "merge_eligible": False,
        "test_split_access": TEST_SPLIT_ACCESS,
        "label": EXECUTION_CLASS_FULL_STRENGTH,
    }
    return validate_work_unit_request(request)


def build_bounded_smoke_request(
    *,
    work_unit_id: str,
    bler_identity: Mapping[str, Any],
    snr_db: Any,
    source_packet_config_ids: Sequence[str],
    trials_requested: int,
) -> dict[str, Any]:
    """Build a visibly non-scientific bounded-smoke request."""

    bindings = campaign_bindings()
    request = {
        "schema_version": BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "artifact_role": REQUEST_ARTIFACT_ROLE,
        "execution_class": EXECUTION_CLASS_BOUNDED_SMOKE,
        **bindings,
        "work_unit_id": work_unit_id,
        "bler_identity": dict(bler_identity),
        "snr_db": snr_db,
        "source_packet_config_ids": list(source_packet_config_ids),
        "trials_requested": trials_requested,
        "trial_count_source": BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
        "seed_derivation_identity": SEED_DERIVATION_IDENTITY,
        "seed_domain_separator": SEED_DOMAIN_SEPARATOR,
        "stream_seeds": _seed_block(bindings["campaign_id"], work_unit_id),
        "scientific_evidence": False,
        "merge_eligible": False,
        "test_split_access": TEST_SPLIT_ACCESS,
        "label": BOUNDED_SMOKE_LABEL,
    }
    return validate_work_unit_request(request)


def validate_work_unit_request(
    request: Any,
    *,
    execution_class: str | None = None,
) -> dict[str, Any]:
    """Strictly validate one work-unit request and return it unchanged."""

    _require(isinstance(request, Mapping), "work-unit request is not a mapping")
    missing = [name for name in REQUEST_FIELDS if name not in request]
    unexpected = [name for name in request if name not in REQUEST_FIELDS]
    _require(not missing, f"work-unit request is missing fields: {sorted(missing)}")
    _require(not unexpected, f"work-unit request has unknown fields: {sorted(unexpected)}")

    _require(
        request["schema_version"] == BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "unsupported work-unit request schema_version",
    )
    _require(request["artifact_role"] == REQUEST_ARTIFACT_ROLE, "wrong request artifact role")
    observed_class = request["execution_class"]
    _require(observed_class in EXECUTION_CLASSES, f"unknown execution class {observed_class!r}")
    if execution_class is not None:
        _require(
            observed_class == execution_class,
            f"request is {observed_class!r}, not {execution_class!r}",
        )

    bindings = campaign_bindings()
    for name, expected in bindings.items():
        _require(request[name] == expected, f"request binding {name} does not match the campaign")

    work_unit_id = _require_nonblank_str(request["work_unit_id"], "work_unit_id")
    identity = request["bler_identity"]
    _require(isinstance(identity, Mapping), "bler_identity is not a mapping")
    # Refuses a partial or over-specified physical identity.
    BlerIdentity.from_mapping(identity)
    _require(
        not isinstance(request["snr_db"], bool) and isinstance(request["snr_db"], (int, float)),
        "snr_db must be a real number",
    )
    _require(math.isfinite(float(request["snr_db"])), "snr_db must be finite")
    packet_ids = request["source_packet_config_ids"]
    _require(
        isinstance(packet_ids, list)
        and packet_ids
        and all(isinstance(item, str) and item for item in packet_ids),
        "source_packet_config_ids must be a non-empty list of non-blank strings",
    )
    _require(
        packet_ids == sorted(set(packet_ids)),
        "source_packet_config_ids must be unique and canonically ordered",
    )

    _require(
        request["seed_derivation_identity"] == SEED_DERIVATION_IDENTITY,
        "request seed-derivation identity is not the frozen identity",
    )
    _require(
        request["seed_domain_separator"] == SEED_DOMAIN_SEPARATOR,
        "request seed domain separator is not the frozen separator",
    )
    seeds = request["stream_seeds"]
    _require(isinstance(seeds, Mapping), "stream_seeds is not a mapping")
    _require(set(seeds) == set(SEED_PURPOSES), "stream_seeds must cover exactly the allowed purposes")
    for purpose in SEED_PURPOSES:
        _require(
            _identical(seeds[purpose], seed_record(bindings["campaign_id"], work_unit_id, purpose)),
            f"stream seed record for {purpose} does not reproduce from the frozen derivation",
        )
    distinct = {seeds[purpose]["seed_uint64"] for purpose in SEED_PURPOSES}
    _require(len(distinct) == len(SEED_PURPOSES), "random purposes must not share a stream seed")

    trials = _require_nonnegative_int(request["trials_requested"], "trials_requested")
    _require(trials > 0, "trials_requested must be positive")
    _require(request["merge_eligible"] is False, "a request is never merge eligible; only its result can be")
    _require(request["test_split_access"] == TEST_SPLIT_ACCESS, "request claims test-split access")

    full_count = full_strength_trial_count()
    if observed_class == EXECUTION_CLASS_FULL_STRENGTH:
        unit = required_work_unit(work_unit_id)
        _require(
            _identical(identity, unit["identity"]),
            "full-strength identity does not match the required entry exactly",
        )
        _require(
            _identical(request["snr_db"], unit["snr_db"]),
            "full-strength SNR does not match the required entry exactly",
        )
        _require(
            _identical(packet_ids, list(unit["source_packet_config_ids"])),
            "full-strength source packet IDs do not match the required entry exactly",
        )
        _require(
            request["trial_count_source"] == FULL_STRENGTH_TRIAL_COUNT_SOURCE,
            "full-strength trial count must come from params.baseline.bler_characterisation_trials",
        )
        _require(trials == full_count, "full-strength request must use exactly the configured trial count")
        _require(request["scientific_evidence"] is True, "full-strength request must be scientific evidence")
        _require(request["label"] == EXECUTION_CLASS_FULL_STRENGTH, "full-strength label changed")
    else:
        _require(
            request["trial_count_source"] == BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
            "bounded smoke must not claim the full-strength trial-count source",
        )
        _require(
            trials <= BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT,
            f"bounded smoke may not exceed {BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT} trials per unit",
        )
        _require(trials < full_count, "bounded smoke must stay below the full-strength trial count")
        _require(request["scientific_evidence"] is False, "bounded smoke is never scientific evidence")
        _require(request["label"] == BOUNDED_SMOKE_LABEL, "bounded smoke must be visibly labelled")
    return dict(request)


def require_full_strength_request(request: Any) -> dict[str, Any]:
    """Validate a request as full strength; bounded smoke can never pass."""

    return validate_work_unit_request(request, execution_class=EXECUTION_CLASS_FULL_STRENGTH)


def request_digest(request: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(request)))


# --------------------------------------------------------------------------
# Work-unit result schema
# --------------------------------------------------------------------------


def build_work_unit_result(
    *,
    request: Mapping[str, Any],
    status: str,
    trials_completed: int,
    bit_errors: int,
    block_errors: int,
    execution_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a result from authoritative counts; never a measurement."""

    request = validate_work_unit_request(request)
    trials_completed = _require_nonnegative_int(trials_completed, "trials_completed")
    information_length = _require_exact_int(
        request["bler_identity"]["k_and_n"][0], "information_length"
    )
    information_bits = trials_completed * information_length
    derived = recompute_measurements(
        trials_completed=trials_completed,
        information_bits=information_bits,
        bit_errors=bit_errors,
        block_errors=block_errors,
    )
    full_strength = request["execution_class"] == EXECUTION_CLASS_FULL_STRENGTH
    complete = (
        status == STATUS_COMPLETE
        and trials_completed > 0
        and (not full_strength or trials_completed == full_strength_trial_count())
    )
    metadata = dict(execution_metadata or {})
    result = {
        "schema_version": BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
        "artifact_role": RESULT_ARTIFACT_ROLE,
        "status": status,
        "identity": {
            "execution_class": request["execution_class"],
            "request_sha256": request_digest(request),
            "campaign_id": request["campaign_id"],
            "campaign_manifest_sha256": request["campaign_manifest_sha256"],
            "required_bler_artifact_sha256": request["required_bler_artifact_sha256"],
            "selection_policy_sha256": request["selection_policy_sha256"],
            "work_unit_id": request["work_unit_id"],
            "bler_identity": dict(request["bler_identity"]),
            "snr_db": request["snr_db"],
            "source_packet_config_ids": list(request["source_packet_config_ids"]),
            "trials_requested": request["trials_requested"],
            "trial_count_source": request["trial_count_source"],
            "seed_derivation_identity": request["seed_derivation_identity"],
            "seed_domain_separator": request["seed_domain_separator"],
            "stream_seeds": dict(request["stream_seeds"]),
            "implementation": implementation_binding(),
        },
        "measurement": {
            "trials_completed": trials_completed,
            "information_bits": information_bits,
            "bit_errors": _require_nonnegative_int(bit_errors, "bit_errors"),
            "block_errors": _require_nonnegative_int(block_errors, "block_errors"),
            **derived,
            "confidence_interval_method": CONFIDENCE_INTERVAL_METHOD,
            "confidence_interval_percent": CONFIDENCE_INTERVAL_PERCENT,
            "confidence_interval_role": CONFIDENCE_INTERVAL_ROLE,
        },
        "execution_metadata": {
            name: metadata.get(name) for name in RESULT_EXECUTION_METADATA_FIELDS
        },
        "disposition": {
            "scientific_evidence": bool(full_strength),
            "merge_eligible": bool(full_strength and complete),
            "test_split_access": TEST_SPLIT_ACCESS,
            "required_coverage_contribution": 1 if (full_strength and complete) else 0,
        },
    }
    return validate_work_unit_result(result, request=request)


def measurement_identity_digest(result: Mapping[str, Any]) -> str:
    """Digest the identity and measurement sections only.

    Wall time, hostname, device, shard and attempt are provenance; they can
    never move this digest.
    """

    return sha256_bytes(
        canonical_json({"identity": dict(result["identity"]), "measurement": dict(result["measurement"])})
    )


def validate_work_unit_result(
    result: Any,
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate one work-unit result; counts are authoritative."""

    _require(isinstance(result, Mapping), "work-unit result is not a mapping")
    _require(set(result) == set(RESULT_FIELDS), "work-unit result has missing or unknown sections")
    _require(
        result["schema_version"] == BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
        "unsupported work-unit result schema_version",
    )
    _require(result["artifact_role"] == RESULT_ARTIFACT_ROLE, "wrong result artifact role")
    status = result["status"]
    _require(status in RESULT_STATUSES, f"unknown result status {status!r}")

    identity = result["identity"]
    measurement = result["measurement"]
    metadata = result["execution_metadata"]
    disposition = result["disposition"]
    _require(isinstance(identity, Mapping) and set(identity) == set(RESULT_IDENTITY_FIELDS),
             "result identity section has missing or unknown fields")
    _require(isinstance(measurement, Mapping) and set(measurement) == set(RESULT_MEASUREMENT_FIELDS),
             "result measurement section has missing or unknown fields")
    _require(isinstance(metadata, Mapping) and set(metadata) == set(RESULT_EXECUTION_METADATA_FIELDS),
             "result execution metadata has missing or unknown fields")
    _require(isinstance(disposition, Mapping) and set(disposition) == set(RESULT_DISPOSITION_FIELDS),
             "result disposition section has missing or unknown fields")
    _require(isinstance(identity["implementation"], Mapping)
             and set(identity["implementation"]) == set(IMPLEMENTATION_FIELDS),
             "result implementation binding has missing or unknown fields")
    _require(_identical(identity["implementation"], implementation_binding()),
             "result implementation/dependency binding does not match the frozen contract")

    # The identity section must be exactly a validated request plus its digest.
    rebuilt = {
        "schema_version": BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "artifact_role": REQUEST_ARTIFACT_ROLE,
        "execution_class": identity["execution_class"],
        "campaign_id": identity["campaign_id"],
        "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": identity["required_bler_artifact_sha256"],
        "selection_policy_sha256": identity["selection_policy_sha256"],
        "work_unit_id": identity["work_unit_id"],
        "bler_identity": dict(identity["bler_identity"]),
        "snr_db": identity["snr_db"],
        "source_packet_config_ids": list(identity["source_packet_config_ids"]),
        "trials_requested": identity["trials_requested"],
        "trial_count_source": identity["trial_count_source"],
        "seed_derivation_identity": identity["seed_derivation_identity"],
        "seed_domain_separator": identity["seed_domain_separator"],
        "stream_seeds": dict(identity["stream_seeds"]),
        "scientific_evidence": identity["execution_class"] == EXECUTION_CLASS_FULL_STRENGTH,
        "merge_eligible": False,
        "test_split_access": TEST_SPLIT_ACCESS,
        "label": (
            EXECUTION_CLASS_FULL_STRENGTH
            if identity["execution_class"] == EXECUTION_CLASS_FULL_STRENGTH
            else BOUNDED_SMOKE_LABEL
        ),
    }
    validate_work_unit_request(rebuilt)
    _require(identity["request_sha256"] == request_digest(rebuilt),
             "result request digest does not reproduce from its own identity section")
    if request is not None:
        _require(identity["request_sha256"] == request_digest(validate_work_unit_request(request)),
                 "result does not bind the request it claims")
        for field in ("work_unit_id", "trials_requested", "execution_class", "trial_count_source"):
            _require(_identical(identity[field], request[field]),
                     f"result {field} does not match its request exactly")
        _require(_identical(identity["bler_identity"], request["bler_identity"]),
                 "result identity does not match its request exactly")
        _require(_identical(identity["snr_db"], request["snr_db"]),
                 "result SNR does not match its request exactly")

    full_strength = identity["execution_class"] == EXECUTION_CLASS_FULL_STRENGTH
    full_count = full_strength_trial_count()
    trials_requested = _require_nonnegative_int(identity["trials_requested"], "trials_requested")
    trials_completed = _require_nonnegative_int(measurement["trials_completed"], "trials_completed")
    information_bits = _require_nonnegative_int(measurement["information_bits"], "information_bits")
    bit_errors = _require_nonnegative_int(measurement["bit_errors"], "bit_errors")
    block_errors = _require_nonnegative_int(measurement["block_errors"], "block_errors")
    _require(trials_completed <= trials_requested, "trials_completed exceeds trials_requested")
    _require(block_errors <= trials_completed, "block_errors exceeds trials_completed")
    information_length = _require_exact_int(identity["bler_identity"]["k_and_n"][0], "K")
    _require(information_bits == trials_completed * information_length,
             "information_bits is not trials_completed x K")
    _require(bit_errors <= information_bits, "bit_errors exceeds information_bits")

    _require(measurement["confidence_interval_method"] == CONFIDENCE_INTERVAL_METHOD,
             "confidence interval method changed")
    _require(measurement["confidence_interval_percent"] == CONFIDENCE_INTERVAL_PERCENT,
             "confidence interval percent changed")
    _require(measurement["confidence_interval_role"] == CONFIDENCE_INTERVAL_ROLE,
             "confidence interval role changed; it is diagnostic only")

    derived = recompute_measurements(
        trials_completed=trials_completed,
        information_bits=information_bits,
        bit_errors=bit_errors,
        block_errors=block_errors,
    )
    for name, expected in derived.items():
        stored = measurement[name]
        if expected is None:
            _require(stored is None,
                     f"{name} must be null at zero completed trials, never zero")
        else:
            _require(stored is not None, f"{name} is missing")
            _require(stored == _require_finite(stored, name),
                     f"{name} must be finite; NaN and infinity are rejected")
            _require(stored == expected, f"stored {name} does not reproduce from the counts")

    if status == STATUS_COMPLETE:
        _require(trials_completed > 0, "completed evidence requires trials_completed > 0")
        if full_strength:
            _require(trials_completed == trials_requested == full_count,
                     "a completed full-strength result needs exactly the configured trial count")

    scientific = disposition["scientific_evidence"]
    merge_eligible = disposition["merge_eligible"]
    _require(isinstance(scientific, bool) and isinstance(merge_eligible, bool),
             "disposition flags must be booleans")
    _require(scientific is full_strength,
             "only full-strength execution is scientific evidence")
    _require(disposition["test_split_access"] == TEST_SPLIT_ACCESS, "result claims test-split access")
    contribution = _require_nonnegative_int(
        disposition["required_coverage_contribution"], "required_coverage_contribution"
    )
    expected_merge = (
        full_strength and status == STATUS_COMPLETE and trials_completed == full_count
    )
    _require(merge_eligible is expected_merge,
             "merge eligibility must follow exactly from status, class and counts")
    _require(contribution == (1 if expected_merge else 0),
             "required coverage contribution must follow merge eligibility")
    if not full_strength:
        _require(merge_eligible is False, "bounded smoke is never merge eligible")
        _require(contribution == 0, "bounded smoke never contributes required coverage")
    if status in (STATUS_INCOMPLETE, STATUS_FAILED):
        _require(merge_eligible is False, "an incomplete or failed result is never merge eligible")
    return dict(result)


# --------------------------------------------------------------------------
# Frozen fixed test vectors
# --------------------------------------------------------------------------

#: A fixture campaign/work-unit pair used only to pin the seed and stream
#: contract.  It is not a required G8 identity and never enters campaign state.
FIXTURE_CAMPAIGN_ID = "g8-fixture-campaign"
FIXTURE_WORK_UNIT_ID = "bler-fixture-unit"

_VECTOR_WORDS = 4  # literal-ok: width of the pinned raw-word vector, not a scientific parameter
_VECTOR_BITS = 8  # literal-ok: width of the pinned bit vector, not a scientific parameter
_VECTOR_BIT_OFFSET = 60  # literal-ok: pinned vector start, chosen to straddle the 64-bit word boundary
_VECTOR_TRIALS = BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT


def seed_test_vectors() -> dict[str, Any]:
    """Return the frozen seed and stream vectors bound by the contract."""

    vectors: dict[str, Any] = {
        "fixture_campaign_id": FIXTURE_CAMPAIGN_ID,
        "fixture_work_unit_id": FIXTURE_WORK_UNIT_ID,
        "seeds": {},
    }
    for purpose in SEED_PURPOSES:
        record = seed_record(FIXTURE_CAMPAIGN_ID, FIXTURE_WORK_UNIT_ID, purpose)
        seed = record["seed_uint64"]
        entry: dict[str, Any] = {
            "material_sha256": record["material_sha256"],
            "seed_uint64": seed,
            "first_raw_words": [int(word) for word in philox_words(seed, 0, _VECTOR_WORDS)],
        }
        if purpose == PURPOSE_INFORMATION_BITS:
            entry["bits_0_to_8"] = [
                int(bit) for bit in information_bit_stream(seed, 0, _VECTOR_BITS)
            ]
            entry["bits_60_to_68"] = [
                int(bit)
                for bit in information_bit_stream(seed, _VECTOR_BIT_OFFSET, _VECTOR_BITS)
            ]
        else:
            entry["first_normals"] = [float(value) for value in normal_stream(seed, _VECTOR_WORDS)]
        vectors["seeds"][purpose] = entry
    vectors["wilson"] = {
        "zero_errors_16_trials": list(wilson_interval(0, _VECTOR_TRIALS)),
        "one_error_16_trials": list(wilson_interval(1, _VECTOR_TRIALS)),
        "all_errors_16_trials": list(wilson_interval(_VECTOR_TRIALS, _VECTOR_TRIALS)),
    }
    return vectors
