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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from baseline.ldpc.modulation import bits_per_symbol
from config.params import REPO_ROOT, get
from evaluation.w10_scope import HEADLINE_RATIO, W10_DATASET, W10_VALIDATION_DENOMINATOR, snr_grid
from runtime.source_epochs import load_w10_manifest, source_record
from training.deterministic_core import canonical_sha256

JPEG_SELECTION_PATH = "results/learned/w10/jpeg_validation_selection.json"
ER12_SELECTION_PATH = "results/learned/w10/er12_validation_selection.json"
JPEG_SELECTION_PREFIX = "w10jpegselection-"
ER12_SELECTION_PREFIX = "w10er12selection-"
JPEG_SELECTION_ROLE = "W10_JPEG_SECONDARY_VALIDATION_SELECTION"
ER12_SELECTION_ROLE = "W10_ER12_VALIDATION_SELECTION"
SELECTION_STATUS = "FROZEN_VALIDATION_ONLY_SELECTION"
SELECTION_SCHEMA_VERSION = 1
PASS_TWO_STATE_PATH = "results/baseline/g8_f/pass_two_state.json"
CANDIDATE_AUTHORITY_PATH = "results/baseline/g8_e/candidate_authority.json"
ER9_CLOSEOUT_PATH = "results/learned/er9/er9_production_closeout_v4.json"
JPEG_PHY_SOURCE = "pass_two_state:classical_adaptive:r_1_6"
ER12_PHY_SOURCE = "er9_production_closeout:train0_channel0"
JPEG_SCORER = "br12_artifact_finetuned_reference_classifier"
ER12_SCORER = "transmitted_predicted_label_then_true_label_scoring_only"
OUTAGE_POLICY = "results/baseline/w4/outage_policy.json"
NOISE_CONVENTION = "w10_scheduled_noise_id_per_stable_image_snr_k_shared_across_candidates"
ER12_PROTOCOL_VERSION_PARAMETER = "params.evaluation.w10_er12_protocol_version"

JPEG_TIE_BREAK = (
    "n_correct_descending",
    "n_delivered_descending",
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
        -_n_correct(entry),
        -int(entry["n_delivered"]),
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
    ordered = sorted(scores, key=key)
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
    }
    if kind == "jpeg_secondary":
        required |= {"encode_axis_px", "quality"}
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
        _require(
            {_candidate_id(entry) for entry in candidates} == expected_candidates[str(int(snr))],
            "W10 candidate score point does not cover the exact frozen candidate set",
        )
        winner = rank_candidates(kind, candidates)
        stored = dict(selections[grid.index(snr)])
        _require(stored == winner, "W10 selection winner is not the recomputed frozen winner")
    return dict(value)


def jpeg_phy_shortlist(root: Path) -> dict[int, dict[str, Any]]:
    """The exact frozen pass-two adaptive PHY at each SNR (DEC-9 codec contrast)."""

    from evaluation.w10_bindings import _classical_selection  # noqa: PLC0415

    binding = _classical_selection(root, "classical_adaptive", HEADLINE_RATIO)
    table = _read_json(Path(root) / CANDIDATE_AUTHORITY_PATH, "G8 candidate authority")
    candidates = {str(item["candidate_id"]): item for item in table["candidates"]}
    shortlist: dict[int, dict[str, Any]] = {}
    for snr, item in zip(snr_grid(), binding["selections"], strict=True):
        _require(int(float(item["snr_db"])) == int(snr), "pass-two selection SNR order differs")
        candidate = candidates[str(item["authority_candidate_id"])]
        shortlist[int(snr)] = {
            "authority_candidate_id": str(item["authority_candidate_id"]),
            "modulation": str(candidate["modulation"]),
            "ldpc_rate": str(candidate["ldpc_rate"]),
            "encode_axis_px": int(candidate["encode_axis_px"]),
        }
    return shortlist


def _expected_candidate_set(kind: str, root: Path) -> dict[str, set[str]]:
    """The exact prospective candidate space at each SNR, from its frozen source."""

    grid = list(snr_grid())
    if kind == "jpeg_secondary":
        shortlist = jpeg_phy_shortlist(root)
        qualities = tuple(int(value) for value in get("baseline.jpeg_quality_grid"))
        expected: dict[str, set[str]] = {}
        for snr in grid:
            phy = shortlist[int(snr)]
            expected[str(int(snr))] = {
                _candidate_id(
                    {
                        "snr_db": int(snr),
                        "modulation": phy["modulation"],
                        "ldpc_rate": phy["ldpc_rate"],
                        "encode_axis_px": phy["encode_axis_px"],
                        "quality": quality,
                    }
                )
                for quality in qualities
            }
        return expected
    from evaluation.er9_search import configured_phy_candidates  # noqa: PLC0415

    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))
    pairs = tuple((str(modulation), str(rate)) for modulation, rate, _packet in configured_phy_candidates(k))
    return {
        str(int(snr)): {
            _candidate_id({"snr_db": int(snr), "modulation": modulation, "ldpc_rate": rate})
            for modulation, rate in pairs
        }
        for snr in grid
    }


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
    "er12_contract_sha256",
    "er12_rank_key",
    "er12_selection_contract",
    "jpeg_contract_sha256",
    "jpeg_phy_shortlist",
    "jpeg_rank_key",
    "jpeg_selection_contract",
    "rank_candidates",
    "score_vector_digest",
    "selection_identity",
    "validate_selected_point",
    "verify_selection_artifact",
]
