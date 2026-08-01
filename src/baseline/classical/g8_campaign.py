"""Fail-closed contracts for the validation-only G-8 campaign.

G8_A freezes metadata and state machinery only.  This module deliberately has
no simulation, codec, dataset-decoding, classifier, training, selection, or
authorization entry point.
"""

from __future__ import annotations

import hashlib
import json
import csv
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from baseline.classical.composition import BlerIdentity, Candidate, g2_bler_table
from baseline.ldpc.transport import build_packet_plan
from config.params import REPO_ROOT, get

CAMPAIGN = "G-8"
CAMPAIGN_MANIFEST = REPO_ROOT / "results/baseline/g8/campaign_manifest.json"
REQUIRED_BLER_IDENTITIES = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
CAMPAIGN_STATE = REPO_ROOT / "results/baseline/g8/campaign_state.json"
PHASE_ORDER = tuple(f"G8_{letter}" for letter in "ABCDEFG")
PB3C_TERMINAL_SHA = "39c43e327573f33011c561c6de22bd05ff93c068"
SELECTION_POLICY_FIELDS = (
    "tie_break_order",
    "tie_equality",
    "fixed_modulation.source",
    "fixed_modulation.configured_value",
    "selection_passes",
    "selection_termination_pass",
)
PRE_DATA_FLAGS = {
    "campaign_started": False,
    "characterization_started": False,
    "validation_measurements_started": False,
    "pass_one_executed": False,
    "training_started": False,
    "pass_two_executed": False,
    "adjudication_complete": False,
    "test_split_access": 0,
    "authorization_issued": False,
}
STATE_STAGES = {
    "G8_A": ("contract_open", "preflight_complete"),
    "G8_B": ("tooling_open", "tooling_smoke_complete"),
    "G8_C": ("characterization_open", "characterization_complete"),
    "G8_D": ("measurement_tooling_open", "measurement_smoke_complete"),
    "G8_E": ("validation_measurement_open", "pass_one_complete"),
    "G8_F": ("training_open", "pass_two_complete"),
    "G8_G": ("adjudication_open", "adjudication_complete"),
}
COUNTER_FIELDS = (
    "validation_decoding",
    "inference",
    "training",
    "test_access",
)


class G8ContractError(RuntimeError):
    """The persisted campaign contract is missing, malformed, or has drifted."""


def canonical_json(value: Any) -> bytes:
    """Canonical identity bytes; presentation whitespace is never identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered_json(value: Any) -> bytes:
    """Stable tracked-file rendering."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def campaign_identifier(payload: Mapping[str, Any]) -> str:
    """Derive the stable ID from every manifest field except the ID itself."""

    basis = dict(payload)
    basis.pop("campaign_id", None)
    return f"g8-{sha256_bytes(canonical_json(basis))}"


def load_campaign_manifest(path: Path = CAMPAIGN_MANIFEST) -> dict[str, Any]:
    """Load and minimally type-check a G8_A manifest without trusting it."""

    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8ContractError(f"cannot read campaign manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise G8ContractError("campaign manifest is not a JSON object")
    if raw != rendered_json(payload):
        raise G8ContractError("campaign manifest is not canonical rendered JSON")
    if payload.get("schema_version") != 1:
        raise G8ContractError("unsupported campaign manifest schema_version")
    if payload.get("campaign") != CAMPAIGN:
        raise G8ContractError("campaign manifest names the wrong campaign")
    if payload.get("campaign_id") != campaign_identifier(payload):
        raise G8ContractError("campaign_id does not reproduce from manifest content")
    return payload


def canonical_rate(value: str) -> str:
    """Canonical rational spelling shared by required and legacy identities."""

    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise G8ContractError(f"invalid LDPC rate {value!r}") from exc
    if fraction <= 0:
        raise G8ContractError(f"LDPC rate is not positive: {value!r}")
    return f"{fraction.numerator}/{fraction.denominator}"


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{sha256_bytes(canonical_json(value))[:24]}"  # literal-ok: 96-bit display prefix


def _identity_record(
    *,
    information_length: int,
    codeword_length: int,
    base_graph: int,
    lifting_size: int,
    rate: str,
    modulation: str,
) -> dict[str, Any]:
    identity = BlerIdentity(
        k_and_n=(information_length, codeword_length),
        base_graph=base_graph,
        lifting_size=lifting_size,
        modulation=modulation,
        decoder_algorithm=str(get("baseline.ldpc_decoder")),
        decoder_offset=float(get("baseline.ldpc_decoder_offset")),
        iterations=int(get("baseline.ldpc_max_iters")),
        snr_convention="es_n0_per_symbol",
        rate=canonical_rate(rate),
    )
    return identity.as_key()


def _normalized_identity(identity: BlerIdentity) -> BlerIdentity:
    values = identity.as_key()
    values["rate"] = canonical_rate(identity.rate)
    return BlerIdentity.from_mapping(values)


def g2_measured_work_units() -> list[dict[str, Any]]:
    """Read only the already hash-checked G-2 CSV and expose exact points."""

    table = g2_bler_table()  # authenticates the CSV against G-2 adjudication first
    normalized = {_normalized_identity(identity) for identity in table.identities}
    rows: list[dict[str, Any]] = []
    path = REPO_ROOT / "results/baseline/g2/bler_results.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["system"] != "reference":
                continue
            for convention, column in (
                ("eb_n0_per_information_bit", "ebn0_db"),
                ("es_n0_per_symbol", "esn0_db"),
            ):
                identity = BlerIdentity(
                    k_and_n=(int(row["k"]), int(row["n"])),
                    base_graph=int(row["base_graph"]),
                    lifting_size=int(row["lifting_size"]),
                    modulation=row["modulation"],
                    decoder_algorithm=row["decoder"],
                    decoder_offset=float(row["offset"]),
                    iterations=int(row["iterations"]),
                    snr_convention=convention,
                    rate=canonical_rate(row["rate"]),
                )
                if identity not in normalized:
                    raise G8ContractError("G-2 CSV identity is absent from authenticated table")
                record = {"identity": identity.as_key(), "snr_db": float(row[column])}
                record["work_unit_id"] = _stable_id("g2", record)
                rows.append(record)
    return sorted(rows, key=lambda row: row["work_unit_id"])


def compare_required_to_g2(
    required_work_units: list[dict[str, Any]],
    g2_work_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify exact reuse versus every fail-closed missing-evidence case."""

    g2_units = g2_measured_work_units() if g2_work_units is None else g2_work_units
    required_keys = {
        canonical_json({"identity": row["identity"], "snr_db": row["snr_db"]}): row
        for row in required_work_units
    }
    g2_keys = {
        canonical_json({"identity": row["identity"], "snr_db": row["snr_db"]}): row
        for row in g2_units
    }
    if len(required_keys) != len(required_work_units):
        raise G8ContractError("required BLER work units contain a duplicate physical cell")
    if len(g2_keys) != len(g2_units):
        raise G8ContractError("G-2 measured work units contain a duplicate physical cell")
    g2_identities = {canonical_json(row["identity"]) for row in g2_units}
    exact_keys = set(required_keys) & set(g2_keys)
    identity_mismatch = [
        row["work_unit_id"]
        for key, row in required_keys.items()
        if key not in exact_keys and canonical_json(row["identity"]) not in g2_identities
    ]
    snr_support = [
        row["work_unit_id"]
        for key, row in required_keys.items()
        if key not in exact_keys and canonical_json(row["identity"]) in g2_identities
    ]
    return {
        "coverage_complete": bool(required_keys) and len(exact_keys) == len(required_keys),
        "complete_coverage_claim_permitted": bool(required_keys) and len(exact_keys) == len(required_keys),
        "already_characterized_exact": sorted(required_keys[key]["work_unit_id"] for key in exact_keys),
        "missing_required": sorted(
            row["work_unit_id"] for key, row in required_keys.items() if key not in exact_keys
        ),
        "g2_present_outside_required": sorted(g2_keys[key]["work_unit_id"] for key in set(g2_keys) - set(required_keys)),
        "uncharacterized_identity_mismatch": sorted(identity_mismatch),
        "uncharacterized_snr_support": sorted(snr_support),
        "interpolation_used": False,
        "extrapolation_used": False,
        "g2_evidence_reused_only_on_exact_identity_and_exact_snr": True,
    }


def build_structural_preflight() -> dict[str, Any]:
    """Enumerate the complete G-8 structure without inspecting scientific data."""

    datasets = tuple(
        name
        for name in ("imagenette160", "stl10")
        if get(f"datasets.{name}.role") in ("headline", "fallback_headline")
    )
    ratios = tuple(get("bandwidth.ratios"))
    modulations = tuple(get("baseline.modulations"))
    rates = tuple(canonical_rate(rate) for rate in get("baseline.ldpc_rates"))
    snr_grid = tuple(get("channel.test_snr_grid_db"))
    if not all(isinstance(snr, int) and not isinstance(snr, bool) for snr in snr_grid):
        raise G8ContractError("G-8 SNR grid must have exact integer-dB points")

    packet_records: list[dict[str, Any]] = []
    packet_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    required_sources: dict[bytes, set[str]] = {}
    required_records: dict[bytes, dict[str, Any]] = {}
    for dataset in datasets:
        symbols_by_ratio = get(f"bandwidth.k_symbols.{dataset}")
        if tuple(symbols_by_ratio) != ratios:
            raise G8ContractError(f"{dataset} symbol-budget ratios do not match bandwidth.ratios")
        for ratio in ratios:
            for modulation in modulations:
                for rate in rates:
                    packet = build_packet_plan(
                        int(symbols_by_ratio[ratio]), modulation, rate
                    )
                    if not packet.feasible or packet.segmentation is None:
                        raise G8ContractError(
                            "configured structural packet plan is infeasible: "
                            f"{dataset}/{ratio}/{modulation}/{rate}: {packet.reason}"
                        )
                    layout = packet.segmentation
                    basis = {
                        "dataset": dataset,
                        "dataset_role": get(f"datasets.{dataset}.role"),
                        "ratio": ratio,
                        "k_symbols": int(symbols_by_ratio[ratio]),
                        "modulation": modulation,
                        "ldpc_rate": rate,
                        "information_length": layout.k_prime,
                        "codeword_lengths": list(packet.e_r),
                        "base_graph": layout.base_graph,
                        "lifting_size": layout.lifting_size,
                        "code_blocks": layout.code_blocks,
                        "payload_bytes": packet.source_bytes,
                    }
                    packet_id = _stable_id("pkt", basis)
                    record = {"packet_config_id": packet_id, **basis}
                    packet_records.append(record)
                    packet_by_key[(dataset, ratio, modulation, rate)] = record
                    for codeword_length in sorted(set(packet.e_r)):
                        identity = _identity_record(
                            information_length=layout.k_prime,
                            codeword_length=codeword_length,
                            base_graph=layout.base_graph,
                            lifting_size=layout.lifting_size,
                            rate=rate,
                            modulation=modulation,
                        )
                        for snr_db in snr_grid:
                            work_basis = {"identity": identity, "snr_db": snr_db}
                            key = canonical_json(work_basis)
                            required_records[key] = work_basis
                            required_sources.setdefault(key, set()).add(packet_id)

    packet_records.sort(key=lambda row: row["packet_config_id"])
    candidates: list[dict[str, Any]] = []
    for dataset in datasets:
        codec = str(get("baseline.source_codec"))
        axes = tuple(get(f"baseline.downsample_axis_px.{dataset}"))
        for ratio in ratios:
            for encode_axis in axes:
                for modulation in modulations:
                    for rate in rates:
                        packet = packet_by_key[(dataset, ratio, modulation, rate)]
                        for snr_db in snr_grid:
                            candidate = Candidate(
                                dataset=dataset,
                                ratio=ratio,
                                modulation=modulation,
                                ldpc_rate=rate,
                                encode_axis_px=int(encode_axis),
                                snr_db=float(snr_db),
                            )
                            basis = {
                                "dataset": dataset,
                                "dataset_role": get(f"datasets.{dataset}.role"),
                                "source_codec": codec,
                                "ratio": ratio,
                                "encode_axis_px": int(encode_axis),
                                "modulation": modulation,
                                "ldpc_rate": rate,
                                "snr_db": snr_db,
                                "packet_config_id": packet["packet_config_id"],
                                "composition_candidate_identity": candidate.candidate_id,
                            }
                            candidates.append({"candidate_id": _stable_id("cand", basis), **basis})
    candidates.sort(key=lambda row: row["candidate_id"])

    work_units: list[dict[str, Any]] = []
    for key in sorted(required_records):
        basis = required_records[key]
        work_units.append(
            {
                "work_unit_id": _stable_id("bler", basis),
                **basis,
                "information_length": basis["identity"]["k_and_n"][0],
                "codeword_length": basis["identity"]["k_and_n"][1],
                "source_packet_config_ids": sorted(required_sources[key]),
            }
        )
    work_units.sort(key=lambda row: row["work_unit_id"])

    coverage = compare_required_to_g2(work_units)
    return {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "artifact_role": "preflight_required_bler_structure",
        "scientific_execution_performed": False,
        "grid_kind": "structural_not_codec_feasible",
        "fallback_invoked": False,
        "dataset_pixels_loaded": 0,
        "axes": {
            "datasets": [
                {"name": dataset, "role": get(f"datasets.{dataset}.role")} for dataset in datasets
            ],
            "ratios": list(ratios),
            "source_codecs": [str(get("baseline.source_codec"))],
            "encode_axis_px": {
                dataset: list(get(f"baseline.downsample_axis_px.{dataset}")) for dataset in datasets
            },
            "modulations": list(modulations),
            "ldpc_rates": list(rates),
            "snr_convention": "es_n0_per_symbol",
            "snr_grid_db": list(snr_grid),
            "decoder": {
                "algorithm": get("baseline.ldpc_decoder"),
                "offset": get("baseline.ldpc_decoder_offset"),
                "maximum_iterations": get("baseline.ldpc_max_iters"),
            },
        },
        "structural_candidates": candidates,
        "packet_configurations": packet_records,
        "required_bler_work_units": work_units,
        "g2_comparison": coverage,
        "counts": {
            "structural_candidates": len(candidates),
            "packet_configurations": len(packet_records),
            "required_unique_bler_work_units": len(work_units),
            "g2_exact_coverage": len(coverage["already_characterized_exact"]),
            "missing_required": len(coverage["missing_required"]),
            "g2_present_outside_required": len(coverage["g2_present_outside_required"]),
            "identity_mismatch": len(coverage["uncharacterized_identity_mismatch"]),
            "snr_support": len(coverage["uncharacterized_snr_support"]),
        },
    }


def load_required_bler_identities(
    path: Path = REQUIRED_BLER_IDENTITIES,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8ContractError(f"cannot read required-BLER artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or raw != rendered_json(payload):
        raise G8ContractError("required-BLER artifact is not canonical JSON object")
    if payload.get("schema_version") != 1 or payload.get("campaign") != CAMPAIGN:
        raise G8ContractError("required-BLER artifact schema or campaign is wrong")
    return payload


def _artifact_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "bytes": len(path.read_bytes()),
    }


def initial_campaign_state(*, stage: str = "contract_open") -> dict[str, Any]:
    """Construct G8_A state; never manufacture a scientific work-unit claim."""

    manifest = load_campaign_manifest()
    if stage not in STATE_STAGES["G8_A"]:
        raise G8ContractError(f"invalid initial G8_A stage {stage!r}")
    return {
        "schema_version": 1,
        "identity": {
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": sha256_file(CAMPAIGN_MANIFEST),
            "phase": "G8_A",
            "stage": stage,
            "completed_work_unit_ids": [],
            "in_progress_work_unit_id": None,
            "produced_artifacts": [
                _artifact_binding(CAMPAIGN_MANIFEST),
                _artifact_binding(REQUIRED_BLER_IDENTITIES),
            ],
            "restart_command": ".venv/bin/python tools/verify_g8_preflight.py",
            "seed_derivation_identity": "sha256(campaign_id,work_unit_id,purpose)-v1",
            "counters": {name: 0 for name in COUNTER_FIELDS},
        },
        "metadata": {"last_successful_checkpoint_time": None},
    }


def validate_campaign_state(
    payload: Any,
    *,
    manifest_path: Path = CAMPAIGN_MANIFEST,
) -> dict[str, Any]:
    """Strictly validate persisted state against the current campaign contract."""

    if not isinstance(payload, dict) or set(payload) != {"schema_version", "identity", "metadata"}:
        raise G8ContractError("campaign state has an invalid top-level schema")
    if payload["schema_version"] != 1:
        raise G8ContractError("unsupported campaign state schema_version")
    identity = payload["identity"]
    metadata = payload["metadata"]
    expected_identity_fields = {
        "campaign_id",
        "campaign_manifest_sha256",
        "phase",
        "stage",
        "completed_work_unit_ids",
        "in_progress_work_unit_id",
        "produced_artifacts",
        "restart_command",
        "seed_derivation_identity",
        "counters",
    }
    if not isinstance(identity, dict) or set(identity) != expected_identity_fields:
        raise G8ContractError("campaign state identity has missing or unexpected fields")
    if not isinstance(metadata, dict) or set(metadata) != {"last_successful_checkpoint_time"}:
        raise G8ContractError("campaign state metadata has missing or unexpected fields")

    manifest = load_campaign_manifest(manifest_path)
    if identity["campaign_id"] != manifest["campaign_id"]:
        raise G8ContractError("campaign state belongs to another campaign manifest")
    if identity["campaign_manifest_sha256"] != sha256_file(manifest_path):
        raise G8ContractError("campaign state manifest hash mismatch")
    phase = identity["phase"]
    stage = identity["stage"]
    if phase not in PHASE_ORDER or stage not in STATE_STAGES[phase]:
        raise G8ContractError("campaign state phase/stage is invalid")
    completed = identity["completed_work_unit_ids"]
    if (
        not isinstance(completed, list)
        or any(not isinstance(item, str) or not item for item in completed)
        or len(completed) != len(set(completed))
    ):
        raise G8ContractError("completed work-unit IDs are malformed or duplicated")
    if completed != sorted(completed):
        raise G8ContractError("completed work-unit IDs are not in canonical order")
    in_progress = identity["in_progress_work_unit_id"]
    if in_progress is not None and (not isinstance(in_progress, str) or not in_progress):
        raise G8ContractError("in-progress work-unit ID is malformed")
    if in_progress in completed:
        raise G8ContractError("a completed work unit is also marked in progress")
    if not isinstance(identity["restart_command"], str) or not identity["restart_command"]:
        raise G8ContractError("campaign restart command is missing")
    if not isinstance(identity["seed_derivation_identity"], str) or not identity["seed_derivation_identity"]:
        raise G8ContractError("campaign seed-derivation identity is missing")
    counters = identity["counters"]
    if not isinstance(counters, dict) or set(counters) != set(COUNTER_FIELDS):
        raise G8ContractError("campaign counters have the wrong schema")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters.values()):
        raise G8ContractError("campaign counters must be non-negative integers")
    artifacts = identity["produced_artifacts"]
    if not isinstance(artifacts, list):
        raise G8ContractError("produced artifacts are not a sequence")
    artifact_paths = [entry.get("path") for entry in artifacts if isinstance(entry, dict)]
    if len(artifact_paths) != len(artifacts) or artifact_paths != sorted(set(artifact_paths)):
        raise G8ContractError("produced artifact paths are malformed, duplicated, or unsorted")
    for entry in artifacts:
        if set(entry) != {"path", "sha256", "bytes"} or Path(entry["path"]).is_absolute():
            raise G8ContractError("produced artifact binding is malformed")
        target = REPO_ROOT / entry["path"]
        try:
            body = target.read_bytes()
        except OSError as exc:
            raise G8ContractError(f"cannot read produced artifact {entry['path']}: {exc}") from exc
        if entry["bytes"] != len(body) or entry["sha256"] != sha256_bytes(body):
            raise G8ContractError(f"produced artifact binding changed: {entry['path']}")
    if phase == "G8_A":
        if completed or in_progress is not None or any(counters.values()):
            raise G8ContractError("G8_A state exposes scientific work or nonzero counters")
    return payload


def load_campaign_state(path: Path = CAMPAIGN_STATE) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8ContractError(f"cannot read campaign state {path}: {exc}") from exc
    if raw != rendered_json(payload):
        raise G8ContractError("campaign state is partial, corrupt, or noncanonical JSON")
    return validate_campaign_state(payload)


def validate_state_transition(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse skipped/reversed stages, campaign swaps, and lost progress."""

    validate_campaign_state(previous)
    validate_campaign_state(current)
    old = previous["identity"]
    new = current["identity"]
    if old["campaign_id"] != new["campaign_id"]:
        raise G8ContractError("state transition changes campaigns")
    old_phase_index = PHASE_ORDER.index(old["phase"])
    new_phase_index = PHASE_ORDER.index(new["phase"])
    if new_phase_index == old_phase_index:
        old_stage_index = STATE_STAGES[old["phase"]].index(old["stage"])
        new_stage_index = STATE_STAGES[new["phase"]].index(new["stage"])
        if new_stage_index not in (old_stage_index, old_stage_index + 1):
            raise G8ContractError("state transition skips or reverses a stage")
    elif new_phase_index == old_phase_index + 1:
        if old["stage"] != STATE_STAGES[old["phase"]][-1] or new["stage"] != STATE_STAGES[new["phase"]][0]:
            raise G8ContractError("state transition exposes a future phase before its boundary")
    else:
        raise G8ContractError("state transition skips or reverses a phase")
    old_completed = set(old["completed_work_unit_ids"])
    new_completed = set(new["completed_work_unit_ids"])
    if not old_completed <= new_completed:
        raise G8ContractError("state transition loses completed work units")
    old_in_progress = old["in_progress_work_unit_id"]
    if old_in_progress is not None and old_in_progress not in new_completed and old_in_progress != new["in_progress_work_unit_id"]:
        raise G8ContractError("state transition abandons its in-progress work unit")
    for name in COUNTER_FIELDS:
        if new["counters"][name] < old["counters"][name]:
            raise G8ContractError(f"state transition reverses counter {name}")


def write_campaign_state_atomically(path: Path, payload: dict[str, Any]) -> str:
    """Flush a same-directory temporary, replace atomically, then sync directory."""

    validate_campaign_state(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = rendered_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(body)
