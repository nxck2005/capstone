"""W9-A/G-10 execution authority and evidence contracts.

AM-94's predicate lives in :mod:`evaluation.g10_crossover` and is deliberately
not reimplemented here.  This module binds the already-frozen inputs to one
validation-only campaign, extracts the frozen classical headline without
rerunning it, and verifies the additive pre-science boundary after W9 files
exist.

The module is intentionally free of Torch and dataset-loading code.  Importing
it therefore cannot itself perform a model-facing evaluation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from config.execution_profiles import profile_definition
from config.params import REPO_ROOT, get
from evaluation import g10_spec_compatibility as am94


SCHEMA_VERSION = 1
AUTHORIZATION_PATH = Path("results/learned/w9/g10_execution_authorization.json")
SOURCE_MANIFEST_PATH = Path("results/learned/w9/g10_source_manifest.json")
CLASSICAL_EXTRACT_PATH = Path(
    "results/learned/w9/g10_classical_adaptive_r1_6_extract.json"
)
RUNTIME_MANIFEST_PATH = Path("results/learned/w9/g10_runtime_manifest.json")
CELL_INDEX_PATH = Path("results/learned/w9/g10_cell_index.json")
HEADLINE_CURVE_PATH = Path("results/learned/w9/g10_headline_curve.json")
ADJUDICATION_PATH = Path("results/learned/w9/g10_adjudication.json")
COMPLETION_PATH = Path("results/learned/w9/w9a_completion.json")
RECONCILIATION_PATH = Path("results/learned/w9/w9a_reconciliation.json")

CLASSICAL_CLOSEOUT_PATH = Path("results/baseline/g8/g8_closeout.json")
CLASSICAL_PASS_TWO_PATH = Path("results/baseline/g8_f/pass_two_state.json")
CLASSICAL_F3_PATH = Path("results/baseline/g8_f/f3/f3_scoring_aggregate.json")
W8_COMPLETION_PATH = Path("results/learned/w8/w8_completion.json")
W8_RECONCILIATION_PATH = Path("results/learned/w8/w8_c_reconciliation.json")

EXPECTED_GRID = (
    -8,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -7,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -6,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -5,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -4,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -3,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -2,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    -1,
    0,
    1,
    2,
    3,
    4,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    5,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    6,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    7,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    9,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    11,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    13,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    15,  # literal-ok: AM-94 corrected frozen G-10 grid binding
    18,  # literal-ok: AM-94 corrected frozen G-10 grid binding
)
EXPECTED_PROFILE_ID = "confessor_pascal_cu126"
EXPECTED_GPU_UUID = "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a"
EXPECTED_GPU_NAME = "NVIDIA TITAN Xp"
EXPECTED_DEVICE = "cuda:0"
EXPECTED_LAMBDA = 3.0
EXPECTED_RATIO = "r_1_6"
EXPECTED_DATASET = "imagenette160"
EXPECTED_SPLIT = "val"
EXPECTED_DENOMINATOR = 1000  # literal-ok: committed Imagenette validation denominator
EXPECTED_CELL_COUNT = 63
AUTH_PREFIX = "g10auth-"
SOURCE_PREFIX = "g10source-"
CLASSICAL_PREFIX = "g10classical-"
RUNTIME_PREFIX = "g10runtime-"
CELL_PREFIX = "g10cell-"
CURVE_PREFIX = "g10curve-"
ADJUDICATION_PREFIX = "g10adjudication-"
COMPLETION_PREFIX = "w9acompletion-"
RECONCILIATION_PREFIX = "w9areconcile-"

PROTECTED_ZERO = {
    "g10_model_facing_evaluations": 0,
    "g10_outcomes_observed": 0,
    "er9_training": 0,
    "er2_randomized_training": 0,
    "g11": 0,
    "w10": 0,
    "learned_test_inference": 0,
    "model_facing_test_access": 0,
}

# These are the only files allowed to appear in results/learned/w9 before the
# 63-cell result matrix is complete.  The AM-94 file itself is historical and
# remains byte-identical.
PRE_EXECUTION_FILES = frozenset(
    {
        "results/learned/w9/am94_pre_science_freeze.json",
        str(AUTHORIZATION_PATH),
        str(SOURCE_MANIFEST_PATH),
        str(CLASSICAL_EXTRACT_PATH),
    }
)
OUTCOME_FILES = frozenset(
    {
        str(RUNTIME_MANIFEST_PATH),
        str(CELL_INDEX_PATH),
        str(HEADLINE_CURVE_PATH),
        str(ADJUDICATION_PATH),
        str(COMPLETION_PATH),
        str(RECONCILIATION_PATH),
    }
)

# All bytes that can affect a G-10 inference are named here.  The manifest is
# generated after the source-only commit and authenticates these exact bytes
# from that commit, so the later authority/result commits cannot silently
# become the scientific implementation epoch.
SOURCE_PATHS = (
    "src/evaluation/g10_crossover.py",
    "src/evaluation/g10_spec_compatibility.py",
    "src/evaluation/g10_protocol.py",
    "src/evaluation/g10_runner.py",
    "src/data/djscc_validation.py",
    "src/channels/awgn.py",
    "src/models/djscc.py",
    "src/config/params.py",
    "src/config/run_config.py",
    "src/config/execution_profiles.py",
    "configs/learned-w8-final.yaml",
    "spec/params.generated.yaml",
    "tools/freeze_g10_authority.py",
    "tools/run_g10_campaign.py",
    "tools/closeout_g10.py",
    "tools/verify_g10_w9.py",
    "tools/reconcile_g10.py",
)


class G10ProtocolHold(RuntimeError):
    """A W9-A/G-10 authority or evidence contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G10ProtocolHold(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def rendered_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G10ProtocolHold(f"cannot read {label}: {exc}") from None
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _identified(body: dict[str, Any], field: str, prefix: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = prefix + canonical_sha256(value)
    without_content = dict(value)
    value["artifact_content_sha256"] = canonical_sha256(without_content)
    return value


def verify_identified(
    value: dict[str, Any], *, field: str, prefix: str, label: str
) -> None:
    without_content = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    require(
        value.get("artifact_content_sha256") == canonical_sha256(without_content),
        f"{label} content identity differs",
    )
    without_id = {key: child for key, child in without_content.items() if key != field}
    require(
        value.get(field) == prefix + canonical_sha256(without_id),
        f"{label} identity differs",
    )


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, f"cannot resolve source bytes: {relative}")
    return result.stdout


def _read_current(path: str | Path, root: Path = REPO_ROOT) -> bytes:
    candidate = root / path
    require(candidate.is_file() and not candidate.is_symlink(), f"missing source/artifact: {path}")
    return candidate.read_bytes()


def _file_binding(path: str | Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    raw = _read_current(path, root)
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def expected_grid(root: Path = REPO_ROOT) -> tuple[int, ...]:
    value = tuple(get("channel.test_snr_grid_db"))
    require(value == EXPECTED_GRID, "params.channel.test_snr_grid_db differs from the corrected G-10 grid")
    require(len(value) == 21 and list(value) == sorted(value), "G-10 normative grid is not 21-point increasing")  # literal-ok: AM-94 fixes 21 measured points
    return value


def protocol_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    grid = list(expected_grid(root))
    body = {
        "dataset": EXPECTED_DATASET,
        "split": EXPECTED_SPLIT,
        "validation_denominator": EXPECTED_DENOMINATOR,
        "ratio": EXPECTED_RATIO,
        "lambda": EXPECTED_LAMBDA,
        "snr_parameter": "params.channel.test_snr_grid_db",
        "snr_grid_db": grid,
        "learned_cell_count": 3,
        "matrix_cell_count": EXPECTED_CELL_COUNT,
        "top1_rule": "torch.Tensor.argmax(dim=1)_first_index",
        "noise_rule": "validation_noise_id(stable_sample_id,dataset_version,split_manifest_hash,channel_seed,channel,ratio,k,snr_db)",
        "noise_rng_purpose": "channel_noise",
        "validation_order": "stable_manifest_order",
        "test": "SEALED",
    }
    body["protocol_sha256"] = canonical_sha256(body)
    return body


def _w8_selected_checkpoints(root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    completion, _ = load_json(root / W8_COMPLETION_PATH, "W8 terminal completion")
    reconciliation, _ = load_json(root / W8_RECONCILIATION_PATH, "W8-C reconciliation")
    require(completion.get("status") == "W8_GREEN_CLOSED", "W8 is not terminally green")
    require(completion.get("g10") == "NOT_EXECUTED" and completion.get("g10_count") == 0, "G-10 moved before authority")
    selected = {
        row["checkpoint_id"]: row
        for row in completion.get("selected_checkpoints", [])
        if isinstance(row, dict)
    }
    per_run = {
        row["run_id"]: row
        for row in reconciliation.get("selection", {}).get("per_run", [])
        if isinstance(row, dict)
    }
    noise = {
        row["run_id"]: row
        for row in reconciliation.get("validation_noise", {}).get("per_run", [])
        if isinstance(row, dict)
    }
    expected = (
        (0, 0, 92, "b0f72a3e16c537984b6afd3dc93bdf3ea87a0cae8a5b49f3565c803750a8826a", "run-01-r_1_6-train0-channel0", "w8-r_1_6-train0-channel0"),
        (1, 1, 77, "5d7e692c723ee9657be9fc45bfdae5dc54adf137ee9e92a04de2b2e5a7bbdcda", "run-03-r_1_6-train1-channel1", "w8-r_1_6-train1-channel1"),
        (2, 2, 78, "d5595f0931b010805f59ded64b3cda88b35730d200c5578aecf5962c8d058f41", "run-05-r_1_6-train2-channel2", "w8-r_1_6-train2-channel2"),
    )
    rows: list[dict[str, Any]] = []
    for train_seed, channel_seed, epoch, checkpoint_id, run_directory, run_id in expected:
        completion_row = selected.get(checkpoint_id)
        selection_row = per_run.get(run_id)
        noise_row = noise.get(run_id)
        require(completion_row is not None, f"W8 selected checkpoint is absent: {checkpoint_id}")
        require(selection_row is not None, f"W8 selection mapping is absent: {run_id}")
        require(noise_row is not None, f"W8 validation-noise mapping is absent: {run_id}")
        require(completion_row["epoch"] == epoch and selection_row["runner_published"]["epoch"] == epoch, f"W8 epoch binding differs: {run_id}")
        require(selection_row["runner_published"]["checkpoint_id"] == checkpoint_id, f"W8 checkpoint mapping differs: {run_id}")
        require(selection_row["independently_reconstructed"]["checkpoint_id"] == checkpoint_id, f"W8 independent checkpoint differs: {run_id}")
        require(selection_row["independently_reconstructed"]["n_total"] == EXPECTED_DENOMINATOR, f"W8 selection denominator differs: {run_id}")
        rows.append(
            {
                "train_seed": train_seed,
                "channel_seed": channel_seed,
                "run_id": run_id,
                "run_directory": run_directory,
                "ratio": EXPECTED_RATIO,
                "epoch": epoch,
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": checkpoint_id,
                "w8_result_id": completion_row["result_id"],
                "w8_result_file_sha256": completion_row["result_file_sha256"],
                "validation_selection_correct": selection_row["independently_reconstructed"]["n_correct"],
                "validation_selection_total": selection_row["independently_reconstructed"]["n_total"],
                "w8_validation_noise_id_digest": noise_row["digest"],
                "checkpoint_path": f"/home/nick/w8-final-pascal-20260901-r1/{run_directory}/checkpoints/epoch-{epoch:04d}.pt",
            }
        )
    return rows


def verify_am94_boundary(root: Path = REPO_ROOT, *, outcomes_allowed: bool = False) -> dict[str, Any]:
    """Re-authenticate AM-94 while admitting only additive W9 files.

    ``g10_spec_compatibility.load`` intentionally rejects every file after the
    pre-science freeze.  That historical behavior proves that AM-94 itself
    was pre-science.  W9 needs this additive verifier so the old freeze bytes
    remain immutable while the new authority/results are present.
    """

    value, raw = load_json(root / am94.FREEZE_RELATIVE_PATH, "AM-94 pre-science freeze")
    require(raw == am94.rendered(value), "AM-94 freeze is not canonical rendered JSON")
    require(sha256_bytes(raw) == am94.FREEZE_SHA256, "AM-94 freeze bytes differ")
    body = dict(value)
    freeze_id = body.pop("freeze_id", None)
    require(freeze_id == am94.FREEZE_ID and freeze_id == "g10semantics-" + am94.sha256_bytes(am94.canonical(body)), "AM-94 identity differs")
    require(value["predecessor_commit"] == am94.PREDECESSOR_COMMIT, "AM-94 predecessor differs")
    prior, prior_raw = load_json(root / am94.PRIOR_COMPATIBILITY_PATH, "AM-93 compatibility")
    require(sha256_bytes(prior_raw) == am94.PRIOR_COMPATIBILITY_SHA256 and prior.get("compatibility_id") == am94.PRIOR_COMPATIBILITY_ID, "AM-93 compatibility differs")
    require(value["prior_compatibility"] == {"path": am94.PRIOR_COMPATIBILITY_PATH, "compatibility_id": am94.PRIOR_COMPATIBILITY_ID, "sha256": am94.PRIOR_COMPATIBILITY_SHA256}, "AM-94 compatibility binding differs")
    require(value["entries"] == am94.expected_entries(), "AM-94 source-view entries differ")
    for relative, base_bytes, base_sha, current_bytes, current_sha in am94.VIEW_HASHES:
        predecessor = _git_bytes(root, am94.PREDECESSOR_COMMIT, relative)
        current = _read_current(relative, root)
        require(len(predecessor) == base_bytes and sha256_bytes(predecessor) == base_sha, f"AM-94 predecessor bytes differ: {relative}")
        require(len(current) == current_bytes and sha256_bytes(current) == current_sha, f"AM-94 current bytes differ: {relative}")
    require(get("channel.test_snr_grid_db") == list(EXPECTED_GRID), "normative G-10 grid moved")
    require(value["scientific_boundary"] == {
        "er2_randomized_training": 0,
        "er9_training": 0,
        "g10_learned_outcomes_observed": 0,
        "g10_model_facing_evaluations": 0,
        "g11": 0,
        "learned_test_inference": 0,
        "test_model_facing_access": 0,
        "test_split": "SEALED",
        "w10": 0,
    }, "AM-94 protected counters differ")
    completion, _ = load_json(root / W8_COMPLETION_PATH, "W8 terminal completion")
    reconciliation, _ = load_json(root / W8_RECONCILIATION_PATH, "W8-C reconciliation")
    evidence = value["w8_terminal_evidence"]
    require(evidence["completion_id"] == completion.get("completion_id") and evidence["completion_sha256"] == sha256_file(root / W8_COMPLETION_PATH), "AM-94 W8 completion binding differs")
    require(evidence["reconciliation_id"] == reconciliation.get("reconciliation_id") and evidence["reconciliation_sha256"] == sha256_file(root / W8_RECONCILIATION_PATH), "AM-94 W8-C binding differs")
    require(completion.get("g10") == "NOT_EXECUTED" and completion.get("g10_count") == 0 and completion.get("test") == "SEALED" and completion.get("test_model_facing_access") == 0, "W8 protected counters moved")
    require(reconciliation.get("protected_boundaries", {}).get("g10") == 0 and reconciliation.get("protected_boundaries", {}).get("test_model_facing_access") == 0, "W8-C protected counters moved")
    require(not (root / "results/freeze_manifest.json").exists(), "test freeze manifest exists before G-12")
    actual = frozenset(path.relative_to(root).as_posix() for path in (root / "results/learned/w9").glob("**/*") if path.is_file())
    allowed = set(PRE_EXECUTION_FILES) if not outcomes_allowed else set(PRE_EXECUTION_FILES) | set(OUTCOME_FILES)
    require(actual <= allowed, f"unexpected W9/G-10 artifact exists: {sorted(actual - allowed)}")
    require(not (root / "results/learned/g10").exists(), "legacy G-10 outcome directory exists")
    require(get("evaluation.test_access_gate") == "G-12", "test access gate moved")
    return value


def build_source_manifest(root: Path, source_commit: str) -> dict[str, Any]:
    require(_git_head(root) == source_commit, "source manifest must be created at the clean source epoch")
    entries = []
    for relative in SOURCE_PATHS:
        raw = _git_bytes(root, source_commit, relative)
        entries.append({"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "G10_SCIENTIFIC_SOURCE_MANIFEST",
        "status": "FROZEN_BEFORE_FIRST_LEARNED_OUTCOME",
        "source_commit": source_commit,
        "entry_count": len(entries),
        "entries": entries,
        "scientific_source_scope": "G-10 validation-only execution machinery and AM-94 predicate consumer; no learned result bytes",
    }
    return _identified(body, "manifest_id", SOURCE_PREFIX)


def verify_source_manifest(value: dict[str, Any], root: Path = REPO_ROOT) -> None:
    verify_identified(value, field="manifest_id", prefix=SOURCE_PREFIX, label="G-10 source manifest")
    require(value.get("schema_version") == SCHEMA_VERSION and value.get("status") == "FROZEN_BEFORE_FIRST_LEARNED_OUTCOME", "G-10 source manifest header differs")
    entries = value.get("entries")
    require(isinstance(entries, list) and value.get("entry_count") == len(SOURCE_PATHS) == len(entries), "G-10 source manifest entry count differs")
    source_commit = str(value.get("source_commit"))
    require(len(source_commit) == 40, "G-10 source commit is not a full SHA")  # literal-ok: Git full object ID width
    for expected_path, entry in zip(SOURCE_PATHS, entries, strict=True):
        require(entry.get("path") == expected_path, f"G-10 source manifest order differs: {expected_path}")
        historical = _git_bytes(root, source_commit, expected_path)
        require(entry.get("bytes") == len(historical) and entry.get("sha256") == sha256_bytes(historical), f"G-10 historical source differs: {expected_path}")


def build_classical_extract(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Make a lossless extract of the already-frozen adaptive headline.

    This only reads G8/F3 records.  It never calls the classical runner,
    codec, LDPC decoder, selection machinery, or interpolation code.
    """

    closeout, _ = load_json(root / CLASSICAL_CLOSEOUT_PATH, "G8 closeout")
    pass_two, _ = load_json(root / CLASSICAL_PASS_TWO_PATH, "G8/F3 pass-two completion")
    f3, _ = load_json(root / CLASSICAL_F3_PATH, "F3 scoring aggregate")
    selected = [
        row
        for row in closeout.get("selected_operating_points", [])
        if row.get("mode") == "classical_adaptive" and row.get("ratio") == EXPECTED_RATIO
    ]
    require(len(selected) == len(EXPECTED_GRID), "frozen classical adaptive r_1_6 does not have 21 points")  # literal-ok: AM-94 fixes 21 measured points
    require(tuple(int(row["snr_db"]) for row in selected) == EXPECTED_GRID, "classical adaptive r_1_6 grid is not exact and ordered")
    calls = [
        row
        for row in pass_two.get("calls", [])
        if row.get("mode") == "classical_adaptive" and row.get("ratio") == EXPECTED_RATIO
    ]
    require(len(calls) == 1, "pass-two has no unique adaptive r_1_6 call")
    per_snr = calls[0].get("per_snr", [])
    require(len(per_snr) == len(EXPECTED_GRID), "pass-two adaptive r_1_6 coverage differs")  # literal-ok: AM-94 fixes 21 measured points
    by_snr = {int(row["snr_db"]): row for row in per_snr}
    f3_by_id = {row["measurement_identity_id"]: row for row in f3.get("objects", [])}
    rows: list[dict[str, Any]] = []
    for point in selected:
        snr = int(point["snr_db"])
        selected_row = by_snr.get(snr)
        require(selected_row is not None, f"pass-two adaptive point is absent at {snr} dB")
        composition = selected_row.get("selected_composition")
        require(isinstance(composition, dict), f"classical composition is absent at {snr} dB")
        identity = point.get("measurement_identity_id")
        f3_row = f3_by_id.get(identity)
        require(f3_row is not None, f"F3 exact count is absent for {identity}")
        require(selected_row.get("authority_candidate_id") == point.get("candidate_id"), f"classical selected candidate differs at {snr} dB")
        require(composition.get("codec_accuracy", {}).get("correct") == f3_row.get("correct_count") and composition.get("codec_accuracy", {}).get("total") == f3_row.get("total_count"), f"F3 count binding differs at {snr} dB")
        require(composition.get("outage_accuracy", {}).get("numerator") == 100 and composition.get("outage_accuracy", {}).get("denominator") == EXPECTED_DENOMINATOR, f"outage count differs at {snr} dB")  # literal-ok: frozen measured constant-class outage record
        probability_decimal = str(composition["success_probability"])
        success = Fraction(probability_decimal)
        clean = Fraction(int(f3_row["correct_count"]), int(f3_row["total_count"]))
        outage = Fraction(int(composition["outage_accuracy"]["numerator"]), int(composition["outage_accuracy"]["denominator"]))
        comparator = success * clean + (1 - success) * outage
        # The historical closeout stores this derived quantity as an IEEE
        # float (at -2 dB its printed value is one ulp below the exact
        # composition).  G-10 never consumes that lossy field: it consumes
        # the exact measured counts plus the losslessly represented decimal
        # success probability below and reconstructs this Fraction.
        rows.append(
            {
                "snr_db": snr,
                "mode": "classical_adaptive",
                "ratio": EXPECTED_RATIO,
                "candidate_id": point["candidate_id"],
                "measurement_identity_id": identity,
                "modulation": point["modulation"],
                "ldpc_rate": point["ldpc_rate"],
                "encode_axis_px": point["encode_axis_px"],
                "success_probability_decimal": probability_decimal,
                "clean_correct_count": int(f3_row["correct_count"]),
                "clean_denominator": int(f3_row["total_count"]),
                "outage_correct_count": int(composition["outage_accuracy"]["numerator"]),
                "outage_denominator": int(composition["outage_accuracy"]["denominator"]),
                "comparator_correct_count": comparator.numerator,
                "comparator_denominator": comparator.denominator,
                "comparator_fraction": f"{comparator.numerator}/{comparator.denominator}",
                "stored_expected_accuracy_decimal": str(point["expected_accuracy"]),
                "stored_expected_accuracy_is_predicate_input": False,
                "f3_unit_id": f3_row["unit_id"],
                "f3_unit_file_sha256": f3_row["unit_file_sha256"],
                "f3_ordered_scoring_sha256": f3_row["ordered_scoring_sha256"],
                "f3_scoring_set_sha256": f3_row["scoring_set_sha256"],
            }
        )
    source = {
        "g8_closeout": _file_binding(CLASSICAL_CLOSEOUT_PATH, root),
        "g8_pass_two": _file_binding(CLASSICAL_PASS_TWO_PATH, root),
        "g8_f3_aggregate": _file_binding(CLASSICAL_F3_PATH, root),
        "headline_role": "G8/F3 adaptive/oracle classical r_1_6 validation curve",
        "fixed_profile_role": "secondary_context_only_excluded_from_g10_predicate",
        "interpolation": False,
        "classical_phy_rerun": False,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "G10_FROZEN_CLASSICAL_ADAPTIVE_R1_6_EXTRACT",
        "status": "FROZEN_REFERENCE_NO_CLASSICAL_RERUN",
        "dataset": EXPECTED_DATASET,
        "split": EXPECTED_SPLIT,
        "ratio": EXPECTED_RATIO,
        "snr_parameter": "params.channel.test_snr_grid_db",
        "snr_grid_db": list(EXPECTED_GRID),
            "point_count": len(rows),
        "denominator_inputs": EXPECTED_DENOMINATOR,
        "source": source,
        "points": rows,
    }
    return _identified(body, "extract_id", CLASSICAL_PREFIX)


def verify_classical_extract(value: dict[str, Any], root: Path = REPO_ROOT) -> None:
    verify_identified(value, field="extract_id", prefix=CLASSICAL_PREFIX, label="G-10 classical extract")
    require(value.get("snr_grid_db") == list(EXPECTED_GRID) and value.get("point_count") == 21 and value.get("interpolation", False) is False, "classical extract scope differs")  # literal-ok: AM-94 fixes 21 measured points
    rebuilt = build_classical_extract(root)
    require(value == rebuilt, "G-10 classical extract does not reproduce from frozen G8/F3 bytes")


def build_authorization(
    *, root: Path = REPO_ROOT, source_commit: str, profile_id: str = EXPECTED_PROFILE_ID
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(_git_head(root) == source_commit, "authority must bind the clean source epoch HEAD")
    verify_am94_boundary(root)
    expected_grid(root)
    source_manifest = build_source_manifest(root, source_commit)
    classical = build_classical_extract(root)
    profile = profile_definition(profile_id)
    require(profile_id == EXPECTED_PROFILE_ID and profile["role"] == "eligible_production_execution_profile", "G-10 profile is not eligible")
    require(EXPECTED_GPU_UUID in profile["allowed_gpu_uuids"] and EXPECTED_GPU_NAME in profile["allowed_gpu_names"], "G-10 GPU binding is not registered")
    protocol = protocol_identity(root)
    completion, completion_raw = load_json(root / W8_COMPLETION_PATH, "W8 completion")
    reconciliation, reconciliation_raw = load_json(root / W8_RECONCILIATION_PATH, "W8-C reconciliation")
    checkpoint_rows = _w8_selected_checkpoints(root)
    manifest_binding = {"path": str(SOURCE_MANIFEST_PATH), "manifest_id": source_manifest["manifest_id"], "file_sha256": sha256_bytes(rendered_json(source_manifest))}
    classical_binding = {"path": str(CLASSICAL_EXTRACT_PATH), "extract_id": classical["extract_id"], "file_sha256": sha256_bytes(rendered_json(classical))}
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W9A_G10_VALIDATION_ONLY_EXECUTION_AUTHORIZATION",
        "status": "AUTHORIZED_PRE_EXECUTION",
        "authorization_scope": "G10_EXACTLY_THREE_W8_R1_6_CHECKPOINTS_X_NORMATIVE_21_SNR_GRID",
        "authorized_by": "repository owner instruction for W9-A separate G-10 authority",
        "amendment": "AM-94",
        "scientific_source": {"commit": source_commit, "manifest": manifest_binding},
        "protocol": protocol,
        "scientific_scope": {
            "dataset": EXPECTED_DATASET,
            "split": EXPECTED_SPLIT,
            "validation_denominator_per_cell": EXPECTED_DENOMINATOR,
            "ratio": EXPECTED_RATIO,
            "learned_lambda": EXPECTED_LAMBDA,
            "frozen_w8_final_headline_seed_cells": 3,
            "snr_count": 21,  # literal-ok: AM-94 fixes 21 measured points
            "complete_learned_evaluations": EXPECTED_CELL_COUNT,
            "test": "SEALED",
        },
        "checkpoints": checkpoint_rows,
        "checkpoint_selection": {
            "selection_performed_by_g10": False,
            "best_seed_selection": False,
            "weight_averaging": False,
            "w7_pilot_weights": False,
            "r_1_24": False,
            "w8_completion_id": completion["completion_id"],
            "w8_completion_sha256": sha256_bytes(completion_raw),
            "w8_reconciliation_id": reconciliation["reconciliation_id"],
            "w8_reconciliation_sha256": sha256_bytes(reconciliation_raw),
        },
        "snr_authority": {
            "parameter": "params.channel.test_snr_grid_db",
            "resolved_ordered_values_db": list(EXPECTED_GRID),
            "classical_one_to_one_coverage": True,
            "interpolation": False,
            "grid_created_by_g10": False,
        },
        "classical_headline": {
            **classical_binding,
            "comparator": "G8/F3 adaptive/oracle classical r_1_6 validation curve only",
            "fixed_profile_curve": "secondary_context_only_not_in_predicate",
            "point_count": 21,  # literal-ok: AM-94 fixes 21 measured points
            "exact_comparator_count_denominator_values": [
                {"snr_db": row["snr_db"], "correct_count": row["comparator_correct_count"], "denominator": row["comparator_denominator"]}
                for row in classical["points"]
            ],
        },
        "validation_noise": {
            "policy": "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs",
            "identity_function": "data.djscc_validation.validation_noise_id",
            "rng_purpose": "channel_noise",
            "ambient_sequential_rng": False,
            "channel_seed_pairing": "w8_zipped_train_seed_equals_channel_seed",
            "schedule_audit": "ordered stable validation IDs plus per-cell noise_id_digest and row digest",
        },
        "execution_profile_selection": {
            "execution_profile_id": profile_id,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "gpu_name": EXPECTED_GPU_NAME,
            "device": EXPECTED_DEVICE,
            "cuda_visible_devices": EXPECTED_GPU_UUID,
            "lock_file": profile["lock_file"],
            "lock_file_sha256": profile["lock_file_sha256"],
            "writer_host": profile["scientific_writer_host"],
            "selection_time": "before_first_scientific_measurement",
            "sole_writer": True,
        },
        "am94_semantics": {
            "freeze_path": am94.FREEZE_RELATIVE_PATH,
            "freeze_id": am94.FREEZE_ID,
            "freeze_sha256": am94.FREEZE_SHA256,
            "predicate_source_path": "src/evaluation/g10_crossover.py",
            "predicate_source_sha256": sha256_file(root / "src/evaluation/g10_crossover.py"),
            "learned_aggregate": "arithmetic_mean_of_three_exact_correct_count_fractions",
            "gap": "learned_minus_classical_adaptive",
            "sign": "exact_fraction_sign",
            "expected_direction": "positive_to_negative",
            "zero_run": "maximal_zero_run_bracketed_by_positive_then_negative",
            "location": "measured_bracket_or_exact_measured_point_or_measured_zero_plateau",
            "headline_event": "first_expected_direction_event",
            "retain_all_events": True,
            "population_sd_ddof": 0,
            "sd_descriptive_only": True,
            "tolerance_epsilon_ci_bootstrap_seed_vote": False,
        },
        "runtime": {
            "runtime_root": "/home/nick/w9-g10-am94-confessor-pascal-20260906",
            "order": "train_seed_ascending_then_snr_grid_order",
            "successful_cell_rerun": False,
            "incomplete_cell_recovery": "fail_closed_if_started_without_success_record",
        },
        "pre_execution_counters": PROTECTED_ZERO,
        "forbidden_scope": [
            "learned_r_1_24",
            "w7_pilot_checkpoints",
            "other_lambda",
            "learned_training",
            "er9",
            "randomized_er2_training",
            "g11",
            "w10",
            "test_inference",
            "test_access",
            "classical_rerun",
            "classical_interpolation",
            "extra_snr",
            "posthoc_tolerance_or_significance_rule",
        ],
        "source_contains_no_learned_g10_results": True,
    }
    return _identified(body, "authorization_id", AUTH_PREFIX), source_manifest, classical


def verify_authorization(
    path: Path = REPO_ROOT / AUTHORIZATION_PATH,
    *,
    root: Path = REPO_ROOT,
    allow_outcomes: bool = False,
) -> dict[str, Any]:
    value, raw = load_json(path, "G-10 execution authorization")
    require(raw == rendered_json(value), "G-10 authorization is not canonical JSON")
    verify_identified(value, field="authorization_id", prefix=AUTH_PREFIX, label="G-10 authorization")
    require(value.get("status") == "AUTHORIZED_PRE_EXECUTION", "G-10 authorization is not pre-execution")
    require(value.get("pre_execution_counters") == PROTECTED_ZERO, "G-10 authority counters are not zero")
    verify_am94_boundary(root, outcomes_allowed=allow_outcomes)
    source_manifest, _ = load_json(root / SOURCE_MANIFEST_PATH, "G-10 source manifest")
    verify_source_manifest(source_manifest, root)
    require(value["scientific_source"]["commit"] == source_manifest["source_commit"], "authority/source manifest commit differs")
    require(value["scientific_source"]["manifest"]["manifest_id"] == source_manifest["manifest_id"], "authority/source manifest ID differs")
    require(value["scientific_source"]["manifest"]["file_sha256"] == sha256_file(root / SOURCE_MANIFEST_PATH), "authority/source manifest SHA differs")
    classical, _ = load_json(root / CLASSICAL_EXTRACT_PATH, "G-10 classical extract")
    verify_classical_extract(classical, root)
    require(value["classical_headline"]["extract_id"] == classical["extract_id"] and value["classical_headline"]["file_sha256"] == sha256_file(root / CLASSICAL_EXTRACT_PATH), "authority/classical extract binding differs")
    require(value["protocol"] == protocol_identity(root), "G-10 protocol identity differs")
    require(value["snr_authority"]["resolved_ordered_values_db"] == list(EXPECTED_GRID), "authority grid differs")
    require(value["execution_profile_selection"]["execution_profile_id"] == EXPECTED_PROFILE_ID and value["execution_profile_selection"]["gpu_uuid"] == EXPECTED_GPU_UUID, "authority profile differs")
    expected_checkpoints = _w8_selected_checkpoints(root)
    require(value["checkpoints"] == expected_checkpoints, "authority W8 checkpoint mappings differ")
    require(value["am94_semantics"]["freeze_id"] == am94.FREEZE_ID and value["am94_semantics"]["freeze_sha256"] == am94.FREEZE_SHA256, "authority AM-94 binding differs")
    require(value["am94_semantics"]["predicate_source_sha256"] == sha256_file(root / "src/evaluation/g10_crossover.py"), "authority predicate source differs")
    return value


def cell_key(train_seed: int, channel_seed: int, snr_db: int | float) -> str:
    return f"train{int(train_seed)}_channel{int(channel_seed)}_snr{int(snr_db):+03d}"


def expected_cell_keys() -> tuple[str, ...]:
    return tuple(cell_key(seed, seed, snr) for seed in range(3) for snr in EXPECTED_GRID)


def expected_cell_count() -> int:
    return len(expected_cell_keys())
