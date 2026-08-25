"""Validation-only G8_F/F3 rescoring of the immutable G8_E reconstruction cache.

F3 is a distinct scoring layer.  It never imports a codec encoder, a training
entry point, or the guarded test split.  The historical records and clean G1
observations remain read-only; only artifact-classifier predictions are new.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from baseline import g8_e_corrected_v2 as v2
from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s
from baseline import g8_e_pass_one
from config.params import REPO_ROOT, get

SCHEMA_VERSION = 1
SCOPE = "G8_F_F3_EXISTING_G8E_VALIDATION_CACHE_RESCORING_ONLY"
RUNTIME_ROOT = v3s.V3S_RUNTIME_ROOT
F3_ROOT = REPO_ROOT / "results/baseline/g8_f/f3"
CACHE_MANIFEST_PATH = F3_ROOT / "cache_manifest.json"
CONTRACT_PATH = F3_ROOT / "f3_contract.json"
SUPERSEDED_CONTRACT_PATH = F3_ROOT / "f3_contract_v1_superseded_before_inference.json"
AGGREGATE_PATH = F3_ROOT / "f3_scoring_aggregate.json"
REMOTE_SCORE_DIRNAME = "f3_artifact_scores"
F2_FREEZE_PATH = REPO_ROOT / "results/baseline/g8_f/artifact_classifier_freeze.json"
F2_COMPLETION_PATH = REPO_ROOT / "results/baseline/g8_f/f2_completion.json"
G1_ADJUDICATION_PATH = REPO_ROOT / "results/reference_classifier/g1_adjudication.json"
PASS_ONE_PATH = REPO_ROOT / "results/baseline/g8_e/pass_one_state.json"
EXPECTED_ROWS = 288_000
EXPECTED_STRUCTURAL = 288  # literal-ok: frozen G8_E initial-dataset structural universe
EXPECTED_DELIVERED = 264_000
EXPECTED_OUTAGE = 24_000
EXPECTED_RECONSTRUCTIONS = 104_000
EXPECTED_PER_STRUCTURAL = int(get("datasets.imagenette160.val_images"))
CANONICAL_AXIS = int(get("datasets.imagenette160.image_size")[0])
DEFAULT_BATCH_SIZE = int(get("reference_classifier.batch_size"))
CONTRACT_PREFIX = "g8ff3contract-"
CACHE_PREFIX = "g8ff3cache-"
AGGREGATE_PREFIX = "g8ff3scores-"


class F3Hold(RuntimeError):
    """A fail-closed F3 cache, contract, checkpoint, or exact-set violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise F3Hold(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):  # literal-ok: streaming I/O block size only
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def rendered_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def identified(body: Mapping[str, Any], *, field: str, prefix: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = prefix + sha256_bytes(canonical_json(value))
    value["artifact_content_sha256"] = sha256_bytes(canonical_json(value))
    return value


def atomic_bytes(path: Path, raw: bytes, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise F3Hold(f"immutable F3 output already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_existing and path.exists():
            raise F3Hold(f"immutable F3 output already exists: {path}")
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise F3Hold(f"cannot load {path}: {exc}") from None
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _verify_identified(value: Mapping[str, Any], *, field: str, prefix: str) -> None:
    without_digest = {k: v for k, v in value.items() if k != "artifact_content_sha256"}
    require(value.get("artifact_content_sha256") == sha256_bytes(canonical_json(without_digest)), f"{field} content digest differs")
    without_id = {k: v for k, v in without_digest.items() if k != field}
    require(value.get(field) == prefix + sha256_bytes(canonical_json(without_id)), f"{field} differs")


def frozen_context() -> dict[str, Any]:
    context = g8_e_pass_one.authenticate_frozen_chain()
    pass_one = g8_e_pass_one.verify_pass_one_state()
    require(pass_one["counters"]["pass_two"] == 0 and pass_one["counters"]["pass_three"] == 0 and pass_one["counters"]["test_access"] == 0, "pass two/three/test is not zero")
    freeze = _json(F2_FREEZE_PATH)
    require(freeze.get("status") == "FROZEN_TRAINING_CLOSED_F3_CLOSED", "F2 classifier is not frozen with F3 closed")
    require(freeze.get("protected_state") == {"f3_cached_sweep_rescoring": 0, "fallback": 0, "learned_training": 0, "pass_three": 0, "pass_two": 0, "ratio_adjudication": 0, "test_access": 0}, "F2 protected state differs")
    return {**context, "pass_one_verification": pass_one, "f2_freeze": freeze}


def _e4_record_index(e4: Mapping[str, Any]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    by_record: dict[str, tuple[str, str]] = {}
    ordered: list[str] = []
    for obj in e4["objects"]:
        structural_id = str(obj["measurement_identity_id"])
        ids = obj["source_record_ids"]
        shas = obj["source_record_sha256s"]
        require(len(ids) == len(shas) == EXPECTED_PER_STRUCTURAL, f"E4 structural object {structural_id} denominator differs")
        for record_id, record_sha in zip(ids, shas, strict=True):
            require(record_id not in by_record, "E4 repeats a source record ID")
            by_record[str(record_id)] = (structural_id, str(record_sha))
            ordered.append(str(record_id))
    require(len(by_record) == EXPECTED_ROWS, "E4 source-record exact set differs")
    return by_record, ordered


def _reconstruction_file_identity(path: Path, expected_id: str) -> tuple[str, str]:
    require(path.is_file() and not path.is_symlink(), f"cached reconstruction is missing or unsafe: {expected_id}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise F3Hold(f"cached reconstruction JSON is corrupt: {expected_id}") from None
    require(raw == rendered_json(value), f"cached reconstruction is not canonical: {expected_id}")
    required = {"schema_version", "artifact_role", "identity", "status", "reason", "pixels_b64", "pixels_sha256", "object_id"}
    require(set(value) == required and value["schema_version"] == v2.V2_SCHEMA_VERSION and value["artifact_role"] == "g8_e_v2_reconstruction_cache_object", f"cached reconstruction schema differs: {expected_id}")
    require(value["object_id"] == expected_id == v3._id(v2.V2_RECONSTRUCTION_PREFIX, value["identity"]), f"cached reconstruction identity differs: {expected_id}")
    require(value["status"] == v2.OUTCOME_DELIVERED and value["reason"] is None and isinstance(value["pixels_b64"], str), f"cached delivered reconstruction status differs: {expected_id}")
    try:
        pixels = base64.b64decode(value["pixels_b64"], validate=True)
    except ValueError:
        raise F3Hold(f"cached reconstruction base64 is corrupt: {expected_id}") from None
    require(len(pixels) == CANONICAL_AXIS * CANONICAL_AXIS * 3 and sha256_bytes(pixels) == value["pixels_sha256"], f"cached reconstruction pixels differ: {expected_id}")
    return sha256_bytes(raw), str(value["pixels_sha256"])


def build_cache_manifest(runtime_root: Path = RUNTIME_ROOT) -> tuple[dict[str, Any], bytes]:
    """Authenticate the full historical cache and freeze its exact F3 universe.

    This performs no classifier inference.  It first independently rebuilds E3
    with complete codec/reconstruction/old-observation authentication, then
    binds every record and every unique reconstruction file by exact bytes.
    """

    runtime_root = Path(runtime_root).resolve()
    context = frozen_context()
    rebuilt_e3 = v3.build_e3_artifact(
        authority=context["measurement_authority"],
        sample_ids=context["sample_ids"],
        sample_labels=context["sample_labels"],
        runtime_root=runtime_root,
        contract=context["contract"],
        production=True,
        authenticate_caches=True,
    )
    require(rebuilt_e3 == context["e3"], "live historical runtime does not exactly reproduce frozen E3")
    record_index, e4_order = _e4_record_index(context["e4"])
    expected = v3.expected_work_units(context["measurement_authority"], context["sample_ids"])
    require(len(expected) == EXPECTED_ROWS, "F3 expected work-unit count differs")
    outcomes: Counter[str] = Counter()
    record_rows: list[list[Any]] = []
    ordered_record_ids: list[str] = []
    reconstruction_refs: list[str] = []
    unique_reconstruction_ids: set[str] = set()
    for ordinal, unit in enumerate(expected):
        path = runtime_root / "records" / f"{unit['work_unit_id']}.json"
        require(path.is_file() and not path.is_symlink(), f"historical record is missing or unsafe at ordinal {ordinal}")
        raw = path.read_bytes()
        value = v3.MeasurementRecordV3.from_mapping(json.loads(raw)).value
        require(value["authority_ordinal"] == ordinal and value["work_unit_id"] == unit["work_unit_id"] and value["measurement_identity_id"] == unit["measurement_identity_id"] and value["stable_sample_id"] == unit["stable_sample_id"], f"historical record authority binding differs at ordinal {ordinal}")
        expected_structural, expected_sha = record_index.get(value["record_id"], (None, None))
        require(expected_structural == value["measurement_identity_id"] and expected_sha == sha256_bytes(raw), f"historical record differs from frozen E4 at ordinal {ordinal}")
        outcomes[value["outcome"]] += 1
        reconstruction_id = None
        if value["outcome"] == v2.OUTCOME_DELIVERED:
            require(value["reconstruction"] is not None and value["classifier_observation"] is not None and value["outage_applied"] is False, f"delivered F3 row semantics differ at ordinal {ordinal}")
            reconstruction_id = str(value["reconstruction"]["object_id"])
            reconstruction_refs.append(reconstruction_id)
            unique_reconstruction_ids.add(reconstruction_id)
        else:
            require(value["outage_applied"] is True and value["classifier_observation"] is None, f"outage F3 row semantics differ at ordinal {ordinal}")
        ordered_record_ids.append(str(value["record_id"]))
        record_rows.append([ordinal, value["work_unit_id"], value["record_id"], sha256_bytes(raw), value["measurement_identity_id"], value["stable_sample_id"], value["label"], value["outcome"], reconstruction_id])
    require(ordered_record_ids == e4_order, "E4 source-record order differs from authority order")
    object_rows: list[list[str]] = []
    for reconstruction_id in sorted(unique_reconstruction_ids):
        file_sha, pixels_sha = _reconstruction_file_identity(runtime_root / "reconstruction" / f"{reconstruction_id}.json", reconstruction_id)
        object_rows.append([reconstruction_id, file_sha, pixels_sha])
    require(outcomes == Counter({v2.OUTCOME_DELIVERED: EXPECTED_DELIVERED, v2.OUTCOME_CODEC_INFEASIBILITY: EXPECTED_OUTAGE}), f"historical F3 outcome mix differs: {dict(outcomes)}")
    require(len(object_rows) == EXPECTED_RECONSTRUCTIONS, "historical reconstruction-object count differs")
    inventory_raw = b"".join(canonical_json(row) for row in object_rows)
    inventory_path = runtime_root / "f3_cache_inventory.jsonl"
    atomic_bytes(inventory_path, inventory_raw)
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "g8_f_f3_authenticated_historical_cache_manifest",
        "status": "CACHE_EXACT_AUTHENTICATED_NO_INFERENCE",
        "runtime_root": str(runtime_root),
        "e3_id": context["e3"]["e3_id"],
        "e3_sha256": sha256_file(v3s.V3S_E3_PATH),
        "e4_id": context["e4"]["e4_id"],
        "e4_sha256": sha256_file(v3s.V3S_E4_PATH),
        "validation_manifest_sha256": context["contract"]["scientific_data_identity"]["manifest_sha256"],
        "row_count": len(record_rows),
        "structural_identity_count": EXPECTED_STRUCTURAL,
        "outcomes": {"delivered": outcomes[v2.OUTCOME_DELIVERED], "codec_infeasibility": outcomes[v2.OUTCOME_CODEC_INFEASIBILITY], "decode_failure": outcomes[v2.OUTCOME_DECODE_FAILURE], "structural_infeasibility": outcomes[v2.OUTCOME_STRUCTURAL_INFEASIBILITY]},
        "ordered_record_identity_sha256": sha256_bytes(canonical_json(record_rows)),
        "record_identity_set_sha256": sha256_bytes(canonical_json(sorted(record_rows, key=lambda row: row[1]))),
        "ordered_reconstruction_reference_sha256": sha256_bytes(canonical_json(reconstruction_refs)),
        "reconstruction_reference_set_sha256": sha256_bytes(canonical_json(sorted(set(reconstruction_refs)))),
        "unique_reconstruction_object_count": len(object_rows),
        "reconstruction_object_identity_set_sha256": sha256_bytes(canonical_json(object_rows)),
        "inventory_path": str(inventory_path),
        "inventory_bytes": len(inventory_raw),
        "inventory_sha256": sha256_bytes(inventory_raw),
        "no_reencode": True,
        "classifier_inference": 0,
        "optimizer_steps": 0,
        "test_access": 0,
    }
    value = identified(body, field="cache_manifest_id", prefix=CACHE_PREFIX)
    return value, rendered_json(value)


def build_contract(*, source_commit: str, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    context = frozen_context()
    cache_raw = CACHE_MANIFEST_PATH.read_bytes()
    cache = json.loads(cache_raw)
    _verify_identified(cache, field="cache_manifest_id", prefix=CACHE_PREFIX)
    require(cache["status"] == "CACHE_EXACT_AUTHENTICATED_NO_INFERENCE" and cache["classifier_inference"] == 0, "F3 cache manifest is not pre-inference")
    inventory = Path(cache["inventory_path"])
    require(inventory.is_file() and inventory.stat().st_size == cache["inventory_bytes"] and sha256_file(inventory) == cache["inventory_sha256"], "F3 cache inventory bytes differ")
    freeze = context["f2_freeze"]
    source_paths = ["src/baseline/g8_f_f3.py", "tools/preflight_g8_f_f3.py", "tools/run_g8_f_f3.py"]
    source_manifest = [{"path": path, "sha256": sha256_file(REPO_ROOT / path), "bytes": (REPO_ROOT / path).stat().st_size} for path in source_paths]
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "g8_f_f3_pre_inference_contract",
        "status": "FROZEN_PRE_INFERENCE",
        "scope": SCOPE,
        "source_commit": source_commit,
        "source_manifest": source_manifest,
        "execution_profile": freeze["execution_profile"],
        "device": "cuda:0",
        "artifact_classifier": {"freeze_id": freeze["freeze_id"], "freeze_file_sha256": sha256_file(F2_FREEZE_PATH), "checkpoint_id": freeze["checkpoint_id"], "checkpoint_sha256": freeze["checkpoint_file_sha256"], "checkpoint_bytes": freeze["checkpoint_bytes"], "checkpoint_path": freeze["checkpoint_repository_path"], "scorer_identity": freeze["scorer_identity"]},
        "clean_g1_scorer": {"checkpoint_id": freeze["parent_g1_checkpoint_id"], "checkpoint_sha256": freeze["parent_g1_checkpoint_sha256"], "adjudication_file_sha256": sha256_file(G1_ADJUDICATION_PATH), "historical_scores_immutable": True},
        "e4": {"id": context["e4"]["e4_id"], "file_sha256": sha256_file(v3s.V3S_E4_PATH)},
        "pass_one": {"id": context["pass_one_verification"]["state_id"], "file_sha256": context["pass_one_verification"]["file_sha256"], "immutable": True},
        "cache_manifest": {"id": cache["cache_manifest_id"], "file_sha256": sha256_bytes(cache_raw), "inventory_sha256": cache["inventory_sha256"], "ordered_record_identity_sha256": cache["ordered_record_identity_sha256"], "reconstruction_object_identity_set_sha256": cache["reconstruction_object_identity_set_sha256"]},
        "validation_membership": {"dataset": "imagenette160", "split": "val", "manifest_sha256": context["contract"]["scientific_data_identity"]["manifest_sha256"], "rows_per_structural_identity": EXPECTED_PER_STRUCTURAL, "structural_identity_count": EXPECTED_STRUCTURAL, "row_count": EXPECTED_ROWS},
        "outage_policy": context["contract"]["outage_policy"],
        "denominator_semantics": "all 1000 frozen validation rows per structural identity; delivered correctness from artifact scorer and every failure from frozen binary constant-class outage",
        "aggregation_rule": "integer correct-count sum over exactly 1000 authority-ordered binary rows per structural identity; no omissions or substitutions",
        "output_schema": {"unit_schema_version": 1, "unit_path": f"{REMOTE_SCORE_DIRNAME}/<measurement_identity_id>.json", "aggregate_path": str(AGGREGATE_PATH.relative_to(REPO_ROOT))},
        "protected_starting_state": {"f3_cached_sweep_rescoring": 0, "pass_two": 0, "pass_three": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0},
        "supersedes_before_inference": None if not SUPERSEDED_CONTRACT_PATH.exists() else {"path": str(SUPERSEDED_CONTRACT_PATH.relative_to(REPO_ROOT)), "file_sha256": sha256_file(SUPERSEDED_CONTRACT_PATH), "reason": "v1 clean-checkout verifier required worker-only inventory bytes; no inference or science occurred"},
        "inference_only": {"model_eval": True, "torch_inference_mode": True, "optimizer": "absent", "checkpoint_writes": "evidence_metadata_only", "random_eval_transforms": False, "jpeg2000_encoding": False},
    }
    return identified(body, field="contract_id", prefix=CONTRACT_PREFIX)


def verify_contract(path: Path = CONTRACT_PATH, *, require_inventory: bool = False) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(raw == rendered_json(value), "F3 contract is not canonical")
    _verify_identified(value, field="contract_id", prefix=CONTRACT_PREFIX)
    require(value["status"] == "FROZEN_PRE_INFERENCE" and value["scope"] == SCOPE, "F3 contract status/scope differs")
    require(value["protected_starting_state"] == {"f3_cached_sweep_rescoring": 0, "pass_two": 0, "pass_three": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0}, "F3 starting counters differ")
    for row in value["source_manifest"]:
        path = REPO_ROOT / row["path"]
        require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"], f"F3 source bytes differ: {row['path']}")
    cache_raw = CACHE_MANIFEST_PATH.read_bytes()
    cache = json.loads(cache_raw)
    require(value["cache_manifest"]["id"] == cache["cache_manifest_id"] and value["cache_manifest"]["file_sha256"] == sha256_bytes(cache_raw), "F3 cache-manifest binding differs")
    inventory = Path(cache["inventory_path"])
    if require_inventory:
        require(inventory.is_file() and sha256_file(inventory) == value["cache_manifest"]["inventory_sha256"], "F3 inventory differs")
    freeze = _json(F2_FREEZE_PATH)
    require(value["artifact_classifier"]["freeze_id"] == freeze["freeze_id"] and value["artifact_classifier"]["checkpoint_sha256"] == freeze["checkpoint_file_sha256"], "F3 artifact-scorer binding differs")
    return value


def _load_inventory(cache: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    inventory: dict[str, tuple[str, str]] = {}
    for raw in Path(cache["inventory_path"]).read_bytes().splitlines():
        row = json.loads(raw)
        require(isinstance(row, list) and len(row) == 3 and row[0] not in inventory, "F3 cache inventory schema/uniqueness differs")
        inventory[row[0]] = (row[1], row[2])
    require(len(inventory) == EXPECTED_RECONSTRUCTIONS, "F3 cache inventory count differs")
    return inventory


def _load_artifact_model(checkpoint_path: Path, device: str) -> Any:
    import torch
    from models.frozen_reference_classifier import load_frozen_reference_classifier
    from training.g8_f_f2_closeout import verify_checkpoint

    verify_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_frozen_reference_classifier(device, allow_download=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()), "F3 model is not frozen in eval mode")
    return model


def _score_unit(*, structural: Mapping[str, Any], e4_object: Mapping[str, Any], expected_units: Sequence[Mapping[str, Any]], runtime_root: Path, inventory: Mapping[str, tuple[str, str]], model: Any, device: str, outage_class: int, contract_id: str, batch_size: int) -> dict[str, Any]:
    import torch
    from data.preprocessing import reconstruction_input

    rows: list[dict[str, Any]] = []
    batch_tensors: list[Any] = []
    batch_positions: list[int] = []
    outcomes: Counter[str] = Counter()
    inference_count = 0

    def flush() -> None:
        nonlocal inference_count
        if not batch_tensors:
            return
        tensor = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            predictions = model(tensor).argmax(dim=1).cpu().tolist()
        require(len(predictions) == len(batch_positions), "F3 prediction batch length differs")
        for position, prediction in zip(batch_positions, predictions, strict=True):
            rows[position]["predicted_label"] = int(prediction)
            rows[position]["correct"] = int(prediction == rows[position]["label"])
        inference_count += len(predictions)
        batch_tensors.clear(); batch_positions.clear()

    for index, unit in enumerate(expected_units):
        path = runtime_root / "records" / f"{unit['work_unit_id']}.json"
        raw = path.read_bytes(); value = v3.MeasurementRecordV3.from_mapping(json.loads(raw)).value
        require(value["record_id"] == e4_object["source_record_ids"][index] and sha256_bytes(raw) == e4_object["source_record_sha256s"][index], "F3 record differs from E4 during scoring")
        outcome = value["outcome"]; outcomes[outcome] += 1
        row = {"ordinal_within_structural": index, "work_unit_id": value["work_unit_id"], "record_id": value["record_id"], "record_sha256": sha256_bytes(raw), "stable_sample_id": value["stable_sample_id"], "label": value["label"], "outcome": outcome, "reconstruction_object_id": None, "reconstruction_pixels_sha256": None, "predicted_label": None, "correct": None}
        rows.append(row)
        if outcome == v2.OUTCOME_DELIVERED:
            reconstruction_id = value["reconstruction"]["object_id"]
            expected_file_sha, expected_pixels_sha = inventory.get(reconstruction_id, (None, None))
            reconstruction_path = runtime_root / "reconstruction" / f"{reconstruction_id}.json"
            require(expected_file_sha is not None and sha256_file(reconstruction_path) == expected_file_sha, "F3 reconstruction changed after contract freeze")
            reconstruction = json.loads(reconstruction_path.read_bytes())
            pixels_raw = base64.b64decode(reconstruction["pixels_b64"], validate=True)
            require(sha256_bytes(pixels_raw) == expected_pixels_sha == reconstruction["pixels_sha256"], "F3 reconstruction pixels changed after contract freeze")
            pixels = np.frombuffer(pixels_raw, dtype=np.uint8).reshape(CANONICAL_AXIS, CANONICAL_AXIS, 3)
            row["reconstruction_object_id"] = reconstruction_id; row["reconstruction_pixels_sha256"] = expected_pixels_sha
            batch_tensors.append(reconstruction_input(pixels)); batch_positions.append(len(rows) - 1)
            if len(batch_tensors) >= batch_size: flush()
        else:
            require(value["outage_applied"] is True and value["outage_prediction"]["selected_class"] == outage_class, "F3 outage semantics differ")
            row["predicted_label"] = outage_class; row["correct"] = int(value["label"] == outage_class)
    flush()
    require(len(rows) == EXPECTED_PER_STRUCTURAL and all(row["correct"] in (0, 1) for row in rows), "F3 structural denominator/scoring differs")
    scoring_hashes = [sha256_bytes(canonical_json(row)) for row in rows]
    body = {"schema_version": 1, "artifact_role": "g8_f_f3_structural_scoring_unit", "status": "COMPLETE", "contract_id": contract_id, "measurement_identity_id": structural["structural_identity_id"], "structural_identity": dict(structural), "row_count": len(rows), "delivered_count": outcomes[v2.OUTCOME_DELIVERED], "outage_count": len(rows) - outcomes[v2.OUTCOME_DELIVERED], "correct_count": sum(row["correct"] for row in rows), "inference_count": inference_count, "ordered_scoring_sha256": sha256_bytes(canonical_json(scoring_hashes)), "scoring_set_sha256": sha256_bytes(canonical_json(sorted(scoring_hashes))), "rows": rows}
    body["unit_id"] = "g8ff3unit-" + sha256_bytes(canonical_json(body))
    body["artifact_content_sha256"] = sha256_bytes(canonical_json(body))
    return body


def run_f3(*, runtime_root: Path, checkpoint_path: Path, device: str = "cuda:0", batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    contract = verify_contract(require_inventory=True); cache = _json(CACHE_MANIFEST_PATH); inventory = _load_inventory(cache)
    require(str(Path(runtime_root).resolve()) == cache["runtime_root"], "F3 runtime root differs from contract cache")
    require(device == contract["device"] and batch_size > 0, "F3 device/batch size differs")
    require(checkpoint_path.stat().st_size == contract["artifact_classifier"]["checkpoint_bytes"] and sha256_file(checkpoint_path) == contract["artifact_classifier"]["checkpoint_sha256"], "F3 artifact checkpoint bytes differ")
    context = frozen_context(); model = _load_artifact_model(checkpoint_path, device)
    authority_rows = [row for row in context["measurement_authority"]["structural_identities"] if row["dataset"] == "imagenette160"]
    authority_rows.sort(key=lambda row: row["structural_identity_id"])
    e4_by_id = {obj["measurement_identity_id"]: obj for obj in context["e4"]["objects"]}
    expected = v3.expected_work_units(context["measurement_authority"], context["sample_ids"])
    units_by_structural: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in expected: units_by_structural[unit["measurement_identity_id"]].append(unit)
    output_dir = Path(runtime_root) / REMOTE_SCORE_DIRNAME; output_dir.mkdir(parents=True, exist_ok=True)
    for structural in authority_rows:
        structural_id = structural["structural_identity_id"]; output = output_dir / f"{structural_id}.json"
        if output.exists():
            verify_scoring_unit(output, contract_id=contract["contract_id"], structural_id=structural_id)
            continue
        unit = _score_unit(structural=structural, e4_object=e4_by_id[structural_id], expected_units=units_by_structural[structural_id], runtime_root=Path(runtime_root), inventory=inventory, model=model, device=device, outage_class=int(contract["outage_policy"]["selected_class"]), contract_id=contract["contract_id"], batch_size=batch_size)
        atomic_bytes(output, rendered_json(unit), refuse_existing=True)
        print(f"F3 {structural_id} delivered={unit['delivered_count']} outage={unit['outage_count']} correct={unit['correct_count']}", flush=True)


def verify_scoring_unit(path: Path, *, contract_id: str, structural_id: str) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == rendered_json(value), f"F3 scoring unit is not canonical: {structural_id}")
    without_digest = {k: v for k, v in value.items() if k != "artifact_content_sha256"}
    require(value["artifact_content_sha256"] == sha256_bytes(canonical_json(without_digest)), f"F3 scoring unit digest differs: {structural_id}")
    without_id = {k: v for k, v in without_digest.items() if k != "unit_id"}
    require(value["unit_id"] == "g8ff3unit-" + sha256_bytes(canonical_json(without_id)), f"F3 scoring unit ID differs: {structural_id}")
    require(value["contract_id"] == contract_id and value["measurement_identity_id"] == structural_id and value["row_count"] == EXPECTED_PER_STRUCTURAL and len(value["rows"]) == EXPECTED_PER_STRUCTURAL, f"F3 scoring unit binding/denominator differs: {structural_id}")
    hashes = [sha256_bytes(canonical_json(row)) for row in value["rows"]]
    require(value["ordered_scoring_sha256"] == sha256_bytes(canonical_json(hashes)) and value["scoring_set_sha256"] == sha256_bytes(canonical_json(sorted(hashes))), f"F3 scoring digests differ: {structural_id}")
    require(value["correct_count"] == sum(row["correct"] for row in value["rows"]), f"F3 correct count differs: {structural_id}")
    return value


def build_aggregate(runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    contract = verify_contract(); context = frozen_context(); output_dir = Path(runtime_root) / REMOTE_SCORE_DIRNAME
    structural = {row["structural_identity_id"]: row for row in context["measurement_authority"]["structural_identities"] if row["dataset"] == "imagenette160"}
    require(len(structural) == EXPECTED_STRUCTURAL, "F3 structural authority count differs")
    objects: list[dict[str, Any]] = []; all_hashes: list[str] = []; outcomes: Counter[str] = Counter(); total_correct = total_inference = 0
    quality_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for structural_id in sorted(structural):
        path = output_dir / f"{structural_id}.json"; require(path.is_file() and not path.is_symlink(), f"F3 scoring unit is missing: {structural_id}")
        unit = verify_scoring_unit(path, contract_id=contract["contract_id"], structural_id=structural_id)
        row_hashes = [sha256_bytes(canonical_json(row)) for row in unit["rows"]]; all_hashes.extend(row_hashes)
        for row in unit["rows"]: outcomes[row["outcome"]] += 1
        total_correct += unit["correct_count"]; total_inference += unit["inference_count"]
        item = {"measurement_identity_id": structural_id, "ratio": structural[structural_id]["ratio"], "modulation": structural[structural_id]["modulation"], "ldpc_rate": structural[structural_id]["ldpc_rate"], "encode_axis_px": structural[structural_id]["encode_axis_px"], "payload_budget_bytes": structural[structural_id]["payload_budget_bytes"], "correct_count": unit["correct_count"], "total_count": EXPECTED_PER_STRUCTURAL, "delivered_count": unit["delivered_count"], "outage_count": unit["outage_count"], "ordered_scoring_sha256": unit["ordered_scoring_sha256"], "scoring_set_sha256": unit["scoring_set_sha256"], "unit_id": unit["unit_id"], "unit_file_sha256": sha256_file(path)}
        objects.append(item)
        quality_key = (item["ratio"], item["payload_budget_bytes"], item["encode_axis_px"])
        quality_groups[quality_key].append(item)
    require(len(all_hashes) == EXPECTED_ROWS and total_inference == EXPECTED_DELIVERED and outcomes == Counter({v2.OUTCOME_DELIVERED: EXPECTED_DELIVERED, v2.OUTCOME_CODEC_INFEASIBILITY: EXPECTED_OUTAGE}), "F3 aggregate exact counts differ")
    qualities = [{"ratio": key[0], "payload_budget_bytes": key[1], "encode_axis_px": key[2], "structural_identity_count": len(items), "correct_counts": sorted({item["correct_count"] for item in items}), "total_count_per_structural": EXPECTED_PER_STRUCTURAL} for key, items in sorted(quality_groups.items())]
    body = {"schema_version": SCHEMA_VERSION, "artifact_role": "g8_f_f3_artifact_scorer_aggregate", "status": "F3_COMPLETE_EXACT", "contract_id": contract["contract_id"], "contract_file_sha256": sha256_file(CONTRACT_PATH), "scorer": contract["artifact_classifier"], "historical_clean_scorer": contract["clean_g1_scorer"], "e4": contract["e4"], "cache_manifest": contract["cache_manifest"], "row_count": len(all_hashes), "structural_identity_count": len(objects), "outcomes": {"delivered": outcomes[v2.OUTCOME_DELIVERED], "codec_infeasibility": outcomes[v2.OUTCOME_CODEC_INFEASIBILITY], "decode_failure": outcomes[v2.OUTCOME_DECODE_FAILURE], "structural_infeasibility": outcomes[v2.OUTCOME_STRUCTURAL_INFEASIBILITY]}, "artifact_classifier_inference_count": total_inference, "correct_count": total_correct, "ordered_scoring_sha256": sha256_bytes(canonical_json(all_hashes)), "scoring_set_sha256": sha256_bytes(canonical_json(sorted(all_hashes))), "objects": objects, "quality_summary": qualities, "protected_counters": {"f3_cached_sweep_rescoring": 1, "f2_optimizer_steps_during_f3": 0, "pass_two": 0, "pass_three": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0}, "selection_ready": True}
    return identified(body, field="aggregate_id", prefix=AGGREGATE_PREFIX)


def verify_aggregate(path: Path = AGGREGATE_PATH, *, live_runtime: Path | None = None) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    require(raw == rendered_json(value), "F3 aggregate is not canonical")
    _verify_identified(value, field="aggregate_id", prefix=AGGREGATE_PREFIX)
    require(value["status"] == "F3_COMPLETE_EXACT" and value["row_count"] == EXPECTED_ROWS and value["structural_identity_count"] == EXPECTED_STRUCTURAL and value["artifact_classifier_inference_count"] == EXPECTED_DELIVERED, "F3 aggregate closure counts differ")
    require(value["outcomes"] == {"delivered": EXPECTED_DELIVERED, "codec_infeasibility": EXPECTED_OUTAGE, "decode_failure": 0, "structural_infeasibility": 0}, "F3 aggregate outcome mix differs")
    require(value["protected_counters"] == {"f3_cached_sweep_rescoring": 1, "f2_optimizer_steps_during_f3": 0, "pass_two": 0, "pass_three": 0, "fallback_training": 0, "learned_training": 0, "test_access": 0}, "F3 aggregate protected counters differ")
    contract = verify_contract(); require(value["contract_id"] == contract["contract_id"] and value["scorer"] == contract["artifact_classifier"] and value["historical_clean_scorer"] == contract["clean_g1_scorer"], "F3 scorer/contract binding differs")
    if live_runtime is not None:
        rebuilt = build_aggregate(live_runtime); require(rebuilt == value, "F3 live scoring units do not reproduce compact aggregate")
    return value
