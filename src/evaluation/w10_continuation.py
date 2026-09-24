"""Fail-closed bridge from the failed v8 prefix to a separately authorized v9 suffix."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline.classical.records import make_pair_id
from config.params import get
from evaluation.downstream_v4 import read_json, require
from evaluation.w10_classical import _candidate_table, _selection_map, resolve_classical_configuration
from evaluation.w10_evidence import per_image_relative_path, scheduled_noise_id, unit_relative_path
from evaluation.w10_rehearsal import (
    _validate_persisted_streams,
    per_image_manifest,
    unit_manifest,
    validate_unit,
    work_units,
)
from evaluation.w10_scope import W10_VALIDATION_DENOMINATOR
from runtime.source_epochs import (
    W10_V8_AUTHORITY_ID,
    W10_V8_AUTHORITY_SHA256,
    W10_V8_CUSTODY_ID,
    W10_V8_CUSTODY_SHA256,
    W10_V8_EXECUTION_COMMIT,
    W10_V8_MANIFEST_ID,
    W10_V8_MANIFEST_SHA256,
    W10_V8_PREFIX_DIGEST,
    W10_V9_MANIFEST_KIND,
    assert_active_epoch_closure,
    load_w10_manifest,
    source_record,
    v8_continuation_custody,
)
from runtime.source_guard import assert_manifest_commit_bytes, git_blob_bytes
from training.deterministic_core import canonical_sha256

CONTINUATION_AUTHORITY_PATH = "results/learned/w10/w10_continuation_authorization_v9.json"
CONTINUATION_LAUNCH_PATH = "results/learned/w10/w10_continuation_launch_authorization_v9.json"
CONTINUATION_PLAN_PATH = "continuation_plan_v9.json"
CONTINUATION_START = 126
CONTINUATION_STOP = 252
CONTINUATION_AUTHORITY_KIND = "W10_VALIDATION_CONTINUATION_AUTHORITY_V9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_scheduled_rows(rows: Sequence[Mapping[str, Any]], unit: Mapping[str, Any]) -> None:
    k = int(get(f"bandwidth.k_symbols.imagenette160.{unit['bw_ratio']}"))
    for row in rows:
        stable_id = row["stable_sample_id"]
        noise_id = scheduled_noise_id(
            stable_sample_id=stable_id,
            bw_ratio=unit["bw_ratio"],
            test_snr_db=unit["snr_db"],
            channel_seed=unit["channel_seed"],
            k=k,
        )
        require(row["noise_id"] == noise_id, "W10 scheduled noise identity differs")
        pair_id = make_pair_id({
            "analysis_cell_id": row["analysis_cell_id"],
            "stable_sample_id": stable_id,
            "bw_ratio": unit["bw_ratio"],
            "test_snr_db": float(unit["snr_db"]),
            "noise_id": noise_id,
        })
        require(row["pair_id"] == pair_id, "W10 pair identity differs")
        require(row["correct"] == (row["pred_label"] == row["true_label"]), "W10 correctness differs")


def _verify_completed_code_paths(root: Path) -> None:
    """Bind completed learned transport, noise, scoring and scope to v8 bytes."""

    for relative in (
        "src/evaluation/w10_dispatch.py",
        "src/evaluation/w10_backends.py",
        "src/evaluation/w10_evidence.py",
        "src/evaluation/w10_scope.py",
        "src/evaluation/w10_bindings.py",
        "src/evaluation/w10_selections.py",
        "src/data/djscc_validation.py",
    ):
        require((root / relative).read_bytes() == git_blob_bytes(root, W10_V8_EXECUTION_COMMIT, relative), f"completed W10 code path changed: {relative}")


def _verify_completed_binding(
    original: Mapping[str, Any],
    unit: Mapping[str, Any],
    value: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    frozen = next(item for item in original["bindings"] if item["scope"]["role"] == unit["role"])
    binding = value["binding"]
    if unit["ordinal"] < 84:  # literal-ok: four completed learned arms, 21 SNRs each
        checkpoint = frozen["checkpoint"]
        require(binding.get("checkpoint_id") == checkpoint.get("checkpoint_id", checkpoint.get("checkpoint_sha256")), "completed learned checkpoint differs")
        if unit["system"] == "learned_snr_randomised":
            require(binding.get("task_head_identity") == checkpoint["task_head_identity"], "completed randomized task head differs")
        if unit["system"] == "learned_papr_constrained":
            require(binding.get("config_hash") == checkpoint["config_hash"] and binding.get("protocol_config_hash") == checkpoint["protocol_config_hash"], "completed PAPR protocol differs")
        return
    selection = frozen["selection"]
    point = _selection_map(selection)[int(unit["snr_db"])]
    expected = resolve_classical_configuration(
        binding=selection, point=point, candidates=candidates, codec_kind="jpeg2000", quality=None
    )
    actual = (binding.get("modulation"), binding.get("ldpc_rate"), binding.get("encode_axis_px"), binding.get("config_hash"))
    require(actual == expected and binding.get("selection_kind") == "classical_pass_two", "completed adaptive configuration differs from v8")


def verify_historical_authority(root: Path, *, live: bool) -> dict[str, Any]:
    """Authenticate the original authority against its own v8 source epoch."""

    custody = v8_continuation_custody(root)
    source = load_w10_manifest(root, live=False, epoch="v8")
    assert_manifest_commit_bytes(root, source)
    if live:
        assert_active_epoch_closure(root, source)
    authority = read_json(Path(root) / custody["authority"]["path"], "original W10 authority")
    require(authority["source_manifest"] == source_record(root, source), "original authority source record differs")
    require(authority["unit_count"] == CONTINUATION_STOP and authority["scope_sha256"] == canonical_sha256(authority["scope"]), "original authority scope differs")
    require(authority["gpu_uuid"] == "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a" and authority["host"] == "confessor" and authority["device"] == "cuda:0", "original authority worker differs")
    require(authority["authority_id"] == W10_V8_AUTHORITY_ID and _sha(Path(root) / custody["authority"]["path"]) == W10_V8_AUTHORITY_SHA256, "original authority bytes differ")
    return authority


def build_continuation_authority(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    """Project the original frozen science into a separate suffix-only grant."""

    require(source.get("manifest_kind") == W10_V9_MANIFEST_KIND, "W10 continuation needs source-v9")
    original = verify_historical_authority(root, live=False)
    body = {key: value for key, value in original.items() if key != "authority_id"}
    body.update({
        "schema_version": 3,
        "authority_kind": CONTINUATION_AUTHORITY_KIND,
        "status": "FROZEN_SUFFIX_ONLY_PRE_EXECUTION",
        "source_manifest": source_record(root, source),
        "source_binding": dict(source),
        "source_commit": source["source_commit"],
        "original_authority_id": W10_V8_AUTHORITY_ID,
        "original_authority_sha256": W10_V8_AUTHORITY_SHA256,
        "original_source_manifest_id": W10_V8_MANIFEST_ID,
        "original_source_manifest_sha256": W10_V8_MANIFEST_SHA256,
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "historical_custody_sha256": W10_V8_CUSTODY_SHA256,
        "historical_prefix_digest": W10_V8_PREFIX_DIGEST,
        "historical_completed_ordinals": list(range(CONTINUATION_START)),
        "new_authorized_ordinals": list(range(CONTINUATION_START, CONTINUATION_STOP)),
        "plan_path": CONTINUATION_PLAN_PATH,
        "suffix_only": True,
    })
    return body


def verify_continuation_authority(root: Path, *, live: bool = True) -> dict[str, Any]:
    root = Path(root)
    source = load_w10_manifest(root, live=live, epoch="v9")
    path = root / CONTINUATION_AUTHORITY_PATH
    require(path.is_file() and not path.is_symlink(), "W10 v9 continuation authority is missing or unsafe")
    value = read_json(path, "W10 v9 continuation authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "w10continuationauth-" + canonical_sha256(body), "W10 v9 continuation authority ID differs")
    require(body == build_continuation_authority(root, source), "W10 v9 continuation authority fields differ")
    from evaluation.w10_bindings import resolve_scope_bindings

    require(value["bindings"] == resolve_scope_bindings(root, verify_runtime=False), "W10 v9 frozen scientific bindings differ")
    require(value["scope_sha256"] == canonical_sha256(value["scope"]) and len(value["historical_completed_ordinals"]) == CONTINUATION_START and len(value["new_authorized_ordinals"]) == CONTINUATION_STOP - CONTINUATION_START, "W10 v9 scope frontier differs")
    require(value["gpu_uuid"] == "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a" and value["host"] == "confessor" and value["device"] == "cuda:0", "W10 v9 exact worker differs")
    require(value["validation_only"] is True and value["test"] == "SEALED" and value["test_access"] == 0 and value["test_authorized"] is False, "W10 v9 authority crossed test boundary")
    return value


def verify_worker_prefix(
    root: Path,
    *,
    expected_stable_ids: Sequence[str],
    live_source: bool = True,
    allow_suffix: bool = False,
) -> dict[str, Any]:
    """Rebuild every historical identity and compare the immutable custody record."""

    root = Path(root)
    authority = verify_historical_authority(root, live=live_source)
    _verify_completed_code_paths(root)
    custody = v8_continuation_custody(root)
    runtime = root / custody["runtime_root"]
    require(runtime.is_dir() and not runtime.is_symlink(), "historical W10 runtime is missing or unsafe")
    plan_path = runtime / custody["plan"]["path"]
    plan = read_json(plan_path, "original W10 plan")
    expected = work_units()
    require(plan["work_units"] == list(expected) and plan["authority_id"] == W10_V8_AUTHORITY_ID, "original W10 plan differs")
    require(_sha(plan_path) == custody["plan"]["sha256"], "original W10 plan bytes differ")
    require(plan["validation_only"] is True and plan["test"] == "SEALED" and plan["test_access"] == 0, "original W10 plan crossed test boundary")
    require(len(expected_stable_ids) == W10_VALIDATION_DENOMINATOR and len(set(expected_stable_ids)) == W10_VALIDATION_DENOMINATOR, "validation stable IDs differ")
    unit_paths = {str(path.relative_to(runtime)) for path in (runtime / "units").iterdir()}
    stream_paths = {str(path.relative_to(runtime)) for path in (runtime / "per_image").iterdir()}
    historical_unit_paths = {row["path"] for row in custody["units"]}
    historical_stream_paths = {row["path"] for row in custody["streams"]}
    if allow_suffix:
        require(historical_unit_paths <= unit_paths, "historical W10 unit file set differs")
        require(historical_stream_paths <= stream_paths, "historical W10 scorer file set differs")
    else:
        require(unit_paths == historical_unit_paths, "historical W10 unit file set differs")
        require(stream_paths == historical_stream_paths, "historical W10 scorer file set differs")
    unit_rows: list[dict[str, Any]] = []
    stream_rows: list[dict[str, Any]] = []
    adaptive_selection = next(item["selection"] for item in authority["bindings"] if item["scope"]["role"] == "er1_headline_classical")
    candidates = _candidate_table(root, expected_sha256=adaptive_selection["candidate_authority"]["sha256"])
    for expected_unit in expected[:CONTINUATION_START]:
        ordinal = expected_unit["ordinal"]
        relative = unit_relative_path(expected_unit, "json")
        path = runtime / relative
        require(path.is_file() and not path.is_symlink(), "historical W10 unit is missing or unsafe")
        value = read_json(path, "historical W10 unit")
        validate_unit(value, expected_unit)
        _verify_completed_binding(authority, expected_unit, value, candidates)
        require("execution_authority_id" not in value["binding"], "historical unit was relabelled")
        _validate_persisted_streams(runtime, expected_unit, value, expected_stable_ids=list(expected_stable_ids))
        unit_rows.append({"ordinal": ordinal, "path": relative, "unit_id": value["unit_id"], "sha256": _sha(path)})
        for index, variant in enumerate(value["scorer_variants"]):
            srel = per_image_relative_path(expected_unit, index)
            spath = runtime / srel
            record = read_json(spath, "historical W10 scorer stream")
            _verify_scheduled_rows(record["rows"], expected_unit)
            stream_rows.append({
                "ordinal": ordinal,
                "path": srel,
                "classifier_variant": variant["classifier_variant"],
                "sha256": _sha(spath),
                "canonical_sha256": variant["sha256"],
            })
    require(unit_rows == custody["units"] and stream_rows == custody["streams"], "historical W10 prefix bytes differ")
    require(canonical_sha256({"units": unit_rows, "streams": stream_rows}) == custody["complete_prefix_digest"] == W10_V8_PREFIX_DIGEST, "historical W10 prefix digest differs")
    require(custody["unit_count"] == CONTINUATION_START and custody["stream_count"] == 168 and custody["failed_ordinal"] == CONTINUATION_START, "historical prefix boundary differs")
    return custody


def build_continuation_plan(authority: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_CONTINUATION_PLAN_V9",
        "authority_id": authority["authority_id"],
        "original_authority_id": W10_V8_AUTHORITY_ID,
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "historical_custody_sha256": W10_V8_CUSTODY_SHA256,
        "scope_sha256": authority["scope_sha256"],
        "historical_ordinals": [0, 125],
        "new_ordinals": [126, 251],
        "work_units": list(work_units()),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["plan_id"] = "w10continuationplan-" + canonical_sha256(body)
    return body


def verify_continuation_plan(runtime: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    plan = read_json(runtime / CONTINUATION_PLAN_PATH, "W10 v9 continuation plan")
    require(plan == build_continuation_plan(authority), "W10 v9 continuation plan differs")
    return plan


def build_launch_authorization(authority: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a later, separate owner grant; this task never writes it."""

    require(dict(plan) == build_continuation_plan(authority), "W10 v9 launch plan differs")
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_CONTINUATION_SUFFIX_LAUNCH_AUTHORIZATION_V9",
        "status": "OWNER_AUTHORIZED_SUFFIX_ONLY",
        "continuation_authority_id": authority["authority_id"],
        "source_manifest_id": authority["source_manifest"]["manifest_id"],
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "plan_id": plan["plan_id"],
        "authorized_ordinals": list(range(CONTINUATION_START, CONTINUATION_STOP)),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["launch_id"] = "w10continuationlaunch-" + canonical_sha256(body)
    return body


def verify_launch_authorization(root: Path, authority: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(root) / CONTINUATION_LAUNCH_PATH
    require(path.is_file() and not path.is_symlink(), "W10 v9 suffix launch lacks separate owner authorization")
    value = read_json(path, "W10 v9 suffix launch authorization")
    require(value == build_launch_authorization(authority, plan), "W10 v9 suffix launch authorization differs")
    return value


def verify_suffix_evidence_set(runtime: Path, custody: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    """Allow only an ordered, v9-marked suffix prefix after the original 126."""

    expected = work_units()
    unit_files = {str(path.relative_to(runtime)) for path in (runtime / "units").iterdir()}
    stream_files = {str(path.relative_to(runtime)) for path in (runtime / "per_image").iterdir()}
    historical_units = {row["path"] for row in custody["units"]}
    historical_streams = {row["path"] for row in custody["streams"]}
    require(historical_units <= unit_files and historical_streams <= stream_files, "historical evidence disappeared")
    suffix_ordinals = []
    expected_suffix_units = set()
    expected_suffix_streams = set()
    for unit in expected[CONTINUATION_START:]:
        relative = unit_relative_path(unit, "json")
        path = runtime / relative
        if not path.exists():
            continue
        require(path.is_file() and not path.is_symlink(), "v9 suffix unit is unsafe")
        value = read_json(path, "W10 v9 suffix unit")
        validate_unit(value, unit)
        require(value["binding"].get("execution_authority_id") == authority["authority_id"], "cross-authority suffix evidence")
        suffix_ordinals.append(unit["ordinal"])
        expected_suffix_units.add(relative)
        expected_suffix_streams.update(variant["path"] for variant in value["scorer_variants"])
    require(suffix_ordinals == list(range(CONTINUATION_START, CONTINUATION_START + len(suffix_ordinals))), "v9 suffix has a hole")
    require(unit_files == historical_units | expected_suffix_units, "unexpected W10 unit evidence")
    require(stream_files == historical_streams | expected_suffix_streams, "unexpected W10 scorer evidence")


def verify_suffix_streams(runtime: Path, *, expected_stable_ids: Sequence[str], authority: Mapping[str, Any]) -> None:
    """Authenticate every persisted v9 scorer row before a terminal closeout."""

    for unit in work_units()[CONTINUATION_START:]:
        path = runtime / unit_relative_path(unit, "json")
        require(path.is_file() and not path.is_symlink(), "W10 v9 suffix is incomplete")
        value = read_json(path, "W10 v9 suffix unit")
        validate_unit(value, unit)
        require(value["binding"].get("execution_authority_id") == authority["authority_id"], "W10 v9 suffix authority differs")
        _validate_persisted_streams(runtime, unit, value, expected_stable_ids=list(expected_stable_ids))
        for variant in value["scorer_variants"]:
            record = read_json(runtime / variant["path"], "W10 v9 scorer stream")
            _verify_scheduled_rows(record["rows"], unit)


def build_continuation_closeout(runtime: Path, results: list[Mapping[str, Any]], authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Closeout names the actual source and authority for every ordinal."""

    require(len(results) == CONTINUATION_STOP, "W10 continuation has incomplete scope")
    for value, unit in zip(results, work_units(), strict=True):
        validate_unit(value, unit)
        if unit["ordinal"] >= CONTINUATION_START:
            require(value["binding"].get("execution_authority_id") == authority["authority_id"], "v9 closeout suffix provenance differs")
        else:
            require("execution_authority_id" not in value["binding"], "v8 closeout prefix was relabelled")
    units = unit_manifest(runtime, [dict(item) for item in results])
    images = per_image_manifest(runtime, units)
    provenance = [
        {
            "ordinal": index,
            "source_manifest_id": W10_V8_MANIFEST_ID if index < CONTINUATION_START else authority["source_manifest"]["manifest_id"],
            "authority_id": W10_V8_AUTHORITY_ID if index < CONTINUATION_START else authority["authority_id"],
            "unit_id": value["unit_id"],
        }
        for index, value in enumerate(results)
    ]
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_CONTINUATION_CLOSEOUT_V9",
        "status": "COMPLETE",
        "original_authority_id": W10_V8_AUTHORITY_ID,
        "continuation_authority_id": authority["authority_id"],
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "historical_custody_sha256": W10_V8_CUSTODY_SHA256,
        "scope_sha256": authority["scope_sha256"],
        "unit_count": CONTINUATION_STOP,
        "stream_count": images["stream_count"],
        "ordered_unit_ids_digest": units["ordered_unit_ids_digest"],
        "ordered_per_image_digest": images["ordered_per_image_digest"],
        "unit_manifest_sha256": canonical_sha256(units),
        "per_image_manifest_sha256": canonical_sha256(images),
        "ordinal_provenance": provenance,
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["closeout_id"] = "w10continuationcloseout-" + canonical_sha256(body)
    return body, units, images


def published_continuation_units(results: list[Mapping[str, Any]], authority: Mapping[str, Any]) -> dict[str, Any]:
    require(len(results) == CONTINUATION_STOP, "W10 continuation published unit count differs")
    for value, unit in zip(results, work_units(), strict=True):
        validate_unit(value, unit)
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_CONTINUATION_PUBLISHED_UNITS_V9",
        "original_authority_id": W10_V8_AUTHORITY_ID,
        "continuation_authority_id": authority["authority_id"],
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "unit_count": CONTINUATION_STOP,
        "units": [dict(value) for value in results],
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["published_units_id"] = "w10continuationunits-" + canonical_sha256(body)
    return body
