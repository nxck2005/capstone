"""Authenticated v8/v9 custody and an ordinal-231-only W10 v10 continuation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.downstream_v4 import read_json, require
from evaluation.w10_continuation import (
    _verify_scheduled_rows,
    build_continuation_authority,
)
from evaluation.w10_evidence import per_image_relative_path, unit_relative_path
from evaluation.w10_rehearsal import (
    _validate_persisted_streams,
    per_image_manifest,
    unit_manifest,
    validate_unit,
    work_units,
)
from runtime.source_epochs import (
    W10_V8_AUTHORITY_ID,
    W10_V8_CUSTODY_ID,
    W10_V8_CUSTODY_SHA256,
    W10_V8_MANIFEST_ID,
    W10_V8_PREFIX_DIGEST,
    W10_V9_MANIFEST_KIND,
    W10_V10_MANIFEST_KIND,
    W10_V11_MANIFEST_KIND,
    assert_active_epoch_closure,
    load_w10_manifest,
    source_record,
    v8_continuation_custody,
)
from runtime.source_guard import assert_manifest_commit_bytes
from training.deterministic_core import canonical_bytes, canonical_sha256

V9_CUSTODY_PATH = "results/learned/w10/w10_v9_failed_prefix_custody.json"
V9_AUTHORITY_SHA256 = "15c1bb0c479c46146d1c66eb17bae1432cc4eb127ea06c1f8d8023aa0dd0f421"
V10_AUTHORITY_PATH = "results/learned/w10/w10_continuation_authorization_v10.json"
V10_LAUNCH_PATH = "results/learned/w10/w10_continuation_launch_authorization_v10.json"
V10_PLAN_PATH = "continuation_plan_v10.json"
V10_START = 231
V10_STOP = 252
V10_AUTHORITY_KIND = "W10_VALIDATION_CONTINUATION_AUTHORITY_V10"
V11_AUTHORITY_PATH = "results/learned/w10/w10_continuation_authorization_v11.json"
V11_LAUNCH_PATH = "results/learned/w10/w10_continuation_launch_authorization_v11.json"
V11_PLAN_PATH = "continuation_plan_v11.json"
V11_AUTHORITY_KIND = "W10_VALIDATION_CONTINUATION_AUTHORITY_V11"
V10_CLOSEOUT_FILES = {
    "continuation_unit_manifest_v10.json",
    "continuation_per_image_manifest_v10.json",
    "continuation_closeout_v10.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_v9_authority_for_succession(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate v9 at its own commit before the v10 source can be active."""

    root = Path(root)
    source = load_w10_manifest(root, live=False, epoch="v9")
    require(source["manifest_kind"] == W10_V9_MANIFEST_KIND, "v9 custody source kind differs")
    assert_manifest_commit_bytes(root, source)
    path = root / "results/learned/w10/w10_continuation_authorization_v9.json"
    require(path.is_file() and not path.is_symlink(), "W10 v9 authority is missing or unsafe")
    authority = read_json(path, "W10 v9 authority")
    body = dict(authority)
    identifier = body.pop("authority_id", None)
    require(identifier == "w10continuationauth-" + canonical_sha256(body), "W10 v9 authority ID differs")
    require(body == build_continuation_authority(root, source), "W10 v9 authority differs from its frozen v8 predecessor")
    require(_sha(path) == V9_AUTHORITY_SHA256, "W10 v9 historical authority bytes differ")
    require(authority["validation_only"] is True and authority["test"] == "SEALED" and authority["test_access"] == 0 and authority["test_authorized"] is False, "W10 v9 authority crossed test boundary")
    return source, authority


def _historical_records(
    root: Path,
    *,
    stable_ids: Sequence[str],
    authority: Mapping[str, Any],
    allow_v10_suffix: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reauthenticate every old record; never publish or rewrite a record."""

    root = Path(root)
    runtime = root / authority["runtime_root"]
    require(runtime.is_dir() and not runtime.is_symlink(), "W10 historical runtime is missing or unsafe")
    units_dir = runtime / "units"
    streams_dir = runtime / "per_image"
    require(
        units_dir.is_dir() and not units_dir.is_symlink()
        and streams_dir.is_dir() and not streams_dir.is_symlink(),
        "W10 historical evidence directories are missing or unsafe",
    )
    v8 = v8_continuation_custody(root)
    unit_paths: set[str] = set()
    stream_paths: set[str] = set()
    v9_units: list[dict[str, Any]] = []
    v9_streams: list[dict[str, Any]] = []
    for unit in work_units()[:V10_START]:
        ordinal = unit["ordinal"]
        relative = unit_relative_path(unit, "json")
        path = runtime / relative
        require(path.is_file() and not path.is_symlink(), f"historical W10 unit {ordinal} is missing or unsafe")
        value = read_json(path, f"historical W10 unit {ordinal}")
        validate_unit(value, unit)
        require(path.read_bytes() == canonical_bytes(value), f"historical W10 unit {ordinal} bytes are not canonical")
        if ordinal < 126:
            require("execution_authority_id" not in value["binding"], f"historical W10 unit {ordinal} authority differs")
        else:
            require(value["binding"].get("execution_authority_id") == authority["authority_id"], f"historical W10 unit {ordinal} authority differs")
        _validate_persisted_streams(runtime, unit, value, expected_stable_ids=list(stable_ids))
        unit_row = {"ordinal": ordinal, "path": relative, "unit_id": value["unit_id"], "sha256": _sha(path)}
        if ordinal < 126:
            require(unit_row == v8["units"][ordinal], f"v8 W10 unit {ordinal} bytes differ")
        else:
            v9_units.append(unit_row)
        unit_paths.add(relative)
        for index, variant in enumerate(value["scorer_variants"]):
            stream_relative = per_image_relative_path(unit, index)
            stream_path = runtime / stream_relative
            record = read_json(stream_path, f"historical W10 scorer {ordinal}/{index}")
            require(stream_path.read_bytes() == canonical_bytes(record), "historical W10 scorer bytes are not canonical")
            _verify_scheduled_rows(record["rows"], unit)
            stream_row = {
                "ordinal": ordinal,
                "path": stream_relative,
                "classifier_variant": variant["classifier_variant"],
                "sha256": _sha(stream_path),
                "canonical_sha256": variant["sha256"],
            }
            if ordinal < 126:
                # Historical scorer ordering is the order of work units and variants.
                require(stream_row == v8["streams"][len(stream_paths)], "v8 W10 scorer bytes differ")
            else:
                v9_streams.append(stream_row)
            stream_paths.add(stream_relative)
    actual_units = {str(path.relative_to(runtime)) for path in units_dir.iterdir()}
    actual_streams = {str(path.relative_to(runtime)) for path in streams_dir.iterdir()}
    if allow_v10_suffix:
        require(unit_paths <= actual_units and stream_paths <= actual_streams, "historical W10 evidence disappeared")
    else:
        require(actual_units == unit_paths and actual_streams == stream_paths, "failed v9 prefix contains unexpected evidence")
    require(len(v9_units) == 105 and len(v9_streams) == 168, "failed v9 prefix cardinality differs")
    return v9_units, v9_streams


def build_v9_custody(root: Path, *, stable_ids: Sequence[str]) -> dict[str, Any]:
    """Capture the failed v9 prefix after authenticating its original sources."""

    root = Path(root)
    source, authority = verify_v9_authority_for_succession(root)
    require(len(stable_ids) == 1000 and len(set(stable_ids)) == 1000, "v9 custody validation IDs differ")
    units, streams = _historical_records(root, stable_ids=stable_ids, authority=authority, allow_v10_suffix=False)
    body = {
        "schema_version": 1,
        "artifact_role": "W10_V9_FAILED_PREFIX_CUSTODY",
        "status": "FAILED_PARTIAL_NOT_CLOSEOUT",
        "source": source_record(root, source),
        "authority": {
            "path": "results/learned/w10/w10_continuation_authorization_v9.json",
            "authority_id": authority["authority_id"],
            "sha256": _sha(root / "results/learned/w10/w10_continuation_authorization_v9.json"),
        },
        "v8_custody_id": W10_V8_CUSTODY_ID,
        "v8_custody_sha256": W10_V8_CUSTODY_SHA256,
        "completed_ordinals": [126, 230],
        "failed_ordinal": V10_START,
        "unit_count": len(units),
        "stream_count": len(streams),
        "units": units,
        "streams": streams,
        "complete_prefix_digest": canonical_sha256({
            "v8_prefix_digest": W10_V8_PREFIX_DIGEST, "units": units, "streams": streams,
        }),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
        "g12_unopened": True,
    }
    body["custody_id"] = "w10v9prefix-" + canonical_sha256(body)
    return body


def load_v9_custody(
    root: Path, *, source: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root)
    path = root / V9_CUSTODY_PATH
    require(path.is_file() and not path.is_symlink(), "W10 v9 failed-prefix custody is missing or unsafe")
    value = read_json(path, "W10 v9 failed-prefix custody")
    body = dict(value)
    identifier = body.pop("custody_id", None)
    require(identifier == "w10v9prefix-" + canonical_sha256(body), "W10 v9 custody ID differs")
    require(path.read_bytes() == canonical_bytes(value), "W10 v9 custody bytes are not canonical")
    require(value["source"] == source_record(root, source), "W10 v9 custody source differs")
    require(value["authority"] == {
        "path": "results/learned/w10/w10_continuation_authorization_v9.json",
        "authority_id": authority["authority_id"],
        "sha256": _sha(root / "results/learned/w10/w10_continuation_authorization_v9.json"),
    }, "W10 v9 custody authority differs")
    require(value["v8_custody_id"] == W10_V8_CUSTODY_ID and value["v8_custody_sha256"] == W10_V8_CUSTODY_SHA256, "W10 v9 custody v8 lineage differs")
    require(value["completed_ordinals"] == [126, 230] and value["failed_ordinal"] == V10_START, "W10 v9 custody frontier differs")
    require(value["unit_count"] == 105 and [item["ordinal"] for item in value["units"]] == list(range(126, V10_START)), "W10 v9 custody unit order differs")
    require(value["stream_count"] == 168 and len(value["streams"]) == 168, "W10 v9 custody stream count differs")
    require(len({item["path"] for item in value["units"]}) == 105 and len({item["path"] for item in value["streams"]}) == 168, "W10 v9 custody paths repeat")
    require(value["complete_prefix_digest"] == canonical_sha256({
        "v8_prefix_digest": W10_V8_PREFIX_DIGEST, "units": value["units"], "streams": value["streams"],
    }), "W10 v9 custody prefix digest differs")
    require(value["validation_only"] is True and value["test"] == "SEALED" and value["test_access"] == 0 and value["g12_unopened"] is True, "W10 v9 custody crossed test boundary")
    return value


def verify_historical_prefix(root: Path, *, stable_ids: Sequence[str], allow_v10_suffix: bool = True) -> dict[str, Any]:
    root = Path(root)
    require(len(stable_ids) == 1000 and len(set(stable_ids)) == 1000, "W10 validation stable IDs differ")
    v8_source = load_w10_manifest(root, live=False, epoch="v8")
    v9_source, authority = verify_v9_authority_for_succession(root)
    assert_manifest_commit_bytes(root, v8_source)
    assert_active_epoch_closure(root, v9_source)
    custody = load_v9_custody(root, source=v9_source, authority=authority)
    units, streams = _historical_records(root, stable_ids=stable_ids, authority=authority, allow_v10_suffix=allow_v10_suffix)
    require(units == custody["units"] and streams == custody["streams"], "W10 v9 historical evidence bytes differ")
    return custody


def _epoch(authority: Mapping[str, Any]) -> str:
    kind = authority.get("authority_kind")
    require(kind in {V10_AUTHORITY_KIND, V11_AUTHORITY_KIND}, "W10 suffix authority kind differs")
    return "v11" if kind == V11_AUTHORITY_KIND else "v10"


def build_v10_authority(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    require(source.get("manifest_kind") in {W10_V10_MANIFEST_KIND, W10_V11_MANIFEST_KIND}, "W10 suffix authority needs source-v10/v11")
    root = Path(root)
    epoch = "v11" if source["manifest_kind"] == W10_V11_MANIFEST_KIND else "v10"
    v9_source, v9 = verify_v9_authority_for_succession(root)
    custody = load_v9_custody(root, source=v9_source, authority=v9)
    body = {key: value for key, value in v9.items() if key != "authority_id"}
    body.update({
        "schema_version": 4,
        "authority_kind": V11_AUTHORITY_KIND if epoch == "v11" else V10_AUTHORITY_KIND,
        "status": "FROZEN_231_251_ONLY_PRE_EXECUTION",
        "source_manifest": source_record(root, source),
        "source_binding": dict(source),
        "source_commit": source["source_commit"],
        "v9_authority": {
            "path": "results/learned/w10/w10_continuation_authorization_v9.json",
            "authority_id": v9["authority_id"],
            "sha256": _sha(root / "results/learned/w10/w10_continuation_authorization_v9.json"),
            "source_manifest_id": v9["source_manifest"]["manifest_id"],
        },
        "v9_custody": {"path": V9_CUSTODY_PATH, "custody_id": custody["custody_id"], "sha256": _sha(root / V9_CUSTODY_PATH)},
        "historical_completed_ordinals": list(range(V10_START)),
        "new_authorized_ordinals": list(range(V10_START, V10_STOP)),
        "plan_path": V11_PLAN_PATH if epoch == "v11" else V10_PLAN_PATH,
        "suffix_only": True,
    })
    if epoch == "v11":
        body["superseded_v10"] = dict(source["transition_from_v10"])
    return body


def verify_v10_authority(root: Path, *, live: bool = True, epoch: str = "v10", historical: bool = False) -> dict[str, Any]:
    root = Path(root)
    require(epoch in {"v10", "v11"}, "unknown W10 suffix epoch")
    source = load_w10_manifest(root, live=live, epoch=epoch)
    path = root / (V11_AUTHORITY_PATH if epoch == "v11" else V10_AUTHORITY_PATH)
    require(path.is_file() and not path.is_symlink(), "W10 v10 authority is missing or unsafe")
    value = read_json(path, "W10 v10 authority")
    require(path.read_bytes() == canonical_bytes(value), "W10 v10 authority bytes are not canonical")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == f"w10continuationauth{epoch}-" + canonical_sha256(body), "W10 suffix authority ID differs")
    require(body == build_v10_authority(root, source), "W10 v10 authority fields differ")
    from evaluation.w10_bindings import resolve_scope_bindings

    if not historical:
        require(value["bindings"] == resolve_scope_bindings(root, verify_runtime=False), "W10 v10 frozen scientific bindings differ")
    require(value["scope_sha256"] == canonical_sha256(value["scope"]), "W10 v10 scope digest differs")
    require(value["validation_only"] is True and value["test"] == "SEALED" and value["test_access"] == 0 and value["test_authorized"] is False, "W10 v10 authority crossed test boundary")
    return value


def build_v10_plan(authority: Mapping[str, Any]) -> dict[str, Any]:
    epoch = _epoch(authority)
    require(authority["historical_completed_ordinals"] == list(range(V10_START)) and authority["new_authorized_ordinals"] == list(range(V10_START, V10_STOP)), "W10 v10 plan frontier differs")
    body = {
        "schema_version": 1,
        "artifact_role": f"W10_VALIDATION_CONTINUATION_PLAN_{epoch.upper()}",
        "authority_id": authority["authority_id"],
        "scope_sha256": authority["scope_sha256"],
        "v8_ordinals": [0, 125],
        "v9_ordinals": [126, 230],
        f"{epoch}_ordinals": [V10_START, V10_STOP - 1],
        "v9_custody_id": authority["v9_custody"]["custody_id"],
        "work_units": list(work_units()),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["plan_id"] = f"w10continuationplan{epoch}-" + canonical_sha256(body)
    return body


def verify_v10_plan(runtime: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(runtime) / authority["plan_path"]
    require(path.is_file() and not path.is_symlink(), "W10 v10 plan is missing or unsafe")
    value = read_json(path, "W10 v10 plan")
    require(value == build_v10_plan(authority), "W10 v10 plan differs")
    require(path.read_bytes() == canonical_bytes(value), "W10 v10 plan bytes are not canonical")
    return value


def build_v10_launch(authority: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    epoch = _epoch(authority)
    require(dict(plan) == build_v10_plan(authority), "W10 v10 launch plan differs")
    body = {
        "schema_version": 1,
        "artifact_role": f"W10_VALIDATION_CONTINUATION_LAUNCH_AUTHORIZATION_{epoch.upper()}",
        "status": "OWNER_AUTHORIZED_231_251_ONLY",
        "authority_id": authority["authority_id"],
        "source_manifest_id": authority["source_manifest"]["manifest_id"],
        "v9_custody_id": authority["v9_custody"]["custody_id"],
        "plan_id": plan["plan_id"],
        "authorized_ordinals": list(range(V10_START, V10_STOP)),
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["launch_id"] = f"w10continuationlaunch{epoch}-" + canonical_sha256(body)
    return body


def verify_v10_launch(root: Path, authority: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(root) / (V11_LAUNCH_PATH if _epoch(authority) == "v11" else V10_LAUNCH_PATH)
    require(path.is_file() and not path.is_symlink(), "W10 v10 launch lacks separate owner authorization")
    value = read_json(path, "W10 v10 launch")
    require(value == build_v10_launch(authority, plan), "W10 v10 launch authorization differs")
    require(path.read_bytes() == canonical_bytes(value), "W10 v10 launch bytes are not canonical")
    return value


def verify_v10_evidence_set(
    root: Path, *, authority: Mapping[str, Any], stable_ids: Sequence[str], phase: str = "execute",
) -> int:
    """Accept exactly a contiguous v10 suffix after the immutable 0–230 prefix."""

    require(phase in {"plan", "execute", "closeout"}, "unknown W10 v10 phase")
    epoch = _epoch(authority)
    root = Path(root)
    require(not (root / "results/freeze_manifest.json").exists(), "G12 is already open")
    custody = verify_historical_prefix(root, stable_ids=stable_ids)
    require(custody["custody_id"] == authority["v9_custody"]["custody_id"], "W10 v10 historical custody differs")
    runtime = root / authority["runtime_root"]
    units = work_units()
    unit_files = {str(path.relative_to(runtime)) for path in (runtime / "units").iterdir()}
    stream_files = {str(path.relative_to(runtime)) for path in (runtime / "per_image").iterdir()}
    expected_units = {unit_relative_path(unit, "json") for unit in units[:V10_START]}
    expected_streams = {
        row["path"] for row in v8_continuation_custody(root)["streams"]
    } | {row["path"] for row in custody["streams"]}
    complete = 0
    for unit in units[V10_START:]:
        relative = unit_relative_path(unit, "json")
        if relative not in unit_files:
            break
        path = runtime / relative
        require(path.is_file() and not path.is_symlink(), "W10 v10 unit is unsafe")
        value = read_json(path, "W10 v10 unit")
        validate_unit(value, unit)
        require(value["binding"].get("execution_authority_id") == authority["authority_id"], "W10 v10 unit has wrong authority")
        require(path.read_bytes() == canonical_bytes(value), "W10 v10 unit bytes are not canonical")
        _validate_persisted_streams(runtime, unit, value, expected_stable_ids=list(stable_ids))
        for variant in value["scorer_variants"]:
            stream = runtime / variant["path"]
            record = read_json(stream, "W10 v10 scorer")
            require(stream.read_bytes() == canonical_bytes(record), "W10 v10 scorer bytes are not canonical")
            _verify_scheduled_rows(record["rows"], unit)
            expected_streams.add(variant["path"])
        expected_units.add(relative)
        complete += 1
    require(unit_files == expected_units, "W10 v10 units have a gap or unexpected record")
    pending = V10_START + complete
    if pending < V10_STOP:
        pending_path = per_image_relative_path(units[pending], 0)
        if pending_path in stream_files:
            from evaluation.w10_continuation import _verify_orphan_stream

            _verify_orphan_stream(runtime / pending_path, units[pending], 0, stable_ids)
            expected_streams.add(pending_path)
    require(stream_files == expected_streams, "W10 v10 scorer streams have a gap or unexpected record")
    allowed = {"units", "per_image", "j2k_cache", "plan.json", "continuation_plan_v9.json", authority["plan_path"]}
    if phase == "closeout":
        allowed |= {f"continuation_unit_manifest_{epoch}.json", f"continuation_per_image_manifest_{epoch}.json", f"continuation_closeout_{epoch}.json"}
    names = {path.name for path in runtime.iterdir()}
    require(names <= allowed, "W10 v10 runtime contains unexpected files")
    for name in names & {"units", "per_image", "j2k_cache"}:
        path = runtime / name
        require(path.is_dir() and not path.is_symlink(), "W10 v10 runtime directory is unsafe")
    for name in names - {"units", "per_image", "j2k_cache"}:
        path = runtime / name
        require(path.is_file() and not path.is_symlink(), "W10 v10 runtime file is unsafe")
    return complete


def build_v10_closeout(
    runtime: Path, results: list[Mapping[str, Any]], authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(len(results) == V10_STOP, "W10 v10 closeout requires 252 units")
    unit_rows = unit_manifest(runtime, [dict(item) for item in results])
    image_rows = per_image_manifest(runtime, unit_rows)
    return _closeout_from_manifests(results, authority, unit_rows, image_rows), unit_rows, image_rows


def _closeout_from_manifests(
    results: Sequence[Mapping[str, Any]], authority: Mapping[str, Any],
    unit_rows: Mapping[str, Any], image_rows: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(results) == V10_STOP, "W10 v10 closeout requires 252 units")
    epoch = _epoch(authority)
    v9_source_id = authority["v9_authority"]["source_manifest_id"]
    v9_authority_id = authority["v9_authority"]["authority_id"]
    provenance = []
    for index, (unit, value) in enumerate(zip(work_units(), results, strict=True)):
        validate_unit(value, unit)
        actual = value["binding"].get("execution_authority_id")
        expected = None if index < 126 else v9_authority_id if index < V10_START else authority["authority_id"]
        require(actual == expected, "W10 v10 closeout unit authority differs")
        source_id = W10_V8_MANIFEST_ID if index < 126 else v9_source_id if index < V10_START else authority["source_manifest"]["manifest_id"]
        authority_id = W10_V8_AUTHORITY_ID if index < 126 else v9_authority_id if index < V10_START else authority["authority_id"]
        provenance.append({"ordinal": index, "source_manifest_id": source_id, "authority_id": authority_id, "unit_id": value["unit_id"]})
    body = {
        "schema_version": 1,
        "artifact_role": f"W10_VALIDATION_CONTINUATION_CLOSEOUT_{epoch.upper()}",
        "status": "COMPLETE",
        "v8_authority_id": W10_V8_AUTHORITY_ID,
        "v9_authority_id": v9_authority_id,
        f"{epoch}_authority_id": authority["authority_id"],
        "v8_custody_id": W10_V8_CUSTODY_ID,
        "v9_custody_id": authority["v9_custody"]["custody_id"],
        "scope_sha256": authority["scope_sha256"],
        "unit_count": V10_STOP,
        "stream_count": image_rows["stream_count"],
        "ordered_unit_ids_digest": unit_rows["ordered_unit_ids_digest"],
        "ordered_per_image_digest": image_rows["ordered_per_image_digest"],
        "unit_manifest_sha256": canonical_sha256(unit_rows),
        "per_image_manifest_sha256": canonical_sha256(image_rows),
        "ordinal_provenance": provenance,
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    if epoch == "v11":
        body["superseded_v10_authority_id"] = authority["superseded_v10"]["v10_authority_id"]
    body["closeout_id"] = f"w10continuationcloseout{epoch}-" + canonical_sha256(body)
    return body


def published_v10_units(results: list[Mapping[str, Any]], authority: Mapping[str, Any]) -> dict[str, Any]:
    require(len(results) == V10_STOP, "W10 v10 publication requires 252 units")
    epoch = _epoch(authority)
    body = {
        "schema_version": 1,
        "artifact_role": f"W10_VALIDATION_CONTINUATION_PUBLISHED_UNITS_{epoch.upper()}",
        "v8_authority_id": W10_V8_AUTHORITY_ID,
        "v9_authority_id": authority["v9_authority"]["authority_id"],
        f"{epoch}_authority_id": authority["authority_id"],
        "v8_custody_id": W10_V8_CUSTODY_ID,
        "v9_custody_id": authority["v9_custody"]["custody_id"],
        "unit_count": V10_STOP,
        "units": [dict(value) for value in results],
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    if epoch == "v11":
        body["superseded_v10_authority_id"] = authority["superseded_v10"]["v10_authority_id"]
    body["published_units_id"] = f"w10continuationunits{epoch}-" + canonical_sha256(body)
    return body


def verify_v10_published(root: Path, *, epoch: str = "v10") -> dict[str, Any]:
    """Authenticate committed v10 closeout without claiming access to worker bytes."""

    root = Path(root)
    authority = verify_v10_authority(root) if epoch == "v10" else verify_v10_authority(root, epoch=epoch)
    verify_v10_launch(root, authority, build_v10_plan(authority))
    v8 = v8_continuation_custody(root)
    v9_source, v9_authority = verify_v9_authority_for_succession(root)
    v9 = load_v9_custody(root, source=v9_source, authority=v9_authority)
    paths = {
        "units": root / f"results/learned/w10/w10_continuation_units_{epoch}.json",
        "unit_manifest": root / f"results/learned/w10/w10_continuation_unit_manifest_{epoch}.json",
        "images": root / f"results/learned/w10/w10_continuation_per_image_manifest_{epoch}.json",
        "closeout": root / f"results/learned/w10/w10_continuation_closeout_{epoch}.json",
    }
    values = {}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"W10 v10 published {name} is missing or unsafe")
        value = read_json(path, f"W10 v10 published {name}")
        require(path.read_bytes() == canonical_bytes(value), f"W10 v10 published {name} bytes are not canonical")
        values[name] = value
    published = values["units"]
    unit_rows = values["unit_manifest"]
    image_rows = values["images"]
    closeout = values["closeout"]
    results = published.get("units")
    require(isinstance(results, list) and len(results) == V10_STOP, "W10 v10 published unit count differs")
    require(published == published_v10_units(results, authority), "W10 v10 published unit body differs")
    for value, field, prefix in (
        (unit_rows, "unit_manifest_id", "w10unitmanifest-"),
        (image_rows, "per_image_manifest_id", "w10imagemanifest-"),
    ):
        body = dict(value)
        identifier = body.pop(field, None)
        require(identifier == prefix + canonical_sha256(body), f"W10 v10 published {field} differs")
    require(unit_rows.get("unit_count") == V10_STOP and len(unit_rows.get("units", [])) == V10_STOP, "W10 v10 unit manifest scope differs")
    require(unit_rows.get("artifact_role") == "W10_VALIDATION_REHEARSAL_UNIT_MANIFEST" and image_rows.get("artifact_role") == "W10_VALIDATION_REHEARSAL_PER_IMAGE_MANIFEST", "W10 v10 manifest roles differ")
    expected_streams = []
    v8_stream_index = 0
    v9_stream_index = 0
    for index, (unit, value, row) in enumerate(zip(work_units(), results, unit_rows["units"], strict=True)):
        validate_unit(value, unit)
        expected_row = {
            "ordinal": index,
            "unit_id": value["unit_id"],
            "unit_path": unit_relative_path(unit, "json"),
            "unit_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
            "per_image_path": value["per_image_path"],
            "per_image_sha256": value["per_image_sha256"],
            "n_correct": value["n_correct"],
            "n_total": value["n_total"],
            "scorer_variants": value["scorer_variants"],
        }
        require(row == expected_row, f"W10 v10 published unit manifest row {index} differs")
        if index < 126:
            require(row["unit_sha256"] == v8["units"][index]["sha256"] and value["unit_id"] == v8["units"][index]["unit_id"], "W10 v8 published unit differs from custody")
        elif index < V10_START:
            require(row["unit_sha256"] == v9["units"][index - 126]["sha256"] and value["unit_id"] == v9["units"][index - 126]["unit_id"], "W10 v9 published unit differs from custody")
        for variant_index, variant in enumerate(value["scorer_variants"]):
            require(variant["path"] == per_image_relative_path(unit, variant_index), "W10 v10 published scorer path differs")
            stream = {
                "ordinal": index,
                "classifier_variant": variant["classifier_variant"],
                "per_image_path": variant["path"],
                "per_image_sha256": variant["sha256"],
                "n_correct": variant["n_correct"],
                "n_total": variant["n_total"],
            }
            expected_streams.append(stream)
            if index < 126:
                frozen = v8["streams"][v8_stream_index]
                v8_stream_index += 1
            elif index < V10_START:
                frozen = v9["streams"][v9_stream_index]
                v9_stream_index += 1
            else:
                continue
            require(stream["ordinal"] == frozen["ordinal"] and stream["per_image_path"] == frozen["path"] and stream["per_image_sha256"] == frozen["canonical_sha256"], "W10 historical published scorer differs from custody")
    require(v8_stream_index == len(v8["streams"]) and v9_stream_index == len(v9["streams"]), "W10 historical published stream count differs")
    require(unit_rows["ordered_unit_ids_digest"] == canonical_sha256({"unit_ids": [value["unit_id"] for value in results]}), "W10 v10 ordered unit digest differs")
    require(image_rows.get("stream_count") == len(expected_streams) and image_rows.get("streams") == expected_streams, "W10 v10 published stream manifest differs")
    require(image_rows["ordered_per_image_digest"] == canonical_sha256({"streams": expected_streams}), "W10 v10 ordered stream digest differs")
    require(closeout == _closeout_from_manifests(results, authority, unit_rows, image_rows), "W10 v10 published closeout differs")
    return {"closeout_id": closeout["closeout_id"], "unit_count": V10_STOP, "stream_count": len(expected_streams)}
